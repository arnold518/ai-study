import math
import time
import torch
import torch.nn as nn
from torch.utils.data import IterableDataset, DataLoader
import matplotlib.pyplot as plt

from torchtext import data as data_legacy
from torchtext import datasets as datasets_legacy


# --- 1) Data loading & vocab ---
TEXT = data_legacy.Field(tokenize='basic_english', lower=True)
train_data, _, test_data = datasets_legacy.PennTreebank.splits(TEXT)
TEXT.build_vocab(train_data)
vocab = TEXT.vocab
dict_size = len(vocab)
print(f"Vocabulary Size: {dict_size}")


# --- 2) Flatten PTB splits into long token streams ---
def create_corpus(dataset, vocabulary):
    """Concatenate all examples into a single 1D LongTensor of token IDs."""
    corpus_tensors = []
    for ex in dataset.examples:
        ids = [vocabulary.stoi[token] for token in ex.text]
        corpus_tensors.append(torch.tensor(ids, dtype=torch.long))
    return torch.cat(corpus_tensors, dim=0)

train_corpus = create_corpus(train_data, vocab)
test_corpus  = create_corpus(test_data,  vocab)

print(f"Total tokens in training corpus: {len(train_corpus)}")
print(f"Total tokens in test corpus: {len(test_corpus)}")


# --- 3) Stream batching into [T, B] ---
def batchify(data_1d, batch_size, device):
    """
    Reshape a 1D token stream into [T, B] where B=batch_size parallel streams.
    Trims the tail so T*B divides the length exactly.
    """
    n_batches = data_1d.size(0) // batch_size
    data_1d = data_1d.narrow(0, 0, n_batches * batch_size)
    data_tb = data_1d.view(batch_size, -1).t().contiguous()  # [T, B]
    return data_tb.to(device)

# --- 4) TBPTT windowing via IterableDataset (stateful order preserved) ---
class StreamBatchDataset(IterableDataset):
    """
    Iterates over a time-major source tensor [T, batch_size] and yields
    (data_window, target_window) pairs for TBPTT:
      data_window   : [seq_len, batch_size]
      target_window : [seq_len * batch_size]  (flattened next-token labels)
    NOTE: Order is sequential; do NOT shuffle if you want stateful training.
    """
    def __init__(self, source_tb: torch.Tensor, sequence_length: int):
        super().__init__()
        self.source = source_tb
        self.seq_len = sequence_length

    def __iter__(self):
        T = self.source.size(0)
        for i in range(0, T - 1, self.seq_len):
            cur_len = min(self.seq_len, (T - 1) - i)
            data = self.source[i : i + cur_len]                   # [cur_len, B]
            target = self.source[i + 1 : i + 1 + cur_len]         # [cur_len, B]
            yield data, target.reshape(-1)                         # [cur_len*B]
    
# --- 5) Model ---
class RNNModel(nn.Module):
    """Minimal RNN LM: Embedding → RNN → Linear-to-Vocab.
    
    Forward expects:
      tokens : [sequence_length, batch_size]
      h_0    : [num_layers,     batch_size, hidden_size] or None
    Returns:
      logits : [sequence_length, batch_size, vocab_size]
      h_N    : [num_layers,     batch_size, hidden_size]
    """
    def __init__(self, dict_size, embedding_size, hidden_size, num_layers):
        super().__init__()
        self.embedding = nn.Embedding(dict_size, embedding_size)
        self.rnn = nn.RNN(embedding_size, hidden_size, num_layers=num_layers)
        self.linear = nn.Linear(hidden_size, dict_size)

    def forward(self, x, h):
        """
        tokens   : [sequence_length, batch_size]
        h        : [num_layers, batch_size, hidden_size]
        embedded : [sequence_length, batch_size, embedding_size]
        out      : [sequence_length, batch_size, hidden_size]
        logits   : [sequence_length, batch_size, vocab_size]
        """
        emb = self.embedding(x)            # [seq_len, B, E]
        out, h = self.rnn(emb, h)          # out: [seq_len, B, H]
        logits = self.linear(out)          # [seq_len, B, V]
        return logits, h


# --- 6) Hyperparameters & tensors ---
sequence_length = 35
num_layers      = 3
embedding_size  = 256
hidden_size     = 256
batch_size      = 128
learning_rate   = 0.003
epochs          = 20
grad_clip       = 0.25
device          = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_tb = batchify(train_corpus, batch_size, device)
test_tb  = batchify(test_corpus,  batch_size, device)

# Create iterable datasets that yield TBPTT windows sequentially
train_ds = StreamBatchDataset(train_tb, sequence_length)
test_ds  = StreamBatchDataset(test_tb,  sequence_length)

# DataLoaders: keep order, no auto-batching (each dataset item is already a batch window)
# num_workers=0 is safest when source tensors live on GPU
train_loader = DataLoader(train_ds, batch_size=None, shuffle=False, num_workers=0)
test_loader  = DataLoader(test_ds,  batch_size=None, shuffle=False, num_workers=0)

model = RNNModel(dict_size, embedding_size, hidden_size, num_layers).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)


# --- 7) Training (stateful across windows within an epoch) ---
def train_one_epoch(model, loader, optimizer, clip_norm, device):
    """
    Stateful TBPTT training over a sequential DataLoader.
    Initializes hidden once, carries it across batches, and detaches at window boundaries.
    Returns average loss over steps.
    """
    model.train()
    total_loss = 0.0
    steps = 0

    # Initialize hidden once per epoch; carry across windows
    hidden = torch.zeros(num_layers, batch_size, hidden_size, device=device)

    for data_win, tgt_win in loader:
        # data_win: [seq_len, B] (may be shorter on the last step)
        # tgt_win : [seq_len*B]
        hidden = hidden.detach()  # TBPTT boundary: keep values, cut graph
        optimizer.zero_grad()

        logits, hidden = model(data_win, hidden)   # logits: [seq_len, B, V]
        loss = criterion(logits.reshape(-1, dict_size), tgt_win.reshape(-1))
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
        optimizer.step()

        total_loss += loss.item()
        steps += 1

    return total_loss / max(1, steps)


# --- 8) Evaluation (stateful streaming, no grads) ---
@torch.no_grad()
def evaluate(model, loader, device):
    """
    Stateful evaluation over sequential windows.
    Returns (avg_loss, perplexity, accuracy).
    """
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    total_correct = 0

    hidden = torch.zeros(num_layers, batch_size, hidden_size, device=device)

    for data_win, tgt_win in loader:
        logits, hidden = model(data_win, hidden)   # [seq_len, B, V]
        hidden = hidden.detach()

        # Loss weighted by actual tokens in this window
        V = logits.size(-1)
        loss = criterion(logits.reshape(-1, V), tgt_win.reshape(-1))
        seq_tokens = tgt_win.numel()
        total_loss += seq_tokens * loss.item()
        total_tokens += seq_tokens

        # Accuracy
        preds = torch.argmax(logits, dim=-1).reshape(-1)  # [seq_len*B]
        total_correct += (preds == tgt_win.reshape(-1)).sum().item()

    avg_loss = total_loss / max(1, total_tokens)
    ppl = math.exp(avg_loss)
    acc = 100.0 * total_correct / max(1, total_tokens)
    return avg_loss, ppl, acc


# --- 9) Run training ---
train_losses = []
print("\nStarting stateful training with DataLoader-backed TBPTT...")
for epoch in range(1, epochs + 1):
    t0 = time.time()
    train_loss = train_one_epoch(model, train_loader, optimizer, grad_clip, device)
    test_loss, test_ppl, test_acc = evaluate(model, test_loader, device)
    train_losses.append(train_loss)

    dt = time.time() - t0
    print(f"Epoch [{epoch}/{epochs}]  "
          f"Train Loss: {train_loss:.4f}  "
          f"Test Loss: {test_loss:.4f}  "
          f"Test PPL: {test_ppl:.2f}  "
          f"Test Acc: {test_acc:.2f}%  "
          f"Time: {dt:.2f}s")

    scheduler.step()

print("\nTraining finished.")
final_loss, final_ppl, final_acc = evaluate(model, test_loader, device)
print(f"Final → Loss: {final_loss:.4f} | PPL: {final_ppl:.2f} | Acc: {final_acc:.2f}%")


# --- 10) Plot training curve ---
plt.figure(figsize=(10, 5))
plt.plot(range(1, epochs + 1), train_losses, marker='o', linestyle='--')
plt.title("Training Loss Over Epochs (Stateful TBPTT via DataLoader)")
plt.xlabel("Epoch")
plt.ylabel("Average Loss")
plt.grid(True)
plt.xticks(range(1, epochs + 1))
plt.show()
