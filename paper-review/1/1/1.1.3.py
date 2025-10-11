import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import IterableDataset, DataLoader
import matplotlib.pyplot as plt

from torchtext import data as data_legacy
from torchtext import datasets as datasets_legacy


# =========================
# 0) Minimal RNN components
# =========================

class MinimalRNNCell(nn.Module):
    """
    Single vanilla RNN cell with tanh nonlinearity.

    Forward expects:
      x_t   : [batch_size, input_size]
      h_prev: [batch_size, hidden_size]
    Returns:
      h_t   : [batch_size, hidden_size]
    """
    def __init__(self, input_size: int, hidden_size: int, bias: bool = True):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.W_ih = nn.Parameter(torch.empty(hidden_size, input_size))    # input -> hidden
        self.W_hh = nn.Parameter(torch.empty(hidden_size, hidden_size))   # hidden -> hidden
        if bias:
            self.b_ih = nn.Parameter(torch.zeros(hidden_size))
            self.b_hh = nn.Parameter(torch.zeros(hidden_size))
        else:
            self.register_parameter('b_ih', None)
            self.register_parameter('b_hh', None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize weights/biases; called in __init__ (can be called manually to reinit)."""
        nn.init.xavier_uniform_(self.W_ih)
        nn.init.orthogonal_(self.W_hh)
        if self.b_ih is not None:
            nn.init.zeros_(self.b_ih)
            nn.init.zeros_(self.b_hh)

    def forward(self, x_t: torch.Tensor, h_prev: torch.Tensor) -> torch.Tensor:
        return torch.tanh(
            F.linear(x_t, self.W_ih, self.b_ih) +
            F.linear(h_prev, self.W_hh, self.b_hh)
        )


class MinimalRNN(nn.Module):
    """
    Multi-layer vanilla RNN composed of MinimalRNNCell.

    Forward expects:
      x   : [sequence_length, batch_size, input_size]
      h_0 : [num_layers,     batch_size, hidden_size] or None
    Returns:
      y   : [sequence_length, batch_size, hidden_size]
      h_N : [num_layers,     batch_size, hidden_size]
    """
    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 1, bias: bool = True):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.layers = nn.ModuleList([
            MinimalRNNCell(input_size if l == 0 else hidden_size, hidden_size, bias=bias)
            for l in range(num_layers)
        ])

    def reset_parameters(self) -> None:
        for cell in self.layers:
            cell.reset_parameters()

    def forward(self, x: torch.Tensor, h_0: torch.Tensor | None = None):
        """
        x : [sequence_length, batch_size, input_size]
        h : [num_layers, batch_size, hidden_size]
        out_each_layer_t : [batch_size, hidden_size]
        y : [sequence_length, batch_size, hidden_size]
        hN : [num_layers, batch_size, hidden_size]
        """
        sequence_length, batch_size, _ = x.shape
        if h_0 is None:
            h_prev = x.new_zeros(self.num_layers, batch_size, self.hidden_size)
        else:
            h_prev = h_0

        outputs = []
        for t in range(sequence_length):
            inp = x[t]  # [batch_size, input_size]
            h_next_layers = []
            for l, cell in enumerate(self.layers):
                h_prev_l = h_prev[l]                 # [batch_size, hidden_size]
                h_next_l = cell(inp, h_prev_l)       # [batch_size, hidden_size]
                h_next_layers.append(h_next_l)
                inp = h_next_l                       # feed to next layer
            h_prev = torch.stack(h_next_layers, dim=0)    # [num_layers, batch_size, hidden_size]
            outputs.append(h_next_layers[-1])             # top layer output at time t

        y = torch.stack(outputs, dim=0)  # [sequence_length, batch_size, hidden_size]
        return y, h_prev


class MinimalRNNLM(nn.Module):
    """
    Minimal language model using MinimalRNN:
      Embedding → MinimalRNN → Linear projection to vocabulary.

    Forward expects:
      tokens : [sequence_length, batch_size]
      h_0    : [num_layers,     batch_size, hidden_size] or None
    Returns:
      logits : [sequence_length, batch_size, vocab_size]
      h_N    : [num_layers,     batch_size, hidden_size]
    """
    def __init__(self, vocab_size: int, embedding_size: int, hidden_size: int, num_layers: int = 1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.rnn = MinimalRNN(embedding_size, hidden_size, num_layers=num_layers)
        self.proj = nn.Linear(hidden_size, vocab_size)

    def forward(self, tokens: torch.Tensor, h_0: torch.Tensor | None = None):
        """
        tokens   : [sequence_length, batch_size]
        h        : [num_layers, batch_size, hidden_size]
        embedded : [sequence_length, batch_size, embedding_size]
        out      : [sequence_length, batch_size, hidden_size]
        logits   : [sequence_length, batch_size, vocab_size]
        """
        embedded = self.embedding(tokens)          # [sequence_length, batch_size, embedding_size]
        out, hN  = self.rnn(embedded, h_0)         # [sequence_length, batch_size, hidden_size]
        logits   = self.proj(out)                  # [sequence_length, batch_size, vocab_size]
        return logits, hN


# =========================
# 1) Data loading & vocab
# =========================

TEXT = data_legacy.Field(tokenize='basic_english', lower=True)
train_data, _, test_data = datasets_legacy.PennTreebank.splits(TEXT)
TEXT.build_vocab(train_data)
vocab = TEXT.vocab
vocab_size = len(vocab)
print(f"Vocabulary Size: {vocab_size}")


# ==========================================
# 2) Flatten PTB splits into token 1-D streams
# ==========================================

def create_corpus(dataset, vocabulary):
    """Concatenate all examples into a single 1D LongTensor of token IDs."""
    chunks = []
    for ex in dataset.examples:
        ids = [vocabulary.stoi[token] for token in ex.text]
        chunks.append(torch.tensor(ids, dtype=torch.long))
    return torch.cat(chunks, dim=0)

train_corpus = create_corpus(train_data, vocab)
test_corpus  = create_corpus(test_data,  vocab)
print(f"Total tokens in training corpus: {len(train_corpus)}")
print(f"Total tokens in test corpus: {len(test_corpus)}")


# =====================================
# 3) Stream batching into [T, B] tensor
# =====================================

def batchify(data_1d: torch.Tensor, batch_size: int, device: torch.device):
    """
    Reshape a 1D token stream into [T, B] where B=batch_size parallel streams.
    Trims tail so T*B divides the length exactly.
    """
    n_batches = data_1d.size(0) // batch_size
    data_1d = data_1d.narrow(0, 0, n_batches * batch_size)
    data_tb = data_1d.view(batch_size, -1).t().contiguous()  # [T, B]
    return data_tb.to(device)


# ======================================================
# 4) TBPTT windowing via IterableDataset (stateful order)
# ======================================================

class StreamBatchDataset(IterableDataset):
    """
    Iterates over a time-major tensor [T, batch_size] and yields TBPTT windows:
      data_window   : [seq_len, batch_size]
      target_window : [seq_len * batch_size]  (flattened next-token targets)
    Order is sequential; do NOT shuffle if you want stateful training.
    """
    def __init__(self, source_tb: torch.Tensor, sequence_length: int):
        super().__init__()
        self.source = source_tb
        self.seq_len = sequence_length

    def __iter__(self):
        T = self.source.size(0)
        for i in range(0, T - 1, self.seq_len):
            cur_len = min(self.seq_len, (T - 1) - i)
            data = self.source[i : i + cur_len]            # [cur_len, B]
            target = self.source[i + 1 : i + 1 + cur_len]  # [cur_len, B]
            yield data, target.reshape(-1)                  # [cur_len*B]


# ===============
# 5) Hyperparams
# ===============

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

train_ds = StreamBatchDataset(train_tb, sequence_length)
test_ds  = StreamBatchDataset(test_tb,  sequence_length)

# Each dataset item is already a full window batch → no extra batching by DataLoader
train_loader = DataLoader(train_ds, batch_size=None, shuffle=False, num_workers=0)
test_loader  = DataLoader(test_ds,  batch_size=None, shuffle=False, num_workers=0)


# ===========
# 6) Model
# ===========

model = MinimalRNNLM(vocab_size, embedding_size, hidden_size, num_layers=num_layers).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)


# ==========================
# 7) Training (stateful TBPTT)
# ==========================

def train_one_epoch(model, loader, optimizer, clip_norm, device):
    """
    Stateful TBPTT training over sequential windows from DataLoader.
    Initializes hidden once, carries it across windows, and detaches at boundaries.
    Returns average loss over steps.
    """
    model.train()
    total_loss = 0.0
    steps = 0

    hidden = torch.zeros(num_layers, batch_size, hidden_size, device=device)

    for data_win, tgt_win in loader:
        # data_win: [seq_len, B]
        # tgt_win : [seq_len*B]
        hidden = hidden.detach()  # TBPTT boundary
        optimizer.zero_grad()

        logits, hidden = model(data_win, hidden)
        loss = criterion(logits.reshape(-1, vocab_size), tgt_win.reshape(-1))
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
        optimizer.step()

        total_loss += loss.item()
        steps += 1

    return total_loss / max(1, steps)


# ==========================
# 8) Evaluation (stateful)
# ==========================

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
        logits, hidden = model(data_win, hidden)  # [seq_len, B, V]
        hidden = hidden.detach()

        V = logits.size(-1)
        loss = criterion(logits.reshape(-1, V), tgt_win.reshape(-1))
        n_tokens = tgt_win.numel()
        total_loss += n_tokens * loss.item()
        total_tokens += n_tokens

        preds = torch.argmax(logits, dim=-1).reshape(-1)  # [seq_len*B]
        total_correct += (preds == tgt_win.reshape(-1)).sum().item()

    avg_loss = total_loss / max(1, total_tokens)
    ppl = math.exp(avg_loss)
    acc = 100.0 * total_correct / max(1, total_tokens)
    return avg_loss, ppl, acc


# ================
# 9) Run training
# ================

train_losses = []
print("\nStarting stateful training with MinimalRNN + DataLoader-backed TBPTT...")
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


# =========================
# 10) Plot training curve
# =========================

plt.figure(figsize=(10, 5))
plt.plot(range(1, epochs + 1), train_losses, marker='o', linestyle='--')
plt.title("Training Loss Over Epochs (Stateful TBPTT with Handmade RNN)")
plt.xlabel("Epoch")
plt.ylabel("Average Loss")
plt.grid(True)
plt.xticks(range(1, epochs + 1))
plt.show()
