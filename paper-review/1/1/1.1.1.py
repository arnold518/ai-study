import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import time
import math
import matplotlib.pyplot as plt

from torchtext import data as data_legacy
from torchtext import datasets as datasets_legacy

# --- 1. Data Loading and Preprocessing ---
TEXT = data_legacy.Field(
    tokenize='basic_english',
    lower=True
)
train_data, _, test_data = datasets_legacy.PennTreebank.splits(TEXT)
TEXT.build_vocab(train_data)
vocab = TEXT.vocab
dict_size = len(vocab)
print(f"Vocabulary Size: {dict_size}")

# --- 2. Create the Corpus ---
def create_corpus(dataset, vocabulary):
    """Concatenate all examples' tokens into a single 1D LongTensor of token IDs."""
    corpus_tensors = []
    for example in dataset.examples:
        numerical_ids = [vocabulary.stoi[token] for token in example.text]
        tensor = torch.tensor(numerical_ids, dtype=torch.long)
        corpus_tensors.append(tensor)
    return torch.cat(corpus_tensors)

train_corpus = create_corpus(train_data, vocab)
test_corpus = create_corpus(test_data, vocab)

print(f"Total tokens in training corpus: {len(train_corpus)}")
print(f"Total tokens in test corpus: {len(test_corpus)}")
print(f"First 100 tokens (numerical IDs): {train_corpus[:100]}")
print(f"First 100 tokens (natural language): {[vocab.itos[id] for id in train_corpus[:100]]}")

# --- 3. Dataset Class ---
class PTBSequenceDataset:
    """Slice a long 1D corpus into (input, target) pairs of fixed sequence length."""
    def __init__(self, corpus, sequence_length):
        self.corpus = corpus
        self.seq_len = sequence_length

    def __len__(self):
        return len(self.corpus) - self.seq_len

    def __getitem__(self, idx):
        inputs = self.corpus[idx : idx + self.seq_len]
        targets = self.corpus[idx + 1 : idx + self.seq_len + 1]
        return inputs, targets

# --- 4. RNN Model ---
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

# --- New Function to Evaluate the Model ---
def evaluate(model, data_source, criterion, device):
    """
    Compute average cross-entropy over a DataLoader of (inputs, targets) batches.
    Resets hidden state per batch; no gradient computation.
    """
    model.eval()
    total_loss = 0.0
    V = model.linear.out_features
    L = model.rnn.num_layers
    H = model.rnn.hidden_size
    with torch.no_grad():
        for inputs, targets in data_source:
            inputs, targets = inputs.to(device), targets.to(device)
            inputs = inputs.t().contiguous()  # [T,B] for RNN
            hidden = torch.zeros(L, inputs.size(1), H, device=device)
            outputs, hidden = model(inputs, hidden)
            loss = criterion(outputs.reshape(-1, V), targets.reshape(-1))
            total_loss += loss.item()
    return total_loss / len(data_source)

def calculate_accuracy(model, data_loader, device):
    """
    Token-level accuracy over a DataLoader: argmax(next-token) vs targets.
    Resets hidden state per batch.
    """
    model.eval()
    L = model.rnn.num_layers
    H = model.rnn.hidden_size
    total_correct = 0
    total_words = 0
    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            inputs = inputs.t().contiguous()  # [T,B]
            hidden = torch.zeros(L, inputs.size(1), H, device=device)
            outputs, hidden = model(inputs, hidden)  # outputs: [T,B,V]
            predicted_ids = torch.argmax(outputs, dim=-1)  # [T,B]
            total_correct += (predicted_ids.t() == targets).sum().item()
            total_words += targets.numel()
    return (total_correct / total_words) * 100

# --- 5. Setup ---
# Hyperparameters
sequence_length = 35
num_layers = 3
embedding_size = 256
hidden_size = 256
batch_size = 128
learning_rate = 0.003
epochs = 20
grad_clip = 0.25
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_set = PTBSequenceDataset(train_corpus[:100000], sequence_length)
test_set = PTBSequenceDataset(test_corpus, sequence_length)

train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

model = RNNModel(dict_size, embedding_size, hidden_size, num_layers).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# --- 6. Training and Evaluation Loop ---
train_losses = []

for epoch in range(epochs):
    start_time = time.time()
    total_loss = 0.0
    model.train()

    for i, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        inputs = inputs.t().contiguous()  # [T,B]
        hidden = torch.zeros(num_layers, inputs.size(1), hidden_size, device=device)

        optimizer.zero_grad()
        outputs, hidden = model(inputs, hidden)
        loss = criterion(outputs.reshape(-1, dict_size), targets.reshape(-1))
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        total_loss += loss.item()

    avg_train_loss = total_loss / len(train_loader)
    train_losses.append(avg_train_loss)
    
    test_loss = evaluate(model, test_loader, criterion, device)
    test_perplexity = math.exp(test_loss)
    test_accuracy = calculate_accuracy(model, test_loader, device)
    
    epoch_time = time.time() - start_time
    print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {avg_train_loss:.4f}, Test Loss: {test_loss:.4f}, Test Perplexity: {test_perplexity:.2f}, Test Accuracy: {test_accuracy:.2f}%, Time: {epoch_time:.2f}s')

print("\nTraining finished.")

final_accuracy = calculate_accuracy(model, test_loader, device)
print(f"\nFinal Test Accuracy: {final_accuracy:.2f}%")

# --- 7. Final Results ---
plt.figure(figsize=(10, 5))
plt.plot(range(1, epochs + 1), train_losses, marker='o', linestyle='--')
plt.title("Training Loss Over Epochs (Unstateful TBPTT)")
plt.xlabel("Epoch")
plt.ylabel("Average Loss")
plt.grid(True)
plt.xticks(range(1, epochs + 1))
plt.show()
