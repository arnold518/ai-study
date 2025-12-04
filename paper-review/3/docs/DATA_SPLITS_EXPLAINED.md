# Data Splits Explained: Train, Validation, and Test

This document explains how train, validation, and test data are created, stored, and used throughout the project.

## Overview

The project uses three data splits following standard machine learning practices:

| Split | Size | Purpose | When Used |
|-------|------|---------|-----------|
| **Train** | 897,566 pairs | Update model parameters | Every training step |
| **Validation** | 1,896 pairs | Monitor training, select best model | Every `eval_every` epochs |
| **Test** | 4,061 pairs | Final evaluation | After training complete |

## Data Flow Diagram

```
Raw Data Sources
    ├── moo (train, validation, test)
    ├── tatoeba (validation, test)
    └── aihub (train)
         ↓
    [download_data.py]
         ↓
data/raw/{source}/{split}.{ko,en}
         ↓
    [split_data.py]
    (merge + filter + split)
         ↓
data/processed/{split}.{ko,en}
         ↓
    [train.py]
    (create datasets + dataloaders)
         ↓
    Training Loop
    ├── Train data → update weights
    ├── Validation data → monitor, save best model
    └── Test data → final evaluation (not yet implemented)
```

## 1. Data Creation

### Step 1: Download Raw Data (`scripts/download_data.py`)

**Purpose**: Download datasets from HuggingFace Hub

**Code Reference**: `scripts/download_data.py:26-73` (Moo dataset example)

```python
def download_moo_dataset(raw_dir):
    """Download Moo/korean-parallel-corpora dataset."""
    ds = load_dataset("Moo/korean-parallel-corpora")

    # Save to text files
    for split in ['train', 'validation', 'test']:
        ko_path = output_dir / f"{split}.ko"
        en_path = output_dir / f"{split}.en"

        with open(ko_path, 'w', encoding='utf-8') as f_ko, \
             open(en_path, 'w', encoding='utf-8') as f_en:
            for item in ds[split]:
                f_ko.write(item['ko'].strip() + '\n')
                f_en.write(item['en'].strip() + '\n')
```

**Output Structure**:
```
data/raw/
├── moo/
│   ├── train.ko/en       # 96k pairs
│   ├── validation.ko/en  # 1k pairs
│   └── test.ko/en        # 2k pairs
├── tatoeba/
│   ├── validation.ko/en  # 1k pairs
│   └── test.ko/en        # 2.4k pairs
└── aihub/
    └── train.ko/en       # 1.6M pairs
```

**Key Point**: Different sources provide different splits. For example, Tatoeba only has validation and test (no training data).

### Step 2: Merge and Split (`scripts/split_data.py`)

**Purpose**: Merge multiple datasets by split type and apply filtering

**Code Reference**: `scripts/split_data.py:141-186`

```python
def merge_and_process_split(split_name, datasets, min_len, max_len, max_ratio):
    """
    Merge multiple datasets for a split and apply filtering.

    Args:
        split_name: 'train', 'validation', or 'test'
        datasets: List of (dataset_name, ko_lines, en_lines) tuples
    """
    # Merge all datasets for this split
    merged_ko = []
    merged_en = []

    for dataset_name, ko_lines, en_lines in datasets:
        merged_ko.extend(ko_lines)  # Concatenate all sources
        merged_en.extend(en_lines)

    # Apply filtering (length, ratio checks)
    filtered_ko, filtered_en, filter_stats = clean_and_filter(
        merged_ko, merged_en, min_len, max_len, max_ratio
    )

    return filtered_ko, filtered_en, stats
```

**Filtering Applied** (`scripts/split_data.py:44-96`):
- Remove empty sentences
- Remove too short sentences (< 3 characters)
- Remove too long sentences (> 150 characters)
- Remove misaligned pairs (length ratio > 3.0)

**Output Structure**:
```
data/processed/
├── train.ko/en       # 897,566 pairs (from moo + aihub)
├── validation.ko/en  # 1,896 pairs (from moo + tatoeba)
├── test.ko/en        # 4,061 pairs (from moo + tatoeba)
└── statistics.json   # Shows source contribution
```

**Example statistics.json**:
```json
{
  "train": {
    "num_pairs": 897566,
    "sources": {
      "moo": 96000,
      "aihub": 801566
    }
  },
  "validation": {
    "num_pairs": 1896,
    "sources": {
      "moo": 996,
      "tatoeba": 900
    }
  }
}
```

## 2. Data Loading for Training

### Step 3: Create PyTorch Datasets (`scripts/train.py:80-126`)

**Purpose**: Load processed data into PyTorch Dataset objects

```python
# Load training data
train_dataset = TranslationDataset(
    train_ko_path,  # 'data/processed/train.ko'
    train_en_path,  # 'data/processed/train.en'
    ko_tokenizer,
    en_tokenizer,
    max_len=config.max_seq_length
)

# Load validation data
val_dataset = TranslationDataset(
    val_ko_path,  # 'data/processed/validation.ko'
    val_en_path,  # 'data/processed/validation.en'
    ko_tokenizer,
    en_tokenizer,
    max_len=config.max_seq_length
)

# Keep reference to full validation dataset for BLEU computation
full_val_dataset = val_dataset

# Optional: Use small subset for quick testing
if args.small:
    train_dataset = Subset(train_dataset, range(min(1000, len(train_dataset))))
    val_dataset = Subset(val_dataset, range(min(100, len(val_dataset))))
```

**Key Point**: When using `--small` flag, we keep `full_val_dataset` reference so BLEU computation uses the full validation set (more representative) even though the validation loop uses a subset (faster).

### Step 4: Create DataLoaders (`scripts/train.py:129-141`)

**Purpose**: Batch data and apply padding

```python
# Training DataLoader
train_loader = DataLoader(
    train_dataset,
    batch_size=config.batch_size,  # e.g., 32
    shuffle=True,  # ✓ Shuffle for training
    collate_fn=collate_fn,  # Padding and masking
    num_workers=config.num_workers
)

# Validation DataLoader
val_loader = DataLoader(
    val_dataset,
    batch_size=config.batch_size,
    shuffle=False,  # ✗ Don't shuffle for validation
    collate_fn=collate_fn,
    num_workers=config.num_workers
)
```

**Important Differences**:
- **Training**: Shuffled (prevents learning order-dependent patterns)
- **Validation**: Not shuffled (reproducible results)

## 3. How Data is Used During Training

### Training Data Usage (`src/training/trainer.py:58-107`)

**Purpose**: Update model parameters to minimize loss

```python
def train_epoch(self):
    """Train for one epoch."""
    self.model.train()  # Enable dropout and batch norm training mode
    total_loss = 0

    # Iterate through all training batches
    for batch in tqdm(self.train_loader, desc="Training"):
        src = batch['src'].to(self.device)  # Korean sentences
        tgt = batch['tgt'].to(self.device)  # English sentences

        # Prepare decoder inputs and targets
        # Input: [<bos>, token1, token2, ...]
        # Target: [token1, token2, ..., <eos>]
        tgt_input = tgt[:, :-1]
        tgt_output = tgt[:, 1:]

        # Forward pass
        logits = self.model(src, tgt_input, src_mask, tgt_input_mask, cross_mask)

        # Compute loss (excludes padding tokens)
        loss = self.criterion(logits, tgt_output)

        # Backward pass - UPDATE WEIGHTS
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()  # ← Weights updated here!

        total_loss += loss.item()

    return total_loss / num_batches
```

**Key Points**:
1. **Model in training mode**: `model.train()` enables dropout (randomly drops neurons)
2. **Weights updated**: Every batch updates model parameters via backpropagation
3. **Shuffled data**: Each epoch sees data in different order
4. **All batches used**: Iterates through entire training set

### Validation Data Usage (`src/training/trainer.py:109-143`)

**Purpose**: Monitor training progress WITHOUT updating weights

```python
def validate(self):
    """Validate the model."""
    self.model.eval()  # Disable dropout and batch norm training mode
    total_loss = 0

    with torch.no_grad():  # ← No gradient computation!
        for batch in tqdm(self.val_loader, desc="Validation"):
            src = batch['src'].to(self.device)
            tgt = batch['tgt'].to(self.device)

            # Forward pass (same as training)
            logits = self.model(src, tgt_input, src_mask, tgt_input_mask, cross_mask)

            # Compute loss (but NO backward pass!)
            loss = self.criterion(logits, tgt_output)

            total_loss += loss.item()

    return total_loss / num_batches
```

**Key Differences from Training**:
1. **Model in eval mode**: `model.eval()` disables dropout (use all neurons)
2. **No gradients**: `torch.no_grad()` saves memory and computation
3. **No weight updates**: No `.backward()` or `.step()` calls
4. **Not shuffled**: Reproducible results across runs

**Why Two Modes?**
- **Training mode**: Dropout provides regularization (prevents overfitting)
- **Eval mode**: Dropout disabled for stable predictions (use full model capacity)

### BLEU Computation on Validation Data (`src/training/trainer.py:145-189`)

**Purpose**: Measure actual translation quality (beyond just loss)

```python
def compute_bleu_score(self, num_samples=100):
    """Compute BLEU score on a subset of validation data."""
    if not self.translator or not self.val_dataset:
        return None

    self.model.eval()

    # Sample random indices from validation set
    num_samples = min(num_samples, len(self.val_dataset))
    indices = torch.randperm(len(self.val_dataset))[:num_samples]

    predictions = []
    references = []

    with torch.no_grad():
        for idx in tqdm(indices, desc="  BLEU", leave=False):
            # Get original text (not tokenized)
            src_text = self.val_dataset.src_lines[idx].strip()
            tgt_text = self.val_dataset.tgt_lines[idx].strip()

            # Translate using trained model
            pred_text = self.translator.translate(src_text, method='greedy')

            predictions.append(pred_text)
            references.append(tgt_text)

    # Compute corpus-level BLEU
    bleu = compute_bleu(predictions, references)
    return bleu.score
```

**Key Points**:
1. **Random sampling**: Uses 100 samples (not all 1,896) for speed
2. **Full dataset reference**: Even with `--small` flag, uses full validation set for representative BLEU
3. **Greedy decoding**: Faster than beam search, good enough for monitoring
4. **Text-level comparison**: BLEU compares actual strings, not token IDs

### Training Loop Integration (`src/training/trainer.py:225-323`)

**Purpose**: Orchestrate training, validation, and checkpointing

```python
def train(self):
    """Full training loop with checkpointing."""
    for epoch in range(self.config.num_epochs):

        # 1. TRAIN on training data (updates weights)
        train_loss = self.train_epoch()

        # 2. VALIDATE on validation data (every eval_every epochs)
        if (epoch + 1) % self.config.eval_every == 0:
            val_loss = self.validate()  # No weight updates!

            # 3. Compute BLEU on validation data
            if self.translator:
                bleu_score = self.compute_bleu_score(num_samples=100)
                print(f"  BLEU Score: {bleu_score:.2f}")

            # 4. Generate inference examples from validation data
            if self.translator:
                examples = self.generate_inference_examples(num_examples=2)
                # Display source, reference, prediction

            # 5. Save best model based on VALIDATION loss
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                save_checkpoint(..., 'best_model.pt')

            # 6. Save best model based on VALIDATION BLEU
            if bleu_score > self.best_bleu:
                self.best_bleu = bleu_score
                save_checkpoint(..., 'best_bleu_model.pt')

        # 7. Save periodic checkpoint
        if (epoch + 1) % self.config.save_every == 0:
            save_checkpoint(..., f'checkpoint_epoch_{epoch+1}.pt')
```

**Frequency of Operations**:
- **Training**: Every epoch (all training data)
- **Validation loss**: Every `eval_every` epochs (default: 1)
- **BLEU score**: Every `eval_every` epochs (100 samples)
- **Inference examples**: Every `eval_every` epochs (2 examples)
- **Checkpoint saving**: Every `save_every` epochs (default: 5)

## 4. Test Data Usage (Not Yet Implemented)

### Purpose of Test Data

Test data should be used **only once** at the very end of the project:

```python
# Example future implementation
def final_evaluation():
    """Evaluate best model on test set (run only once!)"""

    # Load best model
    checkpoint = torch.load('checkpoints/best_bleu_model.pt')
    model.load_state_dict(checkpoint['model_state_dict'])

    # Load test data (first time seeing this data!)
    test_dataset = TranslationDataset(
        'data/processed/test.ko',
        'data/processed/test.en',
        ko_tokenizer,
        en_tokenizer
    )

    # Compute BLEU on ALL test samples
    bleu_score = compute_bleu_on_dataset(test_dataset)

    print(f"Final Test BLEU: {bleu_score:.2f}")
    return bleu_score
```

**Critical Rules**:
1. **Never train on test data**: Would give artificially high scores
2. **Never tune hyperparameters on test data**: Use validation for that
3. **Evaluate only once**: Multiple evaluations = data leakage
4. **Use for final reporting**: Report test BLEU in paper/documentation

## 5. Why Three Splits?

### Problem: Overfitting

Without proper splits, the model might "memorize" training data rather than learn generalizable patterns.

### Solution: Three-Split Approach

```
┌─────────────┐
│ Train Data  │ → Update model weights
└─────────────┘
       ↓
┌─────────────┐
│  Val Data   │ → Monitor training, select best model, tune hyperparameters
└─────────────┘
       ↓
┌─────────────┐
│ Test Data   │ → Final unbiased evaluation (use only once!)
└─────────────┘
```

### Example Scenario

**Without validation split** (only train + test):
```
1. Train model
2. Evaluate on test set: BLEU = 15.0
3. Try different learning rate
4. Evaluate on test set: BLEU = 18.0  ← Data leakage!
5. Try different model size
6. Evaluate on test set: BLEU = 20.0  ← More data leakage!
7. Report "Test BLEU: 20.0" ← Inflated score!
```

**With validation split** (train + val + test):
```
1. Train model
2. Evaluate on validation set: BLEU = 15.0
3. Try different learning rate
4. Evaluate on validation set: BLEU = 18.0  ✓ OK to iterate
5. Try different model size
6. Evaluate on validation set: BLEU = 20.0  ✓ OK to iterate
7. Select best model based on validation
8. Evaluate ONCE on test set: BLEU = 19.5  ← Unbiased estimate!
9. Report "Test BLEU: 19.5" ← Honest score!
```

## 6. Data Split Best Practices

### Training Data
- **Use all of it**: More data = better learning
- **Shuffle every epoch**: Prevents learning order-dependent patterns
- **Augmentation possible**: Could add noise, back-translation, etc.

### Validation Data
- **Don't shuffle**: Reproducible results across runs
- **Use frequently**: Monitor after every epoch if computationally feasible
- **Sample for BLEU**: 100 samples enough for reliable estimate
- **Multiple metrics**: Track both loss and BLEU

### Test Data
- **Don't touch until the end**: Preserve as unbiased evaluation
- **Don't shuffle**: Reproducible results
- **Use all of it**: Report BLEU on full test set
- **Report only once**: Multiple evaluations = data leakage

## 7. Common Pitfalls

### ❌ Training on Validation Data
```python
# WRONG: Training and validation combined
all_data = train_dataset + val_dataset  # Don't do this!
train_loader = DataLoader(all_data, shuffle=True)
```

### ❌ Tuning on Test Data
```python
# WRONG: Optimizing hyperparameters on test set
for lr in [0.001, 0.01, 0.1]:
    model.train(lr=lr)
    test_bleu = evaluate(test_dataset)  # Don't do this!
    if test_bleu > best_bleu:
        best_lr = lr
```

### ❌ Validation Without No-Grad
```python
# WRONG: Computing gradients during validation
def validate(self):
    self.model.eval()
    # Missing: with torch.no_grad()
    for batch in self.val_loader:
        loss = self.model(batch)  # Still computing gradients!
```

### ✅ Correct Usage
```python
# CORRECT: Proper separation and usage
train_dataset = TranslationDataset(train_path, ...)
val_dataset = TranslationDataset(val_path, ...)      # Separate!
test_dataset = TranslationDataset(test_path, ...)    # Separate!

# Train with weight updates
train_loss = train_epoch(train_dataset)

# Validate without weight updates
with torch.no_grad():
    val_loss = validate(val_dataset)

# Test only at the very end
with torch.no_grad():
    test_bleu = final_evaluate(test_dataset)  # Once!
```

## 8. Summary Table

| Aspect | Training Data | Validation Data | Test Data |
|--------|--------------|-----------------|-----------|
| **Size** | 897k pairs | 1.9k pairs | 4k pairs |
| **Purpose** | Learn patterns | Monitor & select model | Final evaluation |
| **Frequency** | Every epoch | Every eval_every epochs | Once at end |
| **Shuffled?** | ✓ Yes | ✗ No | ✗ No |
| **Gradients?** | ✓ Yes | ✗ No | ✗ No |
| **Weight updates?** | ✓ Yes | ✗ No | ✗ No |
| **Model mode** | `train()` | `eval()` | `eval()` |
| **Used for** | Backprop | Best model selection | Unbiased BLEU |
| **Can iterate?** | ✓ Yes (multiple epochs) | ✓ Yes (tune hyperparams) | ✗ No (once only!) |
| **Dropout** | Enabled | Disabled | Disabled |

## 9. Code References Summary

| File | Purpose | Key Lines |
|------|---------|-----------|
| `scripts/download_data.py` | Download raw datasets | 26-73 (Moo), 76-130 (Tatoeba) |
| `scripts/split_data.py` | Merge and filter splits | 141-186 (merge), 44-96 (filter) |
| `scripts/train.py` | Load and create datasets | 82-126 (datasets), 129-141 (loaders) |
| `src/training/trainer.py` | Use data in training loop | 58-107 (train), 109-143 (validate), 145-189 (BLEU) |
| `src/data/dataset.py` | PyTorch Dataset implementation | 11-95 (TranslationDataset), 98-149 (collate_fn) |

## 10. Workflow Summary

```bash
# 1. Download raw data
/home/arnold/venv/bin/python scripts/download_data.py all

# 2. Merge and split
/home/arnold/venv/bin/python scripts/split_data.py

# 3. Train tokenizers
/home/arnold/venv/bin/python scripts/train_tokenizer.py

# 4. Start training (uses train + validation)
/home/arnold/venv/bin/python scripts/train.py

# During training:
#   - Train data: Updates weights every batch
#   - Validation data: Monitors every epoch, computes BLEU, saves best model

# 5. Final evaluation (after training complete)
# TODO: Implement test evaluation script
# /home/arnold/venv/bin/python scripts/evaluate.py --checkpoint best_bleu_model.pt
#   - Test data: Computes final BLEU (once!)
```

## Conclusion

The three-split approach (train/validation/test) is fundamental to machine learning:
- **Train**: Learn patterns by updating weights
- **Validation**: Monitor learning without touching weights, select best model
- **Test**: Unbiased final evaluation (use once!)

This separation prevents overfitting and ensures honest evaluation of model performance.
