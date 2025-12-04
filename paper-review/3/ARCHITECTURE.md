# Project Architecture - Korean-English NMT

**Status:** Minimal skeleton ready for Phase 1 implementation
**Current Phase:** 1.1 Complete (Data Download & Merging) ✅

---

## Directory Structure

```
.
├── scripts/              # Executable scripts
│   ├── download_data.py      ✅ COMPLETE - Download datasets from HuggingFace
│   ├── split_data.py         ✅ COMPLETE - Merge & clean datasets
│   ├── train_tokenizer.py    📝 SKELETON - Train SentencePiece models
│   ├── train.py              📝 SKELETON - Training pipeline
│   └── translate.py          📝 SKELETON - Inference pipeline
│
├── src/                  # Source code modules
│   ├── data/             # Data processing
│   │   ├── tokenizer.py      📝 SKELETON - SentencePiece tokenizer wrapper
│   │   └── dataset.py        📝 SKELETON - PyTorch Dataset for parallel corpus
│   │
│   ├── models/           # Model architectures
│   │   └── transformer/      🔧 EXISTING - Transformer components (need review)
│   │       ├── transformer.py
│   │       ├── attention.py
│   │       ├── encoder.py
│   │       ├── decoder.py
│   │       ├── feedforward.py
│   │       ├── positional_encoding.py
│   │       └── embeddings.py
│   │
│   ├── training/         # Training infrastructure
│   │   ├── trainer.py        🔧 EXISTING - Training loop (needs review)
│   │   ├── optimizer.py      🔧 EXISTING - Noam scheduler (needs review)
│   │   └── losses.py         🔧 EXISTING - Label smoothing (needs review)
│   │
│   ├── inference/        # Decoding strategies
│   │   ├── beam_search.py    🔧 EXISTING (needs review)
│   │   ├── greedy_search.py  🔧 EXISTING (needs review)
│   │   └── translator.py     🔧 EXISTING (needs review)
│   │
│   └── utils/            # Helper functions
│       ├── masking.py        🔧 EXISTING (needs review)
│       ├── metrics.py        🔧 EXISTING (needs review)
│       └── checkpointing.py  🔧 EXISTING (needs review)
│
├── config/               # Configuration files
│   ├── base_config.py        ✅ Shared settings
│   └── transformer_config.py ✅ Transformer hyperparameters
│
├── data/                 # Data storage
│   ├── raw/              # Downloaded datasets (by source)
│   │   ├── moo/          # train/validation/test.{ko,en}
│   │   ├── tatoeba/      # validation/test.{ko,en}
│   │   └── aihub/        # train.{ko,en}
│   │
│   ├── processed/        # Unified, cleaned datasets ✅
│   │   ├── train.{ko,en}         ✅ 897k pairs
│   │   ├── validation.{ko,en}    ✅ 1.9k pairs
│   │   ├── test.{ko,en}          ✅ 4k pairs
│   │   └── statistics.json       ✅
│   │
│   └── vocab/            # Tokenizer models (to be created)
│       ├── ko_spm.model      📝 TODO
│       ├── ko_spm.vocab      📝 TODO
│       ├── en_spm.model      📝 TODO
│       └── en_spm.vocab      📝 TODO
│
├── checkpoints/          # Saved models
├── logs/                 # Training logs
└── outputs/              # Generated translations

Legend:
  ✅ COMPLETE   - Fully implemented and tested
  📝 SKELETON   - Minimal structure with TODOs
  🔧 EXISTING   - Previously created, needs review/testing
```

---

## Implementation Pipeline

### ✅ Phase 1.1: Data Acquisition (COMPLETE)

**Purpose:** Download and merge multiple datasets into unified splits

**Scripts:**
- `scripts/download_data.py` - Downloads Moo, Tatoeba, AIHub datasets
- `scripts/split_data.py` - Merges datasets, applies filtering, creates unified splits

**Usage:**
```bash
# Download datasets
/home/arnold/venv/bin/python scripts/download_data.py all

# Merge and clean
/home/arnold/venv/bin/python scripts/split_data.py
```

**Output:** `data/processed/train.{ko,en}`, `validation.{ko,en}`, `test.{ko,en}`

---

### 📝 Phase 1.2: Tokenization (NEXT)

**Purpose:** Train subword tokenizers for Korean and English

**Key Module:** `src/data/tokenizer.py`
- Class: `SentencePieceTokenizer`
- Methods: `tokenize()`, `detokenize()`, `encode_ids()`, `decode_ids()`

**Script:** `scripts/train_tokenizer.py`

**Implementation Steps:**
1. Train SentencePiece model on `data/processed/train.ko` → `data/vocab/ko_spm.model`
2. Train SentencePiece model on `data/processed/train.en` → `data/vocab/en_spm.model`
3. Test tokenization on sample sentences

**Key Decisions:**
- Vocab size: 16,000 (configurable)
- Model type: Unigram (SentencePiece default)
- Character coverage: 0.9995 (for Korean)
- Special tokens: `<pad>=0, <unk>=1, <s>=2, </s>=3`

---

### 📝 Phase 1.3: Dataset Implementation

**Purpose:** Create PyTorch Dataset for loading and batching

**Key Module:** `src/data/dataset.py`
- Class: `TranslationDataset`
- Function: `collate_fn()` for padding

**Implementation Steps:**
1. Load text files in `__init__`
2. Tokenize on-the-fly in `__getitem__` (or pre-tokenize)
3. Add BOS/EOS tokens
4. Implement `collate_fn` for padding
5. Test with DataLoader

**Data Flow:**
```
Text file → Load → Tokenize → Add BOS/EOS → Tensor → Batch → Pad → Model
```

---

### 🔧 Phase 2: Model & Training (REVIEW NEEDED)

**Purpose:** Implement Transformer architecture and training loop

**Key Modules:**
- `src/models/transformer/` - Model architecture
- `src/training/trainer.py` - Training loop
- `src/training/optimizer.py` - Noam learning rate scheduler
- `src/training/losses.py` - Label smoothing loss

**Status:** Components exist from ROADMAP template, need review and testing

**Implementation Steps:**
1. Review existing Transformer implementation
2. Test model forward pass with dummy data
3. Review training loop and optimizer
4. Start training on small subset
5. Scale to full dataset

---

### 🔧 Phase 3: Inference (LATER)

**Purpose:** Generate translations from trained model

**Key Modules:**
- `src/inference/greedy_search.py` - Fast, simple decoding
- `src/inference/beam_search.py` - Better quality decoding
- `scripts/translate.py` - User interface

**Status:** Components exist, need implementation

---

## Data Flow Diagram

```
┌─────────────────┐
│ Raw Datasets    │ (download_data.py)
│ - moo           │
│ - tatoeba       │
│ - aihub         │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Merged Dataset  │ (split_data.py)
│ - train.ko/en   │
│ - val.ko/en     │
│ - test.ko/en    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Tokenizers      │ (train_tokenizer.py) 📝 NEXT
│ - ko_spm.model  │
│ - en_spm.model  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ PyTorch Dataset │ (dataset.py) 📝 NEXT
│ TranslationData │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ DataLoader      │ (train.py)
│ Batching+Padding│
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Transformer     │ (model/)
│ Training        │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Trained Model   │
│ Checkpoints     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Translation     │ (translate.py)
│ Inference       │
└─────────────────┘
```

---

## Key Design Decisions

### 1. **Tokenization: SentencePiece (Unigram)**
**Why?**
- Language-agnostic (same approach for Korean & English)
- Handles OOV through subword decomposition
- Industry standard (used in production NMT)
- No preprocessing needed (handles raw text)

**Alternative considered:** Mecab + BPE (more complex, language-specific)

---

### 2. **Unified Dataset Structure**
**Why?**
- Combine multiple data sources for larger training set
- Single vocabulary and processing pipeline
- Easier to manage than per-source datasets

**Output:**
- `data/processed/` contains merged, cleaned splits
- Statistics track contribution from each source

---

### 3. **Vocabulary Handled by SentencePiece**
**Why?**
- SentencePiece has built-in vocabulary management
- `encode_ids()` returns token IDs directly
- Simpler than maintaining separate Vocabulary class

**Removed:** `src/data/vocabulary.py` (redundant)

---

### 4. **Configuration Hierarchy**
**Structure:**
- `config/base_config.py` - Shared settings (batch size, device, etc.)
- `config/transformer_config.py` - Model hyperparameters

**Benefits:** Easy to experiment with different configurations

---

## Implementation Priority

### Immediate (Phase 1.2-1.3):
1. ✅ `scripts/train_tokenizer.py` - Train SentencePiece
2. ✅ `src/data/tokenizer.py` - Implement wrapper
3. ✅ `src/data/dataset.py` - Implement Dataset
4. ✅ Test data pipeline end-to-end

### Soon (Phase 2):
5. Review `src/models/transformer/` components
6. Review `src/training/` components
7. Implement `scripts/train.py`
8. Start training

### Later (Phase 3):
9. Implement inference (`scripts/translate.py`)
10. Implement evaluation (BLEU scores)
11. Hyperparameter tuning

---

## Module Dependencies

```
scripts/train.py
    ├── config/transformer_config.py
    ├── src/data/tokenizer.py
    │   └── sentencepiece (external)
    ├── src/data/dataset.py
    │   └── src/data/tokenizer.py
    ├── src/models/transformer/transformer.py
    │   ├── encoder.py → attention.py, feedforward.py
    │   └── decoder.py → attention.py, feedforward.py
    └── src/training/trainer.py
        ├── optimizer.py
        └── losses.py

scripts/translate.py
    ├── src/data/tokenizer.py
    ├── src/models/transformer/transformer.py
    └── src/inference/beam_search.py (or greedy_search.py)
```

---

## Testing Strategy

### Unit Tests (per module):
- `src/data/tokenizer.py` → Test encode/decode
- `src/data/dataset.py` → Test loading and batching
- `src/models/transformer/` → Test forward pass shapes

### Integration Tests:
- End-to-end data pipeline
- Training loop (single batch)
- Inference (dummy model)

### System Tests:
- Train on small dataset (1000 samples)
- Evaluate BLEU on test set
- Compare with baseline

---

## Next Steps

**Immediate TODO (Phase 1.2):**

1. **Implement `scripts/train_tokenizer.py`:**
   ```python
   import sentencepiece as spm

   spm.SentencePieceTrainer.train(
       input='data/processed/train.ko',
       model_prefix='data/vocab/ko_spm',
       vocab_size=16000,
       ...
   )
   ```

2. **Implement `src/data/tokenizer.py`:**
   ```python
   class SentencePieceTokenizer:
       def __init__(self, model_path):
           self.sp = spm.SentencePieceProcessor(model_file=model_path)

       def encode_ids(self, text):
           return self.sp.encode(text, out_type=int)
   ```

3. **Implement `src/data/dataset.py`:**
   ```python
   def __getitem__(self, idx):
       src_ids = self.src_tokenizer.encode_ids(self.src_lines[idx])
       src_ids = [BOS] + src_ids + [EOS]
       return torch.tensor(src_ids)
   ```

4. **Test pipeline:**
   ```bash
   /home/arnold/venv/bin/python scripts/train_tokenizer.py
   /home/arnold/venv/bin/python scripts/train.py  # Should load data
   ```

---

## Questions to Resolve

1. **Shared vs Separate Vocabularies?**
   - Current: Separate (ko_spm, en_spm)
   - Alternative: Shared vocabulary (single model)

2. **Pre-tokenize or On-the-fly?**
   - Current skeleton: On-the-fly in `__getitem__`
   - Alternative: Pre-tokenize and save token IDs

3. **Maximum sequence length?**
   - Current: 5000 in config (for positional encoding)
   - Training: Could use 150 (from split_data filter)

4. **Batch size?**
   - Need to determine based on GPU memory
   - Start with small (16-32) and increase

---

**Status:** Ready to implement Phase 1.2 (Tokenization)
