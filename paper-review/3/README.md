# Korean-English Neural Machine Translation

A comparative study of neural machine translation models for Korean-English translation, implementing multiple architectures from foundational papers.

## Models Implemented

- **Transformer** (Attention Is All You Need) - Primary focus
- **Seq2Seq with Attention** (Bahdanau et al.) - Planned
- **Seq2Seq Baseline** - Planned

## Project Structure

```
.
├── config/                  # Model and training configurations
├── data/                    # Dataset storage
│   ├── raw/                # Raw parallel corpus
│   ├── processed/          # Preprocessed data
│   └── vocab/              # Vocabulary files
├── src/
│   ├── models/             # Model implementations
│   │   ├── transformer/    # Transformer architecture
│   │   ├── bahdanau/       # Bahdanau attention
│   │   └── seq2seq/        # Seq2Seq baseline
│   ├── data/               # Data processing
│   ├── training/           # Training utilities
│   ├── inference/          # Inference and decoding
│   └── utils/              # Shared utilities
├── scripts/                # Executable scripts
├── notebooks/              # Jupyter notebooks
├── tests/                  # Unit tests
├── checkpoints/            # Model checkpoints
├── logs/                   # Training logs
└── outputs/                # Translation outputs
```

## Setup

```bash
pip install -r requirements.txt
```

## Quick Start

All scripts use configuration from `config/base_config.py`. Edit this file to change parameters before running.

### Step 1: Download Data
```bash
/home/arnold/venv/bin/python scripts/download_data.py
```
Downloads Korean-English parallel corpora (Moo, Tatoeba, AI Hub).
Configure datasets in `config/base_config.py` → `datasets_to_download`.

### Step 2: Preprocess Data
```bash
/home/arnold/venv/bin/python scripts/split_data.py
```
Merges and cleans datasets from multiple sources.
Configure filtering in `config/base_config.py` → `min_length`, `max_length`, `length_ratio`.

### Step 3: Train Tokenizers
```bash
/home/arnold/venv/bin/python scripts/train_tokenizer.py
```
Trains SentencePiece tokenizers for Korean and English.
Configure vocab in `config/base_config.py` → `use_shared_vocab`, `vocab_size`.

### Step 4: Test Setup (Optional)
```bash
# Test tokenizers
/home/arnold/venv/bin/python tests/test_tokenizer.py

# Test dataset
/home/arnold/venv/bin/python tests/test_dataset.py
```

### Step 5: Train Model
```bash
/home/arnold/venv/bin/python scripts/train.py
```

### Step 6: Translate
```bash
/home/arnold/venv/bin/python scripts/translate.py --input "안녕하세요"
```

## Configuration

All parameters are in `config/base_config.py`:
- **Data sources**: `datasets_to_download` (moo, tatoeba, aihub)
- **Filtering**: `min_length`, `max_length`, `length_ratio`
- **Vocabulary**: `use_shared_vocab`, `vocab_size`, `character_coverage`
- **Training**: `batch_size`, `num_epochs`, `learning_rate`

See `ARCHITECTURE.md` and `ROADMAP.md` for detailed documentation.
