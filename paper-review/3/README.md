# Korean-English Neural Machine Translation

A research project implementing the Transformer architecture from "Attention Is All You Need" (Vaswani et al., 2017) for Korean-to-English translation.

## 📋 Table of Contents

- [Overview](#overview)
- [Quick Start](#quick-start)
- [Complete Workflow](#complete-workflow)
  - [1. Download Data](#1-download-data)
  - [2. Preprocess & Split Data](#2-preprocess--split-data)
  - [3. Train Tokenizers](#3-train-tokenizers)
  - [4. Train Model](#4-train-model)
  - [5. Evaluate Model](#5-evaluate-model)
  - [6. Translate](#6-translate)
  - [7. Visualize Attention](#7-visualize-attention)
  - [8. Analyze Training Logs](#8-analyze-training-logs)
- [Project Structure](#project-structure)
- [Configuration](#configuration)
- [Troubleshooting](#troubleshooting)

---

## 📖 Overview

This project implements a Transformer-based neural machine translation system for Korean-English translation following the original paper with optimizations for Korean language characteristics.

**Key Features:**
- ✅ Full Transformer architecture (encoder-decoder with multi-head attention)
- ✅ SentencePiece tokenization (language-agnostic subword units)
- ✅ Label smoothing, dropout, weight decay regularization
- ✅ KV caching for efficient inference
- ✅ Beam search with length normalization
- ✅ Attention visualization tools
- ✅ Comprehensive evaluation metrics (BLEU, chrF++)

**Dataset:**
- Training: ~897k Korean-English sentence pairs
- Sources: AI Hub, Korean Parallel Corpora (Moo), Tatoeba
- Validation: ~1.9k pairs | Test: ~4k pairs

**Model Size:**
- Parameters: ~39M
- Architecture: 6-layer encoder, 6-layer decoder, 8 attention heads
- Embedding dimension: 512 | Feedforward dimension: 2048

---

## 🚀 Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set up Python path (add to ~/.bashrc for convenience)
export PYTHON=/home/arnold/venv/bin/python

# 3. Download and prepare data (~5 minutes)
$PYTHON scripts/download_data.py all
$PYTHON scripts/split_data.py
$PYTHON scripts/train_tokenizer.py

# 4. Train model (~4-6 hours on CPU, ~1 hour on GPU)
$PYTHON scripts/train.py

# 5. Translate
$PYTHON scripts/translate.py --input "안녕하세요"
```

---

## 📚 Complete Workflow

### 1. Download Data

Download Korean-English parallel datasets from Hugging Face.

```bash
# Download all datasets (AI Hub + Moo + Tatoeba)
$PYTHON scripts/download_data.py all

# Or download individual datasets
$PYTHON scripts/download_data.py moo      # ~96k training pairs
$PYTHON scripts/download_data.py tatoeba  # ~1k validation + 2.4k test
$PYTHON scripts/download_data.py aihub    # ~1.6M pairs
```

**Output:** `data/raw/{moo,tatoeba,aihub}/`

**Options:**
- `--max-samples N`: Limit number of samples per dataset
- `--output-dir DIR`: Custom output directory

---

### 2. Preprocess & Split Data

Merge datasets, clean, filter, and split into train/val/test sets.

```bash
# Basic usage
$PYTHON scripts/split_data.py

# Custom filtering
$PYTHON scripts/split_data.py --min-len 1 --max-len 200 --length-ratio 3.5
```

**What it does:**
1. Merges all downloaded datasets by split type
2. Filters by length and length ratio (Korean/English)
3. Removes duplicates
4. Creates unified train/validation/test splits

**Output:** `data/processed/{train,validation,test}.{ko,en}`

---

### 3. Train Tokenizers

Train SentencePiece tokenizers for Korean and English.

```bash
# Train with default settings (16k vocab each)
$PYTHON scripts/train_tokenizer.py

# Custom vocabulary size
$PYTHON scripts/train_tokenizer.py --ko-vocab-size 32000 --en-vocab-size 24000
```

**Output:** `data/vocab/{ko,en}_spm.{model,vocab}`

---

### 4. Train Model

Train the Transformer model.

```bash
# Basic training
$PYTHON scripts/train.py

# Resume from checkpoint
$PYTHON scripts/train.py --resume checkpoints/best_model.pt

# Quick test on small subset
$PYTHON scripts/train.py --small
```

**Training Configuration:**
- Epochs: 50 (with early stopping patience=10)
- Batch size: 128 (effective: 256 with gradient accumulation)
- Learning rate: Noam scheduler with warmup (8000 steps)
- Regularization: Dropout=0.3, Label smoothing=0.05, Weight decay=1e-5

**Output:**
```
checkpoints/
├── best_model.pt           # Best model by validation loss
├── best_bleu_model.pt      # Best model by BLEU score
└── checkpoint_epoch_5.pt   # Periodic checkpoints

logs/
└── training_log_YYYYMMDD_HHMMSS.csv  # Detailed metrics per epoch
```

**Expected Training Time:**
- CPU (8 cores): ~4-6 hours for 20 epochs
- GPU (T4/V100): ~1-2 hours for 20 epochs

---

### 5. Evaluate Model

Compute comprehensive metrics on test set.

```bash
# Evaluate best model
$PYTHON scripts/evaluate.py

# Evaluate specific checkpoint
$PYTHON scripts/evaluate.py --checkpoint checkpoints/checkpoint_epoch_10.pt

# Use beam search (slower but better quality)
$PYTHON scripts/evaluate.py --method beam --beam-size 5

# Evaluate on subset for speed
$PYTHON scripts/evaluate.py --max-samples 500
```

**Output:** Prints BLEU, chrF++ scores and translation examples

---

### 6. Translate

Translate Korean text to English using trained model.

```bash
# Translate single sentence
$PYTHON scripts/translate.py --input "안녕하세요"

# Translate from file
$PYTHON scripts/translate.py --file input.txt

# Use beam search for better quality
$PYTHON scripts/translate.py \
    --input "한국어를 영어로 번역합니다" \
    --method beam --beam-size 4 --length-penalty 0.6

# Save output to file
$PYTHON scripts/translate.py --file input.txt --output translations.txt --method beam
```

**Translation Methods:**
- **Greedy Search** (default): Fast (~50-100 sentences/second on CPU)
- **Beam Search**: Better quality (~10-20 sentences/second on CPU), recommended beam size 4-8

---

### 7. Visualize Attention

Visualize attention weights to understand translation alignment.

```bash
# Visualize validation examples
$PYTHON scripts/visualize_attention.py

# Visualize specific sentence
$PYTHON scripts/visualize_attention.py --input "안녕하세요. 저는 학생입니다."

# Visualize from file
$PYTHON scripts/visualize_attention.py --file examples.txt

# Visualize specific decoder layer
$PYTHON scripts/visualize_attention.py --input "텍스트" --layer 3

# Show plots interactively
$PYTHON scripts/visualize_attention.py --input "텍스트" --show
```

**Generated Visualizations:**
```
outputs/attention_plots/
├── cross_attention_layer5.png              # Decoder→Encoder attention
├── cross_attention_multihead_layer5.png    # All 8 attention heads
└── self_attention_layer5.png               # Decoder self-attention
```

See `docs/ATTENTION_VISUALIZATION.md` for detailed guide.

---

### 8. Analyze Training Logs

Generate comprehensive training analysis report with visualizations.

```bash
# Full analysis (plot + summary)
$PYTHON tests/tool_analyze_training_log.py logs/training_log_20251213_015206.csv

# Custom output path
$PYTHON tests/tool_analyze_training_log.py logs/training_log_*.csv \
    --output reports/my_analysis.png

# Plot only (no text summary)
$PYTHON tests/tool_analyze_training_log.py logs/training_log_*.csv \
    --format plot

# Summary only (no plot)
$PYTHON tests/tool_analyze_training_log.py logs/training_log_*.csv \
    --format summary

# Save summary to file
$PYTHON tests/tool_analyze_training_log.py logs/training_log_*.csv \
    --summary-file reports/training_summary.txt
```

**Generated Report:**

**1. Visualization (4-panel plot):**
```
outputs/training_analysis_<timestamp>.png
├── Panel 1: Loss curves (train vs val)
├── Panel 2: Perplexity curves
├── Panel 3: BLEU score progression
└── Panel 4: Train/Val gap (overfitting indicator)
```

**2. Text Summary:**
```
==================================================================================
TRAINING LOG ANALYSIS
==================================================================================

📊 Training Summary:
   Total Epochs:     30
   Training samples: 897566
   Val samples:      1896
   Config: dropout=0.3, label_smoothing=0.05, batch_size=128

📈 Best Model (Epoch 15):
   Train Loss: 2.234
   Val Loss:   2.456
   Gap:        0.222
   Train PPL:  9.34
   Val PPL:    11.66
   BLEU:       48.23

📉 Final Model (Epoch 30):
   Train Loss: 2.010
   Val Loss:   2.678
   Gap:        0.668
   BLEU:       45.12

🔍 Degradation from Best to Final:
   Val Loss Increase: +0.222 (+9.0%)
   Gap Increase:      +0.446
   BLEU Drop:         -3.11 points

🎯 Verdict:
   ⚠️  Moderate overfitting detected
   - Some degradation after best epoch
   - Consider stronger regularization for future training
```

**Key Metrics Interpretation:**

| Metric | Healthy | Warning | Critical |
|--------|---------|---------|----------|
| **Train/Val Gap** | < 0.5 | 0.5 - 1.0 | > 1.0 |
| **Val Loss Trend** | Decreasing/Stable | Slight increase | Strong increase |
| **BLEU Trend** | Increasing | Stable | Decreasing |
| **Degradation %** | < 5% | 5-10% | > 10% |

**Use Cases:**
- **Monitor training progress**: Check if model is learning properly
- **Diagnose overfitting**: Identify when model starts memorizing
- **Select best checkpoint**: Find optimal stopping point
- **Compare configurations**: Analyze different hyperparameter settings
- **Generate report figures**: Create plots for papers/presentations

---

## 📁 Project Structure

```
project/
├── config/                      # Configuration files
│   ├── base_config.py          # Shared settings
│   └── transformer_config.py   # Transformer-specific settings
│
├── data/                        # Data files
│   ├── raw/                    # Downloaded datasets
│   ├── processed/              # Cleaned & split data
│   └── vocab/                  # Tokenizer models
│
├── src/                         # Source code
│   ├── data/                   # Data loading & tokenization
│   ├── models/transformer/     # Transformer components
│   ├── training/               # Training infrastructure
│   ├── inference/              # Inference & decoding
│   └── utils/                  # Utilities
│
├── scripts/                     # Production scripts (7 essential)
│   ├── download_data.py        # 1. Download datasets
│   ├── split_data.py           # 2. Preprocess & split
│   ├── train_tokenizer.py      # 3. Train tokenizers
│   ├── train.py                # 4. Train model
│   ├── evaluate.py             # 5. Evaluate on test set
│   ├── translate.py            # 6. Translate sentences
│   └── visualize_attention.py  # 7. Attention visualization
│
├── tests/                       # Unit tests & tools
│   ├── test_*.py               # Unit tests
│   └── tool_*.py               # Analysis/debugging tools
│
├── docs/                        # Documentation
├── checkpoints/                 # Model checkpoints
├── logs/                        # Training logs
├── outputs/                     # Generated outputs
│
└── README.md                    # This file
```

---

## ⚙️ Configuration

### Model Configuration (`config/transformer_config.py`)

```python
d_model = 512               # Embedding dimension
d_ff = 2048                # FFN hidden dimension
num_heads = 8              # Attention heads
num_encoder_layers = 6     # Encoder depth
num_decoder_layers = 6     # Decoder depth

learning_rate = 2.0        # LR factor for Noam schedule
warmup_steps = 8000        # Warmup steps
```

### Training Configuration (`config/base_config.py`)

```python
batch_size = 128
gradient_accumulation_steps = 2  # Effective batch = 256
num_epochs = 50
early_stopping_patience = 10

dropout = 0.3              # Regularization
label_smoothing = 0.05
```

---

## 🐛 Troubleshooting

### Out of Memory (OOM)

```python
# config/base_config.py
batch_size = 64              # Reduce from 128
gradient_accumulation_steps = 4
```

### Training Loss Not Decreasing

```python
# Adjust learning rate
learning_rate = 1.0          # Try lower factor
warmup_steps = 16000         # Slower warmup
```

### Overfitting (Val Loss Increases)

```python
# Stronger regularization
dropout = 0.4                # Increase from 0.3
early_stopping_patience = 5  # Reduce from 10
```

### Slow Training

- **CPU**: Expected ~3 hours/epoch, use `--small` for testing
- **GPU**: Enable `use_mixed_precision = True`, increase batch size

### Poor Translation Quality

1. Check BLEU score (should be >40)
2. Visualize attention patterns
3. Use beam search instead of greedy
4. Train longer or increase model capacity

---

## 📊 Expected Results

### Training Metrics

| Epoch | Train Loss | Val Loss | BLEU | Status |
|-------|------------|----------|------|--------|
| 5     | 2.85       | 2.68     | 28   | Improving |
| 10    | 2.52       | 2.51     | 38   | Good |
| 15    | 2.34       | 2.47     | 45   | **Best** |
| 20    | 2.21       | 2.48     | 43   | Overfitting |

**Target Performance:**
- BLEU: 40-50 (good), 50-60 (excellent)
- Train/Val gap: <0.5 (healthy)

---

## 📚 References

- Vaswani et al. (2017) "Attention Is All You Need"
- The Annotated Transformer - http://nlp.seas.harvard.edu/annotated-transformer/
- Korean Parallel Corpora - https://huggingface.co/datasets/moo/korean-parallel-corpora
- SentencePiece - https://github.com/google/sentencepiece

---

**Last Updated:** 2025-12-13
