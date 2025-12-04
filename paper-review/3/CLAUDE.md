# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Korean-English Neural Machine Translation research project implementing the Transformer architecture from "Attention Is All You Need" (Vaswani et al., 2017). The project is structured as a comparative study that will eventually include Seq2Seq baseline and Bahdanau attention models.

**Current Status (2025-12-05):**
- ✅ **Phase 1 Complete**: Data pipeline (download, preprocessing, tokenization, dataset)
- ✅ **Phase 2 Complete**: Model architecture reviewed and tested
- ✅ **Phase 3 Complete**: Training infrastructure (losses, optimizer, trainer, training script)
- ✅ **Phase 4 Complete**: Inference with KV caching, greedy & beam search, translation interface
- ✅ **Data Ready**: 897k training pairs, 1.9k validation, 4k test
- ✅ **Tokenizers**: SentencePiece models (16k vocab, Korean + English)
- ✅ **Training**: Full training pipeline with label smoothing, gradient clipping, checkpointing
- ✅ **Inference**: Cached greedy & beam search with length normalization
- 📝 **Next**: Train model, evaluation (Phase 5)

## Important: Python Path

**ALWAYS use `/home/arnold/venv/bin/python` for running Python scripts in this project.**

```bash
# Correct
/home/arnold/venv/bin/python scripts/train.py

# Wrong
python scripts/train.py
python3 scripts/train.py
```

## Commands

### Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Key dependencies added:
# - datasets>=2.14.0,<3  (for downloading datasets)
# - sentencepiece>=0.1.99 (for tokenization)
```

### Data Pipeline (Phase 1)

```bash
# 1. Download datasets (supports: moo, tatoeba, aihub, all)
/home/arnold/venv/bin/python scripts/download_data.py all
/home/arnold/venv/bin/python scripts/download_data.py moo     # Single dataset

# 2. Merge and clean datasets
/home/arnold/venv/bin/python scripts/split_data.py

# Optional: Adjust filtering parameters
/home/arnold/venv/bin/python scripts/split_data.py --min-len 1 --max-len 200

# 3. Train tokenizers
/home/arnold/venv/bin/python scripts/train_tokenizer.py
/home/arnold/venv/bin/python scripts/test_tokenizer.py  # Test tokenization

# 4. Test dataset (optional but recommended)
/home/arnold/venv/bin/python scripts/test_dataset.py

# Output:
# - data/raw/{moo,tatoeba,aihub}/  (downloaded datasets)
# - data/processed/train.{ko,en}   (unified, cleaned data)
# - data/vocab/ko_spm.{model,vocab} (SentencePiece models)
```

### Training (Phase 3)
```bash
# Run training
/home/arnold/venv/bin/python scripts/train.py

# Resume from checkpoint
/home/arnold/venv/bin/python scripts/train.py --resume checkpoints/latest.pt

# Train on small subset for testing
/home/arnold/venv/bin/python scripts/train.py --small

# Features:
# - Label smoothing loss (ε=0.1)
# - Noam optimizer with warmup (4000 steps)
# - Gradient clipping (max_norm=1.0)
# - Automatic checkpointing
# - TensorBoard logging
```

### Inference (Phase 4)
```bash
# Translate with greedy search (fast)
/home/arnold/venv/bin/python scripts/translate.py --input "안녕하세요"

# Translate with beam search (better quality)
/home/arnold/venv/bin/python scripts/translate.py --input "안녕하세요" --method beam --beam-size 4

# Translate from file
/home/arnold/venv/bin/python scripts/translate.py --file input.txt --method beam

# Advanced options
/home/arnold/venv/bin/python scripts/translate.py \
    --input "한국어 문장" \
    --method beam \
    --beam-size 5 \
    --length-penalty 0.6 \
    --max-length 150 \
    --device cuda
```

### Testing
```bash
# Run unit tests (when implemented)
pytest tests/

# Run specific test file
pytest tests/test_tokenizer.py

# Test with coverage
pytest --cov=src tests/
```

### Code Quality
```bash
# Format code
black src/ scripts/ tests/

# Check formatting
black --check src/
```

## Architecture Overview

### Data Pipeline (src/data/) - **COMPLETE ✅**

**Status:** Phase 1 (1.1, 1.2, 1.3) All Complete

The data pipeline uses a **unified dataset approach**:

1. **Download** (`scripts/download_data.py`) - ✅ Complete
   - Supports multiple datasets: Moo, Tatoeba, AIHub
   - Each dataset saved to `data/raw/{dataset}/`
   - Different datasets have different splits available:
     - Moo: train/validation/test
     - Tatoeba: validation/test only
     - AIHub: train only

2. **Merge & Clean** (`scripts/split_data.py`) - ✅ Complete
   - Merges multiple datasets by split type
   - Applies filtering: min/max length, length ratio
   - Creates unified splits: `data/processed/train.{ko,en}`, etc.
   - Generates statistics showing contribution from each source

3. **Tokenization** (`scripts/train_tokenizer.py`) - ✅ Complete
   - **Uses SentencePiece** (subword tokenization)
   - Trains separate models for Korean and English
   - Output: `data/vocab/ko_spm.model`, `en_spm.model`
   - Test: `scripts/test_tokenizer.py`

4. **Dataset** (`src/data/dataset.py`) - ✅ Complete
   - PyTorch Dataset with on-the-fly tokenization
   - Batching with dynamic padding
   - Attention mask creation
   - Test: `scripts/test_dataset.py`

**Key Modules:**

- **tokenizer.py**: `SentencePieceTokenizer` class
  - Wraps SentencePiece functionality
  - Methods: `tokenize()`, `detokenize()`, `encode_ids()`, `decode_ids()`
  - **No separate Vocabulary class** - SentencePiece handles vocab internally

- **dataset.py**: `TranslationDataset` + `collate_fn`
  - PyTorch Dataset for parallel corpus
  - Loads text files, tokenizes on-the-fly (or pre-tokenized)
  - `collate_fn` handles padding and masking

**Critical implementation notes:**
- **SentencePiece is used for both Korean and English** (language-agnostic approach)
- Vocabulary size: 16,000 (configurable)
- Model type: Unigram (SentencePiece default)
- Special tokens: `<pad>=0, <unk>=1, <s>=2, </s>=3` (built into SentencePiece)

### Model Architecture (src/models/)
The Transformer implementation follows the paper's architecture with modular components:

- **transformer/attention.py**: Implements scaled dot-product attention and multi-head attention mechanism
- **transformer/encoder.py**: Encoder layers with self-attention and feed-forward networks, stacked N times
- **transformer/decoder.py**: Decoder layers with masked self-attention, cross-attention to encoder, and feed-forward networks
- **transformer/transformer.py**: Main model that connects embeddings, positional encoding, encoder, decoder, and output projection

**Status:** Components exist from ROADMAP template, need review and testing

**Key architectural details:**
- The encoder and decoder each use residual connections (`x + Sublayer(x)`) followed by layer normalization
- Embeddings are scaled by `sqrt(d_model)` before adding positional encoding
- Three types of attention are used: encoder self-attention, decoder masked self-attention, and decoder cross-attention
- Masking is critical: padding masks for variable-length sequences, and causal masks to prevent attending to future positions

### Training Infrastructure (src/training/)
- **trainer.py**: Training loop with gradient clipping, learning rate scheduling, and validation
- **optimizer.py**: Noam learning rate scheduler with warmup (formula: `d_model^(-0.5) * min(step^(-0.5), step * warmup^(-1.5))`)
- **losses.py**: Label smoothing cross-entropy that prevents overconfidence (ε=0.1 typically)

**Status:** ✅ Complete (Phase 3)

**Training specifics:**
- Learning rate increases linearly during warmup (4000 steps), then decreases proportionally to inverse square root of step number
- Label smoothing distributes probability mass from correct token to all others, improving generalization
- Gradient clipping (max_norm=1.0) prevents instability

### Inference (src/inference/)
- **greedy_search.py**: Fast decoding (greedy) + cached version with KV caching
- **beam_search.py**: Beam search with KV caching and length normalization (`score / len^α`, α=0.6)
- **translator.py**: High-level API for tokenization → inference → detokenization

**Status:** ✅ Complete (Phase 4)

**Inference specifics:**
- KV caching reduces complexity from O(n³) to O(n²) for autoregressive generation
- Self-attention caches K, V from previous target positions
- Cross-attention does NOT use caching (encoder projections are fast to recompute)
- Beam search with length normalization prevents bias towards shorter sequences
- Both greedy and beam search support incremental decoding

### Utilities (src/utils/)
- **masking.py**: Creates padding masks and causal masks for attention
- **metrics.py**: BLEU score computation for translation quality
- **checkpointing.py**: Model saving/loading with optimizer state

### Configuration (config/)
Configurations follow a hierarchy:
- **base_config.py**: Shared settings (batch size, epochs, device, dropout)
- **transformer_config.py**: Transformer hyperparameters (d_model=512, num_heads=8, num_layers=6)
- **seq2seq_config.py**: Future Seq2Seq configurations

## Development Workflow

### Current Implementation Status

**✅ Phase 1: Data Pipeline (COMPLETE)**
- 1.1: Data Acquisition & Preprocessing - `scripts/download_data.py`, `scripts/split_data.py`
- 1.2: Tokenization - `scripts/train_tokenizer.py`, `src/data/tokenizer.py`
- 1.3: Dataset - `src/data/dataset.py` with batching and padding
- Result: 897,566 training pairs, 1,896 validation, 4,061 test; 16k vocab each

**✅ Phase 2: Model Architecture (COMPLETE)**
- Transformer encoder/decoder with multi-head attention
- Positional encoding, embeddings, FFN
- Reviewed and tested all components
- Fixed masking bug (mask shapes now correct: [batch, 1, seq_len, seq_len2])

**✅ Phase 3: Training Infrastructure (COMPLETE)**
- `src/training/losses.py` - Label smoothing loss with KL divergence
- `src/training/optimizer.py` - Noam scheduler with warmup
- `src/training/trainer.py` - Training loop with validation
- `scripts/train.py` - Complete training script with CLI
- Features: gradient clipping, checkpointing, TensorBoard logging

**✅ Phase 4: Inference (COMPLETE)**
- 4.1: Basic greedy decoding (no cache) - `greedy_decode()`
- 4.2: KV cache infrastructure - Updated attention, decoder, transformer
- 4.3: Cached greedy decoding - `greedy_decode_cached()` (1.5x speedup)
- 4.4: Beam search with caching - `beam_search()` with length normalization
- 4.5: Translation interface - `Translator` class + `scripts/translate.py` CLI

**📝 Phase 5: Next Steps**
- Train model on full dataset
- Evaluate with BLEU scores
- Hyperparameter tuning
- Comparative analysis

### Implementation Order from ROADMAP.md

1. **Phase 1**: Data pipeline - ✅ Complete
2. **Phase 2**: Transformer architecture - ✅ Complete
3. **Phase 3**: Training infrastructure - ✅ Complete
4. **Phase 4**: Inference - ✅ Complete
5. **Phase 5**: Evaluation and hyperparameter tuning - 📝 Next
6. **Phase 6**: Comparative analysis with Seq2Seq and Bahdanau models

### Critical Implementation Pitfalls to Avoid
- **Masking errors**: Always mask future positions in decoder with causal mask; mask padding tokens in both encoder and decoder
- **Embedding scaling**: Must multiply embeddings by `sqrt(d_model)` before adding positional encoding
- **Attention scaling**: Scale attention scores by `sqrt(d_k)` to prevent vanishing gradients
- **Loss computation**: Exclude padding tokens when computing loss (use `ignore_index` or manual masking)
- **Dimension errors**: Pay attention to tensor shapes, especially head splitting in multi-head attention: `[batch, seq_len, d_model]` → `[batch, num_heads, seq_len, d_k]`

### Key Architectural Relationships
- The **Transformer** class composes encoder and decoder, which are stacks of **EncoderLayer** and **DecoderLayer**
- Each layer uses **MultiHeadAttention** for self/cross-attention and **PositionwiseFeedForward** for non-linear transformation
- **Positional encoding** is added to embeddings before entering encoder/decoder (not trainable, uses sinusoidal functions)
- Masks flow through the entire architecture: padding masks prevent attention to padding, causal masks prevent attention to future tokens

### Configuration System
The config classes use inheritance (TransformerConfig extends BaseConfig) and are imported directly in scripts. To modify hyperparameters, edit the config files or subclass them. Key parameters:
- **d_model**: 512 (base), 256 (small for debugging)
- **num_heads**: Must divide d_model evenly (typically 8)
- **num_layers**: 6 for base model, 2-4 for prototyping
- **warmup_steps**: 4000 in paper, can reduce for smaller datasets

## Project Organization - **UPDATED**

### Directory Structure

```
project/
├── data/                   # Data files
│   ├── raw/               # Downloaded datasets (by source)
│   │   ├── moo/           # Moo/korean-parallel-corpora (96k/1k/2k)
│   │   ├── tatoeba/       # Helsinki-NLP/tatoeba_mt (1k/2.4k)
│   │   └── aihub/         # AI Hub (1.6M)
│   ├── processed/         # Unified, cleaned datasets
│   │   ├── train.ko/en    # 897k pairs
│   │   ├── validation.ko/en # 1.9k pairs
│   │   └── test.ko/en     # 4k pairs
│   └── vocab/             # Tokenizer models
│       ├── ko_spm.model/vocab
│       └── en_spm.model/vocab
│
├── src/                    # Source code
│   ├── data/              # Data loading & preprocessing
│   ├── models/            # Model architectures
│   ├── training/          # Training infrastructure
│   ├── inference/         # Inference & decoding
│   └── utils/             # Utilities
│
├── scripts/               # Production scripts
│   ├── download_data.py   # Download datasets
│   ├── split_data.py      # Merge & clean data
│   ├── train_tokenizer.py # Train tokenizers
│   ├── train.py           # Training script
│   └── translate.py       # Translation CLI
│
├── tests/                 # Test files (reorganized)
│   ├── test_tokenizer.py
│   ├── test_dataset.py
│   ├── test_greedy_search.py
│   ├── test_cached_greedy.py
│   └── ... (all test files)
│
├── docs/                  # Documentation (reorganized)
│   ├── PHASE1_COMPLETE.md
│   ├── PHASE3_SUMMARY.md
│   ├── PHASE4_COMPLETE.md
│   ├── MASK_BUG_FIX.md
│   ├── INFERENCE_PLAN.md
│   ├── CACHING_DEEP_DIVE.md
│   └── BEAM_SEARCH_EXPLAINED.md
│
├── examples/              # Demo scripts (reorganized)
│   ├── demo_beam_search.py
│   └── debug_training.py
│
├── checkpoints/           # Model checkpoints
├── logs/                  # TensorBoard logs
├── config/                # Configuration files
├── notebooks/             # Jupyter notebooks
│
├── CLAUDE.md              # This file
├── README.md              # Project README
├── ROADMAP.md             # Implementation roadmap
└── ARCHITECTURE.md        # Detailed architecture
```

## Tokenization Strategy - **NEW**

### Why SentencePiece?

**Chosen approach:** SentencePiece (Unigram) for both Korean and English

**Reasons:**
1. **Language-agnostic**: Same method for both languages
2. **No preprocessing needed**: Handles raw text directly
3. **Handles OOV**: Decomposes unknown words into subwords
4. **Industry standard**: Used in production NMT systems
5. **Built-in vocabulary**: No need for separate Vocabulary class

### Tokenization Example

```python
# Korean
"안녕하세요" → ["▁안녕", "하", "세요"] → [234, 567, 890]

# English
"Hello world" → ["▁Hello", "▁world"] → [123, 456]
```

### Configuration

- Vocabulary size: 16,000 per language
- Model type: Unigram (data-driven subword)
- Character coverage: 0.9995 (high for Korean)
- Special tokens: `<pad>=0, <unk>=1, <s>=2, </s>=3`

### Alternative Approaches (Not Used)

- **Mecab + BPE**: More complex, language-specific
- **Character-level**: Too granular, loses meaning
- **Word-level**: Huge vocabulary, many OOV
- **Shared vocabulary**: Could merge Korean + English (not implemented)

## Testing Strategy

When implementing components, follow this testing approach:
1. **Unit tests**: Test individual components (attention, encoder layers) with small dummy inputs
2. **Shape tests**: Verify tensor dimensions at each layer
3. **Small data tests**: Ensure loss decreases on tiny dataset (overfitting test)
4. **Gradient tests**: Check for NaN/Inf in gradients
5. **Attention visualization**: Plot attention weights to verify meaningful patterns

## Model Comparison Framework

This project is designed for comparing multiple architectures. When implementing Seq2Seq/Bahdanau models:
- Keep the same data pipeline and training infrastructure
- Use the same evaluation metrics (BLEU, perplexity)
- Track training time, inference speed, and model size
- The goal is to demonstrate Transformer advantages: parallelization, long-range dependencies, training stability

## Development Tips

- **Start small**: Use d_model=256, num_layers=2, small dataset for initial debugging
- **Incremental testing**: Test each component before moving to the next
- **Attention visualization**: Always visualize attention weights to debug masking issues
- **Monitor gradients**: Use gradient norm monitoring to detect instability
- **Batch size**: Adjust based on GPU memory; use gradient accumulation if needed

## Project Files

**Documentation:**
- `CLAUDE.md` - This file (for Claude Code instances)
- `ARCHITECTURE.md` - Detailed system architecture and implementation plan
- `ROADMAP.md` - Original phase-by-phase implementation guide
- `README.md` - Project overview

**Key Scripts:**
- `scripts/download_data.py` - ✅ Download datasets
- `scripts/split_data.py` - ✅ Merge and clean data
- `scripts/train_tokenizer.py` - ✅ Train SentencePiece
- `scripts/train.py` - ✅ Complete training pipeline
- `scripts/translate.py` - ✅ Translation with greedy/beam search

## Phase 3 & 4 Key Learnings

### Critical Bug Fixes

**Masking Bug (Phase 3):**
- **Issue**: Attention masks had shape `[batch, 1, 1, seq_len]` instead of `[batch, 1, seq_len, seq_len2]`
- **Impact**: Broadcasting made it work functionally, but semantically incorrect
- **Fix**: Updated masking functions to explicitly create correct shapes:
  - Encoder self-attention: `[batch, 1, src_len, src_len]`
  - Decoder self-attention: `[batch, 1, tgt_len, tgt_len]` (causal + padding)
  - Decoder cross-attention: `[batch, 1, tgt_len, src_len]`
- **Documentation**: See `docs/MASK_BUG_FIX.md`

### KV Caching Strategy (Phase 4)

**Decision: Cache only self-attention K, V (not cross-attention)**
- **Rationale**: Cross-attention K, V projections are fast to recompute (just linear layers)
- **Benefit**: Simplifies implementation, avoids cache management complexity
- **Result**: Still achieves O(n³) → O(n²) speedup

**Implementation Details:**
- Self-attention: Concatenate cached K, V with new K, V
- Cross-attention: Recompute encoder K, V projections each step
- Memory: `num_layers * seq_len * d_model` per sample (self-attention only)

### Testing Insights

1. **Cached vs Uncached Must Match**: Verified that cached greedy produces identical outputs to uncached
2. **Determinism Requires Fixed Seeds**: Both model init and input generation need fixed seeds
3. **Speedup Varies by Model Size**: Small models show ~1.5x, larger models show more

### Length Normalization

**Problem**: Beam search without normalization favors shorter sequences (log probs are negative)

**Solution**: Normalize by `score / (length^alpha)` where alpha=0.6 is standard

**Impact**: Prevents bias, produces more natural translations

## Next Steps (Phase 5+)

**Phase 5: Training & Evaluation**
- Train Transformer on full 897k dataset
- Monitor loss, perplexity, BLEU scores
- Generate sample translations
- Hyperparameter tuning

**Phase 6: Comparative Analysis**
- Implement Seq2Seq baseline
- Implement Bahdanau attention
- Compare: speed, quality, training stability

**Phase 7: Production Optimization**
- Model quantization
- ONNX export
- API deployment

## References and Resources

The implementation should closely follow:
- "Attention Is All You Need" (Vaswani et al., 2017) - https://arxiv.org/abs/1706.03762
- The Annotated Transformer - http://nlp.seas.harvard.edu/annotated-transformer/

For Korean NLP:
- SentencePiece - https://github.com/google/sentencepiece
- Datasets library - https://huggingface.co/docs/datasets/
- AI Hub for Korean-English parallel corpus datasets
