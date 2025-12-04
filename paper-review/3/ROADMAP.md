 # Korean-English Transformer Implementation Roadmap

This roadmap guides you through implementing a Transformer model for Korean-English translation based on "Attention Is All You Need" (Vaswani et al., 2017).

## Project Overview

### Structure Summary

```
ko-en-transformer/
├── config/              # Configuration files for different models
│   ├── base_config.py          # Shared settings (batch size, learning rate, etc.)
│   ├── transformer_config.py   # Transformer-specific hyperparameters
│   └── seq2seq_config.py       # Future: Seq2Seq and Bahdanau configs
│
├── src/
│   ├── models/          # Model architectures
│   │   ├── transformer/        # Transformer components
│   │   ├── bahdanau/          # Future: Bahdanau attention
│   │   └── seq2seq/           # Future: Baseline Seq2Seq
│   │
│   ├── data/            # Data processing pipeline
│   │   ├── tokenizer.py       # Korean/English tokenizers
│   │   ├── vocabulary.py      # Vocabulary management
│   │   └── dataset.py         # PyTorch Dataset
│   │
│   ├── training/        # Training infrastructure
│   │   ├── trainer.py         # Training loop
│   │   ├── optimizer.py       # Noam optimizer with warmup
│   │   └── losses.py          # Label smoothing loss
│   │
│   ├── inference/       # Decoding strategies
│   │   ├── translator.py      # Translation interface
│   │   ├── greedy_search.py   # Greedy decoding
│   │   └── beam_search.py     # Beam search
│   │
│   └── utils/           # Helper functions
│       ├── masking.py         # Attention masks
│       ├── metrics.py         # BLEU score
│       └── checkpointing.py   # Save/load models
│
├── scripts/             # Executable scripts
│   ├── train.py               # Training script
│   └── translate.py           # Inference script
│
├── data/                # Dataset storage
├── checkpoints/         # Saved models
└── logs/                # Training logs
```

---

## Phase 1: Data Pipeline (Week 1)

### 1.1 Data Acquisition
**Goal:** Obtain Korean-English parallel corpus

**Tasks:**
- [ ] Download parallel corpus (recommended datasets):
  - AI Hub Korean-English parallel corpus
  - Tatoeba Korean-English sentences
  - OpenSubtitles corpus
- [ ] Split into train/validation/test sets (80/10/10)
- [ ] Store in `data/raw/`

**Files to create:**
- `scripts/download_data.sh`
- `scripts/split_data.py`

---

### 1.2 Tokenization
**Goal:** Implement Korean and English tokenizers

**Tasks:**
- [ ] Implement Korean tokenizer in `src/data/tokenizer.py`:
  - Use Mecab or KoNLPy for morphological analysis
  - Alternative: Train SentencePiece model
- [ ] Implement English tokenizer:
  - Use BPE (Byte Pair Encoding) or WordPiece
  - Alternative: Simple word-level tokenization for prototyping
- [ ] Write unit tests in `tests/test_tokenizer.py`

**Key decisions:**
- Korean: Morpheme-based (Mecab) vs Subword (SentencePiece)
- English: BPE vs WordPiece vs Word-level
- Shared vocabulary vs separate vocabularies

---

### 1.3 Vocabulary Building
**Goal:** Create token-to-index mappings

**Tasks:**
- [ ] Complete `src/data/vocabulary.py`:
  - Implement `build_from_corpus()` method
  - Implement `encode()` and `decode()` methods
  - Add special tokens: `<pad>`, `<unk>`, `<bos>`, `<eos>`
  - Handle minimum frequency threshold
- [ ] Build vocabularies from training data
- [ ] Save vocabularies to `data/vocab/`

**Script to create:**
- `scripts/build_vocab.py`

---

### 1.4 Dataset and DataLoader
**Goal:** Create PyTorch dataset for batch training

**Tasks:**
- [ ] Complete `src/data/dataset.py`:
  - Implement `TranslationDataset.__getitem__()`
  - Implement `collate_fn()` for batching with padding
  - Add dynamic padding vs fixed-length padding
- [ ] Test dataloader with sample batch
- [ ] Verify shapes: `[batch_size, seq_len]`

---

## Phase 2: Transformer Architecture (Week 2-3)

### 2.1 Core Attention Mechanism
**Goal:** Implement scaled dot-product and multi-head attention

**Priority:** HIGH - This is the heart of the Transformer

**Tasks:**
- [ ] Complete `src/models/transformer/attention.py`:
  - [ ] Implement `scaled_dot_product_attention()`:
    - Formula: `Attention(Q,K,V) = softmax(QK^T / sqrt(d_k))V`
    - Handle attention mask (set masked positions to -inf before softmax)
  - [ ] Implement `MultiHeadAttention.forward()`:
    - Split heads: `[batch, seq_len, d_model]` → `[batch, num_heads, seq_len, d_k]`
    - Apply attention per head
    - Concatenate heads and project: `W_o`
- [ ] Test attention with toy examples
- [ ] Visualize attention weights (optional but helpful)

**Key concepts:**
- Scaled dot-product prevents gradient vanishing
- Multiple heads learn different representations
- Masking prevents looking at future tokens

---

### 2.2 Position-wise Feed-Forward Network
**Goal:** Implement FFN layer

**Tasks:**
- [ ] Complete `src/models/transformer/feedforward.py`:
  - Two linear transformations with ReLU: `FFN(x) = max(0, xW1 + b1)W2 + b2`
  - Dimensions: `d_model → d_ff → d_model`
  - Paper uses `d_ff = 2048` when `d_model = 512`
- [ ] Add dropout after ReLU

---

### 2.3 Positional Encoding
**Goal:** Add position information to embeddings

**Tasks:**
- [ ] Complete `src/models/transformer/positional_encoding.py`:
  - Implement sinusoidal encoding:
    - `PE(pos, 2i) = sin(pos / 10000^(2i/d_model))`
    - `PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))`
  - Pre-compute and register as buffer (not trainable)
  - Add to embeddings: `embedded + positional_encoding`

**Why sinusoidal?**
- Allows model to learn relative positions
- Generalizes to longer sequences than seen in training

---

### 2.4 Encoder Layer
**Goal:** Build single encoder layer

**Tasks:**
- [ ] Complete `src/models/transformer/encoder.py`:
  - [ ] Implement `EncoderLayer`:
    - Multi-head self-attention sublayer
    - Position-wise FFN sublayer
    - Residual connections around each sublayer: `x + Sublayer(x)`
    - Layer normalization: `LayerNorm(x + Sublayer(x))`
    - Dropout on sublayer outputs
  - [ ] Implement `TransformerEncoder`:
    - Stack N encoder layers (N=6 in paper)
    - Process through all layers sequentially

**Architecture:**
```
Input → Self-Attention → Add & Norm → FFN → Add & Norm → Output
  ↓_________________↑         ↓_______________↑
     (residual)                  (residual)
```

---

### 2.5 Decoder Layer
**Goal:** Build single decoder layer

**Tasks:**
- [ ] Complete `src/models/transformer/decoder.py`:
  - [ ] Implement `DecoderLayer`:
    - **Masked** multi-head self-attention (prevents attending to future)
    - Cross-attention with encoder output (Q from decoder, K,V from encoder)
    - Position-wise FFN
    - Three residual connections with layer norm
  - [ ] Implement `TransformerDecoder`:
    - Stack N decoder layers (N=6 in paper)

**Architecture:**
```
Input → Masked Self-Attention → Add & Norm → Cross-Attention → Add & Norm → FFN → Add & Norm → Output
  ↓______________↑                 ↓________________↑         ↓_______________↑
    (residual)                        (residual)                 (residual)
```

---

### 2.6 Complete Transformer Model
**Goal:** Connect all components

**Tasks:**
- [ ] Complete `src/models/transformer/transformer.py`:
  - [ ] Add embedding layers:
    - Source embeddings: `nn.Embedding(src_vocab_size, d_model)`
    - Target embeddings: `nn.Embedding(tgt_vocab_size, d_model)`
    - Scale by `sqrt(d_model)` (see `embeddings.py`)
  - [ ] Add positional encoding to both encoder and decoder
  - [ ] Connect encoder and decoder
  - [ ] Add final linear projection: `d_model → tgt_vocab_size`
  - [ ] Implement masking logic (use `src/utils/masking.py`)

**Forward pass:**
```python
src_embedded = embedding(src) * sqrt(d_model) + positional_encoding
encoder_output = encoder(src_embedded, src_mask)

tgt_embedded = embedding(tgt) * sqrt(d_model) + positional_encoding
decoder_output = decoder(tgt_embedded, encoder_output, src_mask, tgt_mask)

logits = linear(decoder_output)  # [batch, tgt_len, vocab_size]
```

---

### 2.7 Testing & Debugging
**Goal:** Ensure model works correctly

**Tasks:**
- [ ] Write unit tests:
  - Test attention shapes
  - Test encoder output shapes
  - Test decoder output shapes
  - Test full model forward pass
- [ ] Test with small dummy data
- [ ] Verify gradient flow
- [ ] Count parameters (base model ~65M parameters)

---

## Phase 3: Training Infrastructure (Week 4)

### 3.1 Loss Function
**Goal:** Implement label smoothing cross-entropy

**Tasks:**
- [ ] Complete `src/training/losses.py`:
  - Instead of one-hot targets, smooth distribution:
    - Correct token: `confidence = 0.9`
    - Other tokens: `(1 - confidence) / (vocab_size - 2)`
  - Ignore padding tokens in loss calculation
- [ ] Test loss computation

**Why label smoothing?**
- Prevents overconfidence
- Better generalization
- Paper uses ε = 0.1

---

### 3.2 Optimizer with Warmup
**Goal:** Implement Noam learning rate schedule

**Tasks:**
- [ ] Complete `src/training/optimizer.py`:
  - Implement learning rate schedule:
    - `lr = d_model^(-0.5) * min(step^(-0.5), step * warmup^(-1.5))`
  - Use Adam with β1=0.9, β2=0.98, ε=1e-9
  - Warmup steps = 4000 (paper setting)

**Learning rate schedule:**
```
     ^
  lr |    /\
     |   /  \___
     |  /       \___
     | /            \___
     |/________________\___
     0  4k  8k  12k  16k  steps
```

---

### 3.3 Training Loop
**Goal:** Implement training and validation

**Tasks:**
- [ ] Complete `src/training/trainer.py`:
  - [ ] Implement `train_epoch()`:
    - Forward pass
    - Compute loss (ignore padding)
    - Backward pass
    - Gradient clipping (max_norm=1.0)
    - Optimizer step
    - Update learning rate
  - [ ] Implement `validate()`:
    - Compute validation loss
    - Track metrics (loss, perplexity)
  - [ ] Add logging (tensorboard or wandb)
  - [ ] Implement checkpointing
  - [ ] Add early stopping (optional)

---

### 3.4 Training Script
**Goal:** Create executable training script

**Tasks:**
- [ ] Complete `scripts/train.py`:
  - Load configuration
  - Load and preprocess data
  - Initialize model, optimizer, criterion
  - Create trainer and start training
  - Save best model
- [ ] Add command-line arguments:
  - `--config`: Config file path
  - `--resume`: Resume from checkpoint
  - `--eval-only`: Evaluation mode

---

## Phase 4: Inference (Week 5)

### 4.1 Greedy Decoding
**Goal:** Implement simple decoding strategy

**Tasks:**
- [ ] Complete `src/inference/greedy_search.py`:
  - Start with `<bos>` token
  - At each step, select token with highest probability
  - Append to sequence and continue
  - Stop when `<eos>` or max_length reached
- [ ] Test with trained model

**Pseudocode:**
```python
output = [BOS]
for _ in range(max_length):
    logits = model(src, output)
    next_token = argmax(logits[-1])
    output.append(next_token)
    if next_token == EOS:
        break
```

---

### 4.2 Beam Search
**Goal:** Improve translation quality with beam search

**Tasks:**
- [ ] Complete `src/inference/beam_search.py`:
  - Maintain top-k candidates (beam_size=4)
  - Score candidates by log probability
  - Length normalization: `score / len^α` (α=0.6)
  - Expand each beam and keep top-k
- [ ] Compare with greedy decoding

**Beam search improves quality but is slower**

---

### 4.3 Translation Interface
**Goal:** Create user-friendly translation API

**Tasks:**
- [ ] Complete `src/inference/translator.py`:
  - Load trained model and vocabularies
  - Tokenize input
  - Run inference (greedy or beam)
  - Detokenize output
- [ ] Complete `scripts/translate.py`:
  - Interactive mode: translate user input
  - Batch mode: translate file
  - Add BLEU score computation

---

## Phase 5: Evaluation & Optimization (Week 6)

### 5.1 Evaluation
**Goal:** Measure translation quality

**Tasks:**
- [ ] Implement BLEU score computation (already in `utils/metrics.py`)
- [ ] Evaluate on test set
- [ ] Generate translations for analysis
- [ ] Manual inspection of translations

**Metrics:**
- BLEU score (primary)
- Perplexity
- Translation examples

---

### 5.2 Hyperparameter Tuning
**Goal:** Optimize performance

**Tasks:**
- [ ] Experiment with different configurations:
  - Model size: `d_model` (256, 512, 1024)
  - Depth: `num_layers` (4, 6, 8)
  - Attention heads: `num_heads` (4, 8, 16)
  - Dropout rates
  - Learning rate warmup
- [ ] Track experiments (use tensorboard/wandb)

---

### 5.3 Analysis & Visualization
**Goal:** Understand model behavior

**Tasks:**
- [ ] Visualize attention weights
- [ ] Analyze translation errors
- [ ] Compare different decoding strategies
- [ ] Create attention heatmaps for interesting examples

**Suggested notebook:**
- `notebooks/attention_visualization.ipynb`

---

## Phase 6: Model Comparison (Week 7+)

### 6.1 Implement Seq2Seq Baseline
**Goal:** Provide baseline for comparison

**Tasks:**
- [ ] Implement vanilla Seq2Seq in `src/models/seq2seq/`:
  - LSTM/GRU encoder
  - LSTM/GRU decoder
  - No attention mechanism
- [ ] Train and evaluate

---

### 6.2 Implement Bahdanau Attention
**Goal:** Compare with attention-based Seq2Seq

**Tasks:**
- [ ] Implement Bahdanau attention in `src/models/bahdanau/`:
  - Additive attention mechanism
  - Attention over encoder hidden states
- [ ] Train and evaluate

---

### 6.3 Comparative Analysis
**Goal:** Understand advantages of Transformer

**Tasks:**
- [ ] Compare all models:
  - BLEU scores
  - Training time
  - Inference speed
  - Model size
  - Long-range dependency handling
- [ ] Write analysis report
- [ ] Create comparison charts

---

## Quick Start Guide

### Initial Setup

```bash
cd /home/arnold/arnold/github/ai-study/paper-review/3

# Install dependencies
pip install -r requirements.txt

# Download Korean language support
# For Mecab (recommended):
# bash <(curl -s https://raw.githubusercontent.com/konlpy/konlpy/master/scripts/mecab.sh)
```

### Development Order

**Week 1: Data Pipeline**
1. Download data → `data/raw/`
2. Implement tokenizers → `src/data/tokenizer.py`
3. Build vocabularies → `scripts/build_vocab.py`
4. Test dataset → `src/data/dataset.py`

**Week 2-3: Model Architecture**
1. ✅ Multi-head attention → `src/models/transformer/attention.py`
2. ✅ FFN → `src/models/transformer/feedforward.py`
3. ✅ Positional encoding → `src/models/transformer/positional_encoding.py`
4. ✅ Encoder → `src/models/transformer/encoder.py`
5. ✅ Decoder → `src/models/transformer/decoder.py`
6. ✅ Full model → `src/models/transformer/transformer.py`

**Week 4: Training**
1. ✅ Loss function → `src/training/losses.py`
2. ✅ Optimizer → `src/training/optimizer.py`
3. ✅ Training loop → `src/training/trainer.py`
4. Run training → `scripts/train.py`

**Week 5: Inference**
1. ✅ Greedy search → `src/inference/greedy_search.py`
2. ✅ Beam search → `src/inference/beam_search.py`
3. Translation interface → `scripts/translate.py`

**Week 6+: Evaluation & Comparison**
1. Evaluate performance
2. Implement other models
3. Comparative analysis

---

## Key Implementation Tips

### Attention Mechanism
- Use masking correctly: padding mask for encoder, combined mask for decoder
- Apply dropout after attention weights
- Initialize projection matrices carefully

### Training Tips
- Start with small model for debugging (d_model=256, num_layers=2)
- Use gradient accumulation if memory limited
- Monitor gradient norms
- Watch for attention weight collapse (all attention on single position)

### Common Pitfalls
- ❌ Forgetting to mask future positions in decoder
- ❌ Not scaling embeddings by sqrt(d_model)
- ❌ Wrong mask dimensions/broadcasting
- ❌ Not handling padding in loss computation
- ❌ Forgetting to scale attention scores by sqrt(d_k)

### Debugging Checklist
- [ ] Check tensor shapes at each layer
- [ ] Verify mask is applied correctly
- [ ] Ensure loss decreases on small dataset
- [ ] Test with batch_size=1 first
- [ ] Visualize attention weights
- [ ] Check gradient flow (no NaN/Inf)

---

## Recommended Datasets

### Small (for prototyping)
- **Tatoeba**: ~50k sentence pairs
- Good for initial testing

### Medium
- **AI Hub Korean-English Parallel Corpus**: ~1M pairs
- Balanced quality and size

### Large (for production)
- **OpenSubtitles**: Several million pairs
- Noisy but large scale

---

## Expected Results

### Base Model (6 layers, d_model=512)
- Training time: 2-3 days on single GPU
- BLEU score: 25-30 (depending on dataset)
- Parameters: ~65M

### Small Model (2 layers, d_model=256)
- Training time: Few hours
- BLEU score: 15-20
- Parameters: ~10M
- Good for prototyping

---

## References

1. **Attention Is All You Need** (Vaswani et al., 2017)
   - Original Transformer paper
   - https://arxiv.org/abs/1706.03762

2. **The Annotated Transformer**
   - Line-by-line implementation guide
   - http://nlp.seas.harvard.edu/annotated-transformer/

3. **Neural Machine Translation by Jointly Learning to Align and Translate** (Bahdanau et al., 2014)
   - For comparison model

---

## Next Steps After Transformer

1. **Experiment with variants:**
   - Transformer-XL (longer sequences)
   - Universal Transformer (adaptive depth)
   - Evolved Transformer (architecture search)

2. **Pre-training approaches:**
   - Implement BERT-style pre-training
   - Fine-tune on translation task

3. **Advanced techniques:**
   - Back-translation for data augmentation
   - Multi-task learning
   - Knowledge distillation

---

## Project Status

- [x] Template structure created
- [x] Configuration files
- [x] Model architecture templates (with TODOs)
- [x] Training infrastructure templates
- [x] Inference templates
- [x] Utility functions (masking, metrics, checkpointing)
- [ ] Data pipeline implementation
- [ ] Model implementation
- [ ] Training execution
- [ ] Evaluation

**Current Phase:** Ready to start Phase 1 (Data Pipeline)

**Estimated Timeline:** 6-8 weeks for complete Transformer implementation and evaluation
