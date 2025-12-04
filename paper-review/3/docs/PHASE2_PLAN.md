# Phase 2: Transformer Architecture Implementation Plan

## Executive Summary

**Objective:** Implement all Transformer components from "Attention Is All You Need" paper for Korean-English translation.

**Status:**
- ✅ Phase 1 Complete (Data pipeline, tokenization, dataset)
- 📝 Phase 2 Ready (Components exist as templates with TODOs)

**Timeline Estimate:** 5-7 implementation sessions

**Current Configuration:**
- Vocab: Shared (16k tokens) - configurable to separate
- Data: 897k training pairs ready
- Max sequence length: 150 tokens

---

## Architecture Overview

```
                    TRANSFORMER MODEL
┌──────────────────────────────────────────────────────────┐
│                                                          │
│  ┌─────────────────┐         ┌─────────────────┐       │
│  │   ENCODER       │         │   DECODER       │       │
│  │   Stack (N=6)   │────────▶│   Stack (N=6)   │       │
│  └─────────────────┘         └─────────────────┘       │
│         ▲                            ▲                  │
│         │                            │                  │
│  ┌──────┴──────┐             ┌──────┴──────┐          │
│  │ Src Embed + │             │ Tgt Embed + │          │
│  │   Pos Enc   │             │   Pos Enc   │          │
│  └─────────────┘             └─────────────┘          │
│                                                          │
└──────────────────────────────────────────────────────────┘

Each Encoder Layer:
  Input → Multi-Head Attention → Add & Norm → FFN → Add & Norm

Each Decoder Layer:
  Input → Masked Self-Attention → Add & Norm
        → Cross-Attention → Add & Norm
        → FFN → Add & Norm
```

---

## Component Dependency Graph

```
Level 1 (No dependencies):
  ├── MultiHeadAttention (attention.py)
  ├── PositionwiseFeedForward (feedforward.py)
  └── PositionalEncoding (positional_encoding.py)

Level 2 (Depends on Level 1):
  ├── EncoderLayer (uses MultiHeadAttention + FFN)
  └── DecoderLayer (uses MultiHeadAttention + FFN)

Level 3 (Depends on Level 2):
  ├── TransformerEncoder (stacks EncoderLayer)
  └── TransformerDecoder (stacks DecoderLayer)

Level 4 (Depends on Level 3):
  └── Transformer (combines Encoder + Decoder + Embeddings)

Support:
  ├── Embeddings (embeddings.py) - parallel to Level 1
  └── Masking utilities (utils/masking.py) - already partially implemented
```

---

## Implementation Order & Rationale

### Session 1: Core Attention Mechanism ⭐ CRITICAL
**Priority:** HIGHEST - Heart of the Transformer

**Files to implement:**
- `src/models/transformer/attention.py`

**Components:**
1. **`scaled_dot_product_attention()` function**
   - Formula: `Attention(Q,K,V) = softmax(QK^T / sqrt(d_k))V`
   - Input shapes: `[batch, num_heads, seq_len, d_k]`
   - Must handle masking (set masked positions to -inf before softmax)
   - Must apply dropout after softmax

2. **`MultiHeadAttention` class**
   - Linear projections: W_q, W_k, W_v, W_o
   - Split into heads: `d_model → num_heads × d_k`
   - Process each head in parallel
   - Concatenate and project output

**Critical Details:**
- Scaling by `sqrt(d_k)` prevents gradient vanishing
- Masking MUST happen before softmax
- Head splitting: `[B, L, D]` → `[B, H, L, D/H]` where B=batch, L=length, D=d_model, H=num_heads
- Transpose operations for batch matrix multiply

**Testing Strategy:**
```python
# Test with toy data
batch_size = 2
seq_len = 10
d_model = 512
num_heads = 8

x = torch.randn(batch_size, seq_len, d_model)
attn = MultiHeadAttention(d_model, num_heads)
output = attn(x, x, x)  # Self-attention

# Verify shapes
assert output.shape == (batch_size, seq_len, d_model)

# Test with mask
mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
output_masked = attn(x, x, x, mask)
```

**Why First:**
- Most complex component
- Used everywhere (encoder self-attention, decoder self-attention, cross-attention)
- Early testing reveals integration issues

---

### Session 2: Feed-Forward Network & Positional Encoding
**Priority:** HIGH - Building blocks for layers

**Files to implement:**
1. `src/models/transformer/feedforward.py`
2. `src/models/transformer/positional_encoding.py`
3. `src/models/transformer/embeddings.py` (update if needed)

#### 2A: PositionwiseFeedForward

**Implementation:**
```
FFN(x) = max(0, xW1 + b1)W2 + b2

Architecture:
  d_model (512) → d_ff (2048) → d_model (512)
  with ReLU activation in between
```

**Critical Details:**
- Two linear layers
- ReLU activation after first layer
- Dropout after ReLU
- Applied position-wise (same network for each position independently)

#### 2B: PositionalEncoding

**Implementation:**
```
PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

**Critical Details:**
- Pre-compute entire matrix: `[max_len, d_model]`
- Register as buffer (not trainable parameter)
- Add to embeddings: `embedded + pe[:seq_len]`
- Must handle variable sequence lengths

**Testing Strategy:**
```python
# Test FFN
ffn = PositionwiseFeedForward(d_model=512, d_ff=2048)
x = torch.randn(2, 10, 512)
output = ffn(x)
assert output.shape == (2, 10, 512)

# Test Positional Encoding
pe = PositionalEncoding(d_model=512, max_len=5000)
embedded = torch.randn(2, 10, 512)
output = pe(embedded)
assert output.shape == (2, 10, 512)

# Verify sinusoidal pattern
# PE values should alternate sin/cos
```

**Why Second:**
- Simpler than attention
- Independent components
- Can be tested in isolation

---

### Session 3: Encoder Layer & Stack
**Priority:** HIGH - First half of Transformer

**Files to implement:**
- `src/models/transformer/encoder.py`

**Components:**
1. **`EncoderLayer`**
   - Multi-head self-attention
   - Position-wise FFN
   - Two residual connections with layer norm
   - Dropout on sublayer outputs

2. **`TransformerEncoder`**
   - Stack N EncoderLayers (N=6)
   - Sequential processing through layers

**Architecture Detail:**
```
Input: [batch, src_len, d_model]

EncoderLayer:
  1. x2 = LayerNorm(x + MultiHeadAttention(x, x, x, src_mask))
  2. output = LayerNorm(x2 + FFN(x2))

TransformerEncoder:
  for layer in layers:
    x = layer(x, src_mask)
  return x
```

**Critical Details:**
- **Layer Norm placement:** Post-norm (norm after residual)
  - Alternative: Pre-norm (norm before sublayer) - can be more stable
  - Paper uses post-norm
- **Residual connections:** MUST match dimensions
- **Masking:** Padding mask only (no causal mask in encoder)
- **Dropout:** Applied to attention output and FFN output

**Testing Strategy:**
```python
from config.transformer_config import TransformerConfig

config = TransformerConfig()
encoder = TransformerEncoder(config)

# Test single layer
batch_size = 2
src_len = 10
x = torch.randn(batch_size, src_len, config.d_model)
src_mask = torch.ones(batch_size, 1, 1, src_len).bool()

output = encoder(x, src_mask)
assert output.shape == (batch_size, src_len, config.d_model)

# Test gradient flow
loss = output.sum()
loss.backward()
# Check that all parameters have gradients
```

**Why Third:**
- Depends on attention and FFN
- Simpler than decoder (only self-attention)
- Can be tested independently

---

### Session 4: Decoder Layer & Stack
**Priority:** HIGH - Second half of Transformer

**Files to implement:**
- `src/models/transformer/decoder.py`

**Components:**
1. **`DecoderLayer`**
   - **Masked** multi-head self-attention
   - Cross-attention with encoder output
   - Position-wise FFN
   - Three residual connections with layer norm

2. **`TransformerDecoder`**
   - Stack N DecoderLayers (N=6)
   - Takes encoder output as additional input

**Architecture Detail:**
```
Input: [batch, tgt_len, d_model]
Encoder output: [batch, src_len, d_model]

DecoderLayer:
  1. x2 = LayerNorm(x + MaskedSelfAttention(x, x, x, tgt_mask))
  2. x3 = LayerNorm(x2 + CrossAttention(Q=x2, K=enc_out, V=enc_out, src_mask))
  3. output = LayerNorm(x3 + FFN(x3))

TransformerDecoder:
  for layer in layers:
    x = layer(x, encoder_output, src_mask, tgt_mask)
  return x
```

**Critical Details:**
- **Self-attention mask:** Causal mask (prevents attending to future tokens)
  - Combined with padding mask
  - Shape: `[batch, 1, tgt_len, tgt_len]`
- **Cross-attention mask:** Only padding mask for encoder output
  - Shape: `[batch, 1, 1, src_len]`
- **Cross-attention Q/K/V:**
  - Q: From decoder (current layer input)
  - K, V: From encoder output (same for all decoder layers)

**Testing Strategy:**
```python
config = TransformerConfig()
decoder = TransformerDecoder(config)

batch_size = 2
src_len = 10
tgt_len = 8

# Create encoder output (mock)
encoder_output = torch.randn(batch_size, src_len, config.d_model)

# Create decoder input
tgt_input = torch.randn(batch_size, tgt_len, config.d_model)

# Create masks
src_mask = torch.ones(batch_size, 1, 1, src_len).bool()
tgt_mask = create_target_mask(tgt_len)  # Causal + padding

output = decoder(tgt_input, encoder_output, src_mask, tgt_mask)
assert output.shape == (batch_size, tgt_len, config.d_model)

# Verify causal masking: decoder shouldn't see future tokens
# (Check attention weights if available)
```

**Why Fourth:**
- Depends on attention and FFN
- More complex than encoder (masked + cross attention)
- Must be tested with encoder output

---

### Session 5: Complete Transformer Model
**Priority:** CRITICAL - Integration

**Files to implement:**
- `src/models/transformer/transformer.py`
- `src/models/transformer/embeddings.py` (if needed)

**Components:**
1. **Embedding Layers**
   - Source embeddings: `nn.Embedding(src_vocab_size, d_model)`
   - Target embeddings: `nn.Embedding(tgt_vocab_size, d_model)`
   - Embedding scaling by `sqrt(d_model)` - CRITICAL!

2. **Complete Forward Pass**
   - Embed source and target
   - Add positional encoding
   - Pass through encoder
   - Pass through decoder
   - Final linear projection to vocab

**Architecture Detail:**
```python
class Transformer(nn.Module):
    def __init__(self, config, src_vocab_size, tgt_vocab_size):
        # Embeddings
        self.src_embed = nn.Embedding(src_vocab_size, config.d_model)
        self.tgt_embed = nn.Embedding(tgt_vocab_size, config.d_model)

        # Share embeddings if shared vocab
        if config.use_shared_vocab:
            self.tgt_embed = self.src_embed

        # Positional encoding
        self.pos_encoding = PositionalEncoding(config.d_model, max_len=5000)

        # Encoder and Decoder
        self.encoder = TransformerEncoder(config)
        self.decoder = TransformerDecoder(config)

        # Output projection
        self.fc_out = nn.Linear(config.d_model, tgt_vocab_size)

        # Optional: Share embeddings with output projection (weight tying)
        # self.fc_out.weight = self.tgt_embed.weight

    def forward(self, src, tgt, src_mask, tgt_mask):
        # Embed and add positional encoding
        src_embedded = self.pos_encoding(
            self.src_embed(src) * math.sqrt(self.config.d_model)
        )
        tgt_embedded = self.pos_encoding(
            self.tgt_embed(tgt) * math.sqrt(self.config.d_model)
        )

        # Encode
        encoder_output = self.encoder(src_embedded, src_mask)

        # Decode
        decoder_output = self.decoder(tgt_embedded, encoder_output, src_mask, tgt_mask)

        # Project to vocabulary
        logits = self.fc_out(decoder_output)

        return logits  # [batch, tgt_len, vocab_size]
```

**Critical Details:**
- **Embedding scaling:** MUST multiply by `sqrt(d_model)` before adding PE
- **Shared embeddings:** If `use_shared_vocab=True`, src and tgt embeddings are same
- **Weight tying:** Optional but recommended - tie output projection with embeddings
- **Mask creation:** Use utilities from `src/utils/masking.py`

**Mask Shapes:**
- `src_mask`: `[batch, 1, 1, src_len]` - padding mask
- `tgt_mask`: `[batch, 1, tgt_len, tgt_len]` - causal + padding mask

**Testing Strategy:**
```python
from config.transformer_config import TransformerConfig
from src.data.dataset import load_tokenizers

config = TransformerConfig()
src_tokenizer, tgt_tokenizer = load_tokenizers(
    "data/vocab",
    use_shared_vocab=config.use_shared_vocab
)

model = Transformer(
    config,
    src_vocab_size=src_tokenizer.vocab_size,
    tgt_vocab_size=tgt_tokenizer.vocab_size
)

# Test forward pass
batch_size = 4
src_len = 20
tgt_len = 15

src = torch.randint(0, src_tokenizer.vocab_size, (batch_size, src_len))
tgt = torch.randint(0, tgt_tokenizer.vocab_size, (batch_size, tgt_len))

# Create masks
from src.utils.masking import create_padding_mask, create_target_mask
src_mask = create_padding_mask(src, src_tokenizer.pad_id)
tgt_mask = create_target_mask(tgt, tgt_tokenizer.pad_id)

# Forward pass
logits = model(src, tgt, src_mask, tgt_mask)
assert logits.shape == (batch_size, tgt_len, tgt_tokenizer.vocab_size)

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params:,}")  # Should be ~65M for base model

# Test with real data
from src.data.dataset import create_dataloader
dataloader = create_dataloader(...)
batch = next(iter(dataloader))
logits = model(batch['src'], batch['tgt'], batch['src_mask'], batch['tgt_mask'])
```

**Why Fifth:**
- Integrates all components
- End-to-end testing
- Ready for training

---

## Testing Strategy

### Unit Tests (tests/test_transformer.py)

Create comprehensive test file:

```python
# Test each component
def test_attention():
    """Test MultiHeadAttention shapes and masking."""

def test_feedforward():
    """Test FFN shapes and activation."""

def test_positional_encoding():
    """Test PE values and shapes."""

def test_encoder_layer():
    """Test single encoder layer."""

def test_encoder():
    """Test encoder stack."""

def test_decoder_layer():
    """Test single decoder layer with cross-attention."""

def test_decoder():
    """Test decoder stack."""

def test_transformer_forward():
    """Test complete model forward pass."""

def test_transformer_with_real_data():
    """Test with actual batch from dataloader."""

def test_parameter_count():
    """Verify parameter count matches expected."""

def test_gradient_flow():
    """Ensure gradients flow through all parameters."""
```

### Integration Test (tests/test_transformer_integration.py)

```python
def test_overfitting_single_batch():
    """Transformer should be able to overfit a single batch."""
    # This is a sanity check - if model can't overfit, something is wrong

def test_masking_prevents_future():
    """Verify causal masking prevents seeing future tokens."""

def test_shared_vs_separate_vocab():
    """Test both vocabulary modes work."""
```

---

## Critical Implementation Notes

### 1. Dimension Tracking

**Key shapes to remember:**
```
Input:
  src: [B, S]         # B=batch, S=src_len
  tgt: [B, T]         # T=tgt_len

After embedding:
  src_emb: [B, S, D]  # D=d_model
  tgt_emb: [B, T, D]

Attention (multi-head):
  Before split: [B, L, D]
  After split:  [B, H, L, K]  # H=num_heads, K=d_k=D/H
  Scores:       [B, H, L, L]  # Attention weights
  Output:       [B, H, L, K]
  After concat: [B, L, D]

Encoder output:
  [B, S, D]

Decoder output:
  [B, T, D]

Final logits:
  [B, T, V]  # V=vocab_size
```

### 2. Masking Conventions

**Two mask types:**
1. **Padding mask:** Prevents attending to padding tokens
   - Encoder: `[B, 1, 1, S]`
   - Decoder cross-attention: `[B, 1, 1, S]`

2. **Causal mask:** Prevents attending to future tokens
   - Decoder self-attention: `[B, 1, T, T]` (lower triangular)

**Combining masks:**
```python
# Decoder self-attention needs both
tgt_padding_mask = create_padding_mask(tgt, pad_idx)  # [B, 1, 1, T]
causal_mask = create_look_ahead_mask(T)                # [1, T, T]
tgt_mask = tgt_padding_mask & causal_mask              # [B, 1, T, T]
```

### 3. Common Pitfalls

❌ **WRONG:**
- Forgetting to scale embeddings by `sqrt(d_model)`
- Applying softmax before masking
- Using incorrect mask shapes
- Not handling padding in loss
- Forgetting dropout

✅ **CORRECT:**
- Scale embeddings: `embed(x) * sqrt(d_model)`
- Mask then softmax: `softmax(scores.masked_fill(mask, -inf))`
- Verify all tensor shapes at each step
- Exclude padding from loss: `ignore_index=pad_id`
- Apply dropout consistently

### 4. Debugging Checklist

When things don't work:
- [ ] Check all tensor shapes (print intermediate outputs)
- [ ] Verify masks are applied correctly (visualize attention weights)
- [ ] Ensure embeddings are scaled
- [ ] Check positional encoding is added
- [ ] Verify gradient flow (all parameters have gradients)
- [ ] Test with tiny data (10 samples, overfit check)
- [ ] Check for NaN/Inf in outputs

---

## Configuration Update Needed

Update `config/transformer_config.py` to include:

```python
class TransformerConfig(BaseConfig):
    # Model architecture (from paper)
    d_model = 512
    d_ff = 2048
    num_heads = 8
    num_encoder_layers = 6
    num_decoder_layers = 6

    # For debugging: Small model
    # d_model = 256
    # d_ff = 1024
    # num_heads = 4
    # num_encoder_layers = 2
    # num_decoder_layers = 2

    # Positional encoding
    max_position = 5000

    # Regularization
    dropout = 0.1

    # Weight tying
    tie_embeddings = False  # Whether to tie src/tgt embeddings with output
```

---

## Success Criteria

Phase 2 is complete when:

- [ ] All components implemented (no `NotImplementedError`)
- [ ] All unit tests pass
- [ ] Model forward pass works with real data batch
- [ ] Model can overfit single batch (loss decreases)
- [ ] Parameter count is reasonable (~65M for base, ~20M for small)
- [ ] No NaN/Inf in forward pass
- [ ] Gradients flow to all parameters
- [ ] Code is documented with docstrings

---

## Next Steps (Phase 3)

After Phase 2:
1. Implement training loop (`src/training/trainer.py`)
2. Implement loss function with label smoothing (`src/training/losses.py`)
3. Implement Noam optimizer (`src/training/optimizer.py`)
4. Start training on small subset
5. Implement inference (beam search)

---

## Estimated Timeline

| Session | Component | Time | Status |
|---------|-----------|------|--------|
| 1 | Attention mechanism | 2-3 hours | 📝 Ready |
| 2 | FFN + Positional Encoding | 1-2 hours | 📝 Ready |
| 3 | Encoder | 1-2 hours | 📝 Ready |
| 4 | Decoder | 2-3 hours | 📝 Ready |
| 5 | Complete Transformer | 2-3 hours | 📝 Ready |
| 6 | Testing & Debugging | 2-3 hours | 📝 Ready |

**Total:** 10-16 hours over 6 sessions

---

## Resources

**Paper:** "Attention Is All You Need" - https://arxiv.org/abs/1706.03762

**Reference Implementation:** The Annotated Transformer - http://nlp.seas.harvard.edu/annotated-transformer/

**Key Equations:**
- Attention: `Attention(Q,K,V) = softmax(QK^T / sqrt(d_k))V`
- Multi-head: `MultiHead(Q,K,V) = Concat(head_1,...,head_h)W^O`
- Feed-forward: `FFN(x) = max(0, xW_1 + b_1)W_2 + b_2`
- Positional Encoding: `PE_(pos,2i) = sin(pos/10000^(2i/d_model))`

---

## Questions to Resolve

Before starting implementation:

1. **Weight tying:** Tie embeddings with output projection? (saves parameters)
2. **Layer norm placement:** Post-norm (paper) vs Pre-norm (more stable)?
3. **Shared embeddings:** Already configured in base_config - use it!
4. **Model size:** Start with small (d_model=256) or base (d_model=512)?
5. **Testing data:** Use validation set (2k samples) for quick testing?

**Recommendations:**
- Start with small model for faster iteration
- Use post-norm (paper implementation)
- Don't tie embeddings initially (simpler)
- Use validation set for testing
- Switch to base model once everything works

---

## Risk Assessment

### High Risk:
- **Attention masking bugs:** Hard to debug, subtle errors
  - Mitigation: Extensive testing, visualize attention weights

- **Dimension mismatches:** Easy to make mistakes with reshaping
  - Mitigation: Print shapes liberally, use assertions

### Medium Risk:
- **Gradient vanishing/exploding:** Deep network issues
  - Mitigation: Gradient clipping, proper initialization

- **Memory errors:** Large models + long sequences
  - Mitigation: Start small, monitor GPU memory

### Low Risk:
- **Configuration errors:** Easy to fix
  - Mitigation: Config file already set up well
