# Phase 4: Inference Implementation Plan with KV Caching

## Overview

Implement efficient inference for the Transformer model with Key-Value caching to avoid redundant computation during autoregressive decoding.

---

## Why KV Caching?

### Problem: Autoregressive Decoding is Inefficient

**Without caching:**
```
Step 1: Generate token 1 from encoder output
  - Compute attention for position 0

Step 2: Generate token 2 from [token0, token1]
  - Compute attention for positions 0, 1  ← Recomputes position 0!

Step 3: Generate token 3 from [token0, token1, token2]
  - Compute attention for positions 0, 1, 2  ← Recomputes 0, 1!

...

Total complexity: O(n²) where n = sequence length
```

**With KV caching:**
```
Step 1: Generate token 1
  - Compute K₀, V₀ for position 0
  - Cache K₀, V₀

Step 2: Generate token 2
  - Compute K₁, V₁ for position 1 only
  - Use cached K₀, V₀
  - Cache K₁, V₁

Step 3: Generate token 3
  - Compute K₂, V₂ for position 2 only
  - Use cached K₀, K₁, V₀, V₁
  - Cache K₂, V₂

Total complexity: O(n) ✓
```

### What Gets Cached?

**In self-attention:**
- **Keys (K)**: Projections of previous positions
- **Values (V)**: Projections of previous positions
- **NOT Queries (Q)**: Only computed for new position

**In cross-attention:**
- **Keys (K)**: From encoder output (computed once, reused for all steps)
- **Values (V)**: From encoder output (computed once, reused for all steps)
- **NOT Queries (Q)**: Computed for each new decoder position

---

## Implementation Plan

### Phase 4.1: Basic Greedy Decoding (No Cache)

**Goal:** Get basic inference working first, without optimization

**Files to implement:**
1. `src/inference/greedy_search.py`

**What to implement:**
```python
def greedy_decode(model, src, src_mask, max_len, bos_idx, eos_idx):
    """
    Simple greedy decoding without caching.

    Args:
        model: Trained Transformer model
        src: Source sequence [batch_size, src_len]
        src_mask: Source mask [batch_size, 1, src_len, src_len]
        max_len: Maximum generation length
        bos_idx: Beginning of sequence token
        eos_idx: End of sequence token

    Returns:
        output: Generated sequence [batch_size, output_len]
    """
    # 1. Encode source once
    encoder_output = model.encode(src, src_mask)

    # 2. Initialize decoder input with BOS token
    batch_size = src.size(0)
    tgt = torch.full((batch_size, 1), bos_idx, dtype=torch.long)

    # 3. Generate tokens autoregressively
    for i in range(max_len):
        # Create masks for current sequence
        tgt_mask = create_target_mask(tgt, pad_idx=0)
        cross_mask = create_cross_attention_mask(src, tgt, pad_idx=0)

        # Forward pass (recomputes everything each time - inefficient!)
        logits = model.decode(tgt, encoder_output, cross_mask, tgt_mask)

        # Get next token (greedy = argmax)
        next_token = logits[:, -1, :].argmax(dim=-1)  # [batch_size]

        # Append to sequence
        tgt = torch.cat([tgt, next_token.unsqueeze(1)], dim=1)

        # Stop if all sequences have EOS
        if (next_token == eos_idx).all():
            break

    return tgt
```

**Test:** Verify it works correctly (even if slow)

---

### Phase 4.2: KV Cache Infrastructure

**Goal:** Modify model to support incremental decoding with caching

#### 4.2.1: Update MultiHeadAttention

**File:** `src/models/transformer/attention.py`

**Add cache support:**
```python
class MultiHeadAttention(nn.Module):
    # ... existing code ...

    def forward(self, query, key, value, mask=None, cache=None, use_cache=False):
        """
        Forward pass with optional KV caching.

        Args:
            query: [batch, query_len, d_model]
            key: [batch, key_len, d_model]
            value: [batch, value_len, d_model]
            mask: [batch, 1, query_len, key_len]
            cache: Optional dict with 'key' and 'value' tensors from previous steps
            use_cache: Whether to return updated cache

        Returns:
            output: [batch, query_len, d_model]
            attn: Attention weights (optional)
            new_cache: Updated cache if use_cache=True
        """
        batch_size = query.size(0)
        query_len = query.size(1)

        # Project queries (always computed for new positions)
        Q = self.W_q(query)  # [batch, query_len, d_model]
        Q = Q.view(batch_size, query_len, self.num_heads, self.d_k).transpose(1, 2)

        # Project keys and values
        K = self.W_k(key)
        V = self.W_v(value)

        # If cache is provided, concatenate with cached K, V
        if cache is not None:
            # Concatenate with previous K, V
            K = torch.cat([cache['key'], K], dim=1)  # [batch, cached_len + new_len, d_model]
            V = torch.cat([cache['value'], V], dim=1)

        # Reshape K, V
        key_len = K.size(1)
        K = K.view(batch_size, key_len, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, key_len, self.num_heads, self.d_v).transpose(1, 2)

        # Compute attention
        context, attn = self.scaled_dot_product_attention(Q, K, V, mask)

        # Reshape and project output
        context = context.transpose(1, 2).contiguous()
        context = context.view(batch_size, query_len, self.d_model)
        output = self.W_o(context)
        output = self.dropout(output)

        # Return cache if requested
        if use_cache:
            # Store K, V in their original (pre-split) form
            new_cache = {
                'key': K.transpose(1, 2).contiguous().view(batch_size, key_len, self.d_model),
                'value': V.transpose(1, 2).contiguous().view(batch_size, key_len, self.d_model)
            }
            return output, attn, new_cache

        return output, attn
```

**Key points:**
- Cache stores K, V in their **projected** form (after W_k, W_v)
- Query is always computed fresh (only for new position)
- Cached K, V are concatenated with new K, V
- Cache is optional (training doesn't use it)

#### 4.2.2: Update DecoderLayer

**File:** `src/models/transformer/decoder.py`

**Add cache support:**
```python
class DecoderLayer(nn.Module):
    # ... existing code ...

    def forward(self, x, encoder_output, cross_mask=None, tgt_mask=None,
                self_attn_cache=None, cross_attn_cache=None, use_cache=False):
        """
        Forward with optional caching.

        Args:
            x: Target embeddings [batch, tgt_len, d_model]
            encoder_output: Encoder output [batch, src_len, d_model]
            cross_mask: Cross-attention mask
            tgt_mask: Target causal mask
            self_attn_cache: Cache for self-attention
            cross_attn_cache: Cache for cross-attention
            use_cache: Whether to return caches

        Returns:
            output: [batch, tgt_len, d_model]
            new_self_cache: Updated self-attention cache (if use_cache)
            new_cross_cache: Updated cross-attention cache (if use_cache)
        """
        # 1. Masked self-attention
        if use_cache:
            attn_output, _, new_self_cache = self.self_attn(
                x, x, x, tgt_mask, cache=self_attn_cache, use_cache=True
            )
        else:
            attn_output, _ = self.self_attn(x, x, x, tgt_mask)

        attn_output = self.dropout1(attn_output)
        x = self.norm1(x + attn_output)

        # 2. Cross-attention to encoder
        if use_cache:
            cross_attn_output, _, new_cross_cache = self.cross_attn(
                x, encoder_output, encoder_output, cross_mask,
                cache=cross_attn_cache, use_cache=True
            )
        else:
            cross_attn_output, _ = self.cross_attn(x, encoder_output, encoder_output, cross_mask)

        cross_attn_output = self.dropout2(cross_attn_output)
        x = self.norm2(x + cross_attn_output)

        # 3. Feed-forward
        ffn_output = self.ffn(x)
        ffn_output = self.dropout3(ffn_output)
        x = self.norm3(x + ffn_output)

        if use_cache:
            return x, new_self_cache, new_cross_cache
        return x
```

#### 4.2.3: Update TransformerDecoder

**File:** `src/models/transformer/decoder.py`

**Add cache support:**
```python
class TransformerDecoder(nn.Module):
    # ... existing code ...

    def forward(self, x, encoder_output, cross_mask=None, tgt_mask=None,
                cache=None, use_cache=False):
        """
        Forward with layer-wise caching.

        Args:
            cache: List of caches for each layer (or None)
            use_cache: Whether to return updated caches

        Returns:
            output: [batch, tgt_len, d_model]
            new_cache: List of updated caches (if use_cache)
        """
        if cache is None:
            cache = [None] * self.num_layers

        new_cache = []

        for i, layer in enumerate(self.layers):
            layer_cache = cache[i]

            if use_cache:
                x, new_self_cache, new_cross_cache = layer(
                    x, encoder_output, cross_mask, tgt_mask,
                    self_attn_cache=layer_cache.get('self', None) if layer_cache else None,
                    cross_attn_cache=layer_cache.get('cross', None) if layer_cache else None,
                    use_cache=True
                )
                new_cache.append({
                    'self': new_self_cache,
                    'cross': new_cross_cache
                })
            else:
                x = layer(x, encoder_output, cross_mask, tgt_mask)

        if use_cache:
            return x, new_cache
        return x
```

#### 4.2.4: Update Transformer Model

**File:** `src/models/transformer/transformer.py`

**Add incremental decode method:**
```python
class Transformer(nn.Module):
    # ... existing code ...

    def decode_incremental(self, tgt, encoder_output, cross_mask, tgt_mask, cache=None):
        """
        Incremental decoding with KV caching.

        Args:
            tgt: Target tokens [batch, 1] (single new token)
            encoder_output: Cached encoder output [batch, src_len, d_model]
            cross_mask: [batch, 1, 1, src_len] (for single new position)
            tgt_mask: [batch, 1, 1, cached_len+1] (causal mask for new position)
            cache: Previous cache from decoder layers

        Returns:
            logits: [batch, 1, vocab_size]
            new_cache: Updated cache
        """
        # Embed and add positional encoding (only for new token)
        tgt_embedded = self.tgt_embed(tgt) * self.embed_scale  # [batch, 1, d_model]
        tgt_encoded = self.pos_encoding(tgt_embedded)

        # Decode with caching
        decoder_output, new_cache = self.decoder(
            tgt_encoded, encoder_output, cross_mask, tgt_mask,
            cache=cache, use_cache=True
        )

        # Project to vocabulary
        if self.tie_embeddings:
            logits = torch.matmul(decoder_output, self.tgt_embed.weight.T)
        else:
            logits = self.output_projection(decoder_output)

        return logits, new_cache
```

---

### Phase 4.3: Cached Greedy Decoding

**File:** `src/inference/greedy_search_cached.py`

**Implementation:**
```python
def greedy_decode_cached(model, src, src_mask, max_len, bos_idx, eos_idx, device='cpu'):
    """
    Greedy decoding with KV caching for efficiency.

    Complexity: O(n) per step instead of O(n²)
    """
    model.eval()
    batch_size = src.size(0)

    # Step 1: Encode source once (reuse for all decoding steps)
    with torch.no_grad():
        encoder_output = model.encode(src, src_mask)

    # Step 2: Initialize
    tgt = torch.full((batch_size, 1), bos_idx, dtype=torch.long, device=device)
    cache = None  # Will be populated incrementally
    finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

    # Step 3: Generate tokens one by one
    for step in range(max_len):
        # Create masks for current step
        # Target mask: causal mask for positions [0, ..., step]
        current_len = tgt.size(1)
        tgt_mask = create_look_ahead_mask(current_len).to(device)  # [1, current_len, current_len]
        tgt_mask = tgt_mask.unsqueeze(0).expand(batch_size, 1, current_len, current_len)

        # For incremental: only need mask for last position attending to all previous
        tgt_mask_incremental = tgt_mask[:, :, -1:, :]  # [batch, 1, 1, current_len]

        # Cross mask: last decoder position can attend to all encoder positions
        src_len = src.size(1)
        cross_mask = torch.ones(batch_size, 1, 1, src_len, device=device)

        # Get only the last token for incremental decoding
        if step == 0:
            # First step: process BOS token
            tgt_input = tgt  # [batch, 1]
        else:
            # Subsequent steps: only process new token
            tgt_input = tgt[:, -1:]  # [batch, 1]

        # Forward pass with caching
        with torch.no_grad():
            logits, cache = model.decode_incremental(
                tgt_input, encoder_output, cross_mask, tgt_mask_incremental, cache
            )

        # Get next token (greedy)
        next_token = logits[:, -1, :].argmax(dim=-1)  # [batch]

        # Mark finished sequences
        finished |= (next_token == eos_idx)

        # Append to sequence
        tgt = torch.cat([tgt, next_token.unsqueeze(1)], dim=1)

        # Stop if all finished
        if finished.all():
            break

    return tgt
```

**Key optimizations:**
- Encoder output computed once, reused
- Only new token processed at each step (not entire sequence)
- Cache grows incrementally
- Complexity: O(n) instead of O(n²)

---

### Phase 4.4: Beam Search

**File:** `src/inference/beam_search.py`

**Challenges with caching:**
- Multiple beams (hypotheses) run in parallel
- Each beam has its own cache
- Beams can be pruned, caches must be reordered

**Implementation sketch:**
```python
def beam_search(model, src, src_mask, beam_size=4, max_len=100,
                bos_idx=2, eos_idx=3, length_penalty=0.6):
    """
    Beam search with KV caching.

    Key challenge: Managing cache for multiple beams.
    """
    batch_size = src.size(0)

    # Encode source
    encoder_output = model.encode(src, src_mask)

    # Expand for beam search: [batch * beam_size, ...]
    encoder_output = encoder_output.unsqueeze(1).expand(
        batch_size, beam_size, -1, -1
    ).contiguous().view(batch_size * beam_size, -1, encoder_output.size(-1))

    # Initialize beams
    beams = [Beam(beam_size, bos_idx, eos_idx) for _ in range(batch_size)]

    # Maintain cache for each beam
    beam_caches = [None] * (batch_size * beam_size)

    for step in range(max_len):
        # Get current tokens for all beams
        # Process with caching
        # Update beams
        # Reorder caches based on beam selection
        # ...

    # Return best hypothesis from each beam
```

**Note:** Beam search is more complex with caching. Consider implementing:
1. Simple beam search without cache first
2. Add caching once basic version works

---

### Phase 4.5: Translation Interface

**File:** `src/inference/translator.py`

**High-level API:**
```python
class Translator:
    """High-level translation interface."""

    def __init__(self, model, src_tokenizer, tgt_tokenizer, device='cpu'):
        self.model = model
        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer
        self.device = device
        self.model.to(device)
        self.model.eval()

    def translate(self, src_text, max_len=100, method='greedy', beam_size=4):
        """
        Translate text from source to target language.

        Args:
            src_text: Source text string or list of strings
            max_len: Maximum generation length
            method: 'greedy' or 'beam'
            beam_size: Beam size for beam search

        Returns:
            Translated text string or list of strings
        """
        # Handle single string or batch
        is_single = isinstance(src_text, str)
        if is_single:
            src_text = [src_text]

        # Tokenize
        src_tokens = [self.src_tokenizer.encode_ids(text) for text in src_text]
        src_tokens = [[self.src_tokenizer.bos_id] + ids + [self.src_tokenizer.eos_id]
                      for ids in src_tokens]

        # Pad and create batch
        max_src_len = max(len(ids) for ids in src_tokens)
        src_batch = torch.zeros(len(src_text), max_src_len, dtype=torch.long)
        for i, ids in enumerate(src_tokens):
            src_batch[i, :len(ids)] = torch.tensor(ids)
        src_batch = src_batch.to(self.device)

        # Create source mask
        src_mask = create_padding_mask(src_batch, pad_idx=0)

        # Decode
        if method == 'greedy':
            output = greedy_decode_cached(
                self.model, src_batch, src_mask, max_len,
                self.tgt_tokenizer.bos_id, self.tgt_tokenizer.eos_id, self.device
            )
        elif method == 'beam':
            output = beam_search(
                self.model, src_batch, src_mask, beam_size, max_len,
                self.tgt_tokenizer.bos_id, self.tgt_tokenizer.eos_id
            )
        else:
            raise ValueError(f"Unknown method: {method}")

        # Detokenize
        output_text = []
        for ids in output:
            ids = ids.tolist()
            # Remove BOS, EOS, padding
            if self.tgt_tokenizer.eos_id in ids:
                ids = ids[:ids.index(self.tgt_tokenizer.eos_id)]
            if ids and ids[0] == self.tgt_tokenizer.bos_id:
                ids = ids[1:]
            text = self.tgt_tokenizer.decode_ids(ids)
            output_text.append(text)

        return output_text[0] if is_single else output_text

    @classmethod
    def from_checkpoint(cls, checkpoint_path, vocab_dir, device='cpu'):
        """Load translator from checkpoint."""
        # Load config, model, tokenizers
        # Return Translator instance
        pass
```

---

## Implementation Order

### Step 1: Basic Greedy (No Cache) ✓
- Implement simple greedy_decode
- Test that it works
- Establish baseline

### Step 2: Add Cache to Attention ✓
- Modify MultiHeadAttention
- Test with dummy inputs
- Verify cache concatenation works

### Step 3: Add Cache to Decoder ✓
- Modify DecoderLayer and TransformerDecoder
- Test layer-wise caching
- Verify shapes

### Step 4: Add Incremental Decode ✓
- Add decode_incremental to Transformer
- Test end-to-end

### Step 5: Cached Greedy ✓
- Implement greedy_decode_cached
- Compare with uncached version (should match!)
- Measure speedup

### Step 6: Beam Search (Optional)
- Implement basic beam search
- Add caching later

### Step 7: Translation Interface ✓
- High-level API
- scripts/translate.py

---

## Testing Strategy

### Unit Tests
1. **Cache correctness:** Cached and uncached should give same results
2. **Shape tests:** Verify cache shapes at each layer
3. **Mask tests:** Ensure masks work with incremental decoding

### Integration Tests
1. **Small model test:** Overfit on 10 examples, verify can translate them
2. **Speed test:** Measure speedup from caching (should be 5-10x faster)
3. **Beam vs Greedy:** Compare BLEU scores

### Example Test
```python
def test_cache_correctness():
    """Verify cached and uncached give same output."""
    model = load_model()
    src = torch.randint(1, 1000, (2, 10))
    src_mask = create_padding_mask(src, 0)

    # Uncached
    output_uncached = greedy_decode(model, src, src_mask, 20, 2, 3)

    # Cached
    output_cached = greedy_decode_cached(model, src, src_mask, 20, 2, 3)

    # Should be identical
    assert torch.equal(output_uncached, output_cached)
```

---

## Performance Expectations

### Without Caching
- Complexity per step: O(n²) where n = current sequence length
- Total: O(n³) for generating n tokens

### With Caching
- Complexity per step: O(n)
- Total: O(n²) for generating n tokens

### Expected Speedup
- For max_len=50: **~10-20x faster**
- For max_len=100: **~30-50x faster**
- Memory overhead: Minimal (cache size ≈ model hidden states)

---

## Key Implementation Tips

1. **Start simple:** Get uncached version working first
2. **Test incrementally:** Test each component separately
3. **Verify correctness:** Cached must match uncached exactly
4. **Watch shapes:** Cache shapes are tricky, add assertions
5. **Device handling:** Ensure cache tensors are on correct device
6. **Memory:** Monitor memory usage, cache can grow large

---

## Files to Create/Modify

### New Files
- [ ] `src/inference/greedy_search.py` - Basic greedy
- [ ] `src/inference/greedy_search_cached.py` - Cached greedy
- [ ] `src/inference/beam_search.py` - Beam search
- [ ] `src/inference/translator.py` - High-level API
- [ ] `scripts/translate.py` - CLI tool
- [ ] `scripts/test_inference.py` - Test inference

### Modified Files
- [ ] `src/models/transformer/attention.py` - Add cache support
- [ ] `src/models/transformer/decoder.py` - Add cache support
- [ ] `src/models/transformer/transformer.py` - Add decode_incremental

---

## Next Steps

Ready to start implementing? I recommend this order:

1. **First:** Implement basic greedy_search.py (no cache)
   - Get it working
   - Verify output makes sense

2. **Second:** Add cache to attention.py
   - Modify forward() signature
   - Test cache concatenation

3. **Third:** Propagate cache through decoder
   - Modify DecoderLayer
   - Modify TransformerDecoder

4. **Fourth:** Implement cached greedy
   - Use decode_incremental
   - Verify matches uncached

5. **Fifth:** Add translation interface
   - High-level API
   - CLI tool

Which step would you like to start with?
