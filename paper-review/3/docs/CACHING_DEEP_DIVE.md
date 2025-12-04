# Deep Dive: What Can We Cache in Transformer Decoding?

## TL;DR

**You're right!** We can cache ALL intermediate variables in the decoder, not just K and V. This includes the final output after FFN for each layer.

## Why This Works: The Mathematical Proof

### Key Insight: Causal Masking Makes Previous Positions Independent

Consider decoder layer processing at step t:

```
Step t: Process positions [0, 1, 2, ..., t]
├─ Position 0: attends to [0]
├─ Position 1: attends to [0, 1]
├─ Position 2: attends to [0, 1, 2]
└─ Position t: attends to [0, 1, 2, ..., t]

Step t+1: Process positions [0, 1, 2, ..., t, t+1]
├─ Position 0: attends to [0]           ← SAME as step t
├─ Position 1: attends to [0, 1]        ← SAME as step t
├─ Position 2: attends to [0, 1, 2]     ← SAME as step t
├─ Position t: attends to [0, 1, ..., t] ← SAME as step t
└─ Position t+1: attends to [0, 1, ..., t, t+1] ← NEW
```

**Crucial observation:** Due to causal masking, position i never attends to positions after i. Therefore, when we add position t+1, the computations for positions [0, ..., t] remain **exactly the same**.

---

## What Can Be Cached?

### 1. ✅ Keys and Values (Standard KV Cache)

**Self-Attention:**
```python
# Step t: positions [0, ..., t]
Q_t = W_q @ x[0:t+1]  # [t+1, d_model]
K_t = W_k @ x[0:t+1]  # [t+1, d_model]
V_t = W_v @ x[0:t+1]  # [t+1, d_model]

# Step t+1: can reuse K_t, V_t for positions [0:t]
Q_new = W_q @ x[t+1]      # Only new query
K_new = W_k @ x[t+1]      # Only new key
V_new = W_v @ x[t+1]      # Only new value

K_t+1 = concat([K_t, K_new])  # Reuse cached K
V_t+1 = concat([V_t, V_new])  # Reuse cached V
```

**Cross-Attention:**
```python
# Encoder K, V computed ONCE, reused for all decoding steps
K_enc = W_k @ encoder_output  # Computed once
V_enc = W_v @ encoder_output  # Computed once
# Never changes during decoding!
```

**Memory:** `2 * num_layers * 2 (self+cross) * [batch, seq_len, d_model]`

---

### 2. ✅ Full Layer Outputs (Extended Cache)

**Why this works:**

For position i at step t:
```
layer_output[i] = FFN(CrossAttn(SelfAttn(x[i])))
```

At step t+1, position i:
- **Self-attention:** Still attends to same positions [0, ..., i]
- **Cross-attention:** Same query, same encoder K,V
- **FFN:** Position-wise, no dependency on other positions
- **Layer norm:** Position-wise normalization

**Therefore:** `layer_output[i]` at step t+1 = `layer_output[i]` at step t

**Implementation:**
```python
# At step t, cache the full output
cached_outputs[layer_idx] = layer_output[0:t+1]  # [t+1, d_model]

# At step t+1, only compute for new position
new_output = layer(x[t+1])  # [1, d_model]
full_output = concat([cached_outputs[layer_idx], new_output])
cached_outputs[layer_idx] = full_output  # Update cache
```

**Memory:** `num_layers * [batch, seq_len, d_model]`

---

### 3. ✅ Intermediate Sub-layer Outputs

You could even cache after each sub-layer:

```python
cache = {
    'layer_0': {
        'after_self_attn': [...],      # After self-attention + norm
        'after_cross_attn': [...],     # After cross-attention + norm
        'after_ffn': [...]             # After FFN + norm (full output)
    },
    'layer_1': { ... },
    ...
}
```

**Why this works:** Each sub-layer's output at position i depends only on:
- Previous positions [0, ..., i] (due to causal masking)
- Encoder output (fixed)

---

## Comparison: Different Caching Strategies

### Strategy 1: KV Cache Only (Standard)

```python
def decode_step(new_token, encoder_out, kv_cache):
    x = embed(new_token)  # [1, d_model]

    for layer in decoder_layers:
        # Self-attention
        q = layer.self_attn.W_q(x)
        k = layer.self_attn.W_k(x)
        v = layer.self_attn.W_v(x)

        # Use cached K, V
        k_full = concat([kv_cache[layer]['self_k'], k])
        v_full = concat([kv_cache[layer]['self_v'], v])

        x = layer.self_attn(q, k_full, v_full)  # Compute attention
        x = layer.norm1(x)                      # Still need to compute

        # Cross-attention
        q = layer.cross_attn.W_q(x)
        x = layer.cross_attn(q, kv_cache[layer]['cross_k'], kv_cache[layer]['cross_v'])
        x = layer.norm2(x)                      # Still need to compute

        # FFN
        x = layer.ffn(x)                        # Still need to compute
        x = layer.norm3(x)                      # Still need to compute

    return x
```

**Computation per step:**
- Attention: O(1) for new position (using cached K,V)
- FFN: Still computed
- Layer norm: Still computed
- **Total:** Saves attention computation, but FFN still runs

---

### Strategy 2: Full Output Cache (Alternative)

```python
def decode_step(new_token, encoder_out, output_cache):
    x = embed(new_token)  # [1, d_model]

    for layer_idx, layer in enumerate(decoder_layers):
        # Only compute for NEW position using cached context

        # Self-attention needs context from previous positions
        # But we can use pre-computed K, V from cache
        prev_outputs = output_cache[layer_idx - 1] if layer_idx > 0 else prev_layer_output

        x = layer.forward_incremental(x, prev_outputs, encoder_out)
        # layer.forward_incremental internally uses cached K,V

        # Cache this layer's output for position t
        output_cache[layer_idx] = concat([output_cache[layer_idx], x])

    return x
```

**Computation per step:**
- Attention: O(1) using cached K,V
- FFN: Still computed for new position
- Layer norm: Still computed for new position
- **Total:** Same as KV cache (FFN cannot be avoided for new position)

---

## The Verdict: What Should We Cache?

### KV Cache (Standard) - RECOMMENDED ✅

**Pros:**
- ✅ Maximum speedup with minimal memory
- ✅ Well-established pattern, easy to implement
- ✅ Most computation is in attention (O(n²)), which we optimize
- ✅ FFN is cheap (O(d_model²)), less benefit from caching

**Cons:**
- ❌ Still need to compute FFN, LayerNorm for new position

**Memory:** `~2n * num_layers * d_model` per sample

---

### Full Output Cache (Alternative)

**Pros:**
- ✅ Can skip re-computing outputs for previous positions entirely
- ✅ Conceptually simpler: each layer just passes cached outputs + new output

**Cons:**
- ❌ More memory: `~n * num_layers * d_model` per sample
- ❌ Same computational complexity (still need FFN for new position)
- ❌ More complex bookkeeping

**Memory:** `~n * num_layers * d_model` per sample

---

### Hybrid: Cache Everything (Overkill)

**Pros:**
- ✅ Maximum flexibility
- ✅ Can resume from any intermediate state

**Cons:**
- ❌❌ Memory explosion: `~3n * num_layers * d_model` per sample
- ❌ No additional speedup (still compute same things for new position)
- ❌ Complex implementation

---

## Computational Analysis

### What We MUST Compute for New Position

Even with perfect caching, for each new token we must compute:

```python
# For position t (new token)
x_t = embed(token_t)                    # O(d_model)

for layer in layers:
    # Self-attention
    q_t = W_q @ x_t                     # O(d_model²)
    k_t = W_k @ x_t                     # O(d_model²)
    v_t = W_v @ x_t                     # O(d_model²)

    # Attention with cached K, V
    scores = q_t @ K_cached.T           # O(t * d_model)
    attn = softmax(scores / sqrt(d_k))  # O(t)
    out = attn @ V_cached               # O(t * d_model)

    x_t = norm(x_t + out)               # O(d_model)

    # Cross-attention (similar)
    # ...

    # FFN - MUST compute for new position
    ffn_out = W2 @ relu(W1 @ x_t)       # O(d_model * d_ff)
    x_t = norm(x_t + ffn_out)           # O(d_model)
```

**Bottlenecks:**
1. **Attention over cached K,V:** O(t * d_model) - grows with sequence length
2. **FFN:** O(d_model * d_ff) - constant per position

**Key insight:** No matter what we cache, we still need FFN for the new position!

---

## Why KV Cache is Optimal

### Without any cache:
```
Step t: O(t² * d_model + t * d_ff)
Total for n steps: O(n³ * d_model + n² * d_ff)
```

### With KV cache:
```
Step t: O(t * d_model + d_ff)
Total for n steps: O(n² * d_model + n * d_ff)
```

### With full output cache:
```
Step t: O(t * d_model + d_ff)  ← Same as KV cache!
Total for n steps: O(n² * d_model + n * d_ff)  ← Same as KV cache!
```

**Conclusion:** Full output caching gives **no additional speedup** over KV caching, just uses more memory.

---

## Practical Recommendation

### For your implementation: Use KV Cache ✅

**Reasons:**
1. **Optimal memory/speed tradeoff**
2. **Standard in literature** (GPT-2, GPT-3, BERT, etc. use KV cache)
3. **Easier to implement and debug**
4. **No performance penalty** vs more complex schemes

### When full output cache might make sense:

1. **Speculative decoding:** When you might want to "rewind" computation
2. **Beam search with large beams:** Easier to manage beam states
3. **Interactive editing:** When you want to modify middle of sequence

But for standard greedy/beam search translation: **KV cache is perfect**.

---

## Example: Cache Sizes

For a typical model:
- `d_model = 512`
- `num_layers = 6`
- `batch_size = 4`
- `seq_len = 100`

**KV Cache:**
```
2 (K,V) * 2 (self, cross) * 6 layers * 4 batch * 100 seq * 512 dim
= ~24 MB
```

**Full Output Cache:**
```
6 layers * 4 batch * 100 seq * 512 dim
= ~12 MB
```

**Both together:**
```
~36 MB per sample
```

For inference on modern GPUs (16+ GB), this is negligible.

---

## Conclusion

**You're absolutely correct** that we could cache full layer outputs. However:

1. ✅ **KV caching is optimal** for the standard autoregressive decoding use case
2. ✅ **Full output caching** provides no additional speedup
3. ✅ **KV caching is simpler** and is the industry standard
4. ⚠️ **Full caching** makes sense for specific use cases (speculative decoding, interactive editing)

**Recommendation:** Implement KV caching as planned. It's the sweet spot! 🎯

---

## Visual Summary

```
┌─────────────────────────────────────────────────────────┐
│ What Needs to be Computed for NEW Token (Position t)?  │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  ✅ MUST COMPUTE (no matter what we cache):            │
│     • Embedding of new token                           │
│     • Q, K, V projections for new token                │
│     • Attention scores (Q @ cached K)                  │
│     • Attention output (scores @ cached V)             │
│     • FFN for new token  ← CANNOT SKIP                 │
│     • Layer normalization for new token                │
│                                                         │
│  ❌ CAN SKIP (with KV cache):                          │
│     • K, V for previous positions (use cached)         │
│     • Attention for previous positions                 │
│                                                         │
│  ❌ CANNOT SKIP (even with full output cache):         │
│     • FFN for new position (position-wise)             │
│     • Layer norm for new position                      │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

The key insight: **FFN must run for each new token regardless of caching strategy**, so KV caching already captures most of the benefit!
