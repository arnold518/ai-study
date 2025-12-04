# Mask Shape Bug Fix - Detailed Explanation

## Summary

Fixed a major bug in the masking system where masks had incorrect shapes, causing semantic inconsistencies in the attention mechanism.

## The Bug

### Problem
The masking utilities were creating masks with shapes that, while functionally working due to PyTorch broadcasting, were **semantically incorrect** and didn't match the documented specifications in the code.

### What Was Wrong

**Before the fix:**
```python
# masking.py
create_padding_mask(seq, pad_idx)
# Returned: [batch, 1, 1, seq_len] ❌

# dataset.py collate_fn
src_mask = create_padding_mask(src, pad_idx)  # [batch, 1, 1, src_len]
tgt_mask = create_target_mask(tgt, pad_idx)   # [batch, 1, tgt_len, tgt_len] ✓
# No cross_mask created ❌
```

**Expected by attention mechanism:**
```python
# attention.py forward() expects:
mask: [batch_size, 1, seq_len, seq_len2]

# For different attention types:
# 1. Encoder self-attention: [batch, 1, src_len, src_len]
# 2. Decoder self-attention: [batch, 1, tgt_len, tgt_len]
# 3. Decoder cross-attention: [batch, 1, tgt_len, src_len]
```

## Why This is a Major Bug

### 1. **Semantic Mismatch**
The attention scores matrix is `[batch, heads, query_len, key_len]`:
- **Query dimension (dim -2)**: Which position is querying
- **Key dimension (dim -1)**: Which position is being queried

The mask should have shape `[batch, 1, query_len, key_len]` to explicitly specify:
- `mask[b, 0, i, j] = 1` means "position i CAN attend to position j"
- `mask[b, 0, i, j] = 0` means "position i CANNOT attend to position j"

With shape `[batch, 1, 1, key_len]`, we're relying on broadcasting to repeat the mask for all query positions, which:
- Works functionally but is **semantically unclear**
- Hides the explicit query-key relationship
- Makes debugging harder
- Doesn't match documentation

### 2. **Cross-Attention Missing**
The decoder cross-attention needs a mask of shape `[batch, 1, tgt_len, src_len]`:
- Query positions: target sequence (tgt_len)
- Key positions: source sequence (src_len)
- Different lengths mean broadcasting `[batch, 1, 1, src_len]` works but is **wrong semantically**

## Detailed Explanation of Masks

### Mask Convention
In this implementation:
- `1` (or `True`): Position can be attended to
- `0` (or `False`): Position is masked (cannot be attended)

In attention calculation:
```python
scores = scores.masked_fill(mask==0, float('-inf'))
```
Positions with `mask==0` get `-inf`, becoming 0 after softmax (no attention).

### Three Types of Masks

#### 1. Encoder Self-Attention Mask
**Purpose:** Mask padding tokens in source sequence

**Shape:** `[batch, 1, src_len, src_len]`

**Semantics:**
```
mask[b, 0, i, j] = 1 if src[b, j] != PAD
mask[b, 0, i, j] = 0 if src[b, j] == PAD
```
- Every query position i has the same mask pattern
- All positions can attend to non-padding key positions
- Matrix is same along rows (all rows identical)

**Example (src_len=5, last token is padding):**
```
mask[0, 0] = [[1, 1, 1, 1, 0],
              [1, 1, 1, 1, 0],
              [1, 1, 1, 1, 0],
              [1, 1, 1, 1, 0],
              [1, 1, 1, 1, 0]]
```

#### 2. Decoder Self-Attention Mask
**Purpose:** Causal masking (can't see future) + padding masking

**Shape:** `[batch, 1, tgt_len, tgt_len]`

**Semantics:**
```
mask[b, 0, i, j] = 1 if (j <= i) AND (tgt[b, j] != PAD)
mask[b, 0, i, j] = 0 otherwise
```
- Position i can only attend to positions 0...i (causal)
- Cannot attend to future positions (j > i)
- Cannot attend to padding positions

**Example (tgt_len=6, last token is padding):**
```
mask[0, 0] = [[1, 0, 0, 0, 0, 0],   # pos 0 can only see pos 0
              [1, 1, 0, 0, 0, 0],   # pos 1 can see 0,1
              [1, 1, 1, 0, 0, 0],   # pos 2 can see 0,1,2
              [1, 1, 1, 1, 0, 0],   # pos 3 can see 0,1,2,3
              [1, 1, 1, 1, 1, 0],   # pos 4 can see 0,1,2,3,4
              [0, 0, 0, 0, 0, 0]]   # pos 5 (padding) sees nothing
```

#### 3. Decoder Cross-Attention Mask
**Purpose:** Mask padding in source when decoder attends to encoder

**Shape:** `[batch, 1, tgt_len, src_len]`

**Semantics:**
```
mask[b, 0, i, j] = 1 if src[b, j] != PAD
mask[b, 0, i, j] = 0 if src[b, j] == PAD
```
- Query from target (length tgt_len)
- Key/Value from source (length src_len)
- Each target position can attend to all non-padding source positions

**Example (tgt_len=4, src_len=5, last source token is padding):**
```
mask[0, 0] = [[1, 1, 1, 1, 0],   # tgt pos 0 → all non-pad src
              [1, 1, 1, 1, 0],   # tgt pos 1 → all non-pad src
              [1, 1, 1, 1, 0],   # tgt pos 2 → all non-pad src
              [1, 1, 1, 1, 0]]   # tgt pos 3 → all non-pad src
```

## The Fix

### 1. Updated `create_padding_mask()`
```python
def create_padding_mask(seq, pad_idx):
    """Returns: [batch_size, 1, seq_len, seq_len]"""
    mask = (seq != pad_idx).unsqueeze(1).unsqueeze(2)  # [batch, 1, 1, seq_len]
    batch_size, _, _, seq_len = mask.size()
    mask = mask.expand(batch_size, 1, seq_len, seq_len)  # Expand explicitly
    return mask
```

**Changes:**
- Now returns `[batch, 1, seq_len, seq_len]` instead of `[batch, 1, 1, seq_len]`
- Explicitly expands the mask to show all query-key relationships
- Makes the mask pattern semantically clear

### 2. Added `create_cross_attention_mask()`
```python
def create_cross_attention_mask(src, tgt, pad_idx):
    """Returns: [batch_size, 1, tgt_len, src_len]"""
    batch_size, src_len = src.size()
    tgt_len = tgt.size(1)

    # Create mask indicating non-padding positions in source
    src_mask = (src != pad_idx).unsqueeze(1).unsqueeze(2)

    # Expand to [batch_size, 1, tgt_len, src_len]
    src_mask = src_mask.expand(batch_size, 1, tgt_len, src_len)

    return src_mask
```

**What it does:**
- Creates mask for cross-attention with correct shape
- Explicitly shows target-to-source attention pattern
- Handles different sequence lengths (tgt_len != src_len)

### 3. Updated `collate_fn()`
```python
def collate_fn(batch, pad_idx=0):
    # ... padding ...

    src_mask = create_padding_mask(src_padded, pad_idx)
    tgt_mask = create_target_mask(tgt_padded, pad_idx)
    cross_mask = create_cross_attention_mask(src_padded, tgt_padded, pad_idx)

    return {
        'src': src_padded,
        'tgt': tgt_padded,
        'src_mask': src_mask,      # [batch, 1, src_len, src_len]
        'tgt_mask': tgt_mask,      # [batch, 1, tgt_len, tgt_len]
        'cross_mask': cross_mask   # [batch, 1, tgt_len, src_len]
    }
```

**Changes:**
- Now creates three separate masks
- Each mask has the correct explicit shape
- Cross-attention mask is now provided

### 4. Updated Model Forward Pass
```python
# transformer.py
def forward(self, src, tgt, src_mask=None, tgt_mask=None, cross_mask=None):
    encoder_output = self.encoder(src_encoded, src_mask)
    decoder_output = self.decoder(tgt_encoded, encoder_output, cross_mask, tgt_mask)
    # ...

# decoder.py
def forward(self, x, encoder_output, cross_mask=None, tgt_mask=None):
    # Self-attention (uses tgt_mask)
    attn_output, _ = self.self_attn(x, x, x, tgt_mask)

    # Cross-attention (uses cross_mask)
    cross_attn_output, _ = self.cross_attn(x, encoder_output, encoder_output, cross_mask)
    # ...
```

**Changes:**
- Added `cross_mask` parameter throughout the stack
- Properly separate masks for self-attention vs cross-attention
- Clear semantic distinction between mask types

### 5. Updated Trainer
```python
# trainer.py
def train_epoch(self):
    for batch in tqdm(self.train_loader, desc="Training"):
        src_mask = batch['src_mask'].to(self.device)
        tgt_mask = batch['tgt_mask'].to(self.device)
        cross_mask = batch['cross_mask'].to(self.device)

        # Adjust masks for decoder input
        tgt_input_mask = tgt_mask[:, :, :-1, :-1]
        cross_input_mask = cross_mask[:, :, :-1, :]

        logits = self.model(src, tgt_input, src_mask, tgt_input_mask, cross_input_mask)
        # ...
```

**Changes:**
- Uses all three mask types
- Properly adjusts mask dimensions when removing last token
- Clear and explicit mask handling

## Test Results

### Mask Shape Verification
```
Source mask shape: torch.Size([2, 1, 5, 5])
✓ Source mask shape correct

Target mask shape: torch.Size([2, 1, 6, 6])
✓ Target mask shape correct

Cross-attention mask shape: torch.Size([2, 1, 6, 5])
✓ Cross-attention mask shape correct
```

### Target Mask Pattern (Causal)
```
tensor([[1, 0, 0, 0, 0, 0],
        [1, 1, 0, 0, 0, 0],
        [1, 1, 1, 0, 0, 0],
        [1, 1, 1, 1, 0, 0],
        [1, 1, 1, 1, 1, 0],
        [1, 1, 1, 1, 1, 1]])
```
✓ Correct lower triangular pattern

### Full Pipeline Test
```
Batch shapes:
  Source: torch.Size([4, 26])
  Target: torch.Size([4, 28])
  Source mask: torch.Size([4, 1, 26, 26])
  Target mask: torch.Size([4, 1, 28, 28])
  Cross mask: torch.Size([4, 1, 28, 26])

Output logits shape: torch.Size([4, 27, 16000])
Loss: 8.3851

✓ Full pipeline test PASSED!
```

## Files Modified

1. **src/utils/masking.py**
   - Fixed `create_padding_mask()` to return `[batch, 1, seq_len, seq_len]`
   - Added `create_cross_attention_mask()`
   - Updated documentation

2. **src/data/dataset.py**
   - Updated `collate_fn()` to create cross_mask
   - Returns all three mask types with correct shapes

3. **src/models/transformer/transformer.py**
   - Added `cross_mask` parameter to `forward()`
   - Updated `decode()` method
   - Fixed mask passing to decoder

4. **src/models/transformer/decoder.py**
   - Changed parameter from `src_mask` to `cross_mask`
   - Updated documentation
   - Clarified mask usage in cross-attention

5. **src/models/transformer/encoder.py**
   - Updated documentation for clarity

6. **src/training/trainer.py**
   - Added cross_mask handling
   - Properly adjusts all three mask types for decoder input

## Why Broadcasting Worked But Was Wrong

PyTorch broadcasting rules:
```python
scores: [batch, heads, query_len, key_len]
mask:   [batch, 1,     1,         key_len]  # Old (broadcasts)
mask:   [batch, 1,     query_len, key_len]  # New (explicit)
```

Both broadcast to `[batch, heads, query_len, key_len]`, but:

**Old (implicit):**
- Relies on broadcasting magic
- Unclear which dimensions represent what
- Hard to debug
- Doesn't match documentation

**New (explicit):**
- Clear semantic meaning
- Query and key dimensions are explicit
- Matches attention score dimensions
- Self-documenting code
- Easier to debug

## Impact

### Before Fix
✓ Code worked functionally (due to broadcasting)
✗ Semantically incorrect shapes
✗ Confusing documentation
✗ Missing cross-attention mask
✗ Hard to understand and debug

### After Fix
✓ Semantically correct shapes
✓ Clear documentation
✓ All three mask types properly handled
✓ Easy to understand and debug
✓ Matches standard Transformer implementation

## References

1. "Attention Is All You Need" (Vaswani et al., 2017) - Section 3.2.3 on masking
2. The Annotated Transformer - http://nlp.seas.harvard.edu/annotated-transformer/
3. PyTorch Broadcasting Semantics - https://pytorch.org/docs/stable/notes/broadcasting.html
