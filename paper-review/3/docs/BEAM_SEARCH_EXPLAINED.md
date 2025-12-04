# Beam Search: Complete Guide with Implementation

## What is Beam Search?

Beam search is a **heuristic search algorithm** used to find the most likely sequence in autoregressive generation. It's a compromise between:
- **Greedy search** (fast but suboptimal)
- **Exhaustive search** (optimal but exponentially expensive)

## Greedy Search vs Beam Search

### Greedy Search (Baseline)

At each step, pick the **single most likely token**:

```
Step 0: "The" (100%)
Step 1: "The cat" (100% × 80% = 80%)
Step 2: "The cat sat" (80% × 70% = 56%)

Final sequence: "The cat sat" (56% total probability)
```

**Problem:** Greedy choices can lead to suboptimal global sequences.

**Example:**
```
Greedy:
  Step 0: "The" (prob=0.5)
  Step 1: "The dog" (prob=0.4) → Total: 0.2
  Step 2: "The dog sleeps" (prob=0.3) → Total: 0.06

But better sequence exists:
  Step 0: "The" (prob=0.5)
  Step 1: "The cat" (prob=0.3) → Total: 0.15  ← Lower than "dog"!
  Step 2: "The cat is sleeping" (prob=0.8) → Total: 0.12  ← Better overall!
```

### Beam Search (Better)

Maintain **top-k hypotheses** (beams) at each step:

```
beam_size = 2

Step 0: Keep top 2
  1. "The" (100%)
  2. "A" (80%)

Step 1: Expand each, keep top 2
  1. "The cat" (100% × 80% = 80%)
  2. "The dog" (100% × 70% = 70%)

Step 2: Expand each, keep top 2
  1. "The cat sat" (80% × 90% = 72%)
  2. "The cat ran" (80% × 85% = 68%)
```

**Key idea:** By keeping multiple hypotheses, we can recover from local suboptimal choices.

---

## The Algorithm

### High-Level Overview

```python
def beam_search(model, src, beam_size, max_len):
    # 1. Initialize beam with BOS token
    beams = [Hypothesis(tokens=[BOS], score=0.0)]

    # 2. For each step
    for step in range(max_len):
        # 3. Expand each beam by trying all vocab
        candidates = []
        for beam in beams:
            if beam.finished:  # Skip finished beams
                candidates.append(beam)
                continue

            # Get predictions for next token
            logits = model.predict_next(beam.tokens)
            log_probs = log_softmax(logits)

            # Create candidates by appending each token
            for token_id in range(vocab_size):
                new_beam = Hypothesis(
                    tokens=beam.tokens + [token_id],
                    score=beam.score + log_probs[token_id]
                )
                candidates.append(new_beam)

        # 4. Keep top-k candidates
        beams = sorted(candidates, key=lambda x: x.score, reverse=True)[:beam_size]

        # 5. Stop if all beams are finished
        if all(beam.finished for beam in beams):
            break

    # 6. Return best beam
    return beams[0]
```

### Key Components

1. **Hypothesis**: A candidate sequence with its score
2. **Expansion**: For each beam, try all possible next tokens
3. **Pruning**: Keep only top-k scored beams
4. **Termination**: Stop when EOS is generated or max length reached

---

## Detailed Implementation

### Step 1: Hypothesis Class

```python
import torch
from dataclasses import dataclass
from typing import List

@dataclass
class Hypothesis:
    """A single beam hypothesis."""
    tokens: List[int]          # Token IDs generated so far
    score: float               # Cumulative log probability
    finished: bool = False     # Whether EOS has been generated

    def __len__(self):
        return len(self.tokens)

    def get_normalized_score(self, alpha=0.6):
        """
        Length-normalized score to prevent bias towards shorter sequences.

        Score = log_prob / length^alpha

        alpha=0.0: No normalization (favors shorter)
        alpha=1.0: Full normalization (penalizes longer)
        alpha=0.6: Recommended by Wu et al. (2016)
        """
        length = len(self.tokens)
        return self.score / (length ** alpha)
```

### Step 2: Beam Search Function

```python
def beam_search(
    model,
    src,                    # Source sequence [batch_size, src_len]
    src_mask,              # Source mask
    beam_size=4,
    max_len=100,
    bos_idx=2,
    eos_idx=3,
    alpha=0.6,             # Length penalty
    device='cpu'
):
    """
    Beam search for single source sequence.

    Args:
        model: Trained Transformer model
        src: Source tokens [1, src_len] (batch_size must be 1 for simplicity)
        src_mask: Source mask [1, 1, src_len, src_len]
        beam_size: Number of beams to maintain
        max_len: Maximum generation length
        bos_idx: Beginning of sequence token
        eos_idx: End of sequence token
        alpha: Length normalization factor
        device: Device to run on

    Returns:
        best_hypothesis: Hypothesis with highest score
    """
    model.eval()

    # Encode source once (shared across all beams)
    with torch.no_grad():
        encoder_output = model.encode(src, src_mask)  # [1, src_len, d_model]

    # Initialize beam with BOS token
    beams = [Hypothesis(tokens=[bos_idx], score=0.0)]
    completed_beams = []

    # Generate tokens step by step
    for step in range(max_len):
        candidates = []

        # Expand each beam
        for beam in beams:
            if beam.finished:
                # Keep finished beams as-is
                candidates.append(beam)
                continue

            # Prepare input for this beam
            tgt = torch.tensor([beam.tokens], dtype=torch.long, device=device)  # [1, seq_len]

            # Create masks
            tgt_mask = create_target_mask(tgt, pad_idx=0).to(device)
            cross_mask = create_cross_attention_mask(src, tgt, pad_idx=0).to(device)

            # Forward pass
            with torch.no_grad():
                logits = model.decode(tgt, encoder_output, cross_mask, tgt_mask)
                # logits: [1, seq_len, vocab_size]

                # Get logits for last position (next token prediction)
                next_token_logits = logits[0, -1, :]  # [vocab_size]

                # Convert to log probabilities
                log_probs = torch.log_softmax(next_token_logits, dim=-1)

            # Create new hypotheses by appending each possible token
            # Only consider top-k tokens to speed up (optional optimization)
            top_k = min(beam_size * 2, log_probs.size(0))
            top_log_probs, top_indices = torch.topk(log_probs, top_k)

            for log_prob, token_id in zip(top_log_probs, top_indices):
                token_id = token_id.item()
                log_prob = log_prob.item()

                new_tokens = beam.tokens + [token_id]
                new_score = beam.score + log_prob

                # Check if this beam is finished
                is_finished = (token_id == eos_idx)

                new_beam = Hypothesis(
                    tokens=new_tokens,
                    score=new_score,
                    finished=is_finished
                )

                candidates.append(new_beam)

        # Sort candidates by score (or normalized score)
        # Using normalized score prevents bias towards shorter sequences
        candidates = sorted(
            candidates,
            key=lambda h: h.get_normalized_score(alpha),
            reverse=True
        )

        # Keep top beam_size candidates
        beams = candidates[:beam_size]

        # Move finished beams to completed list
        completed_beams.extend([b for b in beams if b.finished])
        beams = [b for b in beams if not b.finished]

        # Stop if we have enough completed beams
        if len(completed_beams) >= beam_size:
            break

        # Stop if all beams are finished
        if len(beams) == 0:
            break

    # Add any remaining beams to completed
    completed_beams.extend(beams)

    # Return best completed beam
    if completed_beams:
        best_beam = max(completed_beams, key=lambda h: h.get_normalized_score(alpha))
        return best_beam
    else:
        # Fallback if nothing completed
        return Hypothesis(tokens=[bos_idx, eos_idx], score=-float('inf'))
```

---

## Visualization: Beam Search in Action

### Example: beam_size=2, vocabulary={a, b, c, d, EOS}

```
Step 0: Initialize
┌─────────────────┐
│ Beams:          │
│ 1. [BOS]        │
│    score=0.0    │
└─────────────────┘

Step 1: Expand BOS → {a, b, c, d, EOS}
┌─────────────────────────────────────────┐
│ Candidates (after expansion):           │
│ 1. [BOS, a]    score=-0.5  ← Keep      │
│ 2. [BOS, b]    score=-0.7  ← Keep      │
│ 3. [BOS, c]    score=-1.2              │
│ 4. [BOS, d]    score=-1.5              │
│ 5. [BOS, EOS]  score=-2.0              │
└─────────────────────────────────────────┘
Keep top 2 beams

Step 2: Expand 2 beams × vocab_size tokens
┌─────────────────────────────────────────┐
│ Expand [BOS, a]:                        │
│   [BOS, a, a]    score=-0.5 + -0.3 = -0.8 ← Keep │
│   [BOS, a, b]    score=-0.5 + -0.6 = -1.1        │
│   [BOS, a, c]    score=-0.5 + -0.8 = -1.3        │
│   [BOS, a, EOS]  score=-0.5 + -1.0 = -1.5        │
│                                         │
│ Expand [BOS, b]:                        │
│   [BOS, b, a]    score=-0.7 + -0.4 = -1.1        │
│   [BOS, b, b]    score=-0.7 + -0.2 = -0.9 ← Keep │
│   [BOS, b, c]    score=-0.7 + -0.9 = -1.6        │
│   [BOS, b, EOS]  score=-0.7 + -0.5 = -1.2        │
└─────────────────────────────────────────┘
Keep top 2: [BOS, a, a] and [BOS, b, b]

Step 3: Continue until EOS or max_len...
```

---

## Length Normalization

### The Problem

Without normalization, beam search **favors shorter sequences** because:
```
Sequence 1: [a, b, EOS]
  score = log(0.5) + log(0.4) + log(0.8)
        = -0.69 + -0.92 + -0.22
        = -1.83

Sequence 2: [a, b, c, d, EOS]
  score = log(0.5) + log(0.4) + log(0.6) + log(0.5) + log(0.7)
        = -0.69 + -0.92 + -0.51 + -0.69 + -0.36
        = -3.17  ← Lower score (worse) just because it's longer!
```

### The Solution: Length Normalization

Normalize by sequence length with penalty factor α:

```python
normalized_score = score / (length ** alpha)
```

**Common values:**
- `alpha = 0.0`: No normalization (favors short)
- `alpha = 1.0`: Full normalization (may favor long)
- `alpha = 0.6`: Recommended (Wu et al., 2016) - good balance

**Example:**
```python
Sequence 1: score=-1.83, length=3
  normalized = -1.83 / (3 ** 0.6) = -1.83 / 2.05 = -0.89

Sequence 2: score=-3.17, length=5
  normalized = -3.17 / (5 ** 0.6) = -3.17 / 2.63 = -1.21

Now Sequence 1 scores better! ✓
```

---

## Batch Beam Search

For efficiency, we can process multiple source sequences in parallel:

```python
def batch_beam_search(model, src_batch, src_mask_batch, beam_size=4, max_len=100,
                      bos_idx=2, eos_idx=3, alpha=0.6, device='cpu'):
    """
    Beam search for batch of source sequences.

    Args:
        src_batch: [batch_size, src_len]
        src_mask_batch: [batch_size, 1, src_len, src_len]

    Returns:
        List of best hypotheses (one per source)
    """
    batch_size = src_batch.size(0)
    results = []

    # Process each source independently
    # (Can be parallelized further with batched beam search)
    for i in range(batch_size):
        src = src_batch[i:i+1]        # [1, src_len]
        src_mask = src_mask_batch[i:i+1]  # [1, 1, src_len, src_len]

        best_hyp = beam_search(
            model, src, src_mask, beam_size, max_len,
            bos_idx, eos_idx, alpha, device
        )

        results.append(best_hyp)

    return results
```

---

## Advanced: Beam Search with KV Caching

Managing cache for multiple beams is more complex:

```python
def beam_search_with_cache(model, src, src_mask, beam_size=4, max_len=100,
                           bos_idx=2, eos_idx=3, alpha=0.6, device='cpu'):
    """
    Beam search with KV caching for efficiency.

    Challenge: Each beam has its own cache that needs to be managed.
    """
    model.eval()

    # Encode source
    with torch.no_grad():
        encoder_output = model.encode(src, src_mask)

    # Initialize beams with caches
    beams = [{
        'hypothesis': Hypothesis(tokens=[bos_idx], score=0.0),
        'cache': None  # Will be populated during decoding
    }]

    for step in range(max_len):
        candidates = []

        for beam_data in beams:
            beam = beam_data['hypothesis']
            cache = beam_data['cache']

            if beam.finished:
                candidates.append(beam_data)
                continue

            # Prepare input (only last token for incremental decoding)
            if step == 0:
                tgt_input = torch.tensor([[bos_idx]], device=device)
            else:
                tgt_input = torch.tensor([[beam.tokens[-1]]], device=device)

            # Create masks for current step
            current_len = len(beam.tokens)
            tgt_mask = create_target_mask_for_position(current_len, device)
            cross_mask = torch.ones(1, 1, 1, src.size(1), device=device)

            # Incremental decode with cache
            with torch.no_grad():
                logits, new_cache = model.decode_incremental(
                    tgt_input, encoder_output, cross_mask, tgt_mask, cache
                )

                next_token_logits = logits[0, -1, :]
                log_probs = torch.log_softmax(next_token_logits, dim=-1)

            # Expand beam
            top_k = min(beam_size * 2, log_probs.size(0))
            top_log_probs, top_indices = torch.topk(log_probs, top_k)

            for log_prob, token_id in zip(top_log_probs, top_indices):
                token_id = token_id.item()
                log_prob = log_prob.item()

                new_beam = Hypothesis(
                    tokens=beam.tokens + [token_id],
                    score=beam.score + log_prob,
                    finished=(token_id == eos_idx)
                )

                # Clone cache for this beam
                # Important: Each beam needs its own cache copy!
                new_beam_cache = clone_cache(new_cache)

                candidates.append({
                    'hypothesis': new_beam,
                    'cache': new_beam_cache
                })

        # Sort and keep top beams
        candidates = sorted(
            candidates,
            key=lambda x: x['hypothesis'].get_normalized_score(alpha),
            reverse=True
        )
        beams = candidates[:beam_size]

        if all(b['hypothesis'].finished for b in beams):
            break

    # Return best beam
    best = max(beams, key=lambda x: x['hypothesis'].get_normalized_score(alpha))
    return best['hypothesis']


def clone_cache(cache):
    """Deep copy cache for beam branching."""
    if cache is None:
        return None

    new_cache = []
    for layer_cache in cache:
        new_layer_cache = {}
        for key in ['self', 'cross']:
            if key in layer_cache:
                new_layer_cache[key] = {
                    'key': layer_cache[key]['key'].clone(),
                    'value': layer_cache[key]['value'].clone()
                }
        new_cache.append(new_layer_cache)

    return new_cache
```

---

## Comparison: Greedy vs Beam Search

### Translation Example

**Source:** "고양이가 자고 있어요" (Korean)

**Greedy Search (beam_size=1):**
```
Output: "The cat sleeping"
BLEU: 32.5
```

**Beam Search (beam_size=4):**
```
Beam 1: "The cat is sleeping"    score: -2.3  ← Best
Beam 2: "A cat is sleeping"      score: -2.5
Beam 3: "The cat sleeps"         score: -2.7
Beam 4: "Cat is sleeping"        score: -3.1

Output: "The cat is sleeping"
BLEU: 45.2  ← Better!
```

### Typical Performance

| Beam Size | BLEU Score | Speed | Memory |
|-----------|------------|-------|---------|
| 1 (greedy) | 30.0 | 1.0x | 1.0x |
| 2 | 32.5 | 0.6x | 2.0x |
| 4 | 34.0 | 0.3x | 4.0x |
| 8 | 34.5 | 0.15x | 8.0x |
| 16 | 34.6 | 0.08x | 16.0x |

**Diminishing returns:** beam_size > 4-8 usually doesn't help much.

---

## Practical Tips

### 1. Choose Beam Size

- **beam_size=1**: Greedy (fastest, lowest quality)
- **beam_size=4-5**: Good tradeoff (recommended for production)
- **beam_size=10+**: Diminishing returns, much slower

### 2. Length Penalty

- Start with `alpha=0.6` (from Wu et al., 2016)
- Tune on validation set if needed
- Higher α → favors longer sequences

### 3. Early Stopping

Stop when you have `beam_size` completed beams:
```python
if len(completed_beams) >= beam_size:
    break
```

### 4. Diversity

Standard beam search can produce similar beams. For diversity:
- **Diverse Beam Search**: Penalize similar beams
- **Sampling**: Add temperature-based sampling
- **Nucleus Sampling (top-p)**: Sample from top-p probability mass

---

## Complete Example

```python
# Load model and tokenizers
from src.inference.beam_search import beam_search
from src.data.tokenizer import SentencePieceTokenizer
from src.models.transformer.transformer import Transformer

model = load_trained_model('checkpoints/best_model.pt')
ko_tokenizer = SentencePieceTokenizer('data/vocab/ko_spm.model')
en_tokenizer = SentencePieceTokenizer('data/vocab/en_spm.model')

# Prepare source
src_text = "고양이가 자고 있어요"
src_ids = [ko_tokenizer.bos_id] + ko_tokenizer.encode_ids(src_text) + [ko_tokenizer.eos_id]
src = torch.tensor([src_ids])
src_mask = create_padding_mask(src, pad_idx=0)

# Run beam search
result = beam_search(
    model=model,
    src=src,
    src_mask=src_mask,
    beam_size=4,
    max_len=50,
    bos_idx=en_tokenizer.bos_id,
    eos_idx=en_tokenizer.eos_id,
    alpha=0.6,
    device='cpu'
)

# Decode
output_text = en_tokenizer.decode_ids(result.tokens[1:-1])  # Remove BOS/EOS
print(f"Translation: {output_text}")
print(f"Score: {result.score:.4f}")
print(f"Normalized Score: {result.get_normalized_score():.4f}")
```

---

## Summary

**Beam Search:**
- ✅ Better quality than greedy (typically +2-5 BLEU)
- ✅ Tractable (unlike exhaustive search)
- ⚠️ Slower than greedy (linear in beam_size)
- ⚠️ More memory (linear in beam_size)

**Key Parameters:**
- `beam_size`: 4-5 recommended
- `alpha`: 0.6 for length normalization
- `max_len`: Maximum sequence length

**Implementation Complexity:**
- Basic: Medium
- With KV cache: High (need to manage cache per beam)

Beam search is the **standard decoding method** for production NMT systems!
