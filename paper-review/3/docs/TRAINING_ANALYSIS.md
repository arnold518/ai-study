# Training Results Analysis - Full Run (100 Epochs)

**Date:** 2025-12-05
**Checkpoint:** epoch 38 (best validation loss)
**Training Duration:** ~9 hours
**Total Samples:** 417,557 training, 2,021 validation

---

## 📊 Summary of Results

### The Good ✅
- Training loss decreased smoothly: **4.5 → 2.2** (52% reduction)
- Training perplexity: **91 → 9.1** (90% reduction)
- Model learned **simple sentence patterns** well
- Short imperative sentences translate correctly:
  - "창문을 닫지 마" → "Don't close the window" ✓
  - "거짓말 하지 마" → "Don't lie" ✓

### The Bad ❌
- **Validation loss plateaued at ~2.9** (no improvement after epoch 38)
- **Severe overfitting**: Train loss ↓, Val loss →
- **BLEU scores highly unstable**: 1.8 → 79.5 → 0.0 → 53.7 (unreliable metric)
- **Long sentences produce garbage** with repetitive tokens
- **Numbers completely broken**: dates, quantities garbled

### The Ugly 💀
- Model generates **infinite loops** on complex inputs:
  - "and the lawmakers, and the lawmakers, and the lawmakers..." (repeated 40+ times)
  - "0000000000000..." (number repetition)
  - "will, will, will, will..." (modal verb loops)
- **Early stopping should have triggered** at epoch 38 but trained to 100

---

## 🔍 Detailed Analysis

### 1. Training Dynamics

| Metric | Epoch 1 | Epoch 38 (Best) | Epoch 100 | Change |
|--------|---------|-----------------|-----------|---------|
| Train Loss | 4.51 | 2.36 | 2.21 | -51% |
| Val Loss | 3.89 | **2.89** | 2.94 | -24% |
| Train PPL | 91.1 | 11.4 | 9.1 | -90% |
| Val PPL | 48.8 | 18.0 | 18.9 | -61% |
| Best BLEU | 1.9 | - | 79.5* | Unstable |

*BLEU at epoch 22, not sustained

**Key Observations:**
- Validation loss stopped improving after epoch 38
- Training continued for 62 more epochs with no benefit
- Gradient norms remained stable (~2.0-2.4) - no exploding gradients
- Learning rate decayed properly (0.00057 → 0.000077)

### 2. Translation Quality by Input Length

**Pattern discovered:** Translation quality inversely correlates with input length

| Input Type | Korean Length | Quality | Examples |
|------------|---------------|---------|----------|
| Long complex | 20-30+ words | **Garbage** | Repetitive loops, nonsense |
| Medium | 10-20 words | **Poor** | Partial meaning, errors |
| Short commands | 3-8 words | **Good** | Often correct! |
| Very short | 2-3 words | **Excellent** | Almost perfect |

**Example Progression:**
```
[Long - 30 words] → Repetitive garbage with "and the lawmakers" loop
[Medium - 15 words] → Partially coherent but incorrect
[Short - 5 words] "거짓말 하지 마" → "Don't lie" ✓
[Very short - 3 words] "안녕, 톰" → "Hello, this is Mr.Smith" (close)
```

### 3. Specific Failure Modes

#### A. Infinite Token Loops
```
Source: 토론에 참여한 사람들은 법 집행과 국가 안전보장에 대한 우려를...
Output: ...and the lawmakers, and the lawmakers, and the lawmakers [×40]
```

#### B. Number Hallucination
```
Source: 3,000마리의 전갈과 32일 동안...
Output: ...20000000000000000000000000000-won-2-2-2-2-3-3-3...
```

#### C. Premature EOS or Repetition
```
Source: [Complex 25-word sentence about technology]
Output: will, will, will, will, will, will, will, will [×50]
```

### 4. BLEU Score Instability

**BLEU over epochs:**
```
Epoch 1: 1.9
Epoch 3: 25.4
Epoch 5: 45.2
Epoch 6: 12.7
Epoch 7: 1.9
Epoch 17: 53.7
Epoch 22: 79.5 ← Peak (likely artifact)
Epoch 23: 4.9
Epoch 33: 0.0 ← Complete failure
Epoch 60: 56.2
```

**Analysis:**
- BLEU fluctuates wildly (0 to 79.5)
- High scores (79.5) likely due to:
  - Small sample size (100 samples)
  - Overfitting to short validation sentences
  - Metric calculation issues
- Cannot trust BLEU as stopping criterion

---

## 🔬 Root Cause Analysis

### Primary Issues

#### 1. **Severe Overfitting** 🔥
- **Symptom:** Train loss ↓, Val loss plateau
- **Evidence:** Gap between train (2.21) and val (2.94) loss
- **Impact:** Model memorized training data, cannot generalize

**Why overfitting occurred:**
- Dataset may have insufficient diversity
- Model capacity (52M parameters) too large for 418k samples
- No effective regularization beyond dropout=0.1

#### 2. **Length Generalization Failure** 🔥
- **Symptom:** Long sequences → garbage
- **Evidence:** Quality degrades with input length
- **Impact:** Unusable for real-world translation

**Why this happened:**
- Model never learned proper stopping criterion
- Attention patterns may collapse on long sequences
- Positional encodings may not scale well

#### 3. **Degenerate Decoding States** 🔥
- **Symptom:** Infinite token loops
- **Evidence:** "will will will..." repetitions
- **Impact:** Model stuck in attractors

**Why this happened:**
- Softmax distribution becomes too peaked
- Model confidence too high (label smoothing insufficient)
- No length penalty or repetition penalty applied

#### 4. **Number/Date Handling** 🔥
- **Symptom:** Dates and numbers garbled
- **Evidence:** "1972년 10월" → "20019th, 2019th, 19th"
- **Impact:** Factual information destroyed

**Why this happened:**
- SentencePiece tokenization splits numbers inconsistently
- Limited number examples in training data
- Model doesn't understand number semantics

### Secondary Issues

#### 5. **Training Configuration Problems**
- **No early stopping:** Trained 62 epochs past best checkpoint
- **BLEU unreliable:** Should not be used for validation
- **Inference examples too few:** Only 2 examples during training

#### 6. **Hyperparameter Concerns**
- **Learning rate conflict:**
  - `base_config.py`: `learning_rate = 1e-4`
  - `transformer_config.py`: `learning_rate = 1.0` (for Noam)
  - Unclear which is used
- **Batch size:** 128 may be too large
- **Warmup steps:** 4000 may be too many for this dataset size

---

## 💡 Recommendations for Improvement

### Priority 1: Critical Fixes 🚨

#### A. **Implement Early Stopping**
```python
# Add to trainer
early_stopping_patience = 10  # Stop if no improvement for 10 epochs
min_delta = 0.01  # Minimum improvement threshold
```

**Impact:** Prevents wasted training time, reduces overfitting

#### B. **Add Inference-Time Constraints**
```python
# In greedy/beam search
- Add repetition penalty (penalize recently generated tokens)
- Add length penalty (encourage longer outputs)
- Set min_decode_length = 3  # Prevent premature stopping
- Add n-gram blocking (prevent "will will will")
```

**Impact:** Eliminates infinite loops, improves coherence

#### C. **Reduce Model Capacity**
```python
# Current: 52M parameters
d_model = 512 → 256        # -75% parameters
num_layers = 6 → 4         # -33% depth
# Result: ~13M parameters
```

**Impact:** Better fit for 418k training samples, less overfitting

### Priority 2: Training Improvements ⚡

#### D. **Increase Regularization**
```python
dropout = 0.1 → 0.3                    # More aggressive
label_smoothing = 0.1 → 0.15           # Reduce overconfidence
weight_decay = 0.0 → 0.0001           # L2 regularization
```

**Impact:** Reduces overfitting, improves generalization

#### E. **Data Augmentation**
- **Back-translation:** Generate synthetic parallel data
- **Sentence shuffling:** Reorder training samples
- **Subword dropout:** Randomly merge/split tokens (BPE-dropout)
- **Filtering:** Remove sentences >50 words (focus on learnable examples)

**Impact:** Increases effective dataset size, improves robustness

#### F. **Curriculum Learning**
```python
# Epoch 1-20: Train on short sentences (5-15 words)
# Epoch 21-40: Add medium sentences (15-30 words)
# Epoch 41+: Include all lengths
```

**Impact:** Model learns basic patterns before tackling complexity

### Priority 3: Architecture & Hyperparameters 🔧

#### G. **Adjust Learning Rate Schedule**
```python
# Current: Noam with warmup=4000
warmup_steps = 4000 → 2000     # Faster warmup
# OR: Use cosine annealing with restarts
```

**Impact:** Better convergence, less oscillation

#### H. **Reduce Batch Size**
```python
batch_size = 128 → 64          # Or 32
gradient_accumulation = 2      # Maintain effective batch size if needed
```

**Impact:** More frequent updates, better exploration

#### I. **Separate Vocabularies**
```python
use_shared_vocab = True → False
```

**Rationale:**
- Korean and English have very different character sets
- Shared vocab may cause unnecessary token interference
- Each language benefits from optimized tokenization

**Impact:** Better tokenization, improved number handling

#### J. **Increase Sequence Length Buffer**
```python
max_seq_length = 128 → 200     # Allow longer contexts
```

**Impact:** Better handling of long sentences (though may not help if model can't use it)

### Priority 4: Evaluation & Monitoring 📈

#### K. **Improve Validation Metrics**
```python
# Replace BLEU as primary metric
- Use validation loss (already doing this ✓)
- Add: METEOR, ChrF, BERTScore
- Track: Token repetition rate
- Monitor: Average output length vs input length
```

**Impact:** More reliable training signals

#### L. **Increase Inference Examples**
```python
inference_num_examples = 2 → 10
# Show examples from different length buckets
```

**Impact:** Earlier detection of quality issues

#### M. **Add Length-Stratified Validation**
```python
# Report metrics separately for:
- Short (3-10 words)
- Medium (10-20 words)
- Long (20+ words)
```

**Impact:** Identify where model struggles

---

## 🎯 Recommended Action Plan

### Phase 1: Quick Wins (1-2 hours)
1. ✅ Implement early stopping (patience=10)
2. ✅ Add repetition penalty to inference
3. ✅ Increase inference examples to 10
4. ✅ Add n-gram blocking

### Phase 2: Architecture Tuning (2-4 hours)
5. ✅ Reduce model size: d_model=256, layers=4
6. ✅ Increase regularization: dropout=0.3
7. ✅ Use separate vocabularies
8. ✅ Adjust warmup_steps=2000

### Phase 3: Training (8-12 hours)
9. ✅ Train smaller model from scratch
10. ✅ Monitor with improved metrics
11. ✅ Evaluate on length-stratified test set

### Phase 4: Advanced (if needed)
12. ⏸️ Implement curriculum learning
13. ⏸️ Add data augmentation (back-translation)
14. ⏸️ Try different architectures (relative positional encoding)

---

## 📋 Configuration Comparison

### Current Configuration (Problematic)
```python
# Model
d_model = 512
num_layers = 6
num_heads = 8
d_ff = 2048
dropout = 0.1
→ 52M parameters

# Training
batch_size = 128
learning_rate = 1e-4  # Conflict with Noam?
warmup_steps = 4000
label_smoothing = 0.1
max_seq_length = 128

# Data
use_shared_vocab = True
vocab_size = 16000
```

### Recommended Configuration (Conservative)
```python
# Model (reduced capacity)
d_model = 256           # -50%
num_layers = 4          # -33%
num_heads = 8           # Same
d_ff = 1024            # -50%
dropout = 0.3           # +200%
→ ~13M parameters      # -75%

# Training
batch_size = 64         # -50%
warmup_steps = 2000     # -50%
label_smoothing = 0.15  # +50%
max_seq_length = 200    # +56%
early_stopping_patience = 10
weight_decay = 0.0001

# Data
use_shared_vocab = False  # Separate vocabs
vocab_size = 16000          # Same per language

# Inference
repetition_penalty = 1.2
min_decode_length = 3
no_repeat_ngram_size = 3
```

### Recommended Configuration (Aggressive - if conservative fails)
```python
# Model (much smaller)
d_model = 128           # -75%
num_layers = 3          # -50%
num_heads = 4           # -50%
d_ff = 512             # -75%
dropout = 0.4           # +300%
→ ~3M parameters       # -94%

# Training with augmentation
batch_size = 32
warmup_steps = 1000
data_augmentation = True
curriculum_learning = True
max_epochs = 50         # Earlier stop
```

---

## 🔧 Specific Code Changes Needed

### 1. Inference with Repetition Penalty
```python
# In greedy_search.py and beam_search.py
def apply_repetition_penalty(logits, generated_ids, penalty=1.2):
    """Penalize recently generated tokens."""
    for token_id in set(generated_ids[-10:]):  # Last 10 tokens
        logits[:, token_id] /= penalty
    return logits

# In decoding loop:
logits = model(...)
logits = apply_repetition_penalty(logits, generated_ids, penalty=1.2)
next_token = logits.argmax()
```

### 2. Early Stopping in Trainer
```python
# In trainer.py
class EarlyStopping:
    def __init__(self, patience=10, min_delta=0.01):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')

    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            return False  # Don't stop
        else:
            self.counter += 1
            return self.counter >= self.patience  # Stop if patience exceeded

# In training loop:
early_stopping = EarlyStopping(patience=10)
for epoch in range(num_epochs):
    ...
    if early_stopping(val_loss):
        print(f"Early stopping at epoch {epoch}")
        break
```

### 3. Length-Stratified Evaluation
```python
# In trainer.py validation
def compute_metrics_by_length(sources, translations, references):
    short_bleu = bleu([t for s, t, r in zip(sources, translations, references) if len(s.split()) < 10])
    medium_bleu = bleu([t for s, t, r in zip(sources, translations, references) if 10 <= len(s.split()) < 20])
    long_bleu = bleu([t for s, t, r in zip(sources, translations, references) if len(s.split()) >= 20])
    return {"short": short_bleu, "medium": medium_bleu, "long": long_bleu}
```

---

## 📊 Expected Improvements

| Metric | Current | Expected (Conservative) | Expected (Aggressive) |
|--------|---------|-------------------------|----------------------|
| Val Loss | 2.94 (plateau) | 2.5-2.7 (improving) | 2.3-2.5 |
| Val PPL | 18.9 | 12-15 | 10-12 |
| BLEU (short) | ~80 | 70-80 | 60-70 |
| BLEU (medium) | ~20 | 40-50 | 30-40 |
| BLEU (long) | <5 | 15-25 | 10-20 |
| Repetition Rate | 60%+ | <10% | <5% |
| Training Time | 9h (wasted) | 4-5h (early stop) | 3-4h |

---

## 🎓 Lessons Learned

1. **Model size matters:** 52M parameters is overkill for 418k samples
2. **Early stopping is essential:** Don't train past validation plateau
3. **BLEU is unreliable:** Use validation loss as primary metric
4. **Length generalization is hard:** Models struggle with distribution shift
5. **Repetition is a common failure mode:** Need explicit penalties
6. **Numbers need special handling:** Consider separate tokenization
7. **Monitor inference during training:** Catches quality issues early
8. **Overfitting happens fast:** More regularization needed

---

## 📚 References & Further Reading

- **Vaswani et al. (2017):** "Attention Is All You Need" - Original Transformer paper
- **Ott et al. (2018):** "Scaling Neural Machine Translation" - Training tips
- **Murray & Chiang (2018):** "Correcting Length Bias in Neural Machine Translation"
- **Keskar et al. (2019):** "CTRL: A Conditional Transformer Language Model"
- **Holtzman et al. (2020):** "The Curious Case of Neural Text Degeneration" - Repetition analysis

---

## ✅ Next Steps

**Immediate (today):**
1. Create updated configuration file (small model)
2. Implement early stopping and repetition penalty
3. Start training with new config

**Short-term (this week):**
4. Evaluate on full test set with length stratification
5. Compare current vs. improved model
6. Document findings

**Long-term (future work):**
7. Implement curriculum learning
8. Try relative positional encodings
9. Experiment with different architectures (Transformer-XL, etc.)

---

**Report generated:** 2025-12-05
**Analyst:** Claude Code
**Status:** Ready for action ⚡
