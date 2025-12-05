# Learning Rate Algorithm Review

**Date:** 2025-12-05
**Reviewer:** Claude Code
**Status:** ✅ **ALGORITHM WORKING CORRECTLY**

---

## Executive Summary

✅ **The Noam learning rate scheduler is implemented correctly and working as designed.**

- Formula implementation matches the paper exactly
- Actual logged values match expected values to machine precision
- No bugs found in the optimizer code
- Configuration is using the correct parameters

**However:** There is a **configuration issue** that may be contributing to poor training results.

---

## 1. Configuration Trace

### 1.1 Config Files

**base_config.py** (line 31):
```python
learning_rate = 1e-4  # ⚠ NOT USED - Overridden by TransformerConfig
```

**transformer_config.py** (lines 24-28):
```python
learning_rate = 1.0    # ✅ USED as scaling factor
warmup_steps = 4000    # ✅ USED
adam_beta1 = 0.9       # ✅ USED
adam_beta2 = 0.98      # ✅ USED
adam_eps = 1e-9        # ✅ USED
```

**Resolution:** TransformerConfig inherits from BaseConfig but **overrides** `learning_rate`. The value `1.0` is used as the `factor` parameter in the Noam scheduler.

### 1.2 Optimizer Initialization

**scripts/train.py** (lines 156-161):
```python
optimizer = NoamOptimizer(
    model.parameters(),
    config.d_model,        # 512
    config.warmup_steps,   # 4000
    factor=config.learning_rate  # 1.0 (from TransformerConfig)
)
```

**Result:** Correctly passes all parameters to NoamOptimizer.

### 1.3 NoamOptimizer Implementation

**src/training/optimizer.py** (lines 34-43):
```python
def _get_lr(self):
    """Calculate learning rate for current step.

    Implements Noam learning rate schedule from the paper:
    lrate = d_model^(-0.5) * min(step^(-0.5), step * warmup^(-1.5))
    """
    return self.factor * (
        self.d_model ** (-0.5) *
        min(self.step_num ** (-0.5), self.step_num * self.warmup_steps ** (-1.5))
    )
```

**Result:** ✅ Formula is **correct** and matches the paper exactly.

**Paper's formula:**
```
lrate = d_model^(-0.5) * min(step^(-0.5), step * warmup^(-1.5))
```

**Our implementation:**
```python
factor * (d_model^(-0.5) * min(step^(-0.5), step * warmup^(-1.5)))
```

Where `factor = 1.0`, so they are equivalent.

---

## 2. Verification Against Training Log

### 2.1 Theoretical vs Actual Values

| Step   | Epoch | Expected LR          | Actual LR (Logged)   | Difference |
|--------|-------|----------------------|----------------------|------------|
| 3,263  | 1     | 0.000570022641452    | 0.000570022641452    | 0.0 ✓      |
| 6,526  | 2     | 0.000547068217585    | 0.000547068217585    | 0.0 ✓      |
| 13,052 | 4     | 0.000386835646426    | 0.000386835646426    | 0.0 ✓      |
| 32,630 | 10    | 0.000244656344570    | 0.000244656344570    | 0.0 ✓      |
| 65,260 | 20    | 0.000172998160306    | 0.000172998160306    | 0.0 ✓      |
| 97,890 | 30    | 0.000141252406396    | 0.000141252406396    | 0.0 ✓      |

**Conclusion:** Perfect match to machine precision. The algorithm is working correctly.

### 2.2 Learning Rate Evolution

```
Step 1:       0.00000017  (warmup start - very low)
Step 500:     0.00008735  (warmup - 12.5% through)
Step 1000:    0.00017469  (warmup - 25% through)
Step 2000:    0.00034939  (warmup - 50% through)
Step 3000:    0.00052408  (warmup - 75% through)
Step 3263:    0.00057002  (end of epoch 1 - 81.6% through warmup)
Step 4000:    0.00069877  (PEAK - warmup complete)
Step 5000:    0.00062500  (decay phase)
Step 8000:    0.00049411  (continued decay)
...
Step 326300:  0.00007737  (epoch 100 - 11% of peak)
```

**Observations:**
- LR increases linearly during warmup (steps 1-4000)
- Peak LR = 0.000699 at step 4000
- LR decreases as `step^(-0.5)` after warmup
- By epoch 100, LR has decayed to 11% of peak

---

## 3. Warmup Analysis

### 3.1 Dataset Context

- **Training samples:** 417,557
- **Batch size:** 128
- **Batches per epoch:** 417,557 ÷ 128 = **3,262**
- **Warmup steps:** 4,000

### 3.2 Warmup Duration

```
Warmup completes at step 4,000
Epoch 1 ends at step 3,263
Epoch 2 ends at step 6,526

→ Warmup completes at epoch 1.23
→ Peak LR reached during epoch 2
```

**Timeline:**
- **Epoch 1:** LR goes from 0.00017 → 0.00057 (increasing)
- **Epoch 2:** LR reaches peak 0.00070, then starts decay
- **Epochs 3-100:** LR continuously decays

### 3.3 Is This Appropriate?

**Paper's setting:**
- d_model = 512, warmup_steps = 4000
- Dataset: WMT 2014 English-German (4.5M sentence pairs)
- Batch size: ~25,000 tokens (variable batch size)

**Our setting:**
- d_model = 512, warmup_steps = 4000 ✓ (same)
- Dataset: Korean-English (418k sentence pairs) ← **10x smaller**
- Batch size: 128 sentences ← Different batching strategy

**Analysis:**

The paper's 4000 warmup steps were designed for:
- Much larger dataset (4.5M vs 418k = 10x larger)
- Different batch size scheme

For our smaller dataset:
- 4000 steps = 1.23 epochs
- Most papers recommend warmup of **0.5-1.0 epochs** for smaller datasets
- Our 1.23 epochs is **slightly high but acceptable**

**Recommendation:** Could reduce to 2000 steps (0.6 epochs) but current setting is not harmful.

---

## 4. Potential Issues Found

### 4.1 ⚠️ Issue #1: Learning Rate Too Low

**Current peak LR:** 0.000699 ≈ **7e-4**

**Comparison with common practices:**

| Model Type | Typical Peak LR | Our LR | Ratio |
|------------|-----------------|--------|-------|
| Transformer (base) | 1e-3 to 5e-4 | 7e-4 | ✓ Reasonable |
| Transformer (large) | 3e-4 | 7e-4 | 2.3x higher |
| Our model (512d, 6 layers) | - | 7e-4 | ? |

**Analysis:**

The peak LR of **0.000699** is actually **reasonable** for a base Transformer:
- Paper uses: `d_model^(-0.5) = 512^(-0.5) = 0.0442`
- With warmup peak: `0.0442 * (4000^(-0.5)) = 0.000699`
- This matches standard practice

**However**, we can adjust via the `factor` parameter:
- Current: `factor = 1.0` → peak LR = 0.000699
- Could try: `factor = 2.0` → peak LR = 0.001398
- Could try: `factor = 0.5` → peak LR = 0.000350

### 4.2 ⚠️ Issue #2: Learning Rate Decays Too Quickly

**LR Decay Over Training:**

| Epoch | Step    | LR       | % of Peak |
|-------|---------|----------|-----------|
| 2     | 6,526   | 0.000547 | 78%       |
| 10    | 32,630  | 0.000245 | 35%       |
| 20    | 65,260  | 0.000173 | 25%       |
| 50    | 163,150 | 0.000109 | 16%       |
| 100   | 326,300 | 0.000077 | 11%       |

**Analysis:**

By epoch 10, LR has already decayed to **35% of peak**.
By epoch 20, LR is at **25% of peak**.

For a 100-epoch training run, the LR decays according to:
```
LR ∝ 1 / √step
```

This means:
- **First 10 epochs:** LR drops from 100% → 35% (steep decay)
- **Last 90 epochs:** LR drops from 35% → 11% (slower decay)

**Problem:** Most of the training happens with a very low learning rate, which may explain:
1. Slow convergence
2. Getting stuck in local minima
3. Inability to escape poor solutions

### 4.3 ⚠️ Issue #3: No Learning Rate Restarts

**Current schedule:** Monotonic decay after warmup

**Problem:** Once LR decays, it never increases again. If the model gets stuck in a bad region of the loss landscape, it cannot escape.

**Alternative approaches:**
1. **Cosine annealing with restarts**
2. **Step decay** (drop by factor every N epochs)
3. **Cyclic learning rates**

---

## 5. Code Quality Assessment

### 5.1 Implementation Correctness

✅ **NoamOptimizer:** Perfect implementation
✅ **Trainer integration:** Correct usage
✅ **Configuration:** Clear and well-documented
✅ **Logging:** LR properly logged to CSV

### 5.2 Potential Improvements

#### A. Add Learning Rate Bounds
```python
def _get_lr(self):
    lr = self.factor * (
        self.d_model ** (-0.5) *
        min(self.step_num ** (-0.5), self.step_num * self.warmup_steps ** (-1.5))
    )
    # Optional: Add min/max bounds
    # lr = max(lr, 1e-6)  # Prevent lr from going too low
    # lr = min(lr, 1e-3)  # Prevent lr from going too high
    return lr
```

#### B. Add LR Scheduler State Saving
```python
def state_dict(self):
    """Return state dict for checkpointing."""
    return {
        'step_num': self.step_num,
        'd_model': self.d_model,
        'warmup_steps': self.warmup_steps,
        'factor': self.factor,
        'optimizer': self.optimizer.state_dict()
    }

def load_state_dict(self, state_dict):
    """Load state from checkpoint."""
    self.step_num = state_dict['step_num']
    self.optimizer.load_state_dict(state_dict['optimizer'])
```

#### C. Add LR Visualization Support
```python
def get_lr_schedule(self, max_steps):
    """Generate full LR schedule for plotting."""
    return [self._get_lr_for_step(step) for step in range(1, max_steps+1)]

def _get_lr_for_step(self, step):
    """Get LR for a specific step (without updating step_num)."""
    return self.factor * (
        self.d_model ** (-0.5) *
        min(step ** (-0.5), step * self.warmup_steps ** (-1.5))
    )
```

---

## 6. Comparison with Failed Training

### 6.1 Training Results Review

From the training log analysis:
- **Validation loss plateaued at epoch 38** (loss ~2.89)
- **Training loss continued decreasing** (→ overfitting)
- **Translation quality poor** (repetitive outputs)

### 6.2 Could Learning Rate Be the Cause?

**At epoch 38:**
- Step = 123,994
- Learning rate = 0.000126 (18% of peak)
- Already in deep decay phase

**Analysis:**

By the time validation loss plateaued (epoch 38), the learning rate had already decayed to only **18% of its peak value**. This very low LR means:

1. **Limited ability to escape local minima:** Model stuck in suboptimal solution
2. **Slow fine-tuning:** Updates too small to fix issues
3. **No exploration:** Cannot try different solutions

**However, the root cause is NOT the LR schedule itself, but rather:**
- Model capacity too large (52M params for 418k samples)
- Insufficient regularization (dropout=0.1)
- Data/model mismatch (overfitting to short sentences)

**The LR schedule is working correctly**, but it cannot compensate for these fundamental issues.

---

## 7. Recommendations

### Priority 1: Keep Current LR Schedule ✅

**Recommendation:** **Do NOT change the learning rate schedule** - it's working correctly.

**Rationale:**
- Implementation is correct
- Formula matches the paper
- Values are appropriate for base Transformer
- Not the root cause of poor results

### Priority 2: Consider These Alternatives (If Needed)

If you want to experiment with different LR schedules **after fixing the primary issues** (model size, regularization):

#### Option A: Reduce Warmup Duration
```python
warmup_steps = 4000 → 2000  # 0.6 epochs instead of 1.23
```
**Effect:** Reach peak LR faster, spend more time at higher LR

#### Option B: Increase LR Factor
```python
learning_rate = 1.0 → 1.5  # or 2.0
```
**Effect:** Higher peak LR (0.00070 → 0.00105 or 0.00140)

#### Option C: Use Cosine Decay Instead of Inverse Square Root
```python
# In optimizer.py
def _get_lr(self):
    if self.step_num < self.warmup_steps:
        # Linear warmup
        return self.factor * self.step_num / self.warmup_steps * (self.d_model ** -0.5)
    else:
        # Cosine decay
        progress = (self.step_num - self.warmup_steps) / (self.max_steps - self.warmup_steps)
        return self.factor * (self.d_model ** -0.5) * 0.5 * (1 + math.cos(math.pi * progress))
```
**Effect:** Slower decay, maintains higher LR for longer

#### Option D: Add Learning Rate Restarts
```python
# Restart every N epochs
if self.step_num % restart_interval == 0:
    self.step_num = 1  # Reset to beginning of warmup
```
**Effect:** Periodically boost LR to escape local minima

---

## 8. Conclusion

### Summary

✅ **The learning rate algorithm is implemented correctly and working as designed.**

**What's working:**
- Formula implementation: ✅ Perfect
- Integration with trainer: ✅ Correct
- Configuration: ✅ Reasonable
- Logging: ✅ Accurate

**What's NOT the problem:**
- ❌ LR schedule is not causing poor translations
- ❌ LR schedule is not causing overfitting
- ❌ LR schedule is not causing repetitive outputs

**What IS the problem:**
- ⚠️ Model too large (52M params for 418k samples)
- ⚠️ Insufficient regularization (dropout=0.1 too low)
- ⚠️ No early stopping (trained 62 epochs past best checkpoint)
- ⚠️ Inference lacks repetition penalty

### Final Recommendation

**Do NOT modify the learning rate schedule.** Focus on:

1. **Reduce model size** (d_model=256, layers=4)
2. **Increase regularization** (dropout=0.3)
3. **Add early stopping** (patience=10)
4. **Add inference constraints** (repetition penalty, n-gram blocking)

The learning rate schedule is a **correctly implemented component** in an otherwise problematic training configuration.

---

## Appendix: LR Schedule Visualization

### Warmup Phase (Steps 1-4000)
```
0.0007 |                                                    ╱
       |                                              ╱╱╱╱╱╱
0.0006 |                                        ╱╱╱╱╱╱
       |                                  ╱╱╱╱╱╱
0.0005 |                            ╱╱╱╱╱╱
       |                      ╱╱╱╱╱╱
0.0004 |                ╱╱╱╱╱╱
       |          ╱╱╱╱╱╱
0.0003 |    ╱╱╱╱╱╱
       |╱╱╱╱
0.0000 +----+----+----+----+----+----+----+----+----+----+
       0    500  1000 1500 2000 2500 3000 3500 4000
                         Steps
```

### Decay Phase (Steps 4000-100000)
```
0.0007 |╲
       | ╲
0.0006 |  ╲
       |   ╲___
0.0005 |       ╲___
       |           ╲____
0.0004 |                ╲_____
       |                      ╲______
0.0003 |                             ╲________
       |                                      ╲________
0.0002 |                                               ╲__________
       |                                                          ╲
0.0001 +----+----+----+----+----+----+----+----+----+----+----+----+
       0    20k  40k  60k  80k  100k
                    Steps

Legend: LR ∝ 1/√step (inverse square root decay)
```

---

**Report generated:** 2025-12-05
**Status:** Review complete ✅
**Action:** No changes needed to LR schedule
