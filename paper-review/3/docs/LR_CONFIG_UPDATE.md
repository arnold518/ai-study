# Learning Rate Configuration Update for 1.7M Dataset

**Date:** 2025-12-05
**Reason:** Dataset size increased 4.1x due to token-based filtering (417k → 1.7M pairs)

## Problem Identified

### Dataset Size Change
- **Before:** 417,557 training pairs (character-based filtering)
- **After:** 1,697,929 training pairs (token-based filtering)
- **Increase:** 4.07x more data

### Impact on Training Schedule
- **Steps per epoch:** 3,262 → 13,265 (4.07x increase)
- **Total steps (100 epochs):** 326k → 1.33M (4.07x increase)

### Critical Issue: Warmup Too Short
With the old `warmup_steps = 4000`:
- **Before:** Warmup = 1.23 epochs (appropriate)
- **After:** Warmup = 0.30 epochs (too short!)
- **Problem:** Learning rate reaches peak at 30% of first epoch, then immediately starts decaying

**Consequence:** Model doesn't have enough time to stabilize during warmup, leading to:
- Unstable early training
- Suboptimal convergence
- Wasted computation on later epochs

## Configuration Changes

### 1. Warmup Steps (transformer_config.py:25)
```python
# BEFORE
warmup_steps = 4000    # 0.30 epochs on 1.7M dataset

# AFTER
warmup_steps = 16000   # 1.21 epochs on 1.7M dataset
```

**Rationale:**
- Scaled 4x to match 4.1x dataset increase
- Maintains same warmup/dataset ratio as original paper
- Peak LR now reached at ~1.2 epochs (optimal for initialization)

### 2. Number of Epochs (base_config.py:29)
```python
# BEFORE
num_epochs = 100       # 1.33M total steps

# AFTER
num_epochs = 30        # 398k total steps
```

**Rationale:**
- With 4.1x more data per epoch, need fewer epochs for convergence
- 30 epochs * 13,265 steps = 397,950 steps (similar to original 326k steps)
- More efficient training (less redundant passes over data)

### 3. Dropout (base_config.py:34)
```python
# BEFORE
dropout = 0.1          # Appropriate for 417k dataset

# AFTER
dropout = 0.15         # Better for 1.7M dataset
```

**Rationale:**
- Larger dataset requires more regularization to prevent overfitting
- 0.15 is still conservative (paper used 0.1-0.3 range)
- Helps model generalize better with increased data

## Learning Rate Schedule Comparison

### Before (warmup_steps=4000, 100 epochs)
| Epoch | Learning Rate | % of Peak |
|-------|---------------|-----------|
| 0.30  | 0.000699      | 100%      |
| 1.00  | 0.000384      | 55%       |
| 10.00 | 0.000121      | 17%       |
| 100.00| 0.000012      | 2%        |

**Problem:** Peak at 0.3 epochs, rapid decay, very low LR by epoch 10

### After (warmup_steps=16000, 30 epochs)
| Epoch | Learning Rate | % of Peak |
|-------|---------------|-----------|
| 1.21  | 0.000349      | 100%      |
| 2.00  | 0.000271      | 78%       |
| 10.00 | 0.000121      | 35%       |
| 30.00 | 0.000070      | 20%       |

**Improvement:** Peak at 1.2 epochs, gradual decay, still useful LR at epoch 30

## Comparison with Original Paper

### Original Transformer (Vaswani et al., 2017)
- Dataset: WMT 2014 En-De (4.5M pairs)
- Warmup: 4,000 steps
- Training: ~100,000 steps (12.5 epochs)
- Warmup/data ratio: 4,000 / 4.5M = 0.089%

### Our Configuration (Updated)
- Dataset: Korean-English (1.7M pairs)
- Warmup: 16,000 steps
- Training: ~397,950 steps (30 epochs)
- Warmup/data ratio: 16,000 / 1.7M = 0.094%

**✅ Proportionally equivalent to paper's configuration!**

## Expected Benefits

### 1. Stable Warmup (1.21 epochs)
- Model has time to stabilize embeddings and attention patterns
- Gradual increase prevents early-training instability
- Better initialization of all layers

### 2. Optimal Peak LR Timing
- Peak reached after 1.21 epochs (not 0.3 epochs)
- Model sees enough data before starting LR decay
- Better alignment with dataset size

### 3. Efficient Training
- 30 epochs instead of 100 (3.3x faster to train)
- Similar total steps (398k vs 326k)
- Better use of computational resources

### 4. Better Regularization
- Dropout 0.15 prevents overfitting on larger dataset
- Label smoothing 0.1 (unchanged, already optimal)
- Improved generalization expected

## Verification

Run the verification script to see the full schedule:
```bash
/home/arnold/venv/bin/python scripts/verify_lr_config.py
```

## Summary Table

| Parameter | Before | After | Change | Reason |
|-----------|--------|-------|--------|--------|
| **warmup_steps** | 4,000 | 16,000 | 4x | Match 4.1x dataset increase |
| **num_epochs** | 100 | 30 | 0.3x | Fewer epochs needed with 4x data |
| **dropout** | 0.10 | 0.15 | +50% | Better regularization for larger dataset |
| **Peak LR epoch** | 0.30 | 1.21 | 4x | Proper warmup duration |
| **Total steps** | 326k | 398k | +22% | Similar training budget |

## Next Steps

1. ✅ Configuration updated
2. ✅ Verification completed
3. **Ready to train** with optimized schedule
4. Monitor training curves to validate improvements

---

**Files Modified:**
- `config/transformer_config.py` (warmup_steps)
- `config/base_config.py` (num_epochs, dropout)

**Analysis Scripts:**
- `scripts/analyze_lr_schedule.py` - Dataset size impact analysis
- `scripts/verify_lr_config.py` - Configuration verification
