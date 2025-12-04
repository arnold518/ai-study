# Config-Based Evaluation Parameters

**Date**: 2025-12-05
**Change**: Moved BLEU and inference example parameters to config file

## Summary

Previously, the number of BLEU samples and inference examples were hardcoded in the training loop. These values are now configurable through the config file, making it easier to adjust evaluation behavior without modifying code.

## Changes Made

### 1. Added Config Parameters (`config/base_config.py:42-44`)

```python
# Evaluation
bleu_num_samples = 100  # Number of validation samples for BLEU computation
inference_num_examples = 2  # Number of inference examples to display
```

**Location**: After "Checkpointing" section, before "Device" section

**Rationale**:
- Makes evaluation behavior configurable
- Easy to adjust for different dataset sizes
- Can use smaller values for quick testing, larger for production

### 2. Updated Trainer Methods (`src/training/trainer.py`)

#### `compute_bleu_score()` - Lines 145-164

**Before**:
```python
def compute_bleu_score(self, num_samples=100):
    # ...
    num_samples = min(num_samples, len(self.val_dataset))
```

**After**:
```python
def compute_bleu_score(self, num_samples=None):
    """
    Args:
        num_samples: Number of samples to use for BLEU computation.
                    If None, uses config.bleu_num_samples
    """
    if num_samples is None:
        num_samples = self.config.bleu_num_samples
    num_samples = min(num_samples, len(self.val_dataset))
```

**Benefits**:
- Defaults to config value
- Can still override for testing (pass explicit value)
- Flexible and backward compatible

#### `generate_inference_examples()` - Lines 194-213

**Before**:
```python
def generate_inference_examples(self, num_examples=3):
    # ...
    num_examples = min(num_examples, len(self.val_dataset))
```

**After**:
```python
def generate_inference_examples(self, num_examples=None):
    """
    Args:
        num_examples: Number of examples to generate.
                     If None, uses config.inference_num_examples
    """
    if num_examples is None:
        num_examples = self.config.inference_num_examples
    num_examples = min(num_examples, len(self.val_dataset))
```

### 3. Updated Training Loop Calls (`src/training/trainer.py`)

#### During Validation (Line 258, 264)

**Before**:
```python
bleu_score = self.compute_bleu_score(num_samples=100)
# ...
examples = self.generate_inference_examples(num_examples=2)
```

**After**:
```python
bleu_score = self.compute_bleu_score()  # Uses config.bleu_num_samples
# ...
examples = self.generate_inference_examples()  # Uses config.inference_num_examples
```

#### During Periodic Checkpoints (Line 321)

**Before**:
```python
examples = self.generate_inference_examples(num_examples=2)
```

**After**:
```python
examples = self.generate_inference_examples()  # Uses config.inference_num_examples
```

### 4. Updated Test File (`tests/test_phase5_integration.py`)

**Added config values** (Lines 83-84):
```python
config.bleu_num_samples = 10  # Small number for testing
config.inference_num_examples = 2  # Small number for testing
```

**Updated test calls** (Lines 140, 153):
```python
# Before:
bleu_score = trainer.compute_bleu_score(num_samples=10)
examples = trainer.generate_inference_examples(num_examples=2)

# After:
bleu_score = trainer.compute_bleu_score()  # Uses config value
examples = trainer.generate_inference_examples()  # Uses config value
```

## Usage

### Default Behavior (Training)

When running training with default config:

```bash
/home/arnold/venv/bin/python scripts/train.py
```

Uses:
- `bleu_num_samples = 100` (from `base_config.py`)
- `inference_num_examples = 2` (from `base_config.py`)

### Custom Config

Create custom config file:

```python
# config/my_config.py
from config.transformer_config import TransformerConfig

class MyConfig(TransformerConfig):
    # Use more samples for more accurate BLEU
    bleu_num_samples = 500

    # Show more examples
    inference_num_examples = 5
```

Run training:
```bash
/home/arnold/venv/bin/python scripts/train.py --config config/my_config.py
```

### Programmatic Override

You can still override config values programmatically:

```python
# In scripts/train.py or custom script
config = TransformerConfig()
config.bleu_num_samples = 200  # Override default
config.inference_num_examples = 3  # Override default

trainer = Trainer(..., config=config)
```

### Method-Level Override

For one-off evaluation, can still pass explicit values:

```python
# In custom evaluation script
bleu_score = trainer.compute_bleu_score(num_samples=1000)  # Use all samples
examples = trainer.generate_inference_examples(num_examples=10)  # More examples
```

## Recommended Values

### For Different Scenarios

| Scenario | `bleu_num_samples` | `inference_num_examples` | Rationale |
|----------|-------------------|-------------------------|-----------|
| **Quick Testing** | 10-20 | 1-2 | Fast feedback |
| **Regular Training** | 100 | 2-3 | Balance speed/accuracy |
| **Large Dataset** | 200-500 | 3-5 | More representative |
| **Final Evaluation** | All (1000+) | 5-10 | Most accurate |
| **Debug Mode** | 5 | 1 | Minimal overhead |

### Adjusting for Dataset Size

```python
# Small validation set (< 100 samples)
config.bleu_num_samples = len(val_dataset)  # Use all
config.inference_num_examples = 3

# Medium validation set (100-1000 samples)
config.bleu_num_samples = 100
config.inference_num_examples = 2

# Large validation set (> 1000 samples)
config.bleu_num_samples = 500
config.inference_num_examples = 5
```

## Benefits of Config-Based Approach

1. **Centralized Configuration**: All training parameters in one place
2. **No Code Changes**: Adjust behavior without modifying trainer.py
3. **Version Control**: Track evaluation settings with git
4. **Experimentation**: Easy A/B testing with different configs
5. **Documentation**: Config file serves as documentation
6. **Flexibility**: Can still override when needed

## Example: Different Evaluation Strategies

### Strategy 1: Fast Iteration
```python
# config/fast_config.py
class FastConfig(TransformerConfig):
    bleu_num_samples = 20  # Quick BLEU estimate
    inference_num_examples = 1  # Minimal examples
    eval_every = 5  # Evaluate less frequently
```

### Strategy 2: Accurate Monitoring
```python
# config/accurate_config.py
class AccurateConfig(TransformerConfig):
    bleu_num_samples = 500  # More accurate BLEU
    inference_num_examples = 5  # More examples for inspection
    eval_every = 1  # Evaluate every epoch
```

### Strategy 3: Production Training
```python
# config/production_config.py
class ProductionConfig(TransformerConfig):
    bleu_num_samples = 1000  # Maximum accuracy
    inference_num_examples = 10  # Comprehensive examples
    eval_every = 1  # Monitor closely
    save_every = 1  # Save frequently
```

## Backward Compatibility

The changes are **fully backward compatible**:

- If you pass explicit values, they override config
- If you pass `None`, uses config value
- If you pass nothing, uses config value

```python
# All of these work:
trainer.compute_bleu_score()  # Uses config
trainer.compute_bleu_score(num_samples=50)  # Overrides config
trainer.compute_bleu_score(num_samples=None)  # Uses config
```

## Testing

Run the integration test to verify config-based evaluation:

```bash
/home/arnold/venv/bin/python tests/test_phase5_integration.py
```

Expected output:
```
Testing BLEU computation (using config: 10 samples)...
✓ BLEU computation successful: 0.00

Testing inference example generation (using config: 2 examples)...
✓ Generated 2 inference examples
```

## Summary of Files Changed

| File | Changes | Lines |
|------|---------|-------|
| `config/base_config.py` | Added `bleu_num_samples` and `inference_num_examples` | 42-44 |
| `src/training/trainer.py` | Updated `compute_bleu_score()` to use config | 145-164 |
| `src/training/trainer.py` | Updated `generate_inference_examples()` to use config | 194-213 |
| `src/training/trainer.py` | Removed hardcoded values in training loop | 258, 264, 321 |
| `tests/test_phase5_integration.py` | Set config values for testing | 83-84 |
| `tests/test_phase5_integration.py` | Use config defaults in test calls | 140, 153 |

## Next Steps

Consider adding more configurable evaluation parameters:

- **BLEU decoding method**: `bleu_decode_method = 'greedy'` or `'beam'`
- **Beam search settings**: `bleu_beam_size = 4`, `bleu_length_penalty = 0.6`
- **Example selection**: `example_selection = 'random'` or `'worst_bleu'`
- **Metrics**: `compute_meteor = False`, `compute_chrf = False`

These can all follow the same pattern of config-based parameters with method-level overrides.
