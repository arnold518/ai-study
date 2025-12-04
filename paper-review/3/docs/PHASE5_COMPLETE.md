# Phase 5 Complete: BLEU Integration and Inference Examples

**Date**: 2025-12-05
**Status**: ✅ Complete

## Overview

Phase 5 integrates BLEU score computation and inference examples into the training pipeline, allowing real-time monitoring of translation quality during training.

## Implementation Details

### 1. Modified `src/training/trainer.py`

#### New Parameters
The Trainer now accepts optional parameters for BLEU computation:

```python
def __init__(self, model, train_loader, val_loader, optimizer, criterion, config,
             src_tokenizer=None, tgt_tokenizer=None, val_dataset=None):
```

- `src_tokenizer`: Source language tokenizer (Korean)
- `tgt_tokenizer`: Target language tokenizer (English)
- `val_dataset`: Validation dataset for accessing original text

#### New Methods

**`compute_bleu_score(num_samples=100)`**
- Computes BLEU score on a random subset of validation data
- Uses greedy decoding for speed
- Returns corpus-level BLEU score using sacrebleu
- Progress shown with tqdm

```python
def compute_bleu_score(self, num_samples=100):
    """Compute BLEU score on a subset of validation data."""
    if not self.translator or not self.val_dataset:
        return None

    self.model.eval()
    num_samples = min(num_samples, len(self.val_dataset))
    indices = torch.randperm(len(self.val_dataset))[:num_samples]

    predictions = []
    references = []

    with torch.no_grad():
        for idx in tqdm(indices, desc="  BLEU", leave=False):
            src_text = self.val_dataset.src_lines[idx].strip()
            tgt_text = self.val_dataset.tgt_lines[idx].strip()

            pred_text = self.translator.translate(src_text, method='greedy')
            predictions.append(pred_text)
            references.append(tgt_text)

    bleu = compute_bleu(predictions, references)
    return bleu.score
```

**`generate_inference_examples(num_examples=3)`**
- Generates translation examples from validation set
- Returns list of (source, reference, prediction) tuples
- Used to visually inspect translation quality

```python
def generate_inference_examples(self, num_examples=3):
    """Generate inference examples from validation set."""
    if not self.translator or not self.val_dataset:
        return []

    self.model.eval()
    num_examples = min(num_examples, len(self.val_dataset))
    indices = torch.randperm(len(self.val_dataset))[:num_examples]

    examples = []

    with torch.no_grad():
        for idx in indices:
            src_text = self.val_dataset.src_lines[idx].strip()
            tgt_text = self.val_dataset.tgt_lines[idx].strip()
            pred_text = self.translator.translate(src_text, method='greedy')
            examples.append((src_text, tgt_text, pred_text))

    return examples
```

#### Updated Training Loop

The `train()` method now:

1. **Computes BLEU during validation epochs**:
   ```python
   if (epoch + 1) % self.config.eval_every == 0:
       val_loss = self.validate()

       # Compute BLEU score
       bleu_score = None
       if self.translator:
           bleu_score = self.compute_bleu_score(num_samples=100)
           if bleu_score is not None:
               print(f"  BLEU Score: {bleu_score:.2f}")
   ```

2. **Displays inference examples**:
   ```python
   # Generate inference examples
   if self.translator:
       examples = self.generate_inference_examples(num_examples=2)
       if examples:
           print(f"\n  Translation Examples:")
           for i, (src, ref, pred) in enumerate(examples, 1):
               print(f"    [{i}] Source:     {src}")
               print(f"        Reference:  {ref}")
               print(f"        Prediction: {pred}")
   ```

3. **Tracks two best models**:
   - `best_model.pt`: Best validation loss
   - `best_bleu_model.pt`: Best BLEU score

   ```python
   # Save best loss model
   if val_loss < self.best_val_loss:
       self.best_val_loss = val_loss
       save_checkpoint(..., 'best_model.pt')

   # Save best BLEU model
   if bleu_score is not None and bleu_score > self.best_bleu:
       self.best_bleu = bleu_score
       save_checkpoint(..., 'best_bleu_model.pt')
   ```

4. **Shows examples at periodic checkpoints**:
   ```python
   if (epoch + 1) % self.config.save_every == 0:
       checkpoint_path = ...
       save_checkpoint(...)

       # Show inference examples for periodic checkpoints too
       if self.translator:
           examples = self.generate_inference_examples(num_examples=2)
           # ... display examples
   ```

### 2. Modified `scripts/train.py`

Updated to pass tokenizers and validation dataset to Trainer:

```python
# Keep reference to full validation dataset for BLEU computation
full_val_dataset = val_dataset

# Use small subset if requested
if args.small:
    train_dataset = Subset(train_dataset, range(min(1000, len(train_dataset))))
    val_dataset = Subset(val_dataset, range(min(100, len(val_dataset))))

# Create trainer with tokenizers and dataset for BLEU
trainer = Trainer(
    model, train_loader, val_loader, optimizer, criterion, config,
    src_tokenizer=ko_tokenizer,
    tgt_tokenizer=en_tokenizer,
    val_dataset=full_val_dataset  # Use full dataset for BLEU/examples
)
```

**Important**: Use `full_val_dataset` (not the subset) so BLEU computation uses representative samples even when `--small` flag is used.

### 3. Created Test

**`tests/test_phase5_integration.py`**
- Tests Trainer instantiation with tokenizers
- Tests BLEU computation
- Tests inference example generation
- Verifies all components work together

## Usage

### During Training

When running training, you'll now see:

```
Epoch 1/10
  Train Loss: 8.5432 | Train PPL: 5100.23
  Val Loss:   7.9234 | Val PPL:   2750.45
  Learning Rate: 0.000100
  Computing BLEU on 100 samples...
  BLEU Score: 2.35

  Translation Examples:
    [1] Source:     안녕하세요
        Reference:  Hello
        Prediction: Hello there

    [2] Source:     오늘 날씨가 좋아요
        Reference:  The weather is nice today
        Prediction: Today the weather is good
```

### Two Best Models

The training will save two separate best models:

1. **`checkpoints/best_model.pt`**: Best validation loss
   - May have lowest perplexity
   - Best for teacher forcing scenarios

2. **`checkpoints/best_bleu_model.pt`**: Best BLEU score
   - Best translation quality
   - Recommended for inference/deployment

### Configuration Options

BLEU computation can be adjusted:

- **Number of samples**: Change `num_samples` in `compute_bleu_score()` call (default: 100)
- **Number of examples**: Change `num_examples` in `generate_inference_examples()` call (default: 2)
- **Decoding method**: Currently uses greedy; could switch to beam search for higher quality (slower)

## Design Decisions

### 1. Greedy vs Beam Search for BLEU

**Choice**: Greedy decoding

**Rationale**:
- Much faster (important for frequent BLEU computation)
- Good enough for monitoring training progress
- Beam search can be used for final evaluation

### 2. Sample Size for BLEU

**Choice**: 100 samples

**Rationale**:
- Balance between accuracy and speed
- Large enough for reliable BLEU estimate
- Fast enough to not slow training significantly
- Can be adjusted based on dataset size

### 3. Full vs Subset Dataset

**Choice**: Use full validation dataset for BLEU, even with `--small` flag

**Rationale**:
- More representative BLEU scores
- Better quality inference examples
- Subset is only for training loop (speed)
- BLEU computed infrequently, so speed less critical

### 4. Two Best Models

**Choice**: Save both best loss and best BLEU models

**Rationale**:
- Loss and BLEU may not correlate perfectly
- Loss is better for convergence monitoring
- BLEU is better for actual translation quality
- Gives flexibility in model selection

## Testing

Run the integration test:

```bash
/home/arnold/venv/bin/python tests/test_phase5_integration.py
```

Expected output:
```
================================================================================
Testing Phase 5: BLEU Integration
================================================================================
Loading tokenizers...
Korean vocab size: 16000
English vocab size: 16000

✓ Trainer has translator: True
✓ Trainer has src_tokenizer: True
✓ Trainer has tgt_tokenizer: True
✓ Trainer has val_dataset: True

Testing BLEU computation (on 10 samples)...
✓ BLEU computation successful: 0.00

Testing inference example generation...
✓ Generated 2 inference examples:
  [1] Source:     먹을래요.
      Reference:  I will eat it.
      Prediction: gone gone gone gone... (untrained model)

Phase 5 Integration Test Passed!
```

Note: BLEU of 0.00 and gibberish predictions are expected for untrained models.

## Dependencies

Phase 5 requires:
- `sacrebleu>=2.3.0` (for BLEU computation)
- Already listed in `requirements.txt`

## Benefits

1. **Real-time Quality Monitoring**: See translation quality improve during training
2. **Better Model Selection**: Choose model by BLEU (quality) or loss (convergence)
3. **Visual Feedback**: Inference examples show what model is learning
4. **Debugging**: Quickly spot issues (e.g., repeating tokens, empty outputs)

## Next Steps

With Phase 5 complete, the project is ready for full training:

1. **Start Training**:
   ```bash
   /home/arnold/venv/bin/python scripts/train.py
   ```

2. **Monitor Progress**:
   - Watch validation loss decrease
   - Watch BLEU score increase
   - Inspect inference examples for quality

3. **Model Selection**:
   - Use `best_model.pt` if loss is priority
   - Use `best_bleu_model.pt` if translation quality is priority
   - Compare both on test set

4. **Hyperparameter Tuning** (Phase 6):
   - Adjust learning rate, warmup steps
   - Experiment with model size
   - Try different label smoothing values
   - Use BLEU scores to guide tuning

## Example Training Output

```
Epoch 10/50
  Train Loss: 3.2145 | Train PPL: 24.87
  Val Loss:   3.5432 | Val PPL:   34.56
  Learning Rate: 0.000234
  Computing BLEU on 100 samples...
  BLEU Score: 15.67

  Translation Examples:
    [1] Source:     저는 학생입니다
        Reference:  I am a student
        Prediction: I am a student

    [2] Source:     이것은 책입니다
        Reference:  This is a book
        Prediction: This is the book

  -> New best BLEU model saved (BLEU: 15.67)!
```

## Summary

Phase 5 successfully integrates BLEU score computation and inference examples into the training pipeline, providing essential feedback for monitoring and improving translation quality. The implementation is efficient, flexible, and ready for production use.

**Status**: ✅ **Phase 5 Complete**
**Tested**: ✅ Integration test passes
**Ready for**: Full model training
