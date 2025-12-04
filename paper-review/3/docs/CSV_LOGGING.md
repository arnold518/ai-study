# CSV Logging for Training Metrics

**Date**: 2025-12-05
**Feature**: Comprehensive CSV logging of training metrics and configuration

## Overview

The training script now automatically logs all training metrics, model parameters, and configuration settings to a timestamped CSV file. This enables easy tracking, analysis, and visualization of training progress.

## Features

- **Automatic Logging**: CSV file created automatically at training start
- **Comprehensive Metrics**: 39 columns capturing all aspects of training
- **Timestamped Files**: Each training run gets a unique CSV file
- **Excel Compatible**: Can be opened in Excel, pandas, or any CSV viewer
- **Per-Epoch Logging**: One row per epoch with complete metrics

## CSV File Location

Files are saved in the `logs/` directory with format:
```
logs/training_log_YYYYMMDD_HHMMSS.csv
```

Example:
```
logs/training_log_20251205_034937.csv
```

## Columns Captured (39 total)

### Identification and Timing
| Column | Description | Example |
|--------|-------------|---------|
| `timestamp` | When this epoch completed | `2025-12-05 03:49:39` |
| `epoch` | Epoch number (1-indexed) | `1`, `2`, `3`, ... |
| `global_step` | Total training steps so far | `500`, `1000`, ... |
| `epoch_time_seconds` | Time for this epoch (seconds) | `12.5` |
| `cumulative_time_seconds` | Total training time so far | `125.3` |

### Training Metrics
| Column | Description | Example |
|--------|-------------|---------|
| `train_loss` | Training loss (KL divergence) | `4.5234` |
| `train_ppl` | Training perplexity (exp(loss)) | `92.15` |
| `train_kl_div` | Training KL divergence (same as loss) | `4.5234` |
| `grad_norm` | Average gradient norm | `0.8523` |

### Validation Metrics
| Column | Description | Example |
|--------|-------------|---------|
| `val_loss` | Validation loss | `4.2345` |
| `val_ppl` | Validation perplexity | `68.94` |
| `val_kl_div` | Validation KL divergence | `4.2345` |
| `val_bleu` | Validation BLEU score | `15.67` |

### Learning and Optimization
| Column | Description | Example |
|--------|-------------|---------|
| `learning_rate` | Current learning rate | `0.000234` |
| `best_train_loss` | Best training loss so far | `3.8234` |
| `best_val_loss` | Best validation loss so far | `4.1234` |
| `best_bleu` | Best BLEU score so far | `18.45` |

### Checkpoint Information
| Column | Description | Example |
|--------|-------------|---------|
| `is_best_loss` | Is this epoch the best loss? | `True`, `False` |
| `is_best_bleu` | Is this epoch the best BLEU? | `True`, `False` |
| `checkpoint_path` | Path to saved checkpoint (if any) | `checkpoints/best_model.pt` |
| `checkpoint_type` | Type of checkpoint | `best_loss`, `best_bleu`, `periodic` |

### Model Architecture (from config)
| Column | Description | Example |
|--------|-------------|---------|
| `d_model` | Model dimension | `512` |
| `num_heads` | Number of attention heads | `8` |
| `num_encoder_layers` | Number of encoder layers | `6` |
| `num_decoder_layers` | Number of decoder layers | `6` |
| `d_ff` | Feed-forward dimension | `2048` |
| `dropout` | Dropout rate | `0.1` |

### Training Hyperparameters
| Column | Description | Example |
|--------|-------------|---------|
| `batch_size` | Training batch size | `64` |
| `max_seq_length` | Maximum sequence length | `128` |
| `learning_rate_factor` | Learning rate factor | `1.0` |
| `warmup_steps` | Warmup steps for LR schedule | `4000` |
| `label_smoothing` | Label smoothing factor | `0.1` |
| `grad_clip` | Gradient clipping threshold | `1.0` |

### Vocabulary and Data
| Column | Description | Example |
|--------|-------------|---------|
| `src_vocab_size` | Source vocabulary size | `16000` |
| `tgt_vocab_size` | Target vocabulary size | `16000` |
| `train_size` | Training dataset size | `897566` |
| `val_size` | Validation dataset size | `1896` |

### Evaluation Settings
| Column | Description | Example |
|--------|-------------|---------|
| `bleu_num_samples` | Samples used for BLEU computation | `100` |
| `inference_num_examples` | Number of inference examples shown | `2` |

## Usage

### Basic Usage

Training automatically enables CSV logging:

```bash
/home/arnold/venv/bin/python scripts/train.py
```

Output:
```
CSV logging enabled: logs/training_log_20251205_123456.csv
```

### Analyzing with Pandas

```python
import pandas as pd

# Load CSV
df = pd.read_csv('logs/training_log_20251205_123456.csv')

# View summary statistics
print(df[['epoch', 'train_loss', 'val_loss', 'val_bleu']].describe())

# Plot training curves
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.plot(df['epoch'], df['train_loss'], label='Train')
plt.plot(df['epoch'], df['val_loss'], label='Val')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Validation Loss')

plt.subplot(1, 3, 2)
plt.plot(df['epoch'], df['val_bleu'])
plt.xlabel('Epoch')
plt.ylabel('BLEU Score')
plt.title('Validation BLEU')

plt.subplot(1, 3, 3)
plt.plot(df['epoch'], df['learning_rate'])
plt.xlabel('Epoch')
plt.ylabel('Learning Rate')
plt.title('Learning Rate Schedule')

plt.tight_layout()
plt.savefig('training_curves.png')
```

### Comparing Multiple Runs

```python
import pandas as pd
import glob

# Load all training logs
log_files = glob.glob('logs/training_log_*.csv')
dfs = []

for log_file in log_files:
    df = pd.read_csv(log_file)
    df['run'] = log_file  # Add run identifier
    dfs.append(df)

# Combine all runs
all_runs = pd.concat(dfs, ignore_index=True)

# Compare final BLEU scores
final_bleu = all_runs.groupby('run')['val_bleu'].max()
print(final_bleu.sort_values(ascending=False))
```

### Finding Best Model

```python
import pandas as pd

df = pd.read_csv('logs/training_log_20251205_123456.csv')

# Find epoch with best BLEU
best_bleu_epoch = df.loc[df['val_bleu'].idxmax()]
print(f"Best BLEU: {best_bleu_epoch['val_bleu']:.2f} at epoch {best_bleu_epoch['epoch']}")
print(f"Checkpoint: {best_bleu_epoch['checkpoint_path']}")

# Find epoch with best validation loss
best_loss_epoch = df.loc[df['val_loss'].idxmin()]
print(f"Best Val Loss: {best_loss_epoch['val_loss']:.4f} at epoch {best_loss_epoch['epoch']}")
```

### Analyzing Training Stability

```python
import pandas as pd

df = pd.read_csv('logs/training_log_20251205_123456.csv')

# Check gradient norms (high values = instability)
print("Gradient norm statistics:")
print(df['grad_norm'].describe())

# Check for loss spikes
loss_std = df['train_loss'].rolling(5).std()
if (loss_std > 1.0).any():
    print("Warning: Training instability detected!")

# Check learning rate schedule
print("\nLearning rate over time:")
print(df[['epoch', 'learning_rate']].head(10))
```

### Export for Excel

The CSV files are Excel-compatible. Simply open in Excel or export:

```python
import pandas as pd

df = pd.read_csv('logs/training_log_20251205_123456.csv')

# Export to Excel with formatted columns
df.to_excel('training_log.xlsx', index=False)
```

## Example CSV Output

```csv
timestamp,epoch,global_step,train_loss,train_ppl,train_kl_div,val_loss,val_ppl,val_kl_div,val_bleu,learning_rate,grad_norm,best_train_loss,best_val_loss,best_bleu,is_best_loss,is_best_bleu,checkpoint_path,checkpoint_type,d_model,num_heads,...
2025-12-05 10:30:15,1,1400,5.234,187.23,5.234,4.891,133.12,4.891,2.34,0.000123,0.912,5.234,4.891,2.34,True,True,checkpoints/best_model.pt,best_loss,512,8,...
2025-12-05 10:45:32,2,2800,4.567,96.34,4.567,4.321,75.45,4.321,8.67,0.000187,0.823,4.567,4.321,8.67,True,True,checkpoints/best_model.pt,best_loss,512,8,...
2025-12-05 11:00:48,3,4200,4.123,61.78,4.123,3.987,53.98,3.987,12.45,0.000234,0.765,4.123,3.987,12.45,True,True,checkpoints/best_model.pt,best_loss,512,8,...
```

## Visualization Examples

### Training Progress Dashboard

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('logs/training_log_20251205_123456.csv')

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Loss curves
axes[0, 0].plot(df['epoch'], df['train_loss'], 'b-', label='Train')
axes[0, 0].plot(df['epoch'], df['val_loss'], 'r-', label='Val')
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('Loss (KL Divergence)')
axes[0, 0].legend()
axes[0, 0].set_title('Training Progress')
axes[0, 0].grid(True, alpha=0.3)

# BLEU score
axes[0, 1].plot(df['epoch'], df['val_bleu'], 'g-')
axes[0, 1].set_xlabel('Epoch')
axes[0, 1].set_ylabel('BLEU Score')
axes[0, 1].set_title('Validation BLEU')
axes[0, 1].grid(True, alpha=0.3)

# Learning rate schedule
axes[0, 2].plot(df['epoch'], df['learning_rate'], 'orange')
axes[0, 2].set_xlabel('Epoch')
axes[0, 2].set_ylabel('Learning Rate')
axes[0, 2].set_title('Learning Rate Schedule')
axes[0, 2].grid(True, alpha=0.3)

# Perplexity
axes[1, 0].plot(df['epoch'], df['train_ppl'], 'b-', label='Train')
axes[1, 0].plot(df['epoch'], df['val_ppl'], 'r-', label='Val')
axes[1, 0].set_xlabel('Epoch')
axes[1, 0].set_ylabel('Perplexity')
axes[1, 0].legend()
axes[1, 0].set_title('Perplexity')
axes[1, 0].grid(True, alpha=0.3)

# Gradient norm
axes[1, 1].plot(df['epoch'], df['grad_norm'], 'purple')
axes[1, 1].set_xlabel('Epoch')
axes[1, 1].set_ylabel('Gradient Norm')
axes[1, 1].set_title('Gradient Norm (Stability)')
axes[1, 1].grid(True, alpha=0.3)

# Training time
axes[1, 2].plot(df['epoch'], df['epoch_time_seconds'], 'brown')
axes[1, 2].set_xlabel('Epoch')
axes[1, 2].set_ylabel('Time (seconds)')
axes[1, 2].set_title('Time per Epoch')
axes[1, 2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('training_dashboard.png', dpi=150)
```

## Implementation Details

### Code Structure

**CSV Logger** (`src/utils/csv_logger.py`):
- `CSVLogger` class handles file creation and writing
- Automatically adds timestamp and config parameters
- Thread-safe writing

**Trainer Integration** (`src/training/trainer.py`):
- Initialized in `__init__()` with timestamped filename
- Logs after each epoch (validation or non-validation)
- Tracks gradient norms, timing, and checkpoint information

### What Gets Logged When

| Scenario | Logged Columns | Notes |
|----------|---------------|-------|
| **Every epoch** | epoch, train_loss, train_ppl, learning_rate, grad_norm, timing | Always logged |
| **Validation epoch** | + val_loss, val_ppl, val_bleu | Only when `eval_every` |
| **Best model saved** | + checkpoint_path, checkpoint_type, is_best_* | When new best found |
| **Periodic checkpoint** | Tracked in separate row | Every `save_every` |

### Empty vs Missing Values

- **Empty string (`""`)**: Intentionally not applicable (e.g., no checkpoint saved)
- **Missing/blank**: Validation not performed this epoch

## Benefits

1. **Complete Record**: Every training run fully documented
2. **Easy Analysis**: CSV format works with any tool
3. **Reproducibility**: All hyperparameters captured
4. **Debugging**: Track instabilities via gradient norms
5. **Comparison**: Easy to compare multiple runs
6. **Publication**: Ready for plots in papers
7. **Monitoring**: Can parse CSV to monitor training remotely

## Tips

### Monitor Training Remotely

```bash
# On remote server
/home/arnold/venv/bin/python scripts/train.py

# On local machine
watch -n 60 "tail -5 logs/training_log_*.csv"
```

### Quick Check Progress

```python
import pandas as pd

df = pd.read_csv('logs/training_log_20251205_123456.csv')
latest = df.iloc[-1]

print(f"Epoch {int(latest['epoch'])}/{100}")
print(f"Val Loss: {latest['val_loss']:.4f}")
print(f"BLEU: {latest['val_bleu']:.2f}")
print(f"Best BLEU: {latest['best_bleu']:.2f}")
print(f"Time: {latest['cumulative_time_seconds']/3600:.1f}h")
```

### Alert on Milestones

```python
import pandas as pd

df = pd.read_csv('logs/training_log_20251205_123456.csv')

if df['val_bleu'].max() > 20:
    print("🎉 BLEU exceeded 20!")

if df['grad_norm'].iloc[-1] > 10:
    print("⚠️  High gradient norm detected!")
```

## Summary

The CSV logging feature provides comprehensive, automatic tracking of all training metrics and configuration. Every training run creates a timestamped CSV with 39 columns capturing:

- Training/validation metrics (loss, PPL, BLEU)
- Model architecture and hyperparameters
- Checkpoint information and best model tracking
- Timing and gradient statistics

This enables easy analysis, visualization, comparison, and debugging of training runs.

**Test**: Run `python tests/test_csv_logging.py` to verify functionality
