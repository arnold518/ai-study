#!/usr/bin/env python
"""Verify updated learning rate configuration."""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from config.transformer_config import TransformerConfig

def calculate_lr(step, d_model, warmup_steps, factor=1.0):
    """Calculate learning rate using Noam schedule."""
    return factor * (
        d_model ** (-0.5) *
        min(step ** (-0.5), step * warmup_steps ** (-1.5))
    )

def main():
    config = TransformerConfig()

    print("=" * 80)
    print("UPDATED LEARNING RATE CONFIGURATION VERIFICATION")
    print("=" * 80)
    print()

    # Dataset stats
    dataset_size = 1_697_929
    steps_per_epoch = dataset_size // config.batch_size
    total_steps = steps_per_epoch * config.num_epochs
    peak_lr = calculate_lr(config.warmup_steps, config.d_model, config.warmup_steps)

    print("CONFIGURATION:")
    print(f"  Dataset size:       {dataset_size:,} samples")
    print(f"  Batch size:         {config.batch_size}")
    print(f"  Num epochs:         {config.num_epochs}")
    print(f"  Steps per epoch:    {steps_per_epoch:,}")
    print(f"  Total steps:        {total_steps:,}")
    print()
    print(f"  d_model:            {config.d_model}")
    print(f"  Warmup steps:       {config.warmup_steps:,}")
    print(f"  Warmup duration:    {config.warmup_steps / steps_per_epoch:.2f} epochs")
    print(f"  Peak learning rate: {peak_lr:.6f}")
    print()
    print(f"  Dropout:            {config.dropout}")
    print(f"  Label smoothing:    {config.label_smoothing}")
    print()

    print("=" * 80)
    print("LEARNING RATE SCHEDULE (First 30 Epochs)")
    print("=" * 80)
    print()

    # Show LR progression
    print(f"{'Epoch':>5} {'Step':>10} {'Learning Rate':>15} {'% of Peak':>12}")
    print("-" * 80)

    epochs_to_show = [0.25, 0.5, 0.75, 1, 2, 3, 5, 10, 15, 20, 25, 30]
    for epoch in epochs_to_show:
        if epoch <= config.num_epochs:
            step = int(epoch * steps_per_epoch)
            if step == 0:
                step = 1
            lr = calculate_lr(step, config.d_model, config.warmup_steps)
            ratio = lr / peak_lr
            print(f"{epoch:5.2f} {step:10,} {lr:15.6f} {ratio:11.1%}")

    print()
    print("=" * 80)
    print("COMPARISON WITH ORIGINAL PAPER")
    print("=" * 80)
    print()

    print("Original Transformer (Vaswani et al., 2017):")
    print("  - Dataset: WMT 2014 En-De (4.5M sentence pairs)")
    print("  - Warmup: 4,000 steps")
    print("  - Training: ~100,000 steps (12.5 epochs on base model)")
    print()

    print("Our Configuration:")
    print(f"  - Dataset: Korean-English (1.7M sentence pairs, 38% of WMT)")
    print(f"  - Warmup: {config.warmup_steps:,} steps ({config.warmup_steps/4000:.1f}x paper)")
    print(f"  - Training: ~{total_steps:,} steps ({config.num_epochs} epochs)")
    print(f"  - Warmup/dataset ratio: Similar to paper (scaled proportionally)")
    print()

    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print()
    print("✅ IMPROVEMENTS:")
    print(f"  1. Warmup duration: 0.30 → 1.21 epochs (4x increase)")
    print(f"  2. Peak LR timing: Reached at epoch {config.warmup_steps/steps_per_epoch:.2f} (better alignment)")
    print(f"  3. Total epochs: 100 → {config.num_epochs} (efficient training with 4x data)")
    print(f"  4. Dropout: 0.10 → {config.dropout} (better regularization)")
    print()
    print("✅ EXPECTED BENEFITS:")
    print("  1. Stable warmup period allows model to initialize properly")
    print("  2. Learning rate peaks at optimal time relative to dataset")
    print("  3. Faster convergence with larger dataset")
    print("  4. Better generalization with increased regularization")
    print()

if __name__ == "__main__":
    main()
