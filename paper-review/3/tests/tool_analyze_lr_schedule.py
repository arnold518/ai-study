#!/usr/bin/env python
"""Analyze learning rate schedule for different dataset sizes."""

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

def analyze_schedule(dataset_size, batch_size, num_epochs, warmup_steps, d_model=512):
    """Analyze learning rate schedule."""
    steps_per_epoch = dataset_size // batch_size
    total_steps = steps_per_epoch * num_epochs
    peak_lr = calculate_lr(warmup_steps, d_model, warmup_steps)

    print(f"Dataset size: {dataset_size:,} samples")
    print(f"Batch size: {batch_size}")
    print(f"Steps per epoch: {steps_per_epoch:,}")
    print(f"Total training steps ({num_epochs} epochs): {total_steps:,}")
    print()
    print(f"Warmup steps: {warmup_steps:,}")
    print(f"Warmup duration: {warmup_steps / steps_per_epoch:.2f} epochs")
    print(f"Peak learning rate: {peak_lr:.6f}")
    print()

    # Show LR at key milestones
    milestones = [
        ("After 1 epoch", steps_per_epoch),
        ("After 5 epochs", 5 * steps_per_epoch),
        ("After 10 epochs", 10 * steps_per_epoch),
        ("After 20 epochs", 20 * steps_per_epoch),
    ]

    print("Learning rate at milestones:")
    for label, step in milestones:
        if step <= total_steps:
            lr = calculate_lr(step, d_model, warmup_steps)
            ratio = lr / peak_lr
            print(f"  {label:20s} (step {step:7,}): {lr:.6f} ({ratio:.2%} of peak)")
    print()

def main():
    config = TransformerConfig()

    print("=" * 80)
    print("LEARNING RATE SCHEDULE ANALYSIS")
    print("=" * 80)
    print()

    # Old dataset
    print("SCENARIO 1: Old Dataset (Character-based filtering)")
    print("-" * 80)
    analyze_schedule(
        dataset_size=417_557,
        batch_size=config.batch_size,
        num_epochs=config.num_epochs,
        warmup_steps=config.warmup_steps,
        d_model=config.d_model
    )

    # New dataset
    print("SCENARIO 2: New Dataset (Token-based filtering)")
    print("-" * 80)
    analyze_schedule(
        dataset_size=1_697_929,
        batch_size=config.batch_size,
        num_epochs=config.num_epochs,
        warmup_steps=config.warmup_steps,
        d_model=config.d_model
    )

    # Recommendations
    print("=" * 80)
    print("ANALYSIS & RECOMMENDATIONS")
    print("=" * 80)
    print()

    old_steps_per_epoch = 417_557 // config.batch_size
    new_steps_per_epoch = 1_697_929 // config.batch_size

    print("PROBLEM:")
    print(f"  - Old dataset: warmup = {config.warmup_steps / old_steps_per_epoch:.2f} epochs")
    print(f"  - New dataset: warmup = {config.warmup_steps / new_steps_per_epoch:.2f} epochs")
    print(f"  ✗ Warmup is now {old_steps_per_epoch / new_steps_per_epoch:.1f}x SHORTER relative to dataset!")
    print()

    print("RECOMMENDED WARMUP STEPS:")
    print()

    # Different warmup strategies
    strategies = [
        ("Conservative (0.5 epoch)", new_steps_per_epoch * 0.5),
        ("Moderate (1.0 epoch)", new_steps_per_epoch * 1.0),
        ("Standard (1.5 epochs)", new_steps_per_epoch * 1.5),
        ("Paper-like (4.1x scale)", config.warmup_steps * 4.1),
    ]

    for name, warmup in strategies:
        warmup = int(warmup)
        peak_lr = calculate_lr(warmup, config.d_model, warmup)
        print(f"{name:30s}: {warmup:6,} steps → peak LR = {peak_lr:.6f}")

    print()
    print("OPTIMAL RECOMMENDATION:")
    optimal_warmup = int(new_steps_per_epoch * 1.2)  # 1.2 epochs
    peak_lr = calculate_lr(optimal_warmup, config.d_model, optimal_warmup)
    print(f"  warmup_steps = {optimal_warmup:,} (~1.2 epochs)")
    print(f"  Peak learning rate: {peak_lr:.6f}")
    print()
    print("RATIONALE:")
    print("  - Original paper used 4000 steps for 4.5M pairs WMT")
    print(f"  - Our dataset: 1.7M pairs (38% of WMT size)")
    print(f"  - Scaled warmup: {optimal_warmup:,} steps maintains same warmup/dataset ratio")
    print("  - Covers ~1.2 epochs, ensuring stable initialization")
    print()

if __name__ == "__main__":
    main()
