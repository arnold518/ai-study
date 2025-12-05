#!/usr/bin/env python
"""Verify learning rate schedule from training log."""

import math

# Configuration values
d_model = 512
warmup_steps = 4000
factor = 1.0

def noam_lr(step, d_model, warmup_steps, factor=1.0):
    """Calculate Noam learning rate."""
    if step == 0:
        step = 1
    return factor * (d_model ** -0.5) * min(step ** -0.5, step * warmup_steps ** -1.5)

# Test values from the training log
test_cases = [
    (3263, 0.0005700226414517043),   # Epoch 1
    (4000, None),  # Warmup peak
    (6526, 0.0005470682175853403),   # Epoch 2
    (13052, 0.00038683564642623186), # Epoch 4
    (32630, 0.0002446563445700934),  # Epoch 10
    (65260, 0.0001729981603058256),  # Epoch 20
    (97890, 0.00014125240639649325), # Epoch 30
    (195780, None),  # Epoch 60
    (326300, None),  # Epoch 100
]

print("=" * 80)
print("Learning Rate Verification")
print("=" * 80)
print(f"\nConfiguration:")
print(f"  d_model = {d_model}")
print(f"  warmup_steps = {warmup_steps}")
print(f"  factor = {factor}")
print()

print("Formula: factor * d_model^(-0.5) * min(step^(-0.5), step * warmup^(-1.5))")
print()

print("-" * 80)
print(f"{'Step':<10} {'Expected LR':<20} {'Actual LR':<20} {'Match?':<10}")
print("-" * 80)

for step, actual_lr in test_cases:
    expected_lr = noam_lr(step, d_model, warmup_steps, factor)

    if actual_lr is not None:
        diff = abs(expected_lr - actual_lr)
        match = "✓ YES" if diff < 1e-10 else "✗ NO"
        print(f"{step:<10} {expected_lr:<20.15f} {actual_lr:<20.15f} {match:<10}")
    else:
        print(f"{step:<10} {expected_lr:<20.15f} {'(not logged)':<20} {'-':<10}")

print("-" * 80)
print()

# Check the warmup phase
print("Warmup Phase Analysis:")
print("-" * 80)
print(f"{'Step':<10} {'Learning Rate':<20} {'Phase':<20}")
print("-" * 80)

warmup_checkpoints = [1, 500, 1000, 2000, 3000, 4000, 5000, 8000]
for step in warmup_checkpoints:
    lr = noam_lr(step, d_model, warmup_steps, factor)

    if step < warmup_steps:
        phase = "WARMUP (increasing)"
    elif step == warmup_steps:
        phase = "PEAK"
    else:
        phase = "DECAY (decreasing)"

    print(f"{step:<10} {lr:<20.15f} {phase:<20}")

print("-" * 80)
print()

# Analyze the issue
print("Analysis:")
print("=" * 80)

# Calculate peak LR
peak_lr = noam_lr(warmup_steps, d_model, warmup_steps, factor)
print(f"Peak learning rate (at step {warmup_steps}): {peak_lr:.10f}")
print()

# First epoch LR
first_epoch_step = 3263
first_epoch_lr = noam_lr(first_epoch_step, d_model, warmup_steps, factor)
print(f"Learning rate at first epoch (step {first_epoch_step}): {first_epoch_lr:.10f}")
print(f"  This is {(first_epoch_step/warmup_steps)*100:.1f}% through warmup")
print(f"  LR is {(first_epoch_lr/peak_lr)*100:.1f}% of peak")
print()

# Check if warmup was completed
steps_per_epoch = 3263  # From epoch 1
warmup_epoch = warmup_steps / steps_per_epoch
print(f"Steps per epoch: ~{steps_per_epoch}")
print(f"Warmup completes at epoch: ~{warmup_epoch:.1f}")
print()

# Compare with paper's recommendation
print("Paper's Recommendation:")
print("  'We varied the learning rate over the course of training, according to")
print("   the formula: lrate = d_model^(-0.5) * min(step^(-0.5), step * warmup^(-1.5))'")
print()
print("  The paper uses:")
print("    - d_model = 512 (base model)")
print("    - warmup_steps = 4000")
print()

# Our settings
print("Our Settings:")
print(f"  - d_model = {d_model}")
print(f"  - warmup_steps = {warmup_steps}")
print(f"  - factor = {factor}")
print()

# Calculate some key values
print("Key Values:")
print(f"  d_model^(-0.5) = {d_model ** -0.5:.10f}")
print(f"  warmup_steps^(-1.5) = {warmup_steps ** -1.5:.10f}")
print(f"  Peak LR (at step 4000) = {peak_lr:.10f}")
print()

# Problem diagnosis
print("=" * 80)
print("DIAGNOSIS:")
print("=" * 80)

if first_epoch_step < warmup_steps:
    print("⚠ WARNING: First epoch completes BEFORE warmup finishes!")
    print(f"  - First epoch ends at step {first_epoch_step}")
    print(f"  - Warmup completes at step {warmup_steps}")
    print(f"  - Only {(first_epoch_step/warmup_steps)*100:.1f}% through warmup at epoch 1")
    print()
    print("ISSUE: With 417,557 samples and batch_size=128:")
    print(f"  - Batches per epoch = 417,557 / 128 = {417557//128}")
    print(f"  - Warmup takes {warmup_steps / (417557//128):.2f} epochs")
    print()
    print("This means:")
    print("  - Learning rate is still increasing during epoch 1")
    print("  - Peak learning rate reached around epoch 2")
    print("  - Most of training uses decaying learning rate")
    print()

actual_batches_per_epoch = 417557 // 128
warmup_in_epochs = warmup_steps / actual_batches_per_epoch

if warmup_in_epochs > 1.5:
    print("⚠ PROBLEM: Warmup takes too long!")
    print(f"  - Current: {warmup_in_epochs:.2f} epochs")
    print(f"  - Recommended: 0.5-1.0 epochs for this dataset size")
    print()
    recommended_warmup = int(actual_batches_per_epoch * 0.5)
    print(f"RECOMMENDATION: Reduce warmup_steps to {recommended_warmup} (0.5 epochs)")
    print(f"                Or use {actual_batches_per_epoch} for 1 full epoch")
else:
    print("✓ Warmup duration is reasonable")

print()
