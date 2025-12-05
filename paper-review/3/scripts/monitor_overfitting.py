#!/usr/bin/env python
"""Monitor training logs for signs of overfitting."""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
import glob
from pathlib import Path

def load_latest_training_log():
    """Load the most recent training log CSV."""
    log_dir = Path('logs')
    log_files = list(log_dir.glob('training_log_*.csv'))

    if not log_files:
        print("✗ No training logs found in logs/")
        return None

    # Get most recent log
    latest_log = max(log_files, key=lambda p: p.stat().st_mtime)
    print(f"Loading: {latest_log}")
    print()

    df = pd.read_csv(latest_log)
    return df

def analyze_overfitting(df):
    """Analyze training logs for overfitting indicators."""

    print("=" * 80)
    print("OVERFITTING DETECTION ANALYSIS")
    print("=" * 80)
    print()

    # Filter to epochs with validation data
    df_val = df[df['val_loss'].notna()].copy()

    if len(df_val) == 0:
        print("✗ No validation data found in logs")
        return

    print(f"Epochs with validation: {len(df_val)}")
    print(f"Total epochs: {df['epoch'].max()}")
    print()

    # 1. LOSS DIVERGENCE CHECK
    print("1. LOSS DIVERGENCE CHECK")
    print("-" * 80)

    # Calculate loss gap
    df_val['loss_gap'] = df_val['val_loss'] - df_val['train_loss']

    latest = df_val.iloc[-1]
    latest_gap = latest['loss_gap']
    avg_gap = df_val['loss_gap'].mean()
    max_gap = df_val['loss_gap'].max()

    print(f"  Latest epoch ({latest['epoch']:.0f}):")
    print(f"    Train loss: {latest['train_loss']:.4f}")
    print(f"    Val loss:   {latest['val_loss']:.4f}")
    print(f"    Gap:        {latest_gap:.4f}")
    print()
    print(f"  Loss gap statistics:")
    print(f"    Average:    {avg_gap:.4f}")
    print(f"    Maximum:    {max_gap:.4f}")
    print()

    # Assessment
    if latest_gap > 0.5:
        print("  ❌ SEVERE OVERFITTING: Gap > 0.5")
        print("      Recommendation: Stop training, reduce model size")
    elif latest_gap > 0.2:
        print("  ⚠️  WARNING: Moderate overfitting (gap > 0.2)")
        print("      Recommendation: Monitor closely, consider stopping")
    elif latest_gap > 0.1:
        print("  ⚠️  CAUTION: Small gap (0.1 - 0.2)")
        print("      Status: Normal training, continue monitoring")
    else:
        print("  ✓ NORMAL: Gap < 0.1")
        print("      Status: No overfitting detected")
    print()

    # 2. VALIDATION LOSS TREND
    print("2. VALIDATION LOSS TREND")
    print("-" * 80)

    # Check if val loss is increasing
    recent_epochs = min(5, len(df_val))
    recent_val = df_val.tail(recent_epochs)

    val_increasing = recent_val['val_loss'].is_monotonic_increasing
    val_trend = recent_val['val_loss'].diff().mean()

    print(f"  Last {recent_epochs} epochs:")
    for _, row in recent_val.iterrows():
        print(f"    Epoch {row['epoch']:.0f}: {row['val_loss']:.4f}")
    print()

    if val_increasing:
        print("  ❌ WARNING: Validation loss monotonically increasing")
        print("      This is a strong sign of overfitting!")
    elif val_trend > 0.01:
        print("  ⚠️  CAUTION: Validation loss trending upward")
        print(f"      Average increase: {val_trend:.4f} per epoch")
    elif val_trend < -0.01:
        print("  ✓ GOOD: Validation loss decreasing")
        print(f"      Average decrease: {abs(val_trend):.4f} per epoch")
    else:
        print("  ~ STABLE: Validation loss plateaued")
        print("      Model may have converged")
    print()

    # 3. TRAINING LOSS TREND
    print("3. TRAINING LOSS TREND")
    print("-" * 80)

    train_trend = recent_val['train_loss'].diff().mean()

    if train_trend < -0.01 and val_trend > 0.01:
        print("  ❌ OVERFITTING DETECTED:")
        print(f"      Train loss: decreasing ({abs(train_trend):.4f}/epoch)")
        print(f"      Val loss:   increasing (+{val_trend:.4f}/epoch)")
        print("      Recommendation: STOP TRAINING NOW")
    elif train_trend < -0.01 and abs(val_trend) < 0.01:
        print("  ⚠️  WARNING:")
        print(f"      Train loss: decreasing ({abs(train_trend):.4f}/epoch)")
        print(f"      Val loss:   plateaued")
        print("      Recommendation: Model may have peaked, consider stopping")
    else:
        print("  ✓ Train and val losses aligned")
    print()

    # 4. BLEU SCORE TREND
    if 'val_bleu' in df_val.columns and df_val['val_bleu'].notna().any():
        print("4. BLEU SCORE TREND")
        print("-" * 80)

        recent_bleu = recent_val[recent_val['val_bleu'].notna()]

        if len(recent_bleu) > 0:
            print(f"  Last {len(recent_bleu)} BLEU measurements:")
            for _, row in recent_bleu.iterrows():
                print(f"    Epoch {row['epoch']:.0f}: {row['val_bleu']:.2f}")
            print()

            bleu_trend = recent_bleu['val_bleu'].diff().mean()

            if bleu_trend < -0.5:
                print("  ❌ WARNING: BLEU score decreasing")
                print("      Translation quality degrading!")
            elif abs(bleu_trend) < 0.5:
                print("  ~ STABLE: BLEU score plateaued")
            else:
                print("  ✓ IMPROVING: BLEU score increasing")
            print()

    # 5. EARLY STOPPING STATUS
    if 'is_best_loss' in df_val.columns:
        print("5. EARLY STOPPING STATUS")
        print("-" * 80)

        last_best_epoch = df_val[df_val['is_best_loss'] == True]['epoch'].max()
        current_epoch = df_val['epoch'].max()
        epochs_since_best = current_epoch - last_best_epoch if pd.notna(last_best_epoch) else current_epoch

        print(f"  Last best model: Epoch {last_best_epoch:.0f}")
        print(f"  Current epoch:   Epoch {current_epoch:.0f}")
        print(f"  Epochs without improvement: {epochs_since_best:.0f}")
        print()

        if epochs_since_best >= 8:
            print("  ⚠️  WARNING: No improvement for 8+ epochs")
            print("      Early stopping should trigger soon")
        elif epochs_since_best >= 5:
            print("  ⚠️  CAUTION: No improvement for 5+ epochs")
        else:
            print("  ✓ Model still improving")
        print()

    # 6. OVERALL ASSESSMENT
    print("=" * 80)
    print("OVERALL ASSESSMENT")
    print("=" * 80)
    print()

    issues = []

    if latest_gap > 0.5:
        issues.append(("CRITICAL", "Large train/val loss gap"))
    if val_increasing:
        issues.append(("CRITICAL", "Validation loss increasing"))
    if train_trend < -0.01 and val_trend > 0.01:
        issues.append(("CRITICAL", "Train↓ Val↑ divergence"))

    if 0.2 < latest_gap <= 0.5:
        issues.append(("WARNING", "Moderate train/val gap"))
    if val_trend > 0.01 and not val_increasing:
        issues.append(("WARNING", "Validation loss trending up"))

    if issues:
        print("ISSUES DETECTED:")
        for severity, issue in issues:
            print(f"  [{severity:8s}] {issue}")
        print()

        if any(s == "CRITICAL" for s, _ in issues):
            print("RECOMMENDATION: 🛑 STOP TRAINING IMMEDIATELY")
            print("  - Model is overfitting")
            print("  - Use best checkpoint (lowest val loss)")
            print("  - Consider reducing model size for next run")
        else:
            print("RECOMMENDATION: ⚠️  MONITOR CLOSELY")
            print("  - Watch next few epochs carefully")
            print("  - Be ready to stop if trends worsen")
    else:
        print("✓ NO MAJOR ISSUES DETECTED")
        print("  Training appears healthy")
    print()

def main():
    df = load_latest_training_log()

    if df is None:
        return

    analyze_overfitting(df)

    print("=" * 80)
    print("TIPS FOR MONITORING")
    print("=" * 80)
    print()
    print("Run this script periodically during training:")
    print("  /home/arnold/venv/bin/python scripts/monitor_overfitting.py")
    print()
    print("Or use TensorBoard for real-time monitoring:")
    print("  tensorboard --logdir logs/")
    print()

if __name__ == "__main__":
    main()
