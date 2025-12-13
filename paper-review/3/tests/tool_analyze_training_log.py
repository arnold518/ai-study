#!/usr/bin/env python
"""
Analyze training logs and generate comprehensive report.

Generates a 4-panel visualization showing:
- Loss curves (train vs val)
- Perplexity curves
- BLEU score progression
- Train/Val gap (overfitting indicator)

Usage:
    python tool_analyze_training_log.py logs/training_log_20251213_015206.csv
    python tool_analyze_training_log.py logs/training_log_*.csv --output report.png
    python tool_analyze_training_log.py logs/training_log_*.csv --format summary
"""

import sys
import os
import argparse
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # For headless environments
import matplotlib.pyplot as plt
from pathlib import Path


def print_summary(df, output_file=None):
    """
    Print detailed training summary.

    Args:
        df: DataFrame with training metrics
        output_file: Optional file to write summary to
    """
    summary_lines = []

    summary_lines.append("=" * 80)
    summary_lines.append("TRAINING LOG ANALYSIS")
    summary_lines.append("=" * 80)

    # Basic stats
    summary_lines.append(f"\n📊 Training Summary:")
    summary_lines.append(f"   Total Epochs:     {int(df['epoch'].iloc[-1])}")
    summary_lines.append(f"   Training samples: {int(df['train_size'].iloc[0])}")
    summary_lines.append(f"   Val samples:      {int(df['val_size'].iloc[0])}")

    config_info = (f"   Config: dropout={df['dropout'].iloc[0]}, "
                   f"label_smoothing={df['label_smoothing'].iloc[0]}, "
                   f"batch_size={int(df['batch_size'].iloc[0])}")
    summary_lines.append(config_info)

    # Best epoch by validation loss
    best_val_idx = df['val_loss'].idxmin()
    best_epoch = int(df.loc[best_val_idx, 'epoch'])
    best_train_loss = df.loc[best_val_idx, 'train_loss']
    best_val_loss = df.loc[best_val_idx, 'val_loss']
    best_gap = best_val_loss - best_train_loss

    summary_lines.append(f"\n📈 Best Model (Epoch {best_epoch}):")
    summary_lines.append(f"   Train Loss: {best_train_loss:.4f}")
    summary_lines.append(f"   Val Loss:   {best_val_loss:.4f}")
    summary_lines.append(f"   Gap:        {best_gap:.4f}")
    summary_lines.append(f"   Train PPL:  {df.loc[best_val_idx, 'train_ppl']:.2f}")
    summary_lines.append(f"   Val PPL:    {df.loc[best_val_idx, 'val_ppl']:.2f}")

    if 'val_bleu' in df.columns and pd.notna(df.loc[best_val_idx, 'val_bleu']):
        best_bleu = df.loc[best_val_idx, 'val_bleu']
        summary_lines.append(f"   BLEU:       {best_bleu:.2f}")

    # Final epoch
    final_epoch = int(df['epoch'].iloc[-1])
    final_train_loss = df['train_loss'].iloc[-1]
    final_val_loss = df['val_loss'].iloc[-1]
    final_gap = final_val_loss - final_train_loss

    summary_lines.append(f"\n📉 Final Model (Epoch {final_epoch}):")
    summary_lines.append(f"   Train Loss: {final_train_loss:.4f}")
    summary_lines.append(f"   Val Loss:   {final_val_loss:.4f}")
    summary_lines.append(f"   Gap:        {final_gap:.4f}")

    if 'val_bleu' in df.columns and pd.notna(df['val_bleu'].iloc[-1]):
        final_bleu = df['val_bleu'].iloc[-1]
        summary_lines.append(f"   BLEU:       {final_bleu:.2f}")

    # Degradation analysis
    degradation = final_val_loss - best_val_loss
    gap_increase = final_gap - best_gap
    degradation_pct = (degradation / best_val_loss) * 100

    summary_lines.append(f"\n🔍 Degradation from Best to Final:")
    summary_lines.append(f"   Val Loss Increase: {degradation:+.4f} ({degradation_pct:+.1f}%)")
    summary_lines.append(f"   Gap Increase:      {gap_increase:+.4f}")

    if 'val_bleu' in df.columns:
        bleu_data = df[df['val_bleu'].notna()]
        if len(bleu_data) > 0:
            best_bleu = bleu_data['val_bleu'].max()
            best_bleu_epoch = int(bleu_data.loc[bleu_data['val_bleu'].idxmax(), 'epoch'])
            final_bleu = bleu_data['val_bleu'].iloc[-1]
            bleu_drop = final_bleu - best_bleu
            summary_lines.append(f"   BLEU Drop:         {bleu_drop:+.2f} points")
            summary_lines.append(f"   Best BLEU Epoch:   {best_bleu_epoch}")

    # Training time
    if 'cumulative_time_seconds' in df.columns:
        total_time_hours = df['cumulative_time_seconds'].iloc[-1] / 3600
        avg_epoch_time = df['epoch_time_seconds'].mean() / 60
        summary_lines.append(f"\n⏱️  Training Time:")
        summary_lines.append(f"   Total:         {total_time_hours:.2f} hours")
        summary_lines.append(f"   Avg per epoch: {avg_epoch_time:.1f} minutes")

    # Verdict
    summary_lines.append(f"\n🎯 Verdict:")
    if degradation > 0.3:
        summary_lines.append("   ❌ SEVERE OVERFITTING DETECTED!")
        summary_lines.append("   - Validation loss increased significantly after best epoch")
        summary_lines.append("   - Model memorizing training data instead of generalizing")
        summary_lines.append("   Recommendation: Use best_model.pt from earlier epoch")
    elif degradation > 0.1:
        summary_lines.append("   ⚠️  Moderate overfitting detected")
        summary_lines.append("   - Some degradation after best epoch")
        summary_lines.append("   - Consider stronger regularization for future training")
    elif final_gap > 1.0:
        summary_lines.append("   ⚠️  Large train/val gap")
        summary_lines.append("   - Model starting to overfit")
        summary_lines.append("   - Early stopping should trigger soon")
    elif final_gap > 0.5:
        summary_lines.append("   ⚠️  Noticeable train/val gap")
        summary_lines.append("   - Acceptable but monitor closely")
        summary_lines.append("   - Consider if training should continue")
    else:
        summary_lines.append("   ✅ Good generalization!")
        summary_lines.append("   - Minimal overfitting")
        summary_lines.append("   - Regularization working well")
        summary_lines.append("   - Model can likely train longer")

    # Epoch-by-epoch gap progression
    summary_lines.append(f"\n📊 Train/Val Gap Evolution:")
    gap = df['val_loss'] - df['train_loss']

    # Show every 5 epochs or fewer if not many epochs
    step = max(1, len(df) // 6)
    for i in range(0, len(df), step):
        epoch = int(df.loc[i, 'epoch'])
        summary_lines.append(f"   Epoch {epoch:2d}: gap = {gap.iloc[i]:+.4f}")

    # Show final if not already shown
    if (len(df) - 1) % step != 0:
        summary_lines.append(f"   Epoch {final_epoch:2d}: gap = {final_gap:+.4f}")

    summary_lines.append("\n" + "=" * 80)

    # Print to console
    summary_text = "\n".join(summary_lines)
    print(summary_text)

    # Write to file if requested
    if output_file:
        with open(output_file, 'w') as f:
            f.write(summary_text)
        print(f"\n✓ Summary saved to: {output_file}")


def plot_training_curves(df, output_path='outputs/training_analysis.png'):
    """
    Generate 4-panel visualization of training curves.

    Args:
        df: DataFrame with training metrics
        output_path: Where to save the plot

    Returns:
        fig: Matplotlib figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # 1. Loss curves
    ax = axes[0, 0]
    ax.plot(df['epoch'], df['train_loss'], 'o-', label='Train Loss',
            linewidth=2, markersize=5, color='blue')
    ax.plot(df['epoch'], df['val_loss'], 's-', label='Val Loss',
            linewidth=2, markersize=5, color='red')

    # Mark best epoch
    best_idx = df['val_loss'].idxmin()
    best_epoch = df.loc[best_idx, 'epoch']
    ax.axvline(best_epoch, color='green', linestyle='--', alpha=0.7,
              linewidth=2, label=f'Best (Epoch {int(best_epoch)})')

    # Shade overfitting region if it exists
    if best_idx < len(df) - 1:
        ax.axvspan(best_epoch, df['epoch'].max(), alpha=0.1, color='red')

    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss (KL Divergence)', fontsize=12)
    ax.set_title('Loss Curves', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    # 2. Perplexity curves
    ax = axes[0, 1]
    ax.plot(df['epoch'], df['train_ppl'], 'o-', label='Train PPL',
            linewidth=2, markersize=5, color='blue')
    ax.plot(df['epoch'], df['val_ppl'], 's-', label='Val PPL',
            linewidth=2, markersize=5, color='red')
    ax.axvline(best_epoch, color='green', linestyle='--', alpha=0.7, linewidth=2)

    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Perplexity', fontsize=12)
    ax.set_title('Perplexity Curves', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    # Set reasonable y-axis limit
    max_ppl = max(df['train_ppl'].max(), df['val_ppl'].max())
    ax.set_ylim(bottom=0, top=min(max_ppl * 1.1, 10000))

    # 3. BLEU score (if available)
    ax = axes[1, 0]
    if 'val_bleu' in df.columns:
        bleu_data = df[df['val_bleu'].notna()]
        if len(bleu_data) > 0:
            ax.plot(bleu_data['epoch'], bleu_data['val_bleu'], 'o-',
                   color='green', linewidth=2, markersize=8, label='Val BLEU')

            # Mark best BLEU
            best_bleu_idx = bleu_data['val_bleu'].idxmax()
            best_bleu = bleu_data.loc[best_bleu_idx, 'val_bleu']
            best_bleu_epoch = bleu_data.loc[best_bleu_idx, 'epoch']
            ax.axvline(best_bleu_epoch, color='darkgreen', linestyle='--', alpha=0.5)
            ax.text(best_bleu_epoch, best_bleu, f'  Best: {best_bleu:.1f}',
                   fontsize=10, color='darkgreen', fontweight='bold')

            ax.set_xlabel('Epoch', fontsize=12)
            ax.set_ylabel('BLEU Score', fontsize=12)
            ax.set_title('BLEU Score Progress', fontsize=14, fontweight='bold')
            ax.legend(fontsize=11)
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'No BLEU data available',
                   ha='center', va='center', transform=ax.transAxes, fontsize=14)
            ax.set_title('BLEU Score (N/A)', fontsize=14, fontweight='bold')
    else:
        ax.text(0.5, 0.5, 'No BLEU data in log',
               ha='center', va='center', transform=ax.transAxes, fontsize=14)
        ax.set_title('BLEU Score (N/A)', fontsize=14, fontweight='bold')

    # 4. Train/Val Gap (overfitting metric)
    ax = axes[1, 1]
    gap = df['val_loss'] - df['train_loss']
    ax.plot(df['epoch'], gap, 'o-', color='purple', linewidth=2, markersize=6)

    # Reference lines
    ax.axhline(0, color='black', linestyle='--', alpha=0.3)
    ax.axhline(0.5, color='orange', linestyle=':', alpha=0.5, label='Acceptable (0.5)')
    ax.axhline(1.0, color='red', linestyle=':', alpha=0.5, label='Warning (1.0)')

    # Fill overfitting region
    ax.fill_between(df['epoch'], 0, gap, where=(gap > 0), alpha=0.3, color='red')

    # Mark best epoch
    ax.axvline(best_epoch, color='green', linestyle='--', alpha=0.7, linewidth=2)

    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Val Loss - Train Loss', fontsize=12)
    ax.set_title('Train/Val Gap (Overfitting Indicator)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    # Add summary annotation
    final_gap = gap.iloc[-1]
    gap_status = "⚠️ OVERFITTING!" if final_gap > 0.5 else "✓ Good"
    ax.text(0.05, 0.95, f'Final Gap: {final_gap:.3f}\n{gap_status}',
           transform=ax.transAxes, fontsize=12, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='yellow' if final_gap > 0.5 else 'lightgreen',
                    alpha=0.3))

    plt.tight_layout()

    # Save plot
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Training curves saved to: {output_path}")

    return fig


def main():
    """Main analysis function."""
    parser = argparse.ArgumentParser(description='Analyze training logs and generate report')
    parser.add_argument('csv_file', type=str, help='Path to training log CSV file')
    parser.add_argument('--output', type=str, default=None,
                       help='Output path for plot (default: outputs/training_analysis.png)')
    parser.add_argument('--format', type=str, choices=['full', 'plot', 'summary'],
                       default='full', help='Output format: full (plot+summary), plot only, or summary only')
    parser.add_argument('--summary-file', type=str, default=None,
                       help='Save text summary to file')
    args = parser.parse_args()

    # Determine output path
    if args.output is None:
        # Extract timestamp from CSV filename if possible
        csv_name = Path(args.csv_file).stem
        output_path = f'outputs/training_analysis_{csv_name}.png'
    else:
        output_path = args.output

    print("=" * 80)
    print("TRAINING LOG ANALYZER")
    print("=" * 80)
    print(f"\nInput CSV: {args.csv_file}")
    print(f"Output format: {args.format}")
    print()

    # Check if file exists
    if not os.path.exists(args.csv_file):
        print(f"ERROR: CSV file not found: {args.csv_file}")
        sys.exit(1)

    # Read CSV
    try:
        df = pd.read_csv(args.csv_file)
        print(f"✓ Loaded {len(df)} epochs from CSV")
    except Exception as e:
        print(f"ERROR: Failed to read CSV: {e}")
        sys.exit(1)

    # Validate required columns
    required_cols = ['epoch', 'train_loss', 'val_loss']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"ERROR: Missing required columns: {missing_cols}")
        sys.exit(1)

    # Generate outputs based on format
    if args.format in ['full', 'plot']:
        print("\nGenerating plots...")
        plot_training_curves(df, output_path)

    if args.format in ['full', 'summary']:
        print("\nAnalyzing training metrics...")
        print_summary(df, args.summary_file)

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
