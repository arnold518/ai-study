#!/usr/bin/env python
"""
Quick overfitting test on small dataset subset.

Tests if regularization fixes (dropout=0.3, weight_decay, etc.) prevent overfitting.
Trains on small subset and plots train/val curves to visualize overfitting behavior.

Usage:
    /home/arnold/venv/bin/python scripts/test_overfitting.py
    /home/arnold/venv/bin/python scripts/test_overfitting.py --train-size 2000 --epochs 30
    /home/arnold/venv/bin/python scripts/test_overfitting.py --old-config  # Test old config
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import argparse
import torch
import matplotlib.pyplot as plt
import pandas as pd
from torch.utils.data import DataLoader, Subset
import time
import math

from config.transformer_config import TransformerConfig
from src.data.tokenizer import SentencePieceTokenizer
from src.data.dataset import TranslationDataset, collate_fn
from src.models.transformer.transformer import Transformer
from src.training.trainer import Trainer
from src.training.optimizer import NoamOptimizer
from src.training.losses import LabelSmoothingLoss


class OverfittingTestConfig(TransformerConfig):
    """Configuration for overfitting test with small dataset."""

    def __init__(self, use_old_config=False):
        super().__init__()

        if use_old_config:
            # OLD CONFIG (for comparison) - should overfit badly
            self.dropout = 0.1
            self.label_smoothing = 0.1
            self.early_stopping_patience = 8
            self.early_stopping_min_delta = 0.0001
            self.learning_rate = 1.0
            self.warmup_steps = 4000
            self.num_epochs = 20
            print("\n⚠️  Using OLD configuration (dropout=0.1, no weight_decay)")
            print("    This should show severe overfitting!\n")
        else:
            # NEW CONFIG (current) - should resist overfitting
            self.dropout = 0.3
            self.label_smoothing = 0.05
            self.early_stopping_patience = 10
            self.early_stopping_min_delta = 0.001
            self.learning_rate = 2.0
            self.warmup_steps = 8000
            self.num_epochs = 30
            print("\n✓ Using NEW configuration (dropout=0.3, weight_decay=1e-5)")
            print("  This should resist overfitting!\n")

        # Test-specific settings
        self.batch_size = 64  # Smaller batch for faster testing
        self.gradient_accumulation_steps = 1  # No accumulation for speed
        self.eval_every = 1  # Evaluate every epoch
        self.save_every = 5  # Don't save too many checkpoints
        self.bleu_num_samples = 50  # Fewer BLEU samples for speed
        self.inference_num_examples = 1  # Just 1 example for speed

        # Disable expensive features for speed
        self.use_mixed_precision = False  # CPU doesn't support AMP well
        self.monitor_gradients = False

        # Use test checkpoint directory
        self.checkpoint_dir = "checkpoints/overfitting_test"


def plot_training_curves(csv_path, output_path='outputs/overfitting_test.png'):
    """
    Plot training and validation curves to visualize overfitting.

    Args:
        csv_path: Path to training log CSV
        output_path: Where to save the plot
    """
    # Read CSV
    df = pd.read_csv(csv_path)

    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # 1. Loss curves
    ax = axes[0, 0]
    ax.plot(df['epoch'], df['train_loss'], 'o-', label='Train Loss', linewidth=2, markersize=6)
    ax.plot(df['epoch'], df['val_loss'], 's-', label='Val Loss', linewidth=2, markersize=6)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss (KL Divergence)', fontsize=12)
    ax.set_title('Loss Curves', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    # Add shaded region where overfitting occurs
    if len(df) > 1:
        # Find where val_loss starts increasing
        best_val_idx = df['val_loss'].idxmin()
        if best_val_idx < len(df) - 1:
            ax.axvline(df.loc[best_val_idx, 'epoch'], color='red',
                      linestyle='--', alpha=0.5, label='Best Model')
            ax.axvspan(df.loc[best_val_idx, 'epoch'], df['epoch'].max(),
                      alpha=0.1, color='red', label='Overfitting Region')

    # 2. Perplexity curves
    ax = axes[0, 1]
    ax.plot(df['epoch'], df['train_ppl'], 'o-', label='Train PPL', linewidth=2, markersize=6)
    ax.plot(df['epoch'], df['val_ppl'], 's-', label='Val PPL', linewidth=2, markersize=6)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Perplexity', fontsize=12)
    ax.set_title('Perplexity Curves', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0, top=min(50, df['val_ppl'].max() * 1.1))

    # 3. BLEU score
    ax = axes[1, 0]
    if 'val_bleu' in df.columns:
        bleu_data = df[df['val_bleu'].notna()]
        if len(bleu_data) > 0:
            ax.plot(bleu_data['epoch'], bleu_data['val_bleu'], 'o-',
                   color='green', linewidth=2, markersize=8, label='Val BLEU')
            ax.set_xlabel('Epoch', fontsize=12)
            ax.set_ylabel('BLEU Score', fontsize=12)
            ax.set_title('BLEU Score Progress', fontsize=14, fontweight='bold')
            ax.legend(fontsize=11)
            ax.grid(True, alpha=0.3)

            # Highlight best BLEU
            best_bleu_idx = bleu_data['val_bleu'].idxmax()
            best_bleu = bleu_data.loc[best_bleu_idx, 'val_bleu']
            best_bleu_epoch = bleu_data.loc[best_bleu_idx, 'epoch']
            ax.axvline(best_bleu_epoch, color='green', linestyle='--', alpha=0.5)
            ax.text(best_bleu_epoch, best_bleu, f'  Best: {best_bleu:.1f}',
                   fontsize=10, color='green', fontweight='bold')

    # 4. Train/Val Gap (overfitting metric)
    ax = axes[1, 1]
    gap = df['val_loss'] - df['train_loss']
    ax.plot(df['epoch'], gap, 'o-', color='purple', linewidth=2, markersize=6)
    ax.axhline(0, color='black', linestyle='--', alpha=0.3)
    ax.fill_between(df['epoch'], 0, gap, where=(gap > 0),
                    alpha=0.3, color='red', label='Overfitting')
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Val Loss - Train Loss', fontsize=12)
    ax.set_title('Train/Val Gap (Overfitting Indicator)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    # Add text annotation
    final_gap = gap.iloc[-1]
    ax.text(0.05, 0.95, f'Final Gap: {final_gap:.3f}\n{"⚠️ OVERFITTING!" if final_gap > 0.5 else "✓ Good"}',
           transform=ax.transAxes, fontsize=12, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='yellow' if final_gap > 0.5 else 'lightgreen',
                    alpha=0.3))

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Training curves saved to: {output_path}")

    return fig


def print_summary(csv_path):
    """Print summary of overfitting test results."""
    df = pd.read_csv(csv_path)

    print("\n" + "=" * 80)
    print("OVERFITTING TEST RESULTS")
    print("=" * 80)

    # Find best epoch
    best_val_idx = df['val_loss'].idxmin()
    best_epoch = df.loc[best_val_idx, 'epoch']
    best_val_loss = df.loc[best_val_idx, 'val_loss']
    best_train_loss = df.loc[best_val_idx, 'train_loss']

    final_epoch = df['epoch'].iloc[-1]
    final_val_loss = df['val_loss'].iloc[-1]
    final_train_loss = df['train_loss'].iloc[-1]

    print(f"\n📊 Training Summary:")
    print(f"   Total Epochs: {int(final_epoch)}")
    print(f"   Best Epoch:   {int(best_epoch)}")
    print(f"   Final Epoch:  {int(final_epoch)}")

    print(f"\n📈 Best Model (Epoch {int(best_epoch)}):")
    print(f"   Train Loss: {best_train_loss:.4f}")
    print(f"   Val Loss:   {best_val_loss:.4f}")
    print(f"   Gap:        {best_val_loss - best_train_loss:.4f}")

    if 'val_bleu' in df.columns:
        best_bleu = df['val_bleu'].max()
        best_bleu_epoch = df.loc[df['val_bleu'].idxmax(), 'epoch']
        print(f"   BLEU:       {best_bleu:.2f} (epoch {int(best_bleu_epoch)})")

    print(f"\n📉 Final Model (Epoch {int(final_epoch)}):")
    print(f"   Train Loss: {final_train_loss:.4f}")
    print(f"   Val Loss:   {final_val_loss:.4f}")
    print(f"   Gap:        {final_val_loss - final_train_loss:.4f}")

    if 'val_bleu' in df.columns and pd.notna(df['val_bleu'].iloc[-1]):
        final_bleu = df['val_bleu'].iloc[-1]
        print(f"   BLEU:       {final_bleu:.2f}")

    # Degradation analysis
    print(f"\n🔍 Degradation Analysis:")
    val_degradation = final_val_loss - best_val_loss
    gap_increase = (final_val_loss - final_train_loss) - (best_val_loss - best_train_loss)

    print(f"   Val Loss Increase: {val_degradation:+.4f} ({val_degradation/best_val_loss*100:+.1f}%)")
    print(f"   Gap Increase:      {gap_increase:+.4f}")

    if 'val_bleu' in df.columns:
        bleu_data = df[df['val_bleu'].notna()]
        if len(bleu_data) > 0:
            best_bleu = bleu_data['val_bleu'].max()
            final_bleu = bleu_data['val_bleu'].iloc[-1]
            bleu_drop = final_bleu - best_bleu
            print(f"   BLEU Drop:         {bleu_drop:+.2f} points")

    # Verdict
    print(f"\n🎯 Verdict:")
    if val_degradation > 0.3:
        print("   ❌ SEVERE OVERFITTING DETECTED!")
        print("   - Validation loss increased significantly after best epoch")
        print("   - Model memorizing training data instead of generalizing")
    elif val_degradation > 0.1:
        print("   ⚠️  Moderate overfitting detected")
        print("   - Some degradation after best epoch")
        print("   - Consider stronger regularization")
    elif gap_increase > 0.2:
        print("   ⚠️  Train/Val gap widening")
        print("   - Model starting to overfit")
        print("   - Early stopping should trigger soon")
    else:
        print("   ✅ Good generalization!")
        print("   - Minimal overfitting")
        print("   - Regularization working well")

    print("\n" + "=" * 80)


def main():
    """Main testing function."""
    parser = argparse.ArgumentParser(description='Test overfitting on small dataset')
    parser.add_argument('--train-size', type=int, default=1000,
                       help='Number of training samples (default: 1000)')
    parser.add_argument('--val-size', type=int, default=200,
                       help='Number of validation samples (default: 200)')
    parser.add_argument('--epochs', type=int, default=None,
                       help='Number of epochs (default: 30 for new config, 20 for old)')
    parser.add_argument('--old-config', action='store_true',
                       help='Use old configuration (dropout=0.1, for comparison)')
    parser.add_argument('--output-dir', type=str, default='outputs',
                       help='Output directory for plots')
    args = parser.parse_args()

    print("=" * 80)
    print("OVERFITTING TEST ON SMALL DATASET")
    print("=" * 80)
    print(f"\nTest Parameters:")
    print(f"  Train samples: {args.train_size}")
    print(f"  Val samples:   {args.val_size}")
    print(f"  Config:        {'OLD (should overfit)' if args.old_config else 'NEW (regularized)'}")

    # Load configuration
    config = OverfittingTestConfig(use_old_config=args.old_config)
    if args.epochs:
        config.num_epochs = args.epochs

    device = config.device
    print(f"  Device:        {device}")
    print(f"  Epochs:        {config.num_epochs}")
    print(f"  Batch size:    {config.batch_size}")
    print()

    # Load tokenizers
    print("Loading tokenizers...")
    ko_model_path = os.path.join(config.vocab_dir, 'ko_spm.model')
    en_model_path = os.path.join(config.vocab_dir, 'en_spm.model')

    if not os.path.exists(ko_model_path) or not os.path.exists(en_model_path):
        print("ERROR: Tokenizer models not found!")
        print(f"Expected: {ko_model_path}, {en_model_path}")
        print("Please run: /home/arnold/venv/bin/python scripts/train_tokenizer.py")
        return

    ko_tokenizer = SentencePieceTokenizer(ko_model_path)
    en_tokenizer = SentencePieceTokenizer(en_model_path)
    print(f"Korean vocab: {ko_tokenizer.vocab_size}, English vocab: {en_tokenizer.vocab_size}")
    print()

    # Load datasets
    print("Loading datasets...")
    train_ko_path = os.path.join(config.processed_data_dir, 'train.ko')
    train_en_path = os.path.join(config.processed_data_dir, 'train.en')
    val_ko_path = os.path.join(config.processed_data_dir, 'validation.ko')
    val_en_path = os.path.join(config.processed_data_dir, 'validation.en')

    if not all(os.path.exists(p) for p in [train_ko_path, train_en_path, val_ko_path, val_en_path]):
        print("ERROR: Processed data not found!")
        return

    full_train_dataset = TranslationDataset(
        train_ko_path, train_en_path,
        ko_tokenizer, en_tokenizer,
        max_len=config.max_seq_length
    )
    full_val_dataset = TranslationDataset(
        val_ko_path, val_en_path,
        ko_tokenizer, en_tokenizer,
        max_len=config.max_seq_length
    )

    # Create small subsets
    train_dataset = Subset(full_train_dataset, range(min(args.train_size, len(full_train_dataset))))
    val_dataset = Subset(full_val_dataset, range(min(args.val_size, len(full_val_dataset))))

    print(f"Using {len(train_dataset)} training samples (from {len(full_train_dataset)} total)")
    print(f"Using {len(val_dataset)} validation samples (from {len(full_val_dataset)} total)")
    print()

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size,
                             shuffle=True, collate_fn=collate_fn, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size,
                           collate_fn=collate_fn, num_workers=0)

    # Initialize model
    print("Initializing model...")
    model = Transformer(config, src_vocab_size=ko_tokenizer.vocab_size,
                       tgt_vocab_size=en_tokenizer.vocab_size)
    model.to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print()

    # Initialize optimizer and loss
    print("Initializing optimizer and loss...")
    optimizer = NoamOptimizer(model.parameters(), config.d_model,
                             config.warmup_steps, factor=config.learning_rate)
    criterion = LabelSmoothingLoss(en_tokenizer.vocab_size, pad_idx=0,
                                   smoothing=config.label_smoothing)
    criterion.to(device)

    print(f"  Learning rate factor: {config.learning_rate}")
    print(f"  Warmup steps: {config.warmup_steps}")
    print(f"  Dropout: {config.dropout}")
    print(f"  Label smoothing: {config.label_smoothing}")
    print(f"  Early stopping patience: {config.early_stopping_patience}")
    print()

    # Create checkpoint directory
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)

    # Create trainer
    print("Starting training...")
    print("-" * 80)
    trainer = Trainer(model, train_loader, val_loader, optimizer, criterion, config,
                     src_tokenizer=ko_tokenizer, tgt_tokenizer=en_tokenizer,
                     val_dataset=full_val_dataset)

    # Train
    start_time = time.time()
    trainer.train()
    elapsed = time.time() - start_time

    print("-" * 80)
    print(f"Training complete! Time: {elapsed/60:.1f} minutes")
    print()

    # Find the log file
    log_files = sorted([f for f in os.listdir(config.log_dir) if f.startswith('training_log_')])
    if not log_files:
        print("ERROR: No training log found!")
        return

    csv_path = os.path.join(config.log_dir, log_files[-1])
    print(f"Training log: {csv_path}")

    # Plot results
    print("\nGenerating plots...")
    output_path = os.path.join(args.output_dir,
                               'overfitting_test_old.png' if args.old_config else 'overfitting_test_new.png')
    plot_training_curves(csv_path, output_path)

    # Print summary
    print_summary(csv_path)

    print(f"\n✓ Test complete!")
    print(f"\nTo compare OLD vs NEW config, run:")
    print(f"  /home/arnold/venv/bin/python scripts/test_overfitting.py --old-config")


if __name__ == "__main__":
    main()
