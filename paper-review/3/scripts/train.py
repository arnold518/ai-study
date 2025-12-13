#!/usr/bin/env python
"""
Training script for Korean-English Transformer model.

Usage:
    /home/arnold/venv/bin/python scripts/train.py
    /home/arnold/venv/bin/python scripts/train.py --resume checkpoints/checkpoint_epoch_10.pt
    /home/arnold/venv/bin/python scripts/train.py --small  # Use small subset for testing
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import argparse
import torch
from torch.utils.data import DataLoader, Subset

from config.transformer_config import TransformerConfig
from src.data.tokenizer import SentencePieceTokenizer
from src.data.dataset import TranslationDataset, collate_fn
from src.models.transformer.transformer import Transformer
from src.training.trainer import Trainer
from src.training.optimizer import NoamOptimizer
from src.training.losses import LabelSmoothingLoss
from src.utils.checkpointing import load_checkpoint


def main():
    """Main training function."""
    # Parse arguments
    parser = argparse.ArgumentParser(description='Train Transformer model')
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')
    parser.add_argument('--small', action='store_true', help='Use small subset for testing')
    parser.add_argument('--config', type=str, default=None, help='Path to custom config')
    args = parser.parse_args()

    print("=" * 60)
    print("Korean-English Transformer Training")
    print("=" * 60)
    print()

    # Load configuration
    if args.config:
        import importlib.util
        spec = importlib.util.spec_from_file_location("config", args.config)
        config_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(config_module)
        config = config_module.TransformerConfig()
    else:
        config = TransformerConfig()

    device = config.device
    print(f"Device: {device}")
    print(f"Config: d_model={config.d_model}, num_layers={config.num_encoder_layers}, heads={config.num_heads}")
    print(f"Batch size: {config.batch_size}, Warmup steps: {config.warmup_steps}")
    print()

    # Check if tokenizers exist
    ko_model_path = os.path.join(config.vocab_dir, 'ko_spm.model')
    en_model_path = os.path.join(config.vocab_dir, 'en_spm.model')

    if not os.path.exists(ko_model_path) or not os.path.exists(en_model_path):
        print("ERROR: Tokenizer models not found!")
        print(f"Expected files:")
        print(f"  - {ko_model_path}")
        print(f"  - {en_model_path}")
        print()
        print("Please run: /home/arnold/venv/bin/python scripts/train_tokenizer.py")
        return

    # Load tokenizers
    print("Loading tokenizers...")
    ko_tokenizer = SentencePieceTokenizer(ko_model_path)
    en_tokenizer = SentencePieceTokenizer(en_model_path)
    print(f"Korean vocab size: {ko_tokenizer.vocab_size}")
    print(f"English vocab size: {en_tokenizer.vocab_size}")
    print()

    # Create datasets
    print("Loading datasets...")
    train_ko_path = os.path.join(config.processed_data_dir, 'train.ko')
    train_en_path = os.path.join(config.processed_data_dir, 'train.en')
    val_ko_path = os.path.join(config.processed_data_dir, 'validation.ko')
    val_en_path = os.path.join(config.processed_data_dir, 'validation.en')

    if not all(os.path.exists(p) for p in [train_ko_path, train_en_path, val_ko_path, val_en_path]):
        print("ERROR: Processed data files not found!")
        print(f"Expected files:")
        print(f"  - {train_ko_path}")
        print(f"  - {train_en_path}")
        print(f"  - {val_ko_path}")
        print(f"  - {val_en_path}")
        print()
        print("Please run:")
        print("  1. /home/arnold/venv/bin/python scripts/download_data.py all")
        print("  2. /home/arnold/venv/bin/python scripts/split_data.py")
        return

    train_dataset = TranslationDataset(
        train_ko_path,
        train_en_path,
        ko_tokenizer,
        en_tokenizer,
        max_len=config.max_seq_length
    )
    val_dataset = TranslationDataset(
        val_ko_path,
        val_en_path,
        ko_tokenizer,
        en_tokenizer,
        max_len=config.max_seq_length
    )

    # Keep reference to full validation dataset for BLEU computation
    full_val_dataset = val_dataset

    # Use small subset if requested
    if args.small:
        print("Using small subset for testing...")
        train_dataset = Subset(train_dataset, range(min(1000, len(train_dataset))))
        val_dataset = Subset(val_dataset, range(min(100, len(val_dataset))))

    print(f"Train size: {len(train_dataset)}")
    print(f"Val size: {len(val_dataset)}")
    print()

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=config.num_workers if not args.small else 0
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        collate_fn=collate_fn,
        num_workers=config.num_workers if not args.small else 0
    )

    # Initialize model
    print("Initializing model...")
    model = Transformer(
        config,
        src_vocab_size=ko_tokenizer.vocab_size,
        tgt_vocab_size=en_tokenizer.vocab_size
    )
    model.to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print()

    # Initialize optimizer and loss
    print("Initializing optimizer and loss function...")
    optimizer = NoamOptimizer(
        model.parameters(),
        config.d_model,
        config.warmup_steps,
        factor=config.learning_rate
    )
    criterion = LabelSmoothingLoss(
        en_tokenizer.vocab_size,
        pad_idx=0,
        smoothing=config.label_smoothing
    )
    criterion.to(device)
    print()

    # Resume from checkpoint if specified
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        model, optimizer.optimizer, epoch, loss = load_checkpoint(model, optimizer.optimizer, args.resume, device)
        print()

    # Create checkpoint directory
    os.makedirs(config.checkpoint_dir, exist_ok=True)

    # Create trainer and start training
    print("Starting training...")
    trainer = Trainer(
        model, train_loader, val_loader, optimizer, criterion, config,
        src_tokenizer=ko_tokenizer,
        tgt_tokenizer=en_tokenizer,
        val_dataset=full_val_dataset
    )
    trainer.train()

    print()
    print("Training complete!")


if __name__ == "__main__":
    main()
