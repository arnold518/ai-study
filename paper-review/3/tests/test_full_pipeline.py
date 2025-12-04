#!/usr/bin/env python
"""
Test the full training pipeline end-to-end.

Usage:
    /home/arnold/venv/bin/python scripts/test_full_pipeline.py
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import torch
from torch.utils.data import DataLoader, Subset

from config.transformer_config import TransformerConfig
from src.data.tokenizer import SentencePieceTokenizer
from src.data.dataset import TranslationDataset, collate_fn
from src.models.transformer.transformer import Transformer
from src.training.optimizer import NoamOptimizer
from src.training.losses import LabelSmoothingLoss


def test_pipeline():
    """Test the complete pipeline with real data."""
    print("=" * 60)
    print("Testing Full Training Pipeline")
    print("=" * 60)
    print()

    # Configuration
    config = TransformerConfig()
    config.d_model = 256  # Small for testing
    config.num_encoder_layers = 2
    config.num_decoder_layers = 2
    config.num_heads = 4
    config.d_ff = 512
    config.device = 'cpu'
    config.batch_size = 4

    # Check if data exists
    ko_model_path = 'data/vocab/ko_spm.model'
    en_model_path = 'data/vocab/en_spm.model'
    train_ko_path = 'data/processed/train.ko'
    train_en_path = 'data/processed/train.en'

    if not all(os.path.exists(p) for p in [ko_model_path, en_model_path, train_ko_path, train_en_path]):
        print("Required files not found!")
        print("Please run:")
        print("  1. /home/arnold/venv/bin/python scripts/download_data.py all")
        print("  2. /home/arnold/venv/bin/python scripts/split_data.py")
        print("  3. /home/arnold/venv/bin/python scripts/train_tokenizer.py")
        return

    # Load tokenizers
    print("Loading tokenizers...")
    ko_tokenizer = SentencePieceTokenizer(ko_model_path)
    en_tokenizer = SentencePieceTokenizer(en_model_path)
    print(f"Korean vocab size: {ko_tokenizer.vocab_size}")
    print(f"English vocab size: {en_tokenizer.vocab_size}")
    print()

    # Create dataset
    print("Loading dataset...")
    dataset = TranslationDataset(
        train_ko_path,
        train_en_path,
        ko_tokenizer,
        en_tokenizer,
        max_len=50
    )

    # Use tiny subset
    subset = Subset(dataset, range(min(20, len(dataset))))
    print(f"Using {len(subset)} samples for testing")
    print()

    # Create dataloader
    dataloader = DataLoader(
        subset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=collate_fn
    )

    # Create model
    print("Initializing model...")
    model = Transformer(
        config,
        src_vocab_size=ko_tokenizer.vocab_size,
        tgt_vocab_size=en_tokenizer.vocab_size
    )
    model.to(config.device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print()

    # Create optimizer and loss
    optimizer = NoamOptimizer(model.parameters(), config.d_model, config.warmup_steps)
    criterion = LabelSmoothingLoss(en_tokenizer.vocab_size, pad_idx=0, smoothing=0.1)

    # Test one batch
    print("Testing one training step...")
    model.train()

    batch = next(iter(dataloader))

    src = batch['src'].to(config.device)
    tgt = batch['tgt'].to(config.device)
    src_mask = batch['src_mask'].to(config.device)
    tgt_mask = batch['tgt_mask'].to(config.device)

    print(f"Source shape: {src.shape}")
    print(f"Target shape: {tgt.shape}")
    print(f"Source mask shape: {src_mask.shape}")
    print(f"Target mask shape: {tgt_mask.shape}")
    print()

    # Prepare inputs
    tgt_input = tgt[:, :-1]
    tgt_output = tgt[:, 1:]
    tgt_input_mask = tgt_mask[:, :, :-1, :-1]

    # Forward pass
    print("Forward pass...")
    logits = model(src, tgt_input, src_mask, tgt_input_mask)
    print(f"Logits shape: {logits.shape}")
    print(f"Logits stats: min={logits.min():.4f}, max={logits.max():.4f}, mean={logits.mean():.4f}")

    # Compute loss
    logits_flat = logits.contiguous().view(-1, logits.size(-1))
    targets_flat = tgt_output.contiguous().view(-1)
    loss = criterion(logits_flat, targets_flat)
    print(f"Loss: {loss.item():.4f}")

    # Check for NaN
    if torch.isnan(loss):
        print("ERROR: Loss is NaN!")
        return False

    # Backward pass
    print("\nBackward pass...")
    optimizer.zero_grad()
    loss.backward()

    # Check gradients
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    print(f"Gradient norm: {grad_norm:.4f}")

    if torch.isnan(grad_norm):
        print("ERROR: Gradient norm is NaN!")
        return False

    # Update
    optimizer.step()
    print(f"Learning rate: {optimizer._get_lr():.6f}")

    print()
    print("=" * 60)
    print("✓ Full pipeline test PASSED!")
    print("=" * 60)
    print()

    return True


if __name__ == "__main__":
    success = test_pipeline()
    sys.exit(0 if success else 1)
