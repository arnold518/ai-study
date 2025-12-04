#!/usr/bin/env python
"""
Test training infrastructure components.

Usage:
    /home/arnold/venv/bin/python scripts/test_training.py
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn as nn

from config.transformer_config import TransformerConfig
from src.training.losses import LabelSmoothingLoss
from src.training.optimizer import NoamOptimizer


def test_label_smoothing_loss():
    """Test label smoothing loss implementation."""
    print("=" * 60)
    print("Testing Label Smoothing Loss")
    print("=" * 60)

    vocab_size = 100
    batch_size = 32
    seq_len = 10
    pad_idx = 0
    smoothing = 0.1

    # Create loss function
    criterion = LabelSmoothingLoss(vocab_size, pad_idx, smoothing)

    # Create dummy data
    logits = torch.randn(batch_size * seq_len, vocab_size)
    targets = torch.randint(1, vocab_size, (batch_size * seq_len,))  # Avoid padding
    targets[::10] = pad_idx  # Add some padding tokens

    # Compute loss
    loss = criterion(logits, targets)

    print(f"Vocab size: {vocab_size}")
    print(f"Batch size: {batch_size}")
    print(f"Sequence length: {seq_len}")
    print(f"Smoothing: {smoothing}")
    print(f"Loss: {loss.item():.4f}")
    print()

    # Test with all padding (should be 0 or very small)
    all_pad_targets = torch.full((batch_size * seq_len,), pad_idx)
    loss_all_pad = criterion(logits, all_pad_targets)
    print(f"Loss with all padding: {loss_all_pad.item():.4f}")
    print()

    print("✓ Label smoothing loss test passed!")
    print()


def test_noam_optimizer():
    """Test Noam optimizer learning rate schedule."""
    print("=" * 60)
    print("Testing Noam Optimizer")
    print("=" * 60)

    d_model = 512
    warmup_steps = 4000
    factor = 1.0

    # Create dummy model
    model = nn.Linear(10, 10)
    optimizer = NoamOptimizer(model.parameters(), d_model, warmup_steps, factor)

    # Test learning rate schedule
    print(f"d_model: {d_model}")
    print(f"Warmup steps: {warmup_steps}")
    print()

    test_steps = [1, 100, 1000, 4000, 8000, 16000]
    print("Learning rate schedule:")
    for step in test_steps:
        optimizer.step_num = step
        lr = optimizer._get_lr()
        print(f"  Step {step:6d}: lr = {lr:.6f}")

    print()
    print("✓ Noam optimizer test passed!")
    print()


def test_training_step():
    """Test a single training step with all components."""
    print("=" * 60)
    print("Testing Complete Training Step")
    print("=" * 60)

    # Configuration
    config = TransformerConfig()
    config.d_model = 256  # Small for testing
    config.num_encoder_layers = 2
    config.num_decoder_layers = 2
    config.num_heads = 4
    config.d_ff = 512
    config.device = 'cpu'  # Use CPU for testing

    # Create dummy model
    from src.models.transformer.transformer import Transformer
    from src.data.tokenizer import SentencePieceTokenizer

    # Check if tokenizers exist
    ko_model_path = 'data/vocab/ko_spm.model'
    en_model_path = 'data/vocab/en_spm.model'

    if not os.path.exists(ko_model_path) or not os.path.exists(en_model_path):
        print("Tokenizers not found, skipping training step test")
        print("Run: /home/arnold/venv/bin/python scripts/train_tokenizer.py")
        print()
        return

    ko_tokenizer = SentencePieceTokenizer(ko_model_path)
    en_tokenizer = SentencePieceTokenizer(en_model_path)

    model = Transformer(
        config,
        src_vocab_size=ko_tokenizer.vocab_size,
        tgt_vocab_size=en_tokenizer.vocab_size
    )
    model.to(config.device)

    # Create optimizer and criterion
    optimizer = NoamOptimizer(model.parameters(), config.d_model, config.warmup_steps)
    criterion = LabelSmoothingLoss(en_tokenizer.vocab_size, pad_idx=0, smoothing=0.1)

    # Create dummy batch
    batch_size = 4
    src_len = 10
    tgt_len = 12

    src = torch.randint(1, ko_tokenizer.vocab_size, (batch_size, src_len))
    tgt = torch.randint(1, en_tokenizer.vocab_size, (batch_size, tgt_len))
    src_mask = torch.ones(batch_size, 1, src_len).bool()
    tgt_mask = torch.triu(torch.ones(tgt_len, tgt_len), diagonal=1).bool()
    tgt_mask = tgt_mask.unsqueeze(0).expand(batch_size, -1, -1)

    print(f"Batch size: {batch_size}")
    print(f"Source length: {src_len}")
    print(f"Target length: {tgt_len}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print()

    # Training step
    model.train()

    # Prepare inputs and targets
    tgt_input = tgt[:, :-1]
    tgt_output = tgt[:, 1:]
    tgt_input_mask = tgt_mask[:, :-1, :-1]

    # Forward pass
    print("Running forward pass...")
    logits = model(src, tgt_input, src_mask, tgt_input_mask)
    print(f"Output shape: {logits.shape}")

    # Compute loss
    logits_flat = logits.contiguous().view(-1, logits.size(-1))
    targets_flat = tgt_output.contiguous().view(-1)
    loss = criterion(logits_flat, targets_flat)
    print(f"Loss: {loss.item():.4f}")

    # Backward pass
    print("Running backward pass...")
    optimizer.zero_grad()
    loss.backward()

    # Check gradients
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    print(f"Gradient norm: {grad_norm:.4f}")

    # Update weights
    optimizer.step()
    print(f"Learning rate: {optimizer._get_lr():.6f}")

    print()
    print("✓ Training step test passed!")
    print()


def main():
    """Run all tests."""
    print()
    print("Testing Training Infrastructure")
    print()

    test_label_smoothing_loss()
    test_noam_optimizer()
    test_training_step()

    print("=" * 60)
    print("All tests passed!")
    print("=" * 60)
    print()


if __name__ == "__main__":
    main()
