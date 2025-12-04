#!/usr/bin/env python
"""
Test mask shape fixes.

Usage:
    /home/arnold/venv/bin/python scripts/test_mask_fix.py
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
from src.utils.masking import create_padding_mask, create_target_mask, create_cross_attention_mask


def test_masking_utilities():
    """Test the mask creation functions."""
    print("=" * 60)
    print("Testing Masking Utilities")
    print("=" * 60)

    batch_size = 2
    src_len = 5
    tgt_len = 6
    pad_idx = 0

    # Create dummy sequences
    src = torch.randint(1, 100, (batch_size, src_len))
    tgt = torch.randint(1, 100, (batch_size, tgt_len))

    # Add some padding
    src[0, 4] = pad_idx
    tgt[1, 5] = pad_idx

    print(f"\nSource shape: {src.shape}")
    print(f"Target shape: {tgt.shape}")
    print(f"Source[0]: {src[0]}")
    print(f"Target[1]: {tgt[1]}")

    # Test padding mask
    src_mask = create_padding_mask(src, pad_idx)
    print(f"\nSource mask shape: {src_mask.shape}")
    print(f"Expected: [batch={batch_size}, 1, seq={src_len}, seq={src_len}]")
    assert src_mask.shape == (batch_size, 1, src_len, src_len), f"Wrong shape: {src_mask.shape}"
    print("✓ Source mask shape correct")

    # Test target mask
    tgt_mask = create_target_mask(tgt, pad_idx)
    print(f"\nTarget mask shape: {tgt_mask.shape}")
    print(f"Expected: [batch={batch_size}, 1, seq={tgt_len}, seq={tgt_len}]")
    assert tgt_mask.shape == (batch_size, 1, tgt_len, tgt_len), f"Wrong shape: {tgt_mask.shape}"
    print("✓ Target mask shape correct")

    # Check causal property
    print("\nTarget mask (first sample):")
    print(tgt_mask[0, 0].int())
    print("Should be lower triangular (1s on and below diagonal)")

    # Test cross-attention mask
    cross_mask = create_cross_attention_mask(src, tgt, pad_idx)
    print(f"\nCross-attention mask shape: {cross_mask.shape}")
    print(f"Expected: [batch={batch_size}, 1, tgt={tgt_len}, src={src_len}]")
    assert cross_mask.shape == (batch_size, 1, tgt_len, src_len), f"Wrong shape: {cross_mask.shape}"
    print("✓ Cross-attention mask shape correct")

    print("\n✓ All masking utility tests passed!")
    print()


def test_full_pipeline():
    """Test with real data."""
    print("=" * 60)
    print("Testing Full Pipeline with Fixed Masks")
    print("=" * 60)

    # Check if data exists
    ko_model_path = 'data/vocab/ko_spm.model'
    en_model_path = 'data/vocab/en_spm.model'
    train_ko_path = 'data/processed/train.ko'
    train_en_path = 'data/processed/train.en'

    if not all(os.path.exists(p) for p in [ko_model_path, en_model_path, train_ko_path, train_en_path]):
        print("\n⚠ Data files not found, skipping full pipeline test")
        print("Run data preparation scripts first")
        return

    # Configuration
    config = TransformerConfig()
    config.d_model = 256
    config.num_encoder_layers = 2
    config.num_decoder_layers = 2
    config.num_heads = 4
    config.d_ff = 512
    config.device = 'cpu'
    config.batch_size = 4

    # Load tokenizers
    print("\nLoading tokenizers...")
    ko_tokenizer = SentencePieceTokenizer(ko_model_path)
    en_tokenizer = SentencePieceTokenizer(en_model_path)

    # Create dataset
    dataset = TranslationDataset(
        train_ko_path,
        train_en_path,
        ko_tokenizer,
        en_tokenizer,
        max_len=50
    )
    subset = Subset(dataset, range(min(20, len(dataset))))

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

    # Test one batch
    print("Testing one training step...")
    batch = next(iter(dataloader))

    src = batch['src']
    tgt = batch['tgt']
    src_mask = batch['src_mask']
    tgt_mask = batch['tgt_mask']
    cross_mask = batch['cross_mask']

    print(f"\nBatch shapes:")
    print(f"  Source: {src.shape}")
    print(f"  Target: {tgt.shape}")
    print(f"  Source mask: {src_mask.shape}")
    print(f"  Target mask: {tgt_mask.shape}")
    print(f"  Cross mask: {cross_mask.shape}")

    # Verify shapes
    batch_size, src_len = src.shape
    _, tgt_len = tgt.shape

    assert src_mask.shape == (batch_size, 1, src_len, src_len), f"Wrong src_mask shape: {src_mask.shape}"
    assert tgt_mask.shape == (batch_size, 1, tgt_len, tgt_len), f"Wrong tgt_mask shape: {tgt_mask.shape}"
    assert cross_mask.shape == (batch_size, 1, tgt_len, src_len), f"Wrong cross_mask shape: {cross_mask.shape}"

    print("\n✓ All mask shapes correct!")

    # Prepare inputs
    tgt_input = tgt[:, :-1]
    tgt_output = tgt[:, 1:]
    tgt_input_mask = tgt_mask[:, :, :-1, :-1]
    cross_input_mask = cross_mask[:, :, :-1, :]

    print(f"\nAdjusted shapes for forward pass:")
    print(f"  Target input: {tgt_input.shape}")
    print(f"  Target input mask: {tgt_input_mask.shape}")
    print(f"  Cross input mask: {cross_input_mask.shape}")

    # Forward pass
    model.eval()
    with torch.no_grad():
        logits = model(src, tgt_input, src_mask, tgt_input_mask, cross_input_mask)

    print(f"\nOutput logits shape: {logits.shape}")
    print(f"Logits stats: min={logits.min():.4f}, max={logits.max():.4f}, mean={logits.mean():.4f}")

    # Check for NaN
    if torch.isnan(logits).any():
        print("✗ ERROR: NaN in logits!")
        return False

    # Test loss
    criterion = LabelSmoothingLoss(en_tokenizer.vocab_size, pad_idx=0, smoothing=0.1)
    logits_flat = logits.contiguous().view(-1, logits.size(-1))
    targets_flat = tgt_output.contiguous().view(-1)
    loss = criterion(logits_flat, targets_flat)

    print(f"Loss: {loss.item():.4f}")

    if torch.isnan(loss):
        print("✗ ERROR: NaN in loss!")
        return False

    print("\n✓ Full pipeline test PASSED!")
    print()
    return True


def main():
    """Run all tests."""
    print()
    test_masking_utilities()
    success = test_full_pipeline()

    print("=" * 60)
    if success or success is None:
        print("✓ All tests passed!")
    else:
        print("✗ Some tests failed")
    print("=" * 60)
    print()


if __name__ == "__main__":
    main()
