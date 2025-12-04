#!/usr/bin/env python
"""Test greedy search decoding."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
from src.models.transformer.transformer import Transformer
from src.inference.greedy_search import greedy_decode
from src.utils.masking import create_padding_mask
from config.transformer_config import TransformerConfig


def test_greedy_decode():
    """Test basic greedy decoding."""
    print("=" * 80)
    print("Testing Greedy Decoding (No Cache)")
    print("=" * 80)

    # Small config for testing
    config = TransformerConfig()
    config.d_model = 64
    config.num_heads = 4
    config.num_encoder_layers = 2
    config.num_decoder_layers = 2
    config.d_ff = 128
    config.dropout = 0.1

    # Create model with small vocab
    src_vocab_size = 100
    tgt_vocab_size = 100
    pad_idx = 0
    bos_idx = 2  # SentencePiece BOS
    eos_idx = 3  # SentencePiece EOS

    device = torch.device('cpu')
    model = Transformer(config, src_vocab_size, tgt_vocab_size, pad_idx).to(device)
    model.eval()

    print(f"Model parameters: {model.count_parameters():,}")
    print()

    # Create a dummy source sequence
    batch_size = 2
    src_len = 10
    src = torch.randint(4, src_vocab_size, (batch_size, src_len), device=device)
    src[:, 0] = bos_idx  # Start with BOS
    src[0, 5:] = pad_idx  # Add some padding to first sequence

    print(f"Source shape: {src.shape}")
    print(f"Source (first): {src[0].tolist()}")
    print()

    # Create source mask
    src_mask = create_padding_mask(src, pad_idx)
    print(f"Source mask shape: {src_mask.shape}")
    print()

    # Run greedy decoding
    max_length = 20
    print(f"Generating with greedy search (max_length={max_length})...")

    output = greedy_decode(
        model=model,
        src=src,
        src_mask=src_mask,
        max_length=max_length,
        bos_idx=bos_idx,
        eos_idx=eos_idx,
        device=device
    )

    print(f"\nOutput shape: {output.shape}")
    print(f"Output (first): {output[0].tolist()}")
    print(f"Output (second): {output[1].tolist()}")
    print()

    # Verify output
    assert output.shape[0] == batch_size, "Batch size mismatch"
    assert output.shape[1] <= max_length + 1, "Output too long (max_length exceeded)"
    assert (output[:, 0] == bos_idx).all(), "Output should start with BOS"

    # Check if EOS was generated
    has_eos = (output == eos_idx).any(dim=1)
    print(f"Sequences finished with EOS: {has_eos.tolist()}")
    print()

    print("✓ Greedy decoding test passed!")
    print()


def test_greedy_decode_early_stop():
    """Test that decoding stops at EOS."""
    print("=" * 80)
    print("Testing Early Stopping at EOS")
    print("=" * 80)

    config = TransformerConfig()
    config.d_model = 64
    config.num_heads = 4
    config.num_encoder_layers = 1
    config.num_decoder_layers = 1
    config.d_ff = 128
    config.dropout = 0.0

    src_vocab_size = 50
    tgt_vocab_size = 50
    pad_idx = 0
    bos_idx = 2
    eos_idx = 3

    device = torch.device('cpu')
    model = Transformer(config, src_vocab_size, tgt_vocab_size, pad_idx).to(device)
    model.eval()

    # Single source sequence
    src = torch.randint(4, src_vocab_size, (1, 5), device=device)
    src[:, 0] = bos_idx
    src_mask = create_padding_mask(src, pad_idx)

    # Decode with large max_length
    max_length = 50
    output = greedy_decode(
        model=model,
        src=src,
        src_mask=src_mask,
        max_length=max_length,
        bos_idx=bos_idx,
        eos_idx=eos_idx,
        device=device
    )

    print(f"Max length: {max_length}")
    print(f"Actual output length: {output.shape[1]}")
    print(f"Output: {output[0].tolist()}")

    # Check if stopped early due to EOS
    if output.shape[1] < max_length + 1:
        print(f"✓ Stopped early at length {output.shape[1]} (< {max_length + 1})")
        assert (output == eos_idx).any(), "Should contain EOS if stopped early"
    else:
        print(f"Generated full max_length sequence")

    print()


if __name__ == "__main__":
    test_greedy_decode()
    test_greedy_decode_early_stop()

    print("=" * 80)
    print("All greedy search tests passed!")
    print("=" * 80)
