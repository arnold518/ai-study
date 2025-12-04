#!/usr/bin/env python
"""Test cached greedy search decoding."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
from src.models.transformer.transformer import Transformer
from src.inference.greedy_search import greedy_decode, greedy_decode_cached
from src.utils.masking import create_padding_mask
from config.transformer_config import TransformerConfig


def test_cached_vs_uncached():
    """Test that cached and uncached greedy decoding give same results."""
    print("=" * 80)
    print("Testing Cached vs Uncached Greedy Decoding")
    print("=" * 80)

    # Small config for testing
    config = TransformerConfig()
    config.d_model = 64
    config.num_heads = 4
    config.num_encoder_layers = 2
    config.num_decoder_layers = 2
    config.d_ff = 128
    config.dropout = 0.0  # Disable dropout for deterministic results

    # Create model
    src_vocab_size = 100
    tgt_vocab_size = 100
    pad_idx = 0
    bos_idx = 2
    eos_idx = 3

    device = torch.device('cpu')
    model = Transformer(config, src_vocab_size, tgt_vocab_size, pad_idx).to(device)
    model.eval()

    print(f"Model parameters: {model.count_parameters():,}")
    print()

    # Create source sequence
    batch_size = 2
    src_len = 8
    torch.manual_seed(42)  # For reproducibility
    src = torch.randint(4, src_vocab_size, (batch_size, src_len), device=device)
    src[:, 0] = bos_idx
    src[0, 5:] = pad_idx  # Add padding to first sequence

    print(f"Source shape: {src.shape}")
    print(f"Source[0]: {src[0].tolist()}")
    print()

    # Create source mask
    src_mask = create_padding_mask(src, pad_idx)

    # Test parameters
    max_length = 15

    # Run uncached greedy decoding
    print("Running UNCACHED greedy decoding...")
    output_uncached = greedy_decode(
        model=model,
        src=src,
        src_mask=src_mask,
        max_length=max_length,
        bos_idx=bos_idx,
        eos_idx=eos_idx,
        device=device
    )
    print(f"Uncached output shape: {output_uncached.shape}")
    print(f"Uncached output[0]: {output_uncached[0].tolist()}")
    print()

    # Run cached greedy decoding
    print("Running CACHED greedy decoding...")
    output_cached = greedy_decode_cached(
        model=model,
        src=src,
        src_mask=src_mask,
        max_length=max_length,
        bos_idx=bos_idx,
        eos_idx=eos_idx,
        device=device
    )
    print(f"Cached output shape: {output_cached.shape}")
    print(f"Cached output[0]: {output_cached[0].tolist()}")
    print()

    # Verify they match
    print("Comparing outputs...")
    assert output_uncached.shape == output_cached.shape, \
        f"Shape mismatch: {output_uncached.shape} vs {output_cached.shape}"

    match = torch.all(output_uncached == output_cached).item()
    if match:
        print("✓ PASS: Cached and uncached outputs are IDENTICAL!")
    else:
        print("✗ FAIL: Outputs differ!")
        print(f"  Uncached: {output_uncached}")
        print(f"  Cached:   {output_cached}")
        # Find first difference
        for i in range(batch_size):
            for j in range(min(output_uncached.shape[1], output_cached.shape[1])):
                if output_uncached[i, j] != output_cached[i, j]:
                    print(f"  First difference at [{i}, {j}]: {output_uncached[i, j]} vs {output_cached[i, j]}")
                    break
        raise AssertionError("Outputs do not match")

    print()


def test_cache_speedup():
    """Measure speedup from caching (informational)."""
    print("=" * 80)
    print("Cache Speedup Test (Informational)")
    print("=" * 80)
    print("Note: For small models, caching overhead may dominate.")
    print("Real speedup is visible with larger models and longer sequences.")
    print()

    torch.manual_seed(123)  # For reproducibility
    config = TransformerConfig()
    config.d_model = 128
    config.num_heads = 8
    config.num_encoder_layers = 3
    config.num_decoder_layers = 3
    config.d_ff = 256
    config.dropout = 0.0

    src_vocab_size = 500
    tgt_vocab_size = 500
    pad_idx = 0
    bos_idx = 2
    eos_idx = 3

    device = torch.device('cpu')
    model = Transformer(config, src_vocab_size, tgt_vocab_size, pad_idx).to(device)
    model.eval()

    # Single longer sequence
    torch.manual_seed(456)  # For input
    src = torch.randint(4, src_vocab_size, (1, 20), device=device)
    src[:, 0] = bos_idx
    src_mask = create_padding_mask(src, pad_idx)

    max_length = 30

    # Time uncached
    import time
    start = time.time()
    output_uncached = greedy_decode(model, src, src_mask, max_length, bos_idx, eos_idx, device)
    time_uncached = time.time() - start

    # Time cached
    start = time.time()
    output_cached = greedy_decode_cached(model, src, src_mask, max_length, bos_idx, eos_idx, device)
    time_cached = time.time() - start

    print(f"Uncached time: {time_uncached:.4f}s")
    print(f"Cached time:   {time_cached:.4f}s")
    if time_cached > 0:
        speedup = time_uncached / time_cached
        print(f"Speedup:       {speedup:.2f}x")
    print()

    # Verify still correct
    assert torch.all(output_uncached == output_cached).item(), "Outputs must match"
    print("✓ Outputs still match with caching")
    print()


if __name__ == "__main__":
    test_cached_vs_uncached()
    test_cache_speedup()

    print("=" * 80)
    print("All cached greedy search tests passed!")
    print("=" * 80)
