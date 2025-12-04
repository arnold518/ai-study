#!/usr/bin/env python
"""
Test MultiHeadAttention implementation.

This script tests the attention mechanism with various scenarios:
- Shape correctness
- Self-attention
- Cross-attention
- Masking (padding and causal)
- Gradient flow

Usage:
    /home/arnold/venv/bin/python tests/test_attention.py
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn as nn
import math
from src.models.transformer.attention import MultiHeadAttention


def test_initialization():
    """Test MultiHeadAttention initialization."""
    print("\n" + "=" * 60)
    print("Test 1: Initialization")
    print("=" * 60)

    d_model = 512
    num_heads = 8

    # Test valid configuration
    attn = MultiHeadAttention(d_model, num_heads)
    assert attn.d_model == d_model
    assert attn.num_heads == num_heads
    assert attn.d_k == d_model // num_heads
    assert attn.d_v == d_model // num_heads
    print(f"✓ Valid configuration works (d_model={d_model}, num_heads={num_heads})")

    # Test invalid configuration (d_model not divisible by num_heads)
    try:
        invalid_attn = MultiHeadAttention(d_model=100, num_heads=7)
        print("✗ Should have raised assertion error for invalid config")
    except AssertionError as e:
        print(f"✓ Invalid configuration properly rejected")

    print("\n✅ Initialization tests passed!")


def test_self_attention_shapes():
    """Test self-attention output shapes."""
    print("\n" + "=" * 60)
    print("Test 2: Self-Attention Shapes")
    print("=" * 60)

    batch_size = 2
    seq_len = 10
    d_model = 512
    num_heads = 8

    attn = MultiHeadAttention(d_model, num_heads)
    attn.eval()  # Disable dropout for testing

    # Create input
    x = torch.randn(batch_size, seq_len, d_model)

    # Self-attention: Q = K = V
    output, attn_weights = attn(x, x, x)

    # Check output shape
    expected_output_shape = (batch_size, seq_len, d_model)
    assert output.shape == expected_output_shape, \
        f"Output shape {output.shape} != expected {expected_output_shape}"
    print(f"✓ Output shape correct: {output.shape}")

    # Check attention weights shape
    expected_attn_shape = (batch_size, num_heads, seq_len, seq_len)
    assert attn_weights.shape == expected_attn_shape, \
        f"Attention weights shape {attn_weights.shape} != expected {expected_attn_shape}"
    print(f"✓ Attention weights shape correct: {attn_weights.shape}")

    # Attention weights should sum to 1 over last dimension
    attn_sum = attn_weights.sum(dim=-1)
    assert torch.allclose(attn_sum, torch.ones_like(attn_sum), atol=1e-5), \
        "Attention weights should sum to 1"
    print(f"✓ Attention weights sum to 1 (softmax is correct)")

    print("\n✅ Self-attention shape tests passed!")


def test_cross_attention_shapes():
    """Test cross-attention with different sequence lengths."""
    print("\n" + "=" * 60)
    print("Test 3: Cross-Attention Shapes")
    print("=" * 60)

    batch_size = 2
    src_len = 15
    tgt_len = 10
    d_model = 512
    num_heads = 8

    attn = MultiHeadAttention(d_model, num_heads)
    attn.eval()

    # Query from target, Key and Value from source
    query = torch.randn(batch_size, tgt_len, d_model)
    key = torch.randn(batch_size, src_len, d_model)
    value = torch.randn(batch_size, src_len, d_model)

    output, attn_weights = attn(query, key, value)

    # Output should match query sequence length
    expected_output_shape = (batch_size, tgt_len, d_model)
    assert output.shape == expected_output_shape, \
        f"Output shape {output.shape} != expected {expected_output_shape}"
    print(f"✓ Output shape correct: {output.shape}")

    # Attention weights: [batch, heads, tgt_len, src_len]
    expected_attn_shape = (batch_size, num_heads, tgt_len, src_len)
    assert attn_weights.shape == expected_attn_shape, \
        f"Attention weights shape {attn_weights.shape} != expected {expected_attn_shape}"
    print(f"✓ Attention weights shape correct: {attn_weights.shape}")

    print("\n✅ Cross-attention shape tests passed!")


def test_padding_mask():
    """Test attention with padding mask."""
    print("\n" + "=" * 60)
    print("Test 4: Padding Mask")
    print("=" * 60)

    batch_size = 2
    seq_len = 10
    d_model = 64  # Smaller for easier inspection
    num_heads = 4

    attn = MultiHeadAttention(d_model, num_heads)
    attn.eval()

    x = torch.randn(batch_size, seq_len, d_model)

    # Create padding mask: first batch has 7 real tokens, second has 5
    # mask shape: [batch_size, 1, seq_len, seq_len]
    mask = torch.ones(batch_size, 1, seq_len, seq_len)
    mask[0, :, :, 7:] = 0  # Mask positions 7-9 for first batch
    mask[1, :, :, 5:] = 0  # Mask positions 5-9 for second batch

    output, attn_weights = attn(x, x, x, mask)

    # Check that masked positions have near-zero attention
    # First batch: positions 7-9 should have ~0 attention
    masked_attention_batch0 = attn_weights[0, :, :, 7:].abs().max()
    print(f"  Max attention to masked positions (batch 0): {masked_attention_batch0:.6f}")
    assert masked_attention_batch0 < 1e-5, "Masked positions should have near-zero attention"
    print(f"✓ Masked positions correctly ignored (batch 0)")

    # Second batch: positions 5-9 should have ~0 attention
    masked_attention_batch1 = attn_weights[1, :, :, 5:].abs().max()
    print(f"  Max attention to masked positions (batch 1): {masked_attention_batch1:.6f}")
    assert masked_attention_batch1 < 1e-5, "Masked positions should have near-zero attention"
    print(f"✓ Masked positions correctly ignored (batch 1)")

    # Unmasked positions should still sum to 1
    attn_sum = attn_weights.sum(dim=-1)
    assert torch.allclose(attn_sum, torch.ones_like(attn_sum), atol=1e-5), \
        "Attention weights should still sum to 1 with masking"
    print(f"✓ Attention weights still sum to 1 with masking")

    print("\n✅ Padding mask tests passed!")


def test_causal_mask():
    """Test attention with causal (look-ahead) mask."""
    print("\n" + "=" * 60)
    print("Test 5: Causal Mask (Look-Ahead)")
    print("=" * 60)

    batch_size = 1
    seq_len = 5
    d_model = 64
    num_heads = 2

    attn = MultiHeadAttention(d_model, num_heads)
    attn.eval()

    x = torch.randn(batch_size, seq_len, d_model)

    # Create causal mask: lower triangular (can only attend to past)
    # mask[i,j] = 1 if i >= j (can attend), 0 otherwise
    mask = torch.tril(torch.ones(seq_len, seq_len))
    mask = mask.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, seq_len]

    output, attn_weights = attn(x, x, x, mask)

    # Check that upper triangle has near-zero attention
    for i in range(seq_len):
        for j in range(i + 1, seq_len):
            future_attention = attn_weights[0, :, i, j].abs().max()
            assert future_attention < 1e-5, \
                f"Position {i} should not attend to future position {j}"

    print(f"✓ Causal masking prevents attending to future positions")

    # Verify attention pattern
    print(f"\nAttention pattern (head 0, batch 0):")
    print(f"{'':>5}", end='')
    for j in range(seq_len):
        print(f"  pos{j}", end='')
    print()

    for i in range(seq_len):
        print(f"pos{i}:", end='')
        for j in range(seq_len):
            val = attn_weights[0, 0, i, j].item()
            if val < 1e-5:
                print(f"  {0:.3f}", end='')
            else:
                print(f"  {val:.3f}", end='')
        print()

    print("\n✅ Causal mask tests passed!")


def test_multiple_heads():
    """Test that different heads learn different attention patterns."""
    print("\n" + "=" * 60)
    print("Test 6: Multiple Heads")
    print("=" * 60)

    batch_size = 1
    seq_len = 8
    d_model = 512
    num_heads = 8

    attn = MultiHeadAttention(d_model, num_heads)
    attn.eval()

    x = torch.randn(batch_size, seq_len, d_model)
    output, attn_weights = attn(x, x, x)

    # Check that we have the right number of heads
    assert attn_weights.shape[1] == num_heads, \
        f"Should have {num_heads} attention heads"
    print(f"✓ Correct number of heads: {num_heads}")

    # Check d_k per head
    assert attn.d_k == d_model // num_heads, \
        f"d_k should be {d_model // num_heads}"
    print(f"✓ Correct d_k per head: {attn.d_k}")

    # Each head should have different attention patterns
    # (With random initialization, they should be different)
    head_diffs = []
    for i in range(num_heads - 1):
        diff = (attn_weights[0, i] - attn_weights[0, i+1]).abs().mean()
        head_diffs.append(diff.item())

    avg_diff = sum(head_diffs) / len(head_diffs)
    print(f"✓ Average difference between adjacent heads: {avg_diff:.6f}")
    assert avg_diff > 0.01, "Heads should have different patterns (with random init)"

    print("\n✅ Multiple heads tests passed!")


def test_gradient_flow():
    """Test that gradients flow through attention."""
    print("\n" + "=" * 60)
    print("Test 7: Gradient Flow")
    print("=" * 60)

    batch_size = 2
    seq_len = 10
    d_model = 512
    num_heads = 8

    attn = MultiHeadAttention(d_model, num_heads)

    x = torch.randn(batch_size, seq_len, d_model, requires_grad=True)

    output, _ = attn(x, x, x)

    # Compute a simple loss
    loss = output.sum()
    loss.backward()

    # Check that input has gradients
    assert x.grad is not None, "Input should have gradients"
    print(f"✓ Input has gradients")

    # Check that all parameters have gradients
    for name, param in attn.named_parameters():
        assert param.grad is not None, f"Parameter {name} should have gradients"
    print(f"✓ All parameters have gradients:")
    for name, param in attn.named_parameters():
        grad_norm = param.grad.norm().item()
        print(f"    {name:20s}: grad_norm = {grad_norm:.6f}")

    # Check for NaN or Inf
    for name, param in attn.named_parameters():
        assert not torch.isnan(param.grad).any(), f"NaN in gradients of {name}"
        assert not torch.isinf(param.grad).any(), f"Inf in gradients of {name}"
    print(f"✓ No NaN or Inf in gradients")

    print("\n✅ Gradient flow tests passed!")


def test_determinism():
    """Test that attention is deterministic in eval mode."""
    print("\n" + "=" * 60)
    print("Test 8: Determinism (eval mode)")
    print("=" * 60)

    batch_size = 2
    seq_len = 10
    d_model = 512
    num_heads = 8

    attn = MultiHeadAttention(d_model, num_heads)
    attn.eval()  # Set to eval mode

    x = torch.randn(batch_size, seq_len, d_model)

    # Run twice
    output1, attn1 = attn(x, x, x)
    output2, attn2 = attn(x, x, x)

    # Should be identical in eval mode
    assert torch.allclose(output1, output2), \
        "Outputs should be identical in eval mode"
    print(f"✓ Outputs are deterministic in eval mode")

    assert torch.allclose(attn1, attn2), \
        "Attention weights should be identical in eval mode"
    print(f"✓ Attention weights are deterministic in eval mode")

    print("\n✅ Determinism tests passed!")


def test_parameter_count():
    """Test parameter count."""
    print("\n" + "=" * 60)
    print("Test 9: Parameter Count")
    print("=" * 60)

    d_model = 512
    num_heads = 8

    attn = MultiHeadAttention(d_model, num_heads)

    # Count parameters
    total_params = sum(p.numel() for p in attn.parameters())

    # Expected: 4 linear layers (W_q, W_k, W_v, W_o)
    # Each has d_model * d_model weights + d_model biases
    expected_params = 4 * (d_model * d_model + d_model)

    print(f"  Total parameters: {total_params:,}")
    print(f"  Expected: {expected_params:,}")
    assert total_params == expected_params, \
        f"Parameter count {total_params} != expected {expected_params}"
    print(f"✓ Parameter count is correct")

    # Break down by component
    print(f"\n  Parameter breakdown:")
    for name, param in attn.named_parameters():
        print(f"    {name:20s}: {param.numel():>8,} ({list(param.shape)})")

    print("\n✅ Parameter count tests passed!")


def test_different_model_sizes():
    """Test with different model configurations."""
    print("\n" + "=" * 60)
    print("Test 10: Different Model Sizes")
    print("=" * 60)

    configs = [
        (256, 4),   # Small
        (512, 8),   # Base
        (1024, 16), # Large
    ]

    batch_size = 2
    seq_len = 10

    for d_model, num_heads in configs:
        attn = MultiHeadAttention(d_model, num_heads)
        attn.eval()

        x = torch.randn(batch_size, seq_len, d_model)
        output, attn_weights = attn(x, x, x)

        assert output.shape == (batch_size, seq_len, d_model)
        assert attn_weights.shape == (batch_size, num_heads, seq_len, seq_len)

        print(f"✓ d_model={d_model:4d}, num_heads={num_heads:2d}: shapes correct")

    print("\n✅ Different model size tests passed!")


def run_all_tests():
    """Run all attention tests."""
    print("\n" + "=" * 70)
    print(" " * 15 + "MULTI-HEAD ATTENTION TEST SUITE")
    print("=" * 70)

    try:
        test_initialization()
        test_self_attention_shapes()
        test_cross_attention_shapes()
        test_padding_mask()
        test_causal_mask()
        test_multiple_heads()
        test_gradient_flow()
        test_determinism()
        test_parameter_count()
        test_different_model_sizes()

        print("\n" + "=" * 70)
        print(" " * 20 + "🎉 ALL TESTS PASSED! 🎉")
        print("=" * 70)
        print("\n✅ MultiHeadAttention implementation is correct!")
        print("\nNext step: Implement Feed-Forward Network")
        print("  File: src/models/transformer/feedforward.py")

    except AssertionError as e:
        print("\n" + "=" * 70)
        print(" " * 25 + "❌ TEST FAILED")
        print("=" * 70)
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()

    except Exception as e:
        print("\n" + "=" * 70)
        print(" " * 23 + "❌ ERROR OCCURRED")
        print("=" * 70)
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Set random seed for reproducibility
    torch.manual_seed(42)
    run_all_tests()
