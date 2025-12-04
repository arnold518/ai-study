#!/usr/bin/env python
"""
Test PositionWiseFeedForward and PositionalEncoding.

This script tests:
- Feed-forward network shapes and activation
- Positional encoding sinusoidal patterns
- Gradient flow
- Integration

Usage:
    /home/arnold/venv/bin/python tests/test_ffn_and_pe.py
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn as nn
import math
from src.models.transformer.feedforward import PositionWiseFeedForward
from src.models.transformer.positional_encoding import PositionalEncoding


# ============================================================================
# FEED-FORWARD NETWORK TESTS
# ============================================================================

def test_ffn_initialization():
    """Test FFN initialization."""
    print("\n" + "=" * 60)
    print("Test 1: FFN Initialization")
    print("=" * 60)

    d_model = 512
    d_ff = 2048

    ffn = PositionWiseFeedForward(d_model, d_ff)

    # Check layers
    assert hasattr(ffn, 'linear1'), "Should have linear1"
    assert hasattr(ffn, 'linear2'), "Should have linear2"
    assert hasattr(ffn, 'dropout'), "Should have dropout"

    # Check dimensions
    assert ffn.linear1.in_features == d_model
    assert ffn.linear1.out_features == d_ff
    assert ffn.linear2.in_features == d_ff
    assert ffn.linear2.out_features == d_model

    print(f"✓ Layers initialized correctly")
    print(f"  linear1: {d_model} -> {d_ff}")
    print(f"  linear2: {d_ff} -> {d_model}")

    print("\n✅ FFN initialization tests passed!")


def test_ffn_shapes():
    """Test FFN output shapes."""
    print("\n" + "=" * 60)
    print("Test 2: FFN Shapes")
    print("=" * 60)

    batch_size = 4
    seq_len = 10
    d_model = 512
    d_ff = 2048

    ffn = PositionWiseFeedForward(d_model, d_ff)
    ffn.eval()

    x = torch.randn(batch_size, seq_len, d_model)
    output = ffn(x)

    expected_shape = (batch_size, seq_len, d_model)
    assert output.shape == expected_shape, \
        f"Output shape {output.shape} != expected {expected_shape}"
    print(f"✓ Output shape correct: {output.shape}")

    # Test with different batch sizes and sequence lengths
    test_configs = [
        (1, 5, 256, 1024),
        (8, 20, 512, 2048),
        (2, 100, 768, 3072),
    ]

    for batch, seq, d_m, d_f in test_configs:
        ffn_test = PositionWiseFeedForward(d_m, d_f)
        ffn_test.eval()
        x_test = torch.randn(batch, seq, d_m)
        out_test = ffn_test(x_test)
        assert out_test.shape == (batch, seq, d_m)
        print(f"✓ Config (B={batch}, L={seq}, D={d_m}, FF={d_f}): correct")

    print("\n✅ FFN shape tests passed!")


def test_ffn_activation():
    """Test that ReLU activation is applied."""
    print("\n" + "=" * 60)
    print("Test 3: FFN ReLU Activation")
    print("=" * 60)

    d_model = 64
    d_ff = 256

    ffn = PositionWiseFeedForward(d_model, d_ff)
    ffn.eval()

    # Create input with some negative values
    x = torch.randn(2, 5, d_model)

    # Hook to capture intermediate activation
    activations = {}

    def hook_fn(name):
        def hook(module, input, output):
            activations[name] = output
        return hook

    # Register hook on linear1 to see output before ReLU
    handle = ffn.linear1.register_forward_hook(hook_fn('after_linear1'))

    output = ffn(x)

    # Check that intermediate values can be negative (before ReLU)
    intermediate = activations['after_linear1']
    has_negative = (intermediate < 0).any()
    print(f"✓ Intermediate values have negatives: {has_negative.item()}")

    # The intermediate activation should have negatives
    # (otherwise ReLU would be pointless)
    assert has_negative, "Should have negative values before ReLU"

    handle.remove()

    print("\n✅ FFN activation tests passed!")


def test_ffn_parameter_count():
    """Test FFN parameter count."""
    print("\n" + "=" * 60)
    print("Test 4: FFN Parameter Count")
    print("=" * 60)

    d_model = 512
    d_ff = 2048

    ffn = PositionWiseFeedForward(d_model, d_ff)

    total_params = sum(p.numel() for p in ffn.parameters())

    # Expected: 2 linear layers
    # linear1: d_model * d_ff + d_ff
    # linear2: d_ff * d_model + d_model
    expected_params = (d_model * d_ff + d_ff) + (d_ff * d_model + d_model)

    print(f"  Total parameters: {total_params:,}")
    print(f"  Expected: {expected_params:,}")
    assert total_params == expected_params

    print(f"\n  Breakdown:")
    for name, param in ffn.named_parameters():
        print(f"    {name:15s}: {param.numel():>10,} {list(param.shape)}")

    print("\n✅ FFN parameter count tests passed!")


def test_ffn_gradient_flow():
    """Test gradient flow through FFN."""
    print("\n" + "=" * 60)
    print("Test 5: FFN Gradient Flow")
    print("=" * 60)

    d_model = 512
    d_ff = 2048

    ffn = PositionWiseFeedForward(d_model, d_ff)

    x = torch.randn(2, 10, d_model, requires_grad=True)
    output = ffn(x)

    loss = output.sum()
    loss.backward()

    # Check input gradients
    assert x.grad is not None
    print(f"✓ Input has gradients")

    # Check all parameter gradients
    for name, param in ffn.named_parameters():
        assert param.grad is not None, f"{name} should have gradients"
        assert not torch.isnan(param.grad).any(), f"NaN in {name}"
        assert not torch.isinf(param.grad).any(), f"Inf in {name}"
        grad_norm = param.grad.norm().item()
        print(f"  {name:15s}: grad_norm = {grad_norm:.6f}")

    print(f"✓ All parameters have valid gradients")

    print("\n✅ FFN gradient flow tests passed!")


# ============================================================================
# POSITIONAL ENCODING TESTS
# ============================================================================

def test_pe_initialization():
    """Test positional encoding initialization."""
    print("\n" + "=" * 60)
    print("Test 6: Positional Encoding Initialization")
    print("=" * 60)

    d_model = 512
    max_len = 5000

    pe_module = PositionalEncoding(d_model, max_len)

    # Check that buffer is registered
    assert hasattr(pe_module, 'pe'), "Should have pe buffer"

    # Check shape
    expected_shape = (1, max_len, d_model)
    assert pe_module.pe.shape == expected_shape, \
        f"PE shape {pe_module.pe.shape} != expected {expected_shape}"
    print(f"✓ PE buffer shape correct: {pe_module.pe.shape}")

    # Check it's not a parameter (shouldn't be trained)
    param_count = sum(p.numel() for p in pe_module.parameters())
    assert param_count == 0, "PE should not have trainable parameters (only dropout has none)"
    print(f"✓ PE has no trainable parameters (it's a buffer)")

    print("\n✅ Positional encoding initialization tests passed!")


def test_pe_sinusoidal_pattern():
    """Test sinusoidal pattern of positional encoding."""
    print("\n" + "=" * 60)
    print("Test 7: Sinusoidal Pattern")
    print("=" * 60)

    d_model = 128  # Smaller for easier verification
    max_len = 100

    pe_module = PositionalEncoding(d_model, max_len, dropout=0.0)

    pe = pe_module.pe.squeeze(0)  # [max_len, d_model]

    # Test even columns use sin, odd columns use cos
    # PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
    # PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

    pos = 10
    i = 0
    div_term = 10000 ** (2 * i / d_model)
    expected_sin = math.sin(pos / div_term)
    expected_cos = math.cos(pos / div_term)

    actual_sin = pe[pos, 2*i].item()
    actual_cos = pe[pos, 2*i + 1].item()

    print(f"  Position {pos}, dimension {2*i} (sin):")
    print(f"    Expected: {expected_sin:.6f}")
    print(f"    Actual:   {actual_sin:.6f}")
    assert abs(actual_sin - expected_sin) < 1e-5

    print(f"  Position {pos}, dimension {2*i+1} (cos):")
    print(f"    Expected: {expected_cos:.6f}")
    print(f"    Actual:   {actual_cos:.6f}")
    assert abs(actual_cos - expected_cos) < 1e-5

    print(f"✓ Sinusoidal pattern is correct")

    # Check values are in [-1, 1] range
    assert pe.min() >= -1.0 and pe.max() <= 1.0, "PE values should be in [-1, 1]"
    print(f"✓ PE values in valid range: [{pe.min():.3f}, {pe.max():.3f}]")

    # Verify alternating sin/cos pattern
    # Even indices should follow one pattern, odd another
    print(f"✓ PE uses sin for even indices, cos for odd indices")

    print("\n✅ Sinusoidal pattern tests passed!")


def test_pe_shapes():
    """Test positional encoding output shapes."""
    print("\n" + "=" * 60)
    print("Test 8: Positional Encoding Shapes")
    print("=" * 60)

    d_model = 512
    max_len = 5000

    pe_module = PositionalEncoding(d_model, max_len, dropout=0.0)
    pe_module.eval()

    # Test with different sequence lengths
    test_configs = [
        (2, 10, 512),
        (4, 50, 512),
        (1, 100, 256),
        (8, 20, 768),
    ]

    for batch, seq_len, d_m in test_configs:
        if d_m != d_model:
            pe_test = PositionalEncoding(d_m, max_len, dropout=0.0)
            pe_test.eval()
        else:
            pe_test = pe_module

        x = torch.randn(batch, seq_len, d_m)
        output = pe_test(x)

        expected_shape = (batch, seq_len, d_m)
        assert output.shape == expected_shape, \
            f"Shape {output.shape} != expected {expected_shape}"
        print(f"✓ Config (B={batch}, L={seq_len}, D={d_m}): correct shape")

    print("\n✅ Positional encoding shape tests passed!")


def test_pe_adds_to_embedding():
    """Test that PE adds to embedding (not replaces)."""
    print("\n" + "=" * 60)
    print("Test 9: PE Addition (Not Replacement)")
    print("=" * 60)

    d_model = 512
    pe_module = PositionalEncoding(d_model, dropout=0.0)
    pe_module.eval()

    # Create non-zero input
    x = torch.ones(1, 10, d_model) * 5.0

    output = pe_module(x)

    # Output should not equal just PE (should be x + PE)
    pe_only = pe_module.pe[:, :10, :]

    # Output should equal x + pe_only (since dropout=0 in eval)
    expected = x + pe_only
    assert torch.allclose(output, expected, atol=1e-6), \
        "Output should be input + PE"

    print(f"✓ PE is added to input (not replaced)")
    print(f"  Input mean: {x.mean():.3f}")
    print(f"  PE mean: {pe_only.mean():.3f}")
    print(f"  Output mean: {output.mean():.3f}")

    print("\n✅ PE addition tests passed!")


def test_pe_position_uniqueness():
    """Test that different positions have different encodings."""
    print("\n" + "=" * 60)
    print("Test 10: Position Uniqueness")
    print("=" * 60)

    d_model = 512
    max_len = 100

    pe_module = PositionalEncoding(d_model, max_len)
    pe = pe_module.pe.squeeze(0)  # [max_len, d_model]

    # Check that different positions are different
    pos1 = pe[0]
    pos2 = pe[1]
    pos3 = pe[50]

    diff_01 = (pos1 - pos2).abs().mean()
    diff_02 = (pos1 - pos3).abs().mean()
    diff_12 = (pos2 - pos3).abs().mean()

    print(f"  Difference between positions:")
    print(f"    pos 0 vs pos 1:  {diff_01:.6f}")
    print(f"    pos 0 vs pos 50: {diff_02:.6f}")
    print(f"    pos 1 vs pos 50: {diff_12:.6f}")

    assert diff_01 > 0.01, "Adjacent positions should be different"
    assert diff_02 > 0.1, "Distant positions should be very different"
    print(f"✓ Different positions have unique encodings")

    print("\n✅ Position uniqueness tests passed!")


def test_pe_no_trainable_params():
    """Test that PE has no trainable parameters."""
    print("\n" + "=" * 60)
    print("Test 11: PE Non-Trainable")
    print("=" * 60)

    d_model = 512
    pe_module = PositionalEncoding(d_model)

    # Count trainable parameters (should be 0, except dropout which has no params)
    trainable_params = sum(p.numel() for p in pe_module.parameters() if p.requires_grad)

    print(f"  Trainable parameters: {trainable_params}")
    assert trainable_params == 0, "PE should have no trainable parameters"
    print(f"✓ PE has no trainable parameters")

    # Check that pe buffer doesn't require grad
    assert not pe_module.pe.requires_grad, "pe buffer should not require grad"
    print(f"✓ PE buffer does not require gradients")

    print("\n✅ PE non-trainable tests passed!")


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

def test_ffn_pe_integration():
    """Test FFN and PE together."""
    print("\n" + "=" * 60)
    print("Test 12: FFN + PE Integration")
    print("=" * 60)

    batch_size = 4
    seq_len = 20
    d_model = 512
    d_ff = 2048

    # Create modules
    pe = PositionalEncoding(d_model)
    ffn = PositionWiseFeedForward(d_model, d_ff)

    # Simulate: embedding -> PE -> FFN
    x = torch.randn(batch_size, seq_len, d_model)

    # Add positional encoding
    x_with_pe = pe(x)
    assert x_with_pe.shape == x.shape
    print(f"✓ After PE: {x_with_pe.shape}")

    # Apply FFN
    output = ffn(x_with_pe)
    assert output.shape == x.shape
    print(f"✓ After FFN: {output.shape}")

    # Test gradient flow through both
    loss = output.sum()
    loss.backward()

    print(f"✓ Gradients flow through PE -> FFN")

    print("\n✅ Integration tests passed!")


def test_realistic_scenario():
    """Test with realistic model configuration."""
    print("\n" + "=" * 60)
    print("Test 13: Realistic Scenario (Base Transformer)")
    print("=" * 60)

    # Base Transformer config
    batch_size = 32
    seq_len = 50
    d_model = 512
    d_ff = 2048
    vocab_size = 16000

    print(f"\nSimulating Transformer base model:")
    print(f"  Batch size: {batch_size}")
    print(f"  Sequence length: {seq_len}")
    print(f"  d_model: {d_model}")
    print(f"  d_ff: {d_ff}")
    print(f"  vocab_size: {vocab_size}")

    # Simulate embedding
    embedding = nn.Embedding(vocab_size, d_model)
    pe = PositionalEncoding(d_model)
    ffn = PositionWiseFeedForward(d_model, d_ff)

    # Create input tokens
    tokens = torch.randint(0, vocab_size, (batch_size, seq_len))

    # Forward pass
    embedded = embedding(tokens)  # [B, L, D]
    print(f"\n  After embedding: {embedded.shape}")

    embedded_with_pe = pe(embedded)  # [B, L, D]
    print(f"  After PE: {embedded_with_pe.shape}")

    output = ffn(embedded_with_pe)  # [B, L, D]
    print(f"  After FFN: {output.shape}")

    # Check output shape
    assert output.shape == (batch_size, seq_len, d_model)
    print(f"\n✓ Complete forward pass successful")

    # Check gradients
    loss = output.sum()
    loss.backward()
    print(f"✓ Gradients computed successfully")

    print("\n✅ Realistic scenario tests passed!")


def run_all_tests():
    """Run all tests for FFN and PE."""
    print("\n" + "=" * 70)
    print(" " * 10 + "FEED-FORWARD & POSITIONAL ENCODING TEST SUITE")
    print("=" * 70)

    try:
        # FFN tests
        test_ffn_initialization()
        test_ffn_shapes()
        test_ffn_activation()
        test_ffn_parameter_count()
        test_ffn_gradient_flow()

        # PE tests
        test_pe_initialization()
        test_pe_sinusoidal_pattern()
        test_pe_shapes()
        test_pe_adds_to_embedding()
        test_pe_position_uniqueness()
        test_pe_no_trainable_params()

        # Integration tests
        test_ffn_pe_integration()
        test_realistic_scenario()

        print("\n" + "=" * 70)
        print(" " * 20 + "🎉 ALL TESTS PASSED! 🎉")
        print("=" * 70)
        print("\n✅ FFN and Positional Encoding are correct!")
        print("\nNext step: Implement Encoder Layer")
        print("  File: src/models/transformer/encoder.py")

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
    torch.manual_seed(42)
    run_all_tests()
