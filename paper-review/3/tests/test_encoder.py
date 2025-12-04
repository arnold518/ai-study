#!/usr/bin/env python
"""
Test Transformer Encoder implementation.

This script tests the encoder with various scenarios:
- EncoderLayer: self-attention + FFN + residual connections + layer norm
- TransformerEncoder: Stack of encoder layers
- Shape correctness
- Masking
- Gradient flow
- Parameter count

Usage:
    /home/arnold/venv/bin/python tests/test_encoder.py
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn as nn
from src.models.transformer.encoder import EncoderLayer, TransformerEncoder
from config.transformer_config import TransformerConfig


def test_encoder_layer_initialization():
    """Test EncoderLayer initialization."""
    print("\n" + "=" * 60)
    print("Test 1: EncoderLayer Initialization")
    print("=" * 60)

    d_model = 512
    num_heads = 8
    d_ff = 2048

    layer = EncoderLayer(d_model, num_heads, d_ff)

    # Check components
    assert hasattr(layer, 'self_attn'), "Should have self_attn"
    assert hasattr(layer, 'ffn'), "Should have ffn"
    assert hasattr(layer, 'norm1'), "Should have norm1"
    assert hasattr(layer, 'norm2'), "Should have norm2"
    print(f"✓ All components initialized")

    # Check layer norm dimensions
    assert layer.norm1.normalized_shape == (d_model,)
    assert layer.norm2.normalized_shape == (d_model,)
    print(f"✓ Layer norms have correct dimensions: {d_model}")

    print("\n✅ EncoderLayer initialization tests passed!")


def test_encoder_layer_shapes():
    """Test EncoderLayer output shapes."""
    print("\n" + "=" * 60)
    print("Test 2: EncoderLayer Shapes")
    print("=" * 60)

    batch_size = 4
    seq_len = 10
    d_model = 512
    num_heads = 8
    d_ff = 2048

    layer = EncoderLayer(d_model, num_heads, d_ff)
    layer.eval()

    x = torch.randn(batch_size, seq_len, d_model)
    output = layer(x)

    # Check output shape
    expected_shape = (batch_size, seq_len, d_model)
    assert output.shape == expected_shape, \
        f"Output shape {output.shape} != expected {expected_shape}"
    print(f"✓ Output shape correct: {output.shape}")

    # Test with different configurations
    configs = [
        (2, 5, 256),    # Small
        (8, 20, 512),   # Medium
        (1, 100, 768),  # Long sequence
    ]

    for batch, seq, dim in configs:
        x = torch.randn(batch, seq, dim)
        layer = EncoderLayer(dim, 8, dim * 4)
        layer.eval()
        output = layer(x)
        assert output.shape == (batch, seq, dim)
        print(f"✓ Config (B={batch}, L={seq}, D={dim}): correct")

    print("\n✅ EncoderLayer shape tests passed!")


def test_encoder_layer_with_mask():
    """Test EncoderLayer with padding mask."""
    print("\n" + "=" * 60)
    print("Test 3: EncoderLayer with Mask")
    print("=" * 60)

    batch_size = 2
    seq_len = 10
    d_model = 256
    num_heads = 4
    d_ff = 1024

    layer = EncoderLayer(d_model, num_heads, d_ff)
    layer.eval()

    x = torch.randn(batch_size, seq_len, d_model)

    # Create padding mask: first batch has 7 real tokens, second has 5
    mask = torch.ones(batch_size, 1, seq_len, seq_len)
    mask[0, :, :, 7:] = 0  # Mask positions 7-9 for first batch
    mask[1, :, :, 5:] = 0  # Mask positions 5-9 for second batch

    output = layer(x, mask)

    # Output should still have correct shape
    assert output.shape == (batch_size, seq_len, d_model)
    print(f"✓ Output shape correct with mask: {output.shape}")

    # Masked positions should have different values than unmasked
    output_no_mask = layer(x)
    assert not torch.allclose(output, output_no_mask), \
        "Output with mask should differ from output without mask"
    print(f"✓ Masking affects output as expected")

    print("\n✅ EncoderLayer mask tests passed!")


def test_residual_connections():
    """Test that residual connections are working."""
    print("\n" + "=" * 60)
    print("Test 4: Residual Connections")
    print("=" * 60)

    batch_size = 2
    seq_len = 10
    d_model = 256

    layer = EncoderLayer(d_model, 8, 1024, dropout=0.0)  # No dropout for testing
    layer.eval()

    x = torch.randn(batch_size, seq_len, d_model)
    output = layer(x)

    # With residual connections, output should not be too different from input
    # (though layer norm makes this harder to test directly)
    # Instead, test that output magnitude is reasonable
    input_norm = x.norm()
    output_norm = output.norm()

    print(f"  Input norm:  {input_norm:.4f}")
    print(f"  Output norm: {output_norm:.4f}")
    print(f"  Ratio: {output_norm / input_norm:.4f}")

    # Output should be in a reasonable range (not exploding or vanishing)
    assert 0.1 < output_norm / input_norm < 10.0, \
        "Output magnitude should be reasonable (residual connections working)"
    print(f"✓ Residual connections preserve reasonable magnitudes")

    print("\n✅ Residual connection tests passed!")


def test_layer_normalization():
    """Test that layer normalization is applied."""
    print("\n" + "=" * 60)
    print("Test 5: Layer Normalization")
    print("=" * 60)

    batch_size = 2
    seq_len = 10
    d_model = 256

    layer = EncoderLayer(d_model, 8, 1024)
    layer.eval()

    x = torch.randn(batch_size, seq_len, d_model)
    output = layer(x)

    # After layer norm, each position should have mean ≈ 0 and std ≈ 1
    # (across d_model dimension)
    mean = output.mean(dim=-1)
    std = output.std(dim=-1, unbiased=False)

    print(f"  Mean across d_model: min={mean.min():.6f}, max={mean.max():.6f}")
    print(f"  Std across d_model:  min={std.min():.6f}, max={std.max():.6f}")

    # Mean should be close to 0, std close to 1
    assert torch.allclose(mean, torch.zeros_like(mean), atol=1e-5), \
        "Layer norm should center to mean ≈ 0"
    assert torch.allclose(std, torch.ones_like(std), atol=1e-5), \
        "Layer norm should scale to std ≈ 1"
    print(f"✓ Layer normalization working correctly")

    print("\n✅ Layer normalization tests passed!")


def test_transformer_encoder_initialization():
    """Test TransformerEncoder initialization."""
    print("\n" + "=" * 60)
    print("Test 6: TransformerEncoder Initialization")
    print("=" * 60)

    config = TransformerConfig()
    encoder = TransformerEncoder(config)

    # Check number of layers
    assert len(encoder.layers) == config.num_encoder_layers
    print(f"✓ Correct number of layers: {config.num_encoder_layers}")

    # Check that all layers are EncoderLayer instances
    for i, layer in enumerate(encoder.layers):
        assert isinstance(layer, EncoderLayer), \
            f"Layer {i} should be EncoderLayer"
    print(f"✓ All layers are EncoderLayer instances")

    print("\n✅ TransformerEncoder initialization tests passed!")


def test_transformer_encoder_shapes():
    """Test TransformerEncoder output shapes."""
    print("\n" + "=" * 60)
    print("Test 7: TransformerEncoder Shapes")
    print("=" * 60)

    config = TransformerConfig()
    encoder = TransformerEncoder(config)
    encoder.eval()

    batch_size = 4
    seq_len = 20
    d_model = config.d_model

    x = torch.randn(batch_size, seq_len, d_model)
    output = encoder(x)

    # Check output shape
    expected_shape = (batch_size, seq_len, d_model)
    assert output.shape == expected_shape, \
        f"Output shape {output.shape} != expected {expected_shape}"
    print(f"✓ Output shape correct: {output.shape}")

    print("\n✅ TransformerEncoder shape tests passed!")


def test_deep_encoder():
    """Test encoder with multiple layers."""
    print("\n" + "=" * 60)
    print("Test 8: Deep Encoder (Multiple Layers)")
    print("=" * 60)

    config = TransformerConfig()
    config.num_encoder_layers = 6
    encoder = TransformerEncoder(config)
    encoder.eval()

    batch_size = 2
    seq_len = 10
    d_model = config.d_model

    x = torch.randn(batch_size, seq_len, d_model)
    output = encoder(x)

    # Check shape
    assert output.shape == (batch_size, seq_len, d_model)
    print(f"✓ 6-layer encoder output shape: {output.shape}")

    # Test with different layer counts
    for num_layers in [1, 2, 4, 8, 12]:
        config.num_encoder_layers = num_layers
        encoder = TransformerEncoder(config)
        encoder.eval()
        output = encoder(x)
        assert output.shape == (batch_size, seq_len, d_model)
        print(f"✓ {num_layers}-layer encoder: correct")

    print("\n✅ Deep encoder tests passed!")


def test_encoder_with_mask():
    """Test TransformerEncoder with mask."""
    print("\n" + "=" * 60)
    print("Test 9: TransformerEncoder with Mask")
    print("=" * 60)

    config = TransformerConfig()
    encoder = TransformerEncoder(config)
    encoder.eval()

    batch_size = 2
    seq_len = 10
    d_model = config.d_model

    x = torch.randn(batch_size, seq_len, d_model)

    # Create padding mask
    mask = torch.ones(batch_size, 1, seq_len, seq_len)
    mask[0, :, :, 7:] = 0
    mask[1, :, :, 5:] = 0

    output = encoder(x, mask)

    # Output shape should be correct
    assert output.shape == (batch_size, seq_len, d_model)
    print(f"✓ Output shape with mask: {output.shape}")

    # Masked output should differ from unmasked
    output_no_mask = encoder(x)
    assert not torch.allclose(output, output_no_mask), \
        "Mask should affect encoder output"
    print(f"✓ Mask propagates through all layers")

    print("\n✅ TransformerEncoder mask tests passed!")


def test_gradient_flow():
    """Test that gradients flow through encoder."""
    print("\n" + "=" * 60)
    print("Test 10: Gradient Flow")
    print("=" * 60)

    config = TransformerConfig()
    encoder = TransformerEncoder(config)

    batch_size = 2
    seq_len = 10
    d_model = config.d_model

    x = torch.randn(batch_size, seq_len, d_model, requires_grad=True)
    output = encoder(x)

    # Compute loss and backprop
    loss = output.sum()
    loss.backward()

    # Check that input has gradients
    assert x.grad is not None, "Input should have gradients"
    print(f"✓ Input has gradients")

    # Check that all parameters have gradients
    params_with_grad = 0
    params_without_grad = 0

    for name, param in encoder.named_parameters():
        if param.grad is not None:
            params_with_grad += 1
            assert not torch.isnan(param.grad).any(), f"NaN in gradients of {name}"
            assert not torch.isinf(param.grad).any(), f"Inf in gradients of {name}"
        else:
            params_without_grad += 1

    print(f"✓ Parameters with gradients: {params_with_grad}")
    print(f"✓ Parameters without gradients: {params_without_grad}")
    assert params_without_grad == 0, "All parameters should have gradients"
    print(f"✓ No NaN or Inf in gradients")

    print("\n✅ Gradient flow tests passed!")


def test_parameter_count():
    """Test encoder parameter count."""
    print("\n" + "=" * 60)
    print("Test 11: Parameter Count")
    print("=" * 60)

    config = TransformerConfig()
    encoder = TransformerEncoder(config)

    # Count parameters
    total_params = sum(p.numel() for p in encoder.parameters())
    print(f"  Total parameters: {total_params:,}")

    # Expected per layer:
    # - MultiHeadAttention: 4 * (d_model * d_model + d_model)
    # - FFN: (d_model * d_ff + d_ff) + (d_ff * d_model + d_model)
    # - LayerNorm: 2 * (d_model + d_model)  [2 norms, each has weight and bias]

    d_model = config.d_model
    d_ff = config.d_ff
    num_layers = config.num_encoder_layers

    attn_params = 4 * (d_model * d_model + d_model)
    ffn_params = (d_model * d_ff + d_ff) + (d_ff * d_model + d_model)
    norm_params = 2 * (d_model + d_model)

    per_layer = attn_params + ffn_params + norm_params
    expected_total = per_layer * num_layers

    print(f"  Expected: {expected_total:,}")
    print(f"  Per layer: {per_layer:,}")
    print(f"    - Attention: {attn_params:,}")
    print(f"    - FFN: {ffn_params:,}")
    print(f"    - LayerNorm: {norm_params:,}")

    assert total_params == expected_total, \
        f"Parameter count {total_params} != expected {expected_total}"
    print(f"✓ Parameter count is correct")

    print("\n✅ Parameter count tests passed!")


def test_determinism():
    """Test that encoder is deterministic in eval mode."""
    print("\n" + "=" * 60)
    print("Test 12: Determinism (eval mode)")
    print("=" * 60)

    config = TransformerConfig()
    encoder = TransformerEncoder(config)
    encoder.eval()

    batch_size = 2
    seq_len = 10
    d_model = config.d_model

    x = torch.randn(batch_size, seq_len, d_model)

    # Run twice
    output1 = encoder(x)
    output2 = encoder(x)

    # Should be identical in eval mode
    assert torch.allclose(output1, output2), \
        "Outputs should be identical in eval mode"
    print(f"✓ Outputs are deterministic in eval mode")

    print("\n✅ Determinism tests passed!")


def test_different_configurations():
    """Test encoder with different configurations."""
    print("\n" + "=" * 60)
    print("Test 13: Different Configurations")
    print("=" * 60)

    configs = [
        (256, 4, 1024, 2),   # Small
        (512, 8, 2048, 6),   # Base
        (768, 12, 3072, 4),  # Medium
    ]

    batch_size = 2
    seq_len = 10

    for d_model, num_heads, d_ff, num_layers in configs:
        config = TransformerConfig()
        config.d_model = d_model
        config.num_heads = num_heads
        config.d_ff = d_ff
        config.num_encoder_layers = num_layers

        encoder = TransformerEncoder(config)
        encoder.eval()

        x = torch.randn(batch_size, seq_len, d_model)
        output = encoder(x)

        assert output.shape == (batch_size, seq_len, d_model)
        print(f"✓ Config (d={d_model}, h={num_heads}, ff={d_ff}, L={num_layers}): correct")

    print("\n✅ Different configuration tests passed!")


def run_all_tests():
    """Run all encoder tests."""
    print("\n" + "=" * 70)
    print(" " * 20 + "TRANSFORMER ENCODER TEST SUITE")
    print("=" * 70)

    try:
        test_encoder_layer_initialization()
        test_encoder_layer_shapes()
        test_encoder_layer_with_mask()
        test_residual_connections()
        test_layer_normalization()
        test_transformer_encoder_initialization()
        test_transformer_encoder_shapes()
        test_deep_encoder()
        test_encoder_with_mask()
        test_gradient_flow()
        test_parameter_count()
        test_determinism()
        test_different_configurations()

        print("\n" + "=" * 70)
        print(" " * 20 + "🎉 ALL TESTS PASSED! 🎉")
        print("=" * 70)
        print("\n✅ Transformer Encoder implementation is correct!")
        print("\nNext step: Implement Decoder Layer and Stack")
        print("  File: src/models/transformer/decoder.py")

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
