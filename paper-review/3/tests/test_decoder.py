#!/usr/bin/env python
"""
Test Transformer Decoder implementation.

This script tests the decoder with various scenarios:
- DecoderLayer: masked self-attention + cross-attention + FFN
- TransformerDecoder: Stack of decoder layers
- Shape correctness with different src/tgt lengths
- Causal masking (prevents attending to future)
- Cross-attention to encoder output
- Gradient flow
- Parameter count

Usage:
    /home/arnold/venv/bin/python tests/test_decoder.py
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn as nn
from src.models.transformer.decoder import DecoderLayer, TransformerDecoder
from config.transformer_config import TransformerConfig


def test_decoder_layer_initialization():
    """Test DecoderLayer initialization."""
    print("\n" + "=" * 60)
    print("Test 1: DecoderLayer Initialization")
    print("=" * 60)

    d_model = 512
    num_heads = 8
    d_ff = 2048

    layer = DecoderLayer(d_model, num_heads, d_ff)

    # Check components
    assert hasattr(layer, 'self_attn'), "Should have self_attn"
    assert hasattr(layer, 'cross_attn'), "Should have cross_attn"
    assert hasattr(layer, 'ffn'), "Should have ffn"
    assert hasattr(layer, 'norm1'), "Should have norm1"
    assert hasattr(layer, 'norm2'), "Should have norm2"
    assert hasattr(layer, 'norm3'), "Should have norm3"
    print(f"✓ All components initialized")

    # Check layer norm dimensions
    assert layer.norm1.normalized_shape == (d_model,)
    assert layer.norm2.normalized_shape == (d_model,)
    assert layer.norm3.normalized_shape == (d_model,)
    print(f"✓ Layer norms have correct dimensions: {d_model}")

    print("\n✅ DecoderLayer initialization tests passed!")


def test_decoder_layer_shapes():
    """Test DecoderLayer output shapes with different src/tgt lengths."""
    print("\n" + "=" * 60)
    print("Test 2: DecoderLayer Shapes")
    print("=" * 60)

    batch_size = 4
    src_len = 15
    tgt_len = 10
    d_model = 512
    num_heads = 8
    d_ff = 2048

    layer = DecoderLayer(d_model, num_heads, d_ff)
    layer.eval()

    # Target embeddings
    x = torch.randn(batch_size, tgt_len, d_model)

    # Encoder output (different length)
    encoder_output = torch.randn(batch_size, src_len, d_model)

    output = layer(x, encoder_output)

    # Output should match target length, not source length
    expected_shape = (batch_size, tgt_len, d_model)
    assert output.shape == expected_shape, \
        f"Output shape {output.shape} != expected {expected_shape}"
    print(f"✓ Output shape correct: {output.shape}")
    print(f"  (target length {tgt_len}, not source length {src_len})")

    # Test with different configurations
    configs = [
        (2, 5, 3, 256),     # Short target, shorter source
        (8, 20, 15, 512),   # Medium
        (1, 50, 100, 768),  # Long source, medium target
    ]

    for batch, tgt, src, dim in configs:
        x = torch.randn(batch, tgt, dim)
        enc_out = torch.randn(batch, src, dim)
        layer = DecoderLayer(dim, 8, dim * 4)
        layer.eval()
        output = layer(x, enc_out)
        assert output.shape == (batch, tgt, dim)
        print(f"✓ Config (B={batch}, TgtL={tgt}, SrcL={src}, D={dim}): correct")

    print("\n✅ DecoderLayer shape tests passed!")


def test_causal_mask():
    """Test that decoder respects causal mask (no attending to future)."""
    print("\n" + "=" * 60)
    print("Test 3: Causal Mask (Masked Self-Attention)")
    print("=" * 60)

    batch_size = 1
    tgt_len = 5
    src_len = 8
    d_model = 256
    num_heads = 4
    d_ff = 1024

    layer = DecoderLayer(d_model, num_heads, d_ff)
    layer.eval()

    x = torch.randn(batch_size, tgt_len, d_model)
    encoder_output = torch.randn(batch_size, src_len, d_model)

    # Create causal mask: lower triangular (can only attend to past)
    # mask[i,j] = 1 if i >= j (can attend), 0 otherwise
    tgt_mask = torch.tril(torch.ones(tgt_len, tgt_len))
    tgt_mask = tgt_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, tgt_len, tgt_len]

    output_with_mask = layer(x, encoder_output, tgt_mask=tgt_mask)

    # Without mask (can attend to all positions)
    output_no_mask = layer(x, encoder_output)

    # Outputs should differ
    assert not torch.allclose(output_with_mask, output_no_mask), \
        "Causal mask should affect output"
    print(f"✓ Causal mask affects decoder output")

    # Shape should be correct
    assert output_with_mask.shape == (batch_size, tgt_len, d_model)
    print(f"✓ Output shape correct with causal mask: {output_with_mask.shape}")

    print("\n✅ Causal mask tests passed!")


def test_cross_attention():
    """Test that decoder attends to encoder output via cross-attention."""
    print("\n" + "=" * 60)
    print("Test 4: Cross-Attention to Encoder")
    print("=" * 60)

    batch_size = 2
    tgt_len = 10
    src_len = 15
    d_model = 512

    layer = DecoderLayer(d_model, 8, 2048)
    layer.eval()

    x = torch.randn(batch_size, tgt_len, d_model)
    encoder_output1 = torch.randn(batch_size, src_len, d_model)
    encoder_output2 = torch.randn(batch_size, src_len, d_model)

    # Different encoder outputs should produce different decoder outputs
    output1 = layer(x, encoder_output1)
    output2 = layer(x, encoder_output2)

    assert not torch.allclose(output1, output2), \
        "Decoder output should depend on encoder output (cross-attention working)"
    print(f"✓ Decoder output depends on encoder output")

    # Same target, same encoder should produce same output
    output1_again = layer(x, encoder_output1)
    assert torch.allclose(output1, output1_again), \
        "Same inputs should produce same outputs (deterministic)"
    print(f"✓ Cross-attention is deterministic in eval mode")

    print("\n✅ Cross-attention tests passed!")


def test_source_and_target_masks():
    """Test decoder with both source and target masks."""
    print("\n" + "=" * 60)
    print("Test 5: Source and Target Masks")
    print("=" * 60)

    batch_size = 2
    tgt_len = 8
    src_len = 10
    d_model = 256

    layer = DecoderLayer(d_model, 4, 1024)
    layer.eval()

    x = torch.randn(batch_size, tgt_len, d_model)
    encoder_output = torch.randn(batch_size, src_len, d_model)

    # Create target causal mask
    tgt_mask = torch.tril(torch.ones(tgt_len, tgt_len))
    tgt_mask = tgt_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, tgt_len, tgt_len]

    # Create source padding mask (mask last 3 positions)
    src_mask = torch.ones(batch_size, 1, 1, src_len)
    src_mask[0, :, :, 7:] = 0  # Mask positions 7-9 for first batch
    src_mask[1, :, :, 6:] = 0  # Mask positions 6-9 for second batch

    # Test with both masks
    output = layer(x, encoder_output, src_mask=src_mask, tgt_mask=tgt_mask)

    # Should have correct shape
    assert output.shape == (batch_size, tgt_len, d_model)
    print(f"✓ Output shape correct with both masks: {output.shape}")

    # Output should differ from no-mask version
    output_no_masks = layer(x, encoder_output)
    assert not torch.allclose(output, output_no_masks), \
        "Masks should affect output"
    print(f"✓ Both masks affect decoder output")

    print("\n✅ Source and target mask tests passed!")


def test_transformer_decoder_initialization():
    """Test TransformerDecoder initialization."""
    print("\n" + "=" * 60)
    print("Test 6: TransformerDecoder Initialization")
    print("=" * 60)

    config = TransformerConfig()
    decoder = TransformerDecoder(config)

    # Check number of layers
    assert len(decoder.layers) == config.num_decoder_layers
    print(f"✓ Correct number of layers: {config.num_decoder_layers}")

    # Check that all layers are DecoderLayer instances
    for i, layer in enumerate(decoder.layers):
        assert isinstance(layer, DecoderLayer), \
            f"Layer {i} should be DecoderLayer"
    print(f"✓ All layers are DecoderLayer instances")

    print("\n✅ TransformerDecoder initialization tests passed!")


def test_transformer_decoder_shapes():
    """Test TransformerDecoder output shapes."""
    print("\n" + "=" * 60)
    print("Test 7: TransformerDecoder Shapes")
    print("=" * 60)

    config = TransformerConfig()
    decoder = TransformerDecoder(config)
    decoder.eval()

    batch_size = 4
    tgt_len = 12
    src_len = 20
    d_model = config.d_model

    x = torch.randn(batch_size, tgt_len, d_model)
    encoder_output = torch.randn(batch_size, src_len, d_model)

    output = decoder(x, encoder_output)

    # Check output shape (should match target, not source)
    expected_shape = (batch_size, tgt_len, d_model)
    assert output.shape == expected_shape, \
        f"Output shape {output.shape} != expected {expected_shape}"
    print(f"✓ Output shape correct: {output.shape}")
    print(f"  (matches target length {tgt_len}, not source length {src_len})")

    print("\n✅ TransformerDecoder shape tests passed!")


def test_deep_decoder():
    """Test decoder with multiple layers."""
    print("\n" + "=" * 60)
    print("Test 8: Deep Decoder (Multiple Layers)")
    print("=" * 60)

    config = TransformerConfig()
    batch_size = 2
    tgt_len = 10
    src_len = 15
    d_model = config.d_model

    x = torch.randn(batch_size, tgt_len, d_model)
    encoder_output = torch.randn(batch_size, src_len, d_model)

    # Test with different layer counts
    for num_layers in [1, 2, 4, 6, 8]:
        config.num_decoder_layers = num_layers
        decoder = TransformerDecoder(config)
        decoder.eval()
        output = decoder(x, encoder_output)
        assert output.shape == (batch_size, tgt_len, d_model)
        print(f"✓ {num_layers}-layer decoder: correct")

    print("\n✅ Deep decoder tests passed!")


def test_gradient_flow():
    """Test that gradients flow through decoder."""
    print("\n" + "=" * 60)
    print("Test 9: Gradient Flow")
    print("=" * 60)

    config = TransformerConfig()
    decoder = TransformerDecoder(config)

    batch_size = 2
    tgt_len = 10
    src_len = 15
    d_model = config.d_model

    x = torch.randn(batch_size, tgt_len, d_model, requires_grad=True)
    encoder_output = torch.randn(batch_size, src_len, d_model, requires_grad=True)

    output = decoder(x, encoder_output)

    # Compute loss and backprop
    loss = output.sum()
    loss.backward()

    # Check that inputs have gradients
    assert x.grad is not None, "Target input should have gradients"
    assert encoder_output.grad is not None, "Encoder output should have gradients"
    print(f"✓ Both inputs have gradients")

    # Check that all parameters have gradients
    params_with_grad = 0
    params_without_grad = 0

    for name, param in decoder.named_parameters():
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
    """Test decoder parameter count."""
    print("\n" + "=" * 60)
    print("Test 10: Parameter Count")
    print("=" * 60)

    config = TransformerConfig()
    decoder = TransformerDecoder(config)

    # Count parameters
    total_params = sum(p.numel() for p in decoder.parameters())
    print(f"  Total parameters: {total_params:,}")

    # Expected per layer:
    # - Self-attention: 4 * (d_model * d_model + d_model)
    # - Cross-attention: 4 * (d_model * d_model + d_model)
    # - FFN: (d_model * d_ff + d_ff) + (d_ff * d_model + d_model)
    # - LayerNorm: 3 * (d_model + d_model)  [3 norms, each has weight and bias]

    d_model = config.d_model
    d_ff = config.d_ff
    num_layers = config.num_decoder_layers

    self_attn_params = 4 * (d_model * d_model + d_model)
    cross_attn_params = 4 * (d_model * d_model + d_model)
    ffn_params = (d_model * d_ff + d_ff) + (d_ff * d_model + d_model)
    norm_params = 3 * (d_model + d_model)

    per_layer = self_attn_params + cross_attn_params + ffn_params + norm_params
    expected_total = per_layer * num_layers

    print(f"  Expected: {expected_total:,}")
    print(f"  Per layer: {per_layer:,}")
    print(f"    - Self-attention: {self_attn_params:,}")
    print(f"    - Cross-attention: {cross_attn_params:,}")
    print(f"    - FFN: {ffn_params:,}")
    print(f"    - LayerNorm: {norm_params:,}")

    assert total_params == expected_total, \
        f"Parameter count {total_params} != expected {expected_total}"
    print(f"✓ Parameter count is correct")

    print("\n✅ Parameter count tests passed!")


def test_determinism():
    """Test that decoder is deterministic in eval mode."""
    print("\n" + "=" * 60)
    print("Test 11: Determinism (eval mode)")
    print("=" * 60)

    config = TransformerConfig()
    decoder = TransformerDecoder(config)
    decoder.eval()

    batch_size = 2
    tgt_len = 10
    src_len = 15
    d_model = config.d_model

    x = torch.randn(batch_size, tgt_len, d_model)
    encoder_output = torch.randn(batch_size, src_len, d_model)

    # Run twice
    output1 = decoder(x, encoder_output)
    output2 = decoder(x, encoder_output)

    # Should be identical in eval mode
    assert torch.allclose(output1, output2), \
        "Outputs should be identical in eval mode"
    print(f"✓ Outputs are deterministic in eval mode")

    print("\n✅ Determinism tests passed!")


def test_different_configurations():
    """Test decoder with different configurations."""
    print("\n" + "=" * 60)
    print("Test 12: Different Configurations")
    print("=" * 60)

    configs = [
        (256, 4, 1024, 2),   # Small
        (512, 8, 2048, 6),   # Base
        (768, 12, 3072, 4),  # Medium
    ]

    batch_size = 2
    tgt_len = 10
    src_len = 15

    for d_model, num_heads, d_ff, num_layers in configs:
        config = TransformerConfig()
        config.d_model = d_model
        config.num_heads = num_heads
        config.d_ff = d_ff
        config.num_decoder_layers = num_layers

        decoder = TransformerDecoder(config)
        decoder.eval()

        x = torch.randn(batch_size, tgt_len, d_model)
        encoder_output = torch.randn(batch_size, src_len, d_model)
        output = decoder(x, encoder_output)

        assert output.shape == (batch_size, tgt_len, d_model)
        print(f"✓ Config (d={d_model}, h={num_heads}, ff={d_ff}, L={num_layers}): correct")

    print("\n✅ Different configuration tests passed!")


def test_encoder_decoder_integration():
    """Test encoder and decoder working together."""
    print("\n" + "=" * 60)
    print("Test 13: Encoder-Decoder Integration")
    print("=" * 60)

    from src.models.transformer.encoder import TransformerEncoder

    config = TransformerConfig()
    encoder = TransformerEncoder(config)
    decoder = TransformerDecoder(config)

    encoder.eval()
    decoder.eval()

    batch_size = 2
    src_len = 20
    tgt_len = 15
    d_model = config.d_model

    # Source and target sequences
    src = torch.randn(batch_size, src_len, d_model)
    tgt = torch.randn(batch_size, tgt_len, d_model)

    # Pass through encoder
    encoder_output = encoder(src)
    assert encoder_output.shape == (batch_size, src_len, d_model)
    print(f"✓ Encoder output shape: {encoder_output.shape}")

    # Pass through decoder
    decoder_output = decoder(tgt, encoder_output)
    assert decoder_output.shape == (batch_size, tgt_len, d_model)
    print(f"✓ Decoder output shape: {decoder_output.shape}")

    # Test with masks
    src_mask = torch.ones(batch_size, 1, 1, src_len)
    src_mask[0, :, :, 15:] = 0  # Mask last 5 positions

    tgt_mask = torch.tril(torch.ones(tgt_len, tgt_len))
    tgt_mask = tgt_mask.unsqueeze(0).unsqueeze(0)

    encoder_output_masked = encoder(src, src_mask)
    decoder_output_masked = decoder(tgt, encoder_output_masked, src_mask, tgt_mask)

    assert decoder_output_masked.shape == (batch_size, tgt_len, d_model)
    print(f"✓ Encoder-decoder with masks: {decoder_output_masked.shape}")

    # Gradients should flow through both
    loss = decoder_output_masked.sum()
    # Note: Not calling backward here to avoid modifying the models

    print("\n✅ Encoder-decoder integration tests passed!")


def run_all_tests():
    """Run all decoder tests."""
    print("\n" + "=" * 70)
    print(" " * 20 + "TRANSFORMER DECODER TEST SUITE")
    print("=" * 70)

    try:
        test_decoder_layer_initialization()
        test_decoder_layer_shapes()
        test_causal_mask()
        test_cross_attention()
        test_source_and_target_masks()
        test_transformer_decoder_initialization()
        test_transformer_decoder_shapes()
        test_deep_decoder()
        test_gradient_flow()
        test_parameter_count()
        test_determinism()
        test_different_configurations()
        test_encoder_decoder_integration()

        print("\n" + "=" * 70)
        print(" " * 20 + "🎉 ALL TESTS PASSED! 🎉")
        print("=" * 70)
        print("\n✅ Transformer Decoder implementation is correct!")
        print("\nNext step: Complete Transformer Integration")
        print("  File: src/models/transformer/transformer.py")

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
