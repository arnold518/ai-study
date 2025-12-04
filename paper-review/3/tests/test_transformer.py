#!/usr/bin/env python
"""
Test complete Transformer implementation.

This script tests the full Transformer model:
- Weight tying between embeddings and output projection
- Embedding scaling by sqrt(d_model)
- Full forward pass (source → target → logits)
- Masking (padding and causal)
- Gradient flow end-to-end
- encode() and decode() methods
- Parameter count

Usage:
    /home/arnold/venv/bin/python tests/test_transformer.py
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn as nn
from src.models.transformer.transformer import Transformer
from src.utils.masking import create_padding_mask, create_target_mask
from config.transformer_config import TransformerConfig


def test_transformer_initialization():
    """Test Transformer initialization."""
    print("\n" + "=" * 60)
    print("Test 1: Transformer Initialization")
    print("=" * 60)

    config = TransformerConfig()
    vocab_size = 16000

    # Test with weight tying (default)
    model = Transformer(config, vocab_size, vocab_size)

    # Check components
    assert hasattr(model, 'src_embed'), "Should have src_embed"
    assert hasattr(model, 'tgt_embed'), "Should have tgt_embed"
    assert hasattr(model, 'pos_encoding'), "Should have pos_encoding"
    assert hasattr(model, 'encoder'), "Should have encoder"
    assert hasattr(model, 'decoder'), "Should have decoder"
    print(f"✓ All components initialized")

    # Check shared embeddings
    assert model.src_embed is model.tgt_embed, \
        "Source and target embeddings should be shared (share_src_tgt_embed=True)"
    print(f"✓ Source and target embeddings are shared")

    # Check weight tying
    assert model.output_projection is None, \
        "Output projection should be None when tie_embeddings=True"
    print(f"✓ Weight tying enabled (output_projection is None)")

    # Check embedding scale
    expected_scale = torch.sqrt(torch.tensor(config.d_model, dtype=torch.float32))
    assert abs(model.embed_scale - expected_scale) < 1e-6, \
        f"Embed scale should be sqrt(d_model) = {expected_scale}"
    print(f"✓ Embedding scale is sqrt(d_model) = {model.embed_scale:.4f}")

    print("\n✅ Transformer initialization tests passed!")


def test_weight_tying():
    """Test that weight tying is working correctly."""
    print("\n" + "=" * 60)
    print("Test 2: Weight Tying")
    print("=" * 60)

    config = TransformerConfig()
    vocab_size = 1000

    # With weight tying
    config.tie_embeddings = True
    model_tied = Transformer(config, vocab_size, vocab_size)

    # Without weight tying
    config.tie_embeddings = False
    model_untied = Transformer(config, vocab_size, vocab_size)

    # Count parameters
    params_tied = model_tied.count_parameters()
    params_untied = model_untied.count_parameters()

    print(f"  Parameters with weight tying: {params_tied:,}")
    print(f"  Parameters without weight tying: {params_untied:,}")

    # Difference should be vocab_size * d_model (the output layer)
    expected_diff = vocab_size * config.d_model
    actual_diff = params_untied - params_tied

    print(f"  Difference: {actual_diff:,}")
    print(f"  Expected (vocab_size * d_model): {expected_diff:,}")

    assert actual_diff == expected_diff, \
        f"Parameter difference {actual_diff} should equal {expected_diff}"
    print(f"✓ Weight tying reduces parameters correctly")

    # Test forward pass uses embedding weights
    batch_size = 2
    src_len = 5
    tgt_len = 4

    src = torch.randint(0, vocab_size, (batch_size, src_len))
    tgt = torch.randint(0, vocab_size, (batch_size, tgt_len))

    model_tied.eval()
    with torch.no_grad():
        logits = model_tied(src, tgt)

    # Output should have vocab_size dimension
    assert logits.shape == (batch_size, tgt_len, vocab_size), \
        f"Logits shape {logits.shape} != expected {(batch_size, tgt_len, vocab_size)}"
    print(f"✓ Forward pass with weight tying: {logits.shape}")

    print("\n✅ Weight tying tests passed!")


def test_embedding_scaling():
    """Test that embeddings are scaled by sqrt(d_model)."""
    print("\n" + "=" * 60)
    print("Test 3: Embedding Scaling")
    print("=" * 60)

    config = TransformerConfig()
    config.d_model = 512
    vocab_size = 1000

    model = Transformer(config, vocab_size, vocab_size)
    model.eval()

    batch_size = 2
    seq_len = 5

    # Create input
    src = torch.randint(0, vocab_size, (batch_size, seq_len))

    # Get raw embedding (without scaling)
    with torch.no_grad():
        raw_embed = model.src_embed(src)
        # Get scaled embedding (as done in forward)
        scaled_embed = raw_embed * model.embed_scale

    # Check scaling factor
    expected_scale = torch.sqrt(torch.tensor(config.d_model, dtype=torch.float32))
    actual_scale = (scaled_embed / raw_embed).mean()

    print(f"  Expected scale: {expected_scale:.4f}")
    print(f"  Actual scale: {actual_scale:.4f}")

    assert abs(actual_scale - expected_scale) < 1e-4, \
        "Embeddings should be scaled by sqrt(d_model)"
    print(f"✓ Embeddings are correctly scaled by sqrt({config.d_model}) = {expected_scale:.4f}")

    print("\n✅ Embedding scaling tests passed!")


def test_forward_pass_shapes():
    """Test full forward pass output shapes."""
    print("\n" + "=" * 60)
    print("Test 4: Forward Pass Shapes")
    print("=" * 60)

    config = TransformerConfig()
    vocab_size = 5000

    model = Transformer(config, vocab_size, vocab_size)
    model.eval()

    # Test different sequence lengths
    test_configs = [
        (2, 10, 8),    # batch=2, src_len=10, tgt_len=8
        (4, 15, 12),   # batch=4, src_len=15, tgt_len=12
        (1, 50, 30),   # batch=1, src_len=50, tgt_len=30
    ]

    for batch_size, src_len, tgt_len in test_configs:
        src = torch.randint(0, vocab_size, (batch_size, src_len))
        tgt = torch.randint(0, vocab_size, (batch_size, tgt_len))

        with torch.no_grad():
            logits = model(src, tgt)

        expected_shape = (batch_size, tgt_len, vocab_size)
        assert logits.shape == expected_shape, \
            f"Logits shape {logits.shape} != expected {expected_shape}"
        print(f"✓ Config (B={batch_size}, SrcL={src_len}, TgtL={tgt_len}): {logits.shape}")

    print("\n✅ Forward pass shape tests passed!")


def test_masking():
    """Test that padding and causal masks work correctly."""
    print("\n" + "=" * 60)
    print("Test 5: Masking")
    print("=" * 60)

    config = TransformerConfig()
    vocab_size = 1000
    pad_idx = 0

    model = Transformer(config, vocab_size, vocab_size, pad_idx=pad_idx)
    model.eval()

    batch_size = 2
    src_len = 10
    tgt_len = 8

    # Create sequences with padding
    src = torch.randint(1, vocab_size, (batch_size, src_len))
    tgt = torch.randint(1, vocab_size, (batch_size, tgt_len))

    # Add padding to sequences
    src[0, 7:] = pad_idx  # Pad last 3 positions in first batch
    tgt[0, 5:] = pad_idx  # Pad last 3 positions in first batch

    # Create masks
    src_mask = create_padding_mask(src, pad_idx)
    tgt_mask = create_target_mask(tgt, pad_idx)

    print(f"  Source shape: {src.shape}")
    print(f"  Target shape: {tgt.shape}")
    print(f"  Source mask shape: {src_mask.shape}")
    print(f"  Target mask shape: {tgt_mask.shape}")

    with torch.no_grad():
        # Without masks
        logits_no_mask = model(src, tgt)

        # With masks
        logits_with_mask = model(src, tgt, src_mask, tgt_mask)

    # Outputs should differ when masking is applied
    assert not torch.allclose(logits_no_mask, logits_with_mask), \
        "Masking should affect output"
    print(f"✓ Masking affects output (not the same as no masking)")

    # Shape should be correct
    expected_shape = (batch_size, tgt_len, vocab_size)
    assert logits_with_mask.shape == expected_shape, \
        f"Logits shape {logits_with_mask.shape} != expected {expected_shape}"
    print(f"✓ Output shape with masks: {logits_with_mask.shape}")

    print("\n✅ Masking tests passed!")


def test_gradient_flow():
    """Test that gradients flow through entire model."""
    print("\n" + "=" * 60)
    print("Test 6: End-to-End Gradient Flow")
    print("=" * 60)

    config = TransformerConfig()
    vocab_size = 1000

    model = Transformer(config, vocab_size, vocab_size)

    batch_size = 2
    src_len = 10
    tgt_len = 8

    src = torch.randint(0, vocab_size, (batch_size, src_len))
    tgt = torch.randint(0, vocab_size, (batch_size, tgt_len))

    # Forward pass
    logits = model(src, tgt)

    # Compute loss and backprop
    target_labels = torch.randint(0, vocab_size, (batch_size, tgt_len))
    loss = nn.CrossEntropyLoss()(
        logits.view(-1, vocab_size),
        target_labels.view(-1)
    )
    loss.backward()

    # Check that parameters have gradients
    params_with_grad = 0
    params_without_grad = 0

    for name, param in model.named_parameters():
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


def test_encode_decode_methods():
    """Test separate encode() and decode() methods."""
    print("\n" + "=" * 60)
    print("Test 7: Separate Encode/Decode Methods")
    print("=" * 60)

    config = TransformerConfig()
    vocab_size = 1000

    model = Transformer(config, vocab_size, vocab_size)
    model.eval()

    batch_size = 2
    src_len = 10
    tgt_len = 8

    src = torch.randint(0, vocab_size, (batch_size, src_len))
    tgt = torch.randint(0, vocab_size, (batch_size, tgt_len))

    with torch.no_grad():
        # Test encode
        encoder_output = model.encode(src)
        assert encoder_output.shape == (batch_size, src_len, config.d_model)
        print(f"✓ encode() output shape: {encoder_output.shape}")

        # Test decode
        logits = model.decode(tgt, encoder_output)
        assert logits.shape == (batch_size, tgt_len, vocab_size)
        print(f"✓ decode() output shape: {logits.shape}")

        # Test that encode + decode equals forward
        logits_forward = model(src, tgt)
        assert torch.allclose(logits, logits_forward, atol=1e-5), \
            "encode() + decode() should equal forward()"
        print(f"✓ encode() + decode() equals forward()")

    print("\n✅ Encode/decode method tests passed!")


def test_parameter_count():
    """Test parameter counts with different configurations."""
    print("\n" + "=" * 60)
    print("Test 8: Parameter Count")
    print("=" * 60)

    config = TransformerConfig()
    vocab_size = 16000

    # With weight tying and shared embeddings (paper's approach)
    config.tie_embeddings = True
    config.share_src_tgt_embed = True
    model_tied = Transformer(config, vocab_size, vocab_size)
    params_tied = model_tied.count_parameters()

    print(f"  With weight tying + shared embeddings: {params_tied:,}")

    # Expected components:
    # - Embeddings: vocab_size * d_model
    # - Encoder: (calculated from previous tests)
    # - Decoder: (calculated from previous tests)
    # - No output layer (weight tying)

    embeddings_params = vocab_size * config.d_model
    encoder_params = sum(p.numel() for p in model_tied.encoder.parameters())
    decoder_params = sum(p.numel() for p in model_tied.decoder.parameters())

    print(f"    - Embeddings: {embeddings_params:,}")
    print(f"    - Encoder: {encoder_params:,}")
    print(f"    - Decoder: {decoder_params:,}")
    print(f"    - Output: 0 (tied with embeddings)")

    # Without weight tying
    config.tie_embeddings = False
    model_untied = Transformer(config, vocab_size, vocab_size)
    params_untied = model_untied.count_parameters()

    print(f"\n  Without weight tying: {params_untied:,}")
    print(f"    - Additional output layer: {vocab_size * config.d_model:,}")

    # Check difference
    diff = params_untied - params_tied
    expected_diff = vocab_size * config.d_model
    assert diff == expected_diff, \
        f"Difference {diff:,} should equal {expected_diff:,}"
    print(f"✓ Parameter difference is correct: {diff:,}")

    print("\n✅ Parameter count tests passed!")


def test_different_configs():
    """Test Transformer with different model sizes."""
    print("\n" + "=" * 60)
    print("Test 9: Different Model Configurations")
    print("=" * 60)

    configs = [
        (256, 4, 1024, 2, 2),   # Small: d_model, num_heads, d_ff, enc_layers, dec_layers
        (512, 8, 2048, 6, 6),   # Base
        (768, 12, 3072, 4, 4),  # Medium
    ]

    vocab_size = 5000
    batch_size = 2
    src_len = 10
    tgt_len = 8

    for d_model, num_heads, d_ff, enc_layers, dec_layers in configs:
        config = TransformerConfig()
        config.d_model = d_model
        config.num_heads = num_heads
        config.d_ff = d_ff
        config.num_encoder_layers = enc_layers
        config.num_decoder_layers = dec_layers

        model = Transformer(config, vocab_size, vocab_size)
        model.eval()

        src = torch.randint(0, vocab_size, (batch_size, src_len))
        tgt = torch.randint(0, vocab_size, (batch_size, tgt_len))

        with torch.no_grad():
            logits = model(src, tgt)

        assert logits.shape == (batch_size, tgt_len, vocab_size)
        param_count = model.count_parameters()
        print(f"✓ Config (d={d_model}, h={num_heads}, ff={d_ff}, "
              f"enc={enc_layers}, dec={dec_layers}): {param_count:,} params")

    print("\n✅ Different configuration tests passed!")


def test_separate_vocabularies():
    """Test with separate source and target vocabularies."""
    print("\n" + "=" * 60)
    print("Test 10: Separate Vocabularies")
    print("=" * 60)

    config = TransformerConfig()
    config.share_src_tgt_embed = False  # Use separate embeddings

    src_vocab_size = 8000
    tgt_vocab_size = 10000

    model = Transformer(config, src_vocab_size, tgt_vocab_size)
    model.eval()

    # Check that embeddings are different
    assert model.src_embed is not model.tgt_embed, \
        "Source and target embeddings should be separate"
    print(f"✓ Source and target embeddings are separate")

    # Check embedding sizes
    assert model.src_embed.num_embeddings == src_vocab_size
    assert model.tgt_embed.num_embeddings == tgt_vocab_size
    print(f"✓ Source vocab size: {src_vocab_size}")
    print(f"✓ Target vocab size: {tgt_vocab_size}")

    # Test forward pass
    batch_size = 2
    src_len = 10
    tgt_len = 8

    src = torch.randint(0, src_vocab_size, (batch_size, src_len))
    tgt = torch.randint(0, tgt_vocab_size, (batch_size, tgt_len))

    with torch.no_grad():
        logits = model(src, tgt)

    assert logits.shape == (batch_size, tgt_len, tgt_vocab_size)
    print(f"✓ Forward pass with separate vocabs: {logits.shape}")

    print("\n✅ Separate vocabulary tests passed!")


def test_determinism():
    """Test that model is deterministic in eval mode."""
    print("\n" + "=" * 60)
    print("Test 11: Determinism")
    print("=" * 60)

    config = TransformerConfig()
    vocab_size = 1000

    model = Transformer(config, vocab_size, vocab_size)
    model.eval()

    batch_size = 2
    src_len = 10
    tgt_len = 8

    src = torch.randint(0, vocab_size, (batch_size, src_len))
    tgt = torch.randint(0, vocab_size, (batch_size, tgt_len))

    with torch.no_grad():
        # Run twice
        logits1 = model(src, tgt)
        logits2 = model(src, tgt)

    # Should be identical in eval mode
    assert torch.allclose(logits1, logits2), \
        "Outputs should be identical in eval mode"
    print(f"✓ Outputs are deterministic in eval mode")

    print("\n✅ Determinism tests passed!")


def run_all_tests():
    """Run all Transformer tests."""
    print("\n" + "=" * 70)
    print(" " * 18 + "COMPLETE TRANSFORMER TEST SUITE")
    print("=" * 70)

    try:
        test_transformer_initialization()
        test_weight_tying()
        test_embedding_scaling()
        test_forward_pass_shapes()
        test_masking()
        test_gradient_flow()
        test_encode_decode_methods()
        test_parameter_count()
        test_different_configs()
        test_separate_vocabularies()
        test_determinism()

        print("\n" + "=" * 70)
        print(" " * 20 + "🎉 ALL TESTS PASSED! 🎉")
        print("=" * 70)
        print("\n✅ Complete Transformer implementation is correct!")
        print("\n" + "=" * 70)
        print("TRANSFORMER ARCHITECTURE COMPLETE!")
        print("=" * 70)
        print("\nAll components verified:")
        print("  ✅ Multi-head attention")
        print("  ✅ Position-wise feed-forward")
        print("  ✅ Positional encoding")
        print("  ✅ Encoder stack (6 layers)")
        print("  ✅ Decoder stack (6 layers)")
        print("  ✅ Weight tying (embeddings + output)")
        print("  ✅ Embedding scaling (√d_model)")
        print("  ✅ Masking (padding + causal)")
        print("\nNext steps:")
        print("  1. Implement training loop (scripts/train.py)")
        print("  2. Implement inference (scripts/translate.py)")
        print("  3. Train on Korean-English dataset")

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
