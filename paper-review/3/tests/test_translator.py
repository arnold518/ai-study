#!/usr/bin/env python
"""Test translator with tokenizer integration."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
from src.models.transformer.transformer import Transformer
from src.data.tokenizer import SentencePieceTokenizer
from src.inference.translator import Translator
from config.transformer_config import TransformerConfig


def test_translator_uses_tokenizer_indices():
    """Test that Translator correctly uses token indices from tokenizer."""
    print("=" * 80)
    print("Testing Translator Token Index Integration")
    print("=" * 80)

    # Check if tokenizers exist
    ko_model_path = 'data/vocab/ko_spm.model'
    en_model_path = 'data/vocab/en_spm.model'

    if not os.path.exists(ko_model_path) or not os.path.exists(en_model_path):
        print("⚠️  Tokenizer models not found. Please train tokenizers first:")
        print("  /home/arnold/venv/bin/python scripts/train_tokenizer.py")
        return

    # Load tokenizers
    src_tokenizer = SentencePieceTokenizer(ko_model_path)
    tgt_tokenizer = SentencePieceTokenizer(en_model_path)

    print(f"Source tokenizer: {src_tokenizer}")
    print(f"Target tokenizer: {tgt_tokenizer}")
    print()

    # Display special token IDs
    print("Special token IDs from target tokenizer:")
    print(f"  pad_id: {tgt_tokenizer.pad_id}")
    print(f"  unk_id: {tgt_tokenizer.unk_id}")
    print(f"  bos_id: {tgt_tokenizer.bos_id}")
    print(f"  eos_id: {tgt_tokenizer.eos_id}")
    print()

    # Create small model
    config = TransformerConfig()
    config.d_model = 64
    config.num_heads = 4
    config.num_encoder_layers = 2
    config.num_decoder_layers = 2
    config.d_ff = 128
    config.dropout = 0.0

    device = torch.device('cpu')
    model = Transformer(
        config,
        src_vocab_size=src_tokenizer.vocab_size,
        tgt_vocab_size=tgt_tokenizer.vocab_size,
        pad_idx=tgt_tokenizer.pad_id
    )

    # Create translator
    translator = Translator(
        model=model,
        src_tokenizer=src_tokenizer,
        tgt_tokenizer=tgt_tokenizer,
        device=device,
        max_length=50
    )

    print("Translator special token IDs:")
    print(f"  pad_idx: {translator.pad_idx}")
    print(f"  unk_idx: {translator.unk_idx}")
    print(f"  bos_idx: {translator.bos_idx}")
    print(f"  eos_idx: {translator.eos_idx}")
    print()

    # Verify they match
    assert translator.pad_idx == tgt_tokenizer.pad_id, "pad_idx mismatch"
    assert translator.unk_idx == tgt_tokenizer.unk_id, "unk_idx mismatch"
    assert translator.bos_idx == tgt_tokenizer.bos_id, "bos_idx mismatch"
    assert translator.eos_idx == tgt_tokenizer.eos_id, "eos_idx mismatch"

    print("✓ Translator correctly uses token indices from tokenizer!")
    print()

    # Test basic translation (will be random since model is untrained)
    print("Testing translation interface (output will be random)...")
    test_sentence = "테스트"
    print(f"Input: {test_sentence}")

    try:
        translation = translator.translate(test_sentence, method='greedy')
        print(f"Output (greedy): {translation}")
        print()
        print("✓ Translation interface works!")
    except Exception as e:
        print(f"✗ Translation failed: {e}")
        raise

    print("=" * 80)
    print("All translator integration tests passed!")
    print("=" * 80)


if __name__ == "__main__":
    test_translator_uses_tokenizer_indices()
