#!/usr/bin/env python
"""Integration test: Translation with repetition penalty."""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import torch
from src.data.tokenizer import SentencePieceTokenizer
from src.inference.translator import Translator
from src.models.transformer.transformer import Transformer
from config.transformer_config import TransformerConfig

def test_translations():
    """Test translation quality with repetition penalty."""

    print("=" * 80)
    print("Translation Integration Test")
    print("=" * 80)
    print()

    # Check if checkpoint exists
    checkpoint_path = 'checkpoints/best_model.pt'
    if not os.path.exists(checkpoint_path):
        print(f"⚠️  Checkpoint not found: {checkpoint_path}")
        print("   Please train a model first or specify a different checkpoint.")
        return

    print("Loading model and tokenizers...")

    # Load tokenizers
    ko_tokenizer = SentencePieceTokenizer('data/vocab/ko_spm.model')
    en_tokenizer = SentencePieceTokenizer('data/vocab/en_spm.model')
    print(f"  ✓ Tokenizers loaded (vocab: ko={ko_tokenizer.vocab_size}, en={en_tokenizer.vocab_size})")

    # Load config and model
    config = TransformerConfig()
    model = Transformer(
        config,
        src_vocab_size=ko_tokenizer.vocab_size,
        tgt_vocab_size=en_tokenizer.vocab_size
    )

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"  ✓ Model loaded from {checkpoint_path} (epoch {checkpoint.get('epoch', '?')})")
    print()

    # Create translator
    translator = Translator(
        model=model,
        src_tokenizer=ko_tokenizer,
        tgt_tokenizer=en_tokenizer,
        device='cpu',  # Use CPU for testing
        max_length=150
    )

    # Test sentences (from the failing translations)
    test_cases = [
        ("안녕하세요", "Simple greeting"),
        ("거짓말 하지 마", "Simple command"),
        ("토론에 참여한 사람들은 법 집행과 국가 안전보장에 대한 우려를 표명해야 할 필요성을 진지하게 받아 들이고 있습니다.", "Long complex sentence"),
        ("3,000마리의 전갈과 32일 동안 동거할 태국 여인", "Sentence with numbers"),
    ]

    print("=" * 80)
    print("TRANSLATIONS WITH REPETITION PENALTY")
    print("=" * 80)
    print()

    for i, (src, description) in enumerate(test_cases, 1):
        print(f"[Test {i}] {description}")
        print(f"Source:  {src}")
        print()

        # Translate with greedy
        try:
            translation = translator.translate(src, method='greedy')

            # Check for repetition
            tokens = translation.split()
            unique_tokens = set(tokens)
            repetition_rate = (len(tokens) - len(unique_tokens)) / len(tokens) if tokens else 0

            print(f"Translation: {translation}")
            print(f"Length: {len(tokens)} tokens, Unique: {len(unique_tokens)} tokens")
            print(f"Repetition rate: {repetition_rate:.1%}")

            # Simple repetition check
            has_triple_repeat = False
            for j in range(len(tokens) - 2):
                if tokens[j] == tokens[j+1] == tokens[j+2]:
                    has_triple_repeat = True
                    print(f"⚠️  Found triple repeat: '{tokens[j]}'")
                    break

            if not has_triple_repeat and repetition_rate < 0.3:
                print("✓ No obvious repetition detected")
            elif repetition_rate >= 0.5:
                print("✗ High repetition rate!")

        except Exception as e:
            print(f"✗ Translation failed: {e}")

        print()
        print("-" * 80)
        print()

    print("=" * 80)
    print("TEST COMPLETE")
    print("=" * 80)
    print()
    print("Expected behavior:")
    print("  ✓ Short sentences should translate well")
    print("  ✓ No 'the the the...' loops")
    print("  ✓ No 'and the lawmakers, and the lawmakers...' patterns")
    print("  ⚠️  Long sentences may still have quality issues (but no loops)")
    print()

if __name__ == "__main__":
    test_translations()
