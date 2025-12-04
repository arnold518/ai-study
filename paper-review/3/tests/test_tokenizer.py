#!/usr/bin/env python
"""
Test trained SentencePiece tokenizers.

This script loads trained tokenizers based on config and tests them on sample sentences.
Supports both shared and separate vocabulary modes from config/base_config.py.

Usage:
    /home/arnold/venv/bin/python tests/test_tokenizer.py
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from pathlib import Path
from src.data.tokenizer import SentencePieceTokenizer
from src.data.dataset import load_tokenizers
from config.base_config import BaseConfig


def test_tokenizer(tokenizer, language, test_sentences):
    """Test a tokenizer with sample sentences."""
    print(f"\n{'=' * 60}")
    print(f"{language} Tokenizer Test")
    print('=' * 60)
    print(f"Vocab size: {tokenizer.vocab_size:,}")
    print(f"PAD ID: {tokenizer.pad_id}")
    print(f"UNK ID: {tokenizer.unk_id}")
    print(f"BOS ID: {tokenizer.bos_id}")
    print(f"EOS ID: {tokenizer.eos_id}")
    print()

    for i, text in enumerate(test_sentences, 1):
        print(f"Test {i}:")
        print(f"  Original: {text}")

        # Tokenize to pieces
        tokens = tokenizer.tokenize(text)
        print(f"  Tokens:   {tokens}")
        print(f"  # Tokens: {len(tokens)}")

        # Tokenize to IDs
        ids = tokenizer.encode_ids(text)
        print(f"  IDs:      {ids[:10]}{'...' if len(ids) > 10 else ''}")

        # Decode back
        decoded = tokenizer.decode_ids(ids)
        print(f"  Decoded:  {decoded}")

        # Verify roundtrip
        if decoded == text:
            print(f"  ✓ Roundtrip successful")
        else:
            print(f"  ✗ Roundtrip failed!")
            print(f"    Expected: {text}")
            print(f"    Got:      {decoded}")
        print()


def load_sample_sentences(file_path, num_samples=5):
    """Load sample sentences from data file."""
    if not Path(file_path).exists():
        return None

    with open(file_path, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip()]

    # Get diverse samples (beginning, middle, end)
    if len(lines) < num_samples:
        return lines

    step = len(lines) // num_samples
    return [lines[i * step] for i in range(num_samples)]


def main():
    config = BaseConfig()

    print("=" * 60)
    print("SentencePiece Tokenizer Test")
    print("=" * 60)
    print(f"Mode: {'SHARED' if config.use_shared_vocab else 'SEPARATE'} vocabulary")
    print()

    # Setup paths
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    vocab_dir = project_root / config.vocab_dir

    # Load tokenizers using config
    print("Loading tokenizers...")
    try:
        ko_tokenizer, en_tokenizer = load_tokenizers(vocab_dir, use_shared_vocab=config.use_shared_vocab)
    except FileNotFoundError as e:
        print(f"\n✗ {e}")
        return

    if config.use_shared_vocab:
        print(f"✓ Shared tokenizer loaded (vocab_size={ko_tokenizer.vocab_size:,})")
        print("  Note: Korean and English use the same tokenizer")
    else:
        print(f"✓ Korean tokenizer loaded (vocab_size={ko_tokenizer.vocab_size:,})")
        print(f"✓ English tokenizer loaded (vocab_size={en_tokenizer.vocab_size:,})")

    # Test Korean tokenizer
    ko_test_sentences = [
        "안녕하세요. 저는 번역 시스템입니다.",
        "개인용 컴퓨터 사용의 상당 부분은 이것보다 뛰어날 수 있느냐?",
        "먹었습니다.",
        "아름다운 풍경",
        "인공지능 번역"
    ]

    # Try to load from actual data
    sample_ko = load_sample_sentences(project_root / config.processed_data_dir / f'train.{config.src_lang}', 3)
    if sample_ko:
        ko_test_sentences = sample_ko + ko_test_sentences[:2]

    test_tokenizer(ko_tokenizer, f"{config.src_lang.upper()} ({config.src_lang})", ko_test_sentences)

    # Test English tokenizer
    en_test_sentences = [
        "Hello. I am a translation system.",
        "Much of personal computing is about can you top this?",
        "I ate.",
        "Beautiful scenery",
        "Artificial intelligence translation"
    ]

    # Try to load from actual data
    sample_en = load_sample_sentences(project_root / config.processed_data_dir / f'train.{config.tgt_lang}', 3)
    if sample_en:
        en_test_sentences = sample_en + en_test_sentences[:2]

    test_tokenizer(en_tokenizer, f"{config.tgt_lang.upper()} ({config.tgt_lang})", en_test_sentences)

    # Test special cases
    print(f"\n{'=' * 60}")
    print("Special Cases Test")
    print('=' * 60)

    special_cases = [
        ("Empty string", ""),
        ("Single space", " "),
        ("Numbers", "12345"),
        ("Mixed", "Hello 안녕 123"),
        ("Punctuation", "!@#$%^&*()"),
    ]

    print(f"\n{config.src_lang} tokenizer:")
    for name, text in special_cases:
        try:
            ids = ko_tokenizer.encode_ids(text)
            decoded = ko_tokenizer.decode_ids(ids)
            status = "✓" if decoded == text else "✗"
            print(f"  {status} {name:20s}: {len(ids):3d} tokens")
        except Exception as e:
            print(f"  ✗ {name:20s}: Error - {e}")

    print(f"\n{config.tgt_lang} tokenizer:")
    for name, text in special_cases:
        try:
            ids = en_tokenizer.encode_ids(text)
            decoded = en_tokenizer.decode_ids(ids)
            status = "✓" if decoded == text else "✗"
            print(f"  {status} {name:20s}: {len(ids):3d} tokens")
        except Exception as e:
            print(f"  ✗ {name:20s}: Error - {e}")

    # Summary
    print(f"\n{'=' * 60}")
    print("Summary")
    print('=' * 60)
    if config.use_shared_vocab:
        print(f"✓ Shared tokenizer: vocab_size={ko_tokenizer.vocab_size:,}")
        print(f"  (Used for both {config.src_lang} and {config.tgt_lang})")
    else:
        print(f"✓ {config.src_lang} tokenizer: vocab_size={ko_tokenizer.vocab_size:,}")
        print(f"✓ {config.tgt_lang} tokenizer: vocab_size={en_tokenizer.vocab_size:,}")
    print()
    print("Next step: Test dataset")
    print("  /home/arnold/venv/bin/python tests/test_dataset.py")


if __name__ == "__main__":
    main()
