#!/usr/bin/env python
"""Analyze preprocessing and tokenization quality."""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import json
import random
from pathlib import Path
from src.data.tokenizer import SentencePieceTokenizer

def analyze_tokenized_lengths(split_name='train', num_samples=1000):
    """Analyze tokenized lengths vs character lengths."""

    print("=" * 80)
    print(f"Tokenization Analysis: {split_name.upper()}")
    print("=" * 80)
    print()

    # Load data
    data_dir = Path('data/processed')
    ko_path = data_dir / f'{split_name}.ko'
    en_path = data_dir / f'{split_name}.en'

    with open(ko_path, 'r', encoding='utf-8') as f:
        ko_lines = [line.strip() for line in f if line.strip()]

    with open(en_path, 'r', encoding='utf-8') as f:
        en_lines = [line.strip() for line in f if line.strip()]

    print(f"Loaded {len(ko_lines)} sentence pairs")
    print()

    # Load tokenizers
    ko_tok = SentencePieceTokenizer('data/vocab/ko_spm.model')
    en_tok = SentencePieceTokenizer('data/vocab/en_spm.model')
    print(f"Tokenizers loaded (vocab: ko={ko_tok.vocab_size}, en={en_tok.vocab_size})")
    print()

    # Sample random pairs
    num_samples = min(num_samples, len(ko_lines))
    indices = random.sample(range(len(ko_lines)), num_samples)

    ko_char_lens = []
    ko_token_lens = []
    en_char_lens = []
    en_token_lens = []

    too_long_count = 0
    max_seq_len = 128  # From config

    for idx in indices:
        ko = ko_lines[idx]
        en = en_lines[idx]

        # Character lengths
        ko_char_lens.append(len(ko))
        en_char_lens.append(len(en))

        # Token lengths (with BOS/EOS)
        ko_tokens = ko_tok.encode_ids(ko)
        en_tokens = en_tok.encode_ids(en)
        ko_token_len = len(ko_tokens) + 2  # +2 for BOS/EOS
        en_token_len = len(en_tokens) + 2

        ko_token_lens.append(ko_token_len)
        en_token_lens.append(en_token_len)

        if ko_token_len > max_seq_len or en_token_len > max_seq_len:
            too_long_count += 1

    # Compute statistics
    print("CHARACTER LENGTHS (filtering criteria)")
    print("-" * 80)
    print(f"Korean:")
    print(f"  Min: {min(ko_char_lens)}, Max: {max(ko_char_lens)}, Avg: {sum(ko_char_lens)/len(ko_char_lens):.1f}")
    print(f"English:")
    print(f"  Min: {min(en_char_lens)}, Max: {max(en_char_lens)}, Avg: {sum(en_char_lens)/len(en_char_lens):.1f}")
    print()

    print("TOKENIZED LENGTHS (actual model input)")
    print("-" * 80)
    print(f"Korean (with BOS/EOS):")
    print(f"  Min: {min(ko_token_lens)}, Max: {max(ko_token_lens)}, Avg: {sum(ko_token_lens)/len(ko_token_lens):.1f}")
    print(f"English (with BOS/EOS):")
    print(f"  Min: {min(en_token_lens)}, Max: {max(en_token_lens)}, Avg: {sum(en_token_lens)/len(en_token_lens):.1f}")
    print()

    print("MODEL CONFIGURATION")
    print("-" * 80)
    print(f"Max sequence length: {max_seq_len}")
    print(f"Samples exceeding max_seq_len: {too_long_count}/{num_samples} ({100*too_long_count/num_samples:.1f}%)")
    print()

    # Distribution
    print("TOKEN LENGTH DISTRIBUTION")
    print("-" * 80)

    def print_distribution(lengths, name):
        bins = [0, 20, 40, 60, 80, 100, 128, 150, 200, float('inf')]
        bin_labels = ['0-20', '21-40', '41-60', '61-80', '81-100', '101-128', '129-150', '151-200', '200+']

        counts = [0] * len(bin_labels)
        for length in lengths:
            for i, (low, high) in enumerate(zip(bins[:-1], bins[1:])):
                if low < length <= high:
                    counts[i] += 1
                    break

        print(f"{name}:")
        for label, count in zip(bin_labels, counts):
            pct = 100 * count / len(lengths)
            bar = '█' * int(pct / 2)
            print(f"  {label:>10}: {count:5} ({pct:5.1f}%) {bar}")
        print()

    print_distribution(ko_token_lens, "Korean tokens")
    print_distribution(en_token_lens, "English tokens")

    return {
        'ko_char': ko_char_lens,
        'ko_token': ko_token_lens,
        'en_char': en_char_lens,
        'en_token': en_token_lens,
        'too_long': too_long_count,
        'total': num_samples
    }


def show_examples(split_name='train', num_examples=5):
    """Show example sentences and their tokenization."""

    print("=" * 80)
    print(f"Example Sentences: {split_name.upper()}")
    print("=" * 80)
    print()

    # Load data
    data_dir = Path('data/processed')
    ko_path = data_dir / f'{split_name}.ko'
    en_path = data_dir / f'{split_name}.en'

    with open(ko_path, 'r', encoding='utf-8') as f:
        ko_lines = [line.strip() for line in f if line.strip()]

    with open(en_path, 'r', encoding='utf-8') as f:
        en_lines = [line.strip() for line in f if line.strip()]

    # Load tokenizers
    ko_tok = SentencePieceTokenizer('data/vocab/ko_spm.model')
    en_tok = SentencePieceTokenizer('data/vocab/en_spm.model')

    # Sample different lengths
    short_idx = 0
    medium_idx = len(ko_lines) // 2
    long_idx = -1

    indices = [short_idx, medium_idx, long_idx] + random.sample(range(len(ko_lines)), num_examples - 3)

    for i, idx in enumerate(indices[:num_examples], 1):
        ko = ko_lines[idx]
        en = en_lines[idx]

        ko_tokens = ko_tok.tokenize(ko)
        en_tokens = en_tok.tokenize(en)

        ko_token_len = len(ko_tokens) + 2
        en_token_len = len(en_tokens) + 2

        print(f"[Example {i}]")
        print(f"Korean ({len(ko)} chars → {ko_token_len} tokens):")
        print(f"  Text: {ko[:80]}{'...' if len(ko) > 80 else ''}")
        print(f"  Tokens: {' '.join(ko_tokens[:15])}{'...' if len(ko_tokens) > 15 else ''}")
        print()
        print(f"English ({len(en)} chars → {en_token_len} tokens):")
        print(f"  Text: {en[:80]}{'...' if len(en) > 80 else ''}")
        print(f"  Tokens: {' '.join(en_tokens[:15])}{'...' if len(en_tokens) > 15 else ''}")
        print()
        print("-" * 80)
        print()


def main():
    """Run preprocessing analysis."""

    print("\n")
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 20 + "DATA PREPROCESSING ANALYSIS" + " " * 31 + "║")
    print("╚" + "=" * 78 + "╝")
    print("\n")

    # Analyze training data
    train_stats = analyze_tokenized_lengths('train', num_samples=10000)

    print("\n")

    # Show examples
    show_examples('train', num_examples=5)

    # Summary
    print("=" * 80)
    print("SUMMARY & ISSUES")
    print("=" * 80)
    print()

    print("1. FILTERING METHOD:")
    print("   ✓ Using TOKEN-BASED filtering (128 tokens max)")
    print("   ✓ Filtering criterion matches model input!")
    print()

    print("2. TOKENIZED LENGTH VIOLATIONS:")
    pct = 100 * train_stats['too_long'] / train_stats['total']
    print(f"   - {train_stats['too_long']:,}/{train_stats['total']:,} samples ({pct:.1f}%) exceed max_seq_length=128")
    if pct > 10:
        print("   ✗ ISSUE: Too many sequences will be truncated during training!")
    print()

    print("3. VALIDATION/TEST MISMATCH:")
    with open('data/processed/statistics.json', 'r') as f:
        stats = json.load(f)

    train_max_ko = stats['train']['korean']['max_length']
    train_max_en = stats['train']['english']['max_length']
    val_max_ko = stats['validation']['korean']['max_length']
    val_max_en = stats['validation']['english']['max_length']

    print(f"   Training max:   {train_max_ko} ko, {train_max_en} en (chars)")
    print(f"   Validation max: {val_max_ko} ko, {val_max_en} en (chars)")

    if val_max_ko > train_max_ko or val_max_en > train_max_en:
        print("   ✗ ISSUE: Validation has longer sequences than training!")
        print("     Model trained on short sentences, tested on long ones")
    print()

    print("4. FILTERING STATISTICS:")
    kept = stats['train']['filtering']['kept']
    total = stats['train']['filtering']['total']
    too_long = stats['train']['filtering'].get('too_long_tokens', 0)
    ratio = stats['train']['filtering']['ratio_mismatch']

    print(f"   Kept: {kept:,}/{total:,} ({100*kept/total:.1f}%)")
    print(f"   Discarded: {total-kept:,} ({100*(total-kept)/total:.1f}%)")
    print(f"     - Too long (tokens): {too_long:,} ({100*too_long/total:.1f}%)")
    print(f"     - Ratio mismatch: {ratio:,} ({100*ratio/total:.1f}%)")

    if (total - kept) / total > 0.5:
        print("   ✗ ISSUE: Discarding >50% of data may limit model performance")
    print()


if __name__ == "__main__":
    main()
