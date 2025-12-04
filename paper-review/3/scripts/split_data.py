#!/usr/bin/env python
"""
Merge and preprocess Korean-English parallel corpora from multiple sources.

This script:
1. Loads data from all available datasets in data/raw/
2. Merges datasets by split type (train/validation/test)
3. Cleans and filters sentence pairs based on config/base_config.py
4. Saves unified splits to data/processed/
5. Generates statistics showing contribution from each source

Purpose: Create a single, clean, unified dataset from multiple sources.

Usage:
    /home/arnold/venv/bin/python scripts/split_data.py
"""

import os
import sys
import json
from pathlib import Path
from collections import defaultdict

# Add parent directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from config.base_config import BaseConfig


def load_parallel_data(ko_path, en_path):
    """Load parallel corpus from text files."""
    with open(ko_path, 'r', encoding='utf-8') as f:
        ko_lines = [line.strip() for line in f]

    with open(en_path, 'r', encoding='utf-8') as f:
        en_lines = [line.strip() for line in f]

    assert len(ko_lines) == len(en_lines), \
        f"Mismatch: {len(ko_lines)} Korean vs {len(en_lines)} English sentences"

    return ko_lines, en_lines


def clean_and_filter(ko_lines, en_lines, min_len=3, max_len=150, max_ratio=3.0):
    """
    Clean and filter sentence pairs.

    Args:
        ko_lines: List of Korean sentences
        en_lines: List of English sentences
        min_len: Minimum sentence length (characters)
        max_len: Maximum sentence length (characters)
        max_ratio: Maximum length ratio between source and target

    Returns:
        Filtered lists of Korean and English sentences, and statistics
    """
    filtered_ko = []
    filtered_en = []

    stats = {
        'total': len(ko_lines),
        'empty': 0,
        'too_short': 0,
        'too_long': 0,
        'ratio_mismatch': 0,
        'kept': 0
    }

    for ko, en in zip(ko_lines, en_lines):
        # Skip empty lines
        if not ko or not en:
            stats['empty'] += 1
            continue

        # Skip too short sentences
        if len(ko) < min_len or len(en) < min_len:
            stats['too_short'] += 1
            continue

        # Skip too long sentences
        if len(ko) > max_len or len(en) > max_len:
            stats['too_long'] += 1
            continue

        # Skip if length ratio is too high (likely misaligned)
        ratio = max(len(ko), len(en)) / min(len(ko), len(en))
        if ratio > max_ratio:
            stats['ratio_mismatch'] += 1
            continue

        filtered_ko.append(ko)
        filtered_en.append(en)
        stats['kept'] += 1

    return filtered_ko, filtered_en, stats


def load_datasets_from_raw(raw_dir):
    """
    Load all available datasets from raw directory.

    Returns:
        Dict mapping split names to lists of (dataset_name, ko_lines, en_lines)
    """
    datasets_by_split = defaultdict(list)

    # Find all dataset directories
    dataset_dirs = [d for d in raw_dir.iterdir()
                   if d.is_dir() and d.name in ['moo', 'tatoeba', 'aihub']]

    if not dataset_dirs:
        print(f"✗ No dataset directories found in {raw_dir}")
        return None

    print(f"Found {len(dataset_dirs)} dataset(s): {[d.name for d in dataset_dirs]}")
    print()

    # Load each dataset
    for dataset_dir in dataset_dirs:
        dataset_name = dataset_dir.name
        print(f"Loading {dataset_name}...")

        # Check which splits exist for this dataset
        for split in ['train', 'validation', 'test']:
            ko_path = dataset_dir / f"{split}.ko"
            en_path = dataset_dir / f"{split}.en"

            if ko_path.exists() and en_path.exists():
                ko_lines, en_lines = load_parallel_data(ko_path, en_path)
                datasets_by_split[split].append((dataset_name, ko_lines, en_lines))
                print(f"  ✓ {split}: {len(ko_lines):,} pairs")
            else:
                print(f"  - {split}: not available")

        print()

    return datasets_by_split


def merge_and_process_split(split_name, datasets, min_len, max_len, max_ratio):
    """
    Merge multiple datasets for a split and apply filtering.

    Args:
        split_name: 'train', 'validation', or 'test'
        datasets: List of (dataset_name, ko_lines, en_lines) tuples
        min_len, max_len, max_ratio: Filtering parameters

    Returns:
        Merged and filtered ko/en lines, plus statistics
    """
    print(f"\n{'=' * 60}")
    print(f"Processing: {split_name}")
    print('=' * 60)

    if not datasets:
        print(f"No datasets available for {split_name} split")
        return None, None, None

    # Merge all datasets
    merged_ko = []
    merged_en = []
    source_stats = {}

    print(f"\nMerging {len(datasets)} dataset(s)...")
    for dataset_name, ko_lines, en_lines in datasets:
        merged_ko.extend(ko_lines)
        merged_en.extend(en_lines)
        source_stats[dataset_name] = len(ko_lines)
        print(f"  {dataset_name}: {len(ko_lines):,} pairs")

    print(f"\nTotal before filtering: {len(merged_ko):,} pairs")

    # Apply filtering
    print("Applying filters...")
    filtered_ko, filtered_en, filter_stats = clean_and_filter(
        merged_ko, merged_en, min_len, max_len, max_ratio
    )

    print(f"\nFiltering results:")
    print(f"  - Empty: {filter_stats['empty']:,}")
    print(f"  - Too short: {filter_stats['too_short']:,}")
    print(f"  - Too long: {filter_stats['too_long']:,}")
    print(f"  - Length ratio mismatch: {filter_stats['ratio_mismatch']:,}")
    print(f"  - Kept: {filter_stats['kept']:,} ({100*filter_stats['kept']/filter_stats['total']:.1f}%)")

    # Compute statistics
    if filtered_ko:
        ko_lengths = [len(s) for s in filtered_ko]
        en_lengths = [len(s) for s in filtered_en]

        stats = {
            'num_pairs': len(filtered_ko),
            'sources': source_stats,
            'korean': {
                'min_length': min(ko_lengths),
                'max_length': max(ko_lengths),
                'avg_length': sum(ko_lengths) / len(ko_lengths),
            },
            'english': {
                'min_length': min(en_lengths),
                'max_length': max(en_lengths),
                'avg_length': sum(en_lengths) / len(en_lengths),
            },
            'filtering': filter_stats
        }

        print(f"\nFinal statistics:")
        print(f"  Korean:  {stats['korean']['avg_length']:.1f} chars (min: {stats['korean']['min_length']}, max: {stats['korean']['max_length']})")
        print(f"  English: {stats['english']['avg_length']:.1f} chars (min: {stats['english']['min_length']}, max: {stats['english']['max_length']})")
    else:
        stats = None

    return filtered_ko, filtered_en, stats


def save_unified_data(ko_lines, en_lines, output_prefix):
    """Save unified data to files."""
    ko_path = f"{output_prefix}.ko"
    en_path = f"{output_prefix}.en"

    with open(ko_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(ko_lines) + '\n')

    with open(en_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(en_lines) + '\n')

    print(f"✓ Saved to {output_prefix}.{{ko,en}}")


def main():
    """Main preprocessing pipeline."""
    config = BaseConfig()

    print("=" * 60)
    print("Korean-English Data Merging and Preprocessing")
    print("=" * 60)
    print()
    print(f"Configuration:")
    print(f"  Min length: {config.min_length}")
    print(f"  Max length: {config.max_length}")
    print(f"  Max ratio: {config.length_ratio}")
    print()

    # Setup paths
    script_dir = Path(__file__).parent
    raw_dir = script_dir.parent / config.raw_data_dir
    processed_dir = script_dir.parent / config.processed_data_dir
    processed_dir.mkdir(parents=True, exist_ok=True)

    # Load all datasets organized by split
    print(f"Loading datasets from {raw_dir}...")
    print()
    datasets_by_split = load_datasets_from_raw(raw_dir)

    if not datasets_by_split:
        print("\n✗ No datasets found. Please run download_data.py first.")
        return

    # Process each split
    all_stats = {}

    for split in ['train', 'validation', 'test']:
        if split not in datasets_by_split or not datasets_by_split[split]:
            print(f"\n{'=' * 60}")
            print(f"Processing: {split}")
            print('=' * 60)
            print(f"No datasets available for {split} split")
            continue

        # Use more lenient settings for validation/test
        if split in ['validation', 'test']:
            min_len, max_len, max_ratio = 1, 300, 5.0
        else:
            min_len = config.min_length
            max_len = config.max_length
            max_ratio = config.length_ratio

        # Merge and process
        ko_lines, en_lines, stats = merge_and_process_split(
            split, datasets_by_split[split], min_len, max_len, max_ratio
        )

        if ko_lines and en_lines:
            # Save unified data
            output_prefix = processed_dir / split
            save_unified_data(ko_lines, en_lines, output_prefix)
            all_stats[split] = stats

    # Save overall statistics
    stats_path = processed_dir / "statistics.json"
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(all_stats, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 60)
    print("Preprocessing Complete!")
    print("=" * 60)
    print("\nUnified dataset summary:")

    for split in ['train', 'validation', 'test']:
        if split in all_stats and all_stats[split]:
            print(f"\n{split.upper()}:")
            print(f"  Total pairs: {all_stats[split]['num_pairs']:,}")
            print(f"  Sources:")
            for source, count in all_stats[split]['sources'].items():
                print(f"    - {source}: {count:,}")

    print(f"\n✓ Statistics saved to {stats_path}")
    print(f"\n✓ Files saved to: {processed_dir}/")
    print("  - train.ko, train.en")
    print("  - validation.ko, validation.en")
    print("  - test.ko, test.en")
    print("\nNext step: Train tokenizers")
    print("  /home/arnold/venv/bin/python scripts/train_tokenizer.py")


if __name__ == "__main__":
    main()
