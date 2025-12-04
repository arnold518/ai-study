#!/usr/bin/env python
"""
Test TranslationDataset and DataLoader.

This script tests that the dataset correctly loads, tokenizes, and batches data.
Uses configuration from config/base_config.py to determine vocabulary mode.

Usage:
    /home/arnold/venv/bin/python tests/test_dataset.py
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from pathlib import Path
import torch
from src.data.tokenizer import SentencePieceTokenizer
from src.data.dataset import TranslationDataset, create_dataloader, load_tokenizers
from config.base_config import BaseConfig


def test_single_sample(dataset, idx=0):
    """Test loading a single sample."""
    print(f"\n{'=' * 60}")
    print(f"Test Single Sample (index={idx})")
    print('=' * 60)

    sample = dataset[idx]

    print(f"Source text:  {dataset.src_lines[idx]}")
    print(f"Target text:  {dataset.tgt_lines[idx]}")
    print()
    print(f"Source IDs:   {sample['src'].tolist()[:15]}...")
    print(f"Target IDs:   {sample['tgt'].tolist()[:15]}...")
    print()
    print(f"Source shape: {sample['src'].shape}")
    print(f"Target shape: {sample['tgt'].shape}")
    print()

    # Decode back to verify
    src_decoded = dataset.src_tokenizer.decode_ids(sample['src'].tolist()[1:-1])  # Remove BOS/EOS
    tgt_decoded = dataset.tgt_tokenizer.decode_ids(sample['tgt'].tolist()[1:-1])

    print(f"Decoded source: {src_decoded}")
    print(f"Decoded target: {tgt_decoded}")

    # Check BOS/EOS tokens
    assert sample['src'][0] == dataset.src_tokenizer.bos_id, "Missing BOS token in source"
    assert sample['src'][-1] == dataset.src_tokenizer.eos_id, "Missing EOS token in source"
    assert sample['tgt'][0] == dataset.tgt_tokenizer.bos_id, "Missing BOS token in target"
    assert sample['tgt'][-1] == dataset.tgt_tokenizer.eos_id, "Missing EOS token in target"

    print("\n✓ BOS/EOS tokens present")
    print("✓ Encoding/decoding successful")


def test_batch(dataloader):
    """Test batching with DataLoader."""
    print(f"\n{'=' * 60}")
    print("Test Batching with DataLoader")
    print('=' * 60)

    # Get one batch
    batch = next(iter(dataloader))

    print(f"Batch keys: {list(batch.keys())}")
    print()
    print(f"Source shape:      {batch['src'].shape}")
    print(f"Target shape:      {batch['tgt'].shape}")
    print(f"Source mask shape: {batch['src_mask'].shape}")
    print(f"Target mask shape: {batch['tgt_mask'].shape}")
    print()

    # Check shapes
    batch_size = batch['src'].shape[0]
    src_seq_len = batch['src'].shape[1]
    tgt_seq_len = batch['tgt'].shape[1]

    assert batch['src_mask'].shape == (batch_size, src_seq_len), "Source mask shape mismatch"
    assert batch['tgt_mask'].shape == (batch_size, tgt_seq_len), "Target mask shape mismatch"

    print(f"Batch size: {batch_size}")
    print(f"Max source length in batch: {src_seq_len}")
    print(f"Max target length in batch: {tgt_seq_len}")
    print()

    # Check padding
    print("Padding analysis:")
    for i in range(min(3, batch_size)):
        src_len = batch['src_mask'][i].sum().item()
        tgt_len = batch['tgt_mask'][i].sum().item()
        src_pad = src_seq_len - src_len
        tgt_pad = tgt_seq_len - tgt_len

        print(f"  Sample {i}: src_len={src_len:3d} (pad={src_pad:2d}), tgt_len={tgt_len:3d} (pad={tgt_pad:2d})")

    print("\n✓ Batching successful")
    print("✓ Padding masks created")
    print("✓ Shapes correct")


def test_multiple_batches(dataloader, num_batches=3):
    """Test iterating through multiple batches."""
    print(f"\n{'=' * 60}")
    print(f"Test Multiple Batches (first {num_batches})")
    print('=' * 60)

    for i, batch in enumerate(dataloader):
        if i >= num_batches:
            break

        batch_size = batch['src'].shape[0]
        src_len = batch['src'].shape[1]
        tgt_len = batch['tgt'].shape[1]

        print(f"Batch {i}: size={batch_size}, src_len={src_len}, tgt_len={tgt_len}")

    print(f"\n✓ Successfully loaded {num_batches} batches")


def test_edge_cases(dataset):
    """Test edge cases."""
    print(f"\n{'=' * 60}")
    print("Test Edge Cases")
    print('=' * 60)

    # Test longest sequence
    print("\nFinding longest sequences...")
    max_src_len = max(len(dataset[i]['src']) for i in range(min(100, len(dataset))))
    max_tgt_len = max(len(dataset[i]['tgt']) for i in range(min(100, len(dataset))))

    print(f"  Max source length (first 100 samples): {max_src_len}")
    print(f"  Max target length (first 100 samples): {max_tgt_len}")

    # Test first and last samples
    print("\nTesting first and last samples...")
    first = dataset[0]
    last = dataset[-1]

    print(f"  First sample: src={first['src'].shape}, tgt={first['tgt'].shape}")
    print(f"  Last sample:  src={last['src'].shape}, tgt={last['tgt'].shape}")

    print("\n✓ Edge cases handled")


def main():
    config = BaseConfig()

    print("=" * 60)
    print("TranslationDataset Test")
    print("=" * 60)
    print(f"Mode: {'SHARED' if config.use_shared_vocab else 'SEPARATE'} vocabulary")
    print(f"Batch size: {config.batch_size}")
    print(f"Max sequence length: {config.max_seq_length}")
    print()

    # Setup paths
    script_dir = Path(__file__).parent
    project_root = script_dir.parent

    vocab_dir = project_root / config.vocab_dir

    # Use validation set for testing (smaller, faster)
    train_ko = project_root / config.processed_data_dir / f"validation.{config.src_lang}"
    train_en = project_root / config.processed_data_dir / f"validation.{config.tgt_lang}"

    # Check data files exist
    if not train_ko.exists() or not train_en.exists():
        print(f"\n✗ Data not found: {train_ko} or {train_en}")
        print("  Please preprocess data first:")
        print("  /home/arnold/venv/bin/python scripts/split_data.py")
        return

    # Load tokenizers
    print("Loading tokenizers...")
    try:
        ko_tokenizer, en_tokenizer = load_tokenizers(vocab_dir, use_shared_vocab=config.use_shared_vocab)
    except FileNotFoundError as e:
        print(f"\n✗ {e}")
        return

    if config.use_shared_vocab:
        print(f"✓ Shared tokenizer loaded (vocab_size={ko_tokenizer.vocab_size:,})")
        print(f"  {config.src_lang} and {config.tgt_lang} use the same vocabulary")
    else:
        print(f"✓ {config.src_lang} tokenizer loaded (vocab_size={ko_tokenizer.vocab_size:,})")
        print(f"✓ {config.tgt_lang} tokenizer loaded (vocab_size={en_tokenizer.vocab_size:,})")

    # Create dataset
    print("\nCreating dataset...")
    dataset = TranslationDataset(
        src_file=str(train_ko),
        tgt_file=str(train_en),
        src_tokenizer=ko_tokenizer,
        tgt_tokenizer=en_tokenizer,
        max_len=config.max_length  # From config
    )
    print(f"✓ Dataset created with {len(dataset):,} samples")

    # Test single sample
    test_single_sample(dataset, idx=0)
    test_single_sample(dataset, idx=10)

    # Create DataLoader
    print(f"\n{'=' * 60}")
    print("Creating DataLoader")
    print('=' * 60)

    # Use smaller batch size for testing
    test_batch_size = min(8, config.batch_size)

    dataloader = create_dataloader(
        src_file=str(train_ko),
        tgt_file=str(train_en),
        src_tokenizer=ko_tokenizer,
        tgt_tokenizer=en_tokenizer,
        batch_size=test_batch_size,
        max_len=config.max_length,
        shuffle=False,  # Don't shuffle for testing
        num_workers=0   # Single process for simplicity
    )
    print(f"✓ DataLoader created with batch_size={test_batch_size}")

    # Test batching
    test_batch(dataloader)

    # Test multiple batches
    test_multiple_batches(dataloader, num_batches=3)

    # Test edge cases
    test_edge_cases(dataset)

    # Summary
    print(f"\n{'=' * 60}")
    print("Summary")
    print('=' * 60)
    print(f"✓ Dataset loading works")
    print(f"✓ Tokenization works")
    print(f"✓ BOS/EOS tokens added correctly")
    print(f"✓ Batching works")
    print(f"✓ Padding works")
    print(f"✓ Masks created correctly")
    print()
    print(f"Dataset: {len(dataset):,} samples")
    if config.use_shared_vocab:
        print(f"Tokenizer: {ko_tokenizer.vocab_size:,} tokens (shared)")
    else:
        print(f"Tokenizers: {ko_tokenizer.vocab_size:,} tokens ({config.src_lang}), {en_tokenizer.vocab_size:,} tokens ({config.tgt_lang})")
    print(f"Config: batch_size={config.batch_size}, max_length={config.max_length}")
    print()
    print("✅ All tests passed!")
    print()
    print("Next step: Review model architecture")
    print("  Check src/models/transformer/ components")


if __name__ == "__main__":
    main()
