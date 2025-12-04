#!/usr/bin/env python
"""Test checkpoint cleanup functionality."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import shutil
import torch
from pathlib import Path

from src.utils.checkpoint_cleanup import (
    get_directory_size,
    get_checkpoint_info,
    cleanup_checkpoints,
    should_cleanup
)


def create_dummy_checkpoint(path: str, size_mb: float = 1.0):
    """Create a dummy checkpoint file of specified size."""
    # Create random tensor data
    size_bytes = int(size_mb * 1024 * 1024)
    num_elements = size_bytes // 4  # float32 = 4 bytes

    dummy_data = {
        'model_state_dict': torch.randn(num_elements),
        'optimizer_state_dict': {},
        'epoch': 1,
        'loss': 1.0
    }

    torch.save(dummy_data, path)


def test_checkpoint_cleanup():
    """Test checkpoint cleanup functionality."""
    print("=" * 80)
    print("Testing Checkpoint Cleanup")
    print("=" * 80)
    print()

    # Create test directory
    test_dir = 'test_checkpoints'
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)
    os.makedirs(test_dir)

    try:
        # Create dummy checkpoints
        print("Creating dummy checkpoints...")
        print()

        # Best models (never delete)
        create_dummy_checkpoint(os.path.join(test_dir, 'best_model.pt'), size_mb=5.0)
        create_dummy_checkpoint(os.path.join(test_dir, 'best_bleu_model.pt'), size_mb=5.0)

        # Periodic checkpoints (can be deleted)
        for i in range(1, 11):
            create_dummy_checkpoint(
                os.path.join(test_dir, f'checkpoint_epoch_{i}.pt'),
                size_mb=10.0
            )

        print(f"Created:")
        print(f"  - 2 best models (5 MB each)")
        print(f"  - 10 periodic checkpoints (10 MB each)")
        print()

        # Get initial info
        initial_size = get_directory_size(test_dir)
        print(f"Initial directory size: {initial_size:.2f} GB ({initial_size*1024:.0f} MB)")
        print()

        # Get checkpoint info
        info = get_checkpoint_info(test_dir)
        print(f"Found:")
        print(f"  - {len(info['best'])} best model(s)")
        print(f"  - {len(info['periodic'])} periodic checkpoint(s)")
        print()

        # Test 1: Keep only 3 recent checkpoints
        print("-" * 80)
        print("Test 1: Keep only 3 most recent periodic checkpoints")
        print("-" * 80)

        stats = cleanup_checkpoints(
            test_dir,
            max_size_gb=None,
            keep_n_recent=3,
            dry_run=False,
            verbose=True
        )

        assert stats['kept_best'] == 2, "Should keep 2 best models"
        assert stats['kept_periodic'] == 3, "Should keep 3 periodic checkpoints"
        assert len(stats['deleted_files']) == 7, "Should delete 7 old checkpoints"

        print("✓ Test 1 passed!")
        print()

        # Recreate for next test
        shutil.rmtree(test_dir)
        os.makedirs(test_dir)
        create_dummy_checkpoint(os.path.join(test_dir, 'best_model.pt'), size_mb=5.0)
        create_dummy_checkpoint(os.path.join(test_dir, 'best_bleu_model.pt'), size_mb=5.0)
        for i in range(1, 11):
            create_dummy_checkpoint(
                os.path.join(test_dir, f'checkpoint_epoch_{i}.pt'),
                size_mb=10.0
            )

        # Test 2: Cleanup based on max size
        print("-" * 80)
        print("Test 2: Cleanup to stay under 0.05 GB (50 MB)")
        print("-" * 80)

        stats = cleanup_checkpoints(
            test_dir,
            max_size_gb=0.05,  # 50 MB
            keep_n_recent=3,
            dry_run=False,
            verbose=True
        )

        final_size = get_directory_size(test_dir)
        print(f"Final size: {final_size:.4f} GB ({final_size*1024:.1f} MB)")

        assert final_size <= 0.05, f"Directory should be under 50 MB, got {final_size*1024:.1f} MB"
        assert stats['kept_best'] == 2, "Should keep 2 best models"

        print("✓ Test 2 passed!")
        print()

        # Test 3: Dry run
        shutil.rmtree(test_dir)
        os.makedirs(test_dir)
        create_dummy_checkpoint(os.path.join(test_dir, 'best_model.pt'), size_mb=5.0)
        for i in range(1, 6):
            create_dummy_checkpoint(
                os.path.join(test_dir, f'checkpoint_epoch_{i}.pt'),
                size_mb=10.0
            )

        print("-" * 80)
        print("Test 3: Dry run (no actual deletion)")
        print("-" * 80)

        size_before = get_directory_size(test_dir)

        stats = cleanup_checkpoints(
            test_dir,
            max_size_gb=None,
            keep_n_recent=2,
            dry_run=True,
            verbose=True
        )

        size_after = get_directory_size(test_dir)

        assert size_before == size_after, "Dry run should not change directory size"
        assert len(stats['deleted_files']) == 3, "Should report 3 files for deletion"

        # Verify files still exist
        for i in range(1, 6):
            filepath = os.path.join(test_dir, f'checkpoint_epoch_{i}.pt')
            assert os.path.exists(filepath), f"File should still exist: {filepath}"

        print("✓ Test 3 passed!")
        print()

        # Test 4: should_cleanup function
        print("-" * 80)
        print("Test 4: should_cleanup detection")
        print("-" * 80)

        current_size = get_directory_size(test_dir)
        print(f"Current size: {current_size:.4f} GB")

        result1 = should_cleanup(test_dir, max_size_gb=0.001)  # Very small limit
        print(f"Should cleanup (limit 0.001 GB): {result1}")
        assert result1 == True, "Should need cleanup with tiny limit"

        result2 = should_cleanup(test_dir, max_size_gb=100.0)  # Very large limit
        print(f"Should cleanup (limit 100 GB): {result2}")
        assert result2 == False, "Should not need cleanup with large limit"

        print("✓ Test 4 passed!")
        print()

        # Test 5: Never delete best models
        print("-" * 80)
        print("Test 5: Best models are never deleted")
        print("-" * 80)

        shutil.rmtree(test_dir)
        os.makedirs(test_dir)
        create_dummy_checkpoint(os.path.join(test_dir, 'best_model.pt'), size_mb=20.0)
        create_dummy_checkpoint(os.path.join(test_dir, 'best_bleu_model.pt'), size_mb=20.0)
        create_dummy_checkpoint(os.path.join(test_dir, 'checkpoint_epoch_1.pt'), size_mb=5.0)

        # Try to cleanup to 0.01 GB (10 MB) - best models are 40 MB total
        stats = cleanup_checkpoints(
            test_dir,
            max_size_gb=0.01,
            keep_n_recent=1,
            dry_run=False,
            verbose=True
        )

        # Verify best models still exist
        assert os.path.exists(os.path.join(test_dir, 'best_model.pt')), "best_model.pt should still exist"
        assert os.path.exists(os.path.join(test_dir, 'best_bleu_model.pt')), "best_bleu_model.pt should still exist"

        # Periodic checkpoint should be deleted
        assert not os.path.exists(os.path.join(test_dir, 'checkpoint_epoch_1.pt')), "periodic checkpoint should be deleted"

        print("✓ Test 5 passed!")
        print()

        print("=" * 80)
        print("All Checkpoint Cleanup Tests Passed!")
        print("=" * 80)

    finally:
        # Cleanup test directory
        if os.path.exists(test_dir):
            shutil.rmtree(test_dir)
            print(f"\nCleaned up test directory: {test_dir}")


if __name__ == "__main__":
    test_checkpoint_cleanup()
