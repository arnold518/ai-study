"""Checkpoint cleanup utilities to manage disk space."""

import os
import glob
from pathlib import Path
from typing import List, Dict, Tuple


def get_directory_size(directory: str) -> float:
    """
    Calculate total size of directory in GB.

    Args:
        directory: Path to directory

    Returns:
        Size in GB
    """
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(directory):
        for filename in filenames:
            filepath = os.path.join(dirpath, filename)
            if os.path.exists(filepath):
                total_size += os.path.getsize(filepath)

    return total_size / (1024 ** 3)  # Convert bytes to GB


def get_checkpoint_info(checkpoint_dir: str) -> Dict[str, List[Tuple[str, float, float]]]:
    """
    Get information about all checkpoints in directory.

    Args:
        checkpoint_dir: Path to checkpoint directory

    Returns:
        Dictionary with:
            'best': List of (path, size_gb, mtime) for best models
            'periodic': List of (path, size_gb, mtime) for periodic checkpoints
    """
    if not os.path.exists(checkpoint_dir):
        return {'best': [], 'periodic': []}

    best_models = []
    periodic_checkpoints = []

    # Find all checkpoint files
    checkpoint_files = glob.glob(os.path.join(checkpoint_dir, '*.pt'))

    for filepath in checkpoint_files:
        filename = os.path.basename(filepath)
        size_gb = os.path.getsize(filepath) / (1024 ** 3)
        mtime = os.path.getmtime(filepath)

        # Classify checkpoint
        if 'best' in filename.lower():
            # Best model (best_model.pt, best_bleu_model.pt, etc.)
            best_models.append((filepath, size_gb, mtime))
        elif 'checkpoint_epoch' in filename:
            # Periodic checkpoint
            periodic_checkpoints.append((filepath, size_gb, mtime))
        else:
            # Other checkpoint (treat as periodic)
            periodic_checkpoints.append((filepath, size_gb, mtime))

    # Sort by modification time (newest first)
    best_models.sort(key=lambda x: x[2], reverse=True)
    periodic_checkpoints.sort(key=lambda x: x[2], reverse=True)

    return {
        'best': best_models,
        'periodic': periodic_checkpoints
    }


def cleanup_checkpoints(
    checkpoint_dir: str,
    max_size_gb: float = None,
    keep_n_recent: int = 3,
    dry_run: bool = False,
    verbose: bool = True
) -> Dict[str, any]:
    """
    Clean up checkpoint directory to save disk space.

    Strategy:
    1. NEVER delete best models (best_model.pt, best_bleu_model.pt)
    2. Keep only N most recent periodic checkpoints
    3. Delete older periodic checkpoints if directory exceeds max_size_gb

    Args:
        checkpoint_dir: Path to checkpoint directory
        max_size_gb: Maximum directory size in GB (None = unlimited)
        keep_n_recent: Number of recent periodic checkpoints to keep
        dry_run: If True, don't actually delete files (just report)
        verbose: If True, print cleanup details

    Returns:
        Dictionary with cleanup statistics:
            'initial_size_gb': Size before cleanup
            'final_size_gb': Size after cleanup
            'deleted_files': List of deleted file paths
            'deleted_size_gb': Total size deleted
            'kept_best': Number of best models kept
            'kept_periodic': Number of periodic checkpoints kept
    """
    if not os.path.exists(checkpoint_dir):
        if verbose:
            print(f"Checkpoint directory does not exist: {checkpoint_dir}")
        return {
            'initial_size_gb': 0,
            'final_size_gb': 0,
            'deleted_files': [],
            'deleted_size_gb': 0,
            'kept_best': 0,
            'kept_periodic': 0
        }

    # Get initial size
    initial_size = get_directory_size(checkpoint_dir)

    # Get checkpoint information
    checkpoint_info = get_checkpoint_info(checkpoint_dir)
    best_models = checkpoint_info['best']
    periodic_checkpoints = checkpoint_info['periodic']

    deleted_files = []
    deleted_size = 0.0

    if verbose:
        print(f"\n{'=' * 60}")
        print("Checkpoint Cleanup")
        print(f"{'=' * 60}")
        print(f"Directory: {checkpoint_dir}")
        print(f"Initial size: {initial_size:.2f} GB")
        print(f"Max size: {max_size_gb:.2f} GB" if max_size_gb else "Max size: Unlimited")
        print(f"Keep N recent: {keep_n_recent}")
        print(f"Dry run: {dry_run}")
        print()
        print(f"Found:")
        print(f"  - {len(best_models)} best model(s)")
        print(f"  - {len(periodic_checkpoints)} periodic checkpoint(s)")
        print()

    # Keep all best models (NEVER delete)
    if verbose and best_models:
        print("Best models (KEEP ALL):")
        for path, size, mtime in best_models:
            print(f"  ✓ {os.path.basename(path)} ({size:.3f} GB)")
        print()

    # Determine which periodic checkpoints to delete
    if len(periodic_checkpoints) > keep_n_recent:
        # Keep N most recent, delete the rest
        to_keep = periodic_checkpoints[:keep_n_recent]
        to_delete = periodic_checkpoints[keep_n_recent:]

        if verbose:
            print(f"Periodic checkpoints (keeping {keep_n_recent} most recent):")
            for path, size, mtime in to_keep:
                print(f"  ✓ {os.path.basename(path)} ({size:.3f} GB)")
            print()

            if to_delete:
                print(f"Periodic checkpoints to delete ({len(to_delete)}):")
                for path, size, mtime in to_delete:
                    print(f"  ✗ {os.path.basename(path)} ({size:.3f} GB)")
                print()

        # Delete old periodic checkpoints
        for path, size, mtime in to_delete:
            if not dry_run:
                try:
                    os.remove(path)
                    deleted_files.append(path)
                    deleted_size += size
                    if verbose:
                        print(f"Deleted: {os.path.basename(path)}")
                except Exception as e:
                    if verbose:
                        print(f"Error deleting {os.path.basename(path)}: {e}")
            else:
                deleted_files.append(path)
                deleted_size += size
    else:
        if verbose:
            print(f"Periodic checkpoints (all kept, {len(periodic_checkpoints)} ≤ {keep_n_recent}):")
            for path, size, mtime in periodic_checkpoints:
                print(f"  ✓ {os.path.basename(path)} ({size:.3f} GB)")
            print()

    # Check if we need to delete more based on max_size_gb
    if max_size_gb is not None:
        current_size = get_directory_size(checkpoint_dir) if not dry_run else (initial_size - deleted_size)

        if current_size > max_size_gb:
            # Still over limit, delete more periodic checkpoints
            remaining_periodic = [p for p in periodic_checkpoints[:keep_n_recent]]

            if verbose:
                print(f"\nDirectory still over limit ({current_size:.2f} GB > {max_size_gb:.2f} GB)")
                print(f"Deleting additional periodic checkpoints...")
                print()

            # Delete from oldest to newest until under limit
            for path, size, mtime in reversed(remaining_periodic):
                if current_size <= max_size_gb:
                    break

                if not dry_run:
                    try:
                        os.remove(path)
                        deleted_files.append(path)
                        deleted_size += size
                        current_size -= size
                        if verbose:
                            print(f"Deleted: {os.path.basename(path)} ({size:.3f} GB)")
                    except Exception as e:
                        if verbose:
                            print(f"Error deleting {os.path.basename(path)}: {e}")
                else:
                    deleted_files.append(path)
                    deleted_size += size
                    current_size -= size

    # Get final size
    final_size = get_directory_size(checkpoint_dir) if not dry_run else (initial_size - deleted_size)

    # Summary
    if verbose:
        print()
        print(f"{'=' * 60}")
        print("Cleanup Summary")
        print(f"{'=' * 60}")
        print(f"Initial size: {initial_size:.2f} GB")
        print(f"Final size: {final_size:.2f} GB")
        print(f"Space freed: {deleted_size:.2f} GB ({100*deleted_size/initial_size:.1f}%)" if initial_size > 0 else "Space freed: 0 GB")
        print(f"Files deleted: {len(deleted_files)}")
        print(f"Best models kept: {len(best_models)}")
        print(f"Periodic checkpoints kept: {len(periodic_checkpoints) - len([p for p in deleted_files if 'checkpoint_epoch' in p])}")
        print(f"{'=' * 60}")
        print()

    return {
        'initial_size_gb': initial_size,
        'final_size_gb': final_size,
        'deleted_files': deleted_files,
        'deleted_size_gb': deleted_size,
        'kept_best': len(best_models),
        'kept_periodic': len(periodic_checkpoints) - len([p for p in deleted_files if 'checkpoint_epoch' in p]),
        'freed_space_gb': deleted_size,
        'freed_space_percent': 100 * deleted_size / initial_size if initial_size > 0 else 0
    }


def should_cleanup(checkpoint_dir: str, max_size_gb: float) -> bool:
    """
    Check if cleanup is needed.

    Args:
        checkpoint_dir: Path to checkpoint directory
        max_size_gb: Maximum directory size in GB

    Returns:
        True if cleanup is needed
    """
    if not os.path.exists(checkpoint_dir):
        return False

    current_size = get_directory_size(checkpoint_dir)
    return current_size > max_size_gb
