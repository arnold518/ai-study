"""Model checkpointing utilities."""

import torch
import os


def save_checkpoint(model, optimizer, epoch, loss, path):
    """
    Save model checkpoint.

    Args:
        model: Model to save
        optimizer: Optimizer state
        epoch: Current epoch
        loss: Current loss
        path: Path to save checkpoint
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)

    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }

    torch.save(checkpoint, path)
    print(f"Checkpoint saved to {path}")


def load_checkpoint(model, optimizer, path, device):
    """
    Load model checkpoint.

    Args:
        model: Model to load weights into
        optimizer: Optimizer to load state into
        path: Path to checkpoint
        device: Device to load to

    Returns:
        epoch: Epoch number from checkpoint
        loss: Loss from checkpoint
    """
    checkpoint = torch.load(path, map_location=device)

    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']

    print(f"Checkpoint loaded from {path} (epoch {epoch})")
    return epoch, loss
