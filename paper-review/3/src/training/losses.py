"""Loss functions for training."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LabelSmoothingLoss(nn.Module):
    """Label smoothing loss for better generalization."""

    def __init__(self, vocab_size, pad_idx, smoothing=0.1):
        """
        Args:
            vocab_size: Size of vocabulary
            pad_idx: Index of padding token
            smoothing: Smoothing factor
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.pad_idx = pad_idx
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing

    def forward(self, logits, targets):
        """
        Args:
            logits: Model predictions [batch_size * seq_len, vocab_size]
            targets: Ground truth labels [batch_size * seq_len]

        Returns:
            loss: Scalar loss value
        """
        # Create smooth label distribution
        batch_size = logits.size(0)

        # Create smoothed target distribution
        # Start with uniform distribution over all tokens except padding
        true_dist = torch.zeros_like(logits)
        true_dist.fill_(self.smoothing / (self.vocab_size - 2))  # -2 for target and pad

        # Set correct token to confidence value
        true_dist.scatter_(1, targets.unsqueeze(1), self.confidence)

        # Zero out padding token probability
        true_dist[:, self.pad_idx] = 0

        # Mask positions where target is padding
        mask = (targets != self.pad_idx)
        true_dist = true_dist * mask.unsqueeze(1)

        # Compute KL divergence loss
        log_probs = F.log_softmax(logits, dim=1)
        loss = F.kl_div(log_probs, true_dist, reduction='sum')

        # Normalize by number of non-padding tokens
        num_tokens = mask.sum().item()
        if num_tokens > 0:
            loss = loss / num_tokens

        return loss
