"""Alternative loss implementation for comparison."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LabelSmoothingLossExplicit(nn.Module):
    """
    Label smoothing loss with more explicit reduction.

    This version makes the reduction steps clearer:
    1. Compute KL divergence per element (no reduction)
    2. Sum over vocabulary dimension per position
    3. Mask out padding positions
    4. Average over non-padding positions
    """

    def __init__(self, vocab_size, pad_idx, smoothing=0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.pad_idx = pad_idx
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing

    def forward(self, logits, targets):
        """
        Args:
            logits: Model predictions [N, vocab_size] where N = batch * seq_len
            targets: Ground truth labels [N]

        Returns:
            loss: Scalar loss value
        """
        N, vocab_size = logits.size()

        # Create smoothed target distribution
        true_dist = torch.zeros_like(logits)
        true_dist.fill_(self.smoothing / (self.vocab_size - 2))
        true_dist.scatter_(1, targets.unsqueeze(1), self.confidence)
        true_dist[:, self.pad_idx] = 0

        # Mask positions where target is padding
        mask = (targets != self.pad_idx)  # [N]
        true_dist = true_dist * mask.unsqueeze(1)  # [N, vocab]

        # Compute KL divergence (more explicit)
        log_probs = F.log_softmax(logits, dim=1)  # [N, vocab]

        # Element-wise KL divergence: target * (log(target) - log_prob)
        kl_per_element = F.kl_div(log_probs, true_dist, reduction='none')  # [N, vocab]

        # Sum over vocabulary for each position (this is KL divergence per position)
        kl_per_position = kl_per_element.sum(dim=1)  # [N]

        # Mask out padding and compute average over non-padding positions
        num_tokens = mask.sum()
        if num_tokens > 0:
            loss = (kl_per_position * mask).sum() / num_tokens
        else:
            loss = torch.tensor(0.0, device=logits.device)

        return loss


# Original for comparison
class LabelSmoothingLoss(nn.Module):
    """Original implementation using reduction='sum'."""

    def __init__(self, vocab_size, pad_idx, smoothing=0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.pad_idx = pad_idx
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing

    def forward(self, logits, targets):
        batch_size = logits.size(0)

        true_dist = torch.zeros_like(logits)
        true_dist.fill_(self.smoothing / (self.vocab_size - 2))
        true_dist.scatter_(1, targets.unsqueeze(1), self.confidence)
        true_dist[:, self.pad_idx] = 0

        mask = (targets != self.pad_idx)
        true_dist = true_dist * mask.unsqueeze(1)

        log_probs = F.log_softmax(logits, dim=1)
        loss = F.kl_div(log_probs, true_dist, reduction='sum')

        num_tokens = mask.sum().item()
        if num_tokens > 0:
            loss = loss / num_tokens

        return loss


def test_equivalence():
    """Test that both implementations give the same result."""
    import torch

    vocab_size = 100
    N = 50  # batch * seq_len
    pad_idx = 0

    # Create random logits and targets
    logits = torch.randn(N, vocab_size)
    targets = torch.randint(1, vocab_size, (N,))
    targets[::5] = pad_idx  # Add some padding

    # Test both implementations
    loss1 = LabelSmoothingLoss(vocab_size, pad_idx, smoothing=0.1)
    loss2 = LabelSmoothingLossExplicit(vocab_size, pad_idx, smoothing=0.1)

    result1 = loss1(logits, targets)
    result2 = loss2(logits, targets)

    print(f"Original:  {result1.item():.6f}")
    print(f"Explicit:  {result2.item():.6f}")
    print(f"Difference: {abs(result1 - result2).item():.10f}")
    print(f"Equal: {torch.allclose(result1, result2)}")


if __name__ == "__main__":
    test_equivalence()
