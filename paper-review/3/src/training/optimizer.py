"""Custom optimizer with warmup for Transformer."""

import torch.optim as optim


class NoamOptimizer:
    """Optimizer with warmup learning rate schedule from the paper.

    lrate = d_model^(-0.5) * min(step_num^(-0.5), step_num * warmup_steps^(-1.5))
    """

    def __init__(self, model_params, d_model, warmup_steps, factor=1.0):
        """
        Args:
            model_params: Model parameters
            d_model: Model dimension
            warmup_steps: Number of warmup steps
            factor: Learning rate scaling factor
        """
        self.optimizer = optim.Adam(model_params, lr=0, betas=(0.9, 0.98), eps=1e-9)
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.factor = factor
        self.step_num = 0

    def step(self):
        """Update parameters and learning rate."""
        self.step_num += 1
        lr = self._get_lr()
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        self.optimizer.step()

    def _get_lr(self):
        """Calculate learning rate for current step.

        Implements Noam learning rate schedule from the paper:
        lrate = d_model^(-0.5) * min(step^(-0.5), step * warmup^(-1.5))
        """
        return self.factor * (
            self.d_model ** (-0.5) *
            min(self.step_num ** (-0.5), self.step_num * self.warmup_steps ** (-1.5))
        )

    def zero_grad(self):
        """Zero out gradients."""
        self.optimizer.zero_grad()
