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
        self.optimizer = optim.Adam(
            model_params,
            lr=0.0,
            betas=(0.9, 0.98),
            eps=1e-9,
        )
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.factor = factor
        self.step_num = 0

    # >>> This is the important bit for GradScaler <<<
    @property
    def param_groups(self):
        """Expose inner optimizer's param_groups so AMP/GradScaler can see params."""
        return self.optimizer.param_groups

    def step(self):
        """Update parameters and learning rate."""
        self.step_num += 1
        lr = self._get_lr()
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr
        self.optimizer.step()

    def zero_grad(self):
        """Zero out gradients."""
        self.optimizer.zero_grad()

    def _get_lr(self):
        """Calculate learning rate for current step.

        Implements Noam learning rate schedule from the paper:
        lrate = d_model^(-0.5) * min(step^(-0.5), step * warmup^(-1.5))
        """
        return self.factor * (
            self.d_model ** (-0.5)
            * min(
                self.step_num ** (-0.5),
                self.step_num * self.warmup_steps ** (-1.5),
            )
        )

    # Optional but nice for checkpointing
    def state_dict(self):
        return {
            "optimizer": self.optimizer.state_dict(),
            "d_model": self.d_model,
            "warmup_steps": self.warmup_steps,
            "factor": self.factor,
            "step_num": self.step_num,
        }

    def load_state_dict(self, state):
        self.optimizer.load_state_dict(state["optimizer"])
        self.d_model = state["d_model"]
        self.warmup_steps = state["warmup_steps"]
        self.factor = state["factor"]
        self.step_num = state["step_num"]
