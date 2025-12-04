"""Position-wise feed-forward network."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionWiseFeedForward(nn.Module):
    """Position-wise FFN: FFN(x) = max(0, xW1 + b1)W2 + b2."""

    def __init__(self, d_model, d_ff, dropout=0.1):
        """
        Args:
            d_model: Model dimension
            d_ff: Hidden dimension of feed-forward network
            dropout: Dropout probability
        """
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Args:
            x: [batch_size, seq_len, d_model]

        Returns:
            output: [batch_size, seq_len, d_model]
        """
        # First linear transformation: d_model -> d_ff
        x = self.linear1(x)

        # ReLU activation
        x = F.relu(x)

        # Dropout after activation
        x = self.dropout(x)

        # Second linear transformation: d_ff -> d_model
        x = self.linear2(x)

        return x
