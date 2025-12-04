"""Embedding layer with scaling."""

import torch.nn as nn
import math


class Embeddings(nn.Module):
    """Token embeddings with scaling factor."""

    def __init__(self, vocab_size, d_model):
        """
        Args:
            vocab_size: Vocabulary size
            d_model: Model dimension
        """
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model

    def forward(self, x):
        """
        Args:
            x: Token indices [batch_size, seq_len]

        Returns:
            embeddings: [batch_size, seq_len, d_model]
        """
        # Scale embeddings by sqrt(d_model) as per paper
        return self.embedding(x) * math.sqrt(self.d_model)
