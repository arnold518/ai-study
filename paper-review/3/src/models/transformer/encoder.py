"""Transformer encoder implementation."""

import torch.nn as nn
from .attention import MultiHeadAttention
from .feedforward import PositionWiseFeedForward


class EncoderLayer(nn.Module):
    """Single encoder layer with self-attention and feed-forward network."""

    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        """
        Args:
            d_model: Model dimension
            num_heads: Number of attention heads
            d_ff: Hidden dimension of feed-forward network
            dropout: Dropout probability
        """
        super().__init__()

        # Multi-head self-attention
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)

        # Position-wise feed-forward network
        self.ffn = PositionWiseFeedForward(d_model, d_ff, dropout)

        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # Dropout for residual connections
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        """
        Args:
            x: Input [batch_size, seq_len, d_model]
            mask: Self-attention mask [batch_size, 1, seq_len, seq_len]
                  (padding mask for encoder self-attention)

        Returns:
            output: [batch_size, seq_len, d_model]
        """
        # Self-attention with residual connection and layer norm
        # x = LayerNorm(x + Attention(x))
        # Note: attention now returns (output, attn, cache), but encoder doesn't use cache
        attn_output, _, _ = self.self_attn(x, x, x, mask)
        attn_output = self.dropout1(attn_output)
        x = self.norm1(x + attn_output)

        # Feed-forward with residual connection and layer norm
        # x = LayerNorm(x + FFN(x))
        ffn_output = self.ffn(x)
        ffn_output = self.dropout2(ffn_output)
        x = self.norm2(x + ffn_output)

        return x


class TransformerEncoder(nn.Module):
    """Transformer encoder stack."""

    def __init__(self, config):
        """
        Args:
            config: Configuration object with model parameters
        """
        super().__init__()

        # Stack of encoder layers
        self.layers = nn.ModuleList([
            EncoderLayer(
                d_model=config.d_model,
                num_heads=config.num_heads,
                d_ff=config.d_ff,
                dropout=config.dropout
            )
            for _ in range(config.num_encoder_layers)
        ])

        self.num_layers = config.num_encoder_layers

    def forward(self, x, mask=None):
        """
        Args:
            x: Input embeddings [batch_size, seq_len, d_model]
            mask: Attention mask [batch_size, 1, seq_len, seq_len]

        Returns:
            encoder_output: [batch_size, seq_len, d_model]
        """
        # Pass through all encoder layers sequentially
        for layer in self.layers:
            x = layer(x, mask)

        return x
