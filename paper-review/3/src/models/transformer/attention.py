"""Multi-head attention mechanism."""

import torch
import torch.nn as nn
import math

class MultiHeadAttention(nn.Module):
    """Multi-head attention from 'Attention Is All You Need'."""

    def __init__(self, d_model, num_heads, dropout=0.1):
        """
        Args:
            d_model: Model dimension
            num_heads: Number of attention heads
            dropout: Dropout probability
        """
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.d_v = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        """
        Args:
            Q: Queries [batch_size, num_heads, seq_len, d_k]
            K: Keys [batch_size, num_heads, seq_len2, d_k]
            V: Values [batch_size, num_heads, seq_len2, d_v]
            mask: Attention mask [batch_size, 1, seq_len, seq_len2]

        Returns:
            output: [batch_size, num_heads, seq_len, d_v]
            attn: [batch_size, num_heads, seq_len, seq_len2]
        """

        # [batch_size, num_heads, seq_len, seq_len2]
        scores = (Q @ K.transpose(-2, -1)) / math.sqrt(self.d_k) 
        if mask is not None:
            scores = scores.masked_fill(mask==0, float('-inf'))
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        # [batch_size, num_heads, seq_len, d_v]
        output = attn @ V
        return output, attn

    def forward(self, query, key, value, mask=None, cache=None, use_cache=False):
        """
        Forward pass with optional KV caching for efficient inference.

        Args:
            query: [batch_size, query_len, d_model]
            key: [batch_size, key_len, d_model]
            value: [batch_size, value_len, d_model]
            mask: [batch_size, 1, query_len, key_len] or compatible
            cache: Optional dict with 'key' and 'value' tensors from previous steps
                   'key': [batch_size, cached_len, d_model]
                   'value': [batch_size, cached_len, d_model]
            use_cache: Whether to return updated cache

        Returns:
            output: [batch_size, query_len, d_model]
            attn: [batch_size, num_heads, query_len, full_key_len]
            new_cache: Updated cache if use_cache=True, else None
        """

        batch_size = query.size(0)
        query_len = query.size(1)

        # Project query (always computed for new positions)
        Q = self.W_q(query)  # [batch_size, query_len, d_model]

        # Project key and value
        K = self.W_k(key)  # [batch_size, key_len, d_model]
        V = self.W_v(value)  # [batch_size, value_len, d_model]

        # If cache is provided, concatenate with cached K, V (for self-attention)
        if cache is not None:
            # Concatenate: [cached_len, d_model] + [key_len, d_model] -> [full_len, d_model]
            K = torch.cat([cache['key'], K], dim=1)
            V = torch.cat([cache['value'], V], dim=1)

        # Get dimensions after concatenation
        key_len = K.size(1)

        # Reshape for multi-head attention
        Q = Q.view(batch_size, query_len, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, key_len, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, key_len, self.num_heads, self.d_v).transpose(1, 2)

        # Compute attention
        # [batch_size, num_heads, query_len, d_v]
        context, attn = self.scaled_dot_product_attention(Q, K, V, mask)

        # Reshape output
        context = context.transpose(1, 2).contiguous()
        context = context.view(batch_size, query_len, self.d_model)

        # Project output
        # [batch_size, query_len, d_model]
        output = self.W_o(context)
        output = self.dropout(output)

        # Return cache if requested
        if use_cache:
            # Store full K, V (including cached + new) in original d_model form
            new_cache = {
                'key': K.transpose(1, 2).contiguous().view(batch_size, key_len, self.d_model),
                'value': V.transpose(1, 2).contiguous().view(batch_size, key_len, self.d_model)
            }
            return output, attn, new_cache

        return output, attn, None
