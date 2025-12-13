"""Transformer decoder implementation."""

import torch.nn as nn
from .attention import MultiHeadAttention
from .feedforward import PositionWiseFeedForward


class DecoderLayer(nn.Module):
    """Single decoder layer with masked self-attention, cross-attention, and FFN."""

    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        """
        Args:
            d_model: Model dimension
            num_heads: Number of attention heads
            d_ff: Hidden dimension of feed-forward network
            dropout: Dropout probability
        """
        super().__init__()

        # Masked self-attention (for target sequence)
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)

        # Cross-attention (attend to encoder output)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)

        # Position-wise feed-forward network
        self.ffn = PositionWiseFeedForward(d_model, d_ff, dropout)

        # Layer normalization (3 norms for 3 sub-layers)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        # Dropout for residual connections
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        # Store attention weights for visualization (set by external flag)
        self.self_attn_weights = None
        self.cross_attn_weights = None
        self.store_attention = False  # Enable via set_store_attention()

    def forward(self, x, encoder_output, cross_mask=None, tgt_mask=None,
                self_attn_cache=None, cross_attn_cache=None, use_cache=False):
        """
        Forward pass with optional KV caching.

        Args:
            x: Target embeddings [batch_size, tgt_len, d_model]
            encoder_output: Encoder output [batch_size, src_len, d_model]
            cross_mask: Cross-attention mask [batch_size, 1, tgt_len, src_len]
            tgt_mask: Target causal mask [batch_size, 1, tgt_len, tgt_len]
            self_attn_cache: Cache for self-attention (dict with 'key', 'value')
            cross_attn_cache: Cache for cross-attention (dict with 'key', 'value')
            use_cache: Whether to return updated caches

        Returns:
            output: [batch_size, tgt_len, d_model]
            new_self_attn_cache: Updated self-attention cache if use_cache=True
            new_cross_attn_cache: Updated cross-attention cache if use_cache=True
        """
        # 1. Masked self-attention on target sequence
        # Q, K, V all come from target (x)
        # Shape: [batch_size, tgt_len, d_model]
        attn_output, self_attn, new_self_attn_cache = self.self_attn(
            x, x, x, tgt_mask,
            cache=self_attn_cache,
            use_cache=use_cache
        )
        if self.store_attention:
            self.self_attn_weights = self_attn  # Store for visualization
        attn_output = self.dropout1(attn_output)  # [batch_size, tgt_len, d_model]
        x = self.norm1(x + attn_output)           # [batch_size, tgt_len, d_model]

        # 2. Cross-attention to encoder output
        # Q comes from decoder (x), K and V come from encoder
        # Note: We don't cache cross-attention K, V since encoder projections are fast
        # The real speedup comes from self-attention caching
        cross_attn_output, cross_attn, _ = self.cross_attn(
            x, encoder_output, encoder_output,
            cross_mask,
            cache=None,
            use_cache=False
        )
        if self.store_attention:
            self.cross_attn_weights = cross_attn  # Store for visualization
        new_cross_attn_cache = None
        cross_attn_output = self.dropout2(cross_attn_output)  # [batch_size, tgt_len, d_model]
        x = self.norm2(x + cross_attn_output)                 # [batch_size, tgt_len, d_model]

        # 3. Feed-forward network
        # Shape: [batch_size, tgt_len, d_model]
        ffn_output = self.ffn(x)                  # [batch_size, tgt_len, d_model]
        ffn_output = self.dropout3(ffn_output)    # [batch_size, tgt_len, d_model]
        x = self.norm3(x + ffn_output)            # [batch_size, tgt_len, d_model]

        if use_cache:
            return x, new_self_attn_cache, new_cross_attn_cache
        return x, None, None

    def set_store_attention(self, store=True):
        """
        Enable or disable attention weight storage for visualization.

        Args:
            store: If True, store attention weights during forward pass
        """
        self.store_attention = store


class TransformerDecoder(nn.Module):
    """Transformer decoder stack."""

    def __init__(self, config):
        """
        Args:
            config: Configuration object with model parameters
        """
        super().__init__()

        # Stack of decoder layers
        self.layers = nn.ModuleList([
            DecoderLayer(
                d_model=config.d_model,
                num_heads=config.num_heads,
                d_ff=config.d_ff,
                dropout=config.dropout
            )
            for _ in range(config.num_decoder_layers)
        ])

        self.num_layers = config.num_decoder_layers

    def forward(self, x, encoder_output, cross_mask=None, tgt_mask=None,
                layer_caches=None, use_cache=False):
        """
        Forward pass through decoder stack with optional KV caching.

        Args:
            x: Target embeddings [batch_size, tgt_len, d_model]
            encoder_output: Encoder output [batch_size, src_len, d_model]
            cross_mask: Cross-attention mask [batch_size, 1, tgt_len, src_len]
            tgt_mask: Target causal mask [batch_size, 1, tgt_len, tgt_len]
            layer_caches: List of caches for each layer (length = num_layers)
                          Each element is a dict: {'self_attn': {...}, 'cross_attn': {...}}
            use_cache: Whether to return updated caches

        Returns:
            decoder_output: [batch_size, tgt_len, d_model]
            new_layer_caches: Updated caches for all layers if use_cache=True
        """
        new_layer_caches = [] if use_cache else None

        # Pass through all decoder layers sequentially
        for i, layer in enumerate(self.layers):
            # Get cache for this layer
            layer_cache = layer_caches[i] if layer_caches is not None else None
            self_attn_cache = layer_cache.get('self_attn', None) if layer_cache else None
            cross_attn_cache = layer_cache.get('cross_attn', None) if layer_cache else None

            # Forward through layer
            x, new_self_attn_cache, new_cross_attn_cache = layer(
                x, encoder_output,
                cross_mask, tgt_mask,
                self_attn_cache, cross_attn_cache,
                use_cache
            )

            # Store updated caches
            if use_cache:
                new_layer_caches.append({
                    'self_attn': new_self_attn_cache,
                    'cross_attn': new_cross_attn_cache
                })

        if use_cache:
            return x, new_layer_caches
        return x, None

    def set_store_attention(self, store=True):
        """
        Enable or disable attention weight storage for all decoder layers.

        Args:
            store: If True, store attention weights during forward pass
        """
        for layer in self.layers:
            layer.set_store_attention(store)
