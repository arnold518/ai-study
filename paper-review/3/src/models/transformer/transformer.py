"""Main Transformer model implementation."""

import torch
import torch.nn as nn
import math
from .encoder import TransformerEncoder
from .decoder import TransformerDecoder
from .positional_encoding import PositionalEncoding


class Transformer(nn.Module):
    """
    Transformer model for sequence-to-sequence tasks.

    Implements weight tying as described in Section 3.4 of the paper:
    - Embedding weights are shared with the pre-softmax linear transformation
    - Embeddings are scaled by sqrt(d_model) in the forward pass
    """

    def __init__(self, config, src_vocab_size, tgt_vocab_size, pad_idx=0):
        """
        Args:
            config: TransformerConfig object
            src_vocab_size: Source vocabulary size
            tgt_vocab_size: Target vocabulary size
            pad_idx: Padding token index (default: 0)
        """
        super().__init__()
        self.config = config
        self.d_model = config.d_model
        self.pad_idx = pad_idx

        # Store vocab sizes
        self.src_vocab_size = src_vocab_size
        self.tgt_vocab_size = tgt_vocab_size

        # Store weight tying decision (copy from config, don't reference it)
        self.tie_embeddings = config.tie_embeddings
        self.share_src_tgt_embed = config.share_src_tgt_embed

        # Embedding layers
        if self.share_src_tgt_embed:
            # Shared embeddings (requires src_vocab_size == tgt_vocab_size)
            assert src_vocab_size == tgt_vocab_size, \
                "Shared embeddings require same vocabulary size"
            self.src_embed = nn.Embedding(src_vocab_size, config.d_model, padding_idx=pad_idx)
            self.tgt_embed = self.src_embed  # Share the same embedding layer
        else:
            # Separate embeddings
            self.src_embed = nn.Embedding(src_vocab_size, config.d_model, padding_idx=pad_idx)
            self.tgt_embed = nn.Embedding(tgt_vocab_size, config.d_model, padding_idx=pad_idx)

        # Positional encoding (shared between encoder and decoder)
        self.pos_encoding = PositionalEncoding(
            config.d_model,
            max_len=config.max_position,
            dropout=config.dropout
        )

        # Encoder stack
        self.encoder = TransformerEncoder(config)

        # Decoder stack
        self.decoder = TransformerDecoder(config)

        # Output projection layer
        if self.tie_embeddings:
            # Weight tying: Use transposed embedding weights for output projection
            # We'll manually compute logits in forward pass
            self.output_projection = None
        else:
            # Separate output layer
            self.output_projection = nn.Linear(config.d_model, tgt_vocab_size, bias=False)

        # Scaling factor for embeddings (Section 3.4 of paper)
        self.embed_scale = math.sqrt(config.d_model)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize parameters with Glorot / fan_avg."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None, cross_mask=None):
        """
        Forward pass through the Transformer.

        Args:
            src: Source token indices [batch_size, src_len]
            tgt: Target token indices [batch_size, tgt_len]
            src_mask: Source padding mask [batch_size, 1, src_len, src_len]
            tgt_mask: Target mask (causal + padding) [batch_size, 1, tgt_len, tgt_len]
            cross_mask: Cross-attention mask [batch_size, 1, tgt_len, src_len]

        Returns:
            logits: Output logits [batch_size, tgt_len, tgt_vocab_size]
        """
        # 1. Encode source sequence
        # Embedding: [batch_size, src_len] → [batch_size, src_len, d_model]
        src_embedded = self.src_embed(src) * self.embed_scale  # Scale by sqrt(d_model)

        # Add positional encoding: [batch_size, src_len, d_model]
        src_encoded = self.pos_encoding(src_embedded)

        # Pass through encoder: [batch_size, src_len, d_model]
        encoder_output = self.encoder(src_encoded, src_mask)

        # 2. Decode target sequence
        # Embedding: [batch_size, tgt_len] → [batch_size, tgt_len, d_model]
        tgt_embedded = self.tgt_embed(tgt) * self.embed_scale  # Scale by sqrt(d_model)

        # Add positional encoding: [batch_size, tgt_len, d_model]
        tgt_encoded = self.pos_encoding(tgt_embedded)

        # Pass through decoder: [batch_size, tgt_len, d_model]
        # During training, we don't use caching
        decoder_output, _ = self.decoder(tgt_encoded, encoder_output, cross_mask, tgt_mask,
                                          layer_caches=None, use_cache=False)

        # 3. Project to vocabulary
        if self.tie_embeddings:
            # Use transposed embedding weights: [batch_size, tgt_len, tgt_vocab_size]
            # decoder_output: [batch_size, tgt_len, d_model]
            # tgt_embed.weight: [tgt_vocab_size, d_model]
            # Result: [batch_size, tgt_len, tgt_vocab_size]
            logits = torch.matmul(decoder_output, self.tgt_embed.weight.T)
        else:
            # Use separate output projection: [batch_size, tgt_len, tgt_vocab_size]
            logits = self.output_projection(decoder_output)

        return logits

    def encode(self, src, src_mask=None):
        """
        Encode source sequence (for inference).

        Args:
            src: Source token indices [batch_size, src_len]
            src_mask: Source padding mask [batch_size, 1, 1, src_len]

        Returns:
            encoder_output: [batch_size, src_len, d_model]
        """
        # Embedding + positional encoding
        src_embedded = self.src_embed(src) * self.embed_scale
        src_encoded = self.pos_encoding(src_embedded)

        # Encode
        encoder_output = self.encoder(src_encoded, src_mask)
        return encoder_output

    def decode(self, tgt, encoder_output, cross_mask=None, tgt_mask=None):
        """
        Decode target sequence (for inference without caching).

        Args:
            tgt: Target token indices [batch_size, tgt_len]
            encoder_output: Encoder output [batch_size, src_len, d_model]
            cross_mask: Cross-attention mask [batch_size, 1, tgt_len, src_len]
            tgt_mask: Target mask [batch_size, 1, tgt_len, tgt_len]

        Returns:
            logits: Output logits [batch_size, tgt_len, tgt_vocab_size]
        """
        # Embedding + positional encoding
        tgt_embedded = self.tgt_embed(tgt) * self.embed_scale
        tgt_encoded = self.pos_encoding(tgt_embedded)

        # Decode (without caching)
        decoder_output, _ = self.decoder(tgt_encoded, encoder_output, cross_mask, tgt_mask,
                                          layer_caches=None, use_cache=False)

        # Project to vocabulary
        if self.tie_embeddings:
            logits = torch.matmul(decoder_output, self.tgt_embed.weight.T)
        else:
            logits = self.output_projection(decoder_output)

        return logits

    def decode_incremental(self, tgt, encoder_output, cross_mask=None, tgt_mask=None,
                           layer_caches=None, use_cache=True):
        """
        Decode with KV caching for incremental generation.

        This method is optimized for autoregressive decoding where we generate
        one token at a time. It caches K, V from previous steps to avoid recomputation.

        Args:
            tgt: Target token indices [batch_size, tgt_len] (usually tgt_len=1 for incremental)
            encoder_output: Encoder output [batch_size, src_len, d_model]
            cross_mask: Cross-attention mask [batch_size, 1, tgt_len, src_len]
            tgt_mask: Target mask [batch_size, 1, tgt_len, tgt_len] (usually [batch, 1, 1, cached_len+1])
            layer_caches: List of caches from previous steps (length = num_layers)
            use_cache: Whether to return updated caches (default: True)

        Returns:
            logits: Output logits [batch_size, tgt_len, tgt_vocab_size]
            new_layer_caches: Updated caches for all decoder layers
        """
        # Embedding + positional encoding
        tgt_embedded = self.tgt_embed(tgt) * self.embed_scale
        tgt_encoded = self.pos_encoding(tgt_embedded)

        # Decode with caching
        decoder_output, new_layer_caches = self.decoder(
            tgt_encoded, encoder_output,
            cross_mask, tgt_mask,
            layer_caches, use_cache
        )

        # Project to vocabulary
        if self.tie_embeddings:
            logits = torch.matmul(decoder_output, self.tgt_embed.weight.T)
        else:
            logits = self.output_projection(decoder_output)

        return logits, new_layer_caches

    def count_parameters(self):
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
