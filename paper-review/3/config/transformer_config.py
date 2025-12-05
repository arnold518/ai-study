"""Transformer model configuration based on 'Attention Is All You Need'."""

from .base_config import BaseConfig


class TransformerConfig(BaseConfig):
    """Configuration for Transformer model."""

    # Model architecture
    d_model = 512           # Dimension of model embeddings
    d_ff = 2048            # Dimension of feedforward network
    num_heads = 8          # Number of attention heads
    num_encoder_layers = 6 # Number of encoder layers
    num_decoder_layers = 6 # Number of decoder layers

    # Positional encoding
    max_position = 5000

    # Embedding and weight tying (Section 3.4 of paper)
    tie_embeddings = True        # Share weights between embeddings and output projection
    share_src_tgt_embed = True   # Share source and target embeddings (requires shared vocab)

    # Training (from paper, adjusted for 1.7M dataset)
    learning_rate = 1.0    # Will use custom scheduler with warmup
    warmup_steps = 16000   # Scaled from 4000 for 4.1x larger dataset (~1.2 epochs)
    adam_beta1 = 0.9
    adam_beta2 = 0.98
    adam_eps = 1e-9

    # Inference
    beam_size = 4
    length_penalty = 0.6
    max_decode_length = 150
