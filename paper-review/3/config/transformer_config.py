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
    share_src_tgt_embed = False  # Separate embeddings for Korean-English (linguistically distant)

    # Training (from paper, adjusted for 897k dataset)
    learning_rate = 2.0    # Factor in Noam schedule (increased from 1.0 for higher peak LR)
    warmup_steps = 8000    # Increased from 4000 for slower LR decay, longer learning
    adam_beta1 = 0.9
    adam_beta2 = 0.98
    adam_eps = 1e-9

    # Inference
    beam_size = 8             # Increased from 4 for diverse beam search
    length_penalty = 0.6
    max_decode_length = 150

    # Diverse Beam Search (Vijayakumar et al., 2018)
    use_diverse_beam_search = True  # Enable diverse beam groups
    num_beam_groups = 4       # Divide beams into groups (must divide beam_size evenly)
    diversity_penalty = 0.5   # Penalty for selecting same token as previous groups

    # Repetition control
    repetition_penalty = 1.5  # Increased from 1.2 (higher = stronger penalty)
    repetition_window = 30    # Increased from 20 (longer memory)
