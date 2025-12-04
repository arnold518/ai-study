"""Seq2Seq model configurations."""

from .base_config import BaseConfig


class Seq2SeqConfig(BaseConfig):
    """Configuration for basic Seq2Seq model."""

    # Model architecture
    embedding_dim = 256
    hidden_dim = 512
    num_layers = 2
    bidirectional = True

    # Training
    teacher_forcing_ratio = 0.5


class BahdanauConfig(BaseConfig):
    """Configuration for Seq2Seq with Bahdanau attention."""

    # Model architecture
    embedding_dim = 256
    hidden_dim = 512
    num_layers = 2
    bidirectional = True
    attention_dim = 512

    # Training
    teacher_forcing_ratio = 0.5
