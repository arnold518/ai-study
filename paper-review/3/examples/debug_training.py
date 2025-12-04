#!/usr/bin/env python
"""
Debug training issues.

Usage:
    /home/arnold/venv/bin/python scripts/debug_training.py
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn as nn

from config.transformer_config import TransformerConfig
from src.models.transformer.transformer import Transformer
from src.data.tokenizer import SentencePieceTokenizer
from src.training.losses import LabelSmoothingLoss
from src.utils.masking import create_padding_mask, create_target_mask


def debug_forward_pass():
    """Debug the forward pass to find NaN source."""
    print("=" * 60)
    print("Debugging Forward Pass")
    print("=" * 60)

    # Configuration
    config = TransformerConfig()
    config.d_model = 256
    config.num_encoder_layers = 2
    config.num_decoder_layers = 2
    config.num_heads = 4
    config.d_ff = 512
    config.device = 'cpu'

    # Check if tokenizers exist
    ko_model_path = 'data/vocab/ko_spm.model'
    en_model_path = 'data/vocab/en_spm.model'

    if not os.path.exists(ko_model_path) or not os.path.exists(en_model_path):
        print("Tokenizers not found!")
        return

    ko_tokenizer = SentencePieceTokenizer(ko_model_path)
    en_tokenizer = SentencePieceTokenizer(en_model_path)

    # Create model
    model = Transformer(
        config,
        src_vocab_size=ko_tokenizer.vocab_size,
        tgt_vocab_size=en_tokenizer.vocab_size
    )
    model.to(config.device)

    # Create dummy batch
    batch_size = 2
    src_len = 5
    tgt_len = 6

    # Create sequences (avoid padding for now)
    src = torch.randint(1, ko_tokenizer.vocab_size, (batch_size, src_len))
    tgt = torch.randint(1, en_tokenizer.vocab_size, (batch_size, tgt_len))

    print(f"Source shape: {src.shape}")
    print(f"Target shape: {tgt.shape}")
    print()

    # Create masks
    src_mask = create_padding_mask(src, pad_idx=0)
    tgt_mask = create_target_mask(tgt, pad_idx=0)

    print(f"Source mask shape: {src_mask.shape}")
    print(f"Target mask shape: {tgt_mask.shape}")
    print()

    print("Source mask sample:")
    print(src_mask[0, 0, 0, :])
    print()

    print("Target mask sample (first sequence):")
    print(tgt_mask[0, 0, :, :])
    print()

    # Prepare decoder inputs
    tgt_input = tgt[:, :-1]
    tgt_output = tgt[:, 1:]
    tgt_input_mask = tgt_mask[:, :, :-1, :-1]

    print(f"Decoder input shape: {tgt_input.shape}")
    print(f"Decoder output shape (targets): {tgt_output.shape}")
    print(f"Decoder input mask shape: {tgt_input_mask.shape}")
    print()

    # Test embedding
    print("Testing embeddings...")
    src_embedded = model.src_embed(src) * model.embed_scale
    print(f"Source embedded shape: {src_embedded.shape}")
    print(f"Source embedded stats: min={src_embedded.min():.4f}, max={src_embedded.max():.4f}, mean={src_embedded.mean():.4f}")
    if torch.isnan(src_embedded).any():
        print("WARNING: NaN in source embeddings!")
    print()

    # Test positional encoding
    print("Testing positional encoding...")
    src_encoded = model.pos_encoding(src_embedded)
    print(f"Source encoded shape: {src_encoded.shape}")
    print(f"Source encoded stats: min={src_encoded.min():.4f}, max={src_encoded.max():.4f}, mean={src_encoded.mean():.4f}")
    if torch.isnan(src_encoded).any():
        print("WARNING: NaN in source encoded!")
    print()

    # Test encoder
    print("Testing encoder...")
    encoder_output = model.encoder(src_encoded, src_mask)
    print(f"Encoder output shape: {encoder_output.shape}")
    print(f"Encoder output stats: min={encoder_output.min():.4f}, max={encoder_output.max():.4f}, mean={encoder_output.mean():.4f}")
    if torch.isnan(encoder_output).any():
        print("WARNING: NaN in encoder output!")
    print()

    # Test decoder embedding
    print("Testing decoder embeddings...")
    tgt_embedded = model.tgt_embed(tgt_input) * model.embed_scale
    print(f"Target embedded shape: {tgt_embedded.shape}")
    print(f"Target embedded stats: min={tgt_embedded.min():.4f}, max={tgt_embedded.max():.4f}, mean={tgt_embedded.mean():.4f}")
    if torch.isnan(tgt_embedded).any():
        print("WARNING: NaN in target embeddings!")
    print()

    # Test decoder
    print("Testing decoder...")
    tgt_encoded = model.pos_encoding(tgt_embedded)
    print(f"Target encoded shape: {tgt_encoded.shape}")
    print(f"Target encoded stats: min={tgt_encoded.min():.4f}, max={tgt_encoded.max():.4f}, mean={tgt_encoded.mean():.4f}")
    if torch.isnan(tgt_encoded).any():
        print("WARNING: NaN in target encoded!")
    print()

    decoder_output = model.decoder(tgt_encoded, encoder_output, src_mask, tgt_input_mask)
    print(f"Decoder output shape: {decoder_output.shape}")
    print(f"Decoder output stats: min={decoder_output.min():.4f}, max={decoder_output.max():.4f}, mean={decoder_output.mean():.4f}")
    if torch.isnan(decoder_output).any():
        print("WARNING: NaN in decoder output!")
    print()

    # Test output projection
    print("Testing output projection...")
    if model.tie_embeddings:
        logits = torch.matmul(decoder_output, model.tgt_embed.weight.T)
    else:
        logits = model.output_projection(decoder_output)

    print(f"Logits shape: {logits.shape}")
    print(f"Logits stats: min={logits.min():.4f}, max={logits.max():.4f}, mean={logits.mean():.4f}")
    if torch.isnan(logits).any():
        print("WARNING: NaN in logits!")
    print()

    # Test loss
    print("Testing loss computation...")
    criterion = LabelSmoothingLoss(en_tokenizer.vocab_size, pad_idx=0, smoothing=0.1)
    logits_flat = logits.contiguous().view(-1, logits.size(-1))
    targets_flat = tgt_output.contiguous().view(-1)

    print(f"Logits flat shape: {logits_flat.shape}")
    print(f"Targets flat shape: {targets_flat.shape}")
    print(f"Target values: min={targets_flat.min()}, max={targets_flat.max()}")
    print()

    loss = criterion(logits_flat, targets_flat)
    print(f"Loss: {loss.item():.4f}")
    if torch.isnan(loss):
        print("WARNING: NaN in loss!")

    print()
    print("=" * 60)


if __name__ == "__main__":
    debug_forward_pass()
