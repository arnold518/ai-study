"""Masking utilities for Transformer."""

import torch


def create_padding_mask(seq, pad_idx):
    """
    Create mask for padding tokens.

    Creates a mask for encoder self-attention where each position can attend to
    all non-padding positions.

    Args:
        seq: Input sequence [batch_size, seq_len]
        pad_idx: Padding token index

    Returns:
        mask: Padding mask [batch_size, 1, seq_len, seq_len]
              where mask[b, 0, i, j] = 1 if seq[b, j] is not padding, 0 otherwise
              (same mask for all query positions i, varies by key position j)
    """
    # Create [batch_size, 1, 1, seq_len] - indicates which positions are not padding
    mask = (seq != pad_idx).unsqueeze(1).unsqueeze(2)  # [batch, 1, 1, seq_len]

    # Expand to [batch_size, 1, seq_len, seq_len]
    # All query positions can attend to the same set of non-padding key positions
    batch_size, _, _, seq_len = mask.size()
    mask = mask.expand(batch_size, 1, seq_len, seq_len)

    return mask


def create_look_ahead_mask(size):
    """
    Create look-ahead mask for decoder self-attention.

    Args:
        size: Sequence length

    Returns:
        mask: Look-ahead mask [1, size, size]
    """
    # Upper triangular matrix with 1s above diagonal
    mask = torch.triu(torch.ones(size, size), diagonal=1).bool()
    # Invert: 1 for allowed positions, 0 for masked
    mask = ~mask
    return mask.unsqueeze(0)


def create_cross_attention_mask(src, tgt, pad_idx):
    """
    Create mask for decoder cross-attention to encoder.

    Decoder queries (from target) attend to encoder keys (from source).
    Only mask out padding positions in the source.

    Args:
        src: Source sequence [batch_size, src_len]
        tgt: Target sequence [batch_size, tgt_len]
        pad_idx: Padding token index

    Returns:
        mask: Cross-attention mask [batch_size, 1, tgt_len, src_len]
              where mask[b, 0, i, j] = 1 if src[b, j] is not padding
    """
    batch_size, src_len = src.size()
    tgt_len = tgt.size(1)

    # Create mask indicating non-padding positions in source
    # [batch_size, src_len] -> [batch_size, 1, 1, src_len]
    src_mask = (src != pad_idx).unsqueeze(1).unsqueeze(2)

    # Expand to [batch_size, 1, tgt_len, src_len]
    # Each target position can attend to all non-padding source positions
    src_mask = src_mask.expand(batch_size, 1, tgt_len, src_len)

    return src_mask


def create_target_mask(tgt, pad_idx):
    """
    Create combined mask for decoder self-attention (padding + look-ahead).

    Combines causal masking (can't attend to future) with padding masking
    (can't attend to padding tokens).

    Args:
        tgt: Target sequence [batch_size, tgt_len]
        pad_idx: Padding token index

    Returns:
        mask: Combined mask [batch_size, 1, tgt_len, tgt_len]
              where mask[b, 0, i, j] = 1 if j <= i AND tgt[b, j] is not padding
    """
    batch_size, tgt_len = tgt.size()

    # Create padding mask: [batch_size, 1, tgt_len, tgt_len]
    tgt_padding_mask = create_padding_mask(tgt, pad_idx)

    # Create causal mask: [1, tgt_len, tgt_len]
    tgt_look_ahead_mask = create_look_ahead_mask(tgt_len).to(tgt.device)

    # Expand causal mask to match batch size
    tgt_look_ahead_mask = tgt_look_ahead_mask.unsqueeze(0)  # [1, 1, tgt_len, tgt_len]

    # Combine masks with AND operation
    # Position i can attend to position j only if:
    # 1. j <= i (causal constraint)
    # 2. j is not padding (padding constraint)
    tgt_mask = tgt_padding_mask & tgt_look_ahead_mask

    return tgt_mask
