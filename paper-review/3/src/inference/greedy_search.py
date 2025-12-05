"""Greedy decoder with repetition penalty."""

import torch
from ..utils.masking import create_target_mask, create_cross_attention_mask


def apply_repetition_penalty(logits, generated_ids, penalty=1.2, window_size=20):
    """
    Apply repetition penalty to discourage generating recently seen tokens.

    Args:
        logits: Token logits [batch_size, vocab_size]
        generated_ids: Previously generated token IDs [batch_size, seq_len]
        penalty: Penalty factor (>1.0 = discourage repetition)
        window_size: Number of recent tokens to penalize

    Returns:
        logits: Penalized logits [batch_size, vocab_size]
    """
    batch_size = logits.size(0)

    for batch_idx in range(batch_size):
        # Get recent tokens (last window_size tokens)
        recent_tokens = generated_ids[batch_idx, -window_size:].tolist()

        # Penalize each unique recent token
        for token_id in set(recent_tokens):
            # Skip special tokens (PAD=0, UNK=1, BOS=2, EOS=3)
            if token_id < 4:
                continue

            # Apply penalty correctly based on logit sign
            # For positive logits: divide by penalty (makes smaller)
            # For negative logits: multiply by penalty (makes more negative)
            if logits[batch_idx, token_id] > 0:
                logits[batch_idx, token_id] = logits[batch_idx, token_id] / penalty
            else:
                logits[batch_idx, token_id] = logits[batch_idx, token_id] * penalty

    return logits


def greedy_decode(model, src, src_mask, max_length, bos_idx, eos_idx, device):
    """
    Greedy decoding: select highest probability token at each step.

    This is the basic version without KV caching. At each step, we recompute
    attention for all previous positions (inefficient but correct).

    Args:
        model: Translation model
        src: Source sequence [batch_size, src_len]
        src_mask: Source mask [batch_size, 1, src_len, src_len]
        max_length: Maximum decoding length
        bos_idx: Beginning-of-sequence token index
        eos_idx: End-of-sequence token index
        device: Device to run on

    Returns:
        output: Decoded sequence [batch_size, out_len]
    """
    model.eval()

    with torch.no_grad():
        # 1. Encode source sequence once
        encoder_output = model.encode(src, src_mask)

        # 2. Initialize decoder input with BOS token
        batch_size = src.size(0)
        tgt = torch.full((batch_size, 1), bos_idx, dtype=torch.long, device=device)

        # Track which sequences have finished
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

        # 3. Generate tokens autoregressively
        for step in range(max_length):
            # Create masks for current target sequence
            tgt_mask = create_target_mask(tgt, pad_idx=0)  # [batch, 1, tgt_len, tgt_len]
            cross_mask = create_cross_attention_mask(src, tgt, pad_idx=0)  # [batch, 1, tgt_len, src_len]

            # Forward pass through decoder (recomputes everything - inefficient!)
            logits = model.decode(tgt, encoder_output, cross_mask, tgt_mask)

            # Get next token prediction (greedy = argmax)
            # logits: [batch_size, tgt_len, vocab_size]
            # We only care about the last position
            next_token_logits = logits[:, -1, :]  # [batch_size, vocab_size]

            # Apply repetition penalty to discourage loops
            next_token_logits = apply_repetition_penalty(
                next_token_logits, tgt, penalty=1.2, window_size=20
            )

            next_token = next_token_logits.argmax(dim=-1)  # [batch_size]

            # For finished sequences, force padding (or keep as is)
            next_token = torch.where(finished, torch.tensor(eos_idx, device=device), next_token)

            # Append to sequence
            tgt = torch.cat([tgt, next_token.unsqueeze(1)], dim=1)  # [batch, tgt_len+1]

            # Mark finished sequences
            finished = finished | (next_token == eos_idx)

            # Stop if all sequences have generated EOS
            if finished.all():
                break

        return tgt


def greedy_decode_cached(model, src, src_mask, max_length, bos_idx, eos_idx, device):
    """
    Greedy decoding with KV caching for efficient inference.

    This version uses KV caching to avoid recomputing attention for previous positions.
    Complexity: O(n) per step instead of O(n²).

    Args:
        model: Translation model
        src: Source sequence [batch_size, src_len]
        src_mask: Source mask [batch_size, 1, src_len, src_len]
        max_length: Maximum decoding length
        bos_idx: Beginning-of-sequence token index
        eos_idx: End-of-sequence token index
        device: Device to run on

    Returns:
        output: Decoded sequence [batch_size, out_len]
    """
    model.eval()

    with torch.no_grad():
        batch_size = src.size(0)
        src_len = src.size(1)

        # 1. Encode source sequence once
        encoder_output = model.encode(src, src_mask)

        # 2. Initialize decoder input with BOS token
        tgt = torch.full((batch_size, 1), bos_idx, dtype=torch.long, device=device)

        # 3. Initialize caches (None = empty cache)
        layer_caches = None

        # Track finished sequences
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

        # 4. Generate tokens autoregressively with caching
        for step in range(max_length):
            # For incremental decoding:
            # - tgt_input is only the NEW token (position t)
            # - Masks need to account for full sequence length (cached + new)

            if layer_caches is None:
                # First step: no cache, process full sequence (just BOS)
                tgt_input = tgt  # [batch, 1]
                current_len = 1
            else:
                # Subsequent steps: process only last token
                tgt_input = tgt[:, -1:]  # [batch, 1]
                # Current length = cached length + 1
                current_len = step + 1

            # Create masks for current sequence length
            # tgt_mask: [batch, 1, 1, current_len] - allows attending to all previous + current
            # This is essentially [1, 1, ..., 1] (all True) for the last position
            tgt_mask = torch.ones(batch_size, 1, 1, current_len, dtype=torch.bool, device=device)

            # cross_mask: [batch, 1, 1, src_len] - attend to all source positions
            cross_mask = (src != 0).unsqueeze(1).unsqueeze(2)  # Assume pad_idx=0
            cross_mask = cross_mask.expand(batch_size, 1, 1, src_len)

            # Forward pass with caching
            logits, layer_caches = model.decode_incremental(
                tgt=tgt_input,
                encoder_output=encoder_output,
                cross_mask=cross_mask,
                tgt_mask=tgt_mask,
                layer_caches=layer_caches,
                use_cache=True
            )

            # Get next token (greedy = argmax)
            # logits: [batch_size, 1, vocab_size] (only for the new position)
            next_token_logits = logits[:, -1, :]  # [batch_size, vocab_size]

            # Apply repetition penalty to discourage loops
            next_token_logits = apply_repetition_penalty(
                next_token_logits, tgt, penalty=1.2, window_size=20
            )

            next_token = next_token_logits.argmax(dim=-1)  # [batch_size]

            # For finished sequences, use EOS
            next_token = torch.where(finished, torch.tensor(eos_idx, device=device), next_token)

            # Append to sequence
            tgt = torch.cat([tgt, next_token.unsqueeze(1)], dim=1)  # [batch, len+1]

            # Mark finished sequences
            finished = finished | (next_token == eos_idx)

            # Stop if all sequences have generated EOS
            if finished.all():
                break

        return tgt
