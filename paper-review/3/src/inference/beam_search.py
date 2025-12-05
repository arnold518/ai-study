"""Beam search decoder with KV caching and repetition penalty."""

import torch
import torch.nn.functional as F
from dataclasses import dataclass
from typing import List
from ..utils.masking import create_target_mask, create_cross_attention_mask


def apply_repetition_penalty_beam(log_probs, tokens, penalty=1.2, window_size=20):
    """
    Apply repetition penalty to log probabilities in beam search.

    Args:
        log_probs: Log probabilities [vocab_size]
        tokens: Previously generated tokens (list of ints)
        penalty: Penalty factor (>1.0 = discourage repetition)
        window_size: Number of recent tokens to penalize

    Returns:
        log_probs: Penalized log probabilities [vocab_size]
    """
    # Get recent tokens
    recent_tokens = tokens[-window_size:] if len(tokens) > window_size else tokens

    # Penalize each unique recent token
    for token_id in set(recent_tokens):
        # Skip special tokens (PAD=0, UNK=1, BOS=2, EOS=3)
        if token_id < 4:
            continue

        # Apply penalty: subtract log(penalty) from log probability
        # This is equivalent to dividing probability by penalty
        log_probs[token_id] = log_probs[token_id] - torch.log(torch.tensor(penalty))

    return log_probs


@dataclass
class Hypothesis:
    """A beam hypothesis."""
    tokens: List[int]          # Token sequence
    score: float               # Log probability score
    layer_caches: List[dict]   # KV caches for each layer
    finished: bool = False     # Whether sequence ended with EOS

    def __repr__(self):
        tokens_str = ' '.join(str(t) for t in self.tokens)
        return f"[{tokens_str}] score={self.score:.3f} finished={self.finished}"


def beam_search(model, src, src_mask, beam_size, max_length, bos_idx, eos_idx,
                length_penalty=0.6, device='cpu'):
    """
    Beam search with length normalization and KV caching.

    Args:
        model: Translation model
        src: Source sequence [1, src_len]
        src_mask: Source mask [1, 1, src_len, src_len]
        beam_size: Number of beams to maintain
        max_length: Maximum generation length
        bos_idx: Beginning-of-sequence token index
        eos_idx: End-of-sequence token index
        length_penalty: Alpha parameter for length normalization (0.0 = no penalty, 0.6 = standard)
        device: Device to run on

    Returns:
        best_sequence: Best predicted sequence [1, seq_len]
    """
    model.eval()

    with torch.no_grad():
        # 1. Encode source once
        encoder_output = model.encode(src, src_mask)
        src_len = src.size(1)

        # 2. Initialize beam with BOS token
        initial_hypothesis = Hypothesis(
            tokens=[bos_idx],
            score=0.0,
            layer_caches=None,
            finished=False
        )
        beams = [initial_hypothesis]
        completed_hypotheses = []

        # 3. Beam search loop
        for step in range(max_length):
            candidates = []

            # Expand each beam
            for beam in beams:
                if beam.finished:
                    # Keep finished beams
                    candidates.append(beam)
                    continue

                # Prepare input (only last token for incremental decoding)
                if beam.layer_caches is None:
                    # First step: full sequence (just BOS)
                    tgt_input = torch.tensor([beam.tokens], dtype=torch.long, device=device)
                    current_len = len(beam.tokens)
                else:
                    # Subsequent steps: only last token
                    tgt_input = torch.tensor([[beam.tokens[-1]]], dtype=torch.long, device=device)
                    current_len = len(beam.tokens)

                # Create masks
                tgt_mask = torch.ones(1, 1, 1, current_len, dtype=torch.bool, device=device)
                cross_mask = (src != 0).unsqueeze(1).unsqueeze(2).expand(1, 1, 1, src_len)

                # Forward pass with caching
                logits, new_layer_caches = model.decode_incremental(
                    tgt=tgt_input,
                    encoder_output=encoder_output,
                    cross_mask=cross_mask,
                    tgt_mask=tgt_mask,
                    layer_caches=beam.layer_caches,
                    use_cache=True
                )

                # Get log probabilities for next token
                log_probs = F.log_softmax(logits[0, -1, :], dim=-1)  # [vocab_size]

                # Apply repetition penalty to discourage loops
                log_probs = apply_repetition_penalty_beam(
                    log_probs, beam.tokens, penalty=1.2, window_size=20
                )

                # Get top-k tokens for this beam
                topk_log_probs, topk_indices = torch.topk(log_probs, beam_size)

                # Create new hypotheses
                for log_prob, token_idx in zip(topk_log_probs, topk_indices):
                    token = token_idx.item()
                    new_score = beam.score + log_prob.item()
                    new_tokens = beam.tokens + [token]
                    is_finished = (token == eos_idx)

                    new_hypothesis = Hypothesis(
                        tokens=new_tokens,
                        score=new_score,
                        layer_caches=new_layer_caches,
                        finished=is_finished
                    )

                    if is_finished:
                        completed_hypotheses.append(new_hypothesis)
                    else:
                        candidates.append(new_hypothesis)

            # If no active beams left, stop
            if not candidates:
                break

            # Sort candidates by normalized score
            def normalized_score(hyp):
                if length_penalty > 0:
                    # Length normalization: score / (length^alpha)
                    return hyp.score / (len(hyp.tokens) ** length_penalty)
                return hyp.score

            candidates.sort(key=normalized_score, reverse=True)

            # Keep top beam_size beams
            beams = candidates[:beam_size]

            # Early stopping if we have enough completed hypotheses
            if len(completed_hypotheses) >= beam_size:
                # Check if best completed is better than best active
                best_completed = max(completed_hypotheses, key=normalized_score)
                best_active = beams[0] if beams else None

                if best_active is None or normalized_score(best_completed) >= normalized_score(best_active):
                    break

        # 4. Select best hypothesis
        all_hypotheses = completed_hypotheses + beams

        if not all_hypotheses:
            # Fallback: return BOS + EOS
            return torch.tensor([[bos_idx, eos_idx]], dtype=torch.long, device=device)

        # Apply length normalization for final selection
        def final_score(hyp):
            if length_penalty > 0:
                return hyp.score / (len(hyp.tokens) ** length_penalty)
            return hyp.score

        best_hypothesis = max(all_hypotheses, key=final_score)
        best_sequence = torch.tensor([best_hypothesis.tokens], dtype=torch.long, device=device)

        return best_sequence
