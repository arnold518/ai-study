#!/usr/bin/env python
"""Test repetition penalty implementation."""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import torch
from src.inference.greedy_search import apply_repetition_penalty
from src.inference.beam_search import apply_repetition_penalty_beam


class TestRepetitionPenalty:
    """Test repetition penalty functions."""

    def test_apply_repetition_penalty_basic(self):
        """Test that repetition penalty reduces logits for repeated tokens."""
        # Create fake logits
        logits = torch.randn(1, 100)  # [batch=1, vocab_size=100]

        # Create sequence with repeated token 50
        generated_ids = torch.tensor([[2, 10, 20, 50, 50, 50, 30]])  # BOS, then tokens

        # Get original logit for token 50
        original_logit_50 = logits[0, 50].item()
        original_logit_30 = logits[0, 30].item()

        # Apply penalty
        penalized_logits = apply_repetition_penalty(
            logits.clone(), generated_ids, penalty=2.0, window_size=10
        )

        # Check that token 50 was penalized (logit reduced)
        assert penalized_logits[0, 50].item() < original_logit_50

        # Check that token 30 was also penalized (it appeared once)
        assert penalized_logits[0, 30].item() < original_logit_30

        # Check that token 40 (not in sequence) was NOT penalized
        assert penalized_logits[0, 40].item() == logits[0, 40].item()

    def test_repetition_penalty_skips_special_tokens(self):
        """Test that special tokens (PAD, UNK, BOS, EOS) are not penalized."""
        logits = torch.randn(1, 100)

        # Sequence with special tokens and repeated normal tokens
        generated_ids = torch.tensor([[2, 0, 1, 3, 2, 10, 10, 20, 20]])  # BOS, PAD, UNK, EOS, BOS

        # Save original logits for special tokens
        original_0 = logits[0, 0].item()
        original_1 = logits[0, 1].item()
        original_2 = logits[0, 2].item()
        original_3 = logits[0, 3].item()

        # Apply penalty
        penalized_logits = apply_repetition_penalty(
            logits.clone(), generated_ids, penalty=2.0, window_size=10
        )

        # Special tokens should NOT be penalized (even though they appear)
        assert penalized_logits[0, 0].item() == original_0
        assert penalized_logits[0, 1].item() == original_1
        assert penalized_logits[0, 2].item() == original_2
        assert penalized_logits[0, 3].item() == original_3

        # But token 10 and 20 should be penalized (they appear in sequence)
        assert penalized_logits[0, 10].item() < logits[0, 10].item()
        assert penalized_logits[0, 20].item() < logits[0, 20].item()

    def test_repetition_penalty_window_size(self):
        """Test that only recent tokens within window are penalized."""
        logits = torch.randn(1, 100)

        # Token 50 appears only in positions outside window
        # Token 60 appears inside window
        generated_ids = torch.tensor([[
            2, 50, 50,  # Far back
            10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,  # Filler
            60, 60  # Recent
        ]])

        # Apply penalty with small window
        penalized_logits = apply_repetition_penalty(
            logits.clone(), generated_ids, penalty=2.0, window_size=5
        )

        # Token 50 is outside window (5 tokens), should NOT be penalized
        assert penalized_logits[0, 50].item() == logits[0, 50].item()

        # Token 60 is inside window, SHOULD be penalized
        assert penalized_logits[0, 60].item() < logits[0, 60].item()

    def test_apply_repetition_penalty_beam(self):
        """Test beam search repetition penalty."""
        # Create fake log probs
        log_probs = torch.randn(100)  # [vocab_size=100]

        # Create sequence with repeated token 50
        tokens = [2, 10, 20, 50, 50, 50, 30]

        # Get original log prob for token 50
        original_logprob_50 = log_probs[50].item()

        # Apply penalty
        penalized_log_probs = apply_repetition_penalty_beam(
            log_probs.clone(), tokens, penalty=2.0, window_size=10
        )

        # Check that token 50 was penalized (log prob reduced)
        assert penalized_log_probs[50].item() < original_logprob_50

        # Check that the penalty is correct: log(P) - log(penalty)
        expected_penalty = torch.log(torch.tensor(2.0)).item()
        actual_penalty = original_logprob_50 - penalized_log_probs[50].item()
        assert abs(actual_penalty - expected_penalty) < 1e-6

    def test_penalty_factor_effect(self):
        """Test that higher penalty factor causes stronger reduction."""
        logits = torch.randn(1, 100)
        generated_ids = torch.tensor([[2, 50, 50, 50]])

        original_logit = logits[0, 50].item()

        # Apply different penalties
        penalized_1_2 = apply_repetition_penalty(
            logits.clone(), generated_ids, penalty=1.2, window_size=10
        )[0, 50].item()

        penalized_2_0 = apply_repetition_penalty(
            logits.clone(), generated_ids, penalty=2.0, window_size=10
        )[0, 50].item()

        # Higher penalty should cause stronger reduction
        assert (original_logit - penalized_2_0) > (original_logit - penalized_1_2)

    def test_batch_processing(self):
        """Test that penalty works correctly for batched inputs."""
        batch_size = 3
        vocab_size = 100

        logits = torch.randn(batch_size, vocab_size)

        # Different sequences for each batch
        generated_ids = torch.tensor([
            [2, 10, 10, 10],  # Batch 0: token 10 repeated
            [2, 20, 20, 20],  # Batch 1: token 20 repeated
            [2, 30, 40, 50],  # Batch 2: no repetition
        ])

        # Apply penalty
        penalized_logits = apply_repetition_penalty(
            logits.clone(), generated_ids, penalty=2.0, window_size=10
        )

        # Batch 0: token 10 penalized, token 20 not
        assert penalized_logits[0, 10] < logits[0, 10]
        assert penalized_logits[0, 20] == logits[0, 20]

        # Batch 1: token 20 penalized, token 10 not
        assert penalized_logits[1, 20] < logits[1, 20]
        assert penalized_logits[1, 10] == logits[1, 10]

        # Batch 2: tokens 30, 40, 50 penalized (appeared once each)
        assert penalized_logits[2, 30] < logits[2, 30]
        assert penalized_logits[2, 40] < logits[2, 40]
        assert penalized_logits[2, 50] < logits[2, 50]


if __name__ == "__main__":
    # Run tests
    test = TestRepetitionPenalty()

    print("Testing repetition penalty implementation...")
    print()

    try:
        print("Test 1: Basic repetition penalty...")
        test.test_apply_repetition_penalty_basic()
        print("  ✓ PASS")

        print("Test 2: Skip special tokens...")
        test.test_repetition_penalty_skips_special_tokens()
        print("  ✓ PASS")

        print("Test 3: Window size...")
        test.test_repetition_penalty_window_size()
        print("  ✓ PASS")

        print("Test 4: Beam search penalty...")
        test.test_apply_repetition_penalty_beam()
        print("  ✓ PASS")

        print("Test 5: Penalty factor effect...")
        test.test_penalty_factor_effect()
        print("  ✓ PASS")

        print("Test 6: Batch processing...")
        test.test_batch_processing()
        print("  ✓ PASS")

        print()
        print("=" * 60)
        print("✅ ALL TESTS PASSED!")
        print("=" * 60)

    except AssertionError as e:
        print(f"  ✗ FAIL: {e}")
        import traceback
        traceback.print_exc()
