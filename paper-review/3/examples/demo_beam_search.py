#!/usr/bin/env python
"""
Visual demonstration of beam search algorithm.

Shows step-by-step how beam search explores multiple hypotheses.
"""

import numpy as np
from dataclasses import dataclass
from typing import List


@dataclass
class Hypothesis:
    """A beam hypothesis."""
    tokens: List[str]
    score: float
    finished: bool = False

    def __repr__(self):
        tokens_str = ' '.join(self.tokens)
        return f"[{tokens_str}] score={self.score:.3f}"


def beam_search_demo(beam_size=2, max_steps=4):
    """
    Demonstrate beam search with a toy example.

    Vocabulary: {a, b, c, EOS}
    """
    print("=" * 80)
    print("BEAM SEARCH DEMONSTRATION")
    print("=" * 80)
    print(f"\nParameters:")
    print(f"  Beam size: {beam_size}")
    print(f"  Max steps: {max_steps}")
    print(f"  Vocabulary: {{BOS, a, b, c, EOS}}")
    print()

    # Simulated probabilities (log probs) for next token
    # vocab_probs[current_token][next_token]
    vocab_probs = {
        'BOS': {'a': -0.5, 'b': -0.7, 'c': -1.2, 'EOS': -3.0},
        'a': {'a': -0.3, 'b': -0.6, 'c': -1.5, 'EOS': -1.0},
        'b': {'a': -0.4, 'b': -0.2, 'c': -0.9, 'EOS': -0.8},
        'c': {'a': -1.0, 'b': -0.5, 'c': -0.3, 'EOS': -0.6},
    }

    # Initialize with BOS
    beams = [Hypothesis(tokens=['BOS'], score=0.0)]

    print("─" * 80)
    print("STEP 0: Initialize")
    print("─" * 80)
    print(f"Active beams: {len(beams)}")
    for i, beam in enumerate(beams, 1):
        print(f"  {i}. {beam}")
    print()

    # Beam search loop
    for step in range(1, max_steps + 1):
        print("─" * 80)
        print(f"STEP {step}: Expand and Prune")
        print("─" * 80)

        candidates = []

        # Expand each beam
        for beam_idx, beam in enumerate(beams):
            if beam.finished:
                candidates.append(beam)
                continue

            last_token = beam.tokens[-1]
            print(f"\nExpanding beam {beam_idx + 1}: {beam}")

            # Get probabilities for next tokens
            probs = vocab_probs.get(last_token, {})

            # Try each possible next token
            for next_token, log_prob in probs.items():
                new_tokens = beam.tokens + [next_token]
                new_score = beam.score + log_prob
                is_finished = (next_token == 'EOS')

                new_beam = Hypothesis(
                    tokens=new_tokens,
                    score=new_score,
                    finished=is_finished
                )

                candidates.append(new_beam)
                print(f"  → {new_beam} {'[FINISHED]' if is_finished else ''}")

        # Sort candidates by score
        candidates.sort(key=lambda h: h.score, reverse=True)

        print(f"\n{len(candidates)} candidates generated")
        print("\nAll candidates (sorted by score):")
        for i, cand in enumerate(candidates[:10], 1):  # Show top 10
            marker = "★" if i <= beam_size else " "
            print(f"  {marker} {i:2d}. {cand}")

        # Keep top beam_size
        beams = candidates[:beam_size]

        print(f"\nKeeping top {beam_size} beams:")
        for i, beam in enumerate(beams, 1):
            print(f"  {i}. {beam}")
        print()

        # Check if all beams finished
        if all(beam.finished for beam in beams):
            print("All beams finished! Stopping early.")
            break

    print("=" * 80)
    print("FINAL RESULT")
    print("=" * 80)
    best_beam = max(beams, key=lambda h: h.score)
    print(f"\nBest beam: {best_beam}")
    print(f"Sequence: {' '.join(best_beam.tokens)}")
    print(f"Score: {best_beam.score:.3f}")
    print()

    print("All final beams:")
    for i, beam in enumerate(beams, 1):
        print(f"  {i}. {beam}")
    print()


def compare_greedy_vs_beam():
    """Compare greedy search vs beam search."""
    print("\n" + "=" * 80)
    print("COMPARISON: Greedy vs Beam Search")
    print("=" * 80)

    # Example where greedy is suboptimal
    print("\nScenario: Greedy makes wrong early choice")
    print()

    # Probabilities designed to show greedy failure
    print("Greedy Search (beam_size=1):")
    print("─" * 40)
    print("Step 0: [BOS]")
    print("  Options: a (0.5), b (0.3)")
    print("  Choose: a (highest) ★")
    print()
    print("Step 1: [BOS, a]")
    print("  Options: x (0.1), y (0.2)")
    print("  Choose: y")
    print()
    print("Final: [BOS, a, y, EOS]")
    print("Score: log(0.5) + log(0.2) = -0.69 + -1.61 = -2.30")
    print()

    print("Beam Search (beam_size=2):")
    print("─" * 40)
    print("Step 0: [BOS]")
    print("  Keep top 2: a (0.5), b (0.3)")
    print()
    print("Step 1:")
    print("  [BOS, a] → y (0.2)  score: -2.30")
    print("  [BOS, b] → x (0.8)  score: -1.42 ★ Better!")
    print()
    print("Final: [BOS, b, x, EOS]")
    print("Score: log(0.3) + log(0.8) = -1.20 + -0.22 = -1.42")
    print()

    print("Result: Beam search found better sequence!")
    print("  Greedy score: -2.30")
    print("  Beam score:   -1.42 ✓")
    print()


def length_normalization_demo():
    """Demonstrate length normalization."""
    print("\n" + "=" * 80)
    print("LENGTH NORMALIZATION")
    print("=" * 80)

    sequences = [
        Hypothesis(tokens=['BOS', 'a', 'b', 'EOS'], score=-2.0),
        Hypothesis(tokens=['BOS', 'a', 'b', 'c', 'd', 'e', 'EOS'], score=-3.5),
    ]

    print("\nTwo completed sequences:")
    for seq in sequences:
        print(f"  {seq}")

    print("\nWithout normalization (alpha=0.0):")
    for seq in sequences:
        print(f"  {seq.tokens}: score = {seq.score:.3f}")
    best = max(sequences, key=lambda h: h.score)
    print(f"  Winner: {' '.join(best.tokens)} ★")

    print("\nWith normalization (alpha=0.6):")
    for seq in sequences:
        norm_score = seq.score / (len(seq.tokens) ** 0.6)
        print(f"  {seq.tokens}: score = {seq.score:.3f} / {len(seq.tokens)}^0.6 = {norm_score:.3f}")

    sequences_sorted = sorted(sequences, key=lambda h: h.score / (len(h.tokens) ** 0.6), reverse=True)
    best = sequences_sorted[0]
    print(f"  Winner: {' '.join(best.tokens)} ★")

    print("\nLength normalization prevents bias towards shorter sequences!")
    print()


if __name__ == "__main__":
    # Run demonstrations
    beam_search_demo(beam_size=2, max_steps=3)
    compare_greedy_vs_beam()
    length_normalization_demo()

    print("=" * 80)
    print("See BEAM_SEARCH_EXPLAINED.md for full implementation details!")
    print("=" * 80)
