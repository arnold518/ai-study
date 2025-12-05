#!/usr/bin/env python
"""Debug repetition penalty."""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import torch
from src.inference.greedy_search import apply_repetition_penalty

# Create simple test
logits = torch.randn(1, 100)
generated_ids = torch.tensor([[2, 10, 10, 20, 20]])

print("Generated IDs:", generated_ids)
print("Unique tokens:", set(generated_ids[0].tolist()))
print()

print("Before penalty:")
print(f"  Token 10 logit: {logits[0, 10].item():.4f}")
print(f"  Token 20 logit: {logits[0, 20].item():.4f}")
print(f"  Token 30 logit: {logits[0, 30].item():.4f}")
print()

penalized = apply_repetition_penalty(logits.clone(), generated_ids, penalty=2.0, window_size=10)

print("After penalty:")
print(f"  Token 10 logit: {penalized[0, 10].item():.4f}")
print(f"  Token 20 logit: {penalized[0, 20].item():.4f}")
print(f"  Token 30 logit: {penalized[0, 30].item():.4f}")
print()

print("Changes:")
print(f"  Token 10: {logits[0, 10].item() - penalized[0, 10].item():.4f}")
print(f"  Token 20: {logits[0, 20].item() - penalized[0, 20].item():.4f}")
print(f"  Token 30: {logits[0, 30].item() - penalized[0, 30].item():.4f}")
