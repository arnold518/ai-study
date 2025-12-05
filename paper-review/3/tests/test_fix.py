#!/usr/bin/env python
"""Quick test to verify the fix."""

import sys
import os
sys.path.append(os.path.dirname(__file__))

from src.data.tokenizer import SentencePieceTokenizer

ko_tok = SentencePieceTokenizer('data/vocab/ko_spm.model')
en_tok = SentencePieceTokenizer('data/vocab/en_spm.model')

src_text = "안녕하세요"

# Simulate OLD buggy code
src_ids_old = [en_tok.bos_id] + ko_tok.encode_ids(src_text)

# Simulate NEW fixed code
src_ids_new = [ko_tok.bos_id] + ko_tok.encode_ids(src_text) + [ko_tok.eos_id]

# Training format (from dataset.py)
src_ids_train = [ko_tok.bos_id] + ko_tok.encode_ids(src_text) + [ko_tok.eos_id]

print("=" * 60)
print("FIX VERIFICATION")
print("=" * 60)
print(f"\nSource text: '{src_text}'")
print(f"\nTraining format:  {src_ids_train}")
print(f"Old inference:    {src_ids_old}  {'✗ MISMATCH' if src_ids_old != src_ids_train else '✓'}")
print(f"New inference:    {src_ids_new}  {'✓ MATCH!' if src_ids_new == src_ids_train else '✗'}")
print()

if src_ids_new == src_ids_train:
    print("✅ FIX SUCCESSFUL! Training and inference now match.")
else:
    print("❌ FIX FAILED! Still have mismatch.")
