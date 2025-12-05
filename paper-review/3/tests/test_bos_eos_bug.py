#!/usr/bin/env python
"""Test BOS/EOS token handling - potential training/inference mismatch."""

import sys
import os
sys.path.append(os.path.dirname(__file__))

from src.data.tokenizer import SentencePieceTokenizer
import torch

print("=" * 80)
print("BOS/EOS TOKEN BUG INVESTIGATION")
print("=" * 80)
print()

# Load tokenizers
ko_tok = SentencePieceTokenizer('data/vocab/ko_spm.model')
en_tok = SentencePieceTokenizer('data/vocab/en_spm.model')

print("Special Token IDs:")
print(f"  Korean:  BOS={ko_tok.bos_id}, EOS={ko_tok.eos_id}, PAD={ko_tok.pad_id}")
print(f"  English: BOS={en_tok.bos_id}, EOS={en_tok.eos_id}, PAD={en_tok.pad_id}")
print()

# Test sentence
src_text = "안녕하세요"
tgt_text = "Hello"

print("=" * 80)
print("TRAINING: How dataset.py processes tokens")
print("=" * 80)
print()

print(f"Source: '{src_text}'")
print(f"Target: '{tgt_text}'")
print()

# Simulate dataset behavior (dataset.py lines 56-62)
src_ids = ko_tok.encode_ids(src_text)
tgt_ids = en_tok.encode_ids(tgt_text)

print("Step 1: Tokenize")
print(f"  src_ids (raw): {src_ids}")
print(f"  tgt_ids (raw): {tgt_ids}")
print()

# Add BOS/EOS as dataset does
src_ids_train = [ko_tok.bos_id] + src_ids + [ko_tok.eos_id]
tgt_ids_train = [en_tok.bos_id] + tgt_ids + [en_tok.eos_id]

print("Step 2: Add BOS and EOS (dataset.py lines 61-62)")
print(f"  src_ids: {src_ids_train}  <- [BOS, ...tokens..., EOS]")
print(f"  tgt_ids: {tgt_ids_train}  <- [BOS, ...tokens..., EOS]")
print()

# Simulate training loop (trainer.py lines 100-101)
tgt_input = tgt_ids_train[:-1]  # [BOS, token1, token2, ...]
tgt_output = tgt_ids_train[1:]   # [token1, token2, ..., EOS]

print("Step 3: Prepare decoder input/output (trainer.py lines 100-101)")
print(f"  tgt_input:  {tgt_input}  <- Feed to decoder (starts with BOS)")
print(f"  tgt_output: {tgt_output}  <- Target for loss (ends with EOS)")
print()

print("=" * 80)
print("INFERENCE: How translator.py processes tokens")
print("=" * 80)
print()

print(f"Source: '{src_text}'")
print()

# Simulate translator behavior (translator.py lines 51-54)
src_ids_infer = en_tok.encode_ids(src_text)  # Wait, it should be ko_tok!

# But the code actually uses tgt_tokenizer for bos_idx!
# Line 30: self.bos_idx = self.tgt_tokenizer.bos_id
# Line 54: src_ids = [self.bos_idx] + src_ids

print("⚠️  BUG CHECK 1: Which tokenizer's BOS is used for source?")
print(f"  translator.py line 30: self.bos_idx = self.tgt_tokenizer.bos_id")
print(f"  translator.py line 54: src_ids = [self.bos_idx] + src_ids")
print(f"  → Uses TARGET tokenizer BOS for SOURCE sequence!")
print()

# Simulate inference source prep
src_ids_infer_raw = ko_tok.encode_ids(src_text)
bos_idx_used = en_tok.bos_id  # Bug: using target tokenizer's BOS
src_ids_infer = [bos_idx_used] + src_ids_infer_raw

print("Step 1: Tokenize and add BOS (translator.py lines 51-54)")
print(f"  src_ids (raw): {src_ids_infer_raw}")
print(f"  BOS used:      {bos_idx_used} (from target tokenizer)")
print(f"  src_ids_infer: {src_ids_infer}  <- [BOS, ...tokens...] (NO EOS!)")
print()

print("⚠️  BUG CHECK 2: Source missing EOS during inference!")
print("  Training:  src = [BOS, ...tokens..., EOS]")
print("  Inference: src = [BOS, ...tokens...]         ← Missing EOS!")
print()

print("=" * 80)
print("COMPARISON: Training vs Inference")
print("=" * 80)
print()

print("SOURCE SEQUENCE:")
print(f"  Training:  {src_ids_train}")
print(f"  Inference: {src_ids_infer}")
print()

if src_ids_train == src_ids_infer:
    print("  ✓ MATCH - No bug")
else:
    print("  ✗ MISMATCH - CRITICAL BUG!")
    print()
    print("  Differences:")
    if len(src_ids_train) != len(src_ids_infer):
        print(f"    - Length: {len(src_ids_train)} (train) vs {len(src_ids_infer)} (infer)")
    if src_ids_train[-1] != src_ids_infer[-1]:
        print(f"    - Last token: {src_ids_train[-1]} (train) vs {src_ids_infer[-1]} (infer)")
    if src_ids_train[0] != src_ids_infer[0]:
        print(f"    - First token: {src_ids_train[0]} (train) vs {src_ids_infer[0]} (infer)")
print()

print("=" * 80)
print("IMPACT ANALYSIS")
print("=" * 80)
print()

# Check if BOS/EOS IDs are the same across tokenizers
if ko_tok.bos_id == en_tok.bos_id and ko_tok.eos_id == en_tok.eos_id:
    print("✓ Korean and English have SAME special token IDs:")
    print(f"  BOS: {ko_tok.bos_id} == {en_tok.bos_id}")
    print(f"  EOS: {ko_tok.eos_id} == {en_tok.eos_id}")
    print()
    print("  → BOS mismatch doesn't cause ID errors")
    print("  → BUT: Missing EOS on source during inference is STILL A BUG!")
else:
    print("✗ Korean and English have DIFFERENT special token IDs!")
    print(f"  BOS: {ko_tok.bos_id} != {en_tok.bos_id}")
    print(f"  EOS: {ko_tok.eos_id} != {en_tok.eos_id}")
    print()
    print("  → This would cause CRITICAL token ID mismatches!")

print()
print("Missing EOS impacts:")
print("  1. Encoder sees different sequence structure")
print("  2. Position encodings off by 1")
print("  3. Attention masks have different shapes")
print("  4. Model never learned to handle source without EOS")
print()

print("=" * 80)
print("ROOT CAUSE")
print("=" * 80)
print()

print("Bug Location: src/inference/translator.py")
print()
print("LINE 54 (WRONG):")
print("  src_ids = [self.bos_idx] + src_ids")
print("           ^^^^^^^^^^^^^^^^          <- Only adds BOS")
print()
print("SHOULD BE:")
print("  src_ids = [self.src_tokenizer.bos_id] + src_ids + [self.src_tokenizer.eos_id]")
print("           ^^^^^^^^^^^^^^^^^^^^^^^^^             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
print("           Use src tokenizer's BOS                Add EOS to match training")
print()

print("=" * 80)
print("VERIFICATION TEST")
print("=" * 80)
print()

# Read actual translator code
with open('src/inference/translator.py', 'r') as f:
    translator_code = f.read()

# Check line 54
import re
match = re.search(r'src_ids = \[self\.bos_idx\] \+ src_ids', translator_code)
if match:
    print("✗ BUG CONFIRMED: Line 54 only adds BOS, not EOS")
else:
    print("? Could not verify (code might have changed)")

# Check if EOS is added anywhere else
if 'self.src_tokenizer.eos_id' in translator_code or 'src_tokenizer.eos_id' in translator_code:
    print("✓ Source EOS is added somewhere")
else:
    print("✗ Source EOS is NEVER added during inference")

print()
