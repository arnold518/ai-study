#!/usr/bin/env python
"""Check special token IDs and their behavior."""

import sys
import os
sys.path.append(os.path.dirname(__file__))

from src.data.tokenizer import SentencePieceTokenizer

print("=" * 80)
print("Special Token Investigation")
print("=" * 80)
print()

# Load tokenizers
ko_tokenizer = SentencePieceTokenizer('data/vocab/ko_spm.model')
en_tokenizer = SentencePieceTokenizer('data/vocab/en_spm.model')

print("KOREAN TOKENIZER")
print("-" * 80)
print(f"Vocab size: {ko_tokenizer.vocab_size}")
print(f"PAD ID: {ko_tokenizer.pad_id}")
print(f"UNK ID: {ko_tokenizer.unk_id}")
print(f"BOS ID: {ko_tokenizer.bos_id}")
print(f"EOS ID: {ko_tokenizer.eos_id}")
print()

# Check what these tokens look like
print("Token strings:")
for name, token_id in [('PAD', ko_tokenizer.pad_id),
                       ('UNK', ko_tokenizer.unk_id),
                       ('BOS', ko_tokenizer.bos_id),
                       ('EOS', ko_tokenizer.eos_id)]:
    piece = ko_tokenizer.id_to_piece(token_id)
    print(f"  {name} (ID {token_id}): '{piece}'")
print()

print("ENGLISH TOKENIZER")
print("-" * 80)
print(f"Vocab size: {en_tokenizer.vocab_size}")
print(f"PAD ID: {en_tokenizer.pad_id}")
print(f"UNK ID: {en_tokenizer.unk_id}")
print(f"BOS ID: {en_tokenizer.bos_id}")
print(f"EOS ID: {en_tokenizer.eos_id}")
print()

print("Token strings:")
for name, token_id in [('PAD', en_tokenizer.pad_id),
                       ('UNK', en_tokenizer.unk_id),
                       ('BOS', en_tokenizer.bos_id),
                       ('EOS', en_tokenizer.eos_id)]:
    piece = en_tokenizer.id_to_piece(token_id)
    print(f"  {name} (ID {token_id}): '{piece}'")
print()

# Test encoding with and without special tokens
print("=" * 80)
print("Encoding Tests")
print("=" * 80)
print()

test_ko = "안녕하세요"
test_en = "Hello world"

print("KOREAN: '안녕하세요'")
print("-" * 80)
ids = ko_tokenizer.encode_ids(test_ko)
tokens = ko_tokenizer.tokenize(test_ko)
print(f"IDs:    {ids}")
print(f"Tokens: {tokens}")
print(f"Decoded: '{ko_tokenizer.decode_ids(ids)}'")
print()

print("Does encoding include BOS/EOS automatically?")
print(f"  First ID: {ids[0]} (is BOS {ko_tokenizer.bos_id}? {ids[0] == ko_tokenizer.bos_id})")
print(f"  Last ID:  {ids[-1]} (is EOS {ko_tokenizer.eos_id}? {ids[-1] == ko_tokenizer.eos_id})")
print()

print("ENGLISH: 'Hello world'")
print("-" * 80)
ids = en_tokenizer.encode_ids(test_en)
tokens = en_tokenizer.tokenize(test_en)
print(f"IDs:    {ids}")
print(f"Tokens: {tokens}")
print(f"Decoded: '{en_tokenizer.decode_ids(ids)}'")
print()

print("Does encoding include BOS/EOS automatically?")
print(f"  First ID: {ids[0]} (is BOS {en_tokenizer.bos_id}? {ids[0] == en_tokenizer.bos_id})")
print(f"  Last ID:  {ids[-1]} (is EOS {en_tokenizer.eos_id}? {ids[-1] == en_tokenizer.eos_id})")
print()

# Test manual BOS/EOS addition
print("=" * 80)
print("Manual BOS/EOS Addition")
print("=" * 80)
print()

test_en = "Hello"
ids_raw = en_tokenizer.encode_ids(test_en)
ids_with_special = [en_tokenizer.bos_id] + ids_raw + [en_tokenizer.eos_id]

print(f"Text: '{test_en}'")
print(f"Raw IDs:          {ids_raw}")
print(f"With BOS/EOS:     {ids_with_special}")
print(f"Decoded (raw):    '{en_tokenizer.decode_ids(ids_raw)}'")
print(f"Decoded (w/ spc): '{en_tokenizer.decode_ids(ids_with_special)}'")
print()

# Check if decoding handles special tokens correctly
print("=" * 80)
print("Decoding Special Tokens")
print("=" * 80)
print()

special_sequences = [
    [en_tokenizer.bos_id],
    [en_tokenizer.eos_id],
    [en_tokenizer.pad_id],
    [en_tokenizer.bos_id, en_tokenizer.eos_id],
    [en_tokenizer.bos_id, 100, 200, en_tokenizer.eos_id],
]

for seq in special_sequences:
    decoded = en_tokenizer.decode_ids(seq)
    print(f"IDs: {seq}")
    print(f"  Decoded: '{decoded}'")
    print(f"  Repr: {repr(decoded)}")
    print()

# Test repetition
print("=" * 80)
print("Repetition Test")
print("=" * 80)
print()

# Simulate model generating same token repeatedly
repeated_token_id = en_tokenizer.encode_ids("the")[0]
repeated_seq = [en_tokenizer.bos_id] + [repeated_token_id] * 20 + [en_tokenizer.eos_id]

print(f"Repeated token: '{en_tokenizer.id_to_piece(repeated_token_id)}' (ID {repeated_token_id})")
print(f"Sequence: {repeated_seq[:10]}... (first 10)")
print(f"Decoded: '{en_tokenizer.decode_ids(repeated_seq)}'")
print()

# Test if EOS stops generation
print("=" * 80)
print("EOS Token Test")
print("=" * 80)
print()

# What if model generates EOS in the middle?
normal_ids = en_tokenizer.encode_ids("Hello world")
with_eos_middle = [en_tokenizer.bos_id] + normal_ids[:2] + [en_tokenizer.eos_id] + normal_ids[2:] + [en_tokenizer.eos_id]

print(f"Normal: {normal_ids}")
print(f"With EOS in middle: {with_eos_middle}")
print(f"Decoded: '{en_tokenizer.decode_ids(with_eos_middle)}'")
print()

print("CRITICAL: Does decode_ids remove BOS/EOS?")
test_cases = [
    ([en_tokenizer.bos_id, 100, 200, en_tokenizer.eos_id], "BOS + tokens + EOS"),
    ([100, 200], "tokens only"),
    ([en_tokenizer.bos_id, 100, 200], "BOS + tokens (no EOS)"),
]

for ids, desc in test_cases:
    decoded = en_tokenizer.decode_ids(ids)
    print(f"{desc}:")
    print(f"  IDs: {ids}")
    print(f"  Decoded: '{decoded}' (len={len(decoded)})")
    print()
