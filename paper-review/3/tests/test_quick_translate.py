#!/usr/bin/env python
"""Quick translation test with debugging."""

import sys
import os
sys.path.append(os.path.dirname(__file__))

import torch
from src.data.tokenizer import SentencePieceTokenizer
from src.inference.translator import Translator
from src.models.transformer.transformer import Transformer
from config.transformer_config import TransformerConfig

print("Loading...")

# Load tokenizers
ko_tok = SentencePieceTokenizer('data/vocab/ko_spm.model')
en_tok = SentencePieceTokenizer('data/vocab/en_spm.model')

# Load model
config = TransformerConfig()
model = Transformer(config, ko_tok.vocab_size, en_tok.vocab_size)
checkpoint = torch.load('checkpoints/best_model.pt', map_location='cpu')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Create translator
translator = Translator(model, ko_tok, en_tok, device='cpu', max_length=20)

# Test
src = "안녕하세요"
print(f"\nSource: {src}")

# Manually check tokenization
src_ids = ko_tok.encode_ids(src)
print(f"Tokens: {src_ids}")
src_with_bos_eos = [ko_tok.bos_id] + src_ids + [ko_tok.eos_id]
print(f"With BOS/EOS: {src_with_bos_eos}")

# Translate
translation = translator.translate(src, method='greedy')
print(f"Translation: '{translation}'")
print(f"Translation length: {len(translation)}")
print(f"Translation repr: {repr(translation)}")
