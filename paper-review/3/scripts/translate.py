#!/usr/bin/env python
"""
Translation script for trained Transformer model.

Usage:
    /home/arnold/venv/bin/python scripts/translate.py --input "한국어 문장"
    /home/arnold/venv/bin/python scripts/translate.py --file input.txt
    /home/arnold/venv/bin/python scripts/translate.py --input "안녕하세요" --method beam
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import argparse
import torch

from config.transformer_config import TransformerConfig
from src.data.tokenizer import SentencePieceTokenizer
from src.models.transformer.transformer import Transformer
from src.inference.translator import Translator


def main():
    parser = argparse.ArgumentParser(description="Translate Korean to English")
    parser.add_argument('--input', type=str, help='Input text to translate')
    parser.add_argument('--file', type=str, help='Input file with sentences (one per line)')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/best_model.pt',
                       help='Path to model checkpoint')
    parser.add_argument('--method', type=str, default='greedy', choices=['greedy', 'beam'],
                       help='Decoding method')
    parser.add_argument('--beam-size', type=int, default=4,
                       help='Beam size for beam search')
    parser.add_argument('--length-penalty', type=float, default=0.6,
                       help='Length penalty for beam search (0.0 = no penalty, 0.6 = standard)')
    parser.add_argument('--max-length', type=int, default=150,
                       help='Maximum generation length')
    parser.add_argument('--device', type=str, default='cpu',
                       help='Device to run on (cpu or cuda)')

    args = parser.parse_args()

    # Check inputs
    if not args.input and not args.file:
        print("Error: Please provide --input or --file")
        parser.print_help()
        return

    # Check if checkpoint exists
    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint not found: {args.checkpoint}")
        print()
        print("To use this script, you need a trained model checkpoint.")
        print("Train a model first using: /home/arnold/venv/bin/python scripts/train.py")
        return

    print("=" * 80)
    print("Korean-English Translation")
    print("=" * 80)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Method: {args.method}")
    if args.method == 'beam':
        print(f"Beam size: {args.beam_size}")
        print(f"Length penalty: {args.length_penalty}")
    print(f"Device: {args.device}")
    print()

    # Load configuration
    config = TransformerConfig()
    device = torch.device(args.device)

    print("Loading model and tokenizers...")

    # Load tokenizers
    src_tokenizer = SentencePieceTokenizer('data/vocab/ko_spm.model')
    tgt_tokenizer = SentencePieceTokenizer('data/vocab/en_spm.model')

    # Load model
    src_vocab_size = src_tokenizer.get_vocab_size()
    tgt_vocab_size = tgt_tokenizer.get_vocab_size()

    model = Transformer(config, src_vocab_size, tgt_vocab_size, pad_idx=0)

    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
    print(f"Model parameters: {model.count_parameters():,}")
    print()

    # Create translator
    translator = Translator(
        model=model,
        src_tokenizer=src_tokenizer,
        tgt_tokenizer=tgt_tokenizer,
        device=device,
        max_length=args.max_length
    )

    # Translate
    if args.input:
        # Single sentence
        print(f"Source: {args.input}")
        translation = translator.translate(
            args.input,
            method=args.method,
            beam_size=args.beam_size,
            length_penalty=args.length_penalty
        )
        print(f"Translation: {translation}")
        print()

    elif args.file:
        # Multiple sentences from file
        if not os.path.exists(args.file):
            print(f"Error: File not found: {args.file}")
            return

        print(f"Translating from file: {args.file}")
        print("-" * 80)

        with open(args.file, 'r', encoding='utf-8') as f:
            sentences = [line.strip() for line in f if line.strip()]

        for i, sentence in enumerate(sentences, 1):
            print(f"\n[{i}/{len(sentences)}]")
            print(f"Source:      {sentence}")
            translation = translator.translate(
                sentence,
                method=args.method,
                beam_size=args.beam_size,
                length_penalty=args.length_penalty
            )
            print(f"Translation: {translation}")

        print()
        print(f"Translated {len(sentences)} sentences")

    print("=" * 80)


if __name__ == "__main__":
    main()
