#!/usr/bin/env python
"""
Train SentencePiece tokenizers for Korean and English.

This script trains subword tokenizers on the processed training data using
parameters from config/base_config.py.

Output:
  - Separate mode: data/vocab/ko_spm.{model,vocab} and en_spm.{model,vocab}
  - Shared mode: data/vocab/shared_spm.{model,vocab}

Usage:
    /home/arnold/venv/bin/python scripts/train_tokenizer.py
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import tempfile
from pathlib import Path
import sentencepiece as spm
from config.base_config import BaseConfig


def train_sentencepiece(input_file, output_prefix, vocab_size=16000,
                       character_coverage=1.0, model_type='unigram'):
    """
    Train a SentencePiece tokenizer.

    Args:
        input_file: Path to training text file
        output_prefix: Output model prefix (e.g., 'data/vocab/ko_spm')
        vocab_size: Size of vocabulary
        character_coverage: Character coverage for the model (0.9995 for Korean, 1.0 for English)
        model_type: Type of model ('unigram', 'bpe', 'char', 'word')
    """
    print(f"Training SentencePiece model...")
    print(f"  Input: {input_file}")
    print(f"  Output: {output_prefix}.{{model,vocab}}")
    print(f"  Vocab size: {vocab_size}")
    print(f"  Character coverage: {character_coverage}")
    print(f"  Model type: {model_type}")

    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_prefix)
    os.makedirs(output_dir, exist_ok=True)

    # Train SentencePiece model
    spm.SentencePieceTrainer.train(
        input=input_file,
        model_prefix=output_prefix,
        vocab_size=vocab_size,
        character_coverage=character_coverage,
        model_type=model_type,
        # Special token IDs (SentencePiece standard)
        pad_id=0,    # <pad>
        unk_id=1,    # <unk>
        bos_id=2,    # <s> (beginning of sentence)
        eos_id=3,    # </s> (end of sentence)
        # Optional: add custom symbols for future use
        user_defined_symbols=['<mask>'],
        # Training parameters
        num_threads=os.cpu_count(),
        split_digits=True,  # Split numbers into individual digits
        byte_fallback=True,  # Use byte encoding for unknown characters
    )

    print(f"✓ Model saved: {output_prefix}.model")
    print(f"✓ Vocab saved: {output_prefix}.vocab")
    print()


def create_combined_corpus(ko_input, en_input, output_file):
    """
    Combine Korean and English training data into a single file.

    Args:
        ko_input: Path to Korean training file
        en_input: Path to English training file
        output_file: Path to output combined file
    """
    print("Creating combined corpus for shared vocabulary...")
    print(f"  Korean input: {ko_input}")
    print(f"  English input: {en_input}")
    print(f"  Combined output: {output_file}")

    line_count = 0
    with open(output_file, 'w', encoding='utf-8') as outf:
        # Add Korean sentences
        with open(ko_input, 'r', encoding='utf-8') as inf:
            for line in inf:
                if line.strip():
                    outf.write(line)
                    line_count += 1

        # Add English sentences
        with open(en_input, 'r', encoding='utf-8') as inf:
            for line in inf:
                if line.strip():
                    outf.write(line)
                    line_count += 1

    print(f"✓ Combined {line_count:,} sentences")
    print()


def main():
    config = BaseConfig()

    print("=" * 60)
    print("SentencePiece Tokenizer Training")
    print("=" * 60)
    print()
    print(f"Configuration:")
    print(f"  Vocab mode: {'SHARED' if config.use_shared_vocab else 'SEPARATE'}")
    print(f"  Vocab size: {config.vocab_size:,}")
    print(f"  Model type: {config.spm_model_type}")
    print()

    # Setup paths relative to script location
    script_dir = Path(__file__).parent
    project_root = script_dir.parent

    # Resolve paths from config
    ko_input = project_root / config.processed_data_dir / f"train.{config.src_lang}"
    en_input = project_root / config.processed_data_dir / f"train.{config.tgt_lang}"
    output_dir = project_root / config.vocab_dir

    # Check input files exist
    if not ko_input.exists():
        print(f"✗ Error: {config.src_lang} training file not found: {ko_input}")
        print("  Please run: /home/arnold/venv/bin/python scripts/split_data.py")
        return

    if not en_input.exists():
        print(f"✗ Error: {config.tgt_lang} training file not found: {en_input}")
        print("  Please run: /home/arnold/venv/bin/python scripts/split_data.py")
        return

    if config.use_shared_vocab:
        # Train shared vocabulary
        print("Training shared vocabulary...")
        print()

        # Create temporary combined corpus
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8',
                                         suffix='.txt', delete=False) as tmp:
            combined_file = tmp.name

        try:
            create_combined_corpus(ko_input, en_input, combined_file)

            print("Training shared tokenizer...")
            train_sentencepiece(
                input_file=combined_file,
                output_prefix=str(output_dir / "shared_spm"),
                vocab_size=config.vocab_size,
                character_coverage=config.character_coverage,
                model_type=config.spm_model_type
            )
        finally:
            # Clean up temporary file
            if os.path.exists(combined_file):
                os.remove(combined_file)
                print(f"✓ Cleaned up temporary file")
                print()

        print("=" * 60)
        print("Training Complete!")
        print("=" * 60)
        print()
        print("Output files:")
        print(f"  Shared: {output_dir}/shared_spm.{{model,vocab}}")
        print()
        print(f"Next step: Test tokenization")
        print(f"  /home/arnold/venv/bin/python tests/test_tokenizer.py")

    else:
        # Train separate vocabularies
        print("Training separate vocabularies...")
        print()

        # Train source language tokenizer
        print(f"Training {config.src_lang} tokenizer...")
        train_sentencepiece(
            input_file=str(ko_input),
            output_prefix=str(output_dir / f"{config.src_lang}_spm"),
            vocab_size=config.vocab_size,
            character_coverage=config.character_coverage,
            model_type=config.spm_model_type
        )

        # Train target language tokenizer
        print(f"Training {config.tgt_lang} tokenizer...")
        train_sentencepiece(
            input_file=str(en_input),
            output_prefix=str(output_dir / f"{config.tgt_lang}_spm"),
            vocab_size=config.vocab_size,
            character_coverage=1.0,  # Full coverage for English
            model_type=config.spm_model_type
        )

        print("=" * 60)
        print("Training Complete!")
        print("=" * 60)
        print()
        print("Output files:")
        print(f"  {config.src_lang}:  {output_dir}/{config.src_lang}_spm.{{model,vocab}}")
        print(f"  {config.tgt_lang}:  {output_dir}/{config.tgt_lang}_spm.{{model,vocab}}")
        print()
        print("Next step: Test tokenization")
        print("  /home/arnold/venv/bin/python tests/test_tokenizer.py")


if __name__ == "__main__":
    main()
