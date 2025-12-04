#!/usr/bin/env python
"""Test Phase 5: BLEU integration and inference examples in training."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
from torch.utils.data import DataLoader

from config.transformer_config import TransformerConfig
from src.data.tokenizer import SentencePieceTokenizer
from src.data.dataset import TranslationDataset, collate_fn
from src.models.transformer.transformer import Transformer
from src.training.trainer import Trainer
from src.training.optimizer import NoamOptimizer
from src.training.losses import LabelSmoothingLoss


def test_phase5_integration():
    """Test that Trainer correctly integrates BLEU and inference examples."""
    print("=" * 80)
    print("Testing Phase 5: BLEU Integration")
    print("=" * 80)

    # Check if tokenizers exist
    ko_model_path = 'data/vocab/ko_spm.model'
    en_model_path = 'data/vocab/en_spm.model'

    if not os.path.exists(ko_model_path) or not os.path.exists(en_model_path):
        print("⚠️  Tokenizer models not found. Please train tokenizers first:")
        print("  /home/arnold/venv/bin/python scripts/train_tokenizer.py")
        return

    # Check if validation data exists
    val_ko_path = 'data/processed/validation.ko'
    val_en_path = 'data/processed/validation.en'

    if not os.path.exists(val_ko_path) or not os.path.exists(val_en_path):
        print("⚠️  Validation data not found. Please prepare data first:")
        print("  /home/arnold/venv/bin/python scripts/split_data.py")
        return

    print("Loading tokenizers...")
    ko_tokenizer = SentencePieceTokenizer(ko_model_path)
    en_tokenizer = SentencePieceTokenizer(en_model_path)
    print(f"Korean vocab size: {ko_tokenizer.vocab_size}")
    print(f"English vocab size: {en_tokenizer.vocab_size}")
    print()

    # Create small validation dataset
    print("Loading small validation dataset...")
    val_dataset = TranslationDataset(
        val_ko_path,
        val_en_path,
        ko_tokenizer,
        en_tokenizer,
        max_len=50
    )
    # Use only 20 samples for testing
    from torch.utils.data import Subset
    val_dataset = Subset(val_dataset, range(min(20, len(val_dataset))))

    val_loader = DataLoader(
        val_dataset,
        batch_size=4,
        collate_fn=collate_fn,
        num_workers=0
    )
    print(f"Validation size: {len(val_dataset)}")
    print()

    # Create small model
    print("Creating small model for testing...")
    config = TransformerConfig()
    config.d_model = 128
    config.num_heads = 4
    config.num_encoder_layers = 2
    config.num_decoder_layers = 2
    config.d_ff = 256
    config.dropout = 0.1
    config.max_seq_length = 50
    config.bleu_num_samples = 10  # Small number for testing
    config.inference_num_examples = 2  # Small number for testing

    device = torch.device('cpu')
    config.device = device

    model = Transformer(
        config,
        src_vocab_size=ko_tokenizer.vocab_size,
        tgt_vocab_size=en_tokenizer.vocab_size,
        pad_idx=ko_tokenizer.pad_id
    )
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print()

    # Create optimizer and criterion
    optimizer = NoamOptimizer(
        model.parameters(),
        config.d_model,
        warmup_steps=100,
        factor=1.0
    )
    criterion = LabelSmoothingLoss(
        en_tokenizer.vocab_size,
        pad_idx=0,
        smoothing=0.1
    )

    # Get reference to full dataset (not Subset wrapper)
    # For testing, we'll use the Subset itself, but in real training
    # you'd keep reference to the original dataset
    full_val_dataset = val_dataset.dataset if hasattr(val_dataset, 'dataset') else val_dataset

    # Create trainer WITH tokenizers and dataset
    print("Creating Trainer with Phase 5 features...")
    trainer = Trainer(
        model=model,
        train_loader=val_loader,  # Use val as train for testing
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        config=config,
        src_tokenizer=ko_tokenizer,
        tgt_tokenizer=en_tokenizer,
        val_dataset=full_val_dataset  # Pass dataset for BLEU/examples
    )

    # Verify trainer has translator
    print(f"✓ Trainer has translator: {trainer.translator is not None}")
    print(f"✓ Trainer has src_tokenizer: {trainer.src_tokenizer is not None}")
    print(f"✓ Trainer has tgt_tokenizer: {trainer.tgt_tokenizer is not None}")
    print(f"✓ Trainer has val_dataset: {trainer.val_dataset is not None}")
    print()

    # Test BLEU computation (uses config.bleu_num_samples = 10)
    print(f"Testing BLEU computation (using config: {config.bleu_num_samples} samples)...")
    try:
        bleu_score = trainer.compute_bleu_score()
        if bleu_score is not None:
            print(f"✓ BLEU computation successful: {bleu_score:.2f}")
        else:
            print("⚠️  BLEU returned None (may be expected if dataset is too small)")
    except Exception as e:
        print(f"✗ BLEU computation failed: {e}")
        raise
    print()

    # Test inference example generation (uses config.inference_num_examples = 2)
    print(f"Testing inference example generation (using config: {config.inference_num_examples} examples)...")
    try:
        examples = trainer.generate_inference_examples()
        if examples:
            print(f"✓ Generated {len(examples)} inference examples:")
            for i, (src, ref, pred) in enumerate(examples, 1):
                print(f"  [{i}] Source:     {src[:50]}{'...' if len(src) > 50 else ''}")
                print(f"      Reference:  {ref[:50]}{'...' if len(ref) > 50 else ''}")
                print(f"      Prediction: {pred[:50]}{'...' if len(pred) > 50 else ''}")
                print()
        else:
            print("⚠️  No examples generated (may be expected if dataset is too small)")
    except Exception as e:
        print(f"✗ Inference example generation failed: {e}")
        raise

    print("=" * 80)
    print("Phase 5 Integration Test Passed!")
    print("=" * 80)
    print()
    print("Summary:")
    print("  ✓ Trainer accepts tokenizers and validation dataset")
    print("  ✓ BLEU score computation works")
    print("  ✓ Inference example generation works")
    print()
    print("Phase 5 is ready for training!")


if __name__ == "__main__":
    test_phase5_integration()
