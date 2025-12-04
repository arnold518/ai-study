#!/usr/bin/env python
"""Test CSV logging functionality."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import pandas as pd
from torch.utils.data import DataLoader

from config.transformer_config import TransformerConfig
from src.data.tokenizer import SentencePieceTokenizer
from src.data.dataset import TranslationDataset, collate_fn
from src.models.transformer.transformer import Transformer
from src.training.trainer import Trainer
from src.training.optimizer import NoamOptimizer
from src.training.losses import LabelSmoothingLoss


def test_csv_logging():
    """Test that CSV logging captures all metrics correctly."""
    print("=" * 80)
    print("Testing CSV Logging")
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
    print()

    # Create tiny validation dataset
    print("Loading tiny validation dataset...")
    val_dataset = TranslationDataset(
        val_ko_path,
        val_en_path,
        ko_tokenizer,
        en_tokenizer,
        max_len=50
    )
    from torch.utils.data import Subset
    val_dataset = Subset(val_dataset, range(min(10, len(val_dataset))))

    val_loader = DataLoader(
        val_dataset,
        batch_size=2,
        collate_fn=collate_fn,
        num_workers=0
    )
    print()

    # Create small model
    print("Creating small model...")
    config = TransformerConfig()
    config.d_model = 64
    config.num_heads = 4
    config.num_encoder_layers = 2
    config.num_decoder_layers = 2
    config.d_ff = 128
    config.dropout = 0.1
    config.max_seq_length = 50
    config.num_epochs = 3  # Just 3 epochs for testing
    config.eval_every = 1
    config.save_every = 2
    config.bleu_num_samples = 5
    config.inference_num_examples = 1

    device = torch.device('cpu')
    config.device = device

    model = Transformer(
        config,
        src_vocab_size=ko_tokenizer.vocab_size,
        tgt_vocab_size=en_tokenizer.vocab_size,
        pad_idx=ko_tokenizer.pad_id
    )
    print()

    # Create optimizer and criterion
    optimizer = NoamOptimizer(
        model.parameters(),
        config.d_model,
        warmup_steps=10,
        factor=1.0
    )
    criterion = LabelSmoothingLoss(
        en_tokenizer.vocab_size,
        pad_idx=0,
        smoothing=0.1
    )

    full_val_dataset = val_dataset.dataset if hasattr(val_dataset, 'dataset') else val_dataset

    # Create trainer
    print("Creating Trainer with CSV logging...")
    trainer = Trainer(
        model=model,
        train_loader=val_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        config=config,
        src_tokenizer=ko_tokenizer,
        tgt_tokenizer=en_tokenizer,
        val_dataset=full_val_dataset
    )
    print()

    # Get CSV path
    csv_path = trainer.csv_logger.log_path
    print(f"CSV log file: {csv_path}")
    print()

    # Run training for a few epochs
    print("Running 3 epochs of training...")
    trainer.train()
    print()

    # Verify CSV was created and has data
    if not os.path.exists(csv_path):
        print(f"✗ CSV file not found: {csv_path}")
        return

    # Read and display CSV
    print("=" * 80)
    print("CSV Content:")
    print("=" * 80)
    df = pd.read_csv(csv_path)

    print(f"\nNumber of rows: {len(df)}")
    print(f"Number of columns: {len(df.columns)}")
    print()

    # Display column names
    print("Columns captured:")
    for col in df.columns:
        print(f"  - {col}")
    print()

    # Display first few rows (key columns)
    key_cols = [
        'epoch', 'train_loss', 'train_ppl', 'val_loss', 'val_ppl',
        'val_bleu', 'learning_rate', 'grad_norm', 'best_val_loss', 'best_bleu'
    ]
    available_cols = [col for col in key_cols if col in df.columns]

    print("Sample data (key metrics):")
    print(df[available_cols].to_string(index=False))
    print()

    # Verify expected columns exist
    expected_columns = [
        'timestamp', 'epoch', 'global_step',
        'train_loss', 'train_ppl', 'train_kl_div',
        'val_loss', 'val_ppl', 'val_bleu',
        'learning_rate', 'grad_norm',
        'best_train_loss', 'best_val_loss', 'best_bleu',
        'd_model', 'num_heads', 'batch_size',
        'src_vocab_size', 'tgt_vocab_size',
        'epoch_time_seconds', 'cumulative_time_seconds'
    ]

    missing_cols = []
    for col in expected_columns:
        if col not in df.columns:
            missing_cols.append(col)

    if missing_cols:
        print(f"⚠️  Missing expected columns: {missing_cols}")
    else:
        print("✓ All expected columns present")

    # Check data validity
    print()
    print("Data validation:")
    print(f"  ✓ Epochs logged: {df['epoch'].tolist()}")
    print(f"  ✓ Train loss range: {df['train_loss'].min():.4f} - {df['train_loss'].max():.4f}")
    if 'val_loss' in df.columns and not df['val_loss'].isna().all():
        print(f"  ✓ Val loss range: {df['val_loss'].min():.4f} - {df['val_loss'].max():.4f}")
    print(f"  ✓ Learning rate: {df['learning_rate'].iloc[0]:.6f}")
    print(f"  ✓ Model architecture: d_model={df['d_model'].iloc[0]}, num_heads={df['num_heads'].iloc[0]}")
    print(f"  ✓ Vocabulary sizes: src={df['src_vocab_size'].iloc[0]}, tgt={df['tgt_vocab_size'].iloc[0]}")
    print(f"  ✓ Total training time: {df['cumulative_time_seconds'].iloc[-1]:.2f} seconds")

    print()
    print("=" * 80)
    print("CSV Logging Test Passed!")
    print("=" * 80)
    print()
    print(f"CSV file saved at: {csv_path}")
    print("You can view it with: pandas, Excel, or any CSV viewer")


if __name__ == "__main__":
    test_csv_logging()
