"""
PyTorch Dataset for parallel translation corpus.

Handles loading, tokenization, and batching of sentence pairs.
Supports both shared and separate vocabularies.
"""

import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from pathlib import Path


class TranslationDataset(Dataset):
    """Dataset for Korean-English parallel corpus."""

    def __init__(self, src_file, tgt_file, src_tokenizer, tgt_tokenizer, max_len=None):
        """
        Initialize dataset with parallel text files and tokenizers.

        Args:
            src_file: Path to source language file (.ko)
            tgt_file: Path to target language file (.en)
            src_tokenizer: Source tokenizer (SentencePieceTokenizer)
            tgt_tokenizer: Target tokenizer (SentencePieceTokenizer)
            max_len: Maximum sequence length (optional, None = no limit)
        """
        # Load data
        with open(src_file, 'r', encoding='utf-8') as f:
            self.src_lines = [line.strip() for line in f if line.strip()]

        with open(tgt_file, 'r', encoding='utf-8') as f:
            self.tgt_lines = [line.strip() for line in f if line.strip()]

        assert len(self.src_lines) == len(self.tgt_lines), \
            f"Mismatch: {len(self.src_lines)} source vs {len(self.tgt_lines)} target sentences"

        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.src_lines)

    def __getitem__(self, idx):
        """
        Get a single sentence pair, tokenized and converted to IDs.

        Returns:
            Dictionary with 'src' and 'tgt' token ID tensors
        """
        src_text = self.src_lines[idx]
        tgt_text = self.tgt_lines[idx]

        # Tokenize and convert to IDs
        src_ids = self.src_tokenizer.encode_ids(src_text)
        tgt_ids = self.tgt_tokenizer.encode_ids(tgt_text)

        # Add BOS (beginning of sequence) and EOS (end of sequence) tokens
        # Format: [BOS, token1, token2, ..., tokenN, EOS]
        src_ids = [self.src_tokenizer.bos_id] + src_ids + [self.src_tokenizer.eos_id]
        tgt_ids = [self.tgt_tokenizer.bos_id] + tgt_ids + [self.tgt_tokenizer.eos_id]

        # Truncate if needed (keep BOS/EOS)
        if self.max_len and len(src_ids) > self.max_len:
            src_ids = src_ids[:self.max_len-1] + [self.src_tokenizer.eos_id]
        if self.max_len and len(tgt_ids) > self.max_len:
            tgt_ids = tgt_ids[:self.max_len-1] + [self.tgt_tokenizer.eos_id]

        # Convert to tensors
        return {
            'src': torch.tensor(src_ids, dtype=torch.long),
            'tgt': torch.tensor(tgt_ids, dtype=torch.long)
        }


def collate_fn(batch, pad_idx=0):
    """
    Collate function for DataLoader.

    Pads sequences in a batch to the same length and creates attention masks.

    Args:
        batch: List of samples from __getitem__ (list of dicts)
        pad_idx: Padding token ID (default: 0)

    Returns:
        Dictionary with:
            - src: Padded source sequences [batch_size, max_src_len]
            - tgt: Padded target sequences [batch_size, max_tgt_len]
            - src_mask: Source padding mask [batch_size, 1, max_src_len, max_src_len]
            - tgt_mask: Target mask (padding + causal) [batch_size, 1, max_tgt_len, max_tgt_len]
            - cross_mask: Cross-attention mask [batch_size, 1, max_tgt_len, max_src_len]
    """
    from src.utils.masking import create_padding_mask, create_target_mask, create_cross_attention_mask

    # Extract src and tgt from batch
    src_batch = [item['src'] for item in batch]
    tgt_batch = [item['tgt'] for item in batch]

    # Pad sequences to same length in batch
    # pad_sequence pads to the longest sequence in the batch
    src_padded = pad_sequence(src_batch, batch_first=True, padding_value=pad_idx)
    tgt_padded = pad_sequence(tgt_batch, batch_first=True, padding_value=pad_idx)

    # Create masks using utility functions
    # src_mask: [batch_size, 1, max_src_len, max_src_len] - for encoder self-attention
    # tgt_mask: [batch_size, 1, max_tgt_len, max_tgt_len] - for decoder self-attention (causal + padding)
    # cross_mask: [batch_size, 1, max_tgt_len, max_src_len] - for decoder cross-attention
    src_mask = create_padding_mask(src_padded, pad_idx)
    tgt_mask = create_target_mask(tgt_padded, pad_idx)
    cross_mask = create_cross_attention_mask(src_padded, tgt_padded, pad_idx)

    return {
        'src': src_padded,           # [batch_size, max_src_len]
        'tgt': tgt_padded,           # [batch_size, max_tgt_len]
        'src_mask': src_mask,        # [batch_size, 1, max_src_len, max_src_len]
        'tgt_mask': tgt_mask,        # [batch_size, 1, max_tgt_len, max_tgt_len]
        'cross_mask': cross_mask     # [batch_size, 1, max_tgt_len, max_src_len]
    }


def create_dataloader(src_file, tgt_file, src_tokenizer, tgt_tokenizer,
                     batch_size=32, max_len=None, shuffle=True, num_workers=0):
    """
    Convenience function to create a DataLoader.

    Args:
        src_file: Path to source file
        tgt_file: Path to target file
        src_tokenizer: Source tokenizer
        tgt_tokenizer: Target tokenizer
        batch_size: Batch size
        max_len: Maximum sequence length
        shuffle: Whether to shuffle data
        num_workers: Number of worker processes for data loading

    Returns:
        DataLoader instance
    """
    from torch.utils.data import DataLoader
    from functools import partial

    dataset = TranslationDataset(
        src_file, tgt_file,
        src_tokenizer, tgt_tokenizer,
        max_len=max_len
    )

    # Create collate_fn with correct pad_idx
    pad_idx = src_tokenizer.pad_id
    collate_fn_with_pad = partial(collate_fn, pad_idx=pad_idx)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn_with_pad,
        num_workers=num_workers
    )

    return dataloader


def load_tokenizers(vocab_dir, use_shared_vocab=False):
    """
    Load tokenizers based on vocabulary configuration.

    Args:
        vocab_dir: Directory containing tokenizer models
        use_shared_vocab: If True, load shared tokenizer for both languages
                         If False, load separate tokenizers for src/tgt

    Returns:
        tuple: (src_tokenizer, tgt_tokenizer)
               If shared vocab, both will be the same tokenizer instance

    Example:
        # Separate vocabularies
        src_tok, tgt_tok = load_tokenizers("data/vocab", use_shared_vocab=False)
        # src_tok uses ko_spm.model, tgt_tok uses en_spm.model

        # Shared vocabulary
        src_tok, tgt_tok = load_tokenizers("data/vocab", use_shared_vocab=True)
        # Both use shared_spm.model (same object)
    """
    from .tokenizer import SentencePieceTokenizer

    vocab_dir = Path(vocab_dir)

    if use_shared_vocab:
        # Load single shared tokenizer for both languages
        shared_model = vocab_dir / "shared_spm.model"
        if not shared_model.exists():
            raise FileNotFoundError(
                f"Shared tokenizer not found: {shared_model}\n"
                "Train it with: /home/arnold/venv/bin/python scripts/train_tokenizer.py --shared"
            )

        tokenizer = SentencePieceTokenizer(str(shared_model))
        return tokenizer, tokenizer  # Return same instance for both

    else:
        # Load separate tokenizers for source and target
        ko_model = vocab_dir / "ko_spm.model"
        en_model = vocab_dir / "en_spm.model"

        if not ko_model.exists():
            raise FileNotFoundError(
                f"Korean tokenizer not found: {ko_model}\n"
                "Train it with: /home/arnold/venv/bin/python scripts/train_tokenizer.py"
            )

        if not en_model.exists():
            raise FileNotFoundError(
                f"English tokenizer not found: {en_model}\n"
                "Train it with: /home/arnold/venv/bin/python scripts/train_tokenizer.py"
            )

        src_tokenizer = SentencePieceTokenizer(str(ko_model))
        tgt_tokenizer = SentencePieceTokenizer(str(en_model))

        return src_tokenizer, tgt_tokenizer
