"""
Tokenizers for Korean and English text.

Supports SentencePiece subword tokenization (recommended for NMT).
"""

import sentencepiece as spm
from pathlib import Path


class SentencePieceTokenizer:
    """SentencePiece-based tokenizer wrapper."""

    def __init__(self, model_path):
        """
        Initialize tokenizer with trained SentencePiece model.

        Args:
            model_path: Path to trained SentencePiece model (.model file)
        """
        if not Path(model_path).exists():
            raise FileNotFoundError(f"SentencePiece model not found: {model_path}")

        self.model_path = model_path
        self.sp = spm.SentencePieceProcessor(model_file=model_path)

    def tokenize(self, text):
        """
        Tokenize text into subword tokens.

        Args:
            text: Input text string

        Returns:
            List of token strings

        Example:
            >>> tokenizer.tokenize("안녕하세요")
            ['▁안녕', '하', '세요']
        """
        return self.sp.encode(text, out_type=str)

    def detokenize(self, tokens):
        """
        Convert tokens back to text.

        Args:
            tokens: List of token strings

        Returns:
            Detokenized text string

        Example:
            >>> tokenizer.detokenize(['▁안녕', '하', '세요'])
            '안녕하세요'
        """
        return self.sp.decode(tokens)

    def encode_ids(self, text):
        """
        Tokenize text and convert to token IDs.

        Args:
            text: Input text string

        Returns:
            List of token IDs (integers)

        Example:
            >>> tokenizer.encode_ids("안녕하세요")
            [234, 567, 890]
        """
        return self.sp.encode(text, out_type=int)

    def decode_ids(self, ids):
        """
        Convert token IDs back to text.

        Args:
            ids: List of token IDs (integers)

        Returns:
            Decoded text string

        Example:
            >>> tokenizer.decode_ids([234, 567, 890])
            '안녕하세요'
        """
        return self.sp.decode(ids)

    def encode_as_pieces(self, text):
        """
        Tokenize text into pieces (same as tokenize).

        Args:
            text: Input text string

        Returns:
            List of token strings
        """
        return self.sp.encode_as_pieces(text)

    @property
    def vocab_size(self):
        """Return vocabulary size."""
        return len(self.sp)

    @property
    def pad_id(self):
        """Return padding token ID."""
        return self.sp.pad_id()

    @property
    def unk_id(self):
        """Return unknown token ID."""
        return self.sp.unk_id()

    @property
    def bos_id(self):
        """Return beginning-of-sentence token ID."""
        return self.sp.bos_id()

    @property
    def eos_id(self):
        """Return end-of-sentence token ID."""
        return self.sp.eos_id()

    def id_to_piece(self, idx):
        """Convert token ID to token string."""
        return self.sp.id_to_piece(idx)

    def piece_to_id(self, piece):
        """Convert token string to token ID."""
        return self.sp.piece_to_id(piece)

    def get_vocab_size(self):
        """Return vocabulary size."""
        return self.vocab_size

    def get_piece_size(self):
        """Alias for vocab_size."""
        return self.vocab_size

    def __len__(self):
        """Return vocabulary size."""
        return self.vocab_size

    def __repr__(self):
        return f"SentencePieceTokenizer(model_path='{self.model_path}', vocab_size={self.vocab_size})"
