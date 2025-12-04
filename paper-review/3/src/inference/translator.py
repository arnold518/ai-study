"""Translation interface."""

import torch
from ..utils.masking import create_padding_mask
from .greedy_search import greedy_decode_cached
from .beam_search import beam_search


class Translator:
    """Interface for translating sentences."""

    def __init__(self, model, src_tokenizer, tgt_tokenizer, device='cpu', max_length=150):
        """
        Args:
            model: Trained translation model
            src_tokenizer: Source tokenizer (SentencePieceTokenizer)
            tgt_tokenizer: Target tokenizer (SentencePieceTokenizer)
            device: Device to run on
            max_length: Maximum generation length
        """
        self.model = model
        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer
        self.device = device
        self.max_length = max_length

        # Get special token indices from tokenizer
        self.pad_idx = self.tgt_tokenizer.pad_id
        self.unk_idx = self.tgt_tokenizer.unk_id
        self.bos_idx = self.tgt_tokenizer.bos_id
        self.eos_idx = self.tgt_tokenizer.eos_id

        self.model.to(device)
        self.model.eval()

    def translate(self, src_sentence, method='greedy', beam_size=4, length_penalty=0.6):
        """
        Translate a source sentence.

        Args:
            src_sentence: Source sentence (string)
            method: Decoding method ('greedy' or 'beam')
            beam_size: Beam size for beam search (ignored for greedy)
            length_penalty: Length penalty for beam search (ignored for greedy)

        Returns:
            translation: Translated sentence (string)
        """
        # 1. Tokenize and encode source sentence
        src_tokens = self.src_tokenizer.tokenize(src_sentence)
        src_ids = self.src_tokenizer.encode_ids(src_sentence)

        # Add BOS token at the beginning
        src_ids = [self.bos_idx] + src_ids

        # Convert to tensor
        src = torch.tensor([src_ids], dtype=torch.long, device=self.device)  # [1, src_len]

        # Create source mask
        src_mask = create_padding_mask(src, self.pad_idx)

        # 2. Run inference
        if method == 'greedy':
            output = greedy_decode_cached(
                model=self.model,
                src=src,
                src_mask=src_mask,
                max_length=self.max_length,
                bos_idx=self.bos_idx,
                eos_idx=self.eos_idx,
                device=self.device
            )
        elif method == 'beam':
            output = beam_search(
                model=self.model,
                src=src,
                src_mask=src_mask,
                beam_size=beam_size,
                max_length=self.max_length,
                bos_idx=self.bos_idx,
                eos_idx=self.eos_idx,
                length_penalty=length_penalty,
                device=self.device
            )
        else:
            raise ValueError(f"Unknown decoding method: {method}")

        # 3. Decode output IDs
        output_ids = output[0].tolist()  # Convert to list

        # Remove BOS and EOS tokens
        if output_ids and output_ids[0] == self.bos_idx:
            output_ids = output_ids[1:]
        if output_ids and output_ids[-1] == self.eos_idx:
            output_ids = output_ids[:-1]

        # 4. Detokenize
        translation = self.tgt_tokenizer.decode_ids(output_ids)

        return translation

    def batch_translate(self, src_sentences, method='greedy', beam_size=4, length_penalty=0.6):
        """
        Translate multiple sentences.

        Args:
            src_sentences: List of source sentences
            method: Decoding method ('greedy' or 'beam')
            beam_size: Beam size for beam search
            length_penalty: Length penalty for beam search

        Returns:
            translations: List of translated sentences
        """
        translations = []
        for src_sentence in src_sentences:
            translation = self.translate(src_sentence, method, beam_size, length_penalty)
            translations.append(translation)
        return translations
