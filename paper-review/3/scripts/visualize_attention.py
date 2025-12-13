#!/usr/bin/env python
"""
Visualize attention weights from trained Transformer model.

Usage:
    /home/arnold/venv/bin/python scripts/visualize_attention.py
    /home/arnold/venv/bin/python scripts/visualize_attention.py --checkpoint checkpoints/best_model.pt
    /home/arnold/venv/bin/python scripts/visualize_attention.py --input "한국어 문장" --layer 5
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import argparse
import torch
import matplotlib.pyplot as plt

from config.transformer_config import TransformerConfig
from src.data.tokenizer import SentencePieceTokenizer
from src.models.transformer.transformer import Transformer
from src.utils.checkpointing import load_checkpoint
from src.utils.visualization import AttentionVisualizer
from src.utils.masking import create_padding_mask, create_look_ahead_mask


def visualize_attention_for_sentence(
    model, src_text, src_tokenizer, tgt_tokenizer,
    device, max_length=150, layer_idx=-1, visualizer=None
):
    """
    Translate a sentence and visualize attention weights.

    Args:
        model: Transformer model
        src_text: Source sentence (Korean)
        src_tokenizer: Source tokenizer
        tgt_tokenizer: Target tokenizer
        device: Device to run on
        max_length: Maximum generation length
        layer_idx: Which decoder layer to visualize (-1 for last)
        visualizer: AttentionVisualizer instance

    Returns:
        translation: Generated translation
        figures: Dictionary of matplotlib figures
    """
    if visualizer is None:
        visualizer = AttentionVisualizer()

    model.eval()

    # Enable attention storage for decoder
    model.decoder.set_store_attention(True)

    # Tokenize source
    src_tokens = src_tokenizer.tokenize(src_text)
    src_ids = src_tokenizer.encode_ids(src_text)
    src = torch.tensor([src_ids], dtype=torch.long).to(device)  # [1, src_len]

    # Create source mask
    src_mask = create_padding_mask(src, pad_idx=0)  # [1, 1, 1, src_len]

    # Start with BOS token
    bos_id = tgt_tokenizer.bos_id
    eos_id = tgt_tokenizer.eos_id
    tgt = torch.tensor([[bos_id]], dtype=torch.long).to(device)  # [1, 1]

    tgt_tokens = ['<s>']

    # Greedy decoding
    with torch.no_grad():
        for _ in range(max_length):
            # Create target mask (causal + padding)
            tgt_len = tgt.size(1)
            tgt_padding_mask = create_padding_mask(tgt, pad_idx=0)  # [1, 1, tgt_len, tgt_len]
            tgt_causal_mask = create_look_ahead_mask(tgt_len).unsqueeze(0).to(device)  # [1, 1, tgt_len, tgt_len]
            tgt_mask = tgt_padding_mask & tgt_causal_mask  # [1, 1, tgt_len, tgt_len]

            # Create cross-attention mask
            cross_mask = src_mask  # [1, 1, 1, src_len] broadcasts to [1, 1, tgt_len, src_len]

            # Forward pass
            logits = model(src, tgt, src_mask, tgt_mask, cross_mask)  # [1, tgt_len, vocab]

            # Get next token (from last position)
            next_token_logits = logits[:, -1, :]  # [1, vocab]
            next_token = torch.argmax(next_token_logits, dim=-1)  # [1]

            # Append to target
            tgt = torch.cat([tgt, next_token.unsqueeze(1)], dim=1)  # [1, tgt_len+1]

            # Decode token
            token_text = tgt_tokenizer.decode_ids([next_token.item()])
            tgt_tokens.append(token_text)

            # Stop if EOS
            if next_token.item() == eos_id:
                break

    # Decode full translation
    translation = tgt_tokenizer.decode_ids(tgt[0].cpu().tolist())

    # Get attention weights from specified layer
    decoder_layer = model.decoder.layers[layer_idx]

    figures = {}

    # Visualize cross-attention (decoder attending to source)
    if decoder_layer.cross_attn_weights is not None:
        cross_attn = decoder_layer.cross_attn_weights  # [batch, num_heads, tgt_len, src_len]

        # Plot average attention across all heads
        save_path = visualizer.save_dir / f"cross_attention_layer{layer_idx}.png"
        fig = visualizer.plot_attention_summary(
            cross_attn, src_tokens, tgt_tokens,
            layer_idx=layer_idx,
            save_path=save_path
        )
        figures['cross_attention_avg'] = fig

        # Plot all heads
        save_path = visualizer.save_dir / f"cross_attention_multihead_layer{layer_idx}.png"
        fig = visualizer.plot_multihead_attention(
            cross_attn, src_tokens, tgt_tokens,
            layer_idx=layer_idx,
            save_path=save_path
        )
        figures['cross_attention_multihead'] = fig

    # Visualize self-attention (decoder attending to itself)
    if decoder_layer.self_attn_weights is not None:
        self_attn = decoder_layer.self_attn_weights  # [batch, num_heads, tgt_len, tgt_len]

        save_path = visualizer.save_dir / f"self_attention_layer{layer_idx}.png"
        fig = visualizer.plot_attention_summary(
            self_attn, tgt_tokens, tgt_tokens,
            layer_idx=layer_idx,
            save_path=save_path
        )
        figures['self_attention_avg'] = fig

    # Disable attention storage
    model.decoder.set_store_attention(False)

    return translation, figures


def main():
    """Main visualization function."""
    parser = argparse.ArgumentParser(description='Visualize attention weights')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/best_model.pt',
                       help='Path to model checkpoint')
    parser.add_argument('--input', type=str, default=None,
                       help='Source sentence to translate (Korean)')
    parser.add_argument('--file', type=str, default=None,
                       help='File with source sentences (one per line)')
    parser.add_argument('--layer', type=int, default=-1,
                       help='Decoder layer to visualize (-1 for last layer)')
    parser.add_argument('--num-examples', type=int, default=3,
                       help='Number of validation examples to visualize')
    parser.add_argument('--output-dir', type=str, default='outputs/attention_plots',
                       help='Directory to save plots')
    parser.add_argument('--show', action='store_true',
                       help='Show plots interactively')
    args = parser.parse_args()

    print("=" * 60)
    print("Transformer Attention Visualization")
    print("=" * 60)
    print()

    # Load configuration
    config = TransformerConfig()
    device = config.device
    print(f"Device: {device}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Layer to visualize: {args.layer}")
    print()

    # Load tokenizers
    print("Loading tokenizers...")
    ko_model_path = os.path.join(config.vocab_dir, 'ko_spm.model')
    en_model_path = os.path.join(config.vocab_dir, 'en_spm.model')

    if not os.path.exists(ko_model_path) or not os.path.exists(en_model_path):
        print("ERROR: Tokenizer models not found!")
        print(f"Expected: {ko_model_path}, {en_model_path}")
        return

    ko_tokenizer = SentencePieceTokenizer(ko_model_path)
    en_tokenizer = SentencePieceTokenizer(en_model_path)
    print(f"Korean vocab size: {ko_tokenizer.vocab_size}")
    print(f"English vocab size: {en_tokenizer.vocab_size}")
    print()

    # Initialize model
    print("Loading model...")
    model = Transformer(
        config,
        src_vocab_size=ko_tokenizer.vocab_size,
        tgt_vocab_size=en_tokenizer.vocab_size
    )

    # Load checkpoint
    if not os.path.exists(args.checkpoint):
        print(f"ERROR: Checkpoint not found: {args.checkpoint}")
        return

    model, _, epoch, loss = load_checkpoint(model, None, args.checkpoint, device)
    model.eval()
    print(f"Loaded checkpoint from epoch {epoch} (loss: {loss:.4f})")
    print()

    # Create visualizer
    visualizer = AttentionVisualizer(save_dir=args.output_dir)
    print(f"Saving plots to: {args.output_dir}")
    print()

    # Visualize attention
    if args.input:
        # Single input sentence
        print(f"Translating: {args.input}")
        print()

        translation, figures = visualize_attention_for_sentence(
            model, args.input, ko_tokenizer, en_tokenizer,
            device, layer_idx=args.layer, visualizer=visualizer
        )

        print(f"Translation: {translation}")
        print()
        print(f"Generated {len(figures)} plots")

        if args.show:
            plt.show()

    elif args.file:
        # Multiple sentences from file
        print(f"Reading sentences from: {args.file}")

        with open(args.file, 'r', encoding='utf-8') as f:
            sentences = [line.strip() for line in f if line.strip()]

        print(f"Found {len(sentences)} sentences")
        print()

        for i, src_text in enumerate(sentences, 1):
            print(f"[{i}/{len(sentences)}] Translating: {src_text}")

            translation, figures = visualize_attention_for_sentence(
                model, src_text, ko_tokenizer, en_tokenizer,
                device, layer_idx=args.layer, visualizer=visualizer
            )

            print(f"  Translation: {translation}")
            print()

        if args.show:
            plt.show()

    else:
        # Use validation examples
        print(f"Visualizing {args.num_examples} validation examples...")

        val_ko_path = os.path.join(config.processed_data_dir, 'validation.ko')
        val_en_path = os.path.join(config.processed_data_dir, 'validation.en')

        if not os.path.exists(val_ko_path) or not os.path.exists(val_en_path):
            print("ERROR: Validation data not found!")
            return

        with open(val_ko_path, 'r', encoding='utf-8') as f:
            ko_lines = [line.strip() for line in f]
        with open(val_en_path, 'r', encoding='utf-8') as f:
            en_lines = [line.strip() for line in f]

        # Take first N examples
        for i in range(min(args.num_examples, len(ko_lines))):
            src_text = ko_lines[i]
            ref_text = en_lines[i]

            print(f"[{i+1}/{args.num_examples}]")
            print(f"  Source:    {src_text}")
            print(f"  Reference: {ref_text}")

            translation, figures = visualize_attention_for_sentence(
                model, src_text, ko_tokenizer, en_tokenizer,
                device, layer_idx=args.layer, visualizer=visualizer
            )

            print(f"  Predicted: {translation}")
            print()

        if args.show:
            plt.show()

    print("Visualization complete!")


if __name__ == "__main__":
    main()
