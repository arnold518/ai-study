"""Attention visualization tools.

Visualize attention weights from Transformer models to understand
translation alignment and debugging attention mechanisms.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


class AttentionVisualizer:
    """Visualize attention weights from Transformer models."""

    def __init__(self, save_dir='outputs/attention_plots'):
        """
        Initialize visualizer.

        Args:
            save_dir: Directory to save plots
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def plot_attention(self, attention_weights, src_tokens, tgt_tokens,
                      layer_idx=0, head_idx=0, title=None, save_path=None):
        """
        Plot attention heatmap.

        Args:
            attention_weights: Attention tensor [batch, num_heads, tgt_len, src_len]
            src_tokens: List of source tokens
            tgt_tokens: List of target tokens
            layer_idx: Layer index (for title)
            head_idx: Head index to visualize
            title: Optional custom title
            save_path: Optional path to save plot

        Returns:
            fig: Matplotlib figure
        """
        # Extract attention for specific head
        if attention_weights.dim() == 4:
            # [batch, num_heads, tgt_len, src_len]
            attn = attention_weights[0, head_idx].cpu().numpy()
        elif attention_weights.dim() == 3:
            # [num_heads, tgt_len, src_len]
            attn = attention_weights[head_idx].cpu().numpy()
        else:
            # [tgt_len, src_len]
            attn = attention_weights.cpu().numpy()

        # Create figure
        fig, ax = plt.subplots(figsize=(max(10, len(src_tokens) * 0.5),
                                        max(8, len(tgt_tokens) * 0.5)))

        # Plot heatmap
        sns.heatmap(attn, xticklabels=src_tokens, yticklabels=tgt_tokens,
                   cmap='viridis', cbar=True, ax=ax, vmin=0, vmax=1,
                   square=True, linewidths=0.5, linecolor='white')

        # Set title
        if title is None:
            title = f'Attention Weights (Layer {layer_idx}, Head {head_idx})'
        ax.set_title(title, fontsize=14, pad=20)
        ax.set_xlabel('Source Tokens', fontsize=12)
        ax.set_ylabel('Target Tokens', fontsize=12)

        # Rotate labels for better readability
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
        plt.setp(ax.get_yticklabels(), rotation=0)

        plt.tight_layout()

        # Save if path provided
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved attention plot to: {save_path}")

        return fig

    def plot_multihead_attention(self, attention_weights, src_tokens, tgt_tokens,
                                layer_idx=0, num_heads=8, save_path=None):
        """
        Plot all attention heads in a grid.

        Args:
            attention_weights: Attention tensor [batch, num_heads, tgt_len, src_len]
            src_tokens: List of source tokens
            tgt_tokens: List of target tokens
            layer_idx: Layer index (for title)
            num_heads: Number of heads to plot
            save_path: Optional path to save plot

        Returns:
            fig: Matplotlib figure
        """
        # Extract batch
        if attention_weights.dim() == 4:
            attn = attention_weights[0].cpu().numpy()  # [num_heads, tgt_len, src_len]
        else:
            attn = attention_weights.cpu().numpy()

        # Determine grid size
        n_cols = 4
        n_rows = (num_heads + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5 * n_rows))
        axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes

        for head_idx in range(num_heads):
            ax = axes[head_idx]

            # Plot heatmap
            sns.heatmap(attn[head_idx], xticklabels=src_tokens, yticklabels=tgt_tokens,
                       cmap='viridis', cbar=True, ax=ax, vmin=0, vmax=1,
                       square=True, linewidths=0.3, linecolor='white')

            ax.set_title(f'Head {head_idx}', fontsize=10)
            ax.set_xlabel('Source', fontsize=8)
            ax.set_ylabel('Target', fontsize=8)

            # Smaller labels
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right', fontsize=7)
            plt.setp(ax.get_yticklabels(), rotation=0, fontsize=7)

        # Hide unused subplots
        for idx in range(num_heads, len(axes)):
            axes[idx].axis('off')

        fig.suptitle(f'Multi-Head Attention (Layer {layer_idx})', fontsize=16, y=1.00)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved multi-head attention plot to: {save_path}")

        return fig

    def plot_attention_summary(self, attention_weights, src_tokens, tgt_tokens,
                             layer_idx=0, save_path=None):
        """
        Plot average attention across all heads.

        Args:
            attention_weights: Attention tensor [batch, num_heads, tgt_len, src_len]
            src_tokens: List of source tokens
            tgt_tokens: List of target tokens
            layer_idx: Layer index (for title)
            save_path: Optional path to save plot

        Returns:
            fig: Matplotlib figure
        """
        # Average across heads
        if attention_weights.dim() == 4:
            attn = attention_weights[0].mean(dim=0).cpu().numpy()  # [tgt_len, src_len]
        elif attention_weights.dim() == 3:
            attn = attention_weights.mean(dim=0).cpu().numpy()
        else:
            attn = attention_weights.cpu().numpy()

        # Create figure
        fig, ax = plt.subplots(figsize=(max(10, len(src_tokens) * 0.5),
                                        max(8, len(tgt_tokens) * 0.5)))

        # Plot heatmap
        sns.heatmap(attn, xticklabels=src_tokens, yticklabels=tgt_tokens,
                   cmap='viridis', cbar=True, ax=ax, vmin=0, vmax=1,
                   square=True, linewidths=0.5, linecolor='white')

        ax.set_title(f'Average Attention (Layer {layer_idx}, All Heads)', fontsize=14, pad=20)
        ax.set_xlabel('Source Tokens', fontsize=12)
        ax.set_ylabel('Target Tokens', fontsize=12)

        plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
        plt.setp(ax.get_yticklabels(), rotation=0)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved average attention plot to: {save_path}")

        return fig

    def visualize_translation_attention(self, model, src, tgt, src_mask, tgt_mask, cross_mask,
                                      src_tokens, tgt_tokens, layer_idx=-1, save_prefix=None):
        """
        Visualize attention for a translation example.

        Args:
            model: Transformer model
            src: Source tensor [batch, src_len]
            tgt: Target tensor [batch, tgt_len]
            src_mask: Source mask
            tgt_mask: Target mask
            cross_mask: Cross-attention mask
            src_tokens: List of source tokens
            tgt_tokens: List of target tokens
            layer_idx: Layer to visualize (-1 for last layer)
            save_prefix: Prefix for saved files

        Returns:
            figures: Dictionary of matplotlib figures
        """
        model.eval()
        figures = {}

        with torch.no_grad():
            # Forward pass
            output = model(src, tgt, src_mask, tgt_mask, cross_mask)

            # Extract attention from decoder
            # Assumes model stores attention weights during forward pass
            if hasattr(model.decoder, 'layers'):
                decoder_layer = model.decoder.layers[layer_idx]

                # Cross-attention (decoder-encoder)
                if hasattr(decoder_layer, 'cross_attn_weights'):
                    cross_attn = decoder_layer.cross_attn_weights

                    # Plot average attention
                    save_path = None
                    if save_prefix:
                        save_path = self.save_dir / f"{save_prefix}_cross_attn_layer{layer_idx}.png"

                    fig = self.plot_attention_summary(
                        cross_attn, src_tokens, tgt_tokens,
                        layer_idx=layer_idx, save_path=save_path
                    )
                    figures['cross_attention'] = fig

                # Self-attention (decoder)
                if hasattr(decoder_layer, 'self_attn_weights'):
                    self_attn = decoder_layer.self_attn_weights

                    save_path = None
                    if save_prefix:
                        save_path = self.save_dir / f"{save_prefix}_self_attn_layer{layer_idx}.png"

                    fig = self.plot_attention_summary(
                        self_attn, tgt_tokens, tgt_tokens,
                        layer_idx=layer_idx, save_path=save_path
                    )
                    figures['self_attention'] = fig

        return figures


def extract_attention_weights(model, layer_idx=-1, attention_type='cross'):
    """
    Extract attention weights from a model layer.

    Note: This requires the model to store attention weights during forward pass.
    Modify MultiHeadAttention to save attention weights as an attribute.

    Args:
        model: Transformer model
        layer_idx: Layer index (-1 for last layer)
        attention_type: 'cross' or 'self'

    Returns:
        attention_weights: Tensor of attention weights
    """
    if not hasattr(model.decoder, 'layers'):
        raise ValueError("Model does not have accessible decoder layers")

    layer = model.decoder.layers[layer_idx]

    if attention_type == 'cross':
        if hasattr(layer, 'cross_attn_weights'):
            return layer.cross_attn_weights
        else:
            raise ValueError("Cross-attention weights not found. Modify MultiHeadAttention to store weights.")

    elif attention_type == 'self':
        if hasattr(layer, 'self_attn_weights'):
            return layer.self_attn_weights
        else:
            raise ValueError("Self-attention weights not found. Modify MultiHeadAttention to store weights.")

    else:
        raise ValueError(f"Unknown attention type: {attention_type}")


def save_attention_weights(attention_weights, save_path):
    """
    Save attention weights to disk.

    Args:
        attention_weights: Attention tensor
        save_path: Path to save (e.g., 'attention_layer0_head0.pt')
    """
    torch.save(attention_weights.cpu(), save_path)
    print(f"Saved attention weights to: {save_path}")


def load_attention_weights(load_path):
    """
    Load attention weights from disk.

    Args:
        load_path: Path to load from

    Returns:
        attention_weights: Loaded attention tensor
    """
    attention_weights = torch.load(load_path)
    return attention_weights
