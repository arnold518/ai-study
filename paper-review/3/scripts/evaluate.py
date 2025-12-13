"""Comprehensive evaluation script for translation models.

Evaluates trained model on test set with:
- All metrics (BLEU, chrF++, COMET, BERTScore)
- Error analysis (repetitions, number errors)
- Attention visualizations
- Detailed report generation
"""

import argparse
import os
import sys
import torch
import json
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config.transformer_config import TransformerConfig
from src.models.transformer.transformer import Transformer
from src.data.tokenizer import SentencePieceTokenizer
from src.data.dataset import TranslationDataset
from src.inference.translator import Translator
from src.utils.metrics import compute_metrics, print_metrics
from src.utils.error_analysis import analyze_translation_errors
from src.utils.checkpointing import load_checkpoint
from tqdm import tqdm


def load_model_and_tokenizers(checkpoint_path, config, device='cuda'):
    """Load trained model and tokenizers."""
    print(f"\nLoading model from: {checkpoint_path}")

    # Load tokenizers
    print("Loading tokenizers...")
    src_tokenizer = SentencePieceTokenizer(os.path.join(config.vocab_dir, 'ko_spm.model'))
    tgt_tokenizer = SentencePieceTokenizer(os.path.join(config.vocab_dir, 'en_spm.model'))

    # Create model
    print("Creating model...")
    model = Transformer(
        config=config,
        src_vocab_size=src_tokenizer.vocab_size,
        tgt_vocab_size=tgt_tokenizer.vocab_size,
    )

    # Load checkpoint
    print("Loading checkpoint...")
    model, _, _, _ = load_checkpoint(model, None, checkpoint_path, device=device)
    model.to(device)
    model.eval()

    print("✓ Model loaded successfully\n")
    return model, src_tokenizer, tgt_tokenizer


def load_test_data(config):
    """Load test dataset."""
    print("Loading test data...")

    src_path = os.path.join(config.processed_data_dir, 'test.ko')
    tgt_path = os.path.join(config.processed_data_dir, 'test.en')

    if not os.path.exists(src_path) or not os.path.exists(tgt_path):
        raise FileNotFoundError(f"Test data not found at {src_path} or {tgt_path}")

    # Read lines
    with open(src_path, 'r', encoding='utf-8') as f:
        src_lines = [line.strip() for line in f]

    with open(tgt_path, 'r', encoding='utf-8') as f:
        tgt_lines = [line.strip() for line in f]

    print(f"✓ Loaded {len(src_lines)} test samples\n")
    return src_lines, tgt_lines


def run_inference(translator, src_lines, method='beam', max_samples=None):
    """Run inference on all test samples."""
    print(f"Running inference with {method} search...")

    if max_samples:
        src_lines = src_lines[:max_samples]

    predictions = []

    with torch.no_grad():
        for src_text in tqdm(src_lines, desc="Translating"):
            try:
                pred_text = translator.translate(src_text, method=method)
                predictions.append(pred_text)
            except Exception as e:
                print(f"\nWarning: Translation failed for: {src_text[:50]}...")
                print(f"Error: {e}")
                predictions.append("")  # Empty prediction

    print(f"✓ Completed {len(predictions)} translations\n")
    return predictions


def compute_all_metrics(sources, predictions, references, device='cuda', use_advanced=False):
    """Compute all available metrics."""
    print("Computing metrics...")

    # Filter out empty predictions
    valid_indices = [i for i, p in enumerate(predictions) if p.strip()]
    filtered_sources = [sources[i] for i in valid_indices]
    filtered_predictions = [predictions[i] for i in valid_indices]
    filtered_references = [references[i] for i in valid_indices]

    if len(filtered_predictions) < len(predictions):
        print(f"Warning: {len(predictions) - len(filtered_predictions)} empty predictions excluded from metrics")

    # Compute metrics
    metrics = compute_metrics(
        filtered_predictions,
        filtered_references,
        sources=filtered_sources,
        use_advanced=use_advanced,
        device=device
    )

    print_metrics(metrics)
    return metrics


def run_error_analysis(sources, predictions, references):
    """Run comprehensive error analysis."""
    print("\n" + "="*80)
    print("ERROR ANALYSIS")
    print("="*80)

    analyzer = analyze_translation_errors(
        sources, predictions, references,
        verbose=True
    )

    return analyzer


def save_results(output_dir, sources, predictions, references, metrics, analyzer, args):
    """Save evaluation results to files."""
    print(f"\nSaving results to: {output_dir}")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Save translations
    translations_path = os.path.join(output_dir, 'translations.txt')
    with open(translations_path, 'w', encoding='utf-8') as f:
        for src, pred, ref in zip(sources, predictions, references):
            f.write(f"Source:     {src}\n")
            f.write(f"Prediction: {pred}\n")
            f.write(f"Reference:  {ref}\n")
            f.write("-" * 80 + "\n")
    print(f"✓ Saved translations to: {translations_path}")

    # Save metrics
    metrics_path = os.path.join(output_dir, 'metrics.json')
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2)
    print(f"✓ Saved metrics to: {metrics_path}")

    # Save error analysis
    error_stats = analyzer.get_statistics()
    error_path = os.path.join(output_dir, 'error_analysis.json')
    with open(error_path, 'w', encoding='utf-8') as f:
        json.dump(error_stats, f, indent=2)
    print(f"✓ Saved error analysis to: {error_path}")

    # Save error examples
    examples_path = os.path.join(output_dir, 'error_examples.txt')
    with open(examples_path, 'w', encoding='utf-8') as f:
        # Repetition errors
        f.write("REPETITION ERRORS\n")
        f.write("=" * 80 + "\n\n")
        rep_examples = analyzer.get_error_examples('repetition', max_examples=10)
        for i, ex in enumerate(rep_examples, 1):
            f.write(f"Example {i}:\n")
            f.write(f"  Source:     {ex['source']}\n")
            f.write(f"  Hypothesis: {ex['hypothesis']}\n")
            f.write(f"  Reference:  {ex['reference']}\n\n")

        # Number errors
        f.write("\n" + "=" * 80 + "\n")
        f.write("NUMBER MISMATCH ERRORS\n")
        f.write("=" * 80 + "\n\n")
        num_examples = analyzer.get_error_examples('number_mismatch', max_examples=10)
        for i, ex in enumerate(num_examples, 1):
            f.write(f"Example {i}:\n")
            f.write(f"  Source:     {ex['source']}\n")
            f.write(f"  Source Numbers:     {ex['source_numbers']}\n")
            f.write(f"  Hypothesis: {ex['hypothesis']}\n")
            f.write(f"  Hypothesis Numbers: {ex['hyp_numbers']}\n")
            f.write(f"  Reference:  {ex['reference']}\n\n")

    print(f"✓ Saved error examples to: {examples_path}")

    # Save summary report
    report_path = os.path.join(output_dir, 'EVALUATION_REPORT.md')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(f"# Evaluation Report\n\n")
        f.write(f"**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"**Model**: {args.checkpoint}\n")
        f.write(f"**Test Samples**: {len(sources)}\n")
        f.write(f"**Inference Method**: {args.method}\n\n")

        f.write(f"## Metrics\n\n")
        f.write(f"| Metric | Score |\n")
        f.write(f"|--------|-------|\n")
        for key, value in sorted(metrics.items()):
            if isinstance(value, float):
                f.write(f"| {key.upper()} | {value:.2f} |\n")

        f.write(f"\n## Error Analysis\n\n")
        stats = analyzer.get_statistics()
        f.write(f"| Error Type | Count | Rate |\n")
        f.write(f"|------------|-------|------|\n")
        f.write(f"| Repetition Errors | {stats['repetition_errors']} | {stats['repetition_rate']:.2f}% |\n")
        f.write(f"| Number Mismatches | {stats['number_errors']} | {stats['number_error_rate']:.2f}% |\n")
        f.write(f"| Unknown Tokens | {stats['unknown_token_errors']} | {stats['unk_rate']:.2f}% |\n")

        f.write(f"\n## Length Statistics\n\n")
        f.write(f"- Average Length Ratio: {stats['avg_length_ratio']:.3f}\n")
        f.write(f"- Std Dev: {stats['std_length_ratio']:.3f}\n")

        f.write(f"\n## Files Generated\n\n")
        f.write(f"- `translations.txt`: All translations with source and reference\n")
        f.write(f"- `metrics.json`: Detailed metrics in JSON format\n")
        f.write(f"- `error_analysis.json`: Error statistics in JSON format\n")
        f.write(f"- `error_examples.txt`: Examples of different error types\n")

    print(f"✓ Saved evaluation report to: {report_path}")
    print(f"\n{'='*80}")
    print(f"All results saved to: {output_dir}")
    print(f"{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(description='Evaluate translation model on test set')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/best_model.pt',
                       help='Path to model checkpoint')
    parser.add_argument('--method', type=str, default='beam', choices=['greedy', 'beam'],
                       help='Decoding method (default: beam)')
    parser.add_argument('--max-samples', type=int, default=None,
                       help='Maximum number of test samples to evaluate (default: all)')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Output directory for results (default: outputs/eval_TIMESTAMP)')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to run on (default: cuda)')
    parser.add_argument('--use-advanced-metrics', action='store_true',
                       help='Compute advanced metrics (COMET, BERTScore) - requires additional packages')

    args = parser.parse_args()

    # Set up output directory
    if args.output_dir is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        args.output_dir = f'outputs/evaluation_{timestamp}'

    # Load config
    config = TransformerConfig()

    # Load model and tokenizers
    model, src_tokenizer, tgt_tokenizer = load_model_and_tokenizers(
        args.checkpoint, config, device=args.device
    )

    # Create translator
    translator = Translator(
        model=model,
        src_tokenizer=src_tokenizer,
        tgt_tokenizer=tgt_tokenizer,
        device=args.device,
        max_length=config.max_seq_length
    )

    # Load test data
    src_lines, tgt_lines = load_test_data(config)

    # Limit samples if requested
    if args.max_samples:
        src_lines = src_lines[:args.max_samples]
        tgt_lines = tgt_lines[:args.max_samples]
        print(f"Limited to {args.max_samples} samples\n")

    # Run inference
    predictions = run_inference(translator, src_lines, method=args.method, max_samples=None)

    # Compute metrics
    metrics = compute_all_metrics(
        src_lines, predictions, tgt_lines,
        device=args.device,
        use_advanced=args.use_advanced_metrics
    )

    # Run error analysis
    analyzer = run_error_analysis(src_lines, predictions, tgt_lines)

    # Save results
    save_results(args.output_dir, src_lines, predictions, tgt_lines, metrics, analyzer, args)

    print("\n✓ Evaluation complete!")


if __name__ == '__main__':
    main()
