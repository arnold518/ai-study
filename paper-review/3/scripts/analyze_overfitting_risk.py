#!/usr/bin/env python
"""Analyze overfitting risk based on model size and dataset size."""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import json
from pathlib import Path
from config.transformer_config import TransformerConfig

def count_model_parameters(config):
    """Calculate total number of trainable parameters."""

    d_model = config.d_model
    d_ff = config.d_ff
    num_heads = config.num_heads
    num_enc_layers = config.num_encoder_layers
    num_dec_layers = config.num_decoder_layers
    vocab_size = config.vocab_size

    params = {}

    # 1. Embeddings
    if config.share_src_tgt_embed:
        # Shared source/target embeddings
        params['src_tgt_embeddings'] = vocab_size * d_model
    else:
        params['src_embeddings'] = vocab_size * d_model
        params['tgt_embeddings'] = vocab_size * d_model

    # Positional encoding is NOT trainable (sinusoidal)
    params['positional_encoding'] = 0

    # 2. Encoder layers
    # Each encoder layer has:
    # - Multi-head attention: Q, K, V, O projections
    # - Feed-forward: W1, W2
    # - Layer norms: gamma, beta for 2 sub-layers

    attn_params = 4 * (d_model * d_model)  # Q, K, V, O
    ffn_params = (d_model * d_ff) + (d_ff * d_model)  # W1, W2
    ln_params = 2 * (d_model * 2)  # 2 layer norms, each has gamma + beta

    encoder_layer_params = attn_params + ffn_params + ln_params
    params['encoder'] = encoder_layer_params * num_enc_layers

    # 3. Decoder layers
    # Each decoder layer has:
    # - Masked self-attention: Q, K, V, O
    # - Cross-attention: Q, K, V, O
    # - Feed-forward: W1, W2
    # - Layer norms: gamma, beta for 3 sub-layers

    self_attn_params = 4 * (d_model * d_model)
    cross_attn_params = 4 * (d_model * d_model)
    dec_ffn_params = (d_model * d_ff) + (d_ff * d_model)
    dec_ln_params = 3 * (d_model * 2)  # 3 layer norms

    decoder_layer_params = self_attn_params + cross_attn_params + dec_ffn_params + dec_ln_params
    params['decoder'] = decoder_layer_params * num_dec_layers

    # 4. Output projection
    if config.tie_embeddings:
        # Tied with target embeddings, don't count twice
        params['output_projection'] = 0
    else:
        params['output_projection'] = d_model * vocab_size

    # Total
    total_params = sum(params.values())

    return params, total_params

def load_dataset_stats():
    """Load dataset statistics."""
    stats_path = Path('data/processed/statistics.json')
    with open(stats_path, 'r') as f:
        stats = json.load(f)
    return stats

def analyze_overfitting_risk(config, dataset_stats, total_params):
    """Analyze overfitting risk factors."""

    train_size = dataset_stats['train']['num_pairs']
    val_size = dataset_stats['validation']['num_pairs']
    test_size = dataset_stats['test']['num_pairs']

    # Calculate key ratios
    params_per_sample = total_params / train_size
    samples_per_param = train_size / total_params
    val_train_ratio = val_size / train_size

    print("=" * 80)
    print("OVERFITTING RISK ANALYSIS")
    print("=" * 80)
    print()

    # 1. Dataset Size
    print("1. DATASET SIZE")
    print("-" * 80)
    print(f"  Training samples:   {train_size:,}")
    print(f"  Validation samples: {val_size:,}")
    print(f"  Test samples:       {test_size:,}")
    print(f"  Val/Train ratio:    {val_train_ratio:.4f} ({val_train_ratio*100:.2f}%)")
    print()

    # Assessment
    if val_size < 1000:
        print("  ⚠️  WARNING: Validation set very small (<1000)")
        print("      Risk: Unreliable validation metrics, high variance")
    elif val_size < 5000:
        print("  ⚠️  CAUTION: Validation set small (<5000)")
        print("      Risk: Moderate variance in validation metrics")
    else:
        print("  ✓ Validation set size adequate")
    print()

    # 2. Model Size
    print("2. MODEL SIZE")
    print("-" * 80)
    print(f"  Total parameters:   {total_params:,}")
    print(f"  d_model:            {config.d_model}")
    print(f"  d_ff:               {config.d_ff}")
    print(f"  Encoder layers:     {config.num_encoder_layers}")
    print(f"  Decoder layers:     {config.num_decoder_layers}")
    print(f"  Vocabulary size:    {config.vocab_size:,}")
    print()

    # 3. Model/Data Ratio
    print("3. MODEL/DATA RATIO")
    print("-" * 80)
    print(f"  Parameters per sample: {params_per_sample:.1f}")
    print(f"  Samples per parameter: {samples_per_param:.4f}")
    print()

    # Assessment based on research
    if params_per_sample > 100:
        print("  ❌ SEVERE RISK: Very high params/sample ratio")
        print("      Typical safe ratio: 10-50 params/sample")
        print(f"      Your ratio: {params_per_sample:.1f} params/sample")
        print("      Risk: Model will likely overfit severely")
    elif params_per_sample > 50:
        print("  ⚠️  HIGH RISK: High params/sample ratio")
        print("      Recommended ratio: 10-50 params/sample")
        print(f"      Your ratio: {params_per_sample:.1f} params/sample")
        print("      Risk: Model may overfit without strong regularization")
    elif params_per_sample > 10:
        print("  ⚠️  MODERATE RISK: Moderate params/sample ratio")
        print(f"      Your ratio: {params_per_sample:.1f} params/sample")
        print("      Recommendation: Use regularization (dropout, label smoothing)")
    else:
        print("  ✓ LOW RISK: Low params/sample ratio")
        print(f"      Your ratio: {params_per_sample:.1f} params/sample")
    print()

    # 4. Regularization
    print("4. REGULARIZATION SETTINGS")
    print("-" * 80)
    print(f"  Dropout:         {config.dropout}")
    print(f"  Label smoothing: {config.label_smoothing}")
    print()

    # Assessment
    regularization_score = 0
    if config.dropout >= 0.15:
        print(f"  ✓ Dropout adequate ({config.dropout})")
        regularization_score += 1
    else:
        print(f"  ⚠️  Dropout low ({config.dropout}), consider 0.15-0.3")

    if config.label_smoothing >= 0.1:
        print(f"  ✓ Label smoothing adequate ({config.label_smoothing})")
        regularization_score += 1
    else:
        print(f"  ⚠️  Label smoothing low ({config.label_smoothing}), consider 0.1-0.2")
    print()

    # 5. Training Configuration
    print("5. TRAINING CONFIGURATION")
    print("-" * 80)
    print(f"  Batch size:      {config.batch_size}")
    print(f"  Num epochs:      {config.num_epochs}")
    print(f"  Gradient clip:   {config.grad_clip}")
    print(f"  Eval every:      {config.eval_every} epoch(s)")
    print()

    # 6. Overall Assessment
    print("=" * 80)
    print("OVERALL OVERFITTING RISK ASSESSMENT")
    print("=" * 80)
    print()

    risk_factors = []

    # Check params/sample
    if params_per_sample > 50:
        risk_factors.append(("HIGH", "Parameters per sample too high", params_per_sample))
    elif params_per_sample > 30:
        risk_factors.append(("MODERATE", "Parameters per sample moderate", params_per_sample))

    # Check validation size
    if val_size < 1000:
        risk_factors.append(("HIGH", "Validation set too small", val_size))
    elif val_size < 5000:
        risk_factors.append(("MODERATE", "Validation set small", val_size))

    # Check regularization
    if config.dropout < 0.15 or config.label_smoothing < 0.1:
        risk_factors.append(("MODERATE", "Regularization could be stronger", config.dropout))

    if risk_factors:
        print("IDENTIFIED RISK FACTORS:")
        for severity, factor, value in risk_factors:
            print(f"  [{severity:8s}] {factor} ({value})")
        print()
    else:
        print("  ✓ No major risk factors identified")
        print()

    # Calculate overall risk
    if any(r[0] == "HIGH" for r in risk_factors):
        overall_risk = "HIGH"
    elif any(r[0] == "MODERATE" for r in risk_factors):
        overall_risk = "MODERATE"
    else:
        overall_risk = "LOW"

    print(f"OVERALL RISK LEVEL: {overall_risk}")
    print()

    return overall_risk, risk_factors

def print_overfitting_detection_methods():
    """Print methods to detect overfitting during training."""

    print("=" * 80)
    print("HOW TO DETECT OVERFITTING DURING TRAINING")
    print("=" * 80)
    print()

    print("1. MONITOR TRAINING VS VALIDATION LOSS")
    print("-" * 80)
    print("  ✓ Normal:     Both decrease together")
    print("  ⚠️  Warning:   Train loss ↓, Val loss →  (plateaus)")
    print("  ❌ Overfitting: Train loss ↓, Val loss ↑  (increases)")
    print()
    print("  Check: Plot loss curves in TensorBoard or logs")
    print("  Command: tensorboard --logdir logs/")
    print()

    print("2. MONITOR LOSS GAP")
    print("-" * 80)
    print("  Calculate: gap = val_loss - train_loss")
    print()
    print("  ✓ Normal:      gap < 0.1-0.2")
    print("  ⚠️  Warning:    gap > 0.2-0.5")
    print("  ❌ Overfitting: gap > 0.5")
    print()
    print("  Check: Compare losses at each evaluation")
    print()

    print("3. MONITOR VALIDATION METRICS (BLEU)")
    print("-" * 80)
    print("  ✓ Normal:      BLEU increases or stabilizes")
    print("  ⚠️  Warning:    BLEU plateaus early")
    print("  ❌ Overfitting: BLEU decreases after initial increase")
    print()
    print("  Check: Track BLEU scores in training logs")
    print()

    print("4. CHECK TRANSLATION QUALITY")
    print("-" * 80)
    print("  Generate translations on validation set periodically")
    print("  Compare quality over epochs")
    print()
    print("  ✓ Normal:      Translations improve or stabilize")
    print("  ❌ Overfitting: Translations become repetitive or degrade")
    print()

    print("5. EARLY STOPPING")
    print("-" * 80)
    print("  Monitor validation loss/BLEU")
    print("  Stop if no improvement for N epochs (e.g., patience=5)")
    print()
    print("  Implementation: Track best_val_loss, stop if not improving")
    print()

    print("6. VALIDATION SET VARIANCE")
    print("-" * 80)
    print("  If validation set is small (<5000), metrics will be noisy")
    print("  Check: Run multiple evaluations on same checkpoint")
    print("  If variance is high, validation set may be too small")
    print()

def print_recommendations(overall_risk, risk_factors, config, total_params, train_size):
    """Print recommendations based on risk analysis."""

    print("=" * 80)
    print("RECOMMENDATIONS TO PREVENT OVERFITTING")
    print("=" * 80)
    print()

    params_per_sample = total_params / train_size

    if overall_risk == "HIGH":
        print("⚠️  HIGH RISK: Immediate action required")
        print()

        if params_per_sample > 50:
            print("PRIORITY 1: REDUCE MODEL SIZE")
            print("-" * 80)
            print(f"  Current: {total_params:,} params ({params_per_sample:.1f} params/sample)")
            print()
            print("  Option A: Reduce d_model")
            print(f"    d_model: {config.d_model} → 256")
            print("    Expected params: ~13M (7.6 params/sample)")
            print()
            print("  Option B: Reduce layers")
            print(f"    Layers: {config.num_encoder_layers}/{config.num_decoder_layers} → 4/4")
            print("    Expected params: ~35M (20.6 params/sample)")
            print()
            print("  Option C: Reduce d_ff")
            print(f"    d_ff: {config.d_ff} → 1024")
            print("    Expected params: ~39M (22.9 params/sample)")
            print()

        print("PRIORITY 2: INCREASE REGULARIZATION")
        print("-" * 80)
        print(f"  Dropout: {config.dropout} → 0.2-0.3")
        print(f"  Label smoothing: {config.label_smoothing} → 0.1-0.15")
        print("  Add weight decay: 1e-4 to 1e-5")
        print()

    elif overall_risk == "MODERATE":
        print("⚠️  MODERATE RISK: Precautions recommended")
        print()

        print("RECOMMENDATION 1: INCREASE REGULARIZATION")
        print("-" * 80)
        if config.dropout < 0.15:
            print(f"  Dropout: {config.dropout} → 0.15-0.2")
        if config.label_smoothing < 0.1:
            print(f"  Label smoothing: {config.label_smoothing} → 0.1")
        print()

        print("RECOMMENDATION 2: EARLY STOPPING")
        print("-" * 80)
        print("  Implement patience-based early stopping")
        print("  Stop if val_loss doesn't improve for 5-10 epochs")
        print()

        print("RECOMMENDATION 3: MONITOR CLOSELY")
        print("-" * 80)
        print("  Check train/val loss gap every epoch")
        print("  Generate sample translations frequently")
        print()

    else:
        print("✓ LOW RISK: Current configuration looks good")
        print()
        print("BEST PRACTICES:")
        print("-" * 80)
        print("  1. Continue monitoring train/val metrics")
        print("  2. Use early stopping as safety net")
        print("  3. Save best model based on validation loss")
        print()

    print("GENERAL RECOMMENDATIONS:")
    print("-" * 80)
    print("  1. Implement early stopping (patience=5-10 epochs)")
    print("  2. Save checkpoint with best validation loss")
    print("  3. Monitor loss curves in TensorBoard")
    print("  4. Generate validation translations every epoch")
    print("  5. If overfitting detected, stop and reduce model size")
    print()

def main():
    config = TransformerConfig()

    # Count parameters
    param_breakdown, total_params = count_model_parameters(config)

    print("=" * 80)
    print("MODEL PARAMETER COUNT")
    print("=" * 80)
    print()
    for name, count in param_breakdown.items():
        print(f"  {name:25s}: {count:15,} ({count/total_params*100:5.1f}%)")
    print(f"  {'─' * 25}   {'─' * 15}")
    print(f"  {'TOTAL':25s}: {total_params:15,}")
    print()

    # Load dataset stats
    dataset_stats = load_dataset_stats()

    # Analyze overfitting risk
    overall_risk, risk_factors = analyze_overfitting_risk(config, dataset_stats, total_params)

    # Print detection methods
    print_overfitting_detection_methods()

    # Print recommendations
    print_recommendations(
        overall_risk,
        risk_factors,
        config,
        total_params,
        dataset_stats['train']['num_pairs']
    )

if __name__ == "__main__":
    main()
