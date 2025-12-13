# Attention Visualization Guide

This guide explains how to visualize attention weights from your trained Transformer model to understand translation alignments and debug attention mechanisms.

## Overview

The attention visualization system includes:

1. **Modified Decoder** (`src/models/transformer/decoder.py`)
   - Optionally stores attention weights during forward pass
   - Controlled via `set_store_attention(True/False)`
   - Minimal overhead when disabled (default)

2. **Visualization Utilities** (`src/utils/visualization.py`)
   - `AttentionVisualizer`: Plots attention heatmaps
   - Supports single head, multi-head, and averaged attention
   - Customizable color schemes and sizes

3. **Visualization Script** (`scripts/visualize_attention.py`)
   - Command-line tool to visualize attention from trained models
   - Supports single sentences, files, or validation examples
   - Generates publication-quality plots

## Quick Start

### Visualize Validation Examples

The simplest way to visualize attention is to use validation examples:

```bash
# Visualize 3 validation examples from best model
/home/arnold/venv/bin/python scripts/visualize_attention.py

# Visualize from specific checkpoint
/home/arnold/venv/bin/python scripts/visualize_attention.py \
    --checkpoint checkpoints/checkpoint_epoch_10.pt

# Visualize 10 examples
/home/arnold/venv/bin/python scripts/visualize_attention.py --num-examples 10
```

This will:
- Load the model and validation data
- Translate examples
- Generate attention heatmaps
- Save plots to `outputs/attention_plots/`

### Visualize Specific Sentence

```bash
# Visualize attention for a specific Korean sentence
/home/arnold/venv/bin/python scripts/visualize_attention.py \
    --input "안녕하세요. 저는 학생입니다."

# Visualize a different decoder layer
/home/arnold/venv/bin/python scripts/visualize_attention.py \
    --input "한국어를 영어로 번역합니다." \
    --layer 3

# Show plots interactively (in addition to saving)
/home/arnold/venv/bin/python scripts/visualize_attention.py \
    --input "기계 번역 시스템" \
    --show
```

### Visualize Multiple Sentences from File

```bash
# Create a file with Korean sentences
cat > examples.txt <<EOF
안녕하세요
오늘 날씨가 좋습니다
기계 번역은 흥미로운 연구 주제입니다
EOF

# Visualize all sentences
/home/arnold/venv/bin/python scripts/visualize_attention.py \
    --file examples.txt \
    --output-dir outputs/my_examples
```

## Command-Line Options

```
--checkpoint PATH          Model checkpoint (default: checkpoints/best_model.pt)
--input TEXT              Single Korean sentence to translate
--file PATH               File with Korean sentences (one per line)
--layer INDEX             Decoder layer to visualize (default: -1 for last layer)
--num-examples N          Number of validation examples (default: 3)
--output-dir DIR          Output directory (default: outputs/attention_plots)
--show                    Display plots interactively
```

## Understanding the Visualizations

The script generates two types of attention plots:

### 1. Cross-Attention (Decoder → Encoder)

**File**: `cross_attention_layer{N}.png`

Shows how each target (English) token attends to source (Korean) tokens.

**Interpretation:**
- **Rows**: Target tokens (English output)
- **Columns**: Source tokens (Korean input)
- **Bright spots**: Strong attention (target token focusing on source token)
- **Expected pattern**: Diagonal or near-diagonal for word-by-word translation

**Example:**
```
Source:  [안녕, 하, 세요]
Target:  [Hello, !]

Cross-attention should show:
- "Hello" attending strongly to "안녕"
- "!" attending to "세요"
```

### 2. Self-Attention (Decoder → Decoder)

**File**: `self_attention_layer{N}.png`

Shows how each target token attends to previous target tokens.

**Interpretation:**
- **Rows**: Current target token
- **Columns**: Previous target tokens
- **Pattern**: Lower-triangular (causal masking prevents future attention)
- **Bright spots**: Dependencies between output tokens

**Example:**
```
Target: [<s>, I, am, a, student]

Self-attention shows:
- "I" can only attend to "<s>"
- "student" can attend to all previous tokens
```

### 3. Multi-Head Attention

**File**: `cross_attention_multihead_layer{N}.png`

Shows all 8 attention heads in a grid.

**Interpretation:**
- Different heads learn different alignment patterns
- Some heads focus on local context (adjacent words)
- Some heads focus on long-range dependencies
- Some heads specialize in syntax, others in semantics

## Debugging with Attention Visualization

### Problem: Model Hallucinates

**Symptom**: Translation contains words not in source

**Diagnosis**: Check cross-attention
- If attention is diffuse (no clear peaks) → Model not learning alignment
- If attention focuses on wrong tokens → Misalignment

**Solution**:
- Increase training data
- Increase model capacity (more layers/heads)
- Check if source tokenization is correct

### Problem: Repetitive Output

**Symptom**: Model repeats same phrase

**Diagnosis**: Check self-attention
- If strong attention to repeated tokens → Copying behavior
- If no diversity across heads → Insufficient capacity

**Solution**:
- Add repetition penalty during inference
- Increase dropout
- Use diverse beam search

### Problem: Poor Long Sentences

**Symptom**: Quality degrades for long inputs

**Diagnosis**: Check cross-attention
- If attention is mostly on first/last tokens → Positional encoding issue
- If uniform attention → Model overwhelmed

**Solution**:
- Increase max_seq_length during training
- Use longer positional encodings
- Add relative position bias

## Visualizing Specific Layers

Different decoder layers learn different abstraction levels:

```bash
# Early layer (layer 0) - local patterns
/home/arnold/venv/bin/python scripts/visualize_attention.py --layer 0

# Middle layer (layer 3) - phrasal patterns
/home/arnold/venv/bin/python scripts/visualize_attention.py --layer 3

# Late layer (layer 5 or -1) - semantic patterns
/home/arnold/venv/bin/python scripts/visualize_attention.py --layer -1
```

**Expected behavior:**
- **Layer 0-1**: Sharp, local attention (word-level alignment)
- **Layer 2-4**: Broader attention (phrase-level patterns)
- **Layer 5**: Diffuse attention (high-level semantics)

## Programmatic Usage

You can also use the visualization utilities in your own code:

```python
from src.utils.visualization import AttentionVisualizer
from src.models.transformer.transformer import Transformer

# Load model
model = Transformer(...)
model.decoder.set_store_attention(True)  # Enable storage

# Forward pass
output = model(src, tgt, src_mask, tgt_mask, cross_mask)

# Get attention from last decoder layer
decoder_layer = model.decoder.layers[-1]
cross_attn = decoder_layer.cross_attn_weights  # [batch, heads, tgt_len, src_len]
self_attn = decoder_layer.self_attn_weights    # [batch, heads, tgt_len, tgt_len]

# Visualize
visualizer = AttentionVisualizer()
fig = visualizer.plot_attention_summary(
    cross_attn, src_tokens, tgt_tokens,
    layer_idx=-1, save_path='my_plot.png'
)

# Disable storage (important for training!)
model.decoder.set_store_attention(False)
```

## Tips and Best Practices

1. **Always disable attention storage during training**
   - Adds memory overhead
   - Only enable for visualization/debugging

2. **Visualize multiple examples**
   - Single examples can be misleading
   - Look for consistent patterns across examples

3. **Compare across layers**
   - Different layers learn different patterns
   - Layer progression shows model learning hierarchy

4. **Compare good vs. bad translations**
   - Attention for correct translations should show clear alignment
   - Attention for hallucinations shows diffuse/wrong patterns

5. **Use high-resolution plots for papers**
   - Default DPI is 150
   - Increase in code: `plt.savefig(..., dpi=300)`

## Output Files

The script generates these files in the output directory:

```
outputs/attention_plots/
├── cross_attention_layer5.png              # Average cross-attention
├── cross_attention_multihead_layer5.png    # All 8 heads
└── self_attention_layer5.png               # Average self-attention
```

## Troubleshooting

### Issue: "Attention weights not found"

**Cause**: `set_store_attention(True)` not called before forward pass

**Solution**: Ensure you call `model.decoder.set_store_attention(True)` before translation

### Issue: Empty/black plots

**Cause**: Attention is all zeros (likely masking issue)

**Solution**: Check that masks are correct shape and not all False

### Issue: Out of memory

**Cause**: Storing attention for very long sequences

**Solution**:
- Reduce `max_seq_length`
- Visualize shorter examples
- Disable attention storage after each example

### Issue: Plots look random

**Cause**: Model not trained or poorly trained

**Solution**:
- Check that model checkpoint is valid
- Verify model loss is reasonable
- Try visualizing from a better checkpoint

## Examples Gallery

See `docs/examples/` for example attention visualizations:

- `good_alignment.png` - Clean word-by-word alignment
- `phrase_alignment.png` - Phrasal translation patterns
- `hallucination.png` - Diffuse attention leading to hallucination
- `repetition.png` - Self-attention causing repetitive output

## References

- Vaswani et al. (2017) "Attention Is All You Need"
- Vig & Belinkov (2019) "Analyzing Multi-Head Self-Attention"
- Clark et al. (2019) "What Does BERT Look At?"
