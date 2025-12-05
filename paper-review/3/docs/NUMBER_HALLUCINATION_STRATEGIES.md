# Strategies to Resolve Number Hallucinations in Neural Machine Translation

**Date:** 2025-12-05
**Problem:** Neural MT models often incorrectly translate numbers (dates, quantities, phone numbers, etc.)

---

## Table of Contents
1. [Problem Overview](#1-problem-overview)
2. [Tokenization-based Strategies](#2-tokenization-based-strategies)
3. [Copy Mechanism & Pointer Networks](#3-copy-mechanism--pointer-networks)
4. [Data Augmentation](#4-data-augmentation)
5. [Model Architecture Modifications](#5-model-architecture-modifications)
6. [Post-processing & Constrained Decoding](#6-post-processing--constrained-decoding)
7. [Training Objectives](#7-training-objectives)
8. [Hybrid & External Tools](#8-hybrid--external-tools)
9. [Implementation Recommendations](#9-implementation-recommendations)
10. [Research Papers](#10-research-papers)

---

## 1. Problem Overview

### Common Number Hallucination Errors

| Error Type | Example | Impact |
|------------|---------|--------|
| **Digit Change** | "2025" → "2024" | Critical for dates |
| **Digit Drop** | "1,234" → "1,23" | Critical for quantities |
| **Digit Add** | "100" → "1000" | Critical for measurements |
| **Complete Hallucination** | "2025" → "1987" | Completely wrong |
| **Format Change** | "12/05/2025" → "05.12.2025" | May be acceptable |

### Why Numbers Are Hard for NMT

1. **Rare in Training Data**: Specific numbers appear infrequently
2. **High Cardinality**: Infinite possible numbers
3. **Subword Splitting**: BPE/SentencePiece splits numbers ("2025" → "20", "25")
4. **Context-dependent**: Sometimes translate (units), sometimes copy (years)
5. **Attention Errors**: Model may attend to wrong position

---

## 2. Tokenization-based Strategies

### Strategy 2.1: Number Placeholder / Masking ⭐ EASIEST

**Concept:** Replace numbers with special tokens, copy them directly during inference.

**Implementation:**
```python
import re

def mask_numbers(text):
    """Replace numbers with placeholders."""
    numbers = []
    def replacer(match):
        numbers.append(match.group())
        return f"<NUM_{len(numbers)-1}>"

    masked = re.sub(r'\b\d+(?:[.,]\d+)*\b', replacer, text)
    return masked, numbers

def unmask_numbers(text, numbers):
    """Restore numbers from placeholders."""
    for i, num in enumerate(numbers):
        text = text.replace(f"<NUM_{i}>", num)
    return text

# Training
src = "I have 123 apples and 456 oranges"
src_masked, src_nums = mask_numbers(src)
# → "I have <NUM_0> apples and <NUM_1> oranges", ["123", "456"]

# Inference
tgt_masked = model.translate(src_masked)
# → "나는 <NUM_0> 사과와 <NUM_1> 오렌지를 가지고 있다"
tgt = unmask_numbers(tgt_masked, src_nums)
# → "나는 123 사과와 456 오렌지를 가지고 있다"
```

**Pros:**
- ✅ Simple to implement
- ✅ 100% number accuracy (if placeholders preserved)
- ✅ No model architecture changes

**Cons:**
- ❌ Assumes 1-to-1 number correspondence
- ❌ Can't handle number reordering
- ❌ Fails if placeholder count mismatches

**Best for:** Dates, IDs, phone numbers, quantities that shouldn't change

---

### Strategy 2.2: Subword-aware Number Tokenization

**Concept:** Modify tokenizer to keep numbers as single tokens.

**Implementation:**
```python
from sentencepiece import SentencePieceTrainer

# Train with user-defined symbols
SentencePieceTrainer.train(
    input='corpus.txt',
    model_prefix='spm',
    vocab_size=16000,
    user_defined_symbols=['<NUM>'],  # Preserve numbers
    # Pre-tokenize numbers before training
)

# Pre-tokenization function
def pre_tokenize(text):
    """Replace numbers with <NUM> before tokenization."""
    return re.sub(r'\b\d+(?:[.,]\d+)*\b', '<NUM>', text)

# Post-tokenization
def restore_numbers(tokens, numbers):
    """Map <NUM> back to actual numbers."""
    result = []
    num_idx = 0
    for token in tokens:
        if token == '<NUM>':
            result.append(numbers[num_idx])
            num_idx += 1
        else:
            result.append(token)
    return result
```

**Pros:**
- ✅ Numbers treated as atomic units
- ✅ No subword splitting of numbers
- ✅ Model learns number as single concept

**Cons:**
- ❌ Requires retraining tokenizer
- ❌ Still doesn't guarantee correct copying

**Best for:** When retraining from scratch

---

### Strategy 2.3: Hybrid Tokenization (Word for Numbers, Subword for Text)

**Concept:** Use character-level or word-level specifically for numbers.

**Implementation:**
```python
class HybridTokenizer:
    def __init__(self, text_tokenizer):
        self.text_tokenizer = text_tokenizer
        self.number_pattern = re.compile(r'\b\d+(?:[.,]\d+)*\b')

    def encode(self, text):
        tokens = []
        last_end = 0

        for match in self.number_pattern.finditer(text):
            # Tokenize text before number
            if match.start() > last_end:
                text_part = text[last_end:match.start()]
                tokens.extend(self.text_tokenizer.encode(text_part))

            # Keep number as single token OR character-level
            number = match.group()
            tokens.append(f"<NUM:{number}>")  # Single token
            # OR: tokens.extend([f"<DIGIT:{d}>" for d in number])  # Char-level

            last_end = match.end()

        # Remaining text
        if last_end < len(text):
            tokens.extend(self.text_tokenizer.encode(text[last_end:]))

        return tokens
```

**Pros:**
- ✅ Best of both worlds
- ✅ Numbers represented optimally
- ✅ No subword splitting issues

**Cons:**
- ❌ Complex implementation
- ❌ Requires custom data pipeline

---

### Strategy 2.4: Number-specific Vocabulary

**Concept:** Add frequent numbers (0-9999, common years) to vocabulary.

**Implementation:**
```python
# When training tokenizer
common_numbers = [str(i) for i in range(10000)]
common_years = [str(y) for y in range(1900, 2100)]
date_formats = ["01", "02", ..., "31"]  # Days

user_defined_symbols = common_numbers + common_years + date_formats

SentencePieceTrainer.train(
    input='corpus.txt',
    model_prefix='spm',
    vocab_size=16000,
    user_defined_symbols=user_defined_symbols
)
```

**Pros:**
- ✅ Common numbers always preserved
- ✅ Simple to implement during tokenizer training

**Cons:**
- ❌ Only works for predefined numbers
- ❌ Inflates vocabulary size

**Best for:** Domain-specific (e.g., years 2020-2025 for news)

---

## 3. Copy Mechanism & Pointer Networks

### Strategy 3.1: Pointer-Generator Networks ⭐ RECOMMENDED

**Concept:** Hybrid model that can generate from vocabulary OR copy from source.

**Architecture:**
```
At each decoding step:
1. Generate probability distribution P_vocab (standard decoder)
2. Copy probability distribution P_copy (attention over source)
3. Generation gate: p_gen = sigmoid(W_h * h_t + W_s * s_t + W_x * x_t)
4. Final distribution: P = p_gen * P_vocab + (1 - p_gen) * P_copy
```

**Implementation (Simplified):**
```python
class PointerGeneratorDecoder(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.vocab_proj = nn.Linear(d_model, vocab_size)
        self.copy_gate = nn.Linear(d_model * 3, 1)  # h_t, s_t, x_t

    def forward(self, decoder_hidden, encoder_outputs, attention_weights, context):
        # Standard generation
        vocab_logits = self.vocab_proj(decoder_hidden)  # [batch, vocab]
        p_vocab = F.softmax(vocab_logits, dim=-1)

        # Copy distribution (from attention)
        p_copy = attention_weights  # [batch, src_len]

        # Generation gate
        gate_input = torch.cat([decoder_hidden, context, decoder_hidden], dim=-1)
        p_gen = torch.sigmoid(self.copy_gate(gate_input))  # [batch, 1]

        # Final distribution
        # Need to map source positions to vocab indices
        p_final = p_gen * p_vocab  # Generation part

        # Add copy probabilities to corresponding vocab positions
        for batch_idx in range(p_copy.size(0)):
            for src_pos in range(p_copy.size(1)):
                src_token_id = source_ids[batch_idx, src_pos]
                p_final[batch_idx, src_token_id] += (1 - p_gen[batch_idx]) * p_copy[batch_idx, src_pos]

        return p_final
```

**Pros:**
- ✅ Learns when to copy vs generate
- ✅ Handles number reordering
- ✅ Works for all rare tokens (not just numbers)
- ✅ Proven in summarization tasks

**Cons:**
- ❌ Requires architecture modification
- ❌ Slower inference (additional computation)
- ❌ Harder to train (additional parameters)

**Research:**
- "Get To The Point: Summarization with Pointer-Generator Networks" (See et al., 2017)
- Applied successfully to NMT in follow-up work

---

### Strategy 3.2: Hard Attention Copy

**Concept:** Use attention to identify numbers, copy them deterministically.

**Implementation:**
```python
def copy_numbers_via_attention(src_tokens, tgt_tokens, attention_weights):
    """Copy numbers from source using attention alignment."""

    # Identify number positions in source
    src_num_positions = [i for i, tok in enumerate(src_tokens) if is_number(tok)]

    result = []
    for tgt_idx, tgt_token in enumerate(tgt_tokens):
        if is_number_placeholder(tgt_token):
            # Find source position with highest attention
            attn = attention_weights[tgt_idx]

            # Prefer source number positions
            max_attn_idx = None
            max_attn_val = -1
            for src_idx in src_num_positions:
                if attn[src_idx] > max_attn_val:
                    max_attn_val = attn[src_idx]
                    max_attn_idx = src_idx

            if max_attn_idx is not None:
                result.append(src_tokens[max_attn_idx])
            else:
                result.append(tgt_token)  # Fallback
        else:
            result.append(tgt_token)

    return result
```

**Pros:**
- ✅ Deterministic copying
- ✅ No training changes needed
- ✅ Post-processing only

**Cons:**
- ❌ Requires access to attention weights
- ❌ Assumes good attention alignment
- ❌ May fail with poor attention

---

### Strategy 3.3: Soft Copy with Coverage Mechanism

**Concept:** Track which source tokens have been copied to avoid over/under-copying.

**Implementation:**
```python
class CoverageAttention(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.coverage_proj = nn.Linear(1, d_model)
        self.attn = MultiHeadAttention(d_model, num_heads=8)

    def forward(self, query, keys, values, coverage):
        """
        coverage: [batch, src_len] - sum of previous attention weights
        """
        # Project coverage to d_model
        coverage_feat = self.coverage_proj(coverage.unsqueeze(-1))

        # Modify keys with coverage
        keys_modified = keys + coverage_feat

        # Standard attention
        output, attn_weights = self.attn(query, keys_modified, values)

        # Update coverage
        new_coverage = coverage + attn_weights.mean(dim=1)  # Avg over heads

        return output, attn_weights, new_coverage
```

**Pros:**
- ✅ Prevents over-copying same numbers
- ✅ Ensures all numbers are copied

**Cons:**
- ❌ Complex training
- ❌ Requires architecture modification

---

## 4. Data Augmentation

### Strategy 4.1: Number Substitution Augmentation ⭐ EASY & EFFECTIVE

**Concept:** Replace numbers in parallel data with random numbers during training.

**Implementation:**
```python
import random

def augment_numbers(src, tgt):
    """Randomly replace numbers in parallel sentences."""

    # Extract numbers
    src_nums = re.findall(r'\b\d+(?:[.,]\d+)*\b', src)
    tgt_nums = re.findall(r'\b\d+(?:[.,]\d+)*\b', tgt)

    # Check if numbers match (assume parallel)
    if src_nums == tgt_nums:
        # Create mapping
        num_map = {}
        for num in set(src_nums):
            # Generate random replacement (same format)
            if '.' in num:
                # Decimal number
                parts = num.split('.')
                new_int = str(random.randint(1, 9999))
                new_dec = ''.join(str(random.randint(0, 9)) for _ in parts[1])
                num_map[num] = f"{new_int}.{new_dec}"
            elif ',' in num:
                # Formatted number
                clean = num.replace(',', '')
                new_num = str(random.randint(1, int(clean) * 10))
                # Add commas back
                num_map[num] = f"{int(new_num):,}"
            else:
                # Plain integer
                num_map[num] = str(random.randint(1, 9999))

        # Replace in both sentences
        for old, new in num_map.items():
            src = src.replace(old, new)
            tgt = tgt.replace(old, new)

    return src, tgt

# During training
for src, tgt in dataset:
    # 50% of time, augment numbers
    if random.random() < 0.5:
        src, tgt = augment_numbers(src, tgt)

    train_batch.append((src, tgt))
```

**Pros:**
- ✅ Forces model to learn number copying
- ✅ Easy to implement
- ✅ No architecture changes
- ✅ Increases data diversity

**Cons:**
- ❌ May hurt if numbers should be translated (e.g., "one" ↔ "1")

**Effectiveness:** Can reduce number errors by 30-50% (empirical)

---

### Strategy 4.2: Synthetic Number Data Generation

**Concept:** Create synthetic parallel sentences with various number formats.

**Implementation:**
```python
templates = [
    ("The meeting is on {date}.", "회의는 {date}에 있습니다."),
    ("I have {num} apples.", "나는 {num}개의 사과를 가지고 있습니다."),
    ("Call me at {phone}.", "{phone}로 전화주세요."),
    ("The price is ${price}.", "가격은 ${price}입니다."),
]

def generate_synthetic_data(n_samples=10000):
    synthetic_pairs = []

    for _ in range(n_samples):
        en_template, ko_template = random.choice(templates)

        # Generate random values
        values = {
            'date': f"{random.randint(1,12)}/{random.randint(1,28)}/{random.randint(2020,2030)}",
            'num': str(random.randint(1, 1000)),
            'phone': f"{random.randint(100,999)}-{random.randint(1000,9999)}-{random.randint(1000,9999)}",
            'price': f"{random.randint(1,1000)}.{random.randint(0,99):02d}",
        }

        en = en_template.format(**values)
        ko = ko_template.format(**values)

        synthetic_pairs.append((en, ko))

    return synthetic_pairs
```

**Pros:**
- ✅ Unlimited data for number patterns
- ✅ Covers various number formats
- ✅ Easy to generate

**Cons:**
- ❌ Synthetic data may not match real distribution
- ❌ Limited template diversity

---

### Strategy 4.3: Back-translation with Number Constraints

**Concept:** During back-translation, ensure numbers are preserved.

**Implementation:**
```python
def constrained_back_translation(monolingual_text):
    """Back-translate while preserving numbers."""

    # Extract numbers
    numbers = extract_numbers(monolingual_text)

    # Translate to source language
    back_translated = model.translate(monolingual_text)

    # Check if numbers preserved
    back_numbers = extract_numbers(back_translated)

    if set(numbers) != set(back_numbers):
        # Force number alignment
        back_translated = force_number_alignment(back_translated, numbers)

    return back_translated
```

---

## 5. Model Architecture Modifications

### Strategy 5.1: Number-aware Embeddings

**Concept:** Special embeddings for numeric tokens.

**Implementation:**
```python
class NumberAwareEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.number_embedding = nn.Embedding(10, d_model)  # 0-9 digits
        self.is_number_flag = nn.Embedding(2, d_model)  # number or not

    def forward(self, tokens):
        # Regular embedding
        emb = self.token_embedding(tokens)

        # Add number-specific features
        for i, token in enumerate(tokens):
            if is_number(token):
                # Add digit embeddings
                digits = [int(d) for d in str(token) if d.isdigit()]
                digit_emb = self.number_embedding(torch.tensor(digits)).mean(dim=0)
                emb[i] += digit_emb

                # Add "is number" flag
                emb[i] += self.is_number_flag(torch.tensor(1))
            else:
                emb[i] += self.is_number_flag(torch.tensor(0))

        return emb
```

**Pros:**
- ✅ Model learns number-specific features
- ✅ Can distinguish numbers from words

**Cons:**
- ❌ Requires architecture modification
- ❌ May not significantly improve copying

---

### Strategy 5.2: Copy Gate in Decoder

**Concept:** Explicit gate to decide copy vs generate for each token.

**Implementation:**
```python
class CopyGateDecoder(nn.Module):
    def __init__(self, d_model, vocab_size):
        super().__init__()
        self.decoder = TransformerDecoder(d_model)
        self.vocab_proj = nn.Linear(d_model, vocab_size)
        self.copy_gate = nn.Linear(d_model, 1)

    def forward(self, tgt, encoder_output, src_tokens):
        # Standard decoding
        decoder_out = self.decoder(tgt, encoder_output)

        # Vocabulary distribution
        vocab_logits = self.vocab_proj(decoder_out)

        # Copy gate
        should_copy = torch.sigmoid(self.copy_gate(decoder_out))

        # If copying, use attention to select source token
        attn_weights = self.decoder.get_attention_weights()

        # Mixed distribution
        final_dist = should_copy * copy_distribution(attn_weights, src_tokens) + \
                     (1 - should_copy) * F.softmax(vocab_logits, dim=-1)

        return final_dist
```

---

## 6. Post-processing & Constrained Decoding

### Strategy 6.1: Number Alignment Post-processing ⭐ PRACTICAL

**Concept:** After generation, align and replace numbers from source.

**Implementation:**
```python
def post_process_numbers(src, tgt):
    """Replace hallucinated numbers with source numbers."""

    # Extract numbers
    src_numbers = extract_numbers_with_positions(src)
    tgt_numbers = extract_numbers_with_positions(tgt)

    # If count matches, replace 1-to-1
    if len(src_numbers) == len(tgt_numbers):
        result = tgt
        for (src_num, _), (tgt_num, tgt_pos) in zip(src_numbers, tgt_numbers):
            if src_num != tgt_num:
                # Replace
                result = result.replace(tgt_num, src_num, 1)
        return result

    # If count doesn't match, use heuristics
    # (e.g., replace numbers that don't exist in source)
    return tgt

def extract_numbers_with_positions(text):
    """Extract numbers and their positions."""
    pattern = r'\b\d+(?:[.,]\d+)*\b'
    return [(m.group(), m.start()) for m in re.finditer(pattern, text)]
```

**Pros:**
- ✅ Simple to implement
- ✅ No model changes
- ✅ Can fix most errors

**Cons:**
- ❌ Assumes number count matches
- ❌ Can't handle number transformations (e.g., "1000" → "one thousand")

---

### Strategy 6.2: Constrained Beam Search

**Concept:** During beam search, penalize beams that hallucinate numbers.

**Implementation:**
```python
class ConstrainedBeamSearch:
    def __init__(self, model, src_numbers):
        self.model = model
        self.src_numbers = set(src_numbers)
        self.hallucination_penalty = 0.5  # Log probability penalty

    def search(self, src, beam_size=4):
        # Standard beam search
        beams = self.model.beam_search(src, beam_size)

        # Re-score beams
        for beam in beams:
            # Check for hallucinated numbers
            tgt_numbers = extract_numbers(beam.sequence)
            hallucinated = [n for n in tgt_numbers if n not in self.src_numbers]

            # Apply penalty
            penalty = len(hallucinated) * self.hallucination_penalty
            beam.score -= penalty

        # Re-rank
        beams.sort(key=lambda b: b.score, reverse=True)
        return beams[0]
```

**Pros:**
- ✅ Steers generation toward correct numbers
- ✅ Works during inference

**Cons:**
- ❌ May be too restrictive
- ❌ Can't handle legitimate number changes

---

### Strategy 6.3: Forced Number Copy (Rule-based)

**Concept:** During decoding, force model to copy numbers at specific positions.

**Implementation:**
```python
def forced_copy_decode(model, src, src_numbers):
    """Force decoder to copy numbers from source."""

    # Track which numbers have been used
    available_numbers = src_numbers.copy()

    # Decode token by token
    decoded = []
    for step in range(max_length):
        # Get next token distribution
        logits = model.decode_step(decoded)

        # Check if next token should be a number
        if should_be_number(decoded, logits):
            # Force copy from available numbers
            if available_numbers:
                next_token = available_numbers.pop(0)
            else:
                next_token = sample_from_logits(logits)
        else:
            next_token = sample_from_logits(logits)

        decoded.append(next_token)

    return decoded
```

---

## 7. Training Objectives

### Strategy 7.1: Number-specific Loss Weighting

**Concept:** Increase loss weight for number tokens.

**Implementation:**
```python
class NumberWeightedLoss(nn.Module):
    def __init__(self, number_weight=3.0):
        super().__init__()
        self.number_weight = number_weight
        self.base_loss = nn.CrossEntropyLoss(reduction='none')

    def forward(self, logits, targets, is_number_mask):
        """
        is_number_mask: [batch, seq_len] - 1 if token is number, 0 otherwise
        """
        # Compute base loss
        loss = self.base_loss(logits.view(-1, logits.size(-1)), targets.view(-1))
        loss = loss.view(targets.size())

        # Apply higher weight to numbers
        weights = torch.where(is_number_mask,
                             torch.tensor(self.number_weight),
                             torch.tensor(1.0))

        weighted_loss = loss * weights
        return weighted_loss.mean()
```

**Pros:**
- ✅ Model focuses more on numbers
- ✅ Easy to implement

**Cons:**
- ❌ May hurt overall translation quality
- ❌ Requires identifying number tokens

---

### Strategy 7.2: Auxiliary Number Classification Task

**Concept:** Multi-task learning - predict which numbers appear in source.

**Implementation:**
```python
class MultiTaskModel(nn.Module):
    def __init__(self, transformer, d_model, num_classes=10):
        super().__init__()
        self.transformer = transformer
        self.number_classifier = nn.Linear(d_model, num_classes)

    def forward(self, src, tgt):
        # Main translation task
        output = self.transformer(src, tgt)

        # Auxiliary task: classify numbers in source
        encoder_output = self.transformer.encoder(src)
        number_logits = self.number_classifier(encoder_output.mean(dim=1))

        return output, number_logits

    def compute_loss(self, output, tgt, number_logits, number_labels):
        # Main loss
        main_loss = F.cross_entropy(output, tgt)

        # Auxiliary loss
        aux_loss = F.binary_cross_entropy_with_logits(number_logits, number_labels)

        # Combined
        total_loss = main_loss + 0.1 * aux_loss
        return total_loss
```

---

### Strategy 7.3: Number Reconstruction Loss

**Concept:** Add loss to reconstruct source numbers.

**Implementation:**
```python
def number_reconstruction_loss(encoder_output, src_numbers, number_decoder):
    """Additional loss to reconstruct source numbers."""

    # Extract features for numbers
    number_features = extract_number_features(encoder_output, src_numbers)

    # Decode numbers
    reconstructed = number_decoder(number_features)

    # Compute reconstruction loss
    loss = F.mse_loss(reconstructed, src_numbers)

    return loss
```

---

## 8. Hybrid & External Tools

### Strategy 8.1: Named Entity Recognition (NER) Pre/Post-processing

**Concept:** Use NER to tag numbers, handle separately.

**Implementation:**
```python
from transformers import pipeline

ner = pipeline("ner", model="dslim/bert-base-NER")

def ner_based_number_handling(src, model):
    """Use NER to identify and preserve numbers."""

    # Tag entities
    entities = ner(src)

    # Extract numbers
    numbers = []
    for entity in entities:
        if entity['entity'] == 'NUMBER' or is_number(entity['word']):
            numbers.append(entity['word'])

    # Translate
    tgt = model.translate(src)

    # Verify numbers present
    for num in numbers:
        if num not in tgt:
            # Insert at appropriate position (use alignment)
            tgt = insert_missing_number(tgt, num)

    return tgt
```

---

### Strategy 8.2: Rule-based Number Extraction & Insertion

**Concept:** Extract numbers before translation, insert after.

**Implementation:**
```python
def rule_based_number_preservation(src, model):
    """Extract numbers, translate, then insert."""

    # Extract numbers and positions
    numbers = []
    positions = []

    def replacer(match):
        numbers.append(match.group())
        positions.append(match.start())
        return "<NUM>"

    src_clean = re.sub(r'\b\d+(?:[.,]\d+)*\b', replacer, src)

    # Translate
    tgt_clean = model.translate(src_clean)

    # Insert numbers back
    tgt = tgt_clean
    for num in numbers:
        tgt = tgt.replace("<NUM>", num, 1)

    return tgt
```

**Pros:**
- ✅ 100% number preservation
- ✅ Simple and reliable

**Cons:**
- ❌ Assumes placeholder preservation
- ❌ Can't handle reordering

---

### Strategy 8.3: Alignment-based Copy (Using Word Aligner)

**Concept:** Use separate alignment model to copy numbers.

**Implementation:**
```python
from fast_align import align

def alignment_based_copy(src, tgt_draft):
    """Use word alignment to copy numbers correctly."""

    # Get alignment
    alignments = align(src, tgt_draft)  # [(src_idx, tgt_idx), ...]

    # Extract numbers from source
    src_tokens = src.split()
    tgt_tokens = tgt_draft.split()

    src_numbers = {i: tok for i, tok in enumerate(src_tokens) if is_number(tok)}

    # Copy aligned numbers
    for src_idx, tgt_idx in alignments:
        if src_idx in src_numbers:
            tgt_tokens[tgt_idx] = src_numbers[src_idx]

    return ' '.join(tgt_tokens)
```

---

## 9. Implementation Recommendations

### Priority 1 (Easy & Effective): ⭐

1. **Number Placeholder / Masking** (Strategy 2.1)
   - Easiest to implement
   - Works for most cases
   - Start here

2. **Number Substitution Augmentation** (Strategy 4.1)
   - Simple data augmentation
   - Trains model to preserve numbers
   - Can reduce errors by 30-50%

3. **Post-processing Alignment** (Strategy 6.1)
   - No model changes
   - Fixes most hallucinations
   - Can combine with other strategies

### Priority 2 (Medium Effort, High Impact): ⭐⭐

4. **Pointer-Generator Network** (Strategy 3.1)
   - Best accuracy
   - Handles complex cases
   - Requires architecture modification

5. **Constrained Beam Search** (Strategy 6.2)
   - Inference-time solution
   - Penalizes hallucinations
   - Good middle ground

6. **Subword-aware Tokenization** (Strategy 2.2)
   - Prevents number splitting
   - Requires retraining tokenizer
   - Worth it for new projects

### Priority 3 (Advanced / Research): ⭐⭐⭐

7. **Multi-task Learning** (Strategy 7.2)
   - Additional supervision signal
   - Complex training

8. **Coverage Mechanism** (Strategy 3.3)
   - Prevents over/under-copying
   - Research-level implementation

### Recommended Combination

**For Production:**
```python
# 1. Data augmentation (training time)
augmented_data = apply_number_substitution(training_data)

# 2. Subword-aware tokenization (if retraining)
tokenizer = train_tokenizer_with_number_preservation()

# 3. Post-processing (inference time)
def translate_with_number_preservation(src):
    src_numbers = extract_numbers(src)

    # Option A: Placeholder
    src_masked, numbers = mask_numbers(src)
    tgt_masked = model.translate(src_masked)
    tgt = unmask_numbers(tgt_masked, numbers)

    # Option B: Post-process alignment
    tgt = model.translate(src)
    tgt = align_and_copy_numbers(src, tgt, src_numbers)

    return tgt
```

---

## 10. Research Papers

### Key Papers

1. **Pointer-Generator Networks:**
   - "Get To The Point: Summarization with Pointer-Generator Networks" (See et al., ACL 2017)
   - https://arxiv.org/abs/1704.04368

2. **Copy Mechanism in NMT:**
   - "Incorporating Copying Mechanism in Sequence-to-Sequence Learning" (Gu et al., ACL 2016)
   - https://arxiv.org/abs/1603.06393

3. **Coverage Mechanism:**
   - "Modeling Coverage for Neural Machine Translation" (Tu et al., ACL 2016)
   - https://arxiv.org/abs/1601.04811

4. **Number Translation in NMT:**
   - "Handling Rare Words in Neural Machine Translation" (Luong et al., ACL 2015)
   - "Numeral Normalization for Neural Machine Translation" (Bansal et al., 2019)

5. **Data Augmentation:**
   - "Data Augmentation for Low-Resource Neural Machine Translation" (Fadaee et al., ACL 2017)

### Recent Advances

- **Constrained Decoding:** "Lexically Constrained Neural Machine Translation with Levenshtein Transformer" (EMNLP 2020)
- **Entity-aware NMT:** "Named Entity Aware Neural Machine Translation" (NAACL 2021)

---

## Summary

| Strategy | Difficulty | Effectiveness | When to Use |
|----------|-----------|---------------|-------------|
| **Number Placeholder** | ⭐ Easy | ⭐⭐⭐ High | Dates, IDs, quantities |
| **Subword-aware Tokenization** | ⭐⭐ Medium | ⭐⭐⭐ High | New projects |
| **Data Augmentation** | ⭐ Easy | ⭐⭐⭐ High | Always |
| **Pointer-Generator** | ⭐⭐⭐ Hard | ⭐⭐⭐⭐ Very High | Best accuracy needed |
| **Post-processing** | ⭐ Easy | ⭐⭐ Medium | Quick fix |
| **Constrained Decoding** | ⭐⭐ Medium | ⭐⭐⭐ High | Inference-time control |
| **Number-weighted Loss** | ⭐⭐ Medium | ⭐⭐ Medium | Emphasis on numbers |

**Best Starting Point:** Combine **Number Placeholder** (2.1) + **Data Augmentation** (4.1) + **Post-processing** (6.1)

**Ultimate Solution:** **Pointer-Generator Network** (3.1) with **Subword-aware Tokenization** (2.2)

---

**Next Steps:**
1. Choose strategy based on your constraints (time, resources, accuracy needs)
2. Implement easiest strategies first (placeholder + augmentation)
3. Evaluate on test set
4. If not satisfactory, move to pointer-generator network
