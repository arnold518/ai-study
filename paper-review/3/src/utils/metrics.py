"""Evaluation metrics for translation quality.

Includes BLEU, chrF++, COMET, and BERTScore.
Optional metrics (COMET, BERTScore) require additional packages:
    pip install unbabel-comet bert-score
"""

import sacrebleu

# Optional imports for advanced metrics
try:
    from comet import download_model, load_from_checkpoint
    COMET_AVAILABLE = True
except ImportError:
    COMET_AVAILABLE = False

try:
    from bert_score import score as bert_score
    BERTSCORE_AVAILABLE = True
except ImportError:
    BERTSCORE_AVAILABLE = False


def compute_bleu(predictions, references):
    """
    Compute BLEU score using SacreBLEU.

    Args:
        predictions: List of predicted sentences
        references: List of reference sentences (or list of lists for multiple references)

    Returns:
        bleu_score: BLEU score object
    """
    # Ensure references is list of lists
    if isinstance(references[0], str):
        references = [[ref] for ref in references]

    bleu = sacrebleu.corpus_bleu(predictions, references)
    return bleu


def compute_chrf(predictions, references):
    """
    Compute chrF++ score (character-level F-score).

    chrF++ has better correlation with human judgment than BLEU,
    especially for morphologically rich languages.

    Args:
        predictions: List of predicted sentences
        references: List of reference sentences

    Returns:
        chrf_score: chrF score object
    """
    # Ensure references is list of lists
    if isinstance(references[0], str):
        references = [[ref] for ref in references]

    chrf = sacrebleu.corpus_chrf(predictions, references)
    return chrf


def compute_comet(sources, predictions, references, model_name='wmt20-comet-da', gpus=0):
    """
    Compute COMET score (neural metric based on cross-lingual representations).

    COMET has state-of-the-art correlation with human judgments.
    Requires: pip install unbabel-comet

    Args:
        sources: List of source sentences
        predictions: List of predicted sentences
        references: List of reference sentences
        model_name: COMET model to use (default: wmt20-comet-da)
        gpus: Number of GPUs to use (0 for CPU)

    Returns:
        comet_score: Float score (0-1, higher is better)
    """
    if not COMET_AVAILABLE:
        print("Warning: COMET not available. Install with: pip install unbabel-comet")
        return None

    try:
        # Download and load model
        model_path = download_model(model_name)
        model = load_from_checkpoint(model_path)

        # Prepare data in COMET format
        data = []
        for src, pred, ref in zip(sources, predictions, references):
            data.append({'src': src, 'mt': pred, 'ref': ref})

        # Compute scores
        output = model.predict(data, batch_size=8, gpus=gpus)

        # Return mean score
        return output.system_score

    except Exception as e:
        print(f"Error computing COMET: {e}")
        return None


def compute_bertscore(predictions, references, lang='en', device='cuda'):
    """
    Compute BERTScore (contextual embedding similarity).

    Measures semantic similarity using BERT embeddings.
    Requires: pip install bert-score

    Args:
        predictions: List of predicted sentences
        references: List of reference sentences
        lang: Language code (default: 'en')
        device: Device to run on ('cuda' or 'cpu')

    Returns:
        bertscore: Dictionary with precision, recall, f1 scores
    """
    if not BERTSCORE_AVAILABLE:
        print("Warning: BERTScore not available. Install with: pip install bert-score")
        return None

    try:
        # Compute BERTScore
        P, R, F1 = bert_score(predictions, references, lang=lang, device=device, verbose=False)

        # Return mean scores
        return {
            'precision': P.mean().item(),
            'recall': R.mean().item(),
            'f1': F1.mean().item()
        }

    except Exception as e:
        print(f"Error computing BERTScore: {e}")
        return None


def compute_metrics(predictions, references, sources=None, use_advanced=False, device='cuda'):
    """
    Compute multiple evaluation metrics.

    Args:
        predictions: List of predicted sentences
        references: List of reference sentences
        sources: List of source sentences (required for COMET)
        use_advanced: Whether to compute advanced metrics (COMET, BERTScore)
        device: Device for neural metrics

    Returns:
        metrics: Dictionary of metric scores
    """
    # Basic metrics (always computed)
    bleu = compute_bleu(predictions, references)
    chrf = compute_chrf(predictions, references)

    metrics = {
        'bleu': bleu.score,
        'bleu_1': bleu.precisions[0],
        'bleu_2': bleu.precisions[1],
        'bleu_3': bleu.precisions[2],
        'bleu_4': bleu.precisions[3],
        'chrf': chrf.score,
    }

    # Advanced metrics (optional)
    if use_advanced:
        # COMET (requires sources)
        if sources and COMET_AVAILABLE:
            comet_score = compute_comet(sources, predictions, references, gpus=1 if device == 'cuda' else 0)
            if comet_score is not None:
                metrics['comet'] = comet_score

        # BERTScore
        if BERTSCORE_AVAILABLE:
            bertscore = compute_bertscore(predictions, references, lang='en', device=device)
            if bertscore is not None:
                metrics['bertscore_p'] = bertscore['precision']
                metrics['bertscore_r'] = bertscore['recall']
                metrics['bertscore_f1'] = bertscore['f1']

    return metrics


def print_metrics(metrics):
    """
    Print metrics in a formatted way.

    Args:
        metrics: Dictionary of metric scores
    """
    print("\n" + "="*50)
    print("EVALUATION METRICS")
    print("="*50)

    # BLEU scores
    if 'bleu' in metrics:
        print(f"\nBLEU Score: {metrics['bleu']:.2f}")
        if 'bleu_1' in metrics:
            print(f"  BLEU-1: {metrics['bleu_1']:.2f}")
            print(f"  BLEU-2: {metrics['bleu_2']:.2f}")
            print(f"  BLEU-3: {metrics['bleu_3']:.2f}")
            print(f"  BLEU-4: {metrics['bleu_4']:.2f}")

    # chrF++
    if 'chrf' in metrics:
        print(f"\nchrF++ Score: {metrics['chrf']:.2f}")

    # COMET
    if 'comet' in metrics:
        print(f"\nCOMET Score: {metrics['comet']:.4f}")

    # BERTScore
    if 'bertscore_f1' in metrics:
        print(f"\nBERTScore:")
        print(f"  Precision: {metrics['bertscore_p']:.4f}")
        print(f"  Recall: {metrics['bertscore_r']:.4f}")
        print(f"  F1: {metrics['bertscore_f1']:.4f}")

    print("="*50 + "\n")
