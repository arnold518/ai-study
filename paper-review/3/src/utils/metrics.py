"""Evaluation metrics."""

import sacrebleu


def compute_bleu(predictions, references):
    """
    Compute BLEU score.

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


def compute_metrics(predictions, references):
    """
    Compute multiple evaluation metrics.

    Args:
        predictions: List of predicted sentences
        references: List of reference sentences

    Returns:
        metrics: Dictionary of metric scores
    """
    bleu = compute_bleu(predictions, references)

    return {
        'bleu': bleu.score,
        'bleu_1': bleu.precisions[0],
        'bleu_2': bleu.precisions[1],
        'bleu_3': bleu.precisions[2],
        'bleu_4': bleu.precisions[3],
    }
