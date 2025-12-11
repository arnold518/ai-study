"""Error analysis tools for translation quality.

Analyzes common translation errors:
- Repetition errors (repeated n-grams)
- Number errors (hallucinated or incorrect numbers)
- Length errors (too short/long translations)
- Coverage errors (missing/extra content)
"""

import re
from collections import Counter, defaultdict
import numpy as np


class ErrorAnalyzer:
    """Analyzes translation errors for debugging and improvement."""

    def __init__(self):
        """Initialize error counters."""
        self.errors = {
            'repetition': [],
            'number_mismatch': [],
            'length_ratio': [],
            'unknown_tokens': []
        }

    def analyze_batch(self, sources, hypotheses, references, source_tokenizer=None, target_tokenizer=None):
        """
        Analyze a batch of translations for errors.

        Args:
            sources: List of source sentences
            hypotheses: List of predicted translations
            references: List of reference translations
            source_tokenizer: Optional tokenizer for source language
            target_tokenizer: Optional tokenizer for target language

        Returns:
            errors_dict: Dictionary with error statistics
        """
        batch_errors = {
            'repetition': 0,
            'number_mismatch': 0,
            'too_short': 0,
            'too_long': 0,
            'unknown_tokens': 0
        }

        for src, hyp, ref in zip(sources, hypotheses, references):
            # Check for repetitions
            if self.has_repetition(hyp):
                batch_errors['repetition'] += 1
                self.errors['repetition'].append({
                    'source': src,
                    'hypothesis': hyp,
                    'reference': ref
                })

            # Check for number mismatches
            if not self.numbers_match(src, hyp):
                batch_errors['number_mismatch'] += 1
                self.errors['number_mismatch'].append({
                    'source': src,
                    'hypothesis': hyp,
                    'reference': ref,
                    'source_numbers': self.extract_numbers(src),
                    'hyp_numbers': self.extract_numbers(hyp)
                })

            # Check length ratio
            ratio = len(hyp.split()) / max(len(ref.split()), 1)
            self.errors['length_ratio'].append(ratio)

            if ratio < 0.5:
                batch_errors['too_short'] += 1
            elif ratio > 2.0:
                batch_errors['too_long'] += 1

            # Check for unknown tokens (if tokenizers provided)
            if target_tokenizer:
                tokens = target_tokenizer.tokenize(hyp)
                unk_count = sum(1 for t in tokens if '<unk>' in t or '�' in t)
                if unk_count > 0:
                    batch_errors['unknown_tokens'] += 1
                    self.errors['unknown_tokens'].append({
                        'hypothesis': hyp,
                        'unk_count': unk_count
                    })

        return batch_errors

    def has_repetition(self, text, n=3, threshold=2):
        """
        Check if text has repeated n-grams.

        Args:
            text: Input text
            n: N-gram size (default: 3)
            threshold: Number of repetitions to flag (default: 2)

        Returns:
            has_rep: True if repetition found
        """
        words = text.split()

        # Check various n-gram sizes
        for ngram_size in range(2, n + 1):
            ngrams = [' '.join(words[i:i+ngram_size]) for i in range(len(words) - ngram_size + 1)]
            counts = Counter(ngrams)

            # Flag if any n-gram appears more than threshold times
            if any(count > threshold for count in counts.values()):
                return True

        return False

    def extract_numbers(self, text):
        """
        Extract all numbers from text.

        Args:
            text: Input text

        Returns:
            numbers: List of numbers (as strings)
        """
        # Match integers and floats
        pattern = r'\d+\.?\d*'
        numbers = re.findall(pattern, text)
        return numbers

    def numbers_match(self, source, hypothesis):
        """
        Check if numbers in source appear in hypothesis.

        Args:
            source: Source sentence
            hypothesis: Translated sentence

        Returns:
            match: True if numbers match (or no numbers in source)
        """
        src_numbers = set(self.extract_numbers(source))
        hyp_numbers = set(self.extract_numbers(hypothesis))

        # No numbers in source - always passes
        if not src_numbers:
            return True

        # Check if all source numbers appear in hypothesis
        # (hypothesis may have additional numbers, which is acceptable)
        return src_numbers.issubset(hyp_numbers)

    def get_statistics(self):
        """
        Get error statistics.

        Returns:
            stats: Dictionary with error counts and percentages
        """
        total = len(self.errors['length_ratio'])

        stats = {
            'total_samples': total,
            'repetition_errors': len(self.errors['repetition']),
            'number_errors': len(self.errors['number_mismatch']),
            'unknown_token_errors': len(self.errors['unknown_tokens']),
            'avg_length_ratio': np.mean(self.errors['length_ratio']) if self.errors['length_ratio'] else 0,
            'std_length_ratio': np.std(self.errors['length_ratio']) if self.errors['length_ratio'] else 0
        }

        # Calculate percentages
        if total > 0:
            stats['repetition_rate'] = 100 * stats['repetition_errors'] / total
            stats['number_error_rate'] = 100 * stats['number_errors'] / total
            stats['unk_rate'] = 100 * stats['unknown_token_errors'] / total
        else:
            stats['repetition_rate'] = 0
            stats['number_error_rate'] = 0
            stats['unk_rate'] = 0

        return stats

    def print_statistics(self):
        """Print error statistics in a formatted way."""
        stats = self.get_statistics()

        print("\n" + "="*60)
        print("ERROR ANALYSIS")
        print("="*60)
        print(f"Total Samples: {stats['total_samples']}")
        print()

        print("Error Rates:")
        print(f"  Repetition Errors:    {stats['repetition_errors']:4d} ({stats['repetition_rate']:5.2f}%)")
        print(f"  Number Mismatches:    {stats['number_errors']:4d} ({stats['number_error_rate']:5.2f}%)")
        print(f"  Unknown Tokens:       {stats['unknown_token_errors']:4d} ({stats['unk_rate']:5.2f}%)")
        print()

        print("Length Statistics:")
        print(f"  Average Length Ratio: {stats['avg_length_ratio']:.3f}")
        print(f"  Std Dev:              {stats['std_length_ratio']:.3f}")
        print("="*60 + "\n")

    def get_error_examples(self, error_type='repetition', max_examples=5):
        """
        Get examples of a specific error type.

        Args:
            error_type: Type of error ('repetition', 'number_mismatch', 'unknown_tokens')
            max_examples: Maximum number of examples to return

        Returns:
            examples: List of error examples
        """
        if error_type not in self.errors:
            return []

        return self.errors[error_type][:max_examples]

    def print_error_examples(self, error_type='repetition', max_examples=3):
        """
        Print examples of a specific error type.

        Args:
            error_type: Type of error
            max_examples: Maximum number of examples to print
        """
        examples = self.get_error_examples(error_type, max_examples)

        if not examples:
            print(f"No {error_type} errors found.")
            return

        print(f"\n{error_type.upper()} ERROR EXAMPLES:")
        print("-" * 60)

        for i, example in enumerate(examples, 1):
            print(f"\nExample {i}:")
            print(f"  Source:     {example['source']}")
            print(f"  Hypothesis: {example['hypothesis']}")
            print(f"  Reference:  {example['reference']}")

            if error_type == 'number_mismatch':
                print(f"  Source Numbers:     {example['source_numbers']}")
                print(f"  Hypothesis Numbers: {example['hyp_numbers']}")
            elif error_type == 'unknown_tokens':
                print(f"  Unknown Token Count: {example['unk_count']}")

        print("-" * 60)

    def reset(self):
        """Reset all error counters."""
        self.errors = {
            'repetition': [],
            'number_mismatch': [],
            'length_ratio': [],
            'unknown_tokens': []
        }


def analyze_translation_errors(sources, predictions, references,
                               source_tokenizer=None, target_tokenizer=None,
                               verbose=True):
    """
    Convenience function for error analysis.

    Args:
        sources: List of source sentences
        predictions: List of predicted translations
        references: List of reference translations
        source_tokenizer: Optional source tokenizer
        target_tokenizer: Optional target tokenizer
        verbose: Whether to print statistics

    Returns:
        analyzer: ErrorAnalyzer with results
    """
    analyzer = ErrorAnalyzer()
    analyzer.analyze_batch(sources, predictions, references, source_tokenizer, target_tokenizer)

    if verbose:
        analyzer.print_statistics()

        # Print examples of each error type
        for error_type in ['repetition', 'number_mismatch']:
            if analyzer.errors[error_type]:
                analyzer.print_error_examples(error_type, max_examples=2)

    return analyzer
