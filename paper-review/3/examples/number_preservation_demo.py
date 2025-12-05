#!/usr/bin/env python
"""
Demonstration of number preservation strategies for NMT.

This script shows 3 easy-to-implement strategies:
1. Number Placeholder/Masking
2. Number Substitution Augmentation
3. Post-processing Alignment
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import re
import random
from typing import List, Tuple

# ==============================================================================
# Strategy 1: Number Placeholder / Masking
# ==============================================================================

def mask_numbers(text: str) -> Tuple[str, List[str]]:
    """
    Replace numbers with placeholders.

    Args:
        text: Input text with numbers

    Returns:
        masked_text: Text with numbers replaced by <NUM_i>
        numbers: List of extracted numbers
    """
    numbers = []

    def replacer(match):
        numbers.append(match.group())
        return f"<NUM_{len(numbers)-1}>"

    # Pattern: Match integers and decimals
    pattern = r'\b\d+(?:[.,]\d+)*\b'
    masked = re.sub(pattern, replacer, text)

    return masked, numbers


def unmask_numbers(text: str, numbers: List[str]) -> str:
    """
    Restore numbers from placeholders.

    Args:
        text: Text with <NUM_i> placeholders
        numbers: List of numbers to restore

    Returns:
        Text with placeholders replaced by actual numbers
    """
    for i, num in enumerate(numbers):
        text = text.replace(f"<NUM_{i}>", num)
    return text


def translate_with_number_masking(text: str, translate_fn) -> str:
    """
    Translate with number preservation via masking.

    Args:
        text: Source text
        translate_fn: Translation function

    Returns:
        Translated text with numbers preserved
    """
    # Mask numbers
    masked_text, numbers = mask_numbers(text)
    print(f"  Masked: {masked_text}")
    print(f"  Numbers: {numbers}")

    # Translate
    translated_masked = translate_fn(masked_text)
    print(f"  Translated (masked): {translated_masked}")

    # Unmask
    translated = unmask_numbers(translated_masked, numbers)
    print(f"  Final: {translated}")

    return translated


# ==============================================================================
# Strategy 2: Number Substitution Augmentation (Data Augmentation)
# ==============================================================================

def extract_numbers_parallel(src: str, tgt: str) -> Tuple[List[str], bool]:
    """
    Extract numbers from parallel sentences and check if they match.

    Args:
        src: Source sentence
        tgt: Target sentence

    Returns:
        numbers: List of numbers
        match: True if source and target have same numbers
    """
    pattern = r'\b\d+(?:[.,]\d+)*\b'
    src_nums = re.findall(pattern, src)
    tgt_nums = re.findall(pattern, tgt)

    return src_nums, src_nums == tgt_nums


def substitute_numbers(text: str, num_map: dict) -> str:
    """
    Substitute numbers in text according to mapping.

    Args:
        text: Input text
        num_map: Dictionary mapping old numbers to new numbers

    Returns:
        Text with substituted numbers
    """
    for old, new in num_map.items():
        text = text.replace(old, new)
    return text


def augment_parallel_pair(src: str, tgt: str) -> Tuple[str, str]:
    """
    Augment a parallel pair by substituting numbers.

    Args:
        src: Source sentence
        tgt: Target sentence

    Returns:
        Augmented source and target
    """
    # Extract numbers
    src_nums, match = extract_numbers_parallel(src, tgt)

    if not match or not src_nums:
        # No augmentation if numbers don't match
        return src, tgt

    # Create substitution mapping
    num_map = {}
    for num in set(src_nums):
        # Generate random replacement (preserve format)
        if '.' in num:
            # Decimal number
            parts = num.split('.')
            new_int = str(random.randint(1, 9999))
            new_dec = ''.join(str(random.randint(0, 9)) for _ in parts[1])
            num_map[num] = f"{new_int}.{new_dec}"
        elif ',' in num:
            # Comma-formatted number
            clean = num.replace(',', '')
            new_num = random.randint(1, int(clean) * 10)
            num_map[num] = f"{new_num:,}"
        else:
            # Plain integer
            num_map[num] = str(random.randint(1, 9999))

    # Apply substitution
    aug_src = substitute_numbers(src, num_map)
    aug_tgt = substitute_numbers(tgt, num_map)

    return aug_src, aug_tgt


# ==============================================================================
# Strategy 3: Post-processing Alignment
# ==============================================================================

def extract_numbers_with_positions(text: str) -> List[Tuple[str, int]]:
    """
    Extract numbers and their positions.

    Args:
        text: Input text

    Returns:
        List of (number, position) tuples
    """
    pattern = r'\b\d+(?:[.,]\d+)*\b'
    return [(m.group(), m.start()) for m in re.finditer(pattern, text)]


def align_and_copy_numbers(src: str, tgt: str) -> str:
    """
    Post-process translation to align numbers with source.

    Strategy:
    - If number counts match, replace 1-to-1
    - If counts don't match, use heuristics

    Args:
        src: Source sentence
        tgt: Target sentence (model output)

    Returns:
        Corrected target sentence
    """
    src_numbers = extract_numbers_with_positions(src)
    tgt_numbers = extract_numbers_with_positions(tgt)

    # Extract just the number strings
    src_nums = [num for num, _ in src_numbers]
    tgt_nums = [num for num, _ in tgt_numbers]

    if len(src_nums) == 0:
        # No numbers to copy
        return tgt

    if len(src_nums) == len(tgt_nums):
        # Same count - replace 1-to-1
        result = tgt
        for src_num, tgt_num in zip(src_nums, tgt_nums):
            if src_num != tgt_num:
                # Replace first occurrence
                result = result.replace(tgt_num, src_num, 1)
        return result
    else:
        # Different counts - replace numbers not in source
        result = tgt
        for tgt_num in tgt_nums:
            if tgt_num not in src_nums:
                # This number is hallucinated
                # Try to find a source number to use
                if src_nums:
                    # Use first unused source number
                    replacement = src_nums[0]
                    result = result.replace(tgt_num, replacement, 1)
                    # Don't reuse
                    src_nums = [n for n in src_nums if n != replacement]
        return result


# ==============================================================================
# Demo
# ==============================================================================

def mock_translate(text: str) -> str:
    """
    Mock translation function that sometimes hallucin ates numbers.

    In real usage, replace with actual model.translate()
    """
    # Simulate number hallucinations
    hallucinations = {
        "2025": "2024",
        "123": "124",
        "456": "789",
        "3.14": "3.15",
    }

    result = text
    for original, hallucinated in hallucinations.items():
        if original in text:
            # 50% chance to hallucinate
            if random.random() < 0.5:
                result = result.replace(original, hallucinated)

    return result


def demo():
    """Demonstrate all three strategies."""

    print("=" * 80)
    print("NUMBER PRESERVATION STRATEGIES DEMO")
    print("=" * 80)
    print()

    # Test sentences
    test_sentences = [
        "The meeting is scheduled for 12/05/2025 at 3:30 PM.",
        "I have 123 apples and 456 oranges.",
        "The temperature is 3.14 degrees.",
        "Call me at 555-1234.",
    ]

    for i, src in enumerate(test_sentences, 1):
        print(f"\n[Example {i}]")
        print(f"Source: {src}")
        print()

        # Strategy 1: Number Masking
        print("Strategy 1: Number Masking")
        print("-" * 40)
        translated = translate_with_number_masking(src, mock_translate)
        print()

        # Strategy 2: Data Augmentation (show augmented version)
        print("Strategy 2: Data Augmentation")
        print("-" * 40)
        # Mock target (same as source for demo)
        tgt = src  # In real case, this would be Korean translation
        aug_src, aug_tgt = augment_parallel_pair(src, tgt)
        print(f"  Original: {src}")
        print(f"  Augmented: {aug_src}")
        print()

        # Strategy 3: Post-processing
        print("Strategy 3: Post-processing Alignment")
        print("-" * 40)
        # Simulate hallucinated translation
        tgt_hallucinated = mock_translate(src)
        print(f"  Model output (hallucinated): {tgt_hallucinated}")

        # Correct with alignment
        tgt_corrected = align_and_copy_numbers(src, tgt_hallucinated)
        print(f"  Corrected: {tgt_corrected}")
        print()

        print("=" * 80)


if __name__ == "__main__":
    # Set seed for reproducibility
    random.seed(42)

    demo()

    print()
    print("SUMMARY")
    print("=" * 80)
    print()
    print("Three Easy Strategies Demonstrated:")
    print()
    print("1. NUMBER MASKING (Strategy 1)")
    print("   - Replace numbers with <NUM_i> placeholders")
    print("   - Translate")
    print("   - Restore numbers from placeholders")
    print("   - ✅ Pros: 100% accurate if placeholders preserved")
    print("   - ❌ Cons: Assumes 1-to-1 correspondence")
    print()
    print("2. DATA AUGMENTATION (Strategy 2)")
    print("   - During training, randomly replace numbers")
    print("   - Forces model to learn number copying")
    print("   - ✅ Pros: Improves model behavior")
    print("   - ❌ Cons: Requires retraining")
    print()
    print("3. POST-PROCESSING (Strategy 3)")
    print("   - After translation, align source/target numbers")
    print("   - Replace hallucinated numbers")
    print("   - ✅ Pros: No model changes needed")
    print("   - ❌ Cons: Heuristic-based")
    print()
    print("RECOMMENDATION:")
    print("  Use Strategy 1 (Masking) + Strategy 3 (Post-processing) together")
    print("  Add Strategy 2 (Augmentation) when retraining model")
    print()
