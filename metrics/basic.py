"""
Per-puzzle metric computation: precision, recall, F1, CAR, pangrams,
difficulty-weighted recall, length coverage, prefix diversity.
"""

from typing import Dict, List

import numpy as np
import pandas as pd

from metrics.config import (
    MIN_WORD_LENGTH, DIFFICULTY_THRESHOLDS, word_difficulty_score
)


def compute_basic_metrics(row: pd.Series) -> Dict[str, float]:
    """
    Compute precision, recall, F1 for a single prediction.

    Args:
        row: DataFrame row with num_correct, num_predicted, num_actual

    Returns:
        Dict with precision, recall, f1
    """
    num_correct = row['num_correct']
    num_predicted = row['num_predicted']
    num_actual = row['num_actual']

    precision = num_correct / num_predicted if num_predicted > 0 else 0.0
    recall = num_correct / num_actual if num_actual > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


def compute_constraint_adherence_rate(row: pd.Series) -> float:
    """
    Compute CAR: proportion of predicted words that satisfy constraints.

    Since our parser already filters out violations, we check if any
    false positives violate constraints (shouldn't happen).
    """
    center_letter = row['center_letter']
    allowed_letters = set(row['all_letters'])
    predicted_words = row['predicted_words']

    violations = 0
    for word in predicted_words:
        # Check constraints
        if len(word) < MIN_WORD_LENGTH:
            violations += 1
        elif center_letter not in word:
            violations += 1
        elif not all(c in allowed_letters for c in word):
            violations += 1

    car = 1.0 - (violations / len(predicted_words)) if predicted_words else 1.0
    return car


def categorize_puzzle_difficulty(num_actual: int) -> str:
    """Categorize puzzle by difficulty based on solution count"""
    for difficulty, (min_val, max_val) in DIFFICULTY_THRESHOLDS.items():
        if min_val <= num_actual < max_val:
            return difficulty
    return 'unknown'


def compute_difficulty_weighted_recall(correctly_predicted: List[str], actual_words: List[str],
                                      difficulty_map: Dict[str, float] = None) -> float:
    """
    Compute difficulty-weighted recall: harder words count more.

    Uses real NYT user success rates when available.

    Args:
        correctly_predicted: List of correctly predicted words
        actual_words: List of all actual words
        difficulty_map: Optional dict mapping word -> difficulty score

    Returns:
        Weighted recall score (0-1)
    """
    if not actual_words:
        return 0.0

    total_difficulty = sum(word_difficulty_score(w, difficulty_map) for w in actual_words)
    found_difficulty = sum(word_difficulty_score(w, difficulty_map) for w in correctly_predicted)

    return found_difficulty / total_difficulty if total_difficulty > 0 else 0.0


def compute_length_distribution_coverage(predicted_words: List[str], actual_words: List[str]) -> Dict[str, float]:
    """
    Compute how well the model covers different word lengths.

    Returns:
        Dict with coverage metrics for each length category
    """
    length_cats = ['4', '5', '6', '7+']

    coverage = {}
    for cat in length_cats:
        actual_in_cat = [w for w in actual_words if categorize_word_length(len(w)) == cat]
        predicted_in_cat = [w for w in predicted_words if categorize_word_length(len(w)) == cat]

        if len(actual_in_cat) > 0:
            coverage[f'length_{cat}_recall'] = len(set(predicted_in_cat) & set(actual_in_cat)) / len(actual_in_cat)
            coverage[f'length_{cat}_count'] = len(actual_in_cat)
        else:
            coverage[f'length_{cat}_recall'] = np.nan
            coverage[f'length_{cat}_count'] = 0

    return coverage


def compute_prefix_diversity(words: List[str], prefix_len: int = 2) -> float:
    """
    Compute prefix diversity: how many unique prefixes are explored.

    Args:
        words: List of words
        prefix_len: Length of prefix to consider

    Returns:
        Number of unique prefixes
    """
    if not words:
        return 0.0

    prefixes = set(w[:prefix_len].lower() for w in words if len(w) >= prefix_len)
    return len(prefixes)


def is_pangram(word: str, all_letters: List[str]) -> bool:
    """Check if word uses all available letters"""
    return set(word.lower()) == set(all_letters)


def categorize_word_length(length: int) -> str:
    """Categorize word by length"""
    if length == 4:
        return '4'
    elif length == 5:
        return '5'
    elif length == 6:
        return '6'
    else:
        return '7+'


def analyze_false_positives(row: pd.Series) -> Dict[str, int]:
    """
    Categorize false positives into constraint violations vs. non-dictionary.

    Returns:
        Dict with counts of each error type
    """
    center_letter = row['center_letter']
    allowed_letters = set(row['all_letters'])
    false_positives = row['false_positives']

    constraint_violations = 0
    non_dictionary = 0

    for word in false_positives:
        # Check if it violates constraints
        if len(word) < MIN_WORD_LENGTH:
            constraint_violations += 1
        elif center_letter not in word:
            constraint_violations += 1
        elif not all(c in allowed_letters for c in word):
            constraint_violations += 1
        else:
            # Satisfies constraints but not in dictionary
            non_dictionary += 1

    return {
        'fp_constraint_violations': constraint_violations,
        'fp_non_dictionary': non_dictionary
    }


def add_metrics_to_dataframe(df: pd.DataFrame, difficulty_data: Dict[int, Dict[str, float]] = None) -> pd.DataFrame:
    """
    Add computed metrics as columns to DataFrame.

    Args:
        df: DataFrame with prediction results
        difficulty_data: Optional dict mapping puzzle_id -> {word: difficulty_score}

    Returns:
        DataFrame with additional metric columns
    """
    df = df.copy()

    # Compute basic metrics
    metrics = df.apply(compute_basic_metrics, axis=1, result_type='expand')
    df = pd.concat([df, metrics], axis=1)

    # Compute CAR (kept for reference, but note it's always 1.0 due to pre-filtering)
    df['car'] = df.apply(compute_constraint_adherence_rate, axis=1)

    # Add puzzle difficulty
    df['puzzle_difficulty'] = df['num_actual'].apply(categorize_puzzle_difficulty)

    # Analyze false positives
    fp_analysis = df.apply(analyze_false_positives, axis=1, result_type='expand')
    df = pd.concat([df, fp_analysis], axis=1)

    # Difficulty-weighted recall (using real user success rates when available)
    def compute_with_difficulty(row):
        puzzle_id = row['puzzle_id']
        difficulty_map = difficulty_data.get(puzzle_id) if difficulty_data else None
        return compute_difficulty_weighted_recall(
            row['correctly_predicted'], row['actual_words'], difficulty_map
        )

    df['difficulty_weighted_recall'] = df.apply(compute_with_difficulty, axis=1)

    # Length distribution coverage
    length_coverage = df.apply(
        lambda row: compute_length_distribution_coverage(
            row['correctly_predicted'], row['actual_words']
        ),
        axis=1, result_type='expand'
    )
    df = pd.concat([df, length_coverage], axis=1)

    # Prefix diversity metrics
    df['predicted_prefix_diversity'] = df.apply(
        lambda row: compute_prefix_diversity(row['predicted_words']),
        axis=1
    )
    df['actual_prefix_diversity'] = df.apply(
        lambda row: compute_prefix_diversity(row['actual_words']),
        axis=1
    )
    df['prefix_coverage'] = df.apply(
        lambda row: row['predicted_prefix_diversity'] / row['actual_prefix_diversity']
        if row['actual_prefix_diversity'] > 0 else 0.0,
        axis=1
    )

    # Average word length metrics
    df['avg_word_length_found'] = df.apply(
        lambda row: np.mean([len(w) for w in row['correctly_predicted']])
        if row['correctly_predicted'] else np.nan,
        axis=1
    )
    df['avg_word_length_missed'] = df.apply(
        lambda row: np.mean([len(w) for w in row['missed_words']])
        if row['missed_words'] else np.nan,
        axis=1
    )
    df['avg_word_length_all'] = df.apply(
        lambda row: np.mean([len(w) for w in row['actual_words']])
        if row['actual_words'] else np.nan,
        axis=1
    )

    # Compute pangram counts (kept but de-emphasized)
    df['num_actual_pangrams'] = df.apply(
        lambda row: sum(1 for w in row['actual_words'] if is_pangram(w, row['all_letters'])),
        axis=1
    )
    df['num_predicted_pangrams'] = df.apply(
        lambda row: sum(1 for w in row['predicted_words'] if is_pangram(w, row['all_letters'])),
        axis=1
    )
    df['num_correct_pangrams'] = df.apply(
        lambda row: sum(1 for w in row['correctly_predicted'] if is_pangram(w, row['all_letters'])),
        axis=1
    )
    df['pangram_recall'] = df.apply(
        lambda row: row['num_correct_pangrams'] / row['num_actual_pangrams']
        if row['num_actual_pangrams'] > 0 else np.nan,
        axis=1
    )

    return df
