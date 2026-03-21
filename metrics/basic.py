"""Per-puzzle metrics: precision, recall, F1, CAR, difficulty-weighted recall, length coverage."""

from typing import Dict, List

import numpy as np
import pandas as pd

from metrics.config import (
    MIN_WORD_LENGTH, DIFFICULTY_THRESHOLDS, word_difficulty_score
)


def compute_basic_metrics(row: pd.Series) -> Dict[str, float]:
    """Compute precision, recall, F1 for a single prediction."""
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
    """Compute CAR: fraction of predictions satisfying all spelling bee constraints.

    Typically 1.0 because the parser pre-filters violations.
    """
    center_letter = row['center_letter']
    allowed_letters = set(row['all_letters'])
    predicted_words = row['predicted_words']

    violations = 0
    for word in predicted_words:
        if len(word) < MIN_WORD_LENGTH:
            violations += 1
        elif center_letter not in word:
            violations += 1
        elif not all(c in allowed_letters for c in word):
            violations += 1

    car = 1.0 - (violations / len(predicted_words)) if predicted_words else 1.0
    return car


def categorize_puzzle_difficulty(num_actual: int) -> str:
    """Categorize puzzle by solution count into easy/medium/hard."""
    for difficulty, (min_val, max_val) in DIFFICULTY_THRESHOLDS.items():
        if min_val <= num_actual < max_val:
            return difficulty
    return 'unknown'


def compute_difficulty_weighted_recall(correctly_predicted: List[str], actual_words: List[str],
                                      difficulty_map: Dict[str, float] = None) -> float:
    """Compute difficulty-weighted recall where harder words count more.

    Uses NYT user success rates when available; falls back to length-based proxy.
    """
    if not actual_words:
        return 0.0

    total_difficulty = sum(word_difficulty_score(w, difficulty_map) for w in actual_words)
    found_difficulty = sum(word_difficulty_score(w, difficulty_map) for w in correctly_predicted)

    return found_difficulty / total_difficulty if total_difficulty > 0 else 0.0


def compute_length_distribution_coverage(predicted_words: List[str], actual_words: List[str]) -> Dict[str, float]:
    """Compute recall for each word-length category (4, 5, 6, 7+)."""
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
    """Count unique prefixes of the given length across words."""
    if not words:
        return 0.0

    prefixes = set(w[:prefix_len].lower() for w in words if len(w) >= prefix_len)
    return len(prefixes)


def is_pangram(word: str, all_letters: List[str]) -> bool:
    """Check if word uses all 7 available letters."""
    return set(word.lower()) == set(all_letters)


def categorize_word_length(length: int) -> str:
    """Map word length to category: '4', '5', '6', or '7+'."""
    if length == 4:
        return '4'
    elif length == 5:
        return '5'
    elif length == 6:
        return '6'
    else:
        return '7+'


def analyze_false_positives(row: pd.Series) -> Dict[str, int]:
    """Categorize false positives into constraint violations vs. non-dictionary words."""
    center_letter = row['center_letter']
    allowed_letters = set(row['all_letters'])
    false_positives = row['false_positives']

    constraint_violations = 0
    non_dictionary = 0

    for word in false_positives:
        if len(word) < MIN_WORD_LENGTH:
            constraint_violations += 1
        elif center_letter not in word:
            constraint_violations += 1
        elif not all(c in allowed_letters for c in word):
            constraint_violations += 1
        else:
            non_dictionary += 1

    return {
        'fp_constraint_violations': constraint_violations,
        'fp_non_dictionary': non_dictionary
    }


def add_metrics_to_dataframe(df: pd.DataFrame, difficulty_data: Dict[int, Dict[str, float]] = None) -> pd.DataFrame:
    """Add all computed metric columns to the predictions DataFrame."""
    df = df.copy()

    metrics = df.apply(compute_basic_metrics, axis=1, result_type='expand')
    df = pd.concat([df, metrics], axis=1)

    df['car'] = df.apply(compute_constraint_adherence_rate, axis=1)
    df['puzzle_difficulty'] = df['num_actual'].apply(categorize_puzzle_difficulty)

    fp_analysis = df.apply(analyze_false_positives, axis=1, result_type='expand')
    df = pd.concat([df, fp_analysis], axis=1)

    def compute_with_difficulty(row):
        puzzle_id = row['puzzle_id']
        difficulty_map = difficulty_data.get(puzzle_id) if difficulty_data else None
        return compute_difficulty_weighted_recall(
            row['correctly_predicted'], row['actual_words'], difficulty_map
        )

    df['difficulty_weighted_recall'] = df.apply(compute_with_difficulty, axis=1)

    length_coverage = df.apply(
        lambda row: compute_length_distribution_coverage(
            row['correctly_predicted'], row['actual_words']
        ),
        axis=1, result_type='expand'
    )
    df = pd.concat([df, length_coverage], axis=1)

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
