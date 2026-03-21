"""
Model-human calibration, failure mode analysis, and inter-model agreement.
"""

from collections import defaultdict
from typing import Dict

import numpy as np
import pandas as pd
from scipy import stats


def compute_model_human_calibration(df: pd.DataFrame, difficulty_data: Dict[int, Dict[str, float]]) -> dict:
    """
    Analyze how well model difficulty aligns with human difficulty.

    Computes correlation between human success rates and model success rates
    at the word level. High correlation means models struggle with the same
    words humans struggle with (good calibration).

    Args:
        df: DataFrame with per-puzzle results
        difficulty_data: Dict mapping puzzle_id -> {word: difficulty_score}

    Returns:
        Dict with calibration metrics per model configuration
    """
    if not difficulty_data:
        return {'error': 'No difficulty data available'}

    # Collect word-level success rates
    word_success_data = defaultdict(lambda: {'human_difficulty': [], 'model_success': []})

    for _, row in df.iterrows():
        puzzle_id = row['puzzle_id']
        if puzzle_id not in difficulty_data:
            continue

        config_key = (row['model_size'], row['thinking'], row.get('thinking_budget'))
        word_difficulties = difficulty_data[puzzle_id]
        actual_words = row['actual_words']
        correctly_predicted = set(row['correctly_predicted'])

        for word in actual_words:
            if word.lower() in word_difficulties:
                human_difficulty = word_difficulties[word.lower()]
                model_found = word in correctly_predicted

                word_success_data[config_key]['human_difficulty'].append(human_difficulty)
                word_success_data[config_key]['model_success'].append(1.0 if model_found else 0.0)

    # Compute calibration metrics
    calibration_results = {}

    for config_key, data in word_success_data.items():
        if len(data['human_difficulty']) < 10:  # Need minimum data points
            continue

        human_diff = np.array(data['human_difficulty'])
        model_success = np.array(data['model_success'])
        model_difficulty = 1 - model_success  # Convert to difficulty

        # Correlation (higher = better calibration)
        # Use Spearman: human_diff and model_difficulty are bounded proportions
        # with potentially skewed marginals; Spearman measures monotonic
        # association without assuming linearity or normality.
        if len(human_diff) > 1:
            correlation, p_value = stats.spearmanr(human_diff, model_difficulty)
        else:
            correlation, p_value = 0, 1.0

        # Mean absolute error
        mae = np.mean(np.abs(human_diff - model_difficulty))

        # Group by difficulty bins and compute model success rate
        bins = [0, 0.25, 0.5, 0.75, 1.0]
        bin_labels = ['Easy', 'Medium-Easy', 'Medium-Hard', 'Hard']

        bin_indices = np.digitize(human_diff, bins)
        bin_model_success = {}
        for i, label in enumerate(bin_labels, start=1):
            mask = bin_indices == i
            if mask.sum() > 0:
                bin_model_success[label] = {
                    'model_recall': float(model_success[mask].mean()),
                    'num_words': int(mask.sum()),
                    'mean_human_difficulty': float(human_diff[mask].mean())
                }

        model_size, thinking, budget = config_key
        config_str = f"{model_size}_{thinking}_{budget if budget else 'NA'}"

        calibration_results[config_str] = {
            'model_size': model_size,
            'thinking': thinking,
            'thinking_budget': budget,
            'correlation': float(correlation),
            'p_value': float(p_value),
            'mae': float(mae),
            'num_words': len(human_diff),
            'bin_performance': bin_model_success
        }

    return calibration_results


def compute_failure_mode_analysis(df: pd.DataFrame, difficulty_data: Dict[int, Dict[str, float]]) -> dict:
    """
    Identify systematic failure patterns - words that are easy for humans but hard for models.

    Args:
        df: DataFrame with per-puzzle results
        difficulty_data: Dict mapping puzzle_id -> {word: difficulty_score}

    Returns:
        Dict with failure analysis including top missed easy words
    """
    if not difficulty_data:
        return {'error': 'No difficulty data available'}

    # Collect misses for easy words (human success > 80%)
    easy_threshold = 0.2  # difficulty < 0.2 means success > 80%

    word_stats = defaultdict(lambda: {'config_appearances': 0, 'model_misses': 0,
                                       'human_difficulties': [], 'puzzle_ids': set()})

    for _, row in df.iterrows():
        puzzle_id = row['puzzle_id']
        if puzzle_id not in difficulty_data:
            continue

        word_difficulties = difficulty_data[puzzle_id]
        actual_words = row['actual_words']
        correctly_predicted = set(row['correctly_predicted'])

        for word in actual_words:
            if word.lower() in word_difficulties:
                human_diff = word_difficulties[word.lower()]

                if human_diff < easy_threshold:  # Easy for humans
                    word_stats[word]['config_appearances'] += 1
                    word_stats[word]['puzzle_ids'].add(puzzle_id)
                    # Collect per-puzzle difficulty (may add duplicates across configs,
                    # but we deduplicate via puzzle_ids for the mean)
                    word_stats[word]['human_difficulties'].append(human_diff)

                    if word not in correctly_predicted:
                        word_stats[word]['model_misses'] += 1

    # Calculate miss rates and find worst offenders
    easy_words_missed = []
    for word, stats_data in word_stats.items():
        num_puzzles = len(stats_data['puzzle_ids'])
        if num_puzzles >= 3:  # Word appears in at least 3 puzzles
            miss_rate = stats_data['model_misses'] / stats_data['config_appearances']
            mean_human_diff = np.mean(stats_data['human_difficulties'])

            easy_words_missed.append({
                'word': word,
                'human_success_rate': float(1 - mean_human_diff),
                'human_difficulty': float(mean_human_diff),
                'appearances': num_puzzles,
                'model_miss_rate': float(miss_rate),
                'total_misses': stats_data['model_misses']
            })

    # Sort by miss rate
    easy_words_missed.sort(key=lambda x: x['model_miss_rate'], reverse=True)

    return {
        'top_easy_words_missed': easy_words_missed[:20],  # Top 20
        'summary': {
            'total_easy_words_analyzed': len(word_stats),
            'mean_miss_rate': np.mean([w['model_miss_rate'] for w in easy_words_missed]) if easy_words_missed else 0
        }
    }


def compute_inter_model_agreement(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute inter-model agreement - do models agree on which words are hard?

    Returns:
        DataFrame with agreement metrics
    """
    # Group by puzzle and word, see which models got it
    word_results = defaultdict(lambda: defaultdict(int))

    for _, row in df.iterrows():
        puzzle_id = row['puzzle_id']
        correctly_predicted = set(row['correctly_predicted'])
        actual_words = row['actual_words']

        config = f"{row['model_size']}_{row['thinking']}"

        for word in actual_words:
            key = f"{puzzle_id}_{word}"
            if word in correctly_predicted:
                word_results[key][config] = 1
            else:
                word_results[key][config] = 0

    # Calculate agreement (what % of words do all models agree on?)
    agreements = []
    for word_key, model_results in word_results.items():
        if len(model_results) > 1:
            values = list(model_results.values())
            # All agree if all 1s or all 0s
            all_found = sum(values) == len(values)
            all_missed = sum(values) == 0
            agreement = 1 if (all_found or all_missed) else 0
            agreements.append(agreement)

    overall_agreement = np.mean(agreements) if agreements else 0

    return pd.DataFrame([{
        'overall_agreement': overall_agreement,
        'total_word_instances': len(agreements),
        'interpretation': f"{overall_agreement*100:.1f}% of words show unanimous model agreement"
    }])
