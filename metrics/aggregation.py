"""Core statistical functions and configuration-level aggregation."""

from typing import Dict, Tuple

import numpy as np
import pandas as pd
from scipy import stats


def compute_confidence_interval(data: np.ndarray, confidence: float = 0.95) -> Tuple[float, float]:
    """Compute confidence interval using t-distribution."""
    if len(data) < 2:
        return (np.nan, np.nan)

    mean = np.mean(data)
    sem = stats.sem(data)
    ci = stats.t.interval(confidence, len(data) - 1, loc=mean, scale=sem)
    return ci


def compute_cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
    """Compute Cohen's d effect size between two samples."""
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))

    if pooled_std == 0:
        return 0.0

    return (np.mean(group1) - np.mean(group2)) / pooled_std


def compute_consistency_metrics(group: pd.DataFrame) -> Dict:
    """Compute CV, percentiles, and success-rate thresholds for a configuration."""
    f1_values = group['f1'].values

    metrics = {
        'cv_f1': np.std(f1_values, ddof=1) / np.mean(f1_values) if np.mean(f1_values) > 0 else np.nan,
        'f1_p5': np.percentile(f1_values, 5),
        'f1_p25': np.percentile(f1_values, 25),
        'f1_p75': np.percentile(f1_values, 75),
        'f1_p95': np.percentile(f1_values, 95),
        'success_rate_20': np.sum(f1_values >= 0.2) / len(f1_values) if len(f1_values) > 0 else 0,
        'success_rate_30': np.sum(f1_values >= 0.3) / len(f1_values) if len(f1_values) > 0 else 0,
        'success_rate_40': np.sum(f1_values >= 0.4) / len(f1_values) if len(f1_values) > 0 else 0,
        'success_rate_50': np.sum(f1_values >= 0.5) / len(f1_values) if len(f1_values) > 0 else 0,
    }

    return metrics


def aggregate_by_configuration(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate metrics by (model_size, thinking, thinking_budget) with CIs and consistency."""
    results = []

    group_cols = ['model_size', 'thinking']
    if 'thinking_budget' in df.columns:
        group_cols.append('thinking_budget')

    for group_keys, group in df.groupby(group_cols):
        metrics_to_aggregate = [
            'precision', 'recall', 'f1', 'car',
            'difficulty_weighted_recall',
            'length_4_recall', 'length_5_recall', 'length_6_recall', 'length_7+_recall',
            'prefix_coverage',
            'avg_word_length_found', 'avg_word_length_missed',
            'pangram_recall',
            'fp_constraint_violations', 'fp_non_dictionary',
            'num_predicted', 'num_actual', 'num_correct'
        ]

        if len(group_cols) == 3:
            model_size, thinking, thinking_budget = group_keys
            row = {
                'model_size': model_size,
                'thinking': thinking,
                'thinking_budget': thinking_budget,
                'num_puzzles': len(group)
            }
        else:
            model_size, thinking = group_keys
            row = {
                'model_size': model_size,
                'thinking': thinking,
                'num_puzzles': len(group)
            }

        for metric in metrics_to_aggregate:
            if metric not in group.columns:
                continue

            values = group[metric].dropna().values
            if len(values) == 0:
                row[f'{metric}_mean'] = np.nan
                row[f'{metric}_std'] = np.nan
                row[f'{metric}_ci_lower'] = np.nan
                row[f'{metric}_ci_upper'] = np.nan
                continue

            mean = np.mean(values)
            std = np.std(values, ddof=1) if len(values) > 1 else 0.0
            ci_lower, ci_upper = compute_confidence_interval(values)

            row[f'{metric}_mean'] = mean
            row[f'{metric}_std'] = std
            row[f'{metric}_ci_lower'] = ci_lower
            row[f'{metric}_ci_upper'] = ci_upper

        consistency = compute_consistency_metrics(group)
        row.update(consistency)

        # Efficiency: recall per prediction, normalized by solution size
        if 'num_predicted_mean' in row and 'num_actual_mean' in row and row['num_actual_mean'] > 0:
            predictions_per_actual = row['num_predicted_mean'] / row['num_actual_mean']
            if predictions_per_actual > 0:
                row['efficiency'] = row.get('recall_mean', 0) / predictions_per_actual
            else:
                row['efficiency'] = 0
        else:
            row['efficiency'] = 0

        results.append(row)

    agg_df = pd.DataFrame(results)

    size_order = {'4b': 0, '8b': 1, '14b': 2, '32b': 3, '30b': 3, 'small': 4}
    agg_df['size_order'] = agg_df['model_size'].map(size_order)

    sort_cols = ['size_order']
    sort_ascending = [True]

    if 'thinking_budget' in agg_df.columns:
        sort_cols.append('thinking_budget')
        sort_ascending.append(True)

    sort_cols.append('thinking')
    sort_ascending.append(False)

    agg_df = agg_df.sort_values(sort_cols, ascending=sort_ascending)
    agg_df = agg_df.drop('size_order', axis=1)

    return agg_df
