"""Statistical tests, scaling analysis, word length stratification, and volume analysis."""

from collections import defaultdict
from typing import Dict

import numpy as np
import pandas as pd
from scipy import stats

from metrics.config import MODEL_SIZES
from metrics.aggregation import compute_cohens_d
from metrics.basic import categorize_word_length


def compute_statistical_significance(df: pd.DataFrame) -> pd.DataFrame:
    """Paired t-tests and Cohen's d for thinking on vs. off, per model size."""
    results = []

    for model_size in MODEL_SIZES:
        thinking_on = df[(df['model_size'] == model_size) & (df['thinking'] == True)]
        thinking_off = df[(df['model_size'] == model_size) & (df['thinking'] == False)]

        if len(thinking_on) == 0 or len(thinking_off) == 0:
            continue

        if len(thinking_on) != len(thinking_off):
            print(f"Warning: Unequal samples for {model_size}, skipping paired test")
            continue

        row = {'model_size': model_size}

        for metric in ['precision', 'recall', 'f1']:
            on_values = thinking_on[metric].values
            off_values = thinking_off[metric].values

            t_stat, p_value = stats.ttest_rel(on_values, off_values)
            cohens_d = compute_cohens_d(on_values, off_values)

            row[f'{metric}_t_stat'] = t_stat
            row[f'{metric}_p_value'] = p_value
            row[f'{metric}_cohens_d'] = cohens_d
            row[f'{metric}_delta'] = np.mean(on_values) - np.mean(off_values)

        results.append(row)

    return pd.DataFrame(results)


def compute_thinking_consistency_effects(agg_df: pd.DataFrame) -> pd.DataFrame:
    """Compute thinking mode effects on consistency (delta CV, worst-5%, success rate)."""
    results = []

    if 'thinking_budget' in agg_df.columns:
        for (model_size, budget) in agg_df[['model_size', 'thinking_budget']].drop_duplicates().values:
            thinking_on = agg_df[(agg_df['model_size'] == model_size) &
                                 (agg_df['thinking_budget'] == budget) &
                                 (agg_df['thinking'] == True)]
            thinking_off = agg_df[(agg_df['model_size'] == model_size) &
                                  (agg_df['thinking_budget'] == budget) &
                                  (agg_df['thinking'] == False)]

            if len(thinking_on) > 0 and len(thinking_off) > 0:
                on_row = thinking_on.iloc[0]
                off_row = thinking_off.iloc[0]

                results.append({
                    'model_size': model_size,
                    'thinking_budget': budget,
                    'delta_f1': on_row['f1_mean'] - off_row['f1_mean'],
                    'delta_cv': on_row['cv_f1'] - off_row['cv_f1'],
                    'delta_worst_5': on_row['f1_p5'] - off_row['f1_p5'],
                    'delta_success_30': on_row['success_rate_30'] - off_row['success_rate_30']
                })
    else:
        for model_size in agg_df['model_size'].unique():
            thinking_on = agg_df[(agg_df['model_size'] == model_size) & (agg_df['thinking'] == True)]
            thinking_off = agg_df[(agg_df['model_size'] == model_size) & (agg_df['thinking'] == False)]

            if len(thinking_on) > 0 and len(thinking_off) > 0:
                on_row = thinking_on.iloc[0]
                off_row = thinking_off.iloc[0]

                results.append({
                    'model_size': model_size,
                    'delta_f1': on_row['f1_mean'] - off_row['f1_mean'],
                    'delta_cv': on_row['cv_f1'] - off_row['cv_f1'],
                    'delta_worst_5': on_row['f1_p5'] - off_row['f1_p5'],
                    'delta_success_30': on_row['success_rate_30'] - off_row['success_rate_30']
                })

    return pd.DataFrame(results)


def compute_scaling_pattern_analysis(agg_df: pd.DataFrame) -> pd.DataFrame:
    """Pairwise model comparisons to identify scaling anomalies (negative delta)."""
    thinking_df = agg_df[agg_df['thinking'] == True].copy()

    if 'thinking_budget' in thinking_df.columns:
        all_results = []

        for budget in thinking_df['thinking_budget'].unique():
            budget_df = thinking_df[thinking_df['thinking_budget'] == budget].copy()

            size_order = {'4b': 0, '8b': 1, '14b': 2, '30b': 3, '32b': 3}
            budget_df['size_order'] = budget_df['model_size'].map(size_order)
            budget_df = budget_df.sort_values('size_order')

            models = budget_df['model_size'].tolist()

            for i in range(len(models) - 1):
                curr = budget_df[budget_df['model_size'] == models[i]].iloc[0]
                next_model = budget_df[budget_df['model_size'] == models[i + 1]].iloc[0]

                delta_f1 = next_model['f1_mean'] - curr['f1_mean']
                delta_recall = next_model['recall_mean'] - curr['recall_mean']

                # Determine if this is an anomaly (negative change)
                is_anomaly = delta_f1 < 0 or delta_recall < 0

                all_results.append({
                    'thinking_budget': budget,
                    'comparison': f"{models[i].upper()}→{models[i+1].upper()}",
                    'delta_f1': delta_f1,
                    'delta_recall': delta_recall,
                    'is_anomaly': is_anomaly,
                    'interpretation': 'DEGRADATION (anomaly)' if is_anomaly else 'Improvement'
                })

        return pd.DataFrame(all_results)
    else:
        size_order = {'4b': 0, '8b': 1, '14b': 2, '30b': 3, '32b': 3}
        thinking_df['size_order'] = thinking_df['model_size'].map(size_order)
        thinking_df = thinking_df.sort_values('size_order')

        results = []
        models = thinking_df['model_size'].tolist()

        for i in range(len(models) - 1):
            curr = thinking_df[thinking_df['model_size'] == models[i]].iloc[0]
            next_model = thinking_df[thinking_df['model_size'] == models[i + 1]].iloc[0]

            delta_f1 = next_model['f1_mean'] - curr['f1_mean']
            delta_recall = next_model['recall_mean'] - curr['recall_mean']
            is_anomaly = delta_f1 < 0 or delta_recall < 0

            results.append({
                'comparison': f"{models[i].upper()}→{models[i+1].upper()}",
                'delta_f1': delta_f1,
                'delta_recall': delta_recall,
                'is_anomaly': is_anomaly,
                'interpretation': 'DEGRADATION (anomaly)' if is_anomaly else 'Improvement'
            })

        return pd.DataFrame(results)


def stratify_by_word_length(df: pd.DataFrame) -> pd.DataFrame:
    """Compute recall by word-length category for each configuration."""
    results = []

    group_cols = ['model_size', 'thinking']
    if 'thinking_budget' in df.columns:
        group_cols.append('thinking_budget')

    for group_keys, group in df.groupby(group_cols):
        length_stats = defaultdict(lambda: {'found': 0, 'total': 0})

        for _, row in group.iterrows():
            for word in row['actual_words']:
                length_cat = categorize_word_length(len(word))
                length_stats[length_cat]['total'] += 1

            for word in row['correctly_predicted']:
                length_cat = categorize_word_length(len(word))
                length_stats[length_cat]['found'] += 1

        for length_cat in ['4', '5', '6', '7+']:
            if length_stats[length_cat]['total'] > 0:
                recall = length_stats[length_cat]['found'] / length_stats[length_cat]['total']
            else:
                recall = np.nan

            if len(group_cols) == 3:
                model_size, thinking, thinking_budget = group_keys
                results.append({
                    'model_size': model_size,
                    'thinking': thinking,
                    'thinking_budget': thinking_budget,
                    'length_category': length_cat,
                    'total_words': length_stats[length_cat]['total'],
                    'found_words': length_stats[length_cat]['found'],
                    'recall': recall
                })
            else:
                model_size, thinking = group_keys
                results.append({
                    'model_size': model_size,
                    'thinking': thinking,
                    'length_category': length_cat,
                    'total_words': length_stats[length_cat]['total'],
                    'found_words': length_stats[length_cat]['found'],
                    'recall': recall
                })

    return pd.DataFrame(results)


def compute_word_length_correlation(df: pd.DataFrame) -> Dict:
    """Test whether missed words are longer than found words (paired t-test + Cohen's d)."""
    valid_rows = df.dropna(subset=['avg_word_length_missed', 'avg_word_length_found'])

    if len(valid_rows) == 0:
        return {'error': 'No valid length data'}

    all_length_missed = valid_rows['avg_word_length_missed'].values
    all_length_found = valid_rows['avg_word_length_found'].values

    t_stat, p_value = stats.ttest_rel(all_length_missed, all_length_found)
    cohens_d = compute_cohens_d(all_length_missed, all_length_found)

    return {
        'mean_length_missed': float(np.mean(all_length_missed)),
        'mean_length_found': float(np.mean(all_length_found)),
        'difference': float(np.mean(all_length_missed) - np.mean(all_length_found)),
        't_statistic': float(t_stat),
        'p_value': float(p_value),
        'cohens_d': float(cohens_d),
        'significant': bool(p_value < 0.05),
        'num_comparisons': int(len(valid_rows)),
        'interpretation': 'Missed words are longer (harder)' if np.mean(all_length_missed) > np.mean(all_length_found) else 'Found words are longer'
    }


def compute_volume_performance_analysis(df: pd.DataFrame) -> Dict:
    """Correlate generation volume with precision, recall, and F1."""
    thinking_df = df[df['thinking'] == True].copy()

    if len(thinking_df) == 0:
        return {'error': 'No thinking mode data'}

    volume_metrics = ['num_predicted', 'precision', 'recall', 'f1']
    available_metrics = [m for m in volume_metrics if m in thinking_df.columns]

    if len(available_metrics) < 2:
        return {'error': 'Insufficient metrics for correlation'}

    corr_matrix = thinking_df[available_metrics].corr()

    analysis = {
        'overall_correlations': {
            'volume_vs_precision': float(corr_matrix.loc['num_predicted', 'precision']) if 'precision' in corr_matrix else None,
            'volume_vs_recall': float(corr_matrix.loc['num_predicted', 'recall']) if 'recall' in corr_matrix else None,
            'volume_vs_f1': float(corr_matrix.loc['num_predicted', 'f1']) if 'f1' in corr_matrix else None,
        },
        'by_model': {}
    }

    for model in thinking_df['model_size'].unique():
        model_data = thinking_df[thinking_df['model_size'] == model]
        if len(model_data) > 1:
            model_corr = model_data[available_metrics].corr()
            analysis['by_model'][model] = {
                'volume_vs_precision': float(model_corr.loc['num_predicted', 'precision']) if 'precision' in model_corr else None,
                'volume_vs_recall': float(model_corr.loc['num_predicted', 'recall']) if 'recall' in model_corr else None,
                'volume_vs_f1': float(model_corr.loc['num_predicted', 'f1']) if 'f1' in model_corr else None,
                'mean_volume': float(model_data['num_predicted'].mean()),
                'mean_precision': float(model_data['precision'].mean()),
                'mean_recall': float(model_data['recall'].mean()),
            }

    analysis['thinking_mode_effect'] = {}
    for model in df['model_size'].unique():
        thinking_on = df[(df['model_size'] == model) & (df['thinking'] == True)]
        thinking_off = df[(df['model_size'] == model) & (df['thinking'] == False)]

        if len(thinking_on) > 0 and len(thinking_off) > 0:
            analysis['thinking_mode_effect'][model] = {
                'volume_on': float(thinking_on['num_predicted'].mean()),
                'volume_off': float(thinking_off['num_predicted'].mean()),
                'volume_increase': float(thinking_on['num_predicted'].mean() - thinking_off['num_predicted'].mean()),
                'volume_increase_pct': float((thinking_on['num_predicted'].mean() / thinking_off['num_predicted'].mean() - 1) * 100) if thinking_off['num_predicted'].mean() > 0 else 0,
            }

    return analysis
