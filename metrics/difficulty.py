"""Difficulty stratification, gap analysis, and word-level difficulty."""

from typing import Dict

import numpy as np
import pandas as pd


def stratify_by_difficulty(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate metrics by puzzle difficulty (easy/medium/hard)."""
    results = []

    group_cols = ['model_size', 'thinking']
    if 'thinking_budget' in df.columns:
        group_cols.append('thinking_budget')
    group_cols.append('puzzle_difficulty')

    for group_keys, group in df.groupby(group_cols):
        if len(group) == 0:
            continue

        if len(group_cols) == 4:
            model_size, thinking, thinking_budget, difficulty = group_keys
            row = {
                'model_size': model_size,
                'thinking': thinking,
                'thinking_budget': thinking_budget,
                'difficulty': difficulty,
                'num_puzzles': len(group),
                'f1_mean': np.mean(group['f1']),
                'f1_std': np.std(group['f1'], ddof=1) if len(group) > 1 else 0.0,
                'recall_mean': np.mean(group['recall']),
                'precision_mean': np.mean(group['precision']),
                'car_mean': np.mean(group['car']),
            }
        else:
            model_size, thinking, difficulty = group_keys
            row = {
                'model_size': model_size,
                'thinking': thinking,
                'difficulty': difficulty,
                'num_puzzles': len(group),
                'f1_mean': np.mean(group['f1']),
                'f1_std': np.std(group['f1'], ddof=1) if len(group) > 1 else 0.0,
                'recall_mean': np.mean(group['recall']),
                'precision_mean': np.mean(group['precision']),
                'car_mean': np.mean(group['car']),
            }
        results.append(row)

    return pd.DataFrame(results)


def compute_difficulty_gap_analysis(difficulty_df: pd.DataFrame) -> pd.DataFrame:
    """Compute recall gap between easy and hard puzzles per configuration."""
    results = []

    group_cols = ['model_size', 'thinking']
    if 'thinking_budget' in difficulty_df.columns:
        group_cols.append('thinking_budget')

    for group_keys, group in difficulty_df.groupby(group_cols):
        easy = group[group['difficulty'] == 'easy']
        hard = group[group['difficulty'] == 'hard']

        if len(easy) > 0 and len(hard) > 0:
            easy_recall = easy['recall_mean'].values[0]
            hard_recall = hard['recall_mean'].values[0]
            gap = easy_recall - hard_recall

            if len(group_cols) == 3:
                model_size, thinking, thinking_budget = group_keys
                results.append({
                    'model_size': model_size,
                    'thinking': thinking,
                    'thinking_budget': thinking_budget,
                    'easy_recall': easy_recall,
                    'hard_recall': hard_recall,
                    'difficulty_gap': gap
                })
            else:
                model_size, thinking = group_keys
                results.append({
                    'model_size': model_size,
                    'thinking': thinking,
                    'easy_recall': easy_recall,
                    'hard_recall': hard_recall,
                    'difficulty_gap': gap
                })

    return pd.DataFrame(results)


def compute_word_level_difficulty_stratification(df: pd.DataFrame, difficulty_data: Dict[int, Dict[str, float]]) -> pd.DataFrame:
    """Compute model recall by human difficulty quartile (Q1=very easy to Q4=very hard)."""
    if not difficulty_data:
        return pd.DataFrame()

    word_level_data = []

    for _, row in df.iterrows():
        puzzle_id = row['puzzle_id']
        if puzzle_id not in difficulty_data:
            continue

        word_difficulties = difficulty_data[puzzle_id]
        actual_words = row['actual_words']
        correctly_predicted = set(row['correctly_predicted'])

        for word in actual_words:
            if word.lower() in word_difficulties:
                human_difficulty = word_difficulties[word.lower()]
                human_success_rate = 1 - human_difficulty
                model_found = word in correctly_predicted

                word_level_data.append({
                    'model_size': row['model_size'],
                    'thinking': row['thinking'],
                    'thinking_budget': row.get('thinking_budget'),
                    'word': word,
                    'human_difficulty': human_difficulty,
                    'human_success_rate': human_success_rate,
                    'model_found': model_found
                })

    if not word_level_data:
        return pd.DataFrame()

    word_df = pd.DataFrame(word_level_data)

    quartile_labels = ['Very Easy (Q1)', 'Easy (Q2)', 'Hard (Q3)', 'Very Hard (Q4)']
    word_df['difficulty_quartile'] = pd.qcut(word_df['human_difficulty'], q=4,
                                               labels=quartile_labels, duplicates='drop')

    results = []
    group_cols = ['model_size', 'thinking']
    if 'thinking_budget' in word_df.columns and word_df['thinking_budget'].notna().any():
        group_cols.append('thinking_budget')
    group_cols.append('difficulty_quartile')

    for group_keys, group in word_df.groupby(group_cols):
        recall = group['model_found'].mean()

        row_dict = dict(zip(group_cols, group_keys))
        row_dict['recall'] = recall
        row_dict['num_words'] = len(group)
        row_dict['mean_human_success_rate'] = group['human_success_rate'].mean()

        results.append(row_dict)

    return pd.DataFrame(results)
