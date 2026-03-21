"""
Thinking budget analysis: effects, model interaction, and optimal budget.
"""

import pandas as pd


def compute_thinking_budget_effects(agg_df: pd.DataFrame) -> pd.DataFrame:
    """
    Analyze how thinking budget affects performance for each model size.

    Returns:
        DataFrame with budget effects on performance metrics
    """
    if 'thinking_budget' not in agg_df.columns:
        return pd.DataFrame()

    results = []

    # Only analyze thinking=True rows
    thinking_df = agg_df[agg_df['thinking'] == True].copy()

    for model_size in thinking_df['model_size'].unique():
        model_data = thinking_df[thinking_df['model_size'] == model_size].copy()

        # Sort by budget
        model_data = model_data.sort_values('thinking_budget')
        budgets = model_data['thinking_budget'].values

        if len(budgets) < 2:
            continue

        for i in range(len(budgets) - 1):
            curr = model_data[model_data['thinking_budget'] == budgets[i]].iloc[0]
            next_budget = model_data[model_data['thinking_budget'] == budgets[i+1]].iloc[0]

            budget_increase = budgets[i+1] - budgets[i]
            delta_f1 = next_budget['f1_mean'] - curr['f1_mean']
            delta_recall = next_budget['recall_mean'] - curr['recall_mean']
            delta_precision = next_budget['precision_mean'] - curr['precision_mean']

            # Efficiency: performance gain per 1K tokens
            efficiency_f1 = (delta_f1 / budget_increase) * 1000
            efficiency_recall = (delta_recall / budget_increase) * 1000

            results.append({
                'model_size': model_size,
                'budget_from': int(budgets[i]),
                'budget_to': int(budgets[i+1]),
                'budget_increase': int(budget_increase),
                'delta_f1': delta_f1,
                'delta_recall': delta_recall,
                'delta_precision': delta_precision,
                'efficiency_f1_per_1k': efficiency_f1,
                'efficiency_recall_per_1k': efficiency_recall,
                'diminishing_returns': 'Yes' if delta_f1 < 0.01 else 'No'
            })

    return pd.DataFrame(results)


def compute_budget_model_interaction(agg_df: pd.DataFrame) -> pd.DataFrame:
    """
    Analyze whether smaller/larger models benefit differently from budget increases.

    Returns:
        DataFrame showing budget sensitivity by model size
    """
    if 'thinking_budget' not in agg_df.columns:
        return pd.DataFrame()

    results = []

    thinking_df = agg_df[agg_df['thinking'] == True].copy()

    for model_size in thinking_df['model_size'].unique():
        model_data = thinking_df[thinking_df['model_size'] == model_size].copy()
        model_data = model_data.sort_values('thinking_budget')

        if len(model_data) < 2:
            continue

        # Compute total improvement from lowest to highest budget
        min_budget_row = model_data.iloc[0]
        max_budget_row = model_data.iloc[-1]

        budget_range = max_budget_row['thinking_budget'] - min_budget_row['thinking_budget']
        f1_improvement = max_budget_row['f1_mean'] - min_budget_row['f1_mean']
        recall_improvement = max_budget_row['recall_mean'] - min_budget_row['recall_mean']

        # Budget sensitivity: how much does this model benefit from budget increases?
        results.append({
            'model_size': model_size,
            'min_budget': int(min_budget_row['thinking_budget']),
            'max_budget': int(max_budget_row['thinking_budget']),
            'budget_range': int(budget_range),
            'f1_min_budget': min_budget_row['f1_mean'],
            'f1_max_budget': max_budget_row['f1_mean'],
            'f1_improvement': f1_improvement,
            'recall_improvement': recall_improvement,
            'f1_improvement_per_1k': (f1_improvement / budget_range) * 1000,
            'budget_sensitivity': 'High' if f1_improvement > 0.05 else ('Medium' if f1_improvement > 0.02 else 'Low')
        })

    return pd.DataFrame(results)


def compute_optimal_budget_analysis(agg_df: pd.DataFrame) -> dict:
    """
    Determine optimal thinking budget for each model size based on performance.

    Returns:
        Dict with optimal budget recommendations
    """
    if 'thinking_budget' not in agg_df.columns:
        return {'error': 'No thinking budget data'}

    recommendations = {}

    thinking_df = agg_df[agg_df['thinking'] == True].copy()

    for model_size in thinking_df['model_size'].unique():
        model_data = thinking_df[thinking_df['model_size'] == model_size].copy()

        if len(model_data) == 0:
            continue

        # Find budget with highest F1
        best_row = model_data.loc[model_data['f1_mean'].idxmax()]

        recommendations[model_size] = {
            'optimal_budget': int(best_row['thinking_budget']),
            'f1_at_optimal': float(best_row['f1_mean']),
            'recall_at_optimal': float(best_row['recall_mean']),
            'precision_at_optimal': float(best_row['precision_mean']),
            'num_budgets_tested': len(model_data)
        }

    return recommendations
