"""Console output and file export for computed metrics."""

import os
import json
from collections import defaultdict
from typing import Dict

import numpy as np
import pandas as pd

from metrics.utils import convert_numpy_types

from metrics.tables import generate_all_latex_tables


def print_summary(agg_df: pd.DataFrame, sig_df: pd.DataFrame, length_stats: Dict,
                  budget_effects: pd.DataFrame = None, budget_interaction: pd.DataFrame = None,
                  optimal_budget: Dict = None, volume_analysis: Dict = None,
                  word_difficulty_strat: pd.DataFrame = None, calibration: Dict = None,
                  failure_analysis: Dict = None, agreement: pd.DataFrame = None):
    """Print human-readable summary of all computed metrics."""
    print("\n" + "="*80)
    print("COMPREHENSIVE METRICS SUMMARY")
    print("="*80)
    print("\nAggregated by Configuration:")
    print("-" * 80)

    for _, row in agg_df.iterrows():
        model = row['model_size'].upper()
        thinking = "ON " if row['thinking'] else "OFF"
        budget_str = f" Budget={int(row['thinking_budget'])}tok" if 'thinking_budget' in row and pd.notna(row['thinking_budget']) else ""

        print(f"\n{model:3} Thinking {thinking}{budget_str} ({row['num_puzzles']} puzzles):")
        print(f"  Precision: {row['precision_mean']:.3f} ± {row['precision_std']:.3f}")
        print(f"  Recall:    {row['recall_mean']:.3f} ± {row['recall_std']:.3f}")
        print(f"  F1:        {row['f1_mean']:.3f} ± {row['f1_std']:.3f}")

        num_pred = row.get('num_predicted_mean', np.nan)
        efficiency = row.get('efficiency', np.nan)
        print(f"  Generation Volume: {num_pred:.1f} predictions/puzzle")
        print(f"  Efficiency: {efficiency:.3f} (recall per prediction)")

        print(f"  Difficulty-Weighted Recall: {row.get('difficulty_weighted_recall_mean', np.nan):.3f}")

        cv = row.get('cv_f1', np.nan)
        success_30 = row.get('success_rate_30', 0) * 100
        p5 = row.get('f1_p5', np.nan)
        print(f"  Consistency: CV={cv:.2f}, Success@30={success_30:.0f}%, Worst 5%={p5:.3f}")

        l4 = row.get('length_4_recall_mean', np.nan) * 100
        l7 = row.get('length_7+_recall_mean', np.nan) * 100
        print(f"  Length Coverage: 4-letter={l4:.1f}%, 7+letter={l7:.1f}%")

    print("\n" + "-" * 80)
    print("THINKING MODE EFFECTS:")
    print("-" * 80)

    if len(sig_df) > 0:
        for _, row in sig_df.iterrows():
            model = row['model_size'].upper()
            print(f"\n{model:3}:")
            print(f"  Delta Precision: {row['precision_delta']:+.3f} (p={row['precision_p_value']:.4f}, d={row['precision_cohens_d']:.2f})")
            print(f"  Delta Recall:    {row['recall_delta']:+.3f} (p={row['recall_p_value']:.4f}, d={row['recall_cohens_d']:.2f})")
            print(f"  Delta F1:        {row['f1_delta']:+.3f} (p={row['f1_p_value']:.4f}, d={row['f1_cohens_d']:.2f})")

    print("\n" + "-" * 80)
    print("WORD LENGTH VS. DIFFICULTY:")
    print("-" * 80)

    if 'error' not in length_stats:
        print(f"\nMean length (found words):  {length_stats['mean_length_found']:.2f} letters")
        print(f"Mean length (missed words): {length_stats['mean_length_missed']:.2f} letters")
        print(f"Difference: {length_stats['difference']:+.2f} letters")
        print(f"t-statistic: {length_stats['t_statistic']:.2f}")
        print(f"p-value: {length_stats['p_value']:.6f} {'***' if length_stats['significant'] else ''}")
        print(f"Cohen's d: {length_stats['cohens_d']:.2f}")
        print(f"\n{length_stats['interpretation']}")
    else:
        print(f"\nError computing length correlation: {length_stats['error']}")

    if budget_effects is not None and len(budget_effects) > 0:
        print("\n" + "-" * 80)
        print("THINKING BUDGET EFFECTS:")
        print("-" * 80)

        for model_size in budget_effects['model_size'].unique():
            model_data = budget_effects[budget_effects['model_size'] == model_size]
            print(f"\n{model_size.upper()}:")
            for _, row in model_data.iterrows():
                print(f"  {row['budget_from']}->{row['budget_to']}tok: Delta F1={row['delta_f1']:+.3f}, "
                      f"Efficiency={row['efficiency_f1_per_1k']:+.4f}/1Ktok, "
                      f"Diminishing Returns: {row['diminishing_returns']}")

    if optimal_budget is not None and 'error' not in optimal_budget:
        print("\n" + "-" * 80)
        print("OPTIMAL BUDGET RECOMMENDATIONS:")
        print("-" * 80)

        for model_size, rec in optimal_budget.items():
            print(f"\n{model_size.upper()}: {rec['optimal_budget']} tokens")
            print(f"  F1 at optimal: {rec['f1_at_optimal']:.3f}")
            print(f"  Recall: {rec['recall_at_optimal']:.3f}")
            print(f"  Tested {rec['num_budgets_tested']} budget levels")

    if budget_interaction is not None and len(budget_interaction) > 0:
        print("\n" + "-" * 80)
        print("BUDGET-MODEL SIZE INTERACTION:")
        print("-" * 80)

        for _, row in budget_interaction.iterrows():
            print(f"\n{row['model_size'].upper()}: Budget Sensitivity = {row['budget_sensitivity']}")
            print(f"  F1 improvement: {row['f1_improvement']:+.3f} over {row['budget_range']} token range")
            print(f"  Efficiency: {row['f1_improvement_per_1k']:+.4f} F1 points per 1K tokens")

    if volume_analysis is not None and 'error' not in volume_analysis:
        print("\n" + "-" * 80)
        print("VOLUME-PERFORMANCE RELATIONSHIP:")
        print("-" * 80)

        corrs = volume_analysis.get('overall_correlations', {})
        print(f"\nOverall Correlations (Thinking Mode ON):")
        print(f"  Volume vs Precision: {corrs.get('volume_vs_precision', 0):+.3f}")
        print(f"  Volume vs Recall:    {corrs.get('volume_vs_recall', 0):+.3f}")
        print(f"  Volume vs F1:        {corrs.get('volume_vs_f1', 0):+.3f}")

        print(f"\nInterpretation:")
        print(f"  - Weak volume-precision correlation indicates models maintain quality")
        print(f"  - Strong volume-recall correlation shows more predictions = more coverage")
        print(f"  - F1 captures combined effect of quality and coverage")

        thinking_effect = volume_analysis.get('thinking_mode_effect', {})
        if thinking_effect:
            print(f"\nThinking Mode Effect on Volume:")
            for model, stats_data in thinking_effect.items():
                vol_inc = stats_data.get('volume_increase', 0)
                vol_pct = stats_data.get('volume_increase_pct', 0)
                print(f"  {model.upper():3}: +{vol_inc:.1f} predictions ({vol_pct:+.1f}%)")

    if word_difficulty_strat is not None and len(word_difficulty_strat) > 0:
        print("\n" + "-" * 80)
        print("WORD-LEVEL DIFFICULTY STRATIFICATION (BY HUMAN SUCCESS RATES):")
        print("-" * 80)

        # Show for optimal budget per model (thinking ON)
        thinking_strat = word_difficulty_strat[word_difficulty_strat['thinking'] == True]

        if 'thinking_budget' in thinking_strat.columns:
            # Get best budget per model
            print("\nUsing optimal budget per model:")

        for model in ['4b', '8b', '14b', '30b', '32b']:
            model_data = thinking_strat[thinking_strat['model_size'] == model]
            if len(model_data) > 0:
                # If multiple budgets, show best one
                if 'thinking_budget' in model_data.columns and model_data['thinking_budget'].notna().any():
                    # Average across budgets for simplicity
                    quartile_recalls = model_data.groupby('difficulty_quartile')['recall'].mean()
                else:
                    quartile_recalls = model_data.set_index('difficulty_quartile')['recall']

                if len(quartile_recalls) > 0:
                    print(f"\n{model.upper()}:")
                    for q in ['Very Easy (Q1)', 'Easy (Q2)', 'Hard (Q3)', 'Very Hard (Q4)']:
                        if q in quartile_recalls.index:
                            print(f"  {q}: {quartile_recalls[q]*100:.1f}% recall")

    if calibration is not None and 'error' not in calibration:
        print("\n" + "-" * 80)
        print("MODEL-HUMAN CALIBRATION:")
        print("-" * 80)
        print("\nCorrelation between model difficulty and human difficulty (higher = better)")
        print("Perfect calibration: models struggle with same words as humans")

        # Show average calibration per model (across budgets)
        calibration_by_model = defaultdict(list)
        for config_str, metrics in calibration.items():
            model = metrics['model_size']
            if metrics['thinking']:
                calibration_by_model[model].append(metrics['correlation'])

        print(f"\nModel Calibration Scores:")
        for model in ['4b', '8b', '14b', '30b', '32b']:
            if model in calibration_by_model:
                avg_corr = np.mean(calibration_by_model[model])
                print(f"  {model.upper():3}: r = {avg_corr:+.3f}")

    if failure_analysis is not None and 'error' not in failure_analysis:
        print("\n" + "-" * 80)
        print("FAILURE MODE ANALYSIS:")
        print("-" * 80)
        print("\nWords that are EASY for humans (>80% success) but models often miss:")

        top_misses = failure_analysis.get('top_easy_words_missed', [])[:10]
        if top_misses:
            print(f"\nTop 10 Systematic Failures:")
            for i, word_info in enumerate(top_misses, 1):
                print(f"  {i:2}. '{word_info['word']:12}' - "
                      f"Human: {word_info['human_success_rate']*100:.1f}%, "
                      f"Model Miss Rate: {word_info['model_miss_rate']*100:.1f}%")

        summary = failure_analysis.get('summary', {})
        if summary:
            print(f"\nAnalyzed {summary.get('total_easy_words_analyzed', 0)} easy words")
            print(f"Mean model miss rate on easy words: {summary.get('mean_miss_rate', 0)*100:.1f}%")

    if agreement is not None and len(agreement) > 0:
        print("\n" + "-" * 80)
        print("INTER-MODEL AGREEMENT:")
        print("-" * 80)

        for _, row in agreement.iterrows():
            overall_agr = row['overall_agreement']
            print(f"\nOverall Agreement: {overall_agr*100:.1f}%")
            print(f"  {row['interpretation']}")
            print(f"  (Based on {row['total_word_instances']} word instances)")

    print("\n" + "="*80)


def save_results(df: pd.DataFrame, agg_df: pd.DataFrame, sig_df: pd.DataFrame,
                 difficulty_df: pd.DataFrame, length_df: pd.DataFrame,
                 length_stats: Dict, difficulty_gap_df: pd.DataFrame,
                 thinking_consistency_df: pd.DataFrame, scaling_pattern_df: pd.DataFrame,
                 budget_effects_df: pd.DataFrame, budget_interaction_df: pd.DataFrame,
                 optimal_budget: Dict, volume_analysis: Dict,
                 word_difficulty_strat_df: pd.DataFrame, model_human_calibration: Dict,
                 failure_mode_analysis: Dict, inter_model_agreement_df: pd.DataFrame,
                 output_dir: str):
    """Save all results to CSV, JSON, and LaTeX files."""
    os.makedirs(output_dir, exist_ok=True)

    csv_path = os.path.join(output_dir, 'detailed_metrics.csv')
    df.to_csv(csv_path, index=False)
    print(f"\nSaved detailed metrics to: {csv_path}")

    agg_path = os.path.join(output_dir, 'aggregated_metrics.csv')
    agg_df.to_csv(agg_path, index=False)
    print(f"Saved aggregated metrics to: {agg_path}")

    sig_path = os.path.join(output_dir, 'statistical_significance.csv')
    sig_df.to_csv(sig_path, index=False)
    print(f"Saved significance tests to: {sig_path}")

    diff_path = os.path.join(output_dir, 'metrics_by_difficulty.csv')
    difficulty_df.to_csv(diff_path, index=False)
    print(f"Saved difficulty metrics to: {diff_path}")

    length_path = os.path.join(output_dir, 'metrics_by_word_length.csv')
    length_df.to_csv(length_path, index=False)
    print(f"Saved word length metrics to: {length_path}")

    if len(budget_effects_df) > 0:
        budget_effects_path = os.path.join(output_dir, 'thinking_budget_effects.csv')
        budget_effects_df.to_csv(budget_effects_path, index=False)
        print(f"Saved thinking budget effects to: {budget_effects_path}")

    if len(budget_interaction_df) > 0:
        budget_interaction_path = os.path.join(output_dir, 'budget_model_interaction.csv')
        budget_interaction_df.to_csv(budget_interaction_path, index=False)
        print(f"Saved budget-model interaction to: {budget_interaction_path}")

    if len(word_difficulty_strat_df) > 0:
        word_diff_path = os.path.join(output_dir, 'word_difficulty_stratification.csv')
        word_difficulty_strat_df.to_csv(word_diff_path, index=False)
        print(f"Saved word difficulty stratification to: {word_diff_path}")

    if len(inter_model_agreement_df) > 0:
        agreement_path = os.path.join(output_dir, 'inter_model_agreement.csv')
        inter_model_agreement_df.to_csv(agreement_path, index=False)
        print(f"Saved inter-model agreement to: {agreement_path}")

    summary = {
        'num_puzzles': int(df['puzzle_id'].nunique()),
        'num_configurations': len(agg_df),
        'configurations': agg_df.to_dict('records'),
        'statistical_significance': sig_df.to_dict('records'),
        'length_difficulty_analysis': length_stats,
        'difficulty_stratification': difficulty_df.to_dict('records'),
        'length_stratification': length_df.to_dict('records'),
        'difficulty_gap_analysis': difficulty_gap_df.to_dict('records'),
        'thinking_consistency_effects': thinking_consistency_df.to_dict('records'),
        'scaling_pattern_analysis': scaling_pattern_df.to_dict('records'),
        'thinking_budget_effects': budget_effects_df.to_dict('records') if len(budget_effects_df) > 0 else [],
        'budget_model_interaction': budget_interaction_df.to_dict('records') if len(budget_interaction_df) > 0 else [],
        'optimal_budget_recommendations': optimal_budget,
        'volume_performance_analysis': volume_analysis,
        'word_difficulty_stratification': word_difficulty_strat_df.to_dict('records') if len(word_difficulty_strat_df) > 0 else [],
        'model_human_calibration': model_human_calibration,
        'failure_mode_analysis': failure_mode_analysis,
        'inter_model_agreement': inter_model_agreement_df.to_dict('records') if len(inter_model_agreement_df) > 0 else []
    }

    summary = convert_numpy_types(summary)

    json_path = os.path.join(output_dir, 'metrics_summary.json')
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Saved JSON summary to: {json_path}")

    num_tables = 12
    if len(budget_effects_df) > 0:
        num_tables += 1
        if len(budget_interaction_df) > 0:
            num_tables += 1
        if optimal_budget and 'error' not in optimal_budget:
            num_tables += 1
        if volume_analysis and 'error' not in volume_analysis:
            num_tables += 1
        if len(word_difficulty_strat_df) > 0:
            num_tables += 1
        if model_human_calibration and 'error' not in model_human_calibration:
            num_tables += 1
        if failure_mode_analysis and 'error' not in failure_mode_analysis:
            num_tables += 1

    latex_tables = generate_all_latex_tables(
        agg_df, sig_df, length_stats, difficulty_df,
        difficulty_gap_df, thinking_consistency_df, scaling_pattern_df,
        budget_effects_df, budget_interaction_df, optimal_budget, volume_analysis,
        word_difficulty_strat_df, model_human_calibration, failure_mode_analysis
    )

    latex_path = os.path.join(output_dir, 'latex_tables.tex')
    with open(latex_path, 'w', encoding='utf-8') as f:
        f.write(latex_tables)
    print(f"Saved LaTeX tables ({num_tables} total) to: {latex_path}")
