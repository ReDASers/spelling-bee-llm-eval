"""Compute metrics from Spelling Bee inference results using the metrics package."""

import os
import argparse

from metrics import (
    load_all_results, load_word_difficulty_data, add_metrics_to_dataframe,
    aggregate_by_configuration, compute_statistical_significance,
    stratify_by_difficulty, stratify_by_word_length,
    compute_word_length_correlation, compute_difficulty_gap_analysis,
    compute_thinking_consistency_effects, compute_scaling_pattern_analysis,
    compute_thinking_budget_effects, compute_budget_model_interaction,
    compute_optimal_budget_analysis, compute_volume_performance_analysis,
    compute_word_level_difficulty_stratification, compute_model_human_calibration,
    compute_failure_mode_analysis, compute_inter_model_agreement,
    print_summary, save_results,
)


def main():
    parser = argparse.ArgumentParser(description='Compute metrics from inference results')
    parser.add_argument('--results-dir', type=str, default='data/qwen-results',
                       help='Directory containing result JSON files')
    parser.add_argument('--output-dir', type=str, default='metrics/output/',
                       help='Directory to save computed metrics')
    parser.add_argument('--difficulty-dir', type=str, default='data/puzzles',
                       help='Directory containing bee_*.json puzzle files with human difficulty data')

    args = parser.parse_args()

    print("="*80)
    print("COMPUTING COMPREHENSIVE METRICS FROM INFERENCE RESULTS")
    print("="*80)
    print(f"Results directory: {args.results_dir}")
    print(f"Output directory: {args.output_dir}")

    print("\nLoading results...")
    additional_dirs = []

    script_dir = os.path.dirname(os.path.abspath(__file__))
    for subdir in ['data/claude-results', 'data/openai-results']:
        candidate = os.path.join(script_dir, subdir)
        if os.path.exists(candidate):
            additional_dirs.append(candidate)

    df = load_all_results(args.results_dir, additional_dirs=additional_dirs if additional_dirs else None)

    print("\nLoading word difficulty data...")
    difficulty_data = load_word_difficulty_data(args.difficulty_dir)

    if difficulty_data:
        puzzles_with_difficulty = set(difficulty_data.keys())
        puzzles_in_results = set(df['puzzle_id'].unique())
        coverage = len(puzzles_with_difficulty & puzzles_in_results)
        print(f"  - Difficulty data available for {coverage}/{len(puzzles_in_results)} puzzles ({coverage/len(puzzles_in_results)*100:.1f}%)")

        df['difficulty_source'] = df['puzzle_id'].apply(
            lambda pid: 'user_data' if pid in puzzles_with_difficulty else 'length_estimate'
        )
    else:
        print("  - No difficulty data found, using length-based estimates")
        df['difficulty_source'] = 'length_estimate'

    print("\nComputing per-puzzle metrics...")
    df = add_metrics_to_dataframe(df, difficulty_data)

    print("\nAggregating statistics by configuration...")
    agg_df = aggregate_by_configuration(df)

    print("Computing statistical significance (paired t-tests)...")
    sig_df = compute_statistical_significance(df)

    print("Stratifying by puzzle difficulty...")
    difficulty_df = stratify_by_difficulty(df)

    print("Stratifying by word length...")
    length_df = stratify_by_word_length(df)

    print("Computing word length vs. difficulty correlation...")
    length_stats = compute_word_length_correlation(df)

    print("Computing difficulty gap analysis...")
    difficulty_gap_df = compute_difficulty_gap_analysis(difficulty_df)

    print("Computing thinking mode consistency effects...")
    thinking_consistency_df = compute_thinking_consistency_effects(agg_df)

    print("Computing scaling pattern analysis...")
    scaling_pattern_df = compute_scaling_pattern_analysis(agg_df)

    print("Computing thinking budget effects...")
    budget_effects_df = compute_thinking_budget_effects(agg_df)

    print("Computing budget-model size interaction...")
    budget_interaction_df = compute_budget_model_interaction(agg_df)

    print("Computing optimal budget recommendations...")
    optimal_budget = compute_optimal_budget_analysis(agg_df)

    print("Computing volume-performance relationship...")
    volume_analysis = compute_volume_performance_analysis(df)

    print("Computing word-level difficulty stratification...")
    word_difficulty_strat_df = compute_word_level_difficulty_stratification(df, difficulty_data)

    print("Computing model-human calibration...")
    model_human_calibration = compute_model_human_calibration(df, difficulty_data)

    print("Computing failure mode analysis...")
    failure_mode_analysis = compute_failure_mode_analysis(df, difficulty_data)

    print("Computing inter-model agreement...")
    inter_model_agreement_df = compute_inter_model_agreement(df)

    print_summary(agg_df, sig_df, length_stats, budget_effects_df, budget_interaction_df,
                  optimal_budget, volume_analysis, word_difficulty_strat_df,
                  model_human_calibration, failure_mode_analysis, inter_model_agreement_df)

    print("\nSaving results...")
    save_results(df, agg_df, sig_df, difficulty_df, length_df, length_stats,
                 difficulty_gap_df, thinking_consistency_df, scaling_pattern_df,
                 budget_effects_df, budget_interaction_df, optimal_budget, volume_analysis,
                 word_difficulty_strat_df, model_human_calibration, failure_mode_analysis,
                 inter_model_agreement_df, args.output_dir)



if __name__ == '__main__':
    main()
