"""
Metrics package for Spelling Bee LLM Evaluation.

Provides comprehensive metrics computation, aggregation, statistical analysis,
and reporting for the NYT Spelling Bee task evaluation.
"""

from metrics.config import (
    MODEL_SIZES, THINKING_MODES, DIFFICULTY_THRESHOLDS,
    MIN_WORD_LENGTH, F1_THRESHOLDS, word_difficulty_score,
)
from metrics.utils import convert_numpy_types
from metrics.loaders import (
    load_word_difficulty_data, load_results_file,
    find_all_results, load_all_results,
)
from metrics.basic import (
    compute_basic_metrics, compute_constraint_adherence_rate,
    categorize_puzzle_difficulty, compute_difficulty_weighted_recall,
    compute_length_distribution_coverage, compute_prefix_diversity,
    is_pangram, categorize_word_length, analyze_false_positives,
    add_metrics_to_dataframe,
)
from metrics.aggregation import (
    compute_confidence_interval, compute_cohens_d,
    compute_consistency_metrics, aggregate_by_configuration,
)
from metrics.statistical import (
    compute_statistical_significance,
    compute_thinking_consistency_effects,
    compute_scaling_pattern_analysis,
    stratify_by_word_length, compute_word_length_correlation,
    compute_volume_performance_analysis,
)
from metrics.budget import (
    compute_thinking_budget_effects,
    compute_budget_model_interaction,
    compute_optimal_budget_analysis,
)
from metrics.difficulty import (
    stratify_by_difficulty, compute_difficulty_gap_analysis,
    compute_word_level_difficulty_stratification,
)
from metrics.calibration import (
    compute_model_human_calibration,
    compute_failure_mode_analysis,
    compute_inter_model_agreement,
)
from metrics.reporting import print_summary, save_results
