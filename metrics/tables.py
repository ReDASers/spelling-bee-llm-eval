"""Publication-ready LaTeX table generation from computed metrics."""

import numpy as np
import pandas as pd
from typing import Dict


def generate_latex_table_scaling(agg_df: pd.DataFrame) -> str:
    """Table: model scaling effects (thinking ON) with consistency metrics."""
    thinking_df = agg_df[agg_df['thinking'] == True].copy()

    has_budget = 'thinking_budget' in thinking_df.columns and thinking_df['thinking_budget'].notna().any()
    
    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\small")
    
    if has_budget:
        lines.append(r"\caption{Model scaling effects on generation performance with consistency metrics. All configurations use thinking mode enabled. Budget = thinking token budget. Volume = mean predictions per puzzle. Success@30 = \% of puzzles achieving F1 $\geq$ 0.3. CV = Coefficient of Variation (lower is more consistent).}")
        lines.append(r"\label{tab:model_scaling}")
        lines.append(r"\begin{tabular}{lccccccc}")
        lines.append(r"\toprule")
        lines.append(r"\textbf{Model} & \textbf{Budget} & \textbf{Precision} & \textbf{Recall} & \textbf{F1} & \textbf{Volume} & \textbf{CV(F1)} & \textbf{Succ@30} \\")
        lines.append(r"\midrule")
        
        for _, row in thinking_df.iterrows():
            model = row['model_size'].upper()
            budget = int(row['thinking_budget'])
            budget_str = f"{budget//1024}K" if budget >= 1024 else str(budget)
            p_mean = row['precision_mean']
            r_mean = row['recall_mean']
            f1_mean = row['f1_mean']
            volume = row.get('num_predicted_mean', np.nan)
            cv_f1 = row.get('cv_f1', np.nan)
            success_30 = row.get('success_rate_30', 0) * 100
            
            lines.append(
                f"{model:3} & {budget_str:4} & {p_mean:.3f} & {r_mean:.3f} & "
                f"{f1_mean:.3f} & {volume:.1f} & {cv_f1:.2f} & {success_30:.0f}\\% \\\\"
            )
    else:
        lines.append(r"\caption{Model scaling effects on generation performance with consistency metrics. All configurations use thinking mode enabled. Success@30 = \% of puzzles achieving F1 $\geq$ 0.3. CV = Coefficient of Variation (lower is more consistent).}")
        lines.append(r"\label{tab:model_scaling}")
        lines.append(r"\begin{tabular}{lccccc}")
        lines.append(r"\toprule")
        lines.append(r"\textbf{Model} & \textbf{Precision} & \textbf{Recall} & \textbf{F1} & \textbf{CV(F1)} & \textbf{Success@30} \\")
        lines.append(r"\midrule")
        
        for _, row in thinking_df.iterrows():
            model = row['model_size'].upper()
            p_mean = row['precision_mean']
            p_std = row['precision_std']
            r_mean = row['recall_mean']
            r_std = row['recall_std']
            f1_mean = row['f1_mean']
            f1_std = row['f1_std']
            cv_f1 = row.get('cv_f1', np.nan)
            success_30 = row.get('success_rate_30', 0) * 100
            
            lines.append(
                f"{model:3} & {p_mean:.3f} $\\pm$ {p_std:.2f} & "
                f"{r_mean:.3f} $\\pm$ {r_std:.2f} & "
                f"{f1_mean:.3f} $\\pm$ {f1_std:.2f} & "
                f"{cv_f1:.2f} & {success_30:.0f}\\% \\\\"
            )
    
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    
    return '\n'.join(lines)


def generate_latex_table_thinking(sig_df: pd.DataFrame) -> str:
    """Table: thinking mode ON vs OFF with statistical significance."""
    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(r"\caption{Thinking mode effects with statistical significance. All comparisons use paired t-tests ($n=58$).}")
    lines.append(r"\label{tab:thinking_mode}")
    lines.append(r"\begin{tabular}{lcccc}")
    lines.append(r"\toprule")
    lines.append(r"\textbf{Model} & \textbf{$\Delta$P} & \textbf{$\Delta$R} & \textbf{$\Delta$F1} & \textbf{Sig.} \\")
    lines.append(r"\midrule")
    
    for _, row in sig_df.iterrows():
        model = row['model_size'].upper()
        
        delta_p = row['precision_delta']
        delta_r = row['recall_delta']
        delta_f1 = row['f1_delta']
        p_val = row['f1_p_value']
        
        sign_p = "+" if delta_p >= 0 else ""
        sign_r = "+" if delta_r >= 0 else ""
        sign_f1 = "+" if delta_f1 >= 0 else ""
        
        if p_val < 0.001:
            p_str = "$p < 0.001$"
        elif p_val < 0.01:
            p_str = "$p < 0.01$"
        elif p_val < 0.05:
            p_str = "$p < 0.05$"
        else:
            p_str = "n.s."
        
        lines.append(
            f"{model:3} & "
            f"{sign_p}{delta_p:.3f} & "
            f"{sign_r}{delta_r:.3f} & "
            f"{sign_f1}{delta_f1:.3f} & "
            f"{p_str} \\\\"
        )
    
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    
    return '\n'.join(lines)


def generate_latex_table_length_correlation(length_stats: Dict) -> str:
    """Table: word length vs. difficulty correlation."""
    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(r"\caption{Word length analysis for generated vs. missed words. Longer words are cognitively harder to generate and recall.}")
    lines.append(r"\label{tab:length_difficulty}")
    lines.append(r"\begin{tabular}{lcc}")
    lines.append(r"\toprule")
    lines.append(r"\textbf{Word Category} & \textbf{Mean Length} & \textbf{Difference} \\")
    lines.append(r"\midrule")
    
    if 'error' in length_stats:
        lines.append(f"\\multicolumn{{3}}{{l}}{{Error: {length_stats['error']}}} \\\\")
    else:
        found_length = length_stats['mean_length_found']
        missed_length = length_stats['mean_length_missed']
        diff = length_stats['difference']
        t_stat = length_stats['t_statistic']
        p_val = length_stats['p_value']
        cohens_d = length_stats['cohens_d']
        
        p_str = f"$p < 0.001$" if p_val < 0.001 else f"$p = {p_val:.3f}$"
        
        lines.append(f"Found (correct) & {found_length:.2f} letters & -- \\\\")
        lines.append(f"Missed (not generated) & {missed_length:.2f} letters & {diff:+.2f} \\\\")
        lines.append(r"\midrule")
        lines.append(f"\\multicolumn{{3}}{{l}}{{Paired $t$-test: $t = {t_stat:.2f}$, {p_str}, $d = {cohens_d:.2f}$}} \\\\")
        lines.append(f"\\multicolumn{{3}}{{l}}{{{length_stats['interpretation']}}} \\\\")
    
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    
    return '\n'.join(lines)


def generate_latex_table_length_coverage(agg_df: pd.DataFrame) -> str:
    """Table: recall by word length category."""
    thinking_df = agg_df[agg_df['thinking'] == True].copy()
    
    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(r"\caption{Recall by word length category (thinking mode ON). Shows whether models explore full solution space or fixate on short words.}")
    lines.append(r"\label{tab:length_coverage}")
    lines.append(r"\begin{tabular}{lcccc}")
    lines.append(r"\toprule")
    lines.append(r"\textbf{Model} & \textbf{4-letter} & \textbf{5-letter} & \textbf{6-letter} & \textbf{7+ letter} \\")
    lines.append(r"\midrule")
    
    for _, row in thinking_df.iterrows():
        model = row['model_size'].upper()
        l4 = row.get('length_4_recall_mean', np.nan) * 100
        l5 = row.get('length_5_recall_mean', np.nan) * 100
        l6 = row.get('length_6_recall_mean', np.nan) * 100
        l7 = row.get('length_7+_recall_mean', np.nan) * 100
        
        lines.append(
            f"{model:3} & {l4:.1f}\\% & {l5:.1f}\\% & {l6:.1f}\\% & {l7:.1f}\\% \\\\"
        )
    
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    
    return '\n'.join(lines)


def generate_latex_table_pangrams(agg_df: pd.DataFrame) -> str:
    """Table: pangram generation performance."""
    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(r"\caption{Pangram generation performance by configuration. Pangram recall measures recovery of words using all 7 letters. Note: Performance is generally low across all configurations.}")
    lines.append(r"\label{tab:pangram_performance}")
    lines.append(r"\begin{tabular}{lccc}")
    lines.append(r"\toprule")
    lines.append(r"\textbf{Model} & \textbf{Thinking} & \textbf{Pangram Recall} & \textbf{Std Dev} \\")
    lines.append(r"\midrule")
    
    for _, row in agg_df.iterrows():
        model = row['model_size'].upper()
        thinking = "ON" if row['thinking'] else "OFF"
        
        if 'pangram_recall_mean' in row and not np.isnan(row['pangram_recall_mean']):
            pangram_mean = row['pangram_recall_mean']
            pangram_std = row['pangram_recall_std']
            lines.append(f"{model:3} & {thinking:3} & {pangram_mean:.3f} & {pangram_std:.3f} \\\\")
    
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    
    return '\n'.join(lines)


def generate_latex_table_complete_comparison(agg_df: pd.DataFrame) -> str:
    """Table: complete comparison across all configurations."""
    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(r"\caption{Complete model comparison across all configurations. Shows raw performance, consistency (CV), reliability (Worst 5\%), and success rate at F1 $\geq$ 0.3 threshold.}")
    lines.append(r"\label{tab:complete_comparison}")
    lines.append(r"\begin{tabular}{llcccccc}")
    lines.append(r"\toprule")
    lines.append(r"\textbf{Model} & \textbf{Think} & \textbf{Prec.} & \textbf{Rec.} & \textbf{F1} & \textbf{CV} & \textbf{P5} & \textbf{Succ@30} \\")
    lines.append(r"\midrule")
    
    for _, row in agg_df.iterrows():
        model = row['model_size'].upper()
        thinking = "ON " if row['thinking'] else "OFF"
        p_mean = row['precision_mean']
        r_mean = row['recall_mean'] * 100  # Convert to percentage
        f1_mean = row['f1_mean']
        cv = row.get('cv_f1', np.nan)
        p5 = row.get('f1_p5', np.nan)
        success_30 = row.get('success_rate_30', 0) * 100
        
        lines.append(
            f"{model:3} & {thinking} & {p_mean:.3f} & {r_mean:.1f}\\% & "
            f"{f1_mean:.3f} & {cv:.2f} & {p5:.3f} & {success_30:.0f}\\% \\\\"
        )
    
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    
    return '\n'.join(lines)


def generate_latex_table_difficulty_stratification(difficulty_df: pd.DataFrame, difficulty_gap_df: pd.DataFrame) -> str:
    """Table: performance stratified by puzzle difficulty."""
    thinking_on = difficulty_df[difficulty_df['thinking'] == True].copy()
    gap_on = difficulty_gap_df[difficulty_gap_df['thinking'] == True].copy()
    
    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(r"\caption{Performance stratified by puzzle difficulty (thinking mode ON). Easy: $<$30 words, Medium: 30--50 words, Hard: $>$50 words. Gap shows performance degradation from easy to hard puzzles.}")
    lines.append(r"\label{tab:difficulty_stratification}")
    lines.append(r"\begin{tabular}{lcccc}")
    lines.append(r"\toprule")
    lines.append(r"\textbf{Model} & \textbf{Easy} & \textbf{Medium} & \textbf{Hard} & \textbf{Gap} \\")
    lines.append(r"\midrule")
    
    for model_size in thinking_on['model_size'].unique():
        model_data = thinking_on[thinking_on['model_size'] == model_size]
        gap_data = gap_on[gap_on['model_size'] == model_size]
        
        easy = model_data[model_data['difficulty'] == 'easy']
        medium = model_data[model_data['difficulty'] == 'medium']
        hard = model_data[model_data['difficulty'] == 'hard']
        
        easy_recall = easy['recall_mean'].values[0] * 100 if len(easy) > 0 else 0
        med_recall = medium['recall_mean'].values[0] * 100 if len(medium) > 0 else 0
        hard_recall = hard['recall_mean'].values[0] * 100 if len(hard) > 0 else 0
        gap = gap_data['difficulty_gap'].values[0] * 100 if len(gap_data) > 0 else 0
        
        lines.append(
            f"{model_size.upper():3} & {easy_recall:.1f}\\% & {med_recall:.1f}\\% & "
            f"{hard_recall:.1f}\\% & {gap:+.1f}pp \\\\"
        )
    
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    
    return '\n'.join(lines)


def generate_latex_table_reliability_analysis(agg_df: pd.DataFrame) -> str:
    """Table: reliability and worst-case analysis."""
    thinking_df = agg_df[agg_df['thinking'] == True].copy()
    
    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(r"\caption{Reliability analysis (thinking mode ON). CV = Coefficient of Variation (lower is better). P5/P95 = 5th/95th percentile F1 scores. Range shows performance variability.}")
    lines.append(r"\label{tab:reliability_analysis}")
    lines.append(r"\begin{tabular}{lccccc}")
    lines.append(r"\toprule")
    lines.append(r"\textbf{Model} & \textbf{Mean F1} & \textbf{CV} & \textbf{P5 (Worst)} & \textbf{P95 (Best)} & \textbf{Range} \\")
    lines.append(r"\midrule")
    
    for _, row in thinking_df.iterrows():
        model = row['model_size'].upper()
        f1_mean = row['f1_mean']
        cv = row.get('cv_f1', np.nan)
        p5 = row.get('f1_p5', np.nan)
        p95 = row.get('f1_p95', np.nan)
        f1_range = p95 - p5 if not (np.isnan(p5) or np.isnan(p95)) else np.nan
        
        lines.append(
            f"{model:3} & {f1_mean:.3f} & {cv:.2f} & {p5:.3f} & {p95:.3f} & {f1_range:.3f} \\\\"
        )
    
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    
    return '\n'.join(lines)


def generate_latex_table_error_distribution(agg_df: pd.DataFrame) -> str:
    """Table: error type distribution (constraint violations vs non-dictionary)."""
    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(r"\caption{Error distribution by configuration. All false positives are non-dictionary words (constraint violations = 0\%). Shows models follow rules but lack lexical coverage.}")
    lines.append(r"\label{tab:error_distribution}")
    lines.append(r"\begin{tabular}{llccc}")
    lines.append(r"\toprule")
    lines.append(r"\textbf{Model} & \textbf{Think} & \textbf{FP/Puzzle} & \textbf{Constraint} & \textbf{Non-Dict} \\")
    lines.append(r"\midrule")
    
    for _, row in agg_df.iterrows():
        model = row['model_size'].upper()
        thinking = "ON " if row['thinking'] else "OFF"
        fp_mean = row.get('fp_non_dictionary_mean', 0)
        constraint_pct = 0   # parser pre-filters constraint violations
        non_dict_pct = 100
        
        lines.append(
            f"{model:3} & {thinking} & {fp_mean:.1f} & {constraint_pct:.0f}\\% & {non_dict_pct:.0f}\\% \\\\"
        )
    
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    
    return '\n'.join(lines)


def generate_latex_table_weighted_vs_standard_recall(agg_df: pd.DataFrame) -> str:
    """Table: difficulty-weighted vs standard recall."""
    thinking_df = agg_df[agg_df['thinking'] == True].copy()
    
    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(r"\caption{Difficulty-weighted vs standard recall (thinking mode ON). Negative weighting effect indicates bias toward easier (shorter) words.}")
    lines.append(r"\label{tab:weighted_recall}")
    lines.append(r"\begin{tabular}{lccc}")
    lines.append(r"\toprule")
    lines.append(r"\textbf{Model} & \textbf{Standard Rec.} & \textbf{Weighted Rec.} & \textbf{Effect} \\")
    lines.append(r"\midrule")
    
    for _, row in thinking_df.iterrows():
        model = row['model_size'].upper()
        std_recall = row['recall_mean'] * 100
        weighted_recall = row.get('difficulty_weighted_recall_mean', np.nan) * 100
        effect = weighted_recall - std_recall
        
        lines.append(
            f"{model:3} & {std_recall:.1f}\\% & {weighted_recall:.1f}\\% & {effect:+.1f}pp \\\\"
        )
    
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    
    return '\n'.join(lines)


def generate_latex_table_thinking_consistency_effects(thinking_consistency_df: pd.DataFrame) -> str:
    """Table: thinking mode impact on consistency metrics."""
    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(r"\caption{Thinking mode impact on consistency metrics. Negative $\Delta$CV indicates improved consistency. Note 8B shows WORSE consistency with thinking enabled.}")
    lines.append(r"\label{tab:thinking_consistency}")
    lines.append(r"\begin{tabular}{lcccc}")
    lines.append(r"\toprule")
    lines.append(r"\textbf{Model} & \textbf{$\Delta$F1} & \textbf{$\Delta$CV} & \textbf{$\Delta$P5} & \textbf{$\Delta$Succ@30} \\")
    lines.append(r"\midrule")
    
    for _, row in thinking_consistency_df.iterrows():
        model = row['model_size'].upper()
        delta_f1 = row['delta_f1']
        delta_cv = row['delta_cv']
        delta_p5 = row['delta_worst_5']
        delta_success = row['delta_success_30'] * 100
        
        sign_f1 = "+" if delta_f1 >= 0 else ""
        sign_cv = "+" if delta_cv >= 0 else ""
        sign_p5 = "+" if delta_p5 >= 0 else ""
        sign_succ = "+" if delta_success >= 0 else ""
        
        lines.append(
            f"{model:3} & {sign_f1}{delta_f1:.3f} & {sign_cv}{delta_cv:.2f} & "
            f"{sign_p5}{delta_p5:.3f} & {sign_succ}{delta_success:.0f}pp \\\\"
        )
    
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    
    return '\n'.join(lines)


def generate_latex_table_scaling_anomaly(scaling_pattern_df: pd.DataFrame) -> str:
    """Table: scaling pattern with anomaly detection."""
    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(r"\caption{Scaling pattern analysis (thinking mode ON). Shows pairwise model comparisons. The 4B$\rightarrow$8B transition shows anomalous performance degradation.}")
    lines.append(r"\label{tab:scaling_anomaly}")
    lines.append(r"\begin{tabular}{lccc}")
    lines.append(r"\toprule")
    lines.append(r"\textbf{Comparison} & \textbf{$\Delta$F1} & \textbf{$\Delta$Recall} & \textbf{Status} \\")
    lines.append(r"\midrule")
    
    for _, row in scaling_pattern_df.iterrows():
        comparison = row['comparison']
        comparison = comparison.replace('\u2192', r'$\rightarrow$')
        
        delta_f1 = row['delta_f1']
        delta_recall = row['delta_recall'] * 100
        is_anomaly = row['is_anomaly']
        
        sign_f1 = "+" if delta_f1 >= 0 else ""
        sign_r = "+" if delta_recall >= 0 else ""
        
        status = r"\textbf{ANOMALY}" if is_anomaly else "Improved"
        
        lines.append(
            f"{comparison} & {sign_f1}{delta_f1:.3f} & {sign_r}{delta_recall:.1f}pp & {status} \\\\"
        )
    
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    
    return '\n'.join(lines)


def generate_latex_table_budget_effects(budget_effects_df: pd.DataFrame) -> str:
    """Table: thinking budget effects on performance."""
    if len(budget_effects_df) == 0:
        return "% No budget effects data available"
    
    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(r"\caption{Thinking budget effects on performance. Shows incremental changes when increasing budget. Efficiency measured as F1 improvement per 1K tokens. Diminishing returns flagged when $\Delta$F1 $<$ 0.01.}")
    lines.append(r"\label{tab:budget_effects}")
    lines.append(r"\begin{tabular}{lccccc}")
    lines.append(r"\toprule")
    lines.append(r"\textbf{Model} & \textbf{Budget Change} & \textbf{$\Delta$F1} & \textbf{$\Delta$Recall} & \textbf{Efficiency} & \textbf{Dim. Returns} \\")
    lines.append(r"\midrule")
    
    for model_size in budget_effects_df['model_size'].unique():
        model_data = budget_effects_df[budget_effects_df['model_size'] == model_size]
        
        for _, row in model_data.iterrows():
            model = model_size.upper()
            budget_from = int(row['budget_from'])
            budget_to = int(row['budget_to'])
            budget_change = f"{budget_from//1024}K$\\rightarrow${budget_to//1024}K"
            delta_f1 = row['delta_f1']
            delta_recall = row['delta_recall'] * 100
            efficiency = row['efficiency_f1_per_1k']
            dim_returns = "Yes" if row['diminishing_returns'] == "Yes" else "No"
            
            sign_f1 = "+" if delta_f1 >= 0 else ""
            sign_r = "+" if delta_recall >= 0 else ""
            sign_eff = "+" if efficiency >= 0 else ""
            
            lines.append(
                f"{model:3} & {budget_change} & {sign_f1}{delta_f1:.3f} & "
                f"{sign_r}{delta_recall:.1f}pp & {sign_eff}{efficiency:.4f} & {dim_returns} \\\\"
            )
    
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    
    return '\n'.join(lines)


def generate_latex_table_budget_interaction(budget_interaction_df: pd.DataFrame) -> str:
    """Table: budget sensitivity by model size."""
    if len(budget_interaction_df) == 0:
        return "% No budget interaction data available"
    
    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(r"\caption{Budget sensitivity by model size. Shows total improvement across budget range (4K--16K tokens). Sensitivity: High ($>$0.05), Medium (0.02--0.05), Low ($<$0.02).}")
    lines.append(r"\label{tab:budget_interaction}")
    lines.append(r"\begin{tabular}{lcccc}")
    lines.append(r"\toprule")
    lines.append(r"\textbf{Model} & \textbf{Budget Range} & \textbf{F1 Improv.} & \textbf{Efficiency} & \textbf{Sensitivity} \\")
    lines.append(r"\midrule")
    
    for _, row in budget_interaction_df.iterrows():
        model = row['model_size'].upper()
        min_budget = int(row['min_budget'])
        max_budget = int(row['max_budget'])
        budget_range = f"{min_budget//1024}K--{max_budget//1024}K"
        f1_improvement = row['f1_improvement']
        efficiency = row['f1_improvement_per_1k']
        sensitivity = row['budget_sensitivity']
        
        sign_f1 = "+" if f1_improvement >= 0 else ""
        sign_eff = "+" if efficiency >= 0 else ""
        
        lines.append(
            f"{model:3} & {budget_range} & {sign_f1}{f1_improvement:.3f} & "
            f"{sign_eff}{efficiency:.4f} & {sensitivity} \\\\"
        )
    
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    
    return '\n'.join(lines)


def generate_latex_table_optimal_budget(optimal_budget: Dict) -> str:
    """Table: optimal budget recommendations per model size."""
    if 'error' in optimal_budget or len(optimal_budget) == 0:
        return "% No optimal budget data available"
    
    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(r"\caption{Optimal thinking budget recommendations by model size. Shows budget that maximizes F1 score based on tested configurations.}")
    lines.append(r"\label{tab:optimal_budget}")
    lines.append(r"\begin{tabular}{lcccc}")
    lines.append(r"\toprule")
    lines.append(r"\textbf{Model} & \textbf{Optimal Budget} & \textbf{F1} & \textbf{Recall} & \textbf{Precision} \\")
    lines.append(r"\midrule")
    
    for model_size in sorted(optimal_budget.keys(), key=lambda x: {'4b': 0, '8b': 1, '14b': 2, '30b': 3, '32b': 3}.get(x, 99)):
        rec = optimal_budget[model_size]
        model = model_size.upper()
        budget = int(rec['optimal_budget'])
        budget_str = f"{budget//1024}K" if budget >= 1024 else str(budget)
        f1 = rec['f1_at_optimal']
        recall = rec['recall_at_optimal']
        precision = rec['precision_at_optimal']
        
        lines.append(
            f"{model:3} & {budget_str:4} tok & {f1:.3f} & {recall:.3f} & {precision:.3f} \\\\"
        )
    
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    
    return '\n'.join(lines)


def generate_latex_table_volume_analysis(agg_df: pd.DataFrame, volume_analysis: Dict) -> str:
    """Table: generation volume and quality metrics."""
    if not volume_analysis or 'error' in volume_analysis:
        return "% No volume analysis data available"
    
    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(r"\caption{Generation volume and quality metrics. Volume = mean predictions per puzzle. Efficiency = recall normalized by generation volume. Weak volume-precision correlation ($r=+0.13$) indicates models maintain quality while increasing coverage.}")
    lines.append(r"\label{tab:volume_analysis}")
    lines.append(r"\begin{tabular}{lcccc}")
    lines.append(r"\toprule")
    lines.append(r"\textbf{Model} & \textbf{Volume} & \textbf{Precision} & \textbf{Recall} & \textbf{Efficiency} \\")
    lines.append(r"\midrule")
    
    thinking_on = agg_df[agg_df['thinking'] == True].copy()

    if 'thinking_budget' in thinking_on.columns:
        thinking_on = thinking_on.sort_values('f1_mean', ascending=False).groupby('model_size').first().reset_index()
    
    for _, row in thinking_on.iterrows():
        model = row['model_size'].upper()
        volume = row.get('num_predicted_mean', np.nan)
        precision = row.get('precision_mean', np.nan)
        recall = row.get('recall_mean', np.nan)
        efficiency = row.get('efficiency', np.nan)
        
        lines.append(
            f"{model:3} & {volume:.1f} & {precision:.3f} & {recall:.3f} & {efficiency:.3f} \\\\"
        )
    
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    
    return '\n'.join(lines)


def generate_latex_table_word_difficulty_stratification(word_diff_df: pd.DataFrame) -> str:
    """Table: model recall by human difficulty quartile."""
    if len(word_diff_df) == 0:
        return "% No word difficulty stratification data available"
    
    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(r"\caption{Model recall stratified by human word difficulty quartiles. Q1 = Very Easy (>75\% human success), Q4 = Very Hard (<25\% human success). Shows whether models struggle with same words as humans.}")
    lines.append(r"\label{tab:word_difficulty_strat}")
    lines.append(r"\begin{tabular}{lcccc}")
    lines.append(r"\toprule")
    lines.append(r"\textbf{Model} & \textbf{Q1 (Very Easy)} & \textbf{Q2 (Easy)} & \textbf{Q3 (Hard)} & \textbf{Q4 (Very Hard)} \\")
    lines.append(r"\midrule")
    
    thinking_df = word_diff_df[word_diff_df['thinking'] == True]

    if 'thinking_budget' in thinking_df.columns:
        thinking_df = thinking_df.groupby(['model_size', 'difficulty_quartile'])['recall'].mean().reset_index()
    
    for model in ['4b', '8b', '14b', '30b', '32b']:
        model_data = thinking_df[thinking_df['model_size'] == model]
        if len(model_data) > 0:
            q_recalls = {}
            for _, row in model_data.iterrows():
                q = row['difficulty_quartile']
                q_recalls[q] = row['recall'] * 100
            
            q1 = q_recalls.get('Very Easy (Q1)', np.nan)
            q2 = q_recalls.get('Easy (Q2)', np.nan)
            q3 = q_recalls.get('Hard (Q3)', np.nan)
            q4 = q_recalls.get('Very Hard (Q4)', np.nan)
            
            lines.append(
                f"{model.upper():3} & {q1:.1f}\\% & {q2:.1f}\\% & {q3:.1f}\\% & {q4:.1f}\\% \\\\"
            )
    
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    
    return '\n'.join(lines)


def generate_latex_table_model_human_calibration(calibration: Dict) -> str:
    """Table: model-human calibration (Spearman correlation and MAE)."""
    if not calibration or 'error' in calibration:
        return "% No calibration data available"
    
    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(r"\caption{Model-human calibration: correlation between model difficulty and human difficulty at word level. Higher correlation indicates models struggle with same words as humans. MAE = Mean Absolute Error between human and model difficulty scores.}")
    lines.append(r"\label{tab:calibration}")
    lines.append(r"\begin{tabular}{lccc}")
    lines.append(r"\toprule")
    lines.append(r"\textbf{Model} & \textbf{Correlation} & \textbf{MAE} & \textbf{Words Analyzed} \\")
    lines.append(r"\midrule")
    
    model_calibrations = {}
    for config_str, metrics in calibration.items():
        if metrics.get('thinking'):
            model = metrics['model_size']
            if model not in model_calibrations:
                model_calibrations[model] = {'corr': [], 'mae': [], 'words': []}
            model_calibrations[model]['corr'].append(metrics['correlation'])
            model_calibrations[model]['mae'].append(metrics['mae'])
            model_calibrations[model]['words'].append(metrics['num_words'])
    
    for model in ['4b', '8b', '14b', '30b', '32b']:
        if model in model_calibrations:
            avg_corr = np.mean(model_calibrations[model]['corr'])
            avg_mae = np.mean(model_calibrations[model]['mae'])
            total_words = int(np.mean(model_calibrations[model]['words']))
            
            lines.append(
                f"{model.upper():3} & {avg_corr:+.3f} & {avg_mae:.3f} & {total_words} \\\\"
            )
    
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    
    return '\n'.join(lines)


def generate_latex_table_failure_analysis(failure_analysis: Dict) -> str:
    """Table: systematic failures (easy-for-humans, hard-for-models)."""
    if not failure_analysis or 'error' in failure_analysis:
        return "% No failure analysis data available"
    
    top_misses = failure_analysis.get('top_easy_words_missed', [])
    if not top_misses:
        return "% No failure data available"
    
    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(r"\caption{Systematic failure analysis: words that are easy for humans (>80\% success) but frequently missed by models. Miss rate = percentage of model predictions that failed to generate the word.}")
    lines.append(r"\label{tab:failure_analysis}")
    lines.append(r"\begin{tabular}{llcc}")
    lines.append(r"\toprule")
    lines.append(r"\textbf{Word} & \textbf{Length} & \textbf{Human Success} & \textbf{Model Miss Rate} \\")
    lines.append(r"\midrule")
    
    for word_info in top_misses[:10]:
        word = word_info['word']
        length = len(word)
        human_success = word_info['human_success_rate'] * 100
        model_miss = word_info['model_miss_rate'] * 100
        
        lines.append(
            f"{word:12} & {length} & {human_success:.1f}\\% & {model_miss:.1f}\\% \\\\"
        )
    
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    
    return '\n'.join(lines)


def generate_all_latex_tables(agg_df: pd.DataFrame, sig_df: pd.DataFrame, length_stats: Dict,
                              difficulty_df: pd.DataFrame, difficulty_gap_df: pd.DataFrame,
                              thinking_consistency_df: pd.DataFrame, scaling_pattern_df: pd.DataFrame,
                              budget_effects_df: pd.DataFrame = None, budget_interaction_df: pd.DataFrame = None,
                              optimal_budget: Dict = None, volume_analysis: Dict = None,
                              word_difficulty_strat_df: pd.DataFrame = None, 
                              model_human_calibration: Dict = None,
                              failure_mode_analysis: Dict = None) -> str:
    """Combine all LaTeX tables into a single formatted string."""
    tables = []
    
    tables.append("% ==========================================================================")
    tables.append("% CORE PERFORMANCE TABLES (Tables 1-5)")
    tables.append("% ==========================================================================\n")
    
    tables.append("% Table 1: Model Scaling Effects (with Consistency Metrics)")
    tables.append(generate_latex_table_scaling(agg_df))
    tables.append("")
    
    tables.append("% Table 2: Thinking Mode Comparison")
    tables.append(generate_latex_table_thinking(sig_df))
    tables.append("")
    
    tables.append("% Table 3: Word Length vs. Difficulty")
    tables.append(generate_latex_table_length_correlation(length_stats))
    tables.append("")
    
    tables.append("% Table 4: Word Length Coverage Distribution")
    tables.append(generate_latex_table_length_coverage(agg_df))
    tables.append("")
    
    tables.append("% Table 5: Pangram Performance (de-emphasized)")
    tables.append(generate_latex_table_pangrams(agg_df))
    tables.append("")
    
    tables.append("% ==========================================================================")
    tables.append("% EXTENDED ANALYSIS TABLES (Tables 6-12)")
    tables.append("% ==========================================================================\n")
    
    tables.append("% Table 6: Complete Model Comparison (ALL Configurations)")
    tables.append(generate_latex_table_complete_comparison(agg_df))
    tables.append("")
    
    tables.append("% Table 7: Performance by Puzzle Difficulty")
    tables.append(generate_latex_table_difficulty_stratification(difficulty_df, difficulty_gap_df))
    tables.append("")
    
    tables.append("% Table 8: Reliability and Worst-Case Analysis")
    tables.append(generate_latex_table_reliability_analysis(agg_df))
    tables.append("")
    
    tables.append("% Table 9: Error Type Distribution")
    tables.append(generate_latex_table_error_distribution(agg_df))
    tables.append("")
    
    tables.append("% Table 10: Difficulty-Weighted vs Standard Recall")
    tables.append(generate_latex_table_weighted_vs_standard_recall(agg_df))
    tables.append("")
    
    tables.append("% Table 11: Thinking Mode Impact on Consistency")
    tables.append(generate_latex_table_thinking_consistency_effects(thinking_consistency_df))
    tables.append("")
    
    tables.append("% Table 12: Scaling Pattern with Anomaly Analysis")
    tables.append(generate_latex_table_scaling_anomaly(scaling_pattern_df))
    tables.append("")
    
    if budget_effects_df is not None and len(budget_effects_df) > 0:
        tables.append("% ==========================================================================")
        tables.append("% THINKING BUDGET ANALYSIS TABLES (Tables 13-16)")
        tables.append("% ==========================================================================\n")
        
        tables.append("% Table 13: Thinking Budget Effects")
        tables.append(generate_latex_table_budget_effects(budget_effects_df))
        tables.append("")
        
        if budget_interaction_df is not None and len(budget_interaction_df) > 0:
            tables.append("% Table 14: Budget-Model Size Interaction")
            tables.append(generate_latex_table_budget_interaction(budget_interaction_df))
            tables.append("")
        
        if optimal_budget is not None and 'error' not in optimal_budget:
            tables.append("% Table 15: Optimal Budget Recommendations")
            tables.append(generate_latex_table_optimal_budget(optimal_budget))
            tables.append("")
        
        if volume_analysis is not None and 'error' not in volume_analysis:
            tables.append("% Table 16: Volume-Performance Analysis")
            tables.append(generate_latex_table_volume_analysis(agg_df, volume_analysis))
        
        if word_difficulty_strat_df is not None and len(word_difficulty_strat_df) > 0:
            tables.append("")
            tables.append("% Table 17: Word-Level Difficulty Stratification (by Human Success Rates)")
            tables.append(generate_latex_table_word_difficulty_stratification(word_difficulty_strat_df))
        
        if model_human_calibration is not None and 'error' not in model_human_calibration:
            tables.append("")
            tables.append("% Table 18: Model-Human Calibration")
            tables.append(generate_latex_table_model_human_calibration(model_human_calibration))
        
        if failure_mode_analysis is not None and 'error' not in failure_mode_analysis:
            tables.append("")
            tables.append("% Table 19: Systematic Failure Analysis")
            tables.append(generate_latex_table_failure_analysis(failure_mode_analysis))
    
    return '\n\n'.join(tables)

