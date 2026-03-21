"""Generate publication figures for the LREC 2026 Spelling Bee paper."""

import json
import argparse
from pathlib import Path
from typing import Dict
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif', 'Liberation Serif'],
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 13,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
    'axes.spines.top': False,
    'axes.spines.right': False,
})


def _desaturate(hex_color: str, factor: float = 0.30) -> str:
    """Blend a hex color toward its luminance gray."""
    r, g, b = int(hex_color[1:3], 16), int(hex_color[3:5], 16), int(hex_color[5:7], 16)
    gray = 0.299 * r + 0.587 * g + 0.114 * b
    r2 = int(max(0, min(255, r + factor * (gray - r))))
    g2 = int(max(0, min(255, g + factor * (gray - g))))
    b2 = int(max(0, min(255, b + factor * (gray - b))))
    return f'#{r2:02x}{g2:02x}{b2:02x}'


# Okabe-Ito palette, desaturated for print
_RAW_COLORS = {
    '4b': '#0173B2', '8b': '#DE8F05', '14b': '#029E73',
    '30b': '#CC78BC', '32b': '#CA0020',
    'claude-haiku': '#56B4E9', 'gpt5-mini': '#5D3A9B',
}
COLORS = {k: _desaturate(v) for k, v in _RAW_COLORS.items()}
COLORS.update({
    '4096': COLORS['4b'], '8192': COLORS['14b'],
    '12288': COLORS['8b'], '16384': COLORS['32b'],
    'thinking_on': COLORS['14b'], 'thinking_off': '#949494',
})

# Grayscale-safe markers, linestyles, hatching
MARKERS = {
    '4b': 'o', '8b': 's', '14b': '^', '30b': 'D', '32b': 'v',
    'claude-haiku': 'P', 'gpt5-mini': 'X',
    '4096': 'o', '8192': 's', '12288': 'D', '16384': '^',
}
LINESTYLES = {
    '4b': '-', '8b': '--', '14b': '-.', '30b': ':', '32b': '-',
    'claude-haiku': '--', 'gpt5-mini': '-.',
    '4096': '-', '8192': '--', '12288': ':', '16384': '-.',
}
HATCHES = {
    '4b': '', '8b': '//', '14b': '\\\\', '30b': 'xx', '32b': '..',
    'claude-haiku': '||', 'gpt5-mini': '--',
}
BUDGET_HATCHES = {'4096': '', '8192': '//', '12288': 'xx', '16384': '\\\\'}

MODEL_ORDER = ['4b', '8b', '14b', '30b', '32b', 'claude-haiku', 'gpt5-mini']
MODEL_LABELS = {
    '4b': 'Qwen-4B', '8b': 'Qwen-8B', '14b': 'Qwen-14B', '30b': 'Qwen-30B', '32b': 'Qwen-32B',
    'claude-haiku': 'Claude-Haiku', 'gpt5-mini': 'GPT-5-mini',
}
BUDGET_LABELS = {4096: '4K', 8192: '8K', 12288: '12K', 16384: '16K'}


def load_metrics_summary(metrics_dir: str) -> Dict:
    json_path = Path(metrics_dir) / 'metrics_summary.json'
    if not json_path.exists():
        raise FileNotFoundError(f"Metrics summary not found: {json_path}")
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_aggregated_metrics(metrics_dir: str) -> pd.DataFrame:
    csv_path = Path(metrics_dir) / 'aggregated_metrics.csv'
    if not csv_path.exists():
        raise FileNotFoundError(f"Aggregated metrics not found: {csv_path}")
    return pd.read_csv(csv_path)


def load_detailed_metrics(metrics_dir: str) -> pd.DataFrame:
    csv_path = Path(metrics_dir) / 'detailed_metrics.csv'
    if not csv_path.exists():
        raise FileNotFoundError(f"Detailed metrics not found: {csv_path}")
    return pd.read_csv(csv_path)


def _save_figure(output_dir: Path, name: str):
    plt.tight_layout()
    plt.savefig(output_dir / f'{name}.pdf', bbox_inches='tight')
    plt.savefig(output_dir / f'{name}.png', bbox_inches='tight')
    plt.close()


def _sort_by_model_order(df: pd.DataFrame) -> pd.DataFrame:
    order_map = {model: idx for idx, model in enumerate(MODEL_ORDER)}
    return df.sort_values('model_size', key=lambda x: x.map(order_map))


# Figure 1: Model & Budget Scaling

def plot_model_budget_scaling(agg_df: pd.DataFrame, output_dir: Path):
    """F1 scaling across model sizes for different thinking budgets."""
    thinking_on = agg_df[agg_df['thinking'] == True].copy()
    has_budget = 'thinking_budget' in thinking_on.columns and thinking_on['thinking_budget'].notna().any()

    fig, ax = plt.subplots(figsize=(7, 4.5))

    if has_budget:
        budgets = sorted(thinking_on['thinking_budget'].dropna().unique())
        for budget in budgets:
            budget_data = _sort_by_model_order(thinking_on[thinking_on['thinking_budget'] == budget].copy())
            models = [MODEL_LABELS.get(m, m.upper()) for m in budget_data['model_size']]
            x = np.arange(len(models))
            budget_key = str(int(budget))
            color = COLORS.get(budget_key, '#000000')
            label = BUDGET_LABELS.get(int(budget), f'{int(budget)} tokens')
            ax.plot(x, budget_data['f1_mean'].values,
                    marker=MARKERS.get(budget_key, 'o'),
                    linestyle=LINESTYLES.get(budget_key, '-'),
                    markersize=7, linewidth=2, color=color, label=label)
        ax.set_xticks(x)
        ax.set_xticklabels(models)
        ax.set_title('Model Scaling Across Thinking Budgets', fontweight='bold', fontsize=14)
    else:
        thinking_on = _sort_by_model_order(thinking_on)
        models = [MODEL_LABELS.get(m, m.upper()) for m in thinking_on['model_size']]
        x = np.arange(len(models))
        ax.plot(x, thinking_on['f1_mean'].values, marker='o', markersize=8,
                linewidth=2.5, color=COLORS['thinking_on'])
        ax.set_xticks(x)
        ax.set_xticklabels(models)
        ax.set_title('Model Scaling Performance', fontweight='bold', fontsize=14)

    ax.set_ylabel('F1 Score', fontweight='bold', fontsize=13)
    ax.set_xlabel('Model Size', fontweight='bold', fontsize=13)
    ax.legend(loc='best', frameon=True, shadow=False)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_ylim(bottom=0)

    _save_figure(output_dir, 'fig1_model_budget_scaling')
    print("  Fig 1: Model & Budget Scaling")


# Figure 2: Reliability Box Plots

def plot_reliability_analysis(agg_df: pd.DataFrame, detailed_df: pd.DataFrame, output_dir: Path):
    """F1 distribution box plots per model (8K budget or optimal)."""
    thinking_on = detailed_df[detailed_df['thinking'] == True].copy()

    if 'thinking_budget' in thinking_on.columns:
        if 8192 in thinking_on['thinking_budget'].values:
            thinking_on = thinking_on[thinking_on['thinking_budget'] == 8192]
        else:
            optimal = agg_df[agg_df['thinking'] == True].copy()
            if 'thinking_budget' in optimal.columns:
                optimal = optimal.sort_values('f1_mean', ascending=False).groupby('model_size').first()
                filtered_data = []
                for model in MODEL_ORDER:
                    if model in optimal.index:
                        opt_budget = optimal.loc[model, 'thinking_budget']
                        filtered_data.append(thinking_on[
                            (thinking_on['model_size'] == model) &
                            (thinking_on['thinking_budget'] == opt_budget)
                        ])
                if filtered_data:
                    thinking_on = pd.concat(filtered_data)

    fig, ax = plt.subplots(figsize=(10, 6))

    data_to_plot, labels, colors_list = [], [], []
    for model in MODEL_ORDER:
        model_data = thinking_on[thinking_on['model_size'] == model]['f1'].values
        if len(model_data) > 0:
            data_to_plot.append(model_data)
            labels.append(MODEL_LABELS.get(model, model.upper()))
            colors_list.append(COLORS[model])

    if not data_to_plot:
        print("  Fig 2: SKIPPED (no data)")
        return

    bp = ax.boxplot(data_to_plot, tick_labels=labels, patch_artist=True, widths=0.6,
                    boxprops=dict(linewidth=1.5),
                    medianprops=dict(color='black', linewidth=2),
                    whiskerprops=dict(linewidth=1.5),
                    capprops=dict(linewidth=1.5))

    models_in_plot = [m for m in MODEL_ORDER
                      if m in detailed_df[detailed_df['thinking'] == True]['model_size'].values]
    for patch, color, model in zip(bp['boxes'], colors_list, models_in_plot):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
        patch.set_hatch(HATCHES.get(model, ''))

    ax.set_ylabel('F1 Score', fontweight='bold', fontsize=13)
    ax.set_xlabel('Model Size', fontweight='bold', fontsize=13)
    ax.set_title('F1 Score Distribution by Model', fontweight='bold', fontsize=14)
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    _save_figure(output_dir, 'fig2_reliability_box_plots')
    print("  Fig 2: Reliability Box Plots")


# Figure 3: Word Length vs Difficulty

def plot_word_length_difficulty(summary: Dict, output_dir: Path):
    """Mean word length for found vs missed words."""
    length_analysis = summary['length_difficulty_analysis']
    if 'error' in length_analysis:
        print("  Fig 3: SKIPPED (no length data)")
        return

    fig, ax = plt.subplots(figsize=(8, 6))

    categories = ['Correctly\nGenerated', 'Missed']
    lengths = [length_analysis['mean_length_found'], length_analysis['mean_length_missed']]
    colors = [_desaturate('#029E73'), _desaturate('#D55E00')]
    hatches = ['', '//']

    bars = ax.bar(categories, lengths, color=colors, alpha=0.8, edgecolor='black',
                  linewidth=1.5, width=0.5)
    for bar, hatch in zip(bars, hatches):
        bar.set_hatch(hatch)

    for bar, length in zip(bars, lengths):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.05,
                f'{length:.2f}', ha='center', va='bottom', fontweight='bold', fontsize=12)

    ax.set_ylabel('Mean Word Length (letters)', fontweight='bold', fontsize=13)
    ax.set_title('Word Length Distribution', fontweight='bold', fontsize=14)
    ax.set_ylim(0, max(lengths) * 1.2)
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    _save_figure(output_dir, 'fig3_word_length_difficulty')
    print("  Fig 3: Word Length vs Difficulty")


# Figure 4: Budget Effects

def plot_budget_effects(agg_df: pd.DataFrame, output_dir: Path):
    """F1 vs thinking budget for each model."""
    thinking_on = agg_df[agg_df['thinking'] == True].copy()

    if 'thinking_budget' not in thinking_on.columns or thinking_on['thinking_budget'].isna().all():
        print("  Fig 4: SKIPPED (no budget data)")
        return

    fig, ax = plt.subplots(figsize=(7, 4.5))

    for model in MODEL_ORDER:
        model_data = thinking_on[thinking_on['model_size'] == model].sort_values('thinking_budget')
        if len(model_data) == 0:
            continue
        budget_k = model_data['thinking_budget'].values / 1024
        f1_mean = model_data['f1_mean'].values
        ax.plot(budget_k, f1_mean,
                marker=MARKERS.get(model, 'o'),
                linestyle=LINESTYLES.get(model, '-'),
                markersize=7, linewidth=1.8, color=COLORS[model],
                label=MODEL_LABELS[model])

    ax.set_xlabel('Thinking Budget (K tokens)', fontweight='bold', fontsize=13)
    ax.set_ylabel('F1 Score', fontweight='bold', fontsize=13)
    ax.set_title('Budget Effects on Performance', fontweight='bold', fontsize=14)
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=True, shadow=False,
              title='Model', fontsize=10, title_fontsize=11)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_ylim(bottom=0)

    _save_figure(output_dir, 'fig4_budget_effects')
    print("  Fig 4: Budget Effects")


# Figure 5: Length Coverage Heatmap

def plot_length_coverage_heatmap(agg_df: pd.DataFrame, metrics_dir: str, output_dir: Path):
    """Model recall heatmap by word length, with optional human comparison panel."""
    thinking_on = agg_df[agg_df['thinking'] == True].copy()

    if 'thinking_budget' in thinking_on.columns:
        thinking_on = thinking_on.sort_values('f1_mean', ascending=False).groupby('model_size').first().reset_index()

    length_cats = ['4', '5', '6', '7+']
    data_matrix, models_in_data = [], []

    for model in MODEL_ORDER:
        model_data = thinking_on[thinking_on['model_size'] == model]
        if len(model_data) > 0:
            row = []
            for cat in length_cats:
                col = f'length_{cat}_recall_mean'
                value = model_data[col].values[0] if col in model_data.columns else 0
                row.append(value * 100)
            data_matrix.append(row)
            models_in_data.append(model)

    if not data_matrix:
        print("  Fig 5: SKIPPED (no length data)")
        return

    data_matrix = np.array(data_matrix)

    human_data_path = Path(metrics_dir) / 'human_difficulty_by_length.json'
    human_success_by_cat = None
    if human_data_path.exists():
        with open(human_data_path, 'r', encoding='utf-8') as f:
            human_success_by_cat = json.load(f).get('by_category', {})

    if human_success_by_cat:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    else:
        fig, ax1 = plt.subplots(figsize=(8, 6))

    im = ax1.imshow(data_matrix, cmap='cividis', aspect='auto', vmin=0, vmax=100)
    ax1.set_xticks(np.arange(len(length_cats)))
    ax1.set_yticks(np.arange(len(models_in_data)))
    ax1.set_xticklabels([f'{cat}-letter' for cat in length_cats], fontsize=11)
    ax1.set_yticklabels([MODEL_LABELS[m] for m in models_in_data], fontsize=11)

    for i in range(len(models_in_data)):
        for j in range(len(length_cats)):
            text_color = 'white' if data_matrix[i, j] < 50 else 'black'
            ax1.text(j, i, f'{data_matrix[i, j]:.1f}',
                     ha="center", va="center", color=text_color, fontweight='bold', fontsize=10)

    ax1.set_title('Model Recall by Word Length', fontweight='bold', fontsize=14)
    ax1.set_xlabel('Word Length', fontweight='bold', fontsize=13)
    ax1.set_ylabel('Model', fontweight='bold', fontsize=13)
    cbar1 = plt.colorbar(im, ax=ax1)
    cbar1.set_label('Recall (%)', rotation=270, labelpad=20, fontweight='bold')
    ax1.text(-0.08, 1.05, '(a)', transform=ax1.transAxes, fontsize=13, fontweight='bold')

    if human_success_by_cat:
        human_means, human_stds = [], []
        for cat in length_cats:
            if human_success_by_cat.get(cat):
                human_means.append(human_success_by_cat[cat]['mean'] * 100)
                human_stds.append(human_success_by_cat[cat]['std'] * 100)
            else:
                human_means.append(0)
                human_stds.append(0)

        x = np.arange(len(length_cats))
        bars = ax2.bar(x, human_means, yerr=human_stds, capsize=4,
                       color=COLORS.get('claude-haiku', '#5499C7'), alpha=0.8,
                       edgecolor='black', linewidth=1, hatch='//',
                       error_kw={'linewidth': 1.5, 'ecolor': '#404040'})

        for i, (bar, mean) in enumerate(zip(bars, human_means)):
            ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + human_stds[i] + 1,
                     f'{mean:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=11)

        ax2.set_xticks(x)
        ax2.set_xticklabels([f'{cat}-letter' for cat in length_cats], fontsize=11)
        ax2.set_ylabel('Success Rate (%)', fontweight='bold', fontsize=13)
        ax2.set_xlabel('Word Length', fontweight='bold', fontsize=13)
        ax2.set_title('Human Success by Word Length (n=10,000/puzzle)', fontweight='bold', fontsize=14)
        ax2.set_ylim(0, 105)
        ax2.grid(axis='y', alpha=0.3, linestyle='--')
        ax2.text(-0.08, 1.05, '(b)', transform=ax2.transAxes, fontsize=13, fontweight='bold')

    _save_figure(output_dir, 'fig5_length_coverage_heatmap')
    print("  Fig 5: Length Coverage Heatmap")


# Figure 6: Difficulty Stratification

def plot_difficulty_stratification(summary: Dict, output_dir: Path):
    """Recall on easy vs hard puzzles per model."""
    difficulty_data = summary.get('difficulty_gap_analysis', [])
    if not difficulty_data:
        print("  Fig 6: SKIPPED (no difficulty data)")
        return

    thinking_on_data = [d for d in difficulty_data if d.get('thinking') == True]
    if not thinking_on_data:
        print("  Fig 6: SKIPPED (no thinking-ON data)")
        return

    configs = summary.get('configurations', [])
    optimal_budgets = {}
    for config in configs:
        if config.get('thinking'):
            model = config['model_size']
            budget = config.get('thinking_budget')
            f1 = config.get('f1_mean', 0)
            if model not in optimal_budgets or f1 > optimal_budgets[model][1]:
                optimal_budgets[model] = (budget, f1)

    models, easy_recalls, hard_recalls = [], [], []
    for entry in thinking_on_data:
        model = entry['model_size']
        entry_budget = entry.get('thinking_budget')
        if optimal_budgets and model in optimal_budgets:
            if entry_budget != optimal_budgets[model][0]:
                continue
        if model in MODEL_ORDER:
            models.append(MODEL_LABELS[model])
            easy_recalls.append(entry['easy_recall'] * 100)
            hard_recalls.append(entry['hard_recall'] * 100)

    if not models:
        print("  Fig 6: SKIPPED (no models)")
        return

    x = np.arange(len(models))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width/2, easy_recalls, width, label='Easy (<30 words)',
           color=_desaturate('#029E73'), alpha=0.8, edgecolor='black', linewidth=1.5)
    ax.bar(x + width/2, hard_recalls, width, label='Hard (>50 words)',
           color=_desaturate('#D55E00'), alpha=0.8, edgecolor='black', linewidth=1.5,
           hatch='//')

    ax.set_ylabel('Recall (%)', fontweight='bold', fontsize=13)
    ax.set_xlabel('Model Size', fontweight='bold', fontsize=13)
    ax.set_title('Performance by Puzzle Difficulty', fontweight='bold', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend(loc='upper left', frameon=True, shadow=False)
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    _save_figure(output_dir, 'fig6_difficulty_stratification')
    print("  Fig 6: Difficulty Stratification")


# Figure 7: Model-Human Calibration

def plot_model_human_calibration(summary: Dict, output_dir: Path):
    """Two-panel: recall by human difficulty quartile + calibration correlation."""
    calibration_data = summary.get('model_human_calibration', {})
    if not calibration_data or 'error' in calibration_data:
        print("  Fig 7: SKIPPED (no calibration data)")
        return

    word_diff_data = summary.get('word_difficulty_stratification', [])
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Panel A: recall by human difficulty quartile
    if word_diff_data:
        thinking_data = [d for d in word_diff_data if d.get('thinking') == True]
        quartiles = ['Very Easy (Q1)', 'Easy (Q2)', 'Hard (Q3)', 'Very Hard (Q4)']
        quartile_colors = [_desaturate('#029E73'), _desaturate('#56B4E9'),
                           _desaturate('#E69F00'), _desaturate('#D55E00')]
        quartile_hatches = ['', '//', '\\\\', 'xx']

        model_quartile_data = defaultdict(lambda: {q: [] for q in quartiles})
        for entry in thinking_data:
            model = entry['model_size']
            quartile = entry.get('difficulty_quartile')
            recall = entry.get('recall', 0)
            if model in MODEL_ORDER and quartile in quartiles:
                model_quartile_data[model][quartile].append(recall * 100)

        plot_data = {q: [] for q in quartiles}
        models_to_plot = []
        for model in MODEL_ORDER:
            if model in model_quartile_data:
                models_to_plot.append(MODEL_LABELS[model])
                for q in quartiles:
                    vals = model_quartile_data[model][q]
                    plot_data[q].append(np.mean(vals) if vals else 0)

        x = np.arange(len(models_to_plot))
        width = 0.2
        offsets = [-1.5*width, -0.5*width, 0.5*width, 1.5*width]

        for i, (quartile, color) in enumerate(zip(quartiles, quartile_colors)):
            label = quartile.split()[0] + ' ' + quartile.split()[1]
            ax1.bar(x + offsets[i], plot_data[quartile], width,
                    label=label, color=color, alpha=0.85, edgecolor='black',
                    linewidth=1, hatch=quartile_hatches[i])

        ax1.set_xlabel('Model Size', fontweight='bold', fontsize=13)
        ax1.set_ylabel('Recall (%)', fontweight='bold', fontsize=13)
        ax1.set_title('Model Performance Across Human Difficulty Levels', fontweight='bold', fontsize=14)
        ax1.set_xticks(x)
        ax1.set_xticklabels(models_to_plot)
        ax1.legend(loc='upper right', frameon=True, shadow=False, ncol=2)
        ax1.grid(axis='y', alpha=0.3, linestyle='--')
        ax1.set_ylim(0, max([max(plot_data[q]) for q in quartiles]) * 1.15)
        ax1.text(-0.08, 1.05, '(a)', transform=ax1.transAxes, fontsize=13, fontweight='bold')

    # Panel B: calibration strength (best thinking config per model)
    best_config = {}  # model -> (config_str, correlation)
    for config_str, metrics in calibration_data.items():
        if metrics.get('thinking'):
            model = metrics['model_size']
            corr = metrics.get('correlation', 0)
            if model not in best_config or corr > best_config[model][1]:
                best_config[model] = (config_str, corr)

    models, correlations, colors_list = [], [], []
    for model in MODEL_ORDER:
        if model in best_config:
            models.append(MODEL_LABELS[model])
            correlations.append(best_config[model][1])
            colors_list.append(COLORS[model])

    if models:
        hatch_list = [HATCHES.get(m, '') for m in MODEL_ORDER if m in best_config]
        bars = ax2.bar(models, correlations, color=colors_list, alpha=0.8,
                       edgecolor='black', linewidth=1.5, width=0.6)
        for bar, hatch in zip(bars, hatch_list):
            bar.set_hatch(hatch)
        for bar, corr in zip(bars, correlations):
            ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                     f'{corr:+.3f}', ha='center', va='bottom', fontweight='bold', fontsize=11)
        ax2.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
        ax2.set_ylabel('Spearman $\\rho$', fontweight='bold', fontsize=13)
        ax2.set_xlabel('Model Size', fontweight='bold', fontsize=13)
        ax2.set_title('Model-Human Calibration Strength', fontweight='bold', fontsize=14)
        ax2.set_ylim(-0.1, max(correlations) * 1.2)
        ax2.grid(axis='y', alpha=0.3, linestyle='--')
        ax2.text(-0.08, 1.05, '(b)', transform=ax2.transAxes, fontsize=13, fontweight='bold')

    _save_figure(output_dir, 'fig7_model_human_calibration')
    print("  Fig 7: Model-Human Calibration")


# Figure 8: Generation Volume Analysis

def plot_generation_volume(agg_df: pd.DataFrame, summary: Dict, output_dir: Path):
    """Generation volume by model/budget and volume-precision scatter."""
    thinking_on = agg_df[agg_df['thinking'] == True].copy()

    if 'num_predicted_mean' not in thinking_on.columns:
        print("  Fig 8: SKIPPED (no volume data)")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Left panel: volume by model and budget
    if 'thinking_budget' in thinking_on.columns and thinking_on['thinking_budget'].notna().any():
        budgets = sorted(thinking_on['thinking_budget'].dropna().unique())
        models = []
        volume_by_budget = {b: [] for b in budgets}

        for model in MODEL_ORDER:
            model_data = thinking_on[thinking_on['model_size'] == model]
            if len(model_data) > 0:
                models.append(MODEL_LABELS[model])
                for budget in budgets:
                    bd = model_data[model_data['thinking_budget'] == budget]
                    volume_by_budget[budget].append(bd['num_predicted_mean'].values[0] if len(bd) > 0 else 0)

        x = np.arange(len(models))
        n_budgets = len(budgets)
        width = 0.8 / n_budgets
        offsets = [(i - (n_budgets - 1) / 2) * width for i in range(n_budgets)]

        for i, budget in enumerate(budgets):
            color = COLORS.get(str(int(budget)), '#000000')
            label = BUDGET_LABELS.get(int(budget), f'{int(budget)} tok')
            hatch = BUDGET_HATCHES.get(str(int(budget)), '')
            ax1.bar(x + offsets[i], volume_by_budget[budget], width,
                    label=label, color=color, alpha=0.8, edgecolor='black', linewidth=1,
                    hatch=hatch)

        ax1.set_xticks(x)
        ax1.set_xticklabels(models)
        ax1.legend(title='Budget', frameon=True, shadow=False)
    else:
        models = [MODEL_LABELS[m] for m in thinking_on['model_size']]
        colors_list = [COLORS[m] for m in thinking_on['model_size']]
        ax1.bar(models, thinking_on['num_predicted_mean'].values,
                color=colors_list, alpha=0.8, edgecolor='black', linewidth=1.5)

    ax1.set_xlabel('Model Size', fontweight='bold', fontsize=13)
    ax1.set_ylabel('Mean Predictions per Puzzle', fontweight='bold', fontsize=13)
    ax1.set_title('Generation Volume by Model and Budget', fontweight='bold', fontsize=14)
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    ax1.text(-0.08, 1.05, '(a)', transform=ax1.transAxes, fontsize=13, fontweight='bold')

    # Right panel: volume vs precision scatter
    by_model = summary.get('volume_performance_analysis', {}).get('by_model', {})
    if by_model:
        for model, stats_data in by_model.items():
            if model in MODEL_ORDER:
                ax2.scatter(stats_data.get('mean_volume', 0), stats_data.get('mean_precision', 0),
                            s=150, color=COLORS[model], label=MODEL_LABELS[model],
                            marker=MARKERS.get(model, 'o'),
                            alpha=0.7, edgecolor='black', linewidth=1.5)

        corr = summary.get('volume_performance_analysis', {}).get('overall_correlations', {}).get('volume_vs_precision', 0)
        ax2.text(0.05, 0.95, f'Correlation: {corr:+.3f}',
                 transform=ax2.transAxes, fontsize=11, fontweight='bold',
                 verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        ax2.set_xlabel('Mean Predictions per Puzzle', fontweight='bold', fontsize=13)
        ax2.set_ylabel('Mean Precision', fontweight='bold', fontsize=13)
        ax2.set_title('Volume vs Quality', fontweight='bold', fontsize=14)
        ax2.legend(frameon=True, shadow=False)
        ax2.grid(True, alpha=0.3, linestyle='--')
        ax2.set_ylim(0, 1)
        ax2.text(-0.08, 1.05, '(b)', transform=ax2.transAxes, fontsize=13, fontweight='bold')

    _save_figure(output_dir, 'fig8_generation_volume')
    print("  Fig 8: Generation Volume")


def main():
    parser = argparse.ArgumentParser(description='Generate figures for LREC 2026 paper')
    parser.add_argument('--metrics-dir', type=str, default='metrics/output/',
                        help='Directory containing computed metrics')
    parser.add_argument('--output-dir', type=str, default='figures/',
                        help='Directory to save generated figures')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading metrics from {args.metrics_dir}...")
    summary = load_metrics_summary(args.metrics_dir)
    agg_df = load_aggregated_metrics(args.metrics_dir)
    detailed_df = load_detailed_metrics(args.metrics_dir)
    print(f"  {len(summary['configurations'])} configs, {len(detailed_df)} per-puzzle rows\n")

    print("Generating figures:")
    plot_model_budget_scaling(agg_df, output_dir)
    plot_reliability_analysis(agg_df, detailed_df, output_dir)
    plot_word_length_difficulty(summary, output_dir)
    plot_budget_effects(agg_df, output_dir)
    plot_length_coverage_heatmap(agg_df, args.metrics_dir, output_dir)
    plot_difficulty_stratification(summary, output_dir)
    plot_model_human_calibration(summary, output_dir)
    plot_generation_volume(agg_df, summary, output_dir)

    print(f"\nDone. 8 figures saved to {output_dir}/ (PDF + PNG)")


if __name__ == '__main__':
    main()
