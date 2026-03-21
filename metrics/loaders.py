"""
Data loading functions for results files and difficulty data.
"""

import os
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd


def load_word_difficulty_data(difficulty_dir: str = 'data/puzzles') -> Dict[int, Dict[str, float]]:
    """
    Load word difficulty data from NYT Spelling Bee user success rates.

    Args:
        difficulty_dir: Directory containing bee_YYYYMMDD.json files

    Returns:
        Dict mapping puzzle_id -> {word: difficulty_score}
        difficulty_score = 1 - (user_success_count / n), where higher = harder.
        n defaults to 10,000 (approximate active user sample size per puzzle).
    """
    if not os.path.exists(difficulty_dir):
        print(f"Warning: Difficulty data directory not found: {difficulty_dir}")
        return {}

    difficulty_data = {}
    files = sorted(Path(difficulty_dir).glob('bee_*.json'))

    for filepath in files:
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)

            puzzle_id = data.get('id')
            answers = data.get('answers', {})
            n = data.get('n', 10000)

            if not puzzle_id or not answers:
                continue

            # Convert user success counts to difficulty scores
            # difficulty = 1 - (success_rate), so higher score = harder word
            word_difficulties = {}
            for word, user_count in answers.items():
                success_rate = user_count / n
                difficulty = 1 - success_rate
                word_difficulties[word.lower()] = difficulty

            difficulty_data[puzzle_id] = word_difficulties

        except Exception as e:
            print(f"Warning: Could not load {filepath}: {e}")
            continue

    print(f"Loaded difficulty data for {len(difficulty_data)} puzzles")
    return difficulty_data


def load_results_file(filepath: str) -> Dict:
    """Load a single results JSON file"""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def find_all_results(results_dir: str) -> List[Tuple[str, str, bool, int]]:
    """
    Find all result files in directory and subdirectories.
    Supports three model families:
    - Qwen: qwen3_4b_thinking_results.json
    - Claude: claude_haiku_thinking_16384_results.json
    - GPT-5: gpt5_reasoning_high_results.json

    Returns:
        List of (filepath, model_size, thinking_enabled, thinking_budget) tuples
    """
    results = []
    results_path = Path(results_dir)

    # Look for JSON files in root and subdirectories
    for filepath in results_path.rglob('*.json'):
        # Skip files in 'old' directory or summary files
        if 'old' in filepath.parts or 'summary' in filepath.name.lower():
            continue

        filename = filepath.stem

        # Try different filename patterns

        # Pattern 1: Qwen - qwen3_4b_thinking_results.json
        match = re.match(r'(qwen\d+)_(\d+b)_(thinking|nothinking)_results', filename)
        if match:
            family, size, mode = match.groups()
            thinking_enabled = (mode == 'thinking')

            # Try to infer thinking_budget from directory structure
            parent_dir = filepath.parent.name
            thinking_budget = None

            if parent_dir in ['4', '8', '16']:
                budget_map = {'4': 4096, '8': 8192, '16': 16384}
                thinking_budget = budget_map[parent_dir]

            results.append((str(filepath), size, thinking_enabled, thinking_budget))
            continue

        # Pattern 2: Claude - claude_haiku_thinking_16384_results.json
        match = re.match(r'(claude)_(haiku|sonnet|opus)_(thinking|nothinking)_(\d+)_results', filename)
        if match:
            family, variant, mode, budget = match.groups()
            model_size = f"{family}-{variant}"
            thinking_enabled = (mode == 'thinking')
            thinking_budget = int(budget)

            results.append((str(filepath), model_size, thinking_enabled, thinking_budget))
            continue

        # Pattern 3: GPT-5 - gpt5_reasoning_high_results.json
        match = re.match(r'(gpt\d+)_reasoning_(low|medium|high)_results', filename)
        if match:
            family, effort = match.groups()
            model_size = f"{family}-mini"  # GPT-5-mini
            thinking_enabled = True  # Reasoning is always enabled for these configs

            # Map reasoning effort to approximate budget equivalent
            effort_to_budget = {
                'low': 4096,
                'medium': 8192,
                'high': 16384
            }
            thinking_budget = effort_to_budget.get(effort)

            results.append((str(filepath), model_size, thinking_enabled, thinking_budget))
            continue

    return sorted(results)


def load_all_results(results_dir: str, additional_dirs: List[str] = None) -> pd.DataFrame:
    """
    Load all results files and combine into DataFrame.
    Supports loading from multiple directories to include Qwen, Claude, and GPT-5 results.

    Args:
        results_dir: Primary results directory (e.g., 'results/')
        additional_dirs: Additional directories to scan (e.g., ['claude-results/', 'openai-results/'])

    Returns:
        DataFrame with columns: model_size, thinking, thinking_budget, puzzle_id, date,
        center_letter, all_letters, predicted_words, actual_words, etc.
    """
    # Collect files from all directories
    all_files = []
    dirs_to_scan = [results_dir]
    if additional_dirs:
        dirs_to_scan.extend(additional_dirs)

    seen_configs = set()
    for directory in dirs_to_scan:
        if os.path.exists(directory):
            files = find_all_results(directory)
            new_files = []
            for file_tuple in files:
                filepath, model_size, thinking, budget = file_tuple
                config_key = (model_size, thinking, budget)
                if config_key not in seen_configs:
                    seen_configs.add(config_key)
                    new_files.append(file_tuple)
                else:
                    print(f"  Skipping duplicate config {config_key}: {filepath}")
            all_files.extend(new_files)
            print(f"Found {len(new_files)} new files in {directory} ({len(files) - len(new_files)} duplicates skipped)")
        else:
            print(f"Warning: Directory not found: {directory}")

    if not all_files:
        raise FileNotFoundError(f"No results files found in any of: {dirs_to_scan}")

    print(f"\nTotal: {len(all_files)} result files:")
    for filepath, size, thinking, budget in all_files:
        budget_str = f"{budget}tok" if budget else "N/A"
        print(f"  - {Path(filepath).name} (budget: {budget_str})")

    all_data = []

    for filepath, model_size, thinking_enabled, thinking_budget_from_path in all_files:
        data = load_results_file(filepath)
        metadata = data.get('metadata', {})

        # Handle different metadata formats
        # - Qwen/Claude: thinking_budget
        # - GPT-5: reasoning_effort (map to budget equivalent)
        thinking_budget = metadata.get('thinking_budget')

        if thinking_budget is None and 'reasoning_effort' in metadata:
            # GPT-5 uses reasoning_effort instead of thinking_budget
            effort_to_budget = {'low': 4096, 'medium': 8192, 'high': 16384}
            thinking_budget = effort_to_budget.get(metadata['reasoning_effort'])

        if thinking_budget is None:
            # Fall back to budget extracted from filename/path
            thinking_budget = thinking_budget_from_path

        for prediction in data['predictions']:
            row = {
                'model_size': model_size,
                'thinking': thinking_enabled,
                'thinking_budget': thinking_budget,
                'puzzle_id': prediction['puzzle_id'],
                'date': prediction['date'],
                'center_letter': prediction['center_letter'],
                'all_letters': prediction['all_letters'],
                'predicted_words': prediction['predicted_words'],
                'actual_words': prediction['actual_words'],
                'correctly_predicted': prediction['correctly_predicted'],
                'missed_words': prediction['missed_words'],
                'false_positives': prediction['false_positives'],
                'num_predicted': len(prediction['predicted_words']),
                'num_actual': len(prediction['actual_words']),
                'num_correct': len(prediction['correctly_predicted']),
            }
            all_data.append(row)

    df = pd.DataFrame(all_data)
    print(f"\nLoaded {len(df)} predictions from {df['puzzle_id'].nunique()} unique puzzles")
    print(f"Configurations: {len(df.groupby(['model_size', 'thinking', 'thinking_budget']))}")

    # Print budget distribution
    if 'thinking_budget' in df.columns and df['thinking_budget'].notna().any():
        print(f"\nThinking Budget Distribution:")
        for budget in sorted(df['thinking_budget'].dropna().unique()):
            count = len(df[df['thinking_budget'] == budget])
            print(f"  - {int(budget)} tokens: {count} predictions")

    return df
