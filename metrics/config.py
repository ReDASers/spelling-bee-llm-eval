"""Configuration constants and word difficulty scoring."""

from typing import Dict


MODEL_SIZES = ['4b', '8b', '14b', '30b', '32b']
THINKING_MODES = ['thinking', 'nothinking']

# Puzzle difficulty thresholds based on solution count
DIFFICULTY_THRESHOLDS = {
    'easy': (0, 30),
    'medium': (30, 50),
    'hard': (50, float('inf'))
}

MIN_WORD_LENGTH = 4

F1_THRESHOLDS = [0.2, 0.3, 0.4, 0.5]


def word_difficulty_score(word: str, difficulty_map: Dict[str, float] = None) -> float:
    """Compute difficulty score (0-1, higher = harder).

    Uses NYT user success rates when available, falls back to length-based estimate.
    """
    if difficulty_map and word.lower() in difficulty_map:
        return difficulty_map[word.lower()]

    # Fallback: length-based proxy calibrated to match user success rate patterns
    length = len(word)
    normalized_length = min((length - 4) / 8.0, 1.0)  # 0 at length=4, 1 at length=12+
    return max(0.15 + normalized_length * 0.45, 0.0)  # Range: 0.15 to 0.60
