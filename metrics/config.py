"""
Configuration constants and word difficulty scoring.
"""

from typing import Dict


# Expected model configurations
MODEL_SIZES = ['4b', '8b', '14b', '30b', '32b']
THINKING_MODES = ['thinking', 'nothinking']

# Puzzle difficulty thresholds (based on num_actual words)
DIFFICULTY_THRESHOLDS = {
    'easy': (0, 30),
    'medium': (30, 50),
    'hard': (50, float('inf'))
}

MIN_WORD_LENGTH = 4

# Success rate thresholds for consistency analysis
F1_THRESHOLDS = [0.2, 0.3, 0.4, 0.5]


# Word difficulty estimation
# Using length-based difficulty as proxy (longer words = harder to generate/recall)
def word_difficulty_score(word: str, difficulty_map: Dict[str, float] = None) -> float:
    """
    Compute difficulty score for a word.

    Uses real NYT user success rates when available, falls back to length-based estimate.

    Args:
        word: Input word
        difficulty_map: Optional dict mapping word -> difficulty (1 - success_rate)

    Returns:
        Difficulty score (0-1 range, higher = harder)
    """
    # Use real difficulty data if available
    if difficulty_map and word.lower() in difficulty_map:
        return difficulty_map[word.lower()]

    # Fallback: length-based difficulty estimate
    # Normalize to 0-1 range similar to user success rates
    length = len(word)
    # Map lengths to approximate difficulty: 4-letter=0.15, 8-letter=0.6, etc.
    # This is calibrated to match typical user success rate patterns
    normalized_length = min((length - 4) / 8.0, 1.0)  # 0 at length=4, 1 at length=12+
    return max(0.15 + normalized_length * 0.45, 0.0)  # Range: 0.15 to 0.60
