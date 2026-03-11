# ============================================================================
# NY Times Spelling Bee Word Prediction - Google Colab Version
# ============================================================================
# This script combines generator.py and run_ablation_experiment.py into a
# single file optimized for Google Colab execution.
#
# Instructions:
# 1. Run: !pip install -q vllm transformers tqdm jedi
# 2. Mount Google Drive (will prompt for authorization)
# 3. Ensure Bee-Daily-Pull/ folder exists in MyDrive
# 4. Edit the USER CONFIGURATION section below (set RUN_MODE)
# 5. Run this entire cell
# 
# Results will be saved to:
# - /content/drive/MyDrive/spelling-bee-results/
# - /content/drive/MyDrive/spelling-bee-logs/
# ============================================================================

# !pip install -q vllm transformers tqdm jedi

try:
    from google.colab import drive
    import os
    if not os.path.exists('/content/drive/MyDrive'):
        drive.mount('/content/drive')
    else:
        print("Google Drive already mounted")
except Exception as e:
    print(f"Note: {e}")

import subprocess
try:
    result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
    if result.returncode == 0:
        print("GPU detected and available")
    else:
        print("WARNING: GPU not detected, vLLM requires GPU")
except FileNotFoundError:
    print("WARNING: nvidia-smi not found, GPU may not be available")

# ============================================================================
# IMPORTS
# ============================================================================

import json
import re
import logging
import time
import random
import statistics
from collections import defaultdict, Counter
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from tqdm import tqdm
from datetime import datetime, timedelta

random.seed(42)

# ============================================================================
# USER CONFIGURATION - EDIT THIS SECTION
# ============================================================================

RUN_MODE = "ablation"  # "single" for one prediction, "ablation" for full experiment

# Single Prediction Configuration
SINGLE_CONFIG = {
    'model': "Qwen/Qwen3-4B",
    'prediction_dates': ["20250606"],
    'historical_days': 1,
    'tensor_parallel_size': 1,
    'gpu_memory_utilization': 0.9,
    'max_model_len': 32768,
    'enable_thinking': True,
    'thinking_budget': 12288
}

# Ablation Experiment Configuration
ABLATION_CONFIG = {
    'models_to_test': ["Qwen/Qwen3-4B", "Qwen/Qwen3-8B", "Qwen/Qwen3-14B", "Qwen/Qwen3-32B"],
    'thinking_modes_to_test': [True, False],
    'start_date': "20250602",
    'end_date': "20250729",
    'historical_days': 0,
    'tensor_parallel_size': 1,
    'gpu_memory_utilization': 0.9,
    'max_model_len': 32768,
    'thinking_budget': 12288
}

# Paths (Google Drive mounted for Colab; adjust for local use)
BEE_DATA_PATH = "/content/drive/MyDrive/Bee-Daily-Pull/"
RESULTS_PATH = "/content/drive/MyDrive/spelling-bee-results/"
LOG_PATH = "/content/drive/MyDrive/spelling-bee-logs/"

# ============================================================================
# CONSTANTS
# ============================================================================

MODEL_NAME = SINGLE_CONFIG['model'] if RUN_MODE == "single" else ABLATION_CONFIG['models_to_test'][0]

# Qwen3 recommended sampling parameters
# See: https://qwen.readthedocs.io/ for best practices
SAMPLING_PARAMS_THINKING = {
    'max_tokens': 16384,
    'temperature': 0.6,
    'top_p': 0.95,
    'top_k': 20,
    'min_p': 0.0,
    'presence_penalty': 1.0,
    'frequency_penalty': 1.2,
    'repetition_penalty': 1.2,
}

SAMPLING_PARAMS_NON_THINKING = {
    'max_tokens': 16384,
    'temperature': 0.7,
    'top_p': 0.8,
    'top_k': 20,
    'min_p': 0.0,
    'presence_penalty': 1.0,
    'frequency_penalty': 1.2,
    'repetition_penalty': 1.2,
}

ENABLE_THINKING = True
THINKING_BUDGET = None
THINKING_END_TOKEN_ID = 151668
COMPLETION_TOKEN_ID = 151645
EARLY_STOP_PROMPT = "\n\nConsidering the limited time by the user, I have to give the solution based on the thinking directly now.\n</think>\n\n"

MIN_WORD_LENGTH = 4
TOTAL_ALPHABET_SIZE = 7

# ============================================================================
# LOGGING SETUP
# ============================================================================

def log_separator(logger, title=None, char="-", width=60):
    """Compact separator for cleaner logs"""
    if title:
        logger.info(f"\n{char*width}")
        logger.info(f" {title}")
        logger.info(char*width)
    else:
        logger.info(char*width)


def setup_logging(model_name=None, thinking_mode=None, date_range=None):
    """
    Setup logging with unique files per configuration.
    Creates separate logs for main execution and thinking traces.
    
    Args:
        model_name: Model identifier (e.g., "Qwen/Qwen3-4B")
        thinking_mode: Boolean for thinking on/off
        date_range: Tuple of (start_date, end_date) strings like ("20250602", "20250729")
    """
    os.makedirs(LOG_PATH, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if model_name and thinking_mode is not None:
        model_size = model_name.split('-')[-1] if '-' in model_name else 'unknown'
        think_str = 'thinkON' if thinking_mode else 'thinkOFF'
        
        if date_range:
            start_date, end_date = date_range
            base_name = f"bee_{model_size}_{think_str}_{start_date}-{end_date}_{timestamp}"
        else:
            base_name = f"bee_{model_size}_{think_str}_{timestamp}"
    else:
        base_name = f"bee_prediction_{timestamp}"
    
    log_file = os.path.join(LOG_PATH, f"{base_name}.log")
    thinking_log_file = os.path.join(LOG_PATH, f"{base_name}_thinking.log")
    
    logger = logging.getLogger('bee_predictor')
    logger.setLevel(logging.INFO)
    logger.handlers = []  # Clear existing handlers to prevent duplicates
    logger.propagate = False  # Prevent propagation to root logger
    
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter('%(message)s'))
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    logger.info("="*80)
    logger.info(f"Logging initialized - Log file: {log_file}")
    logger.info(f"Thinking traces will be saved to: {thinking_log_file}")
    logger.info("="*80)
    
    logger.thinking_log_file = thinking_log_file
    
    return logger

logger = setup_logging()

# ============================================================================
# THINKING TRACE LOGGING
# ============================================================================

def save_thinking_trace(puzzle_id, date, thinking_content, logger):
    """Save thinking trace to separate log file for later analysis"""
    if not hasattr(logger, 'thinking_log_file'):
        return
    
    try:
        with open(logger.thinking_log_file, 'a', encoding='utf-8') as f:
            f.write(f"\n{'='*80}\n")
            f.write(f"Puzzle: {date} (ID: {puzzle_id})\n")
            f.write(f"Timestamp: {datetime.now().isoformat()}\n")
            f.write(f"{'='*80}\n")
            f.write(thinking_content)
            f.write(f"\n{'='*80}\n\n")
    except Exception as e:
        logger.debug(f"Failed to save thinking trace: {e}")


# ============================================================================
# DATA LOADING
# ============================================================================

def load_bee_data(filename):
    with open(filename, 'r') as f:
        return json.load(f)


def get_bee_file_dates(start_date, end_date):
    start = datetime.strptime(start_date, "%Y%m%d")
    end = datetime.strptime(end_date, "%Y%m%d")
    
    dates = []
    current = start
    while current <= end:
        dates.append(current.strftime("%Y%m%d"))
        current += timedelta(days=1)
    
    return [f"bee_{date}.json" for date in dates]


def load_historical_data(start_date, end_date):
    historical_files = get_bee_file_dates(start_date, end_date)
    historical_data = []
    
    logger.info(f"Loading historical data from {start_date} to {end_date} ({len(historical_files)} days)")
    for filename in tqdm(historical_files, desc="Loading historical data"):
        filepath = os.path.join(BEE_DATA_PATH, filename)
        try:
            historical_data.append(load_bee_data(filepath))
            logger.debug(f"  Loaded: {filename}")
        except FileNotFoundError:
            logger.warning(f"File {filename} not found, skipping...")
            continue
    
    logger.info(f"Successfully loaded {len(historical_data)} historical puzzle(s)")
    return historical_data

# ============================================================================
# PUZZLE ANALYSIS
# ============================================================================

def extract_puzzle_letters(words):
    if not words:
        return set()
    return set(''.join(words).lower())


def identify_center_letter(words):
    if not words:
        return None
    
    candidate_letters = set(words[0].lower())
    
    for word in words[1:]:
        candidate_letters &= set(word.lower())
    
    if len(candidate_letters) == 1:
        return list(candidate_letters)[0]
    
    letter_counts = {}
    for word in words:
        for letter in set(word.lower()):
            letter_counts[letter] = letter_counts.get(letter, 0) + 1
    
    if letter_counts:
        return max(letter_counts, key=letter_counts.get)
    
    return None

# ============================================================================
# WORD VALIDATION AND DIFFICULTY METRICS
# ============================================================================

def is_valid_bee_word(word, center_letter, allowed_letters):
    word = word.strip().lower()
    return (word 
            and word.isalpha() 
            and len(word) >= MIN_WORD_LENGTH
            and center_letter in word
            and all(c in allowed_letters for c in word))


def calculate_word_difficulty_metrics(word, all_letters):
    """
    Calculate word difficulty proxies based on letter frequency.
    Frequencies from Cornell Math: https://pi.math.cornell.edu/~mec/2003-2004/cryptography/subs/frequencies.html
    """
    word = word.lower()

    LETTER_FREQ = {
        'e': 12.02, 't': 9.10, 'a': 8.12, 'o': 7.68, 'i': 7.31,
        'n': 6.95, 's': 6.28, 'r': 6.02, 'h': 5.92, 'd': 4.32,
        'l': 3.98, 'u': 2.88, 'c': 2.71, 'm': 2.61, 'f': 2.30,
        'y': 2.11, 'w': 2.09, 'g': 2.03, 'p': 1.82, 'b': 1.49,
        'v': 1.11, 'k': 0.69, 'x': 0.17, 'q': 0.11, 'j': 0.10, 'z': 0.07
    }
    
    # Letter scarcity score: lower frequency letters = higher difficulty
    scarcity_scores = [1.0 / (LETTER_FREQ.get(c, 0.01) + 0.01) for c in word]
    avg_scarcity = sum(scarcity_scores) / len(word) if word else 0
    
    # Positional diversity: more unique letter-position patterns = harder
    letter_positions = defaultdict(list)
    for idx, char in enumerate(word):
        letter_positions[char].append(idx)
    
    positional_diversity = len(letter_positions) / len(word) if word else 0
    
    # Letter utilization: fraction of the 7 available letters used
    letters_used = len(set(word))
    utilization_ratio = letters_used / len(all_letters) if all_letters else 0
    
    return {
        'letter_scarcity_score': avg_scarcity,
        'positional_diversity': positional_diversity,
        'letter_utilization': utilization_ratio,
        'word_length': len(word),
        'unique_letters': letters_used,
        'is_pangram': set(word) == set(all_letters)
    }

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_sampling_params(enable_thinking=True):
    params = SAMPLING_PARAMS_THINKING if enable_thinking else SAMPLING_PARAMS_NON_THINKING
    return SamplingParams(**params.copy())


def detect_qwen3_model(model_name):
    qwen3_pattern = r'Qwen3-(\d+)B'
    match = re.search(qwen3_pattern, model_name, re.IGNORECASE)
    
    if match:
        return {
            'is_qwen3': True,
            'model_size': f"{match.group(1)}B",
            'supports_thinking': True
        }
    
    return {
        'is_qwen3': False,
        'model_size': 'unknown',
        'supports_thinking': False
    }

# ============================================================================
# PROMPT CREATION
# ============================================================================

def create_word_prediction_prompt(center_letter, outer_letters, num_words, historical_examples=None):
    all_letters = sorted([center_letter] + outer_letters)
    all_letters_set = set(all_letters)
    
    all_alphabet = set('abcdefghijklmnopqrstuvwxyz')
    forbidden_letters = sorted(all_alphabet - all_letters_set)
    forbidden_display = ', '.join(forbidden_letters).upper()
    
    letters_display = ', '.join(all_letters).upper()
    letters_bar = ' | '.join(all_letters).upper()
    center_upper = center_letter.upper()
    
    examples_section = ""
    if historical_examples:
        example_lines = ["\nHERE ARE EXAMPLES FROM PREVIOUS PUZZLES:\n"]
        
        for i, example in enumerate(historical_examples, 1):
            ex_center = example['center_letter'].upper()
            ex_letters = ', '.join(sorted(example['all_letters'])).upper()
            ex_words = ', '.join(example['words'])
            
            example_lines.extend([
                f"Example {i}:",
                f"  Letters: {ex_letters}",
                f"  Center letter: {ex_center}",
                f"  Valid words: {ex_words}\n"
            ])
        
        examples_section = '\n'.join(example_lines)
    
    return f"""You are an expert at solving NY Times Spelling Bee puzzles. Find as many valid English words as possible using the given letters.

═══════════════════════════════════════════════════════════════════
AVAILABLE LETTERS (ONLY THESE {TOTAL_ALPHABET_SIZE}):
{letters_bar}

CENTER LETTER (must be in every word): {center_upper}
═══════════════════════════════════════════════════════════════════

RULES:
✓ Use ONLY these {TOTAL_ALPHABET_SIZE} letters: {letters_display}
✓ Every word MUST contain the center letter: {center_upper}
✓ Minimum {MIN_WORD_LENGTH} letters per word
✓ Letters can be reused (e.g., 'meet' uses E twice)
✓ Only common American English words (no proper nouns, abbreviations, or archaic terms)
✓ BONUS: Words using all {TOTAL_ALPHABET_SIZE} letters (pangrams) are especially valuable

═══════════════════════════════════════════════════════════════════

INSTRUCTIONS:
1. Start with the center letter and build words around it
2. Focus on word patterns: common roots, prefixes, suffixes 
4. Skip words with letters NOT in the available set 
5. Find a MINIMUM of 30 valid words
6. Only include words you are confident exist in standard dictionaries

OUTPUT FORMAT:
After your thinking, provide ONLY a clean list of valid words.
- One word per line
- No numbers, bullets, or punctuation
- No explanations, notes, or commentary
- No blank lines between words

Start your word list now:"""


def parse_word_predictions(generated_text, center_letter, allowed_letters, target_count):
    lines = generated_text.strip().split('\n')
    
    valid_words = []
    seen = set()
    rejected_count = 0
    
    logger.info(f"Parsing {len(lines)} lines from model output...")
    logger.debug(f"Allowed letters: {sorted(allowed_letters)}")
    
    for line in lines:
        word = line.strip().lower()
        
        if not word:
            continue
        
        if ',' in word:
            word_candidates = [w.strip() for w in word.split(',')]
        else:
            word_candidates = [word]
        
        for candidate in word_candidates:
            if candidate and candidate[0].isdigit():
                candidate = ''.join(c for c in candidate if c.isalpha())
            
            candidate = ''.join(c for c in candidate if c.isalpha())
            
            if not candidate:
                continue
            
            if is_valid_bee_word(candidate, center_letter, allowed_letters):
                if candidate not in seen:
                    seen.add(candidate)
                    valid_words.append(candidate)
                    logger.debug(f"  Valid word #{len(valid_words)}: {candidate}")
                
                if len(valid_words) >= target_count:
                    break
            else:
                rejected_count += 1
                invalid_letters = set(candidate) - allowed_letters
                if invalid_letters:
                    logger.debug(f"  Rejected '{candidate}': contains forbidden letters {sorted(invalid_letters)}")
                elif center_letter not in candidate:
                    logger.debug(f"  Rejected '{candidate}': missing center letter '{center_letter}'")
                elif len(candidate) < MIN_WORD_LENGTH:
                    logger.debug(f"  Rejected '{candidate}': too short (< {MIN_WORD_LENGTH} letters)")
        
        if len(valid_words) >= target_count:
            break
    
    logger.info(f"Extracted {len(valid_words)} valid words ({rejected_count} rejected)")
    logger.debug(f"Valid words: {', '.join(valid_words[:10])}{'...' if len(valid_words) > 10 else ''}")
    return valid_words[:target_count]

# ============================================================================
# TEXT GENERATION
# ============================================================================

def _generate_with_thinking_budget(llm, tokenizer, formatted_prompt, sampling_params, thinking_budget):
    """
    Implements thinking budget for vLLM by generating in stages:
    1. Generate up to thinking_budget tokens
    2. Check if thinking completed (</think> found)
    3. If not, inject early-stop prompt and continue with remaining budget
    """
    logger.info(f"Using thinking budget: {thinking_budget} tokens")
    budget_sampling_params = SamplingParams(
        max_tokens=thinking_budget,
        temperature=sampling_params.temperature,
        top_p=sampling_params.top_p,
        top_k=sampling_params.top_k,
        min_p=sampling_params.min_p,
        repetition_penalty=sampling_params.repetition_penalty,
    )
    
    start_time = time.time()
    outputs_stage1 = llm.generate([formatted_prompt], budget_sampling_params)
    stage1_time = time.time() - start_time
    
    output_stage1 = outputs_stage1[0]
    stage1_token_ids = output_stage1.outputs[0].token_ids
    stage1_text = output_stage1.outputs[0].text
    prompt_len = len(output_stage1.prompt_token_ids)
    
    logger.info(f"Stage 1: Generated {len(stage1_token_ids)} tokens in {stage1_time:.2f}s")
    logger.debug(f"Stage 1 first 10 token IDs: {stage1_token_ids[:10]}")
    logger.debug(f"Stage 1 last 10 token IDs: {stage1_token_ids[-10:]}")
    logger.debug(f"Checking for COMPLETION_TOKEN_ID ({COMPLETION_TOKEN_ID}) and THINKING_END_TOKEN_ID ({THINKING_END_TOKEN_ID})")
    logger.debug(f"Stage 1 text preview (first 200 chars): {stage1_text[:200]}")
    logger.debug(f"Stage 1 text preview (last 200 chars): {stage1_text[-200:]}")
    logger.debug(f"Contains '<think>': {'<think>' in stage1_text}")
    logger.debug(f"Contains '</think>': {'</think>' in stage1_text}")
    
    if COMPLETION_TOKEN_ID in stage1_token_ids:
        logger.info("Generation completed within thinking budget")
        prompt_tokens = len(output_stage1.prompt_token_ids)
        tokens_generated = len(stage1_token_ids)
        tokens_per_second = tokens_generated / stage1_time if stage1_time > 0 else 0
        
        throughput_stats = {
            'tokens_generated': tokens_generated,
            'generation_time': stage1_time,
            'tokens_per_second': tokens_per_second,
            'prompt_tokens': prompt_tokens,
            'enable_thinking': True,
            'thinking_budget_used': True,
            'budget_exceeded': False
        }
        
        if '</think>' in stage1_text:
            parts = stage1_text.split('</think>', 1)
            answer = parts[1].strip() if len(parts) == 2 else stage1_text
            throughput_stats['thinking_tokens'] = len(parts[0]) // 4
            throughput_stats['answer_tokens'] = tokens_generated - throughput_stats['thinking_tokens']
        else:
            answer = stage1_text
            throughput_stats['thinking_tokens'] = 0
            throughput_stats['answer_tokens'] = tokens_generated
        
        return answer, throughput_stats
    
    thinking_ended = THINKING_END_TOKEN_ID in stage1_token_ids or '</think>' in stage1_text
    
    if thinking_ended:
        logger.info("Thinking completed within budget (found </think>)")
        if '</think>' in stage1_text:
            parts = stage1_text.split('</think>', 1)
            answer_content = parts[1].strip()
            thinking_content = parts[0]
            
            thinking_chars = len(thinking_content)
            answer_chars = len(answer_content)
            total_chars = thinking_chars + answer_chars
            total_tokens_stage1 = len(stage1_token_ids)
            
            if total_chars > 0:
                thinking_ratio = thinking_chars / total_chars
                thinking_tokens = int(total_tokens_stage1 * thinking_ratio)
                answer_tokens = total_tokens_stage1 - thinking_tokens
            else:
                thinking_tokens = total_tokens_stage1
                answer_tokens = 0
            
            prompt_tokens = len(output_stage1.prompt_token_ids)
            tokens_per_second = total_tokens_stage1 / stage1_time if stage1_time > 0 else 0
            
            throughput_stats = {
                'tokens_generated': total_tokens_stage1,
                'generation_time': stage1_time,
                'tokens_per_second': tokens_per_second,
                'prompt_tokens': prompt_tokens,
                'enable_thinking': True,
                'thinking_budget_used': True,
                'budget_exceeded': False,
                'thinking_tokens': thinking_tokens,
                'answer_tokens': answer_tokens,
                'stage1_tokens': total_tokens_stage1,
                'stage2_tokens': 0,
                'early_stop_injected': False
            }
            
            logger.info(f"Thinking budget: {thinking_budget} tokens")
            logger.info(f"Used: {total_tokens_stage1} tokens (within budget)")
            logger.info(f"Thinking tokens: {thinking_tokens}, Answer tokens: {answer_tokens}")
            
            throughput_stats['thinking_content'] = thinking_content
            
            return answer_content, throughput_stats
        else:
            return stage1_text, {'tokens_generated': len(stage1_token_ids), 'generation_time': stage1_time,
                               'tokens_per_second': len(stage1_token_ids)/stage1_time, 'prompt_tokens': prompt_len}
    
    logger.warning("Thinking budget exhausted WITHOUT completing </think>")
    logger.info(f"Budget used: {len(stage1_token_ids)}/{thinking_budget} tokens")
    logger.info("Injecting early-stop prompt to force answer...")
    logger.debug(f"Early stop prompt: {EARLY_STOP_PROMPT[:100]}...")
    
    continuation_prompt = formatted_prompt + stage1_text + EARLY_STOP_PROMPT
    original_max_tokens = sampling_params.max_tokens
    tokens_used = len(stage1_token_ids)
    
    early_stop_token_estimate = 24
    remaining_tokens = original_max_tokens - tokens_used - early_stop_token_estimate
    
    logger.info(f"Remaining for stage 2: {remaining_tokens} tokens")
    
    if remaining_tokens <= 50:
        logger.warning(f"Very few tokens remaining ({remaining_tokens}), setting minimum to 200")
        remaining_tokens = 200
    
    stage2_sampling_params = SamplingParams(
        max_tokens=remaining_tokens,
        temperature=sampling_params.temperature,
        top_p=sampling_params.top_p,
        top_k=sampling_params.top_k,
        min_p=sampling_params.min_p,
        repetition_penalty=sampling_params.repetition_penalty,
    )
    
    start_stage2 = time.time()
    outputs_stage2 = llm.generate([continuation_prompt], stage2_sampling_params)
    stage2_time = time.time() - start_stage2
    
    output_stage2 = outputs_stage2[0]
    stage2_full_text = output_stage2.outputs[0].text
    stage2_token_ids = output_stage2.outputs[0].token_ids

    # Extract only the new generation after the early stop prompt
    expected_prefix_text = stage1_text + EARLY_STOP_PROMPT
    if stage2_full_text.startswith(expected_prefix_text):
        stage2_new_text = stage2_full_text[len(expected_prefix_text):]
        logger.debug(f"Extracted {len(stage2_new_text)} chars of new stage2 text")
    else:
        # Fallback: try to find where the new content starts
        logger.warning("Could not match expected prefix in stage2 output, using full output")
        stage2_new_text = stage2_full_text
    
    logger.info(f"Stage 2: Generated {len(stage2_token_ids)} new tokens in {stage2_time:.2f}s")
    
    full_text = stage1_text + EARLY_STOP_PROMPT + stage2_new_text
    total_time = stage1_time + stage2_time
    total_tokens = len(stage1_token_ids) + early_stop_token_estimate + len(stage2_token_ids)
    
    prompt_tokens = len(output_stage1.prompt_token_ids)
    tokens_per_second = total_tokens / total_time if total_time > 0 else 0
    
    throughput_stats = {
        'tokens_generated': total_tokens,
        'generation_time': total_time,
        'tokens_per_second': tokens_per_second,
        'prompt_tokens': prompt_tokens,
        'enable_thinking': True,
        'thinking_budget_used': True,
        'budget_exceeded': True,
        'stage1_tokens': len(stage1_token_ids),
        'stage2_tokens': len(stage2_token_ids),
        'early_stop_injected': True
    }
    
    if '</think>' in full_text:
        parts = full_text.split('</think>', 1)
        if len(parts) == 2:
            thinking_content = parts[0]
            answer_content = parts[1].strip()
            
            thinking_chars = len(thinking_content)
            answer_chars = len(answer_content)
            total_chars = thinking_chars + answer_chars
            
            if total_chars > 0:
                thinking_ratio = thinking_chars / total_chars
                throughput_stats['thinking_tokens'] = int(total_tokens * thinking_ratio)
                throughput_stats['answer_tokens'] = total_tokens - throughput_stats['thinking_tokens']
            else:
                throughput_stats['thinking_tokens'] = len(stage1_token_ids)
                throughput_stats['answer_tokens'] = len(stage2_token_ids)
            
            logger.info(f"Successfully extracted answer after early-stop injection")
            logger.info(f"Thinking budget: {thinking_budget} tokens (EXCEEDED)")
            logger.info(f"Total tokens: {total_tokens} (Stage 1: {len(stage1_token_ids)}, Early-stop: ~{early_stop_token_estimate}, Stage 2: {len(stage2_token_ids)})")
            logger.info(f"Thinking tokens: {throughput_stats['thinking_tokens']}, Answer tokens: {throughput_stats['answer_tokens']}")
            
            throughput_stats['thinking_content'] = thinking_content
            
            return answer_content, throughput_stats
        else:
            logger.warning("Found </think> but couldn't split properly")
    else:
        logger.warning("Could not find </think> tag even after early-stop injection")
        logger.warning("Model may not have responded to early-stop prompt correctly")
    
    # Fallback: return whatever we got
    throughput_stats['thinking_tokens'] = len(stage1_token_ids)
    throughput_stats['answer_tokens'] = len(stage2_token_ids)
    
    logger.debug(f"Returning full_text as fallback: {full_text[:200]}...")
    return full_text, throughput_stats


def generate_text(llm, tokenizer, prompt, model_metadata=None, thinking_budget=None):
    messages = [{"role": "user", "content": prompt}]
    enable_thinking = model_metadata.get('enable_thinking', ENABLE_THINKING) if model_metadata else ENABLE_THINKING
    
    if thinking_budget is None:
        thinking_budget = THINKING_BUDGET
    
    formatted_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking
    )
    
    logger.debug(f"Formatted prompt (first 300 chars): {formatted_prompt[:300]}")
    logger.debug(f"enable_thinking={enable_thinking} was passed to apply_chat_template")
    if thinking_budget:
        logger.debug(f"Thinking budget enabled: {thinking_budget} tokens")
    
    sampling_params = get_sampling_params(enable_thinking)
    
    if enable_thinking and thinking_budget:
        return _generate_with_thinking_budget(
            llm, tokenizer, formatted_prompt, sampling_params, thinking_budget
        )
        
    start_time = time.time()
    outputs = llm.generate([formatted_prompt], sampling_params)
    generation_time = time.time() - start_time
        
    output = outputs[0]
    generated_text = output.outputs[0].text
    
    prompt_tokens = len(output.prompt_token_ids)
    tokens_generated = len(output.outputs[0].token_ids)
    tokens_per_second = tokens_generated / generation_time if generation_time > 0 else 0
    
    throughput_stats = {
        'tokens_generated': tokens_generated,
        'generation_time': generation_time,
        'tokens_per_second': tokens_per_second,
        'prompt_tokens': prompt_tokens,
        'enable_thinking': enable_thinking,
        'thinking_budget_used': False
    }
    
    if enable_thinking:
        if '</think>' in generated_text:
            parts = generated_text.split('</think>', 1)
            if len(parts) == 2:
                thinking_content = parts[0]
                answer_content = parts[1].strip()
                
                logger.info(f"Thinking mode: Found </think> tag")
                logger.info(f"Thinking content: {len(thinking_content)} chars, Answer: {len(answer_content)} chars")
                
                thinking_tokens = int(len(thinking_content) / 4)
                answer_tokens = tokens_generated - thinking_tokens
                
                throughput_stats['thinking_tokens'] = thinking_tokens
                throughput_stats['answer_tokens'] = answer_tokens
                throughput_stats['has_thinking_content'] = bool(thinking_content.strip())
                throughput_stats['thinking_content'] = thinking_content
                
                return answer_content, throughput_stats
            else:
                logger.warning("Found </think> tag but couldn't split properly")
        else:
            if '<think>' in generated_text[:100]:
                logger.warning("Found <think> opening tag but no </think> closing tag")
                logger.warning("Output likely truncated - consider increasing max_tokens")
            else:
                logger.warning("Thinking mode enabled but no thinking tags found in output")
        
        throughput_stats['thinking_tokens'] = 0
        throughput_stats['answer_tokens'] = tokens_generated
        throughput_stats['has_thinking_content'] = False
    
    return generated_text, throughput_stats


def predict_words_for_puzzle(center_letter, outer_letters, llm, tokenizer, num_words=50, historical_examples=None, model_metadata=None):
    logger.info("")
    log_separator(logger, "PREDICTING WORDS FOR PUZZLE")
    logger.info(f"Center letter: {center_letter.upper()}")
    logger.info(f"Other letters: {', '.join(sorted(outer_letters))}")
    logger.info(f"Target: {num_words} words")
    
    if historical_examples:
        logger.info(f"Using {len(historical_examples)} historical example(s) for few-shot learning")
        for i, ex in enumerate(historical_examples, 1):
            logger.info(f"  Example {i}: Center={ex['center_letter'].upper()}, Letters={', '.join(ex['all_letters'])}")
    log_separator(logger)
    
    prompt = create_word_prediction_prompt(center_letter, outer_letters, num_words, historical_examples)
    
    logger.debug("\n")
    logger.debug("="*80)
    logger.debug("PROMPT SENT TO LLM (FULL):")
    logger.debug("="*80)
    logger.debug(prompt)
    logger.debug("="*80)
    
    logger.info(f"\nPrompt created: {len(prompt)} characters, {num_words} words requested")
    
    logger.info("\nGenerating predictions from LLM...")
    thinking_budget = model_metadata.get('thinking_budget') if model_metadata else None
    output, throughput_stats = generate_text(llm, tokenizer, prompt, model_metadata=model_metadata, thinking_budget=thinking_budget)
    
    logger.info("\n")
    log_separator(logger, "GENERATION THROUGHPUT:")
    logger.info(f"Prompt tokens: {throughput_stats['prompt_tokens']}")
    logger.info(f"Tokens generated: {throughput_stats['tokens_generated']}")
    logger.info(f"Generation time: {throughput_stats['generation_time']:.2f} seconds")
    logger.info(f"Throughput: {throughput_stats['tokens_per_second']:.2f} tokens/second")
    
    if 'thinking_tokens' in throughput_stats:
        thinking_pct = throughput_stats['thinking_tokens'] / throughput_stats['tokens_generated'] * 100
        answer_pct = throughput_stats['answer_tokens'] / throughput_stats['tokens_generated'] * 100
        logger.info(f"  - Thinking tokens: {throughput_stats['thinking_tokens']} ({thinking_pct:.1f}%)")
        logger.info(f"  - Answer tokens: {throughput_stats['answer_tokens']} ({answer_pct:.1f}%)")
    log_separator(logger)
    
    logger.debug("\n")
    logger.debug("="*80)
    logger.debug("RAW MODEL OUTPUT (FULL):")
    logger.debug("="*80)
    logger.debug(output)
    logger.debug("="*80)

    output_preview = output[:500] + "..." if len(output) > 500 else output
    logger.info(f"\nModel output preview (first 500 chars): {output_preview}")
    if len(output) > 500:
        logger.info(f"(Full output: {len(output)} characters - see log file for details)")
    
    all_letters = set([center_letter] + outer_letters)
    predicted_words = parse_word_predictions(output, center_letter, all_letters, num_words)
    
    if 'thinking_content' in throughput_stats and throughput_stats['thinking_content']:
        puzzle_id = model_metadata.get('current_puzzle_id', 'unknown') if model_metadata else 'unknown'
        puzzle_date = model_metadata.get('current_puzzle_date', 'unknown') if model_metadata else 'unknown'
        save_thinking_trace(puzzle_id, puzzle_date, throughput_stats['thinking_content'], logger)

    word_difficulties = {}
    for word in predicted_words:
        word_difficulties[word] = calculate_word_difficulty_metrics(word, all_letters)
    
    throughput_stats['word_difficulty_metrics'] = word_difficulties
    
    logger.info(f"\nSuccessfully predicted {len(predicted_words)} words")
    return predicted_words, throughput_stats

# ============================================================================
# EVALUATION
# ============================================================================

def analyze_constraint_violations(predicted_words, center_letter, allowed_letters):
    """Analyze constraint violations for error taxonomy"""
    violations = {
        'missing_center': [],
        'forbidden_letters': [],
        'too_short': [],
        'all_violations': []
    }
    
    for word in predicted_words:
        word_violations = []
        if center_letter not in word:
            violations['missing_center'].append(word)
            word_violations.append('missing_center')
        
        forbidden = set(word) - allowed_letters
        if forbidden:
            violations['forbidden_letters'].append((word, sorted(forbidden)))
            word_violations.append('forbidden_letters')
        
        if len(word) < MIN_WORD_LENGTH:
            violations['too_short'].append(word)
            word_violations.append('too_short')
        
        if word_violations:
            violations['all_violations'].append((word, word_violations))
    
    return violations


def categorize_errors(predicted_words, actual_words, center_letter, allowed_letters):
    """Detailed error categorization for LREC paper analysis"""
    errors = {
        'constraint_violations': {},
        'non_dictionary_words': [],
        'repetitions': [],
        'total_errors': 0
    }
    
    word_counts = Counter(predicted_words)
    errors['repetitions'] = [(word, count) for word, count in word_counts.items() if count > 1]

    unique_predicted = set(predicted_words)
    actual_words_set = set(actual_words) if isinstance(actual_words, list) else actual_words
    incorrect_words = unique_predicted - actual_words_set
    
    for word in incorrect_words:
        if center_letter not in word:
            errors['constraint_violations'].setdefault('missing_center', []).append(word)
        elif not all(c in allowed_letters for c in word):
            forbidden = sorted(set(word) - allowed_letters)
            errors['constraint_violations'].setdefault('forbidden_letters', []).append((word, forbidden))
        elif len(word) < MIN_WORD_LENGTH:
            errors['constraint_violations'].setdefault('too_short', []).append(word)
        else:
            errors['non_dictionary_words'].append(word)
    
    errors['total_errors'] = len(incorrect_words) + len(errors['repetitions'])
    
    return errors


def calculate_metrics(predicted_words, actual_words, center_letter=None, allowed_letters=None, tokens_generated=None):
    """
    Calculate comprehensive metrics for LREC paper.
    Includes standard IR metrics + constraint adherence + token efficiency.
    """
    true_positives = len(actual_words & predicted_words)
    false_positives = len(predicted_words - actual_words)
    false_negatives = len(actual_words - predicted_words)
    
    precision = true_positives / len(predicted_words) if predicted_words else 0
    recall = true_positives / len(actual_words) if actual_words else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = recall
    jaccard = true_positives / len(actual_words | predicted_words) if (actual_words | predicted_words) else 0
    
    metrics = {
        'num_predicted': len(predicted_words),
        'num_actual': len(actual_words),
        'true_positives': true_positives,
        'false_positives': false_positives,
        'false_negatives': false_negatives,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'accuracy': accuracy,
        'jaccard_index': jaccard
    }
    
    if center_letter and allowed_letters:
        violations = analyze_constraint_violations(predicted_words, center_letter, allowed_letters)
        total_violations = len(violations['all_violations'])
        
        metrics['constraint_adherence_rate'] = 1 - (total_violations / len(predicted_words)) if predicted_words else 0
        metrics['num_constraint_violations'] = total_violations
        metrics['violation_breakdown'] = {
            'missing_center': len(violations['missing_center']),
            'forbidden_letters': len(violations['forbidden_letters']),
            'too_short': len(violations['too_short'])
        }
    
    if tokens_generated:
        metrics['token_efficiency'] = (true_positives / tokens_generated) * 1000 if tokens_generated > 0 else 0
        metrics['tokens_per_correct_word'] = tokens_generated / true_positives if true_positives > 0 else float('inf')
    
    return metrics


def print_summary(predictions):
    logger.info("\n")
    log_separator(logger, "WORD PREDICTION SUMMARY")
    
    for pred in predictions:
        metrics = pred['metrics']
        logger.info(f"\nPuzzle: {pred['date_file']} (ID: {pred['puzzle_id']})")
        logger.info(f"  Center Letter: {pred['center_letter'].upper()}")
        logger.info(f"  All Letters: {', '.join(sorted(pred['all_letters']))}")
        logger.info(f"\n  Results:")
        logger.info(f"    Predicted: {metrics['num_predicted']} words")
        logger.info(f"    Actual: {metrics['num_actual']} words")
        logger.info(f"    Correctly Predicted: {metrics['true_positives']} words")
        logger.info(f"\n  Performance Metrics:")
        logger.info(f"    Precision: {metrics['precision']:.3f} ({metrics['true_positives']}/{metrics['num_predicted']})")
        logger.info(f"    Recall:    {metrics['recall']:.3f} ({metrics['true_positives']}/{metrics['num_actual']})")
        logger.info(f"    F1 Score:  {metrics['f1_score']:.3f}")
        logger.info(f"    Accuracy:  {metrics['accuracy']:.3f} (same as recall)")
        logger.info(f"    Jaccard:   {metrics['jaccard_index']:.3f} (set similarity)")
        
        if 'throughput' in pred:
            throughput = pred['throughput']
            logger.info(f"\n  Throughput Statistics:")
            logger.info(f"    Generation time: {throughput['generation_time']:.2f} seconds")
            logger.info(f"    Tokens generated: {throughput['tokens_generated']}")
            logger.info(f"    Speed: {throughput['tokens_per_second']:.2f} tokens/second")
            if 'thinking_tokens' in throughput:
                thinking_pct = throughput['thinking_tokens'] / throughput['tokens_generated'] * 100
                logger.info(f"    Thinking tokens: {throughput['thinking_tokens']} ({thinking_pct:.1f}%)")
        
        if pred['correctly_predicted']:
            correct_sample = ', '.join(sorted(pred['correctly_predicted'])[:10])
            if len(pred['correctly_predicted']) > 10:
                correct_sample += "..."
            logger.info(f"\n  Sample Correct Predictions: {correct_sample}")
        
        if pred['predicted_words']:
            logger.info(f"\n  All Predicted Words: {', '.join(pred['predicted_words'])}")
    
    if predictions:
        avg_precision = sum(p['metrics']['precision'] for p in predictions) / len(predictions)
        avg_recall = sum(p['metrics']['recall'] for p in predictions) / len(predictions)
        avg_f1 = sum(p['metrics']['f1_score'] for p in predictions) / len(predictions)
        
        logger.info("\n")
        log_separator(logger, "AVERAGE PERFORMANCE:")
        logger.info(f"  Precision: {avg_precision:.3f}")
        logger.info(f"  Recall:    {avg_recall:.3f}")
        logger.info(f"  F1 Score:  {avg_f1:.3f}")
    
    log_separator(logger)
    logger.info("")

# ============================================================================
# MAIN PROCESSING
# ============================================================================

def create_run_metadata(model_name, model_metadata, vllm_config, historical_days, enable_thinking=None):
    if enable_thinking is None:
        enable_thinking = ENABLE_THINKING
    
    sampling_params = get_sampling_params(enable_thinking)
    sampling_params_dict = {
        'max_tokens': sampling_params.max_tokens,
        'temperature': sampling_params.temperature,
        'top_p': sampling_params.top_p,
        'top_k': sampling_params.top_k,
        'min_p': sampling_params.min_p,
        'repetition_penalty': sampling_params.repetition_penalty,
    }
    
    return {
        'timestamp': datetime.now().isoformat(),
        'model_name': model_name,
        'inference_engine': 'vllm',
        'model_config': {
            'is_qwen3': model_metadata.get('is_qwen3', False),
            'model_size': model_metadata.get('model_size', 'unknown'),
            'supports_thinking': model_metadata.get('supports_thinking', False),
            'enable_thinking': enable_thinking,
            'vllm_config': vllm_config
        },
        'sampling_params': sampling_params_dict,
        'puzzle_config': {
            'min_word_length': MIN_WORD_LENGTH,
            'total_letters': TOTAL_ALPHABET_SIZE
        },
        'historical_days': historical_days,
        'platform': 'google_colab'
    }


def _compute_summary_stats(all_predictions):
    """Compute aggregate summary statistics for a set of predictions"""
    if not all_predictions:
        return {}
    
    precisions = [p['metrics']['precision'] for p in all_predictions]
    recalls = [p['metrics']['recall'] for p in all_predictions]
    f1_scores = [p['metrics']['f1_score'] for p in all_predictions]
    cars = [p['metrics'].get('constraint_adherence_rate', 0) for p in all_predictions]
    token_effs = [p['metrics'].get('token_efficiency', 0) for p in all_predictions]
    
    return {
        'total_puzzles': len(all_predictions),
        'avg_precision': statistics.mean(precisions),
        'std_precision': statistics.stdev(precisions) if len(precisions) > 1 else 0,
        'avg_recall': statistics.mean(recalls),
        'std_recall': statistics.stdev(recalls) if len(recalls) > 1 else 0,
        'avg_f1': statistics.mean(f1_scores),
        'std_f1': statistics.stdev(f1_scores) if len(f1_scores) > 1 else 0,
        'avg_car': statistics.mean(cars),
        'std_car': statistics.stdev(cars) if len(cars) > 1 else 0,
        'avg_token_efficiency': statistics.mean(token_effs),
        'std_token_efficiency': statistics.stdev(token_effs) if len(token_effs) > 1 else 0
    }


def _build_results_filename(model_metadata, all_predictions):
    """Build descriptive filename based on model config and date range."""
    model_name = model_metadata.get('name', '')
    if "Qwen3" in model_name or "qwen3" in model_name.lower():
        model_family = "qwen3"
    else:
        model_family = "unknown"
    
    model_size = model_metadata.get('model_size', 'unknown').lower().replace('-fp8', '')
    thinking_str = 'thinkON' if model_metadata.get('enable_thinking', False) else 'thinkOFF'
    
    dates = [p['date_file'].replace('bee_', '').replace('.json', '') for p in all_predictions]
    date_start = min(dates)
    date_end = max(dates)
    
    return f'{model_family}_{model_size}_{thinking_str}_{date_start}-{date_end}_results.json'


def process_bee_predictions(prediction_files, llm, tokenizer, output_path, historical_data=None, model_metadata=None, run_metadata=None):
    all_predictions = []
    
    historical_examples = None
    if historical_data:
        logger.info("\nPreparing historical examples for few-shot learning...")
        historical_examples = []
        for hist_data in historical_data:
            hist_words = list(hist_data['answers'].keys())
            hist_letters = extract_puzzle_letters(hist_words)
            hist_center = identify_center_letter(hist_words)
            
            if hist_center:
                historical_examples.append({
                    'center_letter': hist_center,
                    'all_letters': sorted(hist_letters),
                    'words': hist_words
                })
                logger.info(f"  Added example: Puzzle ID {hist_data['id']}, Center={hist_center.upper()}, {len(hist_words)} total words")
    
    logger.info(f"\nProcessing {len(prediction_files)} puzzle(s)...")
    
    for filename in tqdm(prediction_files, desc="Generating predictions"):
        logger.info(f"\n")
        log_separator(logger, f"PROCESSING: {filename}")
        
        filepath = os.path.join(BEE_DATA_PATH, filename)
        try:
            puzzle_data = load_bee_data(filepath)
            logger.info(f"Loaded puzzle data: ID {puzzle_data['id']}")
        except FileNotFoundError:
            logger.warning(f"File {filename} not found, skipping...")
            continue
        
        actual_words = list(puzzle_data['answers'].keys())
        all_letters = extract_puzzle_letters(actual_words)
        center_letter = identify_center_letter(actual_words)
        
        logger.info(f"Puzzle has {len(actual_words)} actual words")
        logger.info(f"Identified center letter: {center_letter.upper() if center_letter else 'UNKNOWN'}")
        logger.info(f"All letters: {', '.join(sorted(all_letters))}")
        
        if not center_letter:
            logger.warning(f"Could not identify center letter for {filename}, skipping...")
            continue
        
        outer_letters = sorted(all_letters - {center_letter})
        
        if model_metadata:
            model_metadata['current_puzzle_id'] = puzzle_data['id']
            model_metadata['current_puzzle_date'] = filename.replace('bee_', '').replace('.json', '')
        
        predicted_words, throughput_stats = predict_words_for_puzzle(
            center_letter,
            outer_letters,
            llm,
            tokenizer,
            num_words=len(actual_words),
            historical_examples=historical_examples,
            model_metadata=model_metadata
        )
        
        predicted_set = set(predicted_words)
        actual_set = set(actual_words)
        
        tokens_generated = throughput_stats.get('tokens_generated', None)
        metrics = calculate_metrics(
            predicted_set, 
            actual_set, 
            center_letter=center_letter,
            allowed_letters=all_letters,
            tokens_generated=tokens_generated
        )
        
        error_analysis = categorize_errors(predicted_words, actual_words, center_letter, all_letters)
        metrics['error_analysis'] = error_analysis

        actual_word_difficulties = {}
        for word in actual_words:
            actual_word_difficulties[word] = calculate_word_difficulty_metrics(word, all_letters)
        
        found_words = actual_set & predicted_set
        missed_words = actual_set - predicted_set

        if found_words and missed_words:
            found_difficulties = [actual_word_difficulties[w]['letter_scarcity_score'] for w in found_words]
            missed_difficulties = [actual_word_difficulties[w]['letter_scarcity_score'] for w in missed_words]
            
            metrics['difficulty_analysis'] = {
                'avg_difficulty_found': statistics.mean(found_difficulties) if found_difficulties else 0,
                'avg_difficulty_missed': statistics.mean(missed_difficulties) if missed_difficulties else 0,
                'found_word_difficulties': {w: actual_word_difficulties[w] for w in found_words},
                'missed_word_difficulties': {w: actual_word_difficulties[w] for w in missed_words}
            }
        
        result = {
            'date_file': filename,
            'puzzle_id': puzzle_data['id'],
            'center_letter': center_letter,
            'all_letters': sorted(all_letters),
            'predicted_words': predicted_words,
            'actual_words': sorted(actual_words),
            'correctly_predicted': sorted(actual_set & predicted_set),
            'missed_words': sorted(missed_words),
            'false_positives': sorted(predicted_set - actual_set),
            'metrics': metrics,
            'throughput': throughput_stats,
            'word_difficulty_metrics': actual_word_difficulties
        }
        all_predictions.append(result)
    
    os.makedirs(output_path, exist_ok=True)
    
    if model_metadata and all_predictions:
        filename = _build_results_filename(model_metadata, all_predictions)
        output_file = os.path.join(output_path, filename)
    else:
        output_file = os.path.join(output_path, 'bee_predictions.json')
    
    summary = _compute_summary_stats(all_predictions)

    results_with_metadata = {
        'metadata': run_metadata if run_metadata else {},
        'predictions': all_predictions,
        'summary': summary
    }
    
    with open(output_file, 'w') as f:
        json.dump(results_with_metadata, f, indent=2)
    
    logger.info(f"\nPredictions saved to {output_file}")
    if summary:
        logger.info(f"Summary: {summary['total_puzzles']} puzzles | "
                   f"F1={summary['avg_f1']:.3f} | P={summary['avg_precision']:.3f} | R={summary['avg_recall']:.3f}")
    
    print_summary(all_predictions)
    
    return all_predictions


def load_model(model_name, tensor_parallel_size=1, gpu_memory_utilization=0.9, max_model_len=None):
    logger.info(f"\nLoading model with vLLM: {model_name}")
    logger.info(f"Tensor parallel size: {tensor_parallel_size}")
    logger.info(f"GPU memory utilization: {gpu_memory_utilization}")
    logger.info(f"Max model length: {max_model_len if max_model_len else 'auto'}")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        logger.info("Tokenizer loaded")
    except Exception as e:
        logger.error(f"Failed to load tokenizer: {e}")
        raise
    
    vllm_kwargs = {
        "model": model_name,
        "tensor_parallel_size": tensor_parallel_size,
        "gpu_memory_utilization": gpu_memory_utilization,
        # Required for Qwen3 custom tokenizer code; pin model revisions to mitigate supply-chain risk
        "trust_remote_code": True,
        "dtype": "auto",
    }
    
    if max_model_len is not None:
        vllm_kwargs["max_model_len"] = max_model_len
    
    vllm_config = {
        'tensor_parallel_size': tensor_parallel_size,
        'gpu_memory_utilization': gpu_memory_utilization,
        'max_model_len': max_model_len,
        'dtype': 'auto'
    }
    
    try:
        logger.info("Loading model with vLLM (this may take a minute)...")
        llm = LLM(**vllm_kwargs)
        logger.info("Model loaded successfully with vLLM")
    except Exception as e:
        logger.error(f"Failed to load model with vLLM: {e}")
        raise
    
    model_info = detect_qwen3_model(model_name)
    
    model_metadata = {
        'name': model_name,
        'is_qwen3': model_info['is_qwen3'],
        'model_size': model_info['model_size'],
        'supports_thinking': model_info['supports_thinking']
    }
    
    logger.info("Model loaded and ready for inference")
    return llm, tokenizer, model_metadata, vllm_config


def run_prediction(model_name, prediction_dates, historical_days=2, tensor_parallel_size=1, 
                   gpu_memory_utilization=0.9, max_model_len=None, enable_thinking=None, thinking_budget=None):
    if enable_thinking is None:
        enable_thinking = ENABLE_THINKING
    
    if thinking_budget is None:
        thinking_budget = THINKING_BUDGET
    
    logger.info("\n")
    log_separator(logger, "STARTING PREDICTION RUN")
    logger.info(f"Model: {model_name}")
    logger.info(f"Prediction dates: {', '.join(prediction_dates)}")
    logger.info(f"Historical context: {historical_days} day(s)")
    logger.info(f"Thinking mode: {'ON' if enable_thinking else 'OFF'}")
    logger.info(f"Thinking budget: {thinking_budget if thinking_budget else 'Unlimited'}")
    logger.info(f"Inference engine: vLLM")
    log_separator(logger)
    
    llm, tokenizer, model_metadata, vllm_config = load_model(
        model_name, 
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len
    )
        
    model_metadata['enable_thinking'] = enable_thinking
    model_metadata['thinking_budget'] = thinking_budget
    
    run_metadata = create_run_metadata(model_name, model_metadata, vllm_config, historical_days, enable_thinking)
    run_metadata['thinking_budget'] = thinking_budget
    
    historical_data = None
    if historical_days > 0:
        first_pred_date = datetime.strptime(prediction_dates[0], "%Y%m%d")
        hist_end_date = first_pred_date - timedelta(days=1)
        hist_start_date = hist_end_date - timedelta(days=historical_days - 1)
        
        logger.info(f"\nLoading {historical_days} days of historical data for few-shot examples...")
        logger.info(f"  Historical range: {hist_start_date.strftime('%Y%m%d')} to {hist_end_date.strftime('%Y%m%d')}")
        
        historical_data = load_historical_data(
            hist_start_date.strftime("%Y%m%d"),
            hist_end_date.strftime("%Y%m%d")
        )
    
    prediction_filenames = [f"bee_{date}.json" for date in prediction_dates]
    logger.info(f"\nGenerating predictions for {len(prediction_filenames)} puzzles")
    
    predictions = process_bee_predictions(
        prediction_filenames,
        llm,
        tokenizer,
        RESULTS_PATH,
        historical_data=historical_data,
        model_metadata=model_metadata,
        run_metadata=run_metadata
    )
    
    logger.info("\nCleaning up vLLM resources...")
    del llm
    del tokenizer
    
    logger.info("\n")
    log_separator(logger, "PREDICTION RUN COMPLETED")

    return predictions

# ============================================================================
# ABLATION EXPERIMENT
# ============================================================================

def run_ablation_experiment():
    """
    LREC Paper Experiments:
    - Test 4 model sizes (4B, 8B, 14B, 32B)
    - Test thinking ON/OFF for ALL models
    - Test all 58 puzzles (June 2 - July 29, 2025)
    - Zero-shot only (no historical context)
    - Total: 8 configurations x 58 puzzles = 464 experiments
    """
    global logger
    
    logger.info("\n")
    log_separator(logger, "STARTING LREC ABLATION EXPERIMENT")
    logger.info(f"Models to test: {ABLATION_CONFIG['models_to_test']}")
    logger.info(f"Date range: {ABLATION_CONFIG['start_date']} to {ABLATION_CONFIG['end_date']}")
    logger.info(f"Zero-shot evaluation (no historical context)")
    log_separator(logger)
    
    start = datetime.strptime(ABLATION_CONFIG['start_date'], "%Y%m%d")
    end = datetime.strptime(ABLATION_CONFIG['end_date'], "%Y%m%d")
    all_dates = []
    current = start
    while current <= end:
        all_dates.append(current.strftime("%Y%m%d"))
        current += timedelta(days=1)
    
    logger.info(f"Total puzzles to evaluate: {len(all_dates)}")
    
    all_results = []
    
    for model_name in ABLATION_CONFIG['models_to_test']:
        model_size = model_name.split('-')[-1]  # e.g., "4B"
        
        for enable_thinking in ABLATION_CONFIG['thinking_modes_to_test']:
            thinking_mode_str = "ON" if enable_thinking else "OFF"
            
            date_range = (ABLATION_CONFIG['start_date'], ABLATION_CONFIG['end_date'])
            logger = setup_logging(model_name=model_name, thinking_mode=enable_thinking, date_range=date_range)
            
            logger.info("\n")
            logger.info("="*80)
            logger.info(f"EXPERIMENT: Model {model_size}, Thinking {thinking_mode_str}")
            logger.info(f"Evaluating {len(all_dates)} puzzles")
            logger.info("="*80)
            
            try:
                predictions = run_prediction(
                    model_name,
                    prediction_dates=all_dates,
                    historical_days=ABLATION_CONFIG['historical_days'],
                    tensor_parallel_size=ABLATION_CONFIG['tensor_parallel_size'],
                    gpu_memory_utilization=ABLATION_CONFIG['gpu_memory_utilization'],
                    max_model_len=ABLATION_CONFIG['max_model_len'],
                    enable_thinking=enable_thinking,
                    thinking_budget=ABLATION_CONFIG['thinking_budget']
                )
                
                if predictions:
                    for pred in predictions:
                        metrics = pred['metrics']
                        
                        result = {
                            'model': model_name,
                            'model_size': model_size,
                            'enable_thinking': enable_thinking,
                            'thinking_mode': thinking_mode_str,
                            'prediction_date': pred['date_file'].replace('bee_', '').replace('.json', ''),
                            'puzzle_id': pred['puzzle_id'],
                            'center_letter': pred['center_letter'],
                            'all_letters': pred['all_letters'],
                            'num_predicted': metrics['num_predicted'],
                            'num_actual': metrics['num_actual'],
                            'num_correct': metrics['true_positives'],
                            'precision': metrics['precision'],
                            'recall': metrics['recall'],
                            'f1_score': metrics['f1_score'],
                            'constraint_adherence_rate': metrics.get('constraint_adherence_rate', None),
                            'token_efficiency': metrics.get('token_efficiency', None),
                            'violation_breakdown': metrics.get('violation_breakdown', {}),
                            'error_analysis': metrics.get('error_analysis', {}),
                            'correctly_predicted_words': pred['correctly_predicted'],
                            'throughput': pred.get('throughput', {})
                        }
                        all_results.append(result)
                    
                    avg_precision = sum(r['precision'] for r in predictions if 'precision' in r['metrics']) / len(predictions)
                    avg_recall = sum(r['recall'] for r in predictions if 'recall' in r['metrics']) / len(predictions)
                    avg_f1 = sum(r['f1_score'] for r in predictions if 'f1_score' in r['metrics']) / len(predictions)
                    
                    logger.info(f"\nCompleted: Model {model_size}, Thinking {thinking_mode_str}")
                    logger.info(f"  Avg Precision: {avg_precision:.3f}")
                    logger.info(f"  Avg Recall:    {avg_recall:.3f}")
                    logger.info(f"  Avg F1 Score:  {avg_f1:.3f}")
                    logger.info(f"  Puzzles: {len(predictions)}/{len(all_dates)}")
                else:
                    logger.warning(f"No predictions returned for Model {model_size}, Thinking {thinking_mode_str}")
                    
            except Exception as e:
                logger.error(f"Error with Model {model_size}, Thinking {thinking_mode_str}: {e}")
                import traceback
                logger.error(traceback.format_exc())
                continue
    
    date_start = ABLATION_CONFIG['start_date']
    date_end = ABLATION_CONFIG['end_date']
    output_file = os.path.join(RESULTS_PATH, f'lrec_ablation_{date_start}-{date_end}_summary.json')
    
    grouped = defaultdict(list)
    for r in all_results:
        key = (r['model_size'], r['thinking_mode'])
        grouped[key].append(r)

    configs_summary = []
    for (model_size, thinking_mode), group_results in sorted(grouped.items()):
        num_puzzles = len(group_results)
        precisions = [r['precision'] for r in group_results]
        recalls = [r['recall'] for r in group_results]
        f1_scores = [r['f1_score'] for r in group_results]
        cars = [r.get('constraint_adherence_rate', 0) for r in group_results]
        token_effs = [r.get('token_efficiency', 0) for r in group_results]
        
        config_summary = {
            'model_size': model_size,
            'thinking_enabled': thinking_mode == 'ON',
            'num_puzzles': num_puzzles,
            'avg_precision': statistics.mean(precisions),
            'std_precision': statistics.stdev(precisions) if len(precisions) > 1 else 0,
            'avg_recall': statistics.mean(recalls),
            'std_recall': statistics.stdev(recalls) if len(recalls) > 1 else 0,
            'avg_f1': statistics.mean(f1_scores),
            'std_f1': statistics.stdev(f1_scores) if len(f1_scores) > 1 else 0,
            'avg_car': statistics.mean(cars),
            'std_car': statistics.stdev(cars) if len(cars) > 1 else 0,
            'avg_token_efficiency': statistics.mean(token_effs),
            'std_token_efficiency': statistics.stdev(token_effs) if len(token_effs) > 1 else 0,
            'results_file': f'qwen3_{model_size.lower()}_{thinking_mode.lower().replace(" ", "")}_{date_start}-{date_end}_results.json'
        }
        configs_summary.append(config_summary)
    
    ablation_summary = {
        'experiment': {
            'id': 'lrec_ablation_2025',
            'description': 'Zero-shot evaluation of Qwen3 models on NY Times Spelling Bee puzzles',
            'date_range': {'start': date_start, 'end': date_end, 'num_puzzles': len(all_dates)},
            'num_configurations': len(configs_summary),
            'total_experiments': len(all_results),
            'generated_at': datetime.now().isoformat()
        },
        'configurations': configs_summary
    }
    
    with open(output_file, 'w') as f:
        json.dump(ablation_summary, f, indent=2)
    
    logger.info("\n")
    log_separator(logger, "LREC ABLATION EXPERIMENT COMPLETED")
    logger.info(f"Total experiments: {len(all_results)}")
    logger.info(f"Puzzles evaluated: {len(all_dates)}")
    logger.info(f"Results saved to: {output_file}")
    log_separator(logger)
    
    print_ablation_summary(all_results)
    
    return all_results


def print_ablation_summary(results):
    """Print summary table for LREC ablation results"""
    if not results:
        logger.warning("No results to summarize")
        return
    
    logger.info("\n")
    log_separator(logger, "LREC ABLATION SUMMARY")
    logger.info("")
    
    grouped = defaultdict(list)
    for r in results:
        key = (r['model_size'], r['thinking_mode'])
        grouped[key].append(r)

    logger.info("Model | Think | Puzzles | Avg P   | Avg R   | Avg F1  | Const.Adh | Token.Eff")
    logger.info("-" * 90)
    
    for (model_size, thinking_mode), group_results in sorted(grouped.items()):
        num_puzzles = len(group_results)
        avg_precision = sum(r['precision'] for r in group_results) / num_puzzles
        avg_recall = sum(r['recall'] for r in group_results) / num_puzzles
        avg_f1 = sum(r['f1_score'] for r in group_results) / num_puzzles
        avg_adherence = sum(r.get('constraint_adherence_rate', 0) for r in group_results) / num_puzzles
        avg_efficiency = sum(r.get('token_efficiency', 0) for r in group_results) / num_puzzles
        
        logger.info(
            f"{model_size:^5} | "
            f"{thinking_mode:^5} | "
            f"{num_puzzles:^7} | "
            f"{avg_precision:^7.3f} | "
            f"{avg_recall:^7.3f} | "
            f"{avg_f1:^7.3f} | "
            f"{avg_adherence:^9.3f} | "
            f"{avg_efficiency:^9.2f}"
        )
    
    logger.info("=" * 90)
    
    logger.info("\n")
    log_separator(logger, "ERROR ANALYSIS SUMMARY")
    
    for (model_size, thinking_mode), group_results in sorted(grouped.items()):
        logger.info(f"\nModel {model_size}, Thinking {thinking_mode}:")
        
        total_violations = sum(r.get('violation_breakdown', {}).get('missing_center', 0) + 
                              r.get('violation_breakdown', {}).get('forbidden_letters', 0) + 
                              r.get('violation_breakdown', {}).get('too_short', 0) 
                              for r in group_results)
        
        total_repetitions = sum(len(r.get('error_analysis', {}).get('repetitions', [])) for r in group_results)
        total_non_dict = sum(len(r.get('error_analysis', {}).get('non_dictionary_words', [])) for r in group_results)
        
        logger.info(f"  Constraint violations: {total_violations}")
        logger.info(f"  Repetitions: {total_repetitions}")
        logger.info(f"  Non-dictionary words: {total_non_dict}")
    
    logger.info("\n" + "="*90)
    logger.info("")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print(f"Spelling Bee inference (mode={RUN_MODE})")

    if RUN_MODE == "single":
        results = run_prediction(
            SINGLE_CONFIG['model'],
            prediction_dates=SINGLE_CONFIG['prediction_dates'],
            historical_days=SINGLE_CONFIG['historical_days'],
            tensor_parallel_size=SINGLE_CONFIG['tensor_parallel_size'],
            gpu_memory_utilization=SINGLE_CONFIG['gpu_memory_utilization'],
            max_model_len=SINGLE_CONFIG['max_model_len'],
            enable_thinking=SINGLE_CONFIG['enable_thinking'],
            thinking_budget=SINGLE_CONFIG['thinking_budget']
        )

    elif RUN_MODE == "ablation":
        results = run_ablation_experiment()

    else:
        print(f"Error: Unknown RUN_MODE '{RUN_MODE}'. Use 'single' or 'ablation'.")

    print("Done.")

