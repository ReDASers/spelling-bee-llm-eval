"""Claude Haiku Spelling Bee inference across 58 puzzles with thinking budget ablation.

Usage: export ANTHROPIC_API_KEY='...' && python run_claude_inference.py
"""

import os
import json
import time
import asyncio
from datetime import datetime, timedelta
from anthropic import Anthropic, AsyncAnthropic, RateLimitError
from tqdm import tqdm
from tqdm.asyncio import tqdm as async_tqdm
import traceback

# --- Configuration ---

BEE_DATA_PATH = "./Bee-Daily-Pull/"
RESULTS_PATH = "./claude-results/"
LOG_PATH = "./claude-logs/"

START_DATE = "20250602"
END_DATE = "20250729"

MIN_WORD_LENGTH = 4
TOTAL_ALPHABET_SIZE = 7

CLAUDE_MODEL = "claude-haiku-4-5-20251001"

# (thinking_budget, max_tokens) pairs
ABLATION_CONFIGS = [
    (16384, 20480),
    (8192, 12288),
    (4096, 8192),
]

NON_THINKING_ABLATION_CONFIGS = [
    20480,
    12288,
    8192,
]

DEFAULT_THINKING_BUDGET = 16384
DEFAULT_MAX_TOKENS = 20480

# Concurrency tuned for Anthropic Tier 1 (50 RPM). Increase for higher tiers.
BATCH_SIZE = 5
MAX_CONCURRENT_REQUESTS = 5

# Claude defaults; extended thinking forces temperature=1.0
THINKING_PARAMS = {}

NON_THINKING_PARAMS = {}

# --- Data Loading ---

def load_bee_data(filename):
    with open(filename, 'r') as f:
        return json.load(f)


def get_all_puzzle_dates(start_date, end_date):
    start = datetime.strptime(start_date, "%Y%m%d")
    end = datetime.strptime(end_date, "%Y%m%d")
    
    dates = []
    current = start
    while current <= end:
        dates.append(current.strftime("%Y%m%d"))
        current += timedelta(days=1)
    
    return dates


def extract_puzzle_letters(words):
    if not words:
        return set()
    return set(''.join(words).lower())


def identify_center_letter(words):
    """Find the letter present in every word; fall back to most frequent."""
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


# --- Word Validation ---

def is_valid_bee_word(word, center_letter, allowed_letters):
    word = word.strip().lower()
    return (word
            and word.isalpha()
            and len(word) >= MIN_WORD_LENGTH
            and center_letter in word
            and all(c in allowed_letters for c in word))


def parse_word_predictions(generated_text, center_letter, allowed_letters):
    lines = generated_text.strip().split('\n')
    
    valid_words = []
    seen = set()
    
    for line in lines:
        word = line.strip().lower()
        
        if not word:
            continue
        
        if ',' in word:
            word_candidates = [w.strip() for w in word.split(',')]
        else:
            word_candidates = [word]
        
        for candidate in word_candidates:
            candidate = ''.join(c for c in candidate if c.isalpha())
            
            if not candidate:
                continue
            
            if is_valid_bee_word(candidate, center_letter, allowed_letters):
                if candidate not in seen:
                    seen.add(candidate)
                    valid_words.append(candidate)
    
    return valid_words


# --- Prompt Creation ---

def create_word_prediction_prompt(center_letter, outer_letters):
    all_letters = sorted([center_letter] + outer_letters)
    
    letters_display = ', '.join(all_letters).upper()
    letters_bar = ' | '.join(all_letters).upper()
    center_upper = center_letter.upper()
    
    return f"""You are an expert at solving NY Times Spelling Bee puzzles.

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


# --- Claude API Integration ---

class ClaudeGenerator:

    def __init__(self, api_key=None):
        if api_key is None:
            api_key = os.getenv('ANTHROPIC_API_KEY')
        
        if not api_key:
            raise ValueError(
                "ANTHROPIC_API_KEY not found. Set it with:\n"
                "  export ANTHROPIC_API_KEY='your_api_key_here'"
            )
        
        self.client = Anthropic(api_key=api_key)
        self.model = CLAUDE_MODEL
    
    def generate(self, prompt, enable_thinking=False, thinking_budget=None, max_tokens=None):
        """Returns (answer_text, generation_metadata)."""
        if thinking_budget is None:
            thinking_budget = DEFAULT_THINKING_BUDGET
        if max_tokens is None:
            max_tokens = DEFAULT_MAX_TOKENS
        
        params = THINKING_PARAMS.copy() if enable_thinking else NON_THINKING_PARAMS.copy()

        messages = [
            {"role": "user", "content": prompt}
        ]

        start_time = time.time()

        try:
            api_params = {
                'model': self.model,
                'max_tokens': max_tokens,
                'messages': messages,
            }
            
            if 'temperature' in params:
                api_params['temperature'] = params['temperature']

            if enable_thinking:
                api_params['thinking'] = {
                    'type': 'enabled', 
                    'budget_tokens': thinking_budget
                }
            
            response = self.client.messages.create(**api_params)

            generation_time = time.time() - start_time

            raw_output = self._extract_content(response)
            thinking_trace = self._extract_thinking(response) if enable_thinking else ""
            answer_text = self._extract_answer_text(response)
            
            metadata = {
                'raw_output': raw_output,
                'thinking_trace': thinking_trace if enable_thinking else "",
                'final_answer': answer_text,
                'tokens_generated': response.usage.output_tokens,
                'generation_time': generation_time,
                'tokens_per_second': response.usage.output_tokens / generation_time if generation_time > 0 else 0,
                'prompt_tokens': response.usage.input_tokens,
                'model': self.model,
                'stop_reason': response.stop_reason,
                'thinking_enabled': enable_thinking,
                'thinking_budget': thinking_budget if enable_thinking else None,
                'max_tokens': max_tokens
            }

            return answer_text, metadata

        except Exception as e:
            print(f"\n[ERROR] Error calling Claude API: {e}")
            raise

    def _extract_content(self, response):
        if not response.content:
            return ""

        parts = []
        for block in response.content:
            if block.type == 'thinking':
                parts.append(f"<think>{block.thinking}</think>")
            elif block.type == 'text':
                parts.append(block.text)
            elif block.type == 'redacted_thinking':
                parts.append("<redacted_thinking>")
        
        return '\n'.join(parts)
    
    def _extract_thinking(self, response):
        if not response.content:
            return ""
        
        thinking_parts = []
        for block in response.content:
            if block.type == 'thinking':
                thinking_parts.append(block.thinking)
        
        return '\n'.join(thinking_parts)
    
    def _extract_answer_text(self, response):
        if not response.content:
            return ""
        
        text_parts = []
        for block in response.content:
            if block.type == 'text':
                text_parts.append(block.text)
        
        return '\n'.join(text_parts)


class AsyncClaudeGenerator:
    """Async ClaudeGenerator with semaphore-based rate limiting."""

    def __init__(self, api_key=None, max_concurrent=MAX_CONCURRENT_REQUESTS):
        if api_key is None:
            api_key = os.getenv('ANTHROPIC_API_KEY')
        
        if not api_key:
            raise ValueError(
                "ANTHROPIC_API_KEY not found. Set it with:\n"
                "  export ANTHROPIC_API_KEY='your_api_key_here'"
            )
        
        self.client = AsyncAnthropic(api_key=api_key)
        self.model = CLAUDE_MODEL
        self.semaphore = asyncio.Semaphore(max_concurrent)
    
    async def generate(self, prompt, enable_thinking=False, thinking_budget=None, max_tokens=None, max_retries=3):
        """Returns (answer_text, metadata) with exponential backoff on rate limits."""
        if thinking_budget is None:
            thinking_budget = DEFAULT_THINKING_BUDGET
        if max_tokens is None:
            max_tokens = DEFAULT_MAX_TOKENS
        
        params = THINKING_PARAMS.copy() if enable_thinking else NON_THINKING_PARAMS.copy()

        messages = [
            {"role": "user", "content": prompt}
        ]

        async with self.semaphore:
            for attempt in range(max_retries):
                start_time = time.time()

                try:
                    api_params = {
                        'model': self.model,
                        'max_tokens': max_tokens,
                        'messages': messages,
                    }
                    
                    if 'temperature' in params:
                        api_params['temperature'] = params['temperature']

                    if enable_thinking:
                        api_params['thinking'] = {
                            'type': 'enabled', 
                            'budget_tokens': thinking_budget
                        }
                    
                    response = await self.client.messages.create(**api_params)

                    generation_time = time.time() - start_time

                    raw_output = self._extract_content(response)
                    thinking_trace = self._extract_thinking(response) if enable_thinking else ""
                    answer_text = self._extract_answer_text(response)
                    
                    metadata = {
                        'raw_output': raw_output,
                        'thinking_trace': thinking_trace if enable_thinking else "",
                        'final_answer': answer_text,
                        'tokens_generated': response.usage.output_tokens,
                        'generation_time': generation_time,
                        'tokens_per_second': response.usage.output_tokens / generation_time if generation_time > 0 else 0,
                        'prompt_tokens': response.usage.input_tokens,
                        'model': self.model,
                        'stop_reason': response.stop_reason,
                        'thinking_enabled': enable_thinking,
                        'thinking_budget': thinking_budget if enable_thinking else None,
                        'max_tokens': max_tokens,
                        'retry_attempt': attempt
                    }

                    return answer_text, metadata

                except RateLimitError as e:
                    if attempt < max_retries - 1:
                        wait_time = (2 ** attempt) * 2  # 2, 4, 8 seconds
                        print(f"\n[WARN] Rate limit hit, waiting {wait_time}s before retry {attempt + 1}/{max_retries}...")
                        await asyncio.sleep(wait_time)
                    else:
                        print(f"\n[ERROR] Rate limit error after {max_retries} retries: {e}")
                        raise
                        
                except Exception as e:
                    print(f"\n[ERROR] Error calling Claude API: {e}")
                    raise

    def _extract_content(self, response):
        if not response.content:
            return ""
        
        parts = []
        for block in response.content:
            if block.type == 'thinking':
                parts.append(f"<think>{block.thinking}</think>")
            elif block.type == 'text':
                parts.append(block.text)
            elif block.type == 'redacted_thinking':
                parts.append("<redacted_thinking>")
        
        return '\n'.join(parts)
    
    def _extract_thinking(self, response):
        if not response.content:
            return ""
        
        thinking_parts = []
        for block in response.content:
            if block.type == 'thinking':
                thinking_parts.append(block.thinking)
        
        return '\n'.join(thinking_parts)
    
    def _extract_answer_text(self, response):
        if not response.content:
            return ""
        
        text_parts = []
        for block in response.content:
            if block.type == 'text':
                text_parts.append(block.text)
        
        return '\n'.join(text_parts)


# --- File Storage ---

def save_structured_logs(logs_data, enable_thinking, thinking_budget, max_tokens, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    if enable_thinking:
        mode = 'thinking'
        budget_str = f"_{thinking_budget}"
    else:
        mode = 'nothinking'
        budget_str = f"_{max_tokens}"
    
    filename = f"claude_haiku_{mode}{budget_str}_logs.json"
    filepath = os.path.join(output_dir, filename)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(logs_data, f, indent=2, ensure_ascii=False)
    
    print(f"[OK] Logs saved: {filepath}")
    return filepath


def save_results(results_data, enable_thinking, thinking_budget, max_tokens, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    if enable_thinking:
        mode = 'thinking'
        budget_str = f"_{thinking_budget}"
    else:
        mode = 'nothinking'
        budget_str = f"_{max_tokens}"
    
    filename = f"claude_haiku_{mode}{budget_str}_results.json"
    filepath = os.path.join(output_dir, filename)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(results_data, f, indent=2, ensure_ascii=False)
    
    print(f"[OK] Results saved: {filepath}")
    return filepath


# --- Main Inference Pipeline ---

def run_claude_inference(enable_thinking=False, thinking_budget=None, max_tokens=None):
    """Sequential inference on all 58 puzzles. Returns (logs_path, results_path)."""
    if thinking_budget is None:
        thinking_budget = DEFAULT_THINKING_BUDGET
    if max_tokens is None:
        max_tokens = DEFAULT_MAX_TOKENS
    
    print("\n" + "="*80)
    print("NY TIMES SPELLING BEE - CLAUDE MODEL INFERENCE")
    print("="*80)
    print(f"Model: {CLAUDE_MODEL}")
    print(f"Thinking mode: {'ON (Extended Thinking)' if enable_thinking else 'OFF'}")
    if enable_thinking:
        print(f"Thinking budget: {thinking_budget} tokens")
    print(f"Max tokens: {max_tokens}")
    print(f"Date range: {START_DATE} to {END_DATE}")
    print("="*80)
    
    generator = ClaudeGenerator()
    all_dates = get_all_puzzle_dates(START_DATE, END_DATE)
    print(f"\nProcessing {len(all_dates)} puzzles sequentially...")
    
    logs_puzzles = []
    results_predictions = []

    for date in tqdm(all_dates, desc="Processing puzzles"):
        filename = f"bee_{date}.json"
        filepath = os.path.join(BEE_DATA_PATH, filename)
        
        try:
            puzzle_data = load_bee_data(filepath)
        except FileNotFoundError:
            print(f"\nWarning: {filename} not found, skipping...")
            continue
        
        actual_words = list(puzzle_data['answers'].keys())
        all_letters = extract_puzzle_letters(actual_words)
        center_letter = identify_center_letter(actual_words)
        
        if not center_letter:
            print(f"\nWarning: Could not identify center letter for {filename}, skipping...")
            continue
        
        outer_letters = sorted(all_letters - {center_letter})
        
        prompt = create_word_prediction_prompt(center_letter, outer_letters)

        try:
            answer_text, metadata = generator.generate(
                prompt, enable_thinking, thinking_budget, max_tokens
            )
        except Exception as e:
            print(f"\nError generating for {filename}: {e}")
            continue
        
        predicted_words = parse_word_predictions(
            answer_text, center_letter, all_letters
        )
        
        predicted_set = set(predicted_words)
        actual_set = set(actual_words)
        correctly_predicted = sorted(predicted_set & actual_set)
        missed_words = sorted(actual_set - predicted_set)
        false_positives = sorted(predicted_set - actual_set)

        log_entry = {
            'puzzle_id': puzzle_data['id'],
            'date': date,
            'center_letter': center_letter,
            'all_letters': sorted(all_letters),
            'prompt': prompt,
            'generation': metadata
        }
        logs_puzzles.append(log_entry)
        
        result_entry = {
            'puzzle_id': puzzle_data['id'],
            'date': date,
            'center_letter': center_letter,
            'all_letters': sorted(all_letters),
            'predicted_words': predicted_words,
            'actual_words': sorted(actual_words),
            'correctly_predicted': correctly_predicted,
            'missed_words': missed_words,
            'false_positives': false_positives
        }
        results_predictions.append(result_entry)
    
    metadata = {
        'model_name': CLAUDE_MODEL,
        'model_family': 'claude',
        'model_size': 'haiku-4.5',
        'thinking_enabled': enable_thinking,
        'thinking_budget': thinking_budget if enable_thinking else None,
        'max_tokens': max_tokens,
        'sampling_params': THINKING_PARAMS if enable_thinking else NON_THINKING_PARAMS,
        'date_range': {
            'start': START_DATE,
            'end': END_DATE,
            'num_puzzles': len(all_dates)
        },
        'generated_at': datetime.now().isoformat()
    }
    
    logs_data = {
        'metadata': metadata,
        'puzzles': logs_puzzles
    }
    logs_filepath = save_structured_logs(logs_data, enable_thinking, thinking_budget, max_tokens, LOG_PATH)
    
    results_data = {
        'metadata': metadata,
        'predictions': results_predictions
    }
    results_filepath = save_results(results_data, enable_thinking, thinking_budget, max_tokens, RESULTS_PATH)
    
    print("\n" + "="*80)
    print("INFERENCE COMPLETED")
    print(f"Puzzles processed: {len(results_predictions)}/{len(all_dates)}")
    print(f"Logs: {logs_filepath}")
    print(f"Results: {results_filepath}")
    print("="*80)
    
    return logs_filepath, results_filepath


async def run_claude_inference_batched(enable_thinking=False, thinking_budget=None, max_tokens=None, batch_size=BATCH_SIZE):
    """Async batched inference. Returns (logs_path, results_path)."""
    if thinking_budget is None:
        thinking_budget = DEFAULT_THINKING_BUDGET
    if max_tokens is None:
        max_tokens = DEFAULT_MAX_TOKENS
    
    print("\n" + "="*80)
    print("NY TIMES SPELLING BEE - CLAUDE MODEL INFERENCE (BATCHED)")
    print("="*80)
    print(f"Model: {CLAUDE_MODEL}")
    print(f"Thinking mode: {'ON (Extended Thinking)' if enable_thinking else 'OFF'}")
    if enable_thinking:
        print(f"Thinking budget: {thinking_budget} tokens")
    print(f"Max tokens: {max_tokens}")
    print(f"Batch size: {batch_size} concurrent requests")
    print(f"Date range: {START_DATE} to {END_DATE}")
    print("="*80)
    
    generator = AsyncClaudeGenerator(max_concurrent=batch_size)
    all_dates = get_all_puzzle_dates(START_DATE, END_DATE)
    print(f"\nProcessing {len(all_dates)} puzzles in batches of {batch_size}...")
    
    puzzle_tasks = []
    
    for date in all_dates:
        filename = f"bee_{date}.json"
        filepath = os.path.join(BEE_DATA_PATH, filename)
        
        try:
            puzzle_data = load_bee_data(filepath)
        except FileNotFoundError:
            print(f"\nWarning: {filename} not found, skipping...")
            continue
        
        actual_words = list(puzzle_data['answers'].keys())
        all_letters = extract_puzzle_letters(actual_words)
        center_letter = identify_center_letter(actual_words)
        
        if not center_letter:
            print(f"\nWarning: Could not identify center letter for {filename}, skipping...")
            continue
        
        outer_letters = sorted(all_letters - {center_letter})
        
        prompt = create_word_prediction_prompt(center_letter, outer_letters)
        
        puzzle_tasks.append((date, puzzle_data, center_letter, all_letters, prompt))
    
    print(f"[OK] Built {len(puzzle_tasks)} prompts")
    
    async def process_puzzle(task):
        date, puzzle_data, center_letter, all_letters, prompt = task
        
        try:
            answer_text, metadata = await generator.generate(
                prompt, enable_thinking, thinking_budget, max_tokens
            )
            
            predicted_words = parse_word_predictions(
                answer_text, center_letter, all_letters
            )

            actual_words = list(puzzle_data['answers'].keys())
            predicted_set = set(predicted_words)
            actual_set = set(actual_words)
            correctly_predicted = sorted(predicted_set & actual_set)
            missed_words = sorted(actual_set - predicted_set)
            false_positives = sorted(predicted_set - actual_set)
            
            return {
                'success': True,
                'log_entry': {
                    'puzzle_id': puzzle_data['id'],
                    'date': date,
                    'center_letter': center_letter,
                    'all_letters': sorted(all_letters),
                    'prompt': prompt,
                    'generation': metadata
                },
                'result_entry': {
                    'puzzle_id': puzzle_data['id'],
                    'date': date,
                    'center_letter': center_letter,
                    'all_letters': sorted(all_letters),
                    'predicted_words': predicted_words,
                    'actual_words': sorted(actual_words),
                    'correctly_predicted': correctly_predicted,
                    'missed_words': missed_words,
                    'false_positives': false_positives
                }
            }
        except Exception as e:
            print(f"\n[ERROR] Error processing puzzle {date}: {e}")
            return {'success': False, 'date': date, 'error': str(e)}
    
    results = []
    for i in async_tqdm(range(0, len(puzzle_tasks), batch_size), desc="Processing batches"):
        batch = puzzle_tasks[i:i+batch_size]
        batch_results = await asyncio.gather(*[process_puzzle(task) for task in batch])
        results.extend(batch_results)
    
    logs_puzzles = []
    results_predictions = []
    failed_count = 0
    
    for result in results:
        if result['success']:
            logs_puzzles.append(result['log_entry'])
            results_predictions.append(result['result_entry'])
        else:
            failed_count += 1
            print(f"\nFailed puzzle {result['date']}: {result['error']}")
    
    if failed_count > 0:
        print(f"\n[WARN] {failed_count} puzzles failed to process")
    
    metadata = {
        'model_name': CLAUDE_MODEL,
        'model_family': 'claude',
        'model_size': 'haiku-4.5',
        'thinking_enabled': enable_thinking,
        'thinking_budget': thinking_budget if enable_thinking else None,
        'max_tokens': max_tokens,
        'batch_size': batch_size,
        'sampling_params': THINKING_PARAMS if enable_thinking else NON_THINKING_PARAMS,
        'date_range': {
            'start': START_DATE,
            'end': END_DATE,
            'num_puzzles': len(all_dates)
        },
        'generated_at': datetime.now().isoformat()
    }
    
    logs_data = {
        'metadata': metadata,
        'puzzles': logs_puzzles
    }
    logs_filepath = save_structured_logs(logs_data, enable_thinking, thinking_budget, max_tokens, LOG_PATH)
    
    results_data = {
        'metadata': metadata,
        'predictions': results_predictions
    }
    results_filepath = save_results(results_data, enable_thinking, thinking_budget, max_tokens, RESULTS_PATH)
    
    print("\n" + "="*80)
    print("BATCHED INFERENCE COMPLETED")
    print(f"Puzzles processed: {len(results_predictions)}/{len(all_dates)}")
    print(f"Logs: {logs_filepath}")
    print(f"Results: {results_filepath}")
    print("="*80)
    
    return logs_filepath, results_filepath


# --- Run All Configurations ---

def run_claude_inference_wrapper(enable_thinking=False, thinking_budget=None, max_tokens=None, use_batching=True, batch_size=BATCH_SIZE):
    """Dispatch to sequential or batched inference."""
    if use_batching:
        return asyncio.run(run_claude_inference_batched(
            enable_thinking, thinking_budget, max_tokens, batch_size
        ))
    else:
        return run_claude_inference(
            enable_thinking, thinking_budget, max_tokens
        )


def run_all_claude_experiments(run_ablations=True, use_batching=True, batch_size=BATCH_SIZE):
    """Run non-thinking + thinking configs, optionally with budget ablations."""
    print("\n" + "="*80)
    print("RUNNING ALL CLAUDE EXPERIMENTS")
    if use_batching:
        print("(Using BATCHED async processing for 10x speedup!)")
    print("="*80)
    print(f"Model: {CLAUDE_MODEL}")
    if use_batching:
        print(f"Batch size: {batch_size} concurrent requests")
    
    configurations = []

    if run_ablations:
        for max_tokens in NON_THINKING_ABLATION_CONFIGS:
            configurations.append({
                'enable_thinking': False,
                'thinking_budget': None,
                'max_tokens': max_tokens,
                'name': f'Non-Thinking (max={max_tokens})'
            })
        
        for thinking_budget, max_tokens in ABLATION_CONFIGS:
            configurations.append({
                'enable_thinking': True,
                'thinking_budget': thinking_budget,
                'max_tokens': max_tokens,
                'name': f'Thinking (budget={thinking_budget})'
            })
        
        total_configs = 6  # 3 non-thinking + 3 thinking budgets
    else:
        configurations.append({
            'enable_thinking': False,
            'thinking_budget': None,
            'max_tokens': DEFAULT_MAX_TOKENS,
            'name': 'Non-Thinking'
        })
        configurations.append({
            'enable_thinking': True,
            'thinking_budget': DEFAULT_THINKING_BUDGET,
            'max_tokens': DEFAULT_MAX_TOKENS,
            'name': f'Thinking (budget={DEFAULT_THINKING_BUDGET})'
        })
        total_configs = 2
    
    print(f"Configurations: {total_configs}")
    if run_ablations:
        print(f"Non-thinking budgets: {NON_THINKING_ABLATION_CONFIGS}")
        print(f"Thinking budgets: {[f'{b}->{m}' for b, m in ABLATION_CONFIGS]}")
    print(f"Puzzles per configuration: 58")
    print(f"Total API calls: {total_configs * 58}")
    print("="*80)
    
    all_results = []
    start_time = time.time()
    
    for config_idx, config in enumerate(configurations, 1):
        print(f"\n{'='*80}")
        print(f"CONFIGURATION {config_idx}/{total_configs}: {config['name']}")
        if config['enable_thinking']:
            print(f"  Thinking budget: {config['thinking_budget']} tokens")
            print(f"  Max tokens: {config['max_tokens']} tokens")
        print(f"{'='*80}")
        
        try:
            logs_path, results_path = run_claude_inference_wrapper(
                enable_thinking=config['enable_thinking'],
                thinking_budget=config['thinking_budget'],
                max_tokens=config['max_tokens'],
                use_batching=use_batching,
                batch_size=batch_size
            )
            
            all_results.append({
                'config_name': config['name'],
                'thinking_enabled': config['enable_thinking'],
                'thinking_budget': config['thinking_budget'],
                'max_tokens': config['max_tokens'],
                'status': 'success',
                'logs_path': logs_path,
                'results_path': results_path,
                'completed_at': datetime.now().isoformat()
            })
            
            print(f"\n[OK] {config['name']} COMPLETED")
            
        except Exception as e:
            print(f"\n[ERROR] {config['name']} FAILED: {e}")
            
            all_results.append({
                'config_name': config['name'],
                'thinking_enabled': config['enable_thinking'],
                'thinking_budget': config['thinking_budget'],
                'max_tokens': config['max_tokens'],
                'status': 'failed',
                'error': str(e),
                'failed_at': datetime.now().isoformat()
            })
            
            traceback.print_exc()
    
    total_time = time.time() - start_time
    hours = int(total_time // 3600)
    minutes = int((total_time % 3600) // 60)
    seconds = int(total_time % 60)
    
    completed = sum(1 for r in all_results if r['status'] == 'success')
    failed = sum(1 for r in all_results if r['status'] == 'failed')
    
    print("\n" + "="*80)
    print("ALL CLAUDE EXPERIMENTS COMPLETED")
    print("="*80)
    print(f"Total configurations: {total_configs}")
    print(f"Completed successfully: {completed}")
    print(f"Failed: {failed}")
    print(f"Total time: {hours}h {minutes}m {seconds}s")
    print("="*80)
    
    print("\nDETAILED RESULTS:")
    print("-" * 80)
    for i, result in enumerate(all_results, 1):
        status_symbol = "[OK]" if result['status'] == 'success' else "[FAIL]"
        budget_info = f" (budget={result['thinking_budget']})" if result['thinking_budget'] else ""
        print(f"{i}. {status_symbol} {result['config_name']}{budget_info} - {result['status']}")
        if result['status'] == 'success':
            print(f"     Logs: {result['logs_path']}")
            print(f"     Results: {result['results_path']}")
    print("-" * 80)
    
    summary_path = os.path.join(RESULTS_PATH, 'claude_experiment_summary.json')
    os.makedirs(RESULTS_PATH, exist_ok=True)
    
    summary = {
        'experiment': {
            'model': CLAUDE_MODEL,
            'run_ablations': run_ablations,
            'use_batching': use_batching,
            'batch_size': batch_size if use_batching else None,
            'ablation_configs': ABLATION_CONFIGS if run_ablations else None,
            'total_configurations': total_configs,
            'completed': completed,
            'failed': failed,
            'total_time_seconds': total_time,
            'completed_at': datetime.now().isoformat()
        },
        'configurations': all_results
    }
    
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nExperiment summary saved to: {summary_path}")
    
    return summary


# --- Main ---

if __name__ == "__main__":
    print("\n" + "="*80)
    print("CLAUDE SPELLING BEE INFERENCE SCRIPT")
    print("="*80)
    print("\nThis script runs Claude 4.5 haiku on the same 58 puzzles as run_inference.py")
    print("allowing direct comparison with Qwen models.")
    print("\nSEQUENTIAL PROCESSING (THINKING EXPERIMENTS ONLY)")
    print("   - No batching (avoids rate limits)")
    print("   - Only extended thinking configurations")
    print("\nThinking configurations:")
    for thinking_budget, max_tokens in ABLATION_CONFIGS:
        print(f"    - Thinking: {thinking_budget} tokens -> Max: {max_tokens} tokens")
    print("\nEnsure your ANTHROPIC_API_KEY environment variable is set:")
    print("  export ANTHROPIC_API_KEY='your_api_key_here'")
    print("\nRunning THINKING experiments ONLY (3 configurations, sequential)...")
    print("="*80)
    
    all_results = []
    start_time = time.time()
    
    for config_idx, (thinking_budget, max_tokens) in enumerate(ABLATION_CONFIGS, 1):
        print(f"\n{'='*80}")
        print(f"CONFIGURATION {config_idx}/3: Thinking Budget = {thinking_budget}")
        print(f"  Extended thinking: {thinking_budget} tokens")
        print(f"  Max tokens: {max_tokens} tokens")
        print(f"{'='*80}")
        
        try:
            logs_path, results_path = run_claude_inference_wrapper(
                enable_thinking=True,
                thinking_budget=thinking_budget,
                max_tokens=max_tokens,
                use_batching=False  # SEQUENTIAL - No batching to avoid rate limits
            )
            
            all_results.append({
                'config_name': f'Thinking ({thinking_budget})',
                'thinking_enabled': True,
                'thinking_budget': thinking_budget,
                'max_tokens': max_tokens,
                'status': 'success',
                'logs_path': logs_path,
                'results_path': results_path,
                'completed_at': datetime.now().isoformat()
            })
            
            print(f"\n[OK] Thinking ({thinking_budget}) COMPLETED")
            
        except Exception as e:
            print(f"\n[ERROR] Thinking ({thinking_budget}) FAILED: {e}")
            
            all_results.append({
                'config_name': f'Thinking ({thinking_budget})',
                'thinking_enabled': True,
                'thinking_budget': thinking_budget,
                'max_tokens': max_tokens,
                'status': 'failed',
                'error': str(e),
                'failed_at': datetime.now().isoformat()
            })
            
            traceback.print_exc()
    
    total_time = time.time() - start_time
    hours = int(total_time // 3600)
    minutes = int((total_time % 3600) // 60)
    seconds = int(total_time % 60)
    
    completed = sum(1 for r in all_results if r['status'] == 'success')
    failed = sum(1 for r in all_results if r['status'] == 'failed')
    
    print("\n" + "="*80)
    print("ALL THINKING EXPERIMENTS COMPLETED")
    print("="*80)
    print(f"Total configurations: 3")
    print(f"Completed successfully: {completed}")
    print(f"Failed: {failed}")
    print(f"Total time: {hours}h {minutes}m {seconds}s")
    print("="*80)
    
    print("\nDETAILED RESULTS:")
    print("-" * 80)
    for i, result in enumerate(all_results, 1):
        status_symbol = "[OK]" if result['status'] == 'success' else "[FAIL]"
        print(f"{i}. {status_symbol} {result['config_name']} - {result['status']}")
        if result['status'] == 'success':
            print(f"     Logs: {result['logs_path']}")
            print(f"     Results: {result['results_path']}")
    print("-" * 80)
    
    print("\n" + "="*80)
    print("ALL DONE! Results saved to:")
    print(f"  Logs: {LOG_PATH}")
    print(f"  Results: {RESULTS_PATH}")
    print("\nFiles created:")
    print("  Thinking:")
    for thinking_budget, _ in ABLATION_CONFIGS:
        print(f"    - claude_haiku_thinking_{thinking_budget}_logs.json / results.json")
    print("="*80)

