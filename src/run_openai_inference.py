"""
NY Times Spelling Bee - OpenAI GPT Model Inference Script

Runs GPT-5 reasoning model inference on all 58 puzzles with and without extended reasoning.

USAGE:
    export OPENAI_API_KEY='your_api_key_here'
    python run_openai_inference.py

Results saved to ./openai-logs/ and ./openai-results/
"""

import os
import json
import time
import asyncio
import sys
from datetime import datetime, timedelta
from openai import OpenAI, AsyncOpenAI, RateLimitError
from tqdm import tqdm
from tqdm.asyncio import tqdm as async_tqdm

if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# ============================================================================
# CONFIGURATION
# ============================================================================

BEE_DATA_PATH = "./Bee-Daily-Pull/"
RESULTS_PATH = "./openai-results/"
LOG_PATH = "./openai-logs/"

START_DATE = "20250602"
END_DATE = "20250729"

MIN_WORD_LENGTH = 4
TOTAL_ALPHABET_SIZE = 7

OPENAI_REASONING_MODEL = "gpt-5-mini"
OPENAI_NON_REASONING_MODEL = "gpt-4.1"

REASONING_EFFORT = "low"

ABLATION_CONFIGS = [
    ("low", 4096),
    ("medium", 8192),
    ("high", 16384),
]
NON_REASONING_ABLATION_CONFIGS = []

DEFAULT_REASONING_EFFORT = "medium"
DEFAULT_MAX_TOKENS = 32768

BATCH_SIZE = 12
MAX_CONCURRENT_REQUESTS = 12

REASONING_PARAMS = {}
NON_REASONING_PARAMS = {}

# ============================================================================
# DATA LOADING
# ============================================================================

def load_bee_data(filename):
    """Load puzzle data from JSON file"""
    with open(filename, 'r') as f:
        return json.load(f)


def get_all_puzzle_dates(start_date, end_date):
    """Generate list of dates between start and end (inclusive)"""
    start = datetime.strptime(start_date, "%Y%m%d")
    end = datetime.strptime(end_date, "%Y%m%d")
    
    dates = []
    current = start
    while current <= end:
        dates.append(current.strftime("%Y%m%d"))
        current += timedelta(days=1)
    
    return dates


def extract_puzzle_letters(words):
    """Extract unique letters from word list"""
    if not words:
        return set()
    return set(''.join(words).lower())


def identify_center_letter(words):
    """Identify the mandatory center letter (appears in all words)"""
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
# WORD VALIDATION
# ============================================================================

def is_valid_bee_word(word, center_letter, allowed_letters):
    """Check if word satisfies puzzle constraints"""
    word = word.strip().lower()
    return (word
            and word.isalpha()
            and len(word) >= MIN_WORD_LENGTH
            and center_letter in word
            and all(c in allowed_letters for c in word))


def parse_word_predictions(generated_text, center_letter, allowed_letters):
    """Parse model output and extract all valid words"""
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

# ============================================================================
# PROMPT GENERATION
# ============================================================================

def create_word_prediction_prompt(center_letter, outer_letters):
    """Create prompt for word generation task"""
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

# ============================================================================
# SHARED RESPONSE EXTRACTION UTILITIES
# ============================================================================

def extract_text_from_response(response):
    """Extract text from OpenAI response output"""
    if hasattr(response, 'output_text'):
        return response.output_text
    
    if hasattr(response, 'output'):
        texts = []
        for item in response.output:
            if hasattr(item, 'content'):
                for content_block in item.content:
                    if hasattr(content_block, 'text'):
                        texts.append(content_block.text)
        return '\n'.join(texts)
    
    return str(response)


def extract_reasoning_from_response(response):
    """Extract reasoning trace/summary from OpenAI response output"""
    if not hasattr(response, 'output'):
        return ""
    
    reasoning_parts = []
    for item in response.output:
        if hasattr(item, 'type') and item.type == 'reasoning':
            if hasattr(item, 'summary'):
                for summary_block in item.summary:
                    if hasattr(summary_block, 'text'):
                        reasoning_parts.append(summary_block.text)
    
    return '\n'.join(reasoning_parts)

# ============================================================================
# OPENAI API INTEGRATION
# ============================================================================

class OpenAIGenerator:
    """Handles OpenAI API calls with reasoning support"""
    
    def __init__(self, api_key=None):
        if api_key is None:
            api_key = os.getenv('OPENAI_API_KEY')
        
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY not found. Set it with:\n"
                "  export OPENAI_API_KEY='your_api_key_here'"
            )
        
        self.client = OpenAI(api_key=api_key)
        self.reasoning_model = OPENAI_REASONING_MODEL
        self.non_reasoning_model = OPENAI_NON_REASONING_MODEL
    
    def generate(self, prompt, enable_reasoning=False, reasoning_effort=None, max_tokens=None):
        """
        Generate response from OpenAI with optional reasoning.
        
        Args:
            prompt: Input prompt text
            enable_reasoning: Whether to enable reasoning mode
            reasoning_effort: Reasoning effort level ("low", "medium", "high")
            max_tokens: Maximum tokens for generation
        
        Returns:
            tuple: (answer_text, generation_metadata)
        """
        if reasoning_effort is None:
            reasoning_effort = DEFAULT_REASONING_EFFORT
        if max_tokens is None:
            max_tokens = DEFAULT_MAX_TOKENS
        
        model = self.reasoning_model if enable_reasoning else self.non_reasoning_model

        start_time = time.time()

        try:
            api_params = {
                'model': model,
                'max_output_tokens': max_tokens,
                'input': prompt,
                **(REASONING_PARAMS if enable_reasoning else NON_REASONING_PARAMS),
            }

            if enable_reasoning:
                api_params['reasoning'] = {
                    'effort': reasoning_effort,
                    'summary': 'auto'
                }
                api_params['include'] = ['reasoning.encrypted_content']

            response = self.client.responses.create(**api_params)
            
            generation_time = time.time() - start_time
            
            answer_text = extract_text_from_response(response)
            reasoning_trace = extract_reasoning_from_response(response) if enable_reasoning else ""
            
            metadata = {
                'raw_output': answer_text,
                'reasoning_trace': reasoning_trace if enable_reasoning else "",
                'final_answer': answer_text,
                'reasoning_enabled': enable_reasoning,
                'reasoning_effort': reasoning_effort if enable_reasoning else None,
                'generation_time': generation_time,
                'model': model,
                'max_tokens': max_tokens
            }
            
            if hasattr(response, 'usage'):
                metadata['tokens_generated'] = response.usage.output_tokens if hasattr(response.usage, 'output_tokens') else 0
                metadata['prompt_tokens'] = response.usage.input_tokens if hasattr(response.usage, 'input_tokens') else 0
                metadata['total_tokens'] = response.usage.total_tokens if hasattr(response.usage, 'total_tokens') else 0
                metadata['tokens_per_second'] = metadata['tokens_generated'] / generation_time if generation_time > 0 else 0
                
                if hasattr(response.usage, 'output_tokens_details'):
                    details = response.usage.output_tokens_details
                    if hasattr(details, 'reasoning_tokens'):
                        metadata['reasoning_tokens'] = details.reasoning_tokens
            
            return answer_text, metadata
            
        except Exception as e:
            print(f"\n[ERROR] Error calling OpenAI API: {e}")
            raise


class AsyncOpenAIGenerator:
    """Async version of OpenAIGenerator for batch processing"""
    
    def __init__(self, api_key=None, max_concurrent=MAX_CONCURRENT_REQUESTS):
        if api_key is None:
            api_key = os.getenv('OPENAI_API_KEY')
        
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY not found. Set it with:\n"
                "  export OPENAI_API_KEY='your_api_key_here'"
            )
        
        self.client = AsyncOpenAI(api_key=api_key)
        self.reasoning_model = OPENAI_REASONING_MODEL
        self.non_reasoning_model = OPENAI_NON_REASONING_MODEL
        self.semaphore = asyncio.Semaphore(max_concurrent)
    
    async def generate(self, prompt, enable_reasoning=False, reasoning_effort=None, max_tokens=None, max_retries=3):
        """
        Async generate response from OpenAI with optional reasoning.
        Includes exponential backoff retry logic for rate limits.
        
        Args:
            prompt: Input prompt text
            enable_reasoning: Whether to enable reasoning mode
            reasoning_effort: Reasoning effort level ("low", "medium", "high")
            max_tokens: Maximum tokens for generation
            max_retries: Maximum number of retries for rate limit errors
        
        Returns:
            tuple: (answer_text, generation_metadata)
        """
        if reasoning_effort is None:
            reasoning_effort = DEFAULT_REASONING_EFFORT
        if max_tokens is None:
            max_tokens = DEFAULT_MAX_TOKENS
        
        model = self.reasoning_model if enable_reasoning else self.non_reasoning_model

        async with self.semaphore:
            for attempt in range(max_retries):
                start_time = time.time()
                
                try:
                    api_params = {
                        'model': model,
                        'max_output_tokens': max_tokens,
                        'input': prompt,
                        **(REASONING_PARAMS if enable_reasoning else NON_REASONING_PARAMS),
                    }

                    if enable_reasoning:
                        api_params['reasoning'] = {
                            'effort': reasoning_effort,
                            'summary': 'auto'
                        }
                        api_params['include'] = ['reasoning.encrypted_content']
                    
                    response = await self.client.responses.create(**api_params)
                    
                    generation_time = time.time() - start_time
                    
                    answer_text = extract_text_from_response(response)
                    reasoning_trace = extract_reasoning_from_response(response) if enable_reasoning else ""
                    
                    metadata = {
                        'raw_output': answer_text,
                        'reasoning_trace': reasoning_trace if enable_reasoning else "",
                        'final_answer': answer_text,
                        'reasoning_enabled': enable_reasoning,
                        'reasoning_effort': reasoning_effort if enable_reasoning else None,
                        'generation_time': generation_time,
                        'model': model,
                        'max_tokens': max_tokens,
                        'retry_attempt': attempt
                    }
                    
                    if hasattr(response, 'usage'):
                        metadata['tokens_generated'] = response.usage.output_tokens if hasattr(response.usage, 'output_tokens') else 0
                        metadata['prompt_tokens'] = response.usage.input_tokens if hasattr(response.usage, 'input_tokens') else 0
                        metadata['total_tokens'] = response.usage.total_tokens if hasattr(response.usage, 'total_tokens') else 0
                        metadata['tokens_per_second'] = metadata['tokens_generated'] / generation_time if generation_time > 0 else 0
                        
                        if hasattr(response.usage, 'output_tokens_details'):
                            details = response.usage.output_tokens_details
                            if hasattr(details, 'reasoning_tokens'):
                                metadata['reasoning_tokens'] = details.reasoning_tokens
                    
                    return answer_text, metadata
                    
                except RateLimitError as e:
                    if attempt < max_retries - 1:
                        wait_time = (2 ** attempt) * 2
                        print(f"\n[WARN] Rate limit hit, waiting {wait_time}s before retry {attempt + 1}/{max_retries}...")
                        await asyncio.sleep(wait_time)
                    else:
                        print(f"\n[ERROR] Rate limit error after {max_retries} retries: {e}")
                        raise
                        
                except Exception as e:
                    print(f"\n[ERROR] Error calling OpenAI API: {e}")
                    raise

# ============================================================================
# FILE STORAGE
# ============================================================================

def save_structured_logs(logs_data, enable_reasoning, reasoning_effort, max_tokens, output_dir):
    """Save structured logs with complete generation details"""
    os.makedirs(output_dir, exist_ok=True)
    
    if enable_reasoning:
        mode = 'gpt5_reasoning'
        effort_str = f"_{reasoning_effort}"
    else:
        mode = 'gpt4.1'
        effort_str = f"_{max_tokens}"
    
    filename = f"{mode}{effort_str}_logs.json"
    filepath = os.path.join(output_dir, filename)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(logs_data, f, indent=2, ensure_ascii=False)
    
    print(f"[OK] Logs saved: {filepath}")
    return filepath


def save_results(results_data, enable_reasoning, reasoning_effort, max_tokens, output_dir):
    """Save results with predictions vs ground truth"""
    os.makedirs(output_dir, exist_ok=True)
    
    if enable_reasoning:
        mode = 'gpt5_reasoning'
        effort_str = f"_{reasoning_effort}"
    else:
        mode = 'gpt4.1'
        effort_str = f"_{max_tokens}"
    
    filename = f"{mode}{effort_str}_results.json"
    filepath = os.path.join(output_dir, filename)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(results_data, f, indent=2, ensure_ascii=False)
    
    print(f"[OK] Results saved: {filepath}")
    return filepath

# ============================================================================
# MAIN INFERENCE PIPELINE
# ============================================================================

def run_openai_inference(enable_reasoning=False, reasoning_effort=None, max_tokens=None):
    """
    Run inference on all puzzles with GPT (sequential processing).
    
    Args:
        enable_reasoning: Whether to enable reasoning mode
        reasoning_effort: Reasoning effort level
        max_tokens: Maximum tokens
    
    Returns:
        tuple: (logs_filepath, results_filepath)
    """
    if reasoning_effort is None:
        reasoning_effort = DEFAULT_REASONING_EFFORT
    if max_tokens is None:
        max_tokens = DEFAULT_MAX_TOKENS
    
    print("\n" + "="*80)
    print("NY TIMES SPELLING BEE - OPENAI GPT INFERENCE")
    print("="*80)
    print(f"Model: {OPENAI_REASONING_MODEL if enable_reasoning else OPENAI_NON_REASONING_MODEL}")
    print(f"Reasoning mode: {'ON' if enable_reasoning else 'OFF'}")
    if enable_reasoning:
        print(f"Reasoning effort: {reasoning_effort}")
    print(f"Max tokens: {max_tokens}")
    print(f"Date range: {START_DATE} to {END_DATE}")
    print("="*80)
    
    generator = OpenAIGenerator()
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
                prompt, enable_reasoning, reasoning_effort, max_tokens
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
        'model_name': OPENAI_REASONING_MODEL if enable_reasoning else OPENAI_NON_REASONING_MODEL,
        'model_family': 'gpt',
        'model_size': 'gpt-5' if enable_reasoning else 'gpt-4.1',
        'reasoning_enabled': enable_reasoning,
        'reasoning_effort': reasoning_effort if enable_reasoning else None,
        'max_tokens': max_tokens,
        'sampling_params': REASONING_PARAMS if enable_reasoning else NON_REASONING_PARAMS,
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
    logs_filepath = save_structured_logs(logs_data, enable_reasoning, reasoning_effort, max_tokens, LOG_PATH)
    
    results_data = {
        'metadata': metadata,
        'predictions': results_predictions
    }
    results_filepath = save_results(results_data, enable_reasoning, reasoning_effort, max_tokens, RESULTS_PATH)
    
    print("\n" + "="*80)
    print("INFERENCE COMPLETED")
    print(f"Puzzles processed: {len(results_predictions)}/{len(all_dates)}")
    print(f"Logs: {logs_filepath}")
    print(f"Results: {results_filepath}")
    print("="*80)
    
    return logs_filepath, results_filepath


async def run_openai_inference_batched(enable_reasoning=False, reasoning_effort=None, max_tokens=None, batch_size=BATCH_SIZE):
    """
    Run inference on all puzzles using batched async processing (much faster).
    
    Args:
        enable_reasoning: Whether to enable reasoning mode
        reasoning_effort: Reasoning effort level
        max_tokens: Maximum tokens
        batch_size: Number of puzzles to process concurrently
    
    Returns:
        tuple: (logs_filepath, results_filepath)
    """
    if reasoning_effort is None:
        reasoning_effort = DEFAULT_REASONING_EFFORT
    if max_tokens is None:
        max_tokens = DEFAULT_MAX_TOKENS
    
    print("\n" + "="*80)
    print("NY TIMES SPELLING BEE - OPENAI GPT INFERENCE (BATCHED)")
    print("="*80)
    print(f"Model: {OPENAI_REASONING_MODEL if enable_reasoning else OPENAI_NON_REASONING_MODEL}")
    print(f"Reasoning mode: {'ON' if enable_reasoning else 'OFF'}")
    if enable_reasoning:
        print(f"Reasoning effort: {reasoning_effort}")
    print(f"Max tokens: {max_tokens}")
    print(f"Batch size: {batch_size} concurrent requests")
    print(f"Date range: {START_DATE} to {END_DATE}")
    print("="*80)
    
    generator = AsyncOpenAIGenerator(max_concurrent=batch_size)
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
                prompt, enable_reasoning, reasoning_effort, max_tokens
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
        'model_name': OPENAI_REASONING_MODEL if enable_reasoning else OPENAI_NON_REASONING_MODEL,
        'model_family': 'gpt',
        'model_size': 'gpt-5' if enable_reasoning else 'gpt-4.1',
        'reasoning_enabled': enable_reasoning,
        'reasoning_effort': reasoning_effort if enable_reasoning else None,
        'max_tokens': max_tokens,
        'batch_size': batch_size,
        'sampling_params': REASONING_PARAMS if enable_reasoning else NON_REASONING_PARAMS,
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
    logs_filepath = save_structured_logs(logs_data, enable_reasoning, reasoning_effort, max_tokens, LOG_PATH)
    
    results_data = {
        'metadata': metadata,
        'predictions': results_predictions
    }
    results_filepath = save_results(results_data, enable_reasoning, reasoning_effort, max_tokens, RESULTS_PATH)
    
    print("\n" + "="*80)
    print("BATCHED INFERENCE COMPLETED")
    print(f"Puzzles processed: {len(results_predictions)}/{len(all_dates)}")
    print(f"Logs: {logs_filepath}")
    print(f"Results: {results_filepath}")
    print("="*80)
    
    return logs_filepath, results_filepath


def run_openai_inference_wrapper(enable_reasoning=False, reasoning_effort=None, max_tokens=None, use_batching=True, batch_size=BATCH_SIZE):
    """Wrapper to run either sequential or batched inference"""
    if use_batching:
        return asyncio.run(run_openai_inference_batched(
            enable_reasoning, reasoning_effort, max_tokens, batch_size
        ))
    else:
        return run_openai_inference(
            enable_reasoning, reasoning_effort, max_tokens
        )

# ============================================================================
# EXPERIMENT COORDINATOR
# ============================================================================

def run_all_openai_experiments(run_ablations=True, use_batching=True, batch_size=BATCH_SIZE):
    """
    Run all experiment configurations.
    
    Args:
        run_ablations: If True, run all ablation configurations
        use_batching: If True, use batched async processing
        batch_size: Batch size for concurrent processing
    
    Returns:
        dict: Summary of all runs
    """
    print("\n" + "="*80)
    print("RUNNING ALL OPENAI GPT EXPERIMENTS")
    if use_batching:
        print("(Using BATCHED async processing for 10x speedup!)")
    print("="*80)
    print(f"Model: GPT-5 (reasoning only)")
    if use_batching:
        print(f"Batch size: {batch_size} concurrent requests")
    
    configurations = []
    
    if run_ablations:
        for max_tokens in NON_REASONING_ABLATION_CONFIGS:
            configurations.append({
                'enable_reasoning': False,
                'reasoning_effort': None,
                'max_tokens': max_tokens,
                'name': f'GPT-4.1 (max={max_tokens})'
            })
        
        for reasoning_effort, max_tokens in ABLATION_CONFIGS:
            configurations.append({
                'enable_reasoning': True,
                'reasoning_effort': reasoning_effort,
                'max_tokens': max_tokens,
                'name': f'GPT-5 Reasoning (effort={reasoning_effort})'
            })
        
        total_configs = len(NON_REASONING_ABLATION_CONFIGS) + len(ABLATION_CONFIGS)
    else:
        configurations.append({
            'enable_reasoning': False,
            'reasoning_effort': None,
            'max_tokens': DEFAULT_MAX_TOKENS,
            'name': 'GPT-4.1'
        })
        configurations.append({
            'enable_reasoning': True,
            'reasoning_effort': DEFAULT_REASONING_EFFORT,
            'max_tokens': DEFAULT_MAX_TOKENS,
            'name': f'GPT-5 Reasoning (effort={DEFAULT_REASONING_EFFORT})'
        })
        total_configs = 2
    
    print(f"Configurations: {total_configs}")
    if run_ablations:
        if NON_REASONING_ABLATION_CONFIGS:
            print(f"Non-reasoning budgets: {NON_REASONING_ABLATION_CONFIGS}")
        if ABLATION_CONFIGS:
            print(f"Reasoning configs: {[(e, m) for e, m in ABLATION_CONFIGS]}")
    print(f"Puzzles per configuration: 58")
    print(f"Total API calls: {total_configs * 58}")
    print("="*80)
    
    all_results = []
    start_time = time.time()
    
    for config_idx, config in enumerate(configurations, 1):
        print(f"\n{'='*80}")
        print(f"CONFIGURATION {config_idx}/{total_configs}: {config['name']}")
        if config['enable_reasoning']:
            print(f"  Reasoning effort: {config['reasoning_effort']}")
            print(f"  Max tokens: {config['max_tokens']} tokens")
        print(f"{'='*80}")
        
        try:
            logs_path, results_path = run_openai_inference_wrapper(
                enable_reasoning=config['enable_reasoning'],
                reasoning_effort=config['reasoning_effort'],
                max_tokens=config['max_tokens'],
                use_batching=use_batching,
                batch_size=batch_size
            )
            
            all_results.append({
                'config_name': config['name'],
                'reasoning_enabled': config['enable_reasoning'],
                'reasoning_effort': config['reasoning_effort'],
                'max_tokens': config['max_tokens'],
                'status': 'success',
                'logs_path': logs_path,
                'results_path': results_path,
                'completed_at': datetime.now().isoformat()
            })
            
            print(f"\n[OK] {config['name']} COMPLETED")
            
        except Exception as e:
            print(f"\n[FAIL] {config['name']} FAILED: {e}")
            
            all_results.append({
                'config_name': config['name'],
                'reasoning_enabled': config['enable_reasoning'],
                'reasoning_effort': config['reasoning_effort'],
                'max_tokens': config['max_tokens'],
                'status': 'failed',
                'error': str(e),
                'failed_at': datetime.now().isoformat()
            })
            
            import traceback
            traceback.print_exc()
    
    total_time = time.time() - start_time
    hours = int(total_time // 3600)
    minutes = int((total_time % 3600) // 60)
    seconds = int(total_time % 60)
    
    completed = sum(1 for r in all_results if r['status'] == 'success')
    failed = sum(1 for r in all_results if r['status'] == 'failed')
    
    print("\n" + "="*80)
    print("ALL OPENAI EXPERIMENTS COMPLETED")
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
        effort_info = f" (effort={result['reasoning_effort']})" if result['reasoning_effort'] else ""
        print(f"{i}. {status_symbol} {result['config_name']}{effort_info} - {result['status']}")
        if result['status'] == 'success':
            print(f"     Logs: {result['logs_path']}")
            print(f"     Results: {result['results_path']}")
    print("-" * 80)
    
    summary_path = os.path.join(RESULTS_PATH, 'openai_experiment_summary.json')
    os.makedirs(RESULTS_PATH, exist_ok=True)
    
    summary = {
        'experiment': {
            'reasoning_model': OPENAI_REASONING_MODEL,
            'non_reasoning_model': OPENAI_NON_REASONING_MODEL,
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

# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*80)
    print("OPENAI GPT-5 SPELLING BEE INFERENCE SCRIPT")
    print("="*80)
    print("\nRuns GPT-5 on 58 puzzles for direct comparison with other models")
    print("\nBATCHED ASYNC PROCESSING ENABLED")
    print(f"   - {BATCH_SIZE} concurrent requests")
    print(f"   - {MAX_CONCURRENT_REQUESTS} max concurrent connections")
    print("\nGPT-5 Reasoning Effort Ablation:")
    for reasoning_effort, max_tokens in ABLATION_CONFIGS:
        print(f"   - Effort: {reasoning_effort} | Max tokens: {max_tokens}")
    print("\nEnsure OPENAI_API_KEY environment variable is set:")
    print("  export OPENAI_API_KEY='your_api_key_here'")
    print(f"\nRunning {len(ABLATION_CONFIGS)} GPT-5 reasoning configurations...")
    print("="*80)
    
    summary = run_all_openai_experiments(run_ablations=True, use_batching=True, batch_size=BATCH_SIZE)
    
    print("\n" + "="*80)
    print("ALL DONE! Results saved to:")
    print(f"  Logs: {LOG_PATH}")
    print(f"  Results: {RESULTS_PATH}")
    print("\nFiles created (GPT-5 reasoning):")
    for reasoning_effort, _ in ABLATION_CONFIGS:
        print(f"   - gpt5_reasoning_{reasoning_effort}_logs.json / results.json")
    print("="*80)
