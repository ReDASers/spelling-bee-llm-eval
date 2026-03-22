# Orthographic Constraint Satisfaction and Human Difficulty Alignment in Large Language Models

**Bryan E. Tuck and Rakesh M. Verma**
University of Houston

Accepted at LREC 2026. [[arXiv]](https://arxiv.org/abs/2511.21086)

## Overview

This repository provides data and code to reproduce the experiments in our paper evaluating LLM orthographic constraint satisfaction using the NYT Spelling Bee task. We test 28 configurations spanning three model families (Qwen3 4B--32B, Claude Haiku-4.5, GPT-5-mini) on 58 word puzzles, with human difficulty ground truth from 10,000+ NYT users per puzzle.

## Repository Structure

```text
├── src/
│   ├── colab_spelling_bee.py      # Qwen3 inference (Google Colab / vLLM)
│   ├── run_claude_inference.py    # Claude API inference
│   └── run_openai_inference.py    # OpenAI API inference
├── data/
│   ├── puzzles/                   # 58 puzzles with human difficulty data
│   ├── qwen-results/              # Qwen3 results by budget (4K/8K/16K)
│   ├── claude-results/            # Claude Haiku results (6 configs)
│   └── openai-results/            # GPT-5-mini results (3 configs)
├── metrics/                       # Metrics computation package
│   ├── *.py                       #   Python modules
│   └── output/                    #   Computed metrics (CSV, JSON, TEX)
├── compute_metrics.py             # Compute all metrics from results
├── generate_figures.py            # Generate publication figures
├── analyze_tokenization.py        # Tokenization robustness analysis
├── validate_false_positives.py    # False positive validation
├── requirements.txt               # Analysis dependencies
├── requirements-inference.txt     # Inference dependencies (GPU, API clients)
├── LICENSE-CODE                   # MIT License (code + model results)
└── LICENSE-DATA                   # Fair use notice (puzzle data)
```

## Setup

```bash
# Clone and install analysis dependencies
pip install -r requirements.txt

# For running inference (GPU or API keys required)
pip install -r requirements-inference.txt
```

## Reproduction

### Inference

**Qwen3 models on Google Colab (recommended):**
Copy the contents of `src/colab_spelling_bee.py` into a Colab notebook cell with a GPU runtime (A100 recommended). Mount Google Drive and update the paths at the top of the cell. The script runs all five Qwen3 model sizes (4B, 8B, 14B, 30B, 32B) in both thinking and non-thinking modes.

**Qwen3 models locally (requires NVIDIA GPU with vLLM):**

```bash
pip install -r requirements-inference.txt
python src/colab_spelling_bee.py  # Update paths at the top for your environment
```

**Claude Haiku (requires API key):**

```bash
export ANTHROPIC_API_KEY='your_key'
python src/run_claude_inference.py
```

**GPT-5-mini (requires API key):**

```bash
export OPENAI_API_KEY='your_key'
python src/run_openai_inference.py
```

### Analysis

```bash
# Compute all metrics from result files
python compute_metrics.py --results-dir data/qwen-results --output-dir metrics/output

# Generate publication figures
python generate_figures.py --metrics-dir metrics/output --output-dir figures

# Tokenization robustness check (requires transformers from requirements-inference.txt)
python analyze_tokenization.py

# Validate false positives against dictionaries (requires nltk)
python validate_false_positives.py
```

## Data

The puzzle data is also available as a HuggingFace dataset:

```python
from datasets import load_dataset
ds = load_dataset("redasers/spelling-bee-human-difficulty")
```

### Puzzles

`data/puzzles/` contains 58 NYT Spelling Bee puzzles from June 2 to July 29, 2025. Each JSON file maps valid answers to their human solve count from a sample of ~10,000 users:

```json
{
    "id": 15134,
    "answers": {
        "cello": 7858,
        "compile": 7892,
        "lollop": 4874
    }
}
```

Higher counts indicate easier words (more solvers). Lower counts indicate harder words.

### Model Results

Each result file is a JSON object with the following schema:

```json
{
    "metadata": {
        "model_name": "qwen3_8b",
        "model_family": "qwen3",
        "model_size": "8B",
        "thinking_enabled": true,
        "thinking_budget": 4096
    },
    "predictions": [
        {
            "puzzle_id": 15134,
            "date": "20250602",
            "center_letter": "o",
            "all_letters": ["c", "e", "i", "l", "m", "o", "p"],
            "predicted_words": ["cello", "compile", "police"],
            "actual_words": ["cello", "compile", "lollop"],
            "correctly_predicted": ["cello", "compile"],
            "missed_words": ["lollop"],
            "false_positives": ["police"]
        }
    ]
}
```

Metadata fields vary by model family:
- **Qwen3**: `thinking_budget` (4096, 8192, or 16384 tokens)
- **Claude Haiku**: `thinking_budget` (same range)
- **GPT-5-mini**: `reasoning_effort` ("low", "medium", or "high") instead of `thinking_budget`

Result directories:
- `data/qwen-results/{4,8,16}/` -- Qwen3 predictions at 4K/8K/16K thinking token budgets. Each budget directory contains 10 files (5 model sizes x 2 thinking modes).
- `data/claude-results/` -- Claude Haiku-4.5 predictions (3 thinking + 3 non-thinking configs).
- `data/openai-results/` -- GPT-5-mini predictions (3 reasoning effort levels: low/medium/high).

## License

- **Code** (`src/`, `metrics/`, `compute_metrics.py`, `generate_figures.py`): MIT License. See [LICENSE-CODE](LICENSE-CODE).
- **Model results** (`data/qwen-results/`, `data/claude-results/`, `data/openai-results/`): MIT License.
- **Puzzle data** (`data/puzzles/`): Originates from The New York Times Spelling Bee. Redistributed for non-commercial academic research under fair use. See [LICENSE-DATA](LICENSE-DATA).

## Citation

```bibtex
@misc{tuck2025orthographicconstraintsatisfactionhuman,
      title={Orthographic Constraint Satisfaction and Human Difficulty Alignment in Large Language Models},
      author={Bryan E. Tuck and Rakesh M. Verma},
      year={2025},
      eprint={2511.21086},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2511.21086},
}
```
