# Orthographic Constraint Satisfaction and Human Difficulty Alignment in Large Language Models

**Bryan E. Tuck and Rakesh M. Verma**
University of Houston

Accepted at LREC 2026.

## Overview

This repository provides data and code to reproduce the experiments in our paper evaluating LLM orthographic constraint satisfaction using the NYT Spelling Bee task. We test 28 configurations spanning three model families (Qwen3, Claude Haiku-4.5, GPT-5-mini) on 58 word puzzles, with human difficulty ground truth from 10,000+ NYT users per puzzle.

## Repository Structure

```
├── src/                        # All scripts
│   ├── run_claude_inference.py # Claude API inference
│   ├── run_openai_inference.py # OpenAI API inference
│   └── colab_spelling_bee.py   # Google Colab inference script
├── data/
│   ├── puzzles/                # 58 NYT Spelling Bee puzzles with human difficulty
│   ├── qwen-results/           # Qwen3 results (4K/8K/16K token budgets)
│   ├── claude-results/         # Claude Haiku results
│   └── openai-results/         # GPT-5 results
├── requirements.txt            # Analysis dependencies
└── requirements-inference.txt  # Inference dependencies (GPU, API clients)
```

## Setup

```bash
# Clone and install analysis dependencies
pip install -r requirements.txt

# For running inference 
pip install -r requirements-inference.txt
```

## Reproduction

All commands run from the repository root.

**Qwen models (requires GPU with vLLM):**
```bash
python src/run_inference.py
```

**Claude Haiku (requires API key):**
```bash
export ANTHROPIC_API_KEY='your_key'
python src/run_claude_inference.py
```

**GPT-5 (requires API key):**
```bash
export OPENAI_API_KEY='your_key'
python src/run_openai_inference.py
```

**Qwen on Google Colab:**
Upload `src/colab_spelling_bee.py` to Colab and update the Google Drive paths at the top of the file.

## Data

The puzzle data is also available as a HuggingFace dataset:

```python
from datasets import load_dataset
ds = load_dataset("redasers/spelling-bee-human-difficulty")
```

- **Puzzles** (`data/puzzles/`): 58 NYT Spelling Bee puzzles (June 2 -- July 29, 2025). Each JSON contains valid answers and per-word human solve counts from a sample of 10,000 users.
- **Results** (`data/qwen-results/`, `data/claude-results/`, `data/openai-results/`): Model predictions for all configurations. Qwen results are organized by thinking budget (4K/8K/16K tokens).

## Key Findings

1. Architectural differences produce 2.0--2.2x performance gaps (F1=0.761 vs 0.343), larger than eightfold parameter scaling (83% gain)
2. Thinking mode consistently improves performance (+0.179 F1)
3. Budget sensitivity is heterogeneous: high-capacity models gain +0.10--0.14 F1, mid-sized variants saturate or degrade
4. Modest but consistent human calibration (r=0.24--0.38) with systematic failures on orthographically atypical common words

## License

- **Code** (`src/`): MIT License. See [LICENSE-CODE](LICENSE-CODE).
- **Puzzle data** (`data/puzzles/`): Originates from The New York Times Spelling Bee. Redistributed for non-commercial academic research under fair use. See [LICENSE-DATA](LICENSE-DATA).
- **Model results** (`data/qwen-results/`, `data/claude-results/`, `data/openai-results/`): MIT License.

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
