"""Test whether the Qwen tokenizer confounds the length-difficulty relationship (Section 5.4)."""

import ast
import json
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy import stats
from transformers import AutoTokenizer


def main():
    print("Loading Qwen tokenizer...")
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B", trust_remote_code=True)

    print("Loading detailed metrics...")
    df = pd.read_csv("metrics/output/detailed_metrics.csv")

    all_actual = set()
    all_found = set()
    all_missed = set()

    for _, row in df.iterrows():
        actual = ast.literal_eval(row["actual_words"])
        found = ast.literal_eval(row["correctly_predicted"])
        missed = ast.literal_eval(row["missed_words"])
        all_actual.update(actual)
        all_found.update(found)
        all_missed.update(missed)

    all_words = sorted(all_actual)
    print(f"Unique target words across all puzzles: {len(all_words)}")

    word_tokens = {}
    for w in all_words:
        tokens = tok.encode(w, add_special_tokens=False)
        word_tokens[w] = len(tokens)

    char_lengths = [len(w) for w in all_words]
    token_counts = [word_tokens[w] for w in all_words]

    token_dist = defaultdict(int)
    for w in all_words:
        token_dist[word_tokens[w]] += 1

    print("\n=== Token Count Distribution ===")
    for tc in sorted(token_dist.keys()):
        pct = token_dist[tc] / len(all_words) * 100
        print(f"  {tc} token(s): {token_dist[tc]} words ({pct:.1f}%)")

    r, p = stats.pearsonr(char_lengths, token_counts)
    print(f"\n=== Char Length vs Token Count ===")
    print(f"  Pearson r = {r:.3f}, p = {p:.2e}")

    print("\n=== Mean Token Count by Character Length ===")
    length_buckets = defaultdict(list)
    for w in all_words:
        length_buckets[len(w)].append(word_tokens[w])

    for length in sorted(length_buckets.keys()):
        tokens_list = length_buckets[length]
        mean_tok = np.mean(tokens_list)
        single_pct = sum(1 for t in tokens_list if t == 1) / len(tokens_list) * 100
        print(f"  {length}-letter: n={len(tokens_list)}, mean tokens={mean_tok:.2f}, "
              f"single-token={single_pct:.1f}%")

    print("\n=== Model Recall by Token Count vs Character Length ===")

    word_outcomes = defaultdict(lambda: {"found": 0, "missed": 0})

    for _, row in df.iterrows():
        found = set(ast.literal_eval(row["correctly_predicted"]))
        missed = set(ast.literal_eval(row["missed_words"]))
        for w in found:
            word_outcomes[w]["found"] += 1
        for w in missed:
            word_outcomes[w]["missed"] += 1

    recall_by_char = defaultdict(lambda: {"found": 0, "total": 0})
    recall_by_token = defaultdict(lambda: {"found": 0, "total": 0})

    for w, outcomes in word_outcomes.items():
        total = outcomes["found"] + outcomes["missed"]
        char_len = len(w)
        tok_count = word_tokens.get(w, 1)

        recall_by_char[char_len]["found"] += outcomes["found"]
        recall_by_char[char_len]["total"] += total

        recall_by_token[tok_count]["found"] += outcomes["found"]
        recall_by_token[tok_count]["total"] += total

    print("\n  By Character Length:")
    char_recalls, char_lengths_list = [], []
    for length in sorted(recall_by_char.keys()):
        d = recall_by_char[length]
        recall = d["found"] / d["total"] if d["total"] > 0 else 0
        print(f"    {length}-letter: recall={recall:.3f} (n={d['total']})")
        char_recalls.append(recall)
        char_lengths_list.append(length)

    print("\n  By Token Count:")
    tok_recalls, tok_counts_list = [], []
    for tc in sorted(recall_by_token.keys()):
        d = recall_by_token[tc]
        recall = d["found"] / d["total"] if d["total"] > 0 else 0
        print(f"    {tc} token(s): recall={recall:.3f} (n={d['total']})")
        tok_recalls.append(recall)
        tok_counts_list.append(tc)

    # Point-biserial: which length metric better predicts success?
    print("\n=== Point-Biserial Correlation: Success ~ Length Metric ===")

    successes, char_lens, tok_cnts = [], [], []
    for w, outcomes in word_outcomes.items():
        char_len = len(w)
        tok_count = word_tokens.get(w, 1)
        successes.extend([1] * outcomes["found"] + [0] * outcomes["missed"])
        char_lens.extend([char_len] * (outcomes["found"] + outcomes["missed"]))
        tok_cnts.extend([tok_count] * (outcomes["found"] + outcomes["missed"]))

    r_char, p_char = stats.pointbiserialr(successes, char_lens)
    r_tok, p_tok = stats.pointbiserialr(successes, tok_cnts)

    print(f"  Success ~ char_length:  r = {r_char:.4f}, p = {p_char:.2e}")
    print(f"  Success ~ token_count:  r = {r_tok:.4f}, p = {p_tok:.2e}")
    print(f"  Stronger predictor: {'character length' if abs(r_char) > abs(r_tok) else 'token count'}")

    # Qwen-only: isolate tokenizer effect (same tokenizer across all scales)
    print("\n=== Qwen-Only Analysis (Same Tokenizer Across Scales) ===")
    qwen_df = df[df["model_size"].isin(["4b", "8b", "14b", "30b", "32b"])]

    qwen_outcomes = defaultdict(lambda: {"found": 0, "missed": 0})
    for _, row in qwen_df.iterrows():
        found = set(ast.literal_eval(row["correctly_predicted"]))
        missed = set(ast.literal_eval(row["missed_words"]))
        for w in found:
            qwen_outcomes[w]["found"] += 1
        for w in missed:
            qwen_outcomes[w]["missed"] += 1

    q_successes, q_char, q_tok = [], [], []
    for w, outcomes in qwen_outcomes.items():
        char_len = len(w)
        tok_count = word_tokens.get(w, 1)
        q_successes.extend([1] * outcomes["found"] + [0] * outcomes["missed"])
        q_char.extend([char_len] * (outcomes["found"] + outcomes["missed"]))
        q_tok.extend([tok_count] * (outcomes["found"] + outcomes["missed"]))

    r_char_q, p_char_q = stats.pointbiserialr(q_successes, q_char)
    r_tok_q, p_tok_q = stats.pointbiserialr(q_successes, q_tok)

    print(f"  Success ~ char_length:  r = {r_char_q:.4f}, p = {p_char_q:.2e}")
    print(f"  Success ~ token_count:  r = {r_tok_q:.4f}, p = {p_tok_q:.2e}")

    results = {
        "unique_words": len(all_words),
        "single_token_pct": sum(1 for w in all_words if word_tokens[w] == 1) / len(all_words) * 100,
        "token_distribution": {str(k): v for k, v in sorted(token_dist.items())},
        "char_token_correlation": {"r": round(r, 3), "p": float(f"{p:.2e}")},
        "point_biserial_all": {
            "char_length": {"r": round(r_char, 4), "p": float(f"{p_char:.2e}")},
            "token_count": {"r": round(r_tok, 4), "p": float(f"{p_tok:.2e}")},
            "stronger": "character length" if abs(r_char) > abs(r_tok) else "token count",
        },
        "point_biserial_qwen_only": {
            "char_length": {"r": round(r_char_q, 4), "p": float(f"{p_char_q:.2e}")},
            "token_count": {"r": round(r_tok_q, 4), "p": float(f"{p_tok_q:.2e}")},
        },
        "recall_by_char_length": {
            str(k): round(recall_by_char[k]["found"] / recall_by_char[k]["total"], 3)
            for k in sorted(recall_by_char.keys())
        },
        "recall_by_token_count": {
            str(k): round(recall_by_token[k]["found"] / recall_by_token[k]["total"], 3)
            for k in sorted(recall_by_token.keys())
        },
    }

    out_path = Path("metrics/output/tokenization_analysis.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
