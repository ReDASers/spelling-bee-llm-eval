"""Validate false positives against NLTK + WordNet to separate real English words from hallucinations."""

import json
import csv
from pathlib import Path
from collections import defaultdict, Counter

import nltk
nltk.download('words', quiet=True)
nltk.download('wordnet', quiet=True)
from nltk.corpus import words, wordnet


def build_dictionary():
    """Combine NLTK words corpus and WordNet lemmas into a single set."""
    nltk_words = set(w.lower() for w in words.words())
    wn_words = set(
        l.name().lower()
        for s in wordnet.all_synsets()
        for l in s.lemmas()
        if '_' not in l.name()
    )
    combined = nltk_words | wn_words
    return combined


def collect_false_positives(data_dir: Path):
    fps_by_family = defaultdict(Counter)  # family -> {word: count}
    fps_all = Counter()

    for family_dir, family_name in [
        (data_dir / 'qwen-results', 'Qwen'),
        (data_dir / 'claude-results', 'Claude-Haiku'),
        (data_dir / 'openai-results', 'GPT-5-mini'),
    ]:
        if not family_dir.exists():
            continue

        for json_file in sorted(family_dir.rglob('*.json')):
            with open(json_file, encoding='utf-8') as f:
                data = json.load(f)

            preds = data.get('predictions', [])
            if isinstance(preds, dict):
                preds = list(preds.values())

            for pred in preds:
                fp_list = pred.get('false_positives', [])
                for word in fp_list:
                    w = word.lower().strip()
                    if w:
                        fps_by_family[family_name][w] += 1
                        fps_all[w] += 1

    return fps_by_family, fps_all


def main():
    data_dir = Path(__file__).parent / 'data'
    output_dir = Path(__file__).parent / 'metrics' / 'output'

    print("Building dictionary from NLTK words + WordNet...")
    dictionary = build_dictionary()
    print(f"  Dictionary size: {len(dictionary):,} words")

    print("\nCollecting false positives from results...")
    fps_by_family, fps_all = collect_false_positives(data_dir)

    unique_fps = set(fps_all.keys())
    print(f"  Total unique FP words: {len(unique_fps)}")
    print(f"  Total FP occurrences: {sum(fps_all.values())}")

    if not unique_fps:
        print("No false positives found.")
        return

    valid_english = {w for w in unique_fps if w in dictionary}
    hallucinations = unique_fps - valid_english

    print(f"\n=== Dictionary Validation Results ===")
    print(f"Valid English words (in NLTK/WordNet): {len(valid_english)} ({100*len(valid_english)/len(unique_fps):.1f}%)")
    print(f"Genuine hallucinations (not in any dictionary): {len(hallucinations)} ({100*len(hallucinations)/len(unique_fps):.1f}%)")

    valid_occurrences = sum(fps_all[w] for w in valid_english)
    halluc_occurrences = sum(fps_all[w] for w in hallucinations)
    total_occ = valid_occurrences + halluc_occurrences
    print(f"\nBy occurrence:")
    print(f"  Valid English: {valid_occurrences} ({100*valid_occurrences/total_occ:.1f}%)")
    print(f"  Hallucinations: {halluc_occurrences} ({100*halluc_occurrences/total_occ:.1f}%)")

    print(f"\n=== Per-Family Breakdown ===")
    family_stats = {}
    for family in ['Qwen', 'Claude-Haiku', 'GPT-5-mini']:
        if family not in fps_by_family:
            continue
        fam_fps = fps_by_family[family]
        fam_unique = set(fam_fps.keys())
        fam_valid = {w for w in fam_unique if w in dictionary}
        fam_halluc = fam_unique - fam_valid

        fam_valid_occ = sum(fam_fps[w] for w in fam_valid)
        fam_halluc_occ = sum(fam_fps[w] for w in fam_halluc)
        fam_total_occ = fam_valid_occ + fam_halluc_occ

        family_stats[family] = {
            'unique_fps': len(fam_unique),
            'valid_unique': len(fam_valid),
            'halluc_unique': len(fam_halluc),
            'valid_pct': 100 * len(fam_valid) / len(fam_unique) if fam_unique else 0,
            'total_occurrences': fam_total_occ,
            'valid_occurrences': fam_valid_occ,
            'halluc_occurrences': fam_halluc_occ,
            'valid_occ_pct': 100 * fam_valid_occ / fam_total_occ if fam_total_occ else 0,
        }

        print(f"\n{family}:")
        print(f"  Unique FPs: {len(fam_unique)}")
        print(f"  Valid English: {len(fam_valid)} ({family_stats[family]['valid_pct']:.1f}%)")
        print(f"  Hallucinations: {len(fam_halluc)} ({100-family_stats[family]['valid_pct']:.1f}%)")
        print(f"  By occurrence: {fam_valid_occ}/{fam_total_occ} valid ({family_stats[family]['valid_occ_pct']:.1f}%)")

    print(f"\n=== Top 20 Valid English Words Excluded by NYT ===")
    valid_sorted = sorted(valid_english, key=lambda w: fps_all[w], reverse=True)
    for w in valid_sorted[:20]:
        print(f"  {w:15s} (generated {fps_all[w]} times)")

    print(f"\n=== Top 20 Genuine Hallucinations ===")
    halluc_sorted = sorted(hallucinations, key=lambda w: fps_all[w], reverse=True)
    for w in halluc_sorted[:20]:
        print(f"  {w:15s} (generated {fps_all[w]} times)")

    results = {
        'dictionary_size': len(dictionary),
        'total_unique_fps': len(unique_fps),
        'total_fp_occurrences': sum(fps_all.values()),
        'valid_english_unique': len(valid_english),
        'hallucination_unique': len(hallucinations),
        'valid_english_pct': round(100 * len(valid_english) / len(unique_fps), 1),
        'valid_occurrences': valid_occurrences,
        'halluc_occurrences': halluc_occurrences,
        'valid_occ_pct': round(100 * valid_occurrences / total_occ, 1),
        'per_family': family_stats,
        'top_valid_excluded': [{'word': w, 'count': fps_all[w]} for w in valid_sorted[:50]],
        'top_hallucinations': [{'word': w, 'count': fps_all[w]} for w in halluc_sorted[:50]],
    }

    output_path = output_dir / 'fp_dictionary_validation.json'
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == '__main__':
    main()
