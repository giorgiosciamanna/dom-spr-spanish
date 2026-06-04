#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
02b_itemset_logprob.py
----------------------
STEP 2b — Frequencies and log-probabilities of itemset words on OSCAR.

In a single pass over the Counter pickles:
  - sums all counts to obtain N (total number of tokens in the corpus)
  - for each itemset word, accumulates the absolute frequency

Output: CSV with columns
  word | count | total_words | probability | neg_log_prob

Should be run after 02_merge_counts.py and before 03_filter_corpus.py.

Usage:
    python 02b_itemset_logprob.py

All paths are read from paths.py.
"""

import os
import pickle as pkl
import math
import pandas as pd
from collections import Counter

from paths import ITEMSET_PATH, COUNT_CACHE_DIR, OUTPUT_CSV

# =============================================================================


def extract_itemset_words(itemset_path: str) -> frozenset:
    with open(itemset_path, "r", encoding="utf-8") as f:
        df = pd.read_csv(f)
    words = set()
    for sent in df[df["type"] % 6 < 3]["sentence"]:
        tokens = sent.split(".")[0].split()
        if len(tokens) > 4:
            words.update([tokens[0].lower(), tokens[1].lower(), tokens[4].lower()])
    print(f"Itemset words: {len(words)}")
    return frozenset(words)


def compute_stats(count_cache_dir: str, itemset_words: frozenset) -> tuple:
    """
    Single pass over all Counter pickles.
    For each pickle:
      - sum all values → contribution to N_total
      - for each itemset word, add the local count

    Returns (n_total, itemset_counts) where:
      n_total        = int, total number of tokens across OSCAR
      itemset_counts = dict {word: global_count}
    """
    pkl_files = sorted(
        fn for fn in os.listdir(count_cache_dir) if fn.endswith("_count.pkl")
    )
    if not pkl_files:
        raise FileNotFoundError(
            f"No _count.pkl files found in {count_cache_dir}. "
            "Run 01_count_corpus.py first."
        )

    n_total        = 0
    itemset_counts = {w: 0 for w in itemset_words}

    for i, fn in enumerate(pkl_files, 1):
        print(f"  [{i}/{len(pkl_files)}] {fn}...", end=" ", flush=True)
        try:
            with open(os.path.join(count_cache_dir, fn), "rb") as f:
                counter = pkl.load(f)

            n_total += sum(counter.values())

            for word in itemset_words:
                if word in counter:
                    itemset_counts[word] += counter[word]
            print(f"partial N: {n_total:,}")
        except EOFError:
            print(f"WARNING: corrupted pickle skipped: {fn}")
        except Exception as e:
            print(f"ERROR loading {fn}: {e}")
            raise

    return n_total, itemset_counts


def build_csv(itemset_counts: dict, n_total: int, output_csv: str) -> pd.DataFrame:
    """Build and save the CSV with per-word statistics for itemset words."""
    rows = []
    for word, count in itemset_counts.items():
        prob         = count / n_total if n_total > 0 else 0
        neg_log_prob = -math.log(prob) if prob > 0 else float("inf")
        rows.append({
            "word":         word,
            "count":        count,
            "total_words":  n_total,
            "probability":  prob,
            "neg_log_prob": neg_log_prob,
        })

    df = pd.DataFrame(rows).sort_values("count", ascending=False)
    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
    df.to_csv(output_csv, index=False, encoding="utf-8")
    return df


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":

    itemset_words = extract_itemset_words(ITEMSET_PATH)

    print(f"\nPassing over Counter pickles in {COUNT_CACHE_DIR}...")
    n_total, itemset_counts = compute_stats(COUNT_CACHE_DIR, itemset_words)

    print(f"\nTotal tokens in OSCAR: {n_total:,}")

    df = build_csv(itemset_counts, n_total, OUTPUT_CSV)
    print(f"CSV saved to: {OUTPUT_CSV}\n")
    print(df.to_string(index=False))
