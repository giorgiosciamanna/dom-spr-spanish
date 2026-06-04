#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
02_merge_counts.py
------------------
STEP 2 of 4 — Merge of per-file Counters and construction of the allowed vocabulary.

Loads the Counter pickles produced by 01_count_corpus.py and merges them in
batches, applying periodic pruning to keep RAM under control.

Pruning strategy:
  After each batch merge, the accumulated Counter is pruned to KEEP_TOP words.
  Words that do not rank in the top KEEP_TOP after each batch are very unlikely
  to reach the final top-100k, and are discarded early. KEEP_TOP must be much
  larger than TOP_K (e.g. 5-10x) to ensure no relevant word is lost prematurely.

Output:
  - total_counts.pkl   Global Counter (pruned to KEEP_TOP)
  - allowed_vocab.pkl  frozenset: top-TOP_K alphabetic words ∪ itemset words

Next step: 03_filter_corpus.py

Usage:
    python 02_merge_counts.py

All paths are read from paths.py.
"""

import os
import pickle as pkl
import time
import pandas as pd
from collections import Counter

from paths import (
    ITEMSET_PATH, COUNT_CACHE_DIR,
    TOTAL_COUNTS_PKL, ALLOWED_VOCAB_PKL,
)

# =============================================================================
# PARAMETERS
# =============================================================================

TOP_K     = 100_000   # most frequent words to include in the vocabulary
KEEP_TOP  = 500_000   # words to keep in the Counter after each pruning step
                      # must be >> TOP_K; lower to 200_000 if RAM is limited
BATCH_SIZE = 4        # pkl files to merge per batch before pruning

# =============================================================================


def extract_itemset_words(itemset_path: str) -> frozenset:
    """Extract experimental words from the itemset (lowercased)."""
    with open(itemset_path, "r", encoding="utf-8") as f:
        df = pd.read_csv(f)

    words = set()
    for sent in df[df["type"] % 6 < 3]["sentence"]:
        tokens = sent.split(".")[0].split()
        if len(tokens) > 4:
            words.update([tokens[0].lower(), tokens[1].lower(), tokens[4].lower()])

    print(f"Itemset words: {len(words)}")
    return frozenset(words)


def merge_counters(count_cache_dir: str, keep_top: int,
                   batch_size: int) -> Counter:
    """
    Merge Counter pickles in batches, pruning after each merge.

    After every batch:
        total = top-KEEP_TOP words of the accumulated Counter

    This keeps the Counter's size in RAM always <= KEEP_TOP,
    regardless of how many files are processed.
    """
    pkl_files = sorted(
        fn for fn in os.listdir(count_cache_dir) if fn.endswith("_count.pkl")
    )

    if not pkl_files:
        raise FileNotFoundError(
            f"No _count.pkl files found in {count_cache_dir}. "
            "Run 01_count_corpus.py first."
        )

    n_batches = (len(pkl_files) + batch_size - 1) // batch_size
    print(f"\nPkl files found : {len(pkl_files)}")
    print(f"Batches         : {n_batches} x {batch_size}")
    print(f"Pruning to      : top-{keep_top:,} after each batch\n")

    t0    = time.time()
    total = Counter()

    for i in range(0, len(pkl_files), batch_size):
        batch = pkl_files[i : i + batch_size]
        b_num = i // batch_size + 1
        print(f"  batch {b_num}/{n_batches} ({len(batch)} files)...", end=" ", flush=True)
        t_b = time.time()

        for fn in batch:
            file_path = os.path.join(count_cache_dir, fn)
            try:
                with open(file_path, "rb") as f:
                    total += pkl.load(f)
            except EOFError:
                print(f"\n  WARNING: corrupted pickle skipped: {file_path}")
            except Exception as e:
                print(f"\n  ERROR loading {file_path}: {e}")
                raise

        # pruning — keep only the top-KEEP_TOP words
        if len(total) > keep_top:
            total = Counter(dict(total.most_common(keep_top)))

        print(f"{len(total):,} tokens in Counter  [{time.time()-t_b:.1f}s]")

    print(f"\nMerge completed in {time.time()-t0:.1f}s")
    print(f"Distinct tokens in final Counter: {len(total):,}")
    print(f"Total token count (sum):          {sum(total.values()):,}")
    return total


def build_allowed_vocab(total: Counter, itemset_words: frozenset,
                        top_k: int) -> frozenset:
    """
    Build the allowed vocabulary for corpus filtering:
      - top-TOP_K alphabetic words by corpus frequency
      - + all itemset words (even if rare)

    Returns a frozenset for O(1) lookup in 03_filter_corpus.py.
    """
    print(f"\nBuilding vocabulary (top-{top_k:,} + itemset)...")

    top_words = {
        w for w, _ in total.most_common(top_k) if w.isalpha()
    }
    print(f"  Alphabetic words in top-{top_k:,}: {len(top_words):,}")

    extra = itemset_words - top_words
    allowed = frozenset(top_words | itemset_words)
    print(f"  Extra itemset words added:        {len(extra)}")
    print(f"  Final vocabulary:                 {len(allowed):,} words")
    return allowed


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":

    os.makedirs(os.path.dirname(TOTAL_COUNTS_PKL), exist_ok=True)

    itemset_words = extract_itemset_words(ITEMSET_PATH)

    total = merge_counters(COUNT_CACHE_DIR, KEEP_TOP, BATCH_SIZE)

    with open(TOTAL_COUNTS_PKL, "wb") as f:
        pkl.dump(total, f, protocol=pkl.HIGHEST_PROTOCOL)
    print(f"\nGlobal Counter saved to: {TOTAL_COUNTS_PKL}")

    allowed = build_allowed_vocab(total, itemset_words, TOP_K)

    with open(ALLOWED_VOCAB_PKL, "wb") as f:
        pkl.dump(allowed, f, protocol=pkl.HIGHEST_PROTOCOL)
    print(f"Allowed vocabulary saved to: {ALLOWED_VOCAB_PKL}")

    print("\nNext step: python 03_filter_corpus.py")
