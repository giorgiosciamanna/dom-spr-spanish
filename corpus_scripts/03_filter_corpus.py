#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
03_filter_corpus.py
-------------------
STEP 3 of 4 — Corpus filtering based on the allowed vocabulary.

Loads the vocabulary produced by 02_merge_counts.py and filters the corpus:
for each .gz file, only sentences in which ALL words belong to the allowed
vocabulary are kept.

Resumable: already-filtered files are skipped.

Output:
  - one <filename>.gz_filt.pkl per .gz file  (in PARTIAL_DIR)
  - oscar_filt_all.pkl                        (final merge in FINAL_PKL)

Next step: 04_encode_corpus.py

Usage:
    python 03_filter_corpus.py

All paths are read from paths.py.
"""

import os
import gzip
import pickle as pkl
import time
import multiprocessing as mp

from paths import CORPUS_DIR, PARTIAL_DIR, ALLOWED_VOCAB_PKL, FINAL_PKL

# =============================================================================
# PARAMETERS
# =============================================================================

N_WORKERS  = None   # None = all available cores
BATCH_SIZE = 4      # files per batch

# =============================================================================


def filter_file(args: tuple) -> tuple:
    """
    Filter a .gz file: keep only sentences in which ALL words belong to the
    allowed vocabulary. Saves the result as a .pkl.

    If the .pkl already exists, skips the file (resume mode).
    Returns (filename, n_total, n_filtered).

    Notes:
      - sentences are lowercased before checking and saved in lowercase,
        consistent with the rest of the pipeline preprocessing
      - frozenset lookup is O(1)
    """
    gz_path, out_path, allowed_vocab = args

    if os.path.exists(out_path):
        with open(out_path, "rb") as f:
            existing = pkl.load(f)
        return (os.path.basename(gz_path), -1, len(existing))  # -1 = skipped

    sentences = []
    total     = 0

    with gzip.open(gz_path, "rt", encoding="utf-8", errors="ignore") as f:
        for line in f:
            total += 1
            lowered = line.lower()
            if all(token in allowed_vocab for token in lowered.split()):
                sentences.append(lowered)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "wb") as f:
        pkl.dump(sentences, f, protocol=pkl.HIGHEST_PROTOCOL)

    return (os.path.basename(gz_path), total, len(sentences))


def run_filtering(corpus_dir: str, partial_dir: str,
                  allowed_vocab: frozenset,
                  n_workers: int, batch_size: int) -> None:
    """Launch parallel filtering on all .gz files, in batches."""
    os.makedirs(partial_dir, exist_ok=True)

    gz_files = sorted(fn for fn in os.listdir(corpus_dir) if fn.endswith(".gz"))
    tasks = [
        (
            os.path.join(corpus_dir, fn),
            os.path.join(partial_dir, fn + "_filt.pkl"),
            allowed_vocab,
        )
        for fn in gz_files
    ]

    cached    = sum(1 for _, op, _ in tasks if os.path.exists(op))
    pending   = len(tasks) - cached
    n_batches = (len(tasks) + batch_size - 1) // batch_size

    print(f"Files found    : {len(tasks)}")
    print(f"Already filtered: {cached}")
    print(f"To filter      : {pending}")
    print(f"Batches        : {n_batches} x {batch_size} files")
    print(f"Vocabulary     : {len(allowed_vocab):,} words\n")

    t0 = time.time()
    total_all    = 0
    filtered_all = 0

    for i in range(0, len(tasks), batch_size):
        batch = tasks[i : i + batch_size]
        b_num = i // batch_size + 1
        print(f"  batch {b_num}/{n_batches}...")

        with mp.Pool(processes=min(n_workers, len(batch))) as pool:
            results = pool.map(filter_file, batch)

        for fname, tot, filt in results:
            if tot == -1:
                print(f"    {fname}: skip ({filt:,} sentences already saved)")
            else:
                pct = 100 * filt / tot if tot else 0
                print(f"    {fname}: {filt:,}/{tot:,} sentences kept ({pct:.1f}%)")
                total_all    += tot
                filtered_all += filt

    print(f"\nFiltering completed in {time.time()-t0:.1f}s")
    if total_all > 0:
        print(f"Total: {filtered_all:,}/{total_all:,} sentences kept "
              f"({100*filtered_all/total_all:.1f}%)")


def merge_partial_pkls(partial_dir: str, final_pkl: str) -> None:
    """Merge all partial pkl files into a single file."""
    print("\nFinal merge...")

    pkl_files = sorted(fn for fn in os.listdir(partial_dir) if fn.endswith("_filt.pkl"))
    all_sentences = []

    for fn in pkl_files:
        with open(os.path.join(partial_dir, fn), "rb") as f:
            all_sentences.extend(pkl.load(f))

    os.makedirs(os.path.dirname(final_pkl) or ".", exist_ok=True)
    with open(final_pkl, "wb") as f:
        pkl.dump(all_sentences, f, protocol=pkl.HIGHEST_PROTOCOL)

    print(f"  {len(all_sentences):,} total sentences saved to {final_pkl}")


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":

    n_workers = N_WORKERS or os.cpu_count()

    print(f"Loading vocabulary from {ALLOWED_VOCAB_PKL}...")
    with open(ALLOWED_VOCAB_PKL, "rb") as f:
        allowed_vocab = pkl.load(f)
    print(f"  {len(allowed_vocab):,} words loaded\n")

    run_filtering(CORPUS_DIR, PARTIAL_DIR, allowed_vocab, n_workers, BATCH_SIZE)
    merge_partial_pkls(PARTIAL_DIR, FINAL_PKL)

    print("\nNext step: python 04_encode_corpus.py")
