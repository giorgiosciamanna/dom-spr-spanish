#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
01_count_corpus.py
------------------
STEP 1 of 4 — Word-frequency counting on the OSCAR corpus.

For each .gz file in the corpus directory, a parallel worker counts the
frequency of every lowercased token and saves the result as a separate
.pkl file in COUNT_CACHE_DIR.

This script is fully resumable: if interrupted, it automatically skips
.gz files for which a count .pkl already exists.

Output: a folder of  <filename>.gz_count.pkl  files,
        to be read by the next step (02_merge_counts.py).

Usage:
    python 01_count_corpus.py

All paths are read from paths.py.
"""

import os
import gzip
import pickle as pkl
import time
import multiprocessing as mp

from paths import CORPUS_DIR, COUNT_CACHE_DIR

# =============================================================================
# PARAMETERS
# =============================================================================

N_WORKERS  = None  # None = use all available cores
BATCH_SIZE = 4     # .gz files per batch; lower to 2 if RAM is tight

# =============================================================================


def count_file(args: tuple):
    """
    Worker: opens a .gz file, counts token frequencies (lowercased),
    saves the Counter as a .pkl, and returns it.

    If the .pkl already exists (resume mode), loads it directly without
    re-reading the .gz.

    args: (gz_path, cache_path)
    """
    from collections import Counter

    gz_path, cache_path = args

    # resume: if cache exists, load and return
    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            return pkl.load(f)

    # first run: count token by token
    vocab = Counter()
    with gzip.open(gz_path, "rt", encoding="utf-8", errors="ignore") as f:
        for line in f:
            for token in line.split():
                vocab[token.lower()] += 1

    # save Counter to disk
    with open(cache_path, "wb") as f:
        pkl.dump(vocab, f, protocol=pkl.HIGHEST_PROTOCOL)

    return vocab


if __name__ == "__main__":

    os.makedirs(COUNT_CACHE_DIR, exist_ok=True)
    n_workers = N_WORKERS or os.cpu_count()

    # build list of all tasks (gz → pkl)
    gz_files = sorted(fn for fn in os.listdir(CORPUS_DIR) if fn.endswith(".gz"))
    tasks = [
        (
            os.path.join(CORPUS_DIR, fn),
            os.path.join(COUNT_CACHE_DIR, fn + "_count.pkl"),
        )
        for fn in gz_files
    ]

    cached  = sum(1 for _, cp in tasks if os.path.exists(cp))
    pending = len(tasks) - cached
    n_batches = (len(tasks) + BATCH_SIZE - 1) // BATCH_SIZE

    print(f"Files found:       {len(tasks)}")
    print(f"Already cached:    {cached}")
    print(f"To process:        {pending}")
    print(f"Batches of {BATCH_SIZE} files, {n_batches} total, {n_workers} workers\n")

    if pending == 0:
        print("All files already cached. Proceed with 02_merge_counts.py.")
    else:
        t_start = time.time()

        for i in range(0, len(tasks), BATCH_SIZE):
            batch = tasks[i : i + BATCH_SIZE]
            b_num = i // BATCH_SIZE + 1
            n_cached_batch = sum(1 for _, cp in batch if os.path.exists(cp))
            print(f"[batch {b_num}/{n_batches}] "
                  f"{len(batch)} files, {n_cached_batch} cached, "
                  f"{len(batch) - n_cached_batch} to count...")

            t0 = time.time()
            with mp.Pool(processes=min(n_workers, len(batch))) as pool:
                pool.map(count_file, batch)
            print(f"    → batch done in {time.time()-t0:.1f}s")

        print(f"\nDone. {pending} files processed in {time.time()-t_start:.1f}s")
        print(f".pkl files saved in: {COUNT_CACHE_DIR}")
        print("Next step: python 02_merge_counts.py")
