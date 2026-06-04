#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
04_encode_corpus.py
-------------------
STEP 4 of 4 — Encoding and preparation of training files.

Input (produced by previous steps):
  - oscar_filt_all.pkl    filtered sentences (from 03_filter_corpus.py)
  - total_counts.pkl      pruned global Counter (from 02_merge_counts.py)
  - itemset_dom.txt       experimental words

Pipeline:
  1. Builds freq_dict: for each corpus word,
       freq_dict[w] = -log(count(w) / N_total)
     used as the log-frequency predictor in the LMER analysis.
     N_total = sum of counts in total_counts (a conservative approximation:
     rare words discarded during pruning are negligible for itemset and
     filtered vocabulary words).

  2. Sorts sentences: itemset-containing sentences first.

  3. Adds <s> and </s> to each sentence and builds emb_dict:
       emb_dict[w] = integer index  (0 = OOV, reserved)

  4. Saves the three files expected by the training script:
       train_nwp.txt       text with <s>/</ s>, one sentence per line
       train_indices.pkl   emb_dict (defaultdict → index)
       word_freq.pkl       freq_dict (dict → -log P(w))

Usage:
    python 04_encode_corpus.py

All paths are read from paths.py.
"""

import os
import pickle as pkl
import math
import pandas as pd
from collections import defaultdict

from paths import (
    ITEMSET_PATH, FINAL_PKL, TOTAL_COUNTS_PKL,
    TRAIN_NWP_TXT, EMB_DICT_LOC, FREQ_DICT_LOC,
)


def save_pkl(obj, loc: str):
    """Save a pickled object, appending .pkl extension automatically."""
    with open(str(loc) + ".pkl", "wb") as f:
        pkl.dump(obj, f, protocol=pkl.HIGHEST_PROTOCOL)


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


# =============================================================================
# STEP 1 — freq_dict
# =============================================================================

def build_freq_dict(total_counts_pkl: str) -> dict:
    """
    Load the global Counter (from 02_merge_counts.py) and build:
      freq_dict[w] = -log(count(w) / N_total)

    N_total is the sum of counts in the pruned Counter: slightly underestimates
    the true total (rare words discarded by pruning are excluded), but this
    does not affect the LMER predictor validity since all filtered vocabulary
    words are present in the Counter.
    """
    print("[1] Building freq_dict...")

    with open(total_counts_pkl, "rb") as f:
        total_counts = pkl.load(f)

    n_total = sum(total_counts.values())
    print(f"    N_total (from pruned Counter): {n_total:,}")

    freq_dict = {
        w: -math.log(c / n_total)
        for w, c in total_counts.items()
    }

    print(f"    {len(freq_dict):,} words in freq_dict")
    return freq_dict


# =============================================================================
# STEP 2 — Load and sort sentences
# =============================================================================

def load_and_sort(final_pkl: str, itemset_words: frozenset) -> list:
    """
    Load filtered sentences and sort them so that itemset-containing sentences
    come first, ensuring the model encounters them early during training.
    """
    print("[2] Loading filtered sentences...")
    with open(final_pkl, "rb") as f:
        sentences = pkl.load(f)
    print(f"    {len(sentences):,} sentences loaded")

    print("    Sorting (itemset sentences first)...")
    sentences.sort(
        reverse=True,
        key=lambda s: int(any(w in itemset_words for w in s.split()))
    )
    n_with = sum(1 for s in sentences if any(w in itemset_words for w in s.split()))
    print(f"    {n_with:,} sentences contain at least one itemset word")
    return sentences


# =============================================================================
# STEP 3 — Encoding and saving
# =============================================================================

def encode_and_save(sentences: list, freq_dict: dict,
                    train_nwp_txt: str,
                    emb_dict_loc: str,
                    freq_dict_loc: str) -> None:
    """
    For each sentence:
      - add <s> (beginning) and </s> (end)
      - assign an integer index to each new token in emb_dict
        (0 = OOV reserved, assignment starts from 1)

    Saves three files:
      train_nwp.txt      one sentence per line with <s>/</ s>
      train_indices.pkl  emb_dict: defaultdict(int) token → index
      word_freq.pkl      freq_dict: dict token → -log P(w)
    """
    print("[3] Encoding and adding special tokens...")

    emb_dict = defaultdict(int)
    ind = 1   # 0 = OOV, reserved

    for i, sent in enumerate(sentences):
        tokens = ["<s>"] + sent.split() + ["</s>"]
        for w in tokens:
            if emb_dict[w] == 0:
                emb_dict[w] = ind
                ind += 1
        sentences[i] = " ".join(tokens)

    print(f"    Final vocabulary: {ind - 1:,} tokens (+ OOV=0)")

    print("[4] Saving training files...")

    os.makedirs(os.path.dirname(train_nwp_txt) or ".", exist_ok=True)
    with open(train_nwp_txt, "w", encoding="utf-8") as f:
        for line in sentences:
            f.write(line + "\n")
    print(f"    -> {train_nwp_txt}")

    save_pkl(emb_dict, emb_dict_loc)
    print(f"    -> {emb_dict_loc}.pkl")

    save_pkl(freq_dict, freq_dict_loc)
    print(f"    -> {freq_dict_loc}.pkl")


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":

    itemset_words = extract_itemset_words(ITEMSET_PATH)
    freq_dict     = build_freq_dict(TOTAL_COUNTS_PKL)
    sentences     = load_and_sort(FINAL_PKL, itemset_words)

    encode_and_save(
        sentences, freq_dict,
        train_nwp_txt = TRAIN_NWP_TXT,
        emb_dict_loc  = EMB_DICT_LOC,
        freq_dict_loc = FREQ_DICT_LOC,
    )

    print("\nDone. Files ready for training:")
    print(f"  {TRAIN_NWP_TXT}")
    print(f"  {EMB_DICT_LOC}.pkl")
    print(f"  {FREQ_DICT_LOC}.pkl")

    # Example training call (uncomment and adapt):
    # import os
    # save_states = [1000, 10_000, 400_000, 1_000_000, 2_000_000, N_SENTENCES, ...]
    # os.system(
    #     "python -u next_word_prediction/language_modelling/nwp_gru.py "
    #     "-model_ids 1 -param xavier -bias none "
    #     f"-data_loc output/ "
    #     f"-dict_loc {EMB_DICT_LOC} "
    #     f"-results_loc output/oscar_models/ "
    #     f"-save_states '{save_states}' "
    #     "> output/gru_training.out"
    # )
