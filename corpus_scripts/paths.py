#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
paths.py
--------
Central path configuration for the corpus processing pipeline.

Edit this file to match your local directory layout before running any
of the numbered scripts (01_ through 05_).

All scripts import from this module, so you only need to change paths here.
"""

# =============================================================================
# INPUT
# =============================================================================

# Path to the itemset CSV file (columns: id, type, sentence, ...)
ITEMSET_PATH = "data/itemset_dom.txt"

# Directory containing the OSCAR Spanish corpus as gzipped text files (*.gz)
CORPUS_DIR = "path/to/oscar_es/"

# =============================================================================
# INTERMEDIATE OUTPUTS (created automatically by the pipeline)
# =============================================================================

# Cache of per-file word-count pickles (01_count_corpus.py output)
COUNT_CACHE_DIR = "output/oscar_counters/"

# Filtered-sentence pickles, one per .gz file (03_filter_corpus.py output)
PARTIAL_DIR = "output/oscar_filt_partial/"

# =============================================================================
# FINAL OUTPUTS
# =============================================================================

# Merged Counter of the full corpus (02_merge_counts.py output)
TOTAL_COUNTS_PKL = "output/total_counts.pkl"

# frozenset of allowed vocabulary (02_merge_counts.py output)
ALLOWED_VOCAB_PKL = "output/allowed_vocab.pkl"

# CSV with per-word log-probability for itemset words (02b_itemset_logprob.py)
OUTPUT_CSV = "output/itemset_logprob.csv"

# Single merged list of all filtered sentences (03_filter_corpus.py output)
FINAL_PKL = "output/oscar_filt_all.pkl"

# Plain-text training file with <s>/</ s> tokens (04_encode_corpus.py output)
TRAIN_NWP_TXT = "output/train_nwp.txt"

# Embedding dictionary (token → int index), saved as .pkl by 04_encode_corpus.py
# Note: the .pkl extension is added automatically by the script
EMB_DICT_LOC = "output/train_indices"

# Word log-frequency dictionary, saved as .pkl by 04_encode_corpus.py
FREQ_DICT_LOC = "output/word_freq"
