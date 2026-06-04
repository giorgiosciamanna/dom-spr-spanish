#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
06_compute_surprisal_llm.py
---------------------------
Compute per-word surprisal scores for the itemset sentences using
autoregressive large language models (LLMs) from HuggingFace.

Models used in the study (all available on HuggingFace Hub):

    flax-community/gpt-2-spanish        Spanish GPT-2
    meta-llama/Llama-3.1-8B             Llama 3.1 8B (base)
    ecastera/eva-dolphin-llama3-8b-spanish  Llama 3.1 8B fine-tuned on Spanish
    Qwen/Qwen3-8B                       Qwen3 8B (official)

The script can be run once per model by changing MODEL_NAME below.

Output CSV columns:
    id              | item id (from itemset)
    type            | condition code (0–5)
    DOM             | 1 if 'a' marker is present, 0 otherwise
    sentence        | original sentence string
    surprisal_list  | list of (token_str, surprisal) tuples, one per word
    surprisal_obj   | surprisal of the object noun

Object position:
    The object noun is the 4th word (0-indexed, after subject, verb, article)
    in non-DOM sentences, or the 5th word (after subject, verb, 'a', article)
    in DOM sentences. This is encoded as OBJ_WORD_INDEX_DOM /
    OBJ_WORD_INDEX_NODOM below, and derived from the item's DOM column.

Sentence structure (all sentences, 10-11 words):
    Subject  Verb  [a]  Article  Object  PP1_prep  PP1_noun  PP2_prep  PP2_word
                    ^DOM only

Word-level surprisal:
    Because LLMs use subword tokenisation (BPE/SentencePiece), word-level
    surprisal is computed as the SUM of surprisal values over all subword
    tokens belonging to the same word (space-prefixed token = new word).

Usage:
    1. Set MODEL_NAME to a HuggingFace model ID (or local path).
    2. Set OUT_CSV to the desired output path.
    3. Run:
           python 06_compute_surprisal_llm.py

    For gated models (e.g. Llama) you need a HuggingFace access token:
        huggingface-cli login
    or set the environment variable HF_TOKEN before running.

All shared paths are read from paths.py.
"""

import os
import math
import ast
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from paths import ITEMSET_PATH

# =============================================================================
# CONFIGURATION — edit here
# =============================================================================

# HuggingFace model ID — change to the model you want to run.
# Models used in the study:
#   "flax-community/gpt-2-spanish"
#   "meta-llama/Llama-3.1-8B"
#   "ecastera/eva-dolphin-llama3-8b-spanish"
#   "Qwen/Qwen3-8B"
MODEL_NAME = "flax-community/gpt-2-spanish"

# Output CSV path
OUT_CSV = "output/LLM_scores.csv"

# Word indices of the object noun (0-indexed from the start of the sentence)
OBJ_WORD_INDEX_DOM   = 4   # Subject(0) Verb(1) a(2) Article(3) Object(4) ...
OBJ_WORD_INDEX_NODOM = 3   # Subject(0) Verb(1) Article(2) Object(3) ...

# Device: 'cuda' if a GPU is available, otherwise 'cpu'
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =============================================================================


def load_model_and_tokenizer(model_name: str):
    """Load model and tokenizer from HuggingFace."""
    print(f"Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    print(f"Loading model: {model_name}  (device={DEVICE})")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
        device_map=DEVICE,
    )
    model.eval()
    return model, tokenizer


def word_surprisal(sentence: str, model, tokenizer) -> list:
    """
    Compute per-word surprisal for a sentence.

    Returns a list of (word_str, surprisal_nats) tuples, one per word.
    Surprisal of word w_i = -log P(w_i | w_1...w_{i-1}).
    Subword tokens belonging to the same word are summed.
    """
    inputs = tokenizer(sentence, return_tensors="pt").to(DEVICE)
    input_ids = inputs["input_ids"]

    with torch.no_grad():
        outputs = model(**inputs, labels=input_ids)
        # logits: (1, seq_len, vocab_size)
        logits = outputs.logits

    # shifted: logits[t] predicts token[t+1]
    log_probs = torch.log_softmax(logits[0], dim=-1)  # (seq_len, vocab)

    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    n_tok  = len(tokens)

    # surprisal for each token position (excluding the first, which has no
    # preceding context to condition on)
    token_surprisals = []
    for t in range(1, n_tok):
        target_id = input_ids[0, t].item()
        surp = -log_probs[t - 1, target_id].item()
        token_surprisals.append((tokens[t], surp))

    # aggregate subword tokens into words
    # A new word begins when the token has a leading space character
    # (BPE/Ġ prefix) or is the very first token after BOS.
    word_surprisals = []
    current_word    = None
    current_surp    = 0.0

    def is_word_start(tok: str) -> bool:
        """Return True if this token starts a new word."""
        # Covers Ġ (GPT-2/Llama), ▁ (SentencePiece), plain space prefix
        return tok.startswith(("Ġ", "▁", " ")) or tok.startswith("##") is False

    for i, (tok, surp) in enumerate(token_surprisals):
        clean = tok.lstrip("Ġ▁ ")
        starts_new = tok.startswith(("Ġ", "▁", " ")) or i == 0

        if starts_new and current_word is not None:
            word_surprisals.append((current_word, current_surp))
            current_word = clean
            current_surp = surp
        elif starts_new:
            current_word = clean
            current_surp = surp
        else:
            # continuation subword: accumulate surprisal
            current_word  = (current_word or "") + clean
            current_surp += surp

    if current_word is not None:
        word_surprisals.append((current_word, current_surp))

    return word_surprisals


def process_itemset(itemset_path: str, model, tokenizer) -> pd.DataFrame:
    """
    Compute surprisal for all itemset sentences and return a DataFrame.
    """
    df = pd.read_csv(itemset_path)

    # Determine DOM column: try 'DOM' first, fall back to inferring from type
    if "DOM" not in df.columns:
        # types 0,1,2 have DOM; types 3,4,5 do not
        df["DOM"] = (df["type"] % 6 < 3).astype(int)

    surprisal_lists = []
    surprisal_objs  = []

    for _, row in df.iterrows():
        sentence = row["sentence"].strip()
        dom      = int(row["DOM"])

        ws = word_surprisal(sentence, model, tokenizer)

        obj_idx = OBJ_WORD_INDEX_DOM if dom else OBJ_WORD_INDEX_NODOM

        surprisal_lists.append(ws)
        if obj_idx < len(ws):
            surprisal_objs.append(round(ws[obj_idx][1], 6))
        else:
            surprisal_objs.append(float("nan"))

    df["surprisal_list"] = surprisal_lists
    df["surprisal_obj"]  = surprisal_objs

    return df[["id", "type", "DOM", "sentence", "surprisal_list", "surprisal_obj"]]


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":

    model, tokenizer = load_model_and_tokenizer(MODEL_NAME)

    print(f"\nProcessing itemset: {ITEMSET_PATH}")
    result = process_itemset(ITEMSET_PATH, model, tokenizer)

    os.makedirs(os.path.dirname(OUT_CSV) or ".", exist_ok=True)
    result.to_csv(OUT_CSV, index=False)

    print(f"\nSaved: {OUT_CSV}  ({len(result)} rows)")
    print(result[["id", "type", "DOM", "surprisal_obj"]].head(8).to_string(index=False))
