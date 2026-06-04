#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
05_compute_surprisal_baseline.py
---------------------------------
Compute per-sentence surprisal scores from the custom GRU and Transformer
baseline models trained with the next_word_prediction pipeline
(Merkx & Frank, https://github.com/DannyMerkx/next_word_prediction).

Input:
  - itemset_dom.txt             experimental sentences (CSV)
  - train_indices.pkl           token-to-index dictionary (from 04_encode_corpus.py)
  - gru_model_1_<N>             trained GRU model checkpoint
  - tf_model_1_<N>              trained Transformer model checkpoint

Output:
  - baseline_models.csv         CSV with columns:
      id, type, sentence, oov, gru_all, gru_obj, tf_all, tf_obj

    gru_obj / tf_obj = surprisal of the object noun
                       (7th word from the end of the sentence)
    oov              = True if at least one non-object token is OOV

Prerequisites:
  - The next_word_prediction repository must be cloned locally.
    Clone from: https://github.com/DannyMerkx/next_word_prediction
  - Set FUNCTIONS_DIR below to point to the 'functions' subfolder.
  - Set MODEL_GRU and MODEL_TF to the checkpoint paths of your trained models.

Usage:
    python 05_compute_surprisal_baseline.py

All shared paths are read from paths.py.
"""

import sys
import pickle
import torch
import numpy as np
import pandas as pd
import os

from paths import ITEMSET_PATH, EMB_DICT_LOC

# =============================================================================
# PATHS — edit to match your setup
# =============================================================================

# Path to the 'functions' folder inside the next_word_prediction repository
FUNCTIONS_DIR = "path/to/next_word_prediction/functions"

# Trained model checkpoints (produced by nwp_gru.py / nwp_tf.py)
MODEL_GRU = "path/to/oscar_models/gru_model_1_<N>"
MODEL_TF  = "path/to/oscar_models/tf_model_1_<N>"

# Output CSV
OUT_CSV = "output/baseline_models.csv"

# Word position of the object (counting from the end of the sentence)
OBJ_FROM_END = 7

# =============================================================================

sys.path.append(FUNCTIONS_DIR)
from encoders import nwp_rnn_tf_att, nwp_transformer
from prep_text import word_2_index


def load_obj(loc):
    with open(loc + ".pkl", "rb") as f:
        return pickle.load(f)


nwp_dict  = load_obj(EMB_DICT_LOC)
dict_size = len(nwp_dict) + 1

# --- load itemset and normalise sentences ---
df        = pd.read_csv(ITEMSET_PATH)
sentences = [s.lower().rstrip(".").strip() for s in df["sentence"]]
tokenized = [["<s>"] + s.split() + ["</s>"] for s in sentences]

encoded, lengths = word_2_index(tokenized, len(tokenized), nwp_dict)

# --- OOV flag ---
df["oov"] = [
    any(idx == 0 for idx in enc[1 : l - 1])
    for enc, l in zip(encoded, lengths)
]
print(f"OOV sentences: {df['oov'].sum()}")

# --- model configs ---
gru_config = {
    "embed": {"n_embeddings": dict_size, "embedding_dim": 400,
              "sparse": False, "padding_idx": 0},
    "max_len": 41,
    "rnn":  {"in_size": 400, "hidden_size": 500, "n_layers": 1,
             "batch_first": True, "bidirectional": False, "dropout": 0},
    "lin":  {"hidden_size": 400},
    "att":  {"in_size": 500, "heads": 10},
    "cuda": False,
}

tf_config = {
    "embed": {"n_embeddings": dict_size, "embedding_dim": 400,
              "sparse": False, "padding_idx": 0},
    "tf":   {"in_size": 400, "fc_size": 1024, "n_layers": 1,
             "h": 8, "max_len": 41},
    "cuda": False,
}

models = {
    "gru": (nwp_rnn_tf_att(gru_config), MODEL_GRU),
    "tf":  (nwp_transformer(tf_config),  MODEL_TF),
}


def calc_surprisal(sentences, model):
    """Return a list of arrays — one per-word surprisal array per sentence."""
    tokenized        = [["<s>"] + s.split() + ["</s>"] for s in sentences]
    encoded, lengths = word_2_index(tokenized, len(tokenized), nwp_dict)

    predictions, targets = model(torch.FloatTensor(encoded), lengths)
    surp = -torch.log_softmax(predictions, dim=2).squeeze()
    surp = surp.gather(-1, targets.unsqueeze(-1)).squeeze().data.numpy()
    # remove padding and </s>: each array corresponds to sentence tokens
    return [s[:l - 2] for s, l in zip(surp, lengths)]


# --- loop over the two models ---
for name, (arch, path) in models.items():
    print(f"Loading {name}...")
    arch.load_state_dict(torch.load(path, map_location="cpu"))
    for p in arch.parameters():
        p.requires_grad = False
    arch.eval()

    surprisal = calc_surprisal(sentences, arch)

    df[f"{name}_all"] = [[round(float(x), 4) for x in s] for s in surprisal]
    df[f"{name}_obj"] = [
        round(float(s[-OBJ_FROM_END]), 4) if len(s) >= OBJ_FROM_END else np.nan
        for s in surprisal
    ]
    print(f"  -> {name}_all, {name}_obj")

os.makedirs(os.path.dirname(OUT_CSV) or ".", exist_ok=True)
df.to_csv(OUT_CSV, index=False)
print(f"\nSaved: {OUT_CSV}")
print(df[["id", "type", "sentence", "oov", "gru_obj", "tf_obj"]].head(6).to_string(index=False))
