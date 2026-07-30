# dom-spr-spanish

Code, data, and analysis for:

**"Identifying the Factors Underlying Differential Object Marking with Language Models: analysis and comparison of Self-Paced Reading data and Transformer-based Surprisal scores"**  
Giorgio Sciamanna — Scuola Normale Superiore

---

## Overview

This repository contains all materials for a self-paced reading (SPR) study on
**Differential Object Marking (DOM)** in Spanish. The study examines how
animacy, definiteness, and affectedness modulate the processing cost of
DOM violations, and how well a range of language models (LLMs) predict human
reading times via surprisal.

Three methodological strands are combined:

1. **SPR experiment** — 50 transitive Spanish sentences × 6 conditions (300
   items total), with reading times collected word-by-word and offline
   acceptability judgements from 17 native speakers.

2. **Corpus-trained baseline models** — a GRU and a Transformer trained from
   scratch on a filtered subset of the OSCAR Spanish corpus
   (~2.6M sentences), using the next_word_prediction pipeline
   (Merkx & Frank, 2021).

3. **Large language models** — surprisal scores from four autoregressive LLMs:
   `flax-community/gpt-2-spanish`, `meta-llama/Llama-3.1-8B`,
   `ecastera/eva-dolphin-llama3-8b-spanish`, and `Qwen/Qwen3-8B`.

---

## Repository structure

```
dom-spr-spanish/
├── README.md                      This file
├── LICENSE                        MIT licence
│
├── corpus_scripts/                Python pipeline
│   ├── README.md                  Pipeline documentation and attribution
│   ├── paths.py                   Centralised path configuration (edit first)
│   ├── 01_count_corpus.py         Step 1: count token frequencies on OSCAR
│   ├── 02_merge_counts.py         Step 2: merge counters, build vocabulary
│   ├── 02b_itemset_logprob.py     Step 2b: log-prob of itemset words
│   ├── 03_filter_corpus.py        Step 3: filter corpus to allowed vocabulary
│   ├── 04_encode_corpus.py        Step 4: build training files
│   ├── 05_compute_surprisal_baseline.py   Surprisal from GRU/TF baselines
│   └── 06_compute_surprisal_llm.py        Surprisal from HuggingFace LLMs
│
├── experiment/                    OpenSesame SPR experiment
│   ├── exp_sciamanna.txt          Main experiment script (OpenSesame)
│   ├── itemset_dom.txt            Experimental sentences (300 items)
│   ├── distractors.txt            Distractor sentences
│   ├── practice.txt               Practice items
│   └── affected_test.txt          Affectedness classification stimuli
│
├── data/
│   ├── README.md                  Dataset descriptions
│   ├── combined_data.csv          Raw SPR output (reading times as JSON)
│   ├── offline_data.csv           Acceptability judgements
│   ├── affectedness_data.csv      Affectedness classification task
│   ├── comprehension_data.csv     Comprehension checks
│   ├── baseline_models.csv        GRU / Transformer surprisal scores
│   ├── itemset_logprob.csv        OSCAR corpus log-frequencies
│   ├── itemset_ngram.csv          KenLM n-gram surprisal scores
│   ├── LLM_scores_oscarGPT2.csv   GPT-2 surprisal
│   ├── LLM_scores_Llama-3.1-8B.csv  Llama 3.1 8B surprisal
│   ├── llama3-8b-spanish.csv      Llama Spanish fine-tune surprisal
│   └── Qwen3-8B.csv               Qwen3 8B surprisal
│
└── analysis/
│   ├── analysis.html              Step-by-step analysis with tables and figures
    └── analysis.Rmd               Full R Markdown analysis (LMM, CLMM,
                                   surprisal comparison)
```

---

## Getting started

### 1. Reproduce the corpus pipeline

Edit `corpus_scripts/paths.py` to point to your local copy of the OSCAR
Spanish corpus and your preferred output directory, then run the scripts
in order:

```bash
cd corpus_scripts
python 01_count_corpus.py
python 02_merge_counts.py
python 03_filter_corpus.py
python 04_encode_corpus.py
```

To train the baseline models, clone the next_word_prediction repository and
run `nwp_gru.py` / `nwp_tf.py` as described in `corpus_scripts/README.md`.

### 2. Compute surprisal

```bash
# Baseline (GRU / Transformer)
python 05_compute_surprisal_baseline.py

# LLMs (set MODEL_NAME and OUT_CSV inside the script first)
python 06_compute_surprisal_llm.py
```

### 3. Run the statistical analysis

Open `analysis/analysis.Rmd` in RStudio and knit the document.  
Required R packages: `lme4`, `lmerTest`, `ordinal`, `emmeans`, `buildmer`,
`performance`, `influence.ME`.

If the working directory differs from the `data/` folder, add at the top of
the first chunk:

```r
knitr::opts_knit$set(root.dir = "../data")
```

---

## Attribution

The GRU and Transformer training scripts (`nwp_gru.py`, `nwp_tf.py`) and the
surprisal loading script (`load_model.py`) must be obtained from:

> Merkx, D., & Frank, S. L. (2021). Human sentence processing: Recurrence or
> attention? *Proceedings of the Workshop on Cognitive Modeling and
> Computational Linguistics (CMCL 2021)*, 12–22.
> [https://github.com/DannyMerkx/next_word_prediction](https://github.com/DannyMerkx/next_word_prediction)

**Modification applied to the original code:** in `nwp_trainer.py`, the
`save_states` argument was changed from plain string parsing to
`ast.literal_eval()`, enabling custom checkpoint intervals to be passed as a
Python list on the command line (see `corpus_scripts/README.md` for details).

---

## Corpus

The training corpus is a filtered subset of
[OSCAR](https://oscar-project.org/) (Spanish, v1), distributed as gzipped
plain-text files. Filtering retains only sentences whose vocabulary is
entirely within the 100,000 most frequent Spanish words plus the experimental
words, yielding approximately 2.6 million sentences.

---

## Licence

Code: MIT (see `LICENSE`).  
Data: the experimental stimuli and behavioural data are released for
reproducibility purposes only; please cite the paper if you use them.
