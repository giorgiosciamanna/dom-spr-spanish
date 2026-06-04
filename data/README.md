# data/

This folder contains all datasets used in the project.

## Experimental data (raw and processed)

These files come directly from the self-paced reading (SPR) experiment
conducted in OpenSesame.

| File | Description |
|------|-------------|
| `itemset_dom.txt` | The experimental item set: 50 items × 6 conditions (300 sentences). Columns: `id`, `type`, `sentence`, ... |
| `combined_data.csv` | Raw OpenSesame output. Reading times are stored in the `response` column as JSON lists (in seconds). This is the canonical source for RT data. |
| `offline_data.csv` | Acceptability judgements (16 participants, scale 0–4). |
| `affectedness_data.csv` | Affectedness classification task on reversed sentences (16 participants). |
| `comprehension_data.csv` | True/false comprehension check. Correct answer is reconstructed as: `comprehension` (A/E) × `answer` (SI/NO). |
| `reading_data.csv` | Pre-processed RT data (no longer used by the analysis; `combined_data.csv` is used instead). |

## Model scores

Surprisal scores for each itemset sentence × condition, output by the
scripts in `corpus_scripts/`.

| File | Model | Notes |
|------|-------|-------|
| `baseline_models.csv` | Custom GRU + Transformer trained on OSCAR-ES | Columns: `gru_obj`, `tf_obj`, `oov`. Items 17 and 22 are OOV for these models. |
| `itemset_logprob.csv` | OSCAR-ES corpus frequencies | `neg_log_prob` = -log P(w); used as lexical frequency predictor in LMMs. |
| `itemset_ngram.csv` | KenLM n-gram model | Per-region surprisal: `kenlm_object`, `kenlm_pp1`, `kenlm_pp2`. |
| `LLM_scores_oscarGPT2.csv` | GPT-2 fine-tuned on Spanish OSCAR | `surprisal_obj` column. 266 rows (item×condition combinations present in the experiment). |
| `LLM_scores_Llama-3.1-8B.csv` | Llama 3.1 8B (base) | `surprisal_obj` column. |
| `llama3-8b-spanish.csv` | Llama 3.1 8B fine-tuned on Spanish | `surprisal_obj` column. |
| `Qwen3-8B.csv` | Qwen3 8B | `surprisal_obj` column. |

## Experimental design summary

- **50 items**, each in **6 conditions** (`type` 0–5).
- Sentence structure: Subject + Verb + [a] + Article + Object + PP1 + PP2.
- Factors: animacy (human/inanimate), definiteness (definite/indefinite),
  DOM marking (presence/absence of the `a` clitic).
- `correct` = 1 if the DOM marking is grammatically expected (types 0, 1, 5).
- `affected` = 1 if `id < 25` (determined by verb, following Heredero & García 2023).
- 17 Spanish L1 participants; sociolinguistic background: standard European (7),
  Latin American (3), Catalan-dominant (7).
