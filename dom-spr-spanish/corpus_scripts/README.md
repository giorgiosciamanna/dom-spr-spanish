# corpus_scripts/

Scripts for building the Spanish training corpus and computing surprisal scores.

## Pipeline overview

```
01_count_corpus.py          Count token frequencies per OSCAR .gz file (resumable)
        ↓
02_merge_counts.py          Merge per-file Counters → allowed vocabulary
02b_itemset_logprob.py      (Optional) Log-probability of itemset words in OSCAR
        ↓
03_filter_corpus.py         Filter corpus to allowed vocabulary (resumable)
        ↓
04_encode_corpus.py         Build training text + token index + freq dict
        ↓
  [external] nwp_gru.py     Train GRU model (Merkx & Frank)
  [external] nwp_tf.py      Train Transformer model (Merkx & Frank)
        ↓
05_compute_surprisal_baseline.py    Surprisal from trained GRU/Transformer
06_compute_surprisal_llm.py         Surprisal from HuggingFace LLMs
```

## Configuration

All paths are centralised in `paths.py`. Edit that file before running anything.

## Attribution

The GRU and Transformer training scripts (`nwp_gru.py`, `nwp_tf.py`) and
the surprisal loading script (`load_model.py`) must be obtained directly from:

> Merkx, D., & Frank, S. L. (2021). Human sentence processing: Recurrence or
> attention? *Proceedings of the Workshop on Cognitive Modeling and
> Computational Linguistics (CMCL 2021)*, 12–22.
> Repository: https://github.com/DannyMerkx/next_word_prediction

Clone that repository and point `FUNCTIONS_DIR` in `05_compute_surprisal_baseline.py`
to its `functions/` subfolder.

**Modification applied to the original code:**  
In `nwp_trainer.py`, the `save_states` argument was originally parsed as a plain
string. It was changed to:

```python
import ast
save_states = ast.literal_eval(args.save_states)
```

This allows passing a Python list from the command line, enabling custom
checkpoint intervals without editing the training script:

```bash
python nwp_gru.py ... -save_states '[1000, 10000, 400000, 1000000, 2589164]'
```

## Dependencies

```
pip install pandas numpy torch transformers
```

For the OSCAR filtering pipeline only `pandas` and `pickle` (stdlib) are needed.
For the LLM surprisal script (`06_`), `transformers` and `torch` are required.
For the baseline surprisal script (`05_`), the next_word_prediction repository
must be cloned and its `functions/` folder added to `FUNCTIONS_DIR` in the script.
