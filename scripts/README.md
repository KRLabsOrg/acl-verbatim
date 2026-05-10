# Scripts

These scripts are grouped by workflow. The main package entrypoints live under
`acl_verbatim/`; this directory contains repository-level utilities for data
materialization, publishing, and evaluation adapters.

## `corpus/`

Build or materialize the ACL Anthology markdown corpus and index it.

- `export_hf_corpus.py` downloads `KRLabsOrg/acl-anthology-md` into the local
  `paper_data.jsonl` + `acl_md/*.md` layout.
- `get_anthology_metadata.py` extracts metadata from a local
  `acl-org/acl-anthology` checkout.
- `preprocess_acl.py` converts downloaded PDFs to markdown with docling.
- `index_acl.py` chunks markdown and builds a Milvus index.
- `build_corpus_dataset.py` assembles and publishes the HF corpus dataset.

## `publish/`

Create Hugging Face datasets and model repos from prepared local artifacts.

- `build_spans_dataset.py` publishes the ACL-only span dataset.
- `build_generic_spans_dataset.py` publishes the multi-domain span dataset.
- `push_model.py` wraps a trained token classifier with the self-contained
  `.process()` model API and uploads it.

## `experiments/`

Adapters and helpers for non-ACL evaluation sets and exploratory runs.

- `prepare_ragbench_spans.py` and `prepare_squeez_spans.py` convert external
  datasets to the common span format.
- `qasper_to_gold_file.py`, `spans_jsonl_to_gold_file.py`, and
  `multispanqa_to_gold_file.py` convert public evaluation data to the evaluator
  gold-file format.
- `eval_multispanqa_metrics.py` and `eval_squeez_metrics.py` are dataset-specific
  metric helpers.
- `slice_verbatim_val.py` and `search_index.py` are small experiment helpers.

## `maintenance/`

Operational utilities that modify external services.

- `clean_collections.py` deletes Milvus collections matching the supplied
  arguments. Treat it as an admin script.
