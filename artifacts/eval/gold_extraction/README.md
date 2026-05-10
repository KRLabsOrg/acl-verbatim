# Gold Extraction Results

This directory contains small, commit-friendly evaluation artifacts for the
manual gold benchmark in `333_20260206_dense_top5_20260305.json`.

Each file stores per-row predictions and the aggregated summary used in the
paper's extraction table. Evaluations must be run on all 100 benchmark rows:
the 47 relevant rows contain gold spans, and the 53 irrelevant rows are
negative examples with empty gold spans. Predictions on irrelevant rows are
false positives and lower precision.

Included runs:

LLM extractors (via `acl_verbatim/eval/evaluate_extractor.py`):

- `mistral-small-2603.json`
- `mistral-small-2603_paragraph.json`
- `nemotron-120b-a12b.json`
- `nemotron-120b-a12b_paragraph.json`
- `glm-5.json`
- `qwen_default_gold.json`
- `qwen_paragraph_gold.json`

Encoder / pruning baselines (via their respective runners, scored with `evaluate_predictions.py`):

- `zilliz_spans_03_gold.jsonl` — Zilliz Semantic Highlight, token-span mode at threshold 0.3
- `provence_gold.jsonl` — Provence reranker-pruner

Student — GTE-reranker ModernBERT token classifier (main model, pushed as `KRLabsOrg/acl-verbatim-modernbert`):

- `gte-reranker.thr_0.2_merged.json` — headline configuration: threshold $t{=}0.2$ with post-processing (min-span length 10, merge gap 20)

`summary.csv` is the current all-row summary table for the committed runs.

Notes:

- The `.json` files include both per-row records and a `summary` block; they can be loaded with `json.load(...)['summary']`.
- The `.jsonl` files are raw predictions from baseline-specific runners and can be rescored with `acl_verbatim/eval/evaluate_predictions.py`.
- These files are intentionally kept outside `runs/` so they can be versioned and referenced from the paper.
