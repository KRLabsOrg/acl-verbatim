# Gold Extraction Results

This directory contains small, commit-friendly evaluation artifacts for the
manual gold benchmark in `333_20260206_dense_top5_20260305.json`.

Each file stores per-row predictions and the aggregated summary used in the
paper's extraction table.

Included runs:

LLM extractors (via `acl_verbatim/eval/evaluate_extractor.py`):

- `mistral-small-2603.json`
- `nemotron-120b-a12b.json`
- `nemotron-120b-a12b_paragraph.json`
- `glm-5.json`
- `qwen_default_gold.json`
- `qwen_paragraph_gold.json`

Encoder / pruning baselines (via their respective runners, scored with `evaluate_predictions.py`):

- `zilliz_gold.jsonl` — Zilliz Semantic Highlight, default config
- `zilliz_sent_03_gold.jsonl`, `zilliz_spans_03_gold.jsonl`, `zilliz_spans_05_gold.jsonl` — Zilliz threshold / output-mode ablation
- `provence_gold.jsonl` — Provence reranker-pruner

Student — ModernBERT-base (MLM) token classifier (ablation):

- `modernbert-base.gold_eval.json` — argmax eval
- `modernbert-base.thr_0.3.json` — threshold $t{=}0.3$ (the configuration reported in the main table)

Student — GTE-reranker ModernBERT token classifier (main model, pushed as `KRLabsOrg/acl-verbatim-modernbert`):

- `gte-reranker.gold_eval.json` — argmax eval
- `gte-reranker.gold_preds.jsonl` — argmax per-row predictions (for qualitative analysis)
- `gte-reranker.thr_0.2_merged.json` — headline configuration: threshold $t{=}0.2$ with post-processing (min-span length 10, merge gap 20)

Full threshold and silver-validation sweeps are not committed (they are regenerable via `acl_verbatim/span_training/evaluate_token_cls.py` and the sweep numbers are in the paper appendix).

Notes:

- The `.json` files include both per-row records and a `summary` block; they can be loaded with `json.load(...)['summary']`.
- The `.jsonl` files are raw predictions from baseline-specific runners and can be rescored with `acl_verbatim/eval/evaluate_predictions.py`.
- These files are intentionally kept outside `runs/` so they can be versioned and referenced from the paper.
