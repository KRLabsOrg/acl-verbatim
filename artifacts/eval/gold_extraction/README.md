# Gold Extraction Results

This directory contains small, commit-friendly evaluation artifacts for the
manual gold benchmark in `333_20260206_dense_top5_20260305.json`.

Each file stores per-row predictions and the aggregated summary used in the
paper's extraction table.

Included runs:

- `mistral-small-2603.json`
- `nemotron-120b-a12b.json`
- `nemotron-120b-a12b_paragraph.json`
- `glm-5.json`
- `qwen_default_gold.json`
- `qwen_paragraph_gold.json`
- `zilliz_gold.jsonl`
- `zilliz_sent_03_gold.jsonl`
- `zilliz_spans_05_gold.jsonl`
- `zilliz_spans_03_gold.jsonl`
- `provence_gold.jsonl`

Notes:

- The `.json` files were produced by `acl_verbatim/eval/evaluate_extractor.py`
  and include both per-row records and a `summary` block.
- The `.jsonl` files were produced by baseline-specific runners and can be
  rescored with `acl_verbatim/eval/evaluate_predictions.py`.
- The additional `zilliz_*` files are threshold / output-mode ablations used to
  check whether the default Zilliz operating point was overly conservative.
- These files are intentionally kept outside `runs/` so they can be versioned
  and referenced from the paper.
