# Generic Extraction Results

This directory contains the materialized summary for the multi-domain
evaluation of `KRLabsOrg/verbatim-rag-modern-bert-v2`.

The runs compare three extractors on four all-row evaluation slices:

- `generic`: `KRLabsOrg/verbatim-rag-modern-bert-v2`
- `zilliz`: `zilliz/semantic-highlight-bilingual-v1`, token-span output at threshold 0.3
- `provence`: `naver/provence-reranker-debertav3-v1`

Evaluation slices:

- `acl`: ACL-Verbatim gold benchmark, 100 rows.
- `ragbench`: RAGBench native test split converted to span-evaluation format.
- `squeez`: Squeez native test split converted to span-evaluation format.
- `qasper`: QASPER native test split converted to paragraph/table evidence chunks.

All metrics are computed with the same all-row span scorer used for the ACL
paper: rows without gold spans are negatives, and false-positive extracted text
lowers precision.

The source run was executed on an A100 machine using the commands documented in
`docs/GENERIC_EVAL.md`. Latency values are included for reproducibility but are
not treated as controlled benchmark numbers because runtime depends on device,
batching, and model-serving setup.

Files:

- `summary.csv`: current model-card table source.
- `generic.acl_test.json`: detailed all-row ACL gold evaluation for the generic model.
- `generic.acl_test.preds.jsonl`: normalized ACL gold predictions for the generic model.
