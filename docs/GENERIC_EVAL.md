# Generic span-extractor evaluation

This document describes the multi-domain evaluation of
[`KRLabsOrg/verbatim-rag-modern-bert-v2`](https://huggingface.co/KRLabsOrg/verbatim-rag-modern-bert-v2),
the generic counterpart of [`KRLabsOrg/acl-verbatim-modernbert`](https://huggingface.co/KRLabsOrg/acl-verbatim-modernbert).

The ACL model is fine-tuned only on ACL silver labels and reported in the
[main README](../README.md). The generic model is fine-tuned on the
[`KRLabsOrg/verbatim-spans`](https://huggingface.co/datasets/KRLabsOrg/verbatim-spans)
mix (ACL silver + RAGBench + Squeez) and is intended for use *outside* the
ACL domain. We evaluate it on four held-out test sets so the model card can
report per-domain numbers and direct comparisons to public baselines.

## Test sets

| name | source | labels | size | notes |
|---|---|---|---|---|
| ACL gold | this repo (`333_20260206_dense_top5_20260305.json`) | human-curated | 47 relevant rows | ACL Anthology paper chunks; same set used in the main paper |
| RAGBench test | `galileo-ai/ragbench`, all 12 configs, native `test` split | GPT-4o | ~17k rows | Cross-domain QA: finance, medical, legal, general |
| Squeez test | `KRLabsOrg/tool-output-extraction-swebench-gliner` native `test` split | GLiNER-style silver | ~1k rows | Coding-agent tool outputs (SWE-bench) |
| MultiSpanQA valid | [official MultiSpanQA release](https://multi-span.github.io/) | human (BIO) | 653 rows | Direct comparison point — Zilliz and Provence both publish on this set |

For RAGBench and Squeez we use the **source datasets' own test splits**, not
slices of `verbatim-spans/validation`. This keeps our eval data identical to
what the upstream community evaluates on, and avoids re-hosting silver labels
that already live in canonical places.

## Building the test slices

```bash
mkdir -p runs/eval_data runs/eval/test_slices

# RAGBench native test (all 12 configs)
python scripts/experiments/prepare_ragbench_spans.py \
    --split test \
    --output-file runs/eval_data/ragbench_test.spans.jsonl
python scripts/experiments/spans_jsonl_to_gold_file.py \
    runs/eval_data/ragbench_test.spans.jsonl \
    --output runs/eval/test_slices/ragbench.gold.jsonl

# Squeez native test
python scripts/experiments/prepare_squeez_spans.py \
    --split test \
    --output-file runs/eval_data/squeez_test.spans.jsonl
python scripts/experiments/spans_jsonl_to_gold_file.py \
    runs/eval_data/squeez_test.spans.jsonl \
    --output runs/eval/test_slices/squeez.gold.jsonl

# MultiSpanQA valid (download valid.json from https://multi-span.github.io/)
python scripts/experiments/multispanqa_to_gold_file.py \
    /path/to/MultiSpanQA_data/valid.json \
    --output runs/eval/test_slices/multispanqa.gold.jsonl

# ACL gold benchmark (already in the right format)
cp 333_20260206_dense_top5_20260305.json runs/eval/test_slices/acl.gold.jsonl

wc -l runs/eval/test_slices/*.gold.jsonl
```

## Running the sweep

The same four-system grid as the main paper (generic student, ACL-specialized
student, Zilliz semantic-highlight, Provence) on each slice. Best decoding
config per model was selected on the dataset's validation split, not the test
split — see the [main paper](../paper.tex) §4.4 for the threshold-selection
protocol.

```bash
SLICES=(acl ragbench squeez multispanqa)

# Generic student
for S in "${SLICES[@]}"; do
  python acl_verbatim/span_training/evaluate_token_cls.py \
    --gold-file runs/eval/test_slices/${S}.gold.jsonl \
    --model-dir KRLabsOrg/verbatim-rag-modern-bert-v2 \
    --threshold 0.2 --min-span-chars 30 --merge-gap-chars 20 \
    --output-file runs/eval/generic.${S}_test.json
done

# ACL-specialized student (HF release)
for S in "${SLICES[@]}"; do
  python acl_verbatim/span_training/evaluate_token_cls.py \
    --gold-file runs/eval/test_slices/${S}.gold.jsonl \
    --model-dir KRLabsOrg/acl-verbatim-modernbert \
    --threshold 0.2 --min-span-chars 10 --merge-gap-chars 20 \
    --output-file runs/eval/acl-modernbert.${S}_test.json
done

# Zilliz semantic-highlight (predict, then score)
for S in "${SLICES[@]}"; do
  python acl_verbatim/eval/run_semantic_highlight.py \
    --gold-file runs/eval/test_slices/${S}.gold.jsonl \
    --output-file runs/eval/zilliz.${S}_test.preds.jsonl \
    --output-mode spans --threshold 0.3
  python acl_verbatim/eval/evaluate_predictions.py \
    --gold-file runs/eval/test_slices/${S}.gold.jsonl \
    --pred-file runs/eval/zilliz.${S}_test.preds.jsonl \
    --output-file runs/eval/zilliz.${S}_test.json
done

# Provence (predict, then score)
for S in "${SLICES[@]}"; do
  python acl_verbatim/eval/run_provence.py \
    --gold-file runs/eval/test_slices/${S}.gold.jsonl \
    --output-file runs/eval/provence.${S}_test.preds.jsonl
  python acl_verbatim/eval/evaluate_predictions.py \
    --gold-file runs/eval/test_slices/${S}.gold.jsonl \
    --pred-file runs/eval/provence.${S}_test.preds.jsonl \
    --output-file runs/eval/provence.${S}_test.json
done
```

## Tabulating per slice

```bash
for S in acl ragbench squeez multispanqa; do
  echo "=== $S ==="
  python acl_verbatim/eval/compare_span_runs.py \
    --gold-file runs/eval/test_slices/${S}.gold.jsonl \
    --run generic=runs/eval/generic.${S}_test.json \
    --run acl-modernbert=runs/eval/acl-modernbert.${S}_test.json \
    --run zilliz=runs/eval/zilliz.${S}_test.json \
    --run provence=runs/eval/provence.${S}_test.json
done
```

The four printed tables are the spine of the
[`KRLabsOrg/verbatim-rag-modern-bert-v2`](https://huggingface.co/KRLabsOrg/verbatim-rag-modern-bert-v2)
model card.

## Caveats

- **Label provenance differs across slices.** ACL gold and MultiSpanQA are
  human-curated; RAGBench is GPT-4o-annotated; Squeez is GLiNER-style silver.
  All three are the test sets the upstream community uses, but the absolute
  numbers across slices are not directly comparable.
- **Zilliz and Provence were never trained on Squeez-style structured tool
  output.** Their numbers on the Squeez slice say more about scope than
  capability — note this in any comparison.
- **MultiSpanQA reconstruction is space-joined tokens.** The dataset only
  ships token-level annotations; we join with single spaces and derive char
  offsets. All systems see the same reconstructed text.
