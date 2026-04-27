# Evaluation

The repository currently supports three evaluation layers:

1. retrieval evaluation over synthetic query files
2. span-extraction evaluation against the local gold benchmark
3. token-classifier evaluation against that same gold benchmark

## Retrieval

Run batch retrieval with:

```bash
python acl_verbatim/eval/test_index.py \
  --collection-name acl \
  --use-cloud \
  --cloud-uri CLOUD_URI \
  --milvus-token MILVUS_TOKEN \
  --device cpu \
  -r \
  -k 5 \
  --questions-dir QUERIES_DIR \
  --query-field query \
  --output-file runs/search_results_top5_hybrid.jsonl \
  -s hybrid
```

`test_index.py` prints retrieval recall metrics directly. Current code defaults to equal hybrid
weights (`0.5` dense, `0.5` full text).

## Span Evaluation

To evaluate any system that emits normalized `pred_spans` JSONL:

```bash
python acl_verbatim/eval/evaluate_predictions.py \
  --gold-file 333_20260206_dense_top5_20260305.json \
  --pred-file runs/system_preds.jsonl \
  --output-file runs/system_eval.json
```

Supported prediction rows look like:

```json
{
  "question": "query text",
  "paper_id": "2023.acl-long.42",
  "chunk_index": 7,
  "pred_spans": [
    {"start": 120, "end": 248, "text": "exact substring"}
  ]
}
```

The shared metrics are implemented in `span_metrics.py` and include:

- word-level precision / recall / F1
- span IoU F1 at `0.3`, `0.5`, `0.7`
- containment F1 at `0.5`, `0.8`, `1.0`
- gold-coverage recall at `0.5`, `0.8`, `1.0`
- recall@any-overlap
- over-prediction ratio

## LLM Teacher Evaluation

The LLM extractor harness runs the same shared scorer directly:

```bash
python acl_verbatim/eval/evaluate_extractor.py \
  --gold-file 333_20260206_dense_top5_20260305.json \
  --output-file runs/qwen_paragraph_gold.json \
  --extraction-prompt-file acl_verbatim/prompts/extraction_paragraph.txt
```

The endpoint is configured via:

- `OPENAI_API_BASE`
- `OPENAI_API_KEY`
- `OPENAI_MODEL`

`evaluate_extractor.py` uses `LLMSpanExtractor` in `batch` mode, which means one request per
query group, with multiple retrieved chunks packaged into the same prompt.

## Token Classifier Evaluation

Evaluate a trained token-classification model on the gold benchmark with:

```bash
python acl_verbatim/span_training/evaluate_token_cls.py \
  --gold-file 333_20260206_dense_top5_20260305.json \
  --model-dir KRLabsOrg/acl-verbatim-modernbert \
  --threshold 0.2 \
  --min-span-chars 10 \
  --merge-gap-chars 20 \
  --pred-file runs/eval/acl-verbatim-modernbert.gold_preds.jsonl \
  --output-file runs/eval/acl-verbatim-modernbert.gold_eval.json
```

`--threshold` switches from argmax decoding to a positive-class probability
cutoff (lower = more recall). `--min-span-chars` drops tiny noise spans and
`--merge-gap-chars` joins adjacent predictions — together they clean up the
token-level fragmentation that hurts span IoU. The defaults shown reproduce
the headline numbers in the model card.

You can also evaluate against an HF dataset split directly:

```bash
python acl_verbatim/span_training/evaluate_token_cls.py \
  --hf-dataset KRLabsOrg/acl-verbatim-spans \
  --hf-config canonical \
  --gold-split test \
  --model-dir KRLabsOrg/acl-verbatim-modernbert \
  --threshold 0.2 --min-span-chars 10 --merge-gap-chars 20 \
  --output-file runs/eval/acl-verbatim-modernbert.gold_eval.json
```

This uses the same span metrics as the LLM teacher evaluation, so comparisons are directly
comparable.

## Comparing Multiple Runs

Use `compare_span_runs.py` to print one table for multiple systems:

```bash
python acl_verbatim/eval/compare_span_runs.py \
  --gold-file 333_20260206_dense_top5_20260305.json \
  --run qwen=runs/qwen_paragraph_gold.json \
  --run zilliz=runs/zilliz_gold.jsonl \
  --run modernbert=runs/eval/modernbert_qwen_silver_binary.gold_eval.json
```

Each `--run` path may be either:

- a raw predictions JSONL file
- or a detailed evaluation JSON containing a `summary` object
