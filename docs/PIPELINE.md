# Pipeline

This document covers the current end-to-end ACL-Verbatim workflow:

1. build or download the corpus
2. index markdown into Milvus
3. generate synthetic questions and retrieval queries
4. retrieve candidate chunks
5. annotate silver spans with a batched LLM extractor
6. filter silver rows
7. prepare token-classification training data
8. train and evaluate a token classifier

## Prerequisites

Install the base package:

```bash
pip install -e .
```

For the silver-labeling and training pipeline:

```bash
pip install -e ".[training]"
```

For Hugging Face dataset tooling and semantic-highlighting baselines:

```bash
pip install -e ".[hf]"
```

## Recommended Data Path

The simplest way to obtain the corpus is to use the published HF dataset:

- [`KRLabsOrg/acl-anthology-md`](https://huggingface.co/datasets/KRLabsOrg/acl-anthology-md)

Most operational scripts in this repository still expect local files. To materialize those from the
HF dataset:

```bash
python scripts/export_hf_corpus.py \
  --output-metadata-file paper_data.json \
  --output-md-dir acl_md
```

This gives you the local layout expected by indexing and QA-generation scripts:

- `paper_data.json` (JSONL metadata)
- `acl_md/*.md` (markdown fulltext)

## Metadata Extraction

To extract an up-to-date metadata snapshot from a local clone of
[`acl-org/acl-anthology`](https://github.com/acl-org/acl-anthology):

```bash
python scripts/get_anthology_metadata.py \
  --anthology-path /path/to/acl-anthology \
  --output-file paper_data.json
```

The repository expects `paper_data.json` to be a JSONL file.

## PDF to Markdown

If you need to rebuild the markdown corpus locally:

```bash
python scripts/preprocess_acl.py \
  --input-dir ../acl-anthology/build/anthology-files/pdf \
  --output-dir acl_md \
  --metadata-file paper_data.json \
  --doc-batch-size 512 \
  --page-batch-size 1024
```

For most work you can use the published markdown corpus instead of regenerating it:
[`KRLabsOrg/acl-anthology-md`](https://huggingface.co/datasets/KRLabsOrg/acl-anthology-md)

## Indexing

Index into a local Milvus DB:

```bash
python scripts/index_acl.py \
  --input-dir PATH_TO_MARKDOWN_DATA \
  --index-file acl.db \
  --metadata-file paper_data.json \
  --collection-name acl \
  --device cpu
```

Index into a cloud or server Milvus instance:

```bash
python scripts/index_acl.py \
  --input-dir acl_md \
  --metadata-file paper_data.json \
  --collection-name acl \
  --device cuda \
  --use-cloud \
  --cloud-uri http://localhost:19530
```

## Retrieval

Interactive retrieval:

```bash
python acl_verbatim/eval/test_index.py \
  --collection-name acl \
  --device cpu \
  --use-cloud \
  --cloud-uri CLOUD_URI
```

Batch retrieval only:

```bash
python acl_verbatim/eval/test_index.py \
  --collection-name acl \
  --device cpu \
  --use-cloud \
  --cloud-uri CLOUD_URI \
  --milvus-token MILVUS_TOKEN \
  -r \
  -k 5 \
  --questions-dir QUERIES_DIR \
  --query-field query \
  --output-file runs/search_results_top5_hybrid.jsonl \
  -s hybrid
```

Current code defaults to equal hybrid weights (`0.5` dense, `0.5` full text).

## Synthetic Query Generation

The original synthetic benchmark flow is:

```bash
python acl_verbatim/qa_generation/sample_papers.py \
  --input-file paper_data.json \
  --output-file SAMPLE_PAPERS_FILE \
  --n NO_OF_PAPERS_TO_SAMPLE \
  --seed RANDOM_SEED

python acl_verbatim/qa_generation/chunk_and_classify.py \
  --input-dir ACL_MD_PATH \
  --output-dir CHUNKS_DIR \
  --papers-file SAMPLE_PAPERS_FILE \
  --n 1

python acl_verbatim/qa_generation/gen_qa.py \
  --input-dir CHUNKS_DIR \
  --output-dir QUESTIONS_PATH

python acl_verbatim/qa_generation/question_to_query.py \
  --input-dir QUESTIONS_PATH \
  --output-dir QUERIES_PATH
```

The query files created in the last step are also the inputs for retrieval and silver-label
generation.

## Silver Label Generation

Set the environment first:

```bash
export OPENAI_API_BASE=http://127.0.0.1:8000/v1
export OPENAI_API_KEY=dummy
export OPENAI_MODEL='Qwen/Qwen3.6-35B-A3B'

export VERBATIM_COLLECTION_NAME=acl
export MILVUS_URI='https://YOUR-ZILLIZ-OR-MILVUS-ENDPOINT'
export MILVUS_API_KEY='YOUR_MILVUS_API_KEY'

RUN_DIR=runs/silver_qwen_700
N_PAPERS=700
SEED=1337
mkdir -p "$RUN_DIR/chunks" "$RUN_DIR/questions" "$RUN_DIR/queries"
```

Generate the sampled questions and retrieval queries:

```bash
python acl_verbatim/qa_generation/sample_papers.py \
  --input-file paper_data.json \
  --output-file "$RUN_DIR/random_papers_${N_PAPERS}.json" \
  --n "$N_PAPERS" \
  --seed "$SEED"

python acl_verbatim/qa_generation/chunk_and_classify.py \
  --input-dir acl_md \
  --output-dir "$RUN_DIR/chunks" \
  --papers-file "$RUN_DIR/random_papers_${N_PAPERS}.json" \
  --n 1

python acl_verbatim/qa_generation/gen_qa.py \
  --input-dir "$RUN_DIR/chunks" \
  --output-dir "$RUN_DIR/questions"

python acl_verbatim/qa_generation/question_to_query.py \
  --input-dir "$RUN_DIR/questions" \
  --output-dir "$RUN_DIR/queries"
```

Retrieve top-5 chunks:

```bash
python acl_verbatim/eval/test_index.py \
  --collection-name "$VERBATIM_COLLECTION_NAME" \
  --use-cloud \
  --cloud-uri "$MILVUS_URI" \
  --milvus-token "$MILVUS_API_KEY" \
  --device cpu \
  -r \
  -k 5 \
  --questions-dir "$RUN_DIR/queries" \
  --query-field query \
  --output-file "$RUN_DIR/search_results_top5_hybrid.jsonl" \
  -s hybrid
```

Annotate silver spans:

```bash
python acl_verbatim/qa_generation/annotate_spans_from_results_batched.py \
  --results-file "$RUN_DIR/search_results_top5_hybrid.jsonl" \
  --output-file "$RUN_DIR/span_pairs_top5_qwen_paragraph.jsonl" \
  --collection-name "$VERBATIM_COLLECTION_NAME" \
  --milvus-uri "$MILVUS_URI" \
  --milvus-token "$MILVUS_API_KEY" \
  --max-results-per-query 5 \
  --batch-size 5 \
  --workers 6 \
  --resume \
  --flush \
  --extraction-prompt-file acl_verbatim/prompts/extraction_paragraph.txt
```

This is the canonical silver-labeling entrypoint. It supports both sequential mode (`--workers 1`)
and concurrent query-group processing (`--workers > 1`) while still batching multiple retrieved
chunks inside each LLM request.

## Filtering

Recommended caption-preserving split:

```bash
python acl_verbatim/qa_generation/filter_silver_dataset.py \
  --input-file "$RUN_DIR/span_pairs_top5_qwen_paragraph.jsonl" \
  --output-dir "${RUN_DIR}_caption_ok" \
  --max-positive-rank 3
```

If you want a stricter ablation that also drops caption/table/figure-like positives, add:

```bash
--drop-caption-like
```

## Token-Classification Data Prep

```bash
python acl_verbatim/span_training/prepare_token_cls_dataset.py \
  --input-file "${RUN_DIR}_caption_ok/splits/train.jsonl" \
  --output-file "${RUN_DIR}_caption_ok/token_cls/train.modernbert.binary.jsonl" \
  --tokenizer answerdotai/ModernBERT-base \
  --label-scheme binary \
  --drop-unlabeled-positives

python acl_verbatim/span_training/prepare_token_cls_dataset.py \
  --input-file "${RUN_DIR}_caption_ok/splits/dev.jsonl" \
  --output-file "${RUN_DIR}_caption_ok/token_cls/dev.modernbert.binary.jsonl" \
  --tokenizer answerdotai/ModernBERT-base \
  --label-scheme binary \
  --drop-unlabeled-positives
```

## Training

```bash
python acl_verbatim/span_training/train_token_cls.py \
  --train-file "${RUN_DIR}_caption_ok/token_cls/train.modernbert.binary.jsonl" \
  --eval-file "${RUN_DIR}_caption_ok/token_cls/dev.modernbert.binary.jsonl" \
  --model answerdotai/ModernBERT-base \
  --output-dir runs/models/modernbert_qwen_silver_binary \
  --batch-size 8 \
  --lr 5e-5 \
  --epochs 3 \
  --label-scheme binary
```

The resulting model directory is a standard Hugging Face
`AutoModelForTokenClassification` bundle.

## Evaluation

Evaluate the trained encoder on the gold benchmark:

```bash
python acl_verbatim/span_training/evaluate_token_cls.py \
  --gold-file 333_20260206_dense_top5_20260305.json \
  --model-dir runs/models/modernbert_qwen_silver_binary \
  --pred-file runs/eval/modernbert_qwen_silver_binary.gold_preds.jsonl \
  --output-file runs/eval/modernbert_qwen_silver_binary.gold_eval.json
```

For additional evaluation tooling, see:

- [`acl_verbatim/eval/README.md`](../acl_verbatim/eval/README.md)

## Publishing The Span Dataset

To publish the current gold benchmark plus caption-preserving silver splits to Hugging Face:

```bash
python scripts/build_spans_dataset.py \
  --silver-train runs/silver_qwen_2000_caption_ok/splits/train.jsonl \
  --silver-dev runs/silver_qwen_2000_caption_ok/splits/dev.jsonl \
  --gold-file 333_20260206_dense_top5_20260305.json \
  --encoder-train runs/silver_qwen_2000_caption_ok/token_cls/train.modernbert.binary.jsonl \
  --encoder-dev runs/silver_qwen_2000_caption_ok/token_cls/dev.modernbert.binary.jsonl \
  --repo-id KRLabsOrg/acl-verbatim-spans
```

This produces:

- `canonical/train`: silver training rows
- `canonical/validation`: silver dev rows
- `canonical/test`: gold benchmark rows
- `encoder/train`: tokenized train rows
- `encoder/validation`: tokenized dev rows

## Smoke Tests

The repository includes a small smoke-test suite for core supported paths:

```bash
python -m unittest discover -s tests -p 'test_*.py'
```

## Optional Sanity Check

```bash
python - <<'PY'
import json
from pathlib import Path
run_dir = Path("runs/silver_qwen_700_caption_ok")
summary = json.loads((run_dir / "filter_summary.json").read_text())
print(json.dumps(summary, indent=2))
print("train_token_rows =", sum(1 for _ in open(run_dir / "token_cls/train.modernbert.binary.jsonl")))
print("dev_token_rows =", sum(1 for _ in open(run_dir / "token_cls/dev.modernbert.binary.jsonl")))
PY
```
