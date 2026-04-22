# acl-verbatim

Trustworthy question answering on the ACL Anthology using retrieval plus verbatim evidence
extraction.

ACL-Verbatim is a research codebase for:

- indexing ACL Anthology markdown in Milvus
- generating synthetic queries and silver span labels
- evaluating LLM extractors, semantic highlighters, and token classifiers with one shared span scorer
- training a self-contained Hugging Face token-classification model for query-conditioned highlighting

## Datasets

- Corpus: [`KRLabsOrg/acl-anthology-md`](https://huggingface.co/datasets/KRLabsOrg/acl-anthology-md)
  - `metadata` config: bibliographic metadata
  - `fulltext` config: docling-converted markdown
- Span dataset: [`KRLabsOrg/acl-verbatim-spans`](https://huggingface.co/datasets/KRLabsOrg/acl-verbatim-spans)
  - `canonical` config: silver train/dev rows plus gold test rows
  - `encoder` config: token-classification-ready train/dev rows
  - local gold file kept in-repo: [333_20260206_dense_top5_20260305.json](333_20260206_dense_top5_20260305.json)

## Getting The Data

There are two practical ways to get started.

### Option 1: Use The Published Corpus

Use the Hugging Face corpus dataset:

- [`KRLabsOrg/acl-anthology-md`](https://huggingface.co/datasets/KRLabsOrg/acl-anthology-md)
  - `metadata` config: paper metadata
  - `fulltext` config: markdown corpus

This is the easiest way to reproduce retrieval, silver generation, and training without rebuilding
the anthology locally.

If you want to use the existing local-file-based scripts (`scripts/index_acl.py`,
`chunk_and_classify.py`, etc.) with the published corpus, export the HF dataset to the expected
local layout:

```bash
python scripts/export_hf_corpus.py \
  --output-metadata-file paper_data.json \
  --output-md-dir acl_md
```

That materializes:

- `paper_data.json` as JSONL metadata
- `acl_md/*.md` as markdown files keyed by `anthology_id`

The current gold extraction benchmark is the local file:

- [333_20260206_dense_top5_20260305.json](333_20260206_dense_top5_20260305.json)

The same gold and silver span data are also published as:

- [`KRLabsOrg/acl-verbatim-spans`](https://huggingface.co/datasets/KRLabsOrg/acl-verbatim-spans)
  - `canonical/train`: silver training rows
  - `canonical/validation`: silver dev rows
  - `canonical/test`: gold benchmark rows
  - `encoder/train` and `encoder/validation`: tokenized ModernBERT-ready rows

### Option 2: Rebuild The Local Inputs

If you want local metadata and markdown files instead of the published HF corpus:

1. Extract metadata from a local clone of `acl-org/acl-anthology`:

```bash
python scripts/get_anthology_metadata.py \
  --anthology-path /path/to/acl-anthology \
  --output-file paper_data.json
```

2. Convert ACL Anthology PDFs to markdown:

```bash
python scripts/preprocess_acl.py \
  --input-dir ../acl-anthology/build/anthology-files/pdf \
  --output-dir acl_md \
  --metadata-file paper_data.json \
  --doc-batch-size 512 \
  --page-batch-size 1024
```

3. Index that markdown with:

```bash
python scripts/index_acl.py \
  --input-dir acl_md \
  --metadata-file paper_data.json \
  --collection-name acl \
  --device cuda \
  --use-cloud \
  --cloud-uri http://localhost:19530
```

`paper_data.json` is JSONL content stored under a `.json` filename for historical reasons.

## Installation

Base install:

```bash
pip install -e .
```

For silver-label generation and token-classifier training:

```bash
pip install -e ".[training]"
```

For Hugging Face dataset tooling and semantic-highlighting baselines:

```bash
pip install -e ".[hf]"
```

## Quick Start

Index a local markdown corpus:

```bash
python scripts/index_acl.py \
  --input-dir acl_md \
  --metadata-file paper_data.json \
  --collection-name acl \
  --device cuda \
  --use-cloud \
  --cloud-uri http://localhost:19530
```

Run retrieval only:

```bash
python acl_verbatim/eval/test_index.py \
  --collection-name acl \
  --device cpu \
  --use-cloud \
  --cloud-uri CLOUD_URI \
  -r
```

Evaluate a trained token classifier on the gold benchmark:

```bash
python acl_verbatim/span_training/evaluate_token_cls.py \
  --gold-file 333_20260206_dense_top5_20260305.json \
  --model-dir runs/models/modernbert_qwen_silver_binary
```

Or evaluate directly against the published gold split on HF:

```bash
python acl_verbatim/span_training/evaluate_token_cls.py \
  --hf-dataset KRLabsOrg/acl-verbatim-spans \
  --hf-config canonical \
  --gold-split test \
  --model-dir runs/models/modernbert_acl_verbatim_encoder
```

Load the published span dataset:

```python
from datasets import load_dataset

canonical = load_dataset("KRLabsOrg/acl-verbatim-spans", "canonical")
encoder = load_dataset("KRLabsOrg/acl-verbatim-spans", "encoder")
```

Train directly from the published encoder split:

```bash
python acl_verbatim/span_training/train_token_cls.py \
  --hf-dataset KRLabsOrg/acl-verbatim-spans \
  --hf-config encoder \
  --train-split train \
  --eval-split validation \
  --model answerdotai/ModernBERT-base \
  --output-dir runs/models/modernbert_acl_verbatim_encoder \
  --batch-size 8 \
  --lr 5e-5 \
  --epochs 3 \
  --label-scheme binary
```

Run smoke tests:

```bash
python -m unittest discover -s tests -p 'test_*.py'
```

## Main Workflows

- Full silver-data and training pipeline:
  [docs/PIPELINE.md](docs/PIPELINE.md)
- Evaluation commands and supported prediction formats:
  [acl_verbatim/eval/README.md](acl_verbatim/eval/README.md)
- Corpus dataset card:
  [dataset_cards/acl-anthology-md/README.md](dataset_cards/acl-anthology-md/README.md)
- Span dataset card:
  [dataset_cards/acl-verbatim-spans/README.md](dataset_cards/acl-verbatim-spans/README.md)

## Current Supported Pipeline

The current supported path is:

1. sample papers and generate synthetic questions
2. retrieve top-k chunks from Milvus
3. annotate silver spans with `annotate_spans_from_results_batched.py`
4. filter to a caption-preserving `caption_ok` split
5. prepare token-classification training data
6. train a ModernBERT token classifier
7. evaluate the trained model on the local gold benchmark with the shared span metrics

If you want the published rows instead of local files, use:

```python
from datasets import load_dataset

canonical = load_dataset("KRLabsOrg/acl-verbatim-spans", "canonical")
train_rows = canonical["train"]
dev_rows = canonical["validation"]
gold_rows = canonical["test"]
```

The canonical silver-labeling command is:

```bash
python acl_verbatim/qa_generation/annotate_spans_from_results_batched.py \
  --results-file runs/search_results_top5_hybrid.jsonl \
  --output-file runs/span_pairs_top5_qwen_paragraph.jsonl \
  --collection-name acl \
  --milvus-uri MILVUS_URI \
  --milvus-token MILVUS_TOKEN \
  --batch-size 5 \
  --workers 6 \
  --resume \
  --extraction-prompt-file acl_verbatim/prompts/extraction_paragraph.txt
```

## Notes

- `runs/` is intentionally ignored. Large silver datasets, checkpoints, and eval outputs should
  live in Hugging Face or external storage, not Git.
- Current code defaults to equal hybrid retrieval weights (`0.5` dense, `0.5` full text). Older
  benchmark tables using different weights are kept in [EVAL.md](EVAL.md) and marked as historical.

## License

Apache 2.0. See [LICENSE](LICENSE).

## Citation

TODO

## Acknowledgements

TODO
