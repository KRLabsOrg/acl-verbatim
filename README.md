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
- Local gold benchmark:
  [333_20260206_dense_top5_20260305.json](333_20260206_dense_top5_20260305.json)
  - 20 queries
  - 100 retrieved chunks
  - 47 relevant rows
  - 78 gold spans
- Planned span dataset release:
  `KRLabsOrg/acl-verbatim-spans`
  - draft card: [dataset_cards/acl-verbatim-spans/README.md](dataset_cards/acl-verbatim-spans/README.md)

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
- Planned span dataset card:
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

## Local Data Preparation

Extract fresh metadata from a local clone of `acl-org/acl-anthology`:

```bash
python scripts/get_anthology_metadata.py \
  --anthology-path /path/to/acl-anthology \
  --output-file paper_data.json
```

Convert PDFs to markdown locally:

```bash
python scripts/preprocess_acl.py \
  --input-dir ../acl-anthology/build/anthology-files/pdf \
  --output-dir acl_md \
  --metadata-file paper_data.json \
  --doc-batch-size 512 \
  --page-batch-size 1024
```

For most work, the published Hugging Face corpus is easier than rebuilding the markdown locally.

## Notes

- `paper_data.json` is JSONL content stored under a `.json` filename for historical reasons. The
  scripts in this repository now consistently expect that file shape.
- `runs/` is intentionally ignored. Large silver datasets, checkpoints, and eval outputs should
  live in Hugging Face or external storage, not Git.
- Current code defaults to equal hybrid retrieval weights (`0.5` dense, `0.5` full text). Older
  benchmark tables using different weights are kept in [EVAL.md](EVAL.md) and marked as historical.

## License

Apache 2.0. See [LICENSE](LICENSE).

## Citation

An ACL/NAACL-style paper describing ACL-Verbatim is in preparation. Until then, please cite:

- the ACL Anthology for the source corpus
- VerbatimRAG for the underlying retrieval and extraction framework

## Acknowledgements

This project relies on the ACL Anthology, the maintainers of
[`acl-org/acl-anthology`](https://github.com/acl-org/acl-anthology), and the maintainers of
[VerbatimRAG](https://github.com/KRLabsOrg/verbatim-rag).
