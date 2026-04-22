---
license: cc-by-4.0
task_categories:
  - question-answering
  - token-classification
  - text-retrieval
language:
  - en
pretty_name: ACL-Verbatim Span Dataset
tags:
  - acl-anthology
  - extractive-qa
  - evidence-selection
  - semantic-highlighting
  - silver-labels
---

# ACL-Verbatim Span Dataset

This document describes the intended Hugging Face release layout for the ACL-Verbatim span
dataset, planned under `KRLabsOrg/acl-verbatim-spans`.

The dataset is designed for training and evaluating query-conditioned extractive evidence
selection over ACL Anthology chunks. It combines:

- a small manually annotated gold benchmark for evaluation
- a larger silver training set generated from synthetic queries, Milvus retrieval, and batched
  LLM span annotation

## Release Status

The gold benchmark currently lives in this repository as:

- `333_20260206_dense_top5_20260305.json`

Silver datasets are generated locally by the pipeline documented in the repository README and are
not yet published as a stable Hub dataset release.

## Intended Configs

### `canonical`

One row per `(question, chunk)` pair with human- or teacher-provided spans. This is the config
meant for analysis and general downstream reuse.

Expected fields:

| field | type | notes |
|---|---|---|
| `question` | string | Query / question text |
| `paper_id` | string | ACL Anthology identifier |
| `chunk_index` | int | Chunk number within the paper |
| `chunk` | string | Raw chunk text |
| `label` | int | `1` if answer-bearing, `0` otherwise |
| `answerable` | bool | Teacher answerability decision |
| `spans` | list[struct] | `{start, end, text}` evidence spans |
| `source` | string | `gold` or `retrieved` |
| `retrieval_rank` | int or null | Rank among retrieved candidates |
| `gold_paper` | string | Source paper for the synthetic benchmark query |
| `gold_chunk` | int | Source chunk for the synthetic benchmark query |
| `predicted_texts` | list[string] | Raw teacher outputs before alignment |
| `latency_s` | float | Teacher latency metadata |
| `err` | string or null | Teacher/extraction error, if any |

### `encoder`

Token-classification-ready rows for encoder training. These are derived from `canonical` by
tokenization and windowing with a specific tokenizer.

Expected fields:

| field | type | notes |
|---|---|---|
| `input_ids` | list[int] | Token ids |
| `attention_mask` | list[int] | Attention mask |
| `labels` | list[int] | Binary or BIO token labels |

### `generative`

Optional question-conditioned generative training rows. This config is intended for future
sequence-to-sequence or instruction-tuned extractors and is not yet the main supported path.

## Annotation Convention

The benchmark uses paragraph-scale highlighting rather than minimal SQuAD-style spans. Relevant
tables and captions are considered valid evidence. Bibliography/reference sections are generally
out of scope as positives.

## Gold Benchmark Summary

Current local benchmark:

| | |
|---|---|
| Queries | 20 |
| Retrieved chunks per query | 5 |
| Total rows | 100 |
| Relevant rows | 47 |
| Gold spans | 78 |

## Silver Training Data

The recommended silver training variant currently used in this repository is the
caption-preserving `caption_ok` filter mode:

- keep table / figure / caption evidence
- keep all negatives
- optionally cap positive retrieval rank
- drop bibliography/reference positives and clearly pathological spans

## Intended Uses

- Train query-conditioned token classifiers for semantic highlighting
- Compare LLM teachers, token models, sentence-level compressors, and semantic highlighters under
  one span-scoring harness
- Study boundary conventions for evidence extraction in scientific text

## Limitations

- The gold benchmark is small and single-annotator
- Silver labels inherit retrieval noise and teacher noise
- Tables and captions are represented through markdown/caption text, not full table structure
- The benchmark reflects a paragraph-oriented annotation convention rather than a strict minimal
  answer-span convention

## Source Pipeline

The end-to-end generation pipeline is documented in the repository README. The key stages are:

1. sample papers from ACL metadata
2. chunk markdown papers and generate synthetic questions
3. rewrite questions into retrieval queries
4. retrieve top-`k` chunks from Milvus
5. annotate spans with a batched LLM extractor
6. filter silver rows and split by query
7. prepare token-classification windows for encoder training
