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
configs:
  - config_name: canonical
    data_files:
      - split: train
        path: canonical/train-*
      - split: validation
        path: canonical/validation-*
      - split: test
        path: canonical/test-*
  - config_name: encoder
    data_files:
      - split: train
        path: encoder/train-*
      - split: validation
        path: encoder/validation-*
---

# ACL-Verbatim Span Dataset

[`KRLabsOrg/acl-verbatim-spans`](https://huggingface.co/datasets/KRLabsOrg/acl-verbatim-spans)
is a dataset for **query-conditioned extractive evidence selection** over papers from the
[ACL Anthology](https://aclanthology.org/).

The release combines:

- a **gold test benchmark** with manual span annotations
- a larger **silver training set** produced from synthetic questions, retrieval, and LLM-based
  span annotation
- an **encoder-ready config** for training token-classification models directly

The underlying document collection is
[`KRLabsOrg/acl-anthology-md`](https://huggingface.co/datasets/KRLabsOrg/acl-anthology-md).

## What This Dataset Is For

This dataset is intended for systems that, given a **question** and a **retrieved paper chunk**,
must identify the supporting evidence **verbatim** in the chunk.

Typical uses include:

- training token classifiers for semantic highlighting
- evaluating span extractors and evidence selectors
- comparing LLM teachers, token-level students, and sentence-selection baselines
- studying paragraph-scale evidence extraction in scientific text

## Configs

### `canonical`

One row per `(question, chunk)` pair.

- `train` and `validation` are **silver** supervision
- `test` is the **manual gold benchmark**

This is the main config for analysis, evaluation, and downstream reuse.

Current split sizes:

| split | rows | notes |
|---|---:|---|
| `train` | 20,916 | silver |
| `validation` | 2,319 | silver dev |
| `test` | 100 | gold benchmark — 20 queries × 5 retrieved chunks; **47** are marked relevant and carry 78 manual gold spans |

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
| `source` | string | Where the candidate chunk came from in data generation: `gold` = the original chunk used to create the synthetic question, `retrieved` = a chunk retrieved for that question |
| `retrieval_rank` | int or null | Rank among retrieved candidates |
| `gold_paper` | string | Paper from which the synthetic question was derived |
| `gold_chunk` | int | Chunk from which the synthetic question was derived |
| `predicted_texts` | list[string] | Raw teacher outputs before alignment |
| `latency_s` | float | Teacher latency metadata |
| `err` | string or null | Teacher/extraction error, if any |

### `encoder`

Token-classification-ready rows derived from the silver `canonical` data by tokenization and
windowing.

This config is intended for direct encoder training with Hugging Face `transformers`.

Current split sizes:

| split | rows |
|---|---:|
| `train` | 21,099 |
| `validation` | 2,343 |

Expected fields:

| field | type | notes |
|---|---|---|
| `input_ids` | list[int] | Token ids |
| `attention_mask` | list[int] | Attention mask |
| `labels` | list[int] | Binary or BIO token labels |

### How the `encoder` split was built

The `encoder` split was pretokenized with:

| parameter | value |
|---|---|
| tokenizer | `answerdotai/ModernBERT-base` |
| max_length | 8192 |
| doc_stride | 256 |
| truncation | `only_second` (question kept whole, chunk windowed) |
| label_scheme | binary (`0` = outside, `1` = evidence) |
| drop_unlabeled_positives | true |

If you want to train with a different tokenizer or label scheme, rebuild from
the `canonical` config:

```bash
python acl_verbatim/span_training/prepare_token_cls_dataset.py \
  --input-file <canonical_train.jsonl> \
  --output-file train.my_tokenizer.binary.jsonl \
  --tokenizer <your-tokenizer> \
  --label-scheme binary \
  --drop-unlabeled-positives
```

Evaluation always uses `canonical/test` (raw text); the `encoder` config
intentionally does **not** include a test split, because span-level scoring
needs to map token predictions back to character offsets in the original chunk,
which requires retokenizing at inference time.

## Annotation Convention

The benchmark uses **paragraph-oriented evidence annotation** rather than minimal SQuAD-style
answer spans.

Important consequences:

- broader supporting passages are often preferred over minimal snippets
- tables, figure captions, and other structured evidence are considered valid positives
- bibliography/reference sections are generally out of scope as positive evidence

## Gold Benchmark Summary

The `canonical/test` split is the manually annotated benchmark used for extractor evaluation.

| | |
|---|---|
| Queries | 20 |
| Retrieved chunks per query | 5 |
| Total rows | 100 |
| Relevant rows | 47 |
| Gold spans | 78 |

## Silver Training Data

The silver training data was produced by:

1. sampling papers from the ACL Anthology corpus
2. generating synthetic questions from paper chunks
3. rewriting those questions into retrieval-style queries
4. retrieving top-ranked chunks
5. annotating answer-bearing chunks with an LLM span extractor
6. filtering noisy positives while **preserving table/caption evidence**

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

## How To Load The Dataset

Load the canonical config:

```python
from datasets import load_dataset

ds = load_dataset("KRLabsOrg/acl-verbatim-spans", "canonical")
train = ds["train"]
dev = ds["validation"]
test = ds["test"]
```

Load encoder-ready training rows:

```python
from datasets import load_dataset

encoder = load_dataset("KRLabsOrg/acl-verbatim-spans", "encoder")
train = encoder["train"]
dev = encoder["validation"]
```

## Example: inspect gold vs silver splits

```python
from datasets import load_dataset

canonical = load_dataset("KRLabsOrg/acl-verbatim-spans", "canonical")

silver_train = canonical["train"]
silver_dev = canonical["validation"]
gold_test = canonical["test"]
```

## Related Resources

- Raw ACL Anthology markdown corpus:
  [`KRLabsOrg/acl-anthology-md`](https://huggingface.co/datasets/KRLabsOrg/acl-anthology-md)
- Codebase:
  [`KRLabsOrg/acl-verbatim`](https://github.com/KRLabsOrg/acl-verbatim)

## Citation

TODO
