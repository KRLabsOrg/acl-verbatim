---
license: apache-2.0
task_categories:
  - question-answering
  - token-classification
  - text-retrieval
language:
  - en
pretty_name: Verbatim Spans
tags:
  - extractive-qa
  - evidence-selection
  - semantic-highlighting
  - silver-labels
  - multi-domain
configs:
  - config_name: canonical
    data_files:
      - split: train
        path: canonical/train-*
      - split: validation
        path: canonical/validation-*
  - config_name: encoder
    data_files:
      - split: train
        path: encoder/train-*
      - split: validation
        path: encoder/validation-*
---

# Verbatim Spans

A multi-domain training dataset for **query-conditioned extractive evidence
selection**. Given a question and a passage, the task is to highlight the
verbatim substrings of the passage that support the answer.

Combines three sources covering distinct domains and annotation conventions:

| source | domain | convention | annotator | rows (train / val) |
|---|---|---|---|---|
| ACL silver (this project) | NLP research papers | paragraph-scale | Qwen 3.6 35B (paragraph prompt) | 20,916 / 2,319 |
| [RAGBench](https://huggingface.co/datasets/galileo-ai/ragbench) (12 configs, capped) | finance / medical / legal / general QA | sentence-scale | GPT-4o | 101,550 / 15,276 |
| [Squeez](https://huggingface.co/datasets/KRLabsOrg/tool-output-extraction-swebench-gliner) | code / SWE-bench tool outputs | code block / line range | GLiNER-format (this project) | 51,917 / 2,579 |
| **total** | | | | **174,383 / 20,174** |

The dataset is designed for training a **generic span-highlighter encoder** —
the intended model is a ModernBERT token classifier. For the specialized
ACL-only benchmark see
[`KRLabsOrg/acl-verbatim-spans`](https://huggingface.co/datasets/KRLabsOrg/acl-verbatim-spans).

## Configs

### `canonical`

One row per `(question, chunk)` pair, with raw text. Use this config if you
want to train with your own tokenizer or inspect rows.

Fields:

| field | type | notes |
|---|---|---|
| `source_dataset` | string | `"acl"`, `"ragbench"`, or `"squeez"` |
| `question` | string | Query / question text |
| `paper_id` | string | Source-namespaced identifier |
| `chunk_index` | int | Chunk number within the source document |
| `chunk` | string | Raw chunk text |
| `label` | int | `1` if answer-bearing, `0` otherwise |
| `answerable` | bool | |
| `spans` | list[struct] | `{start, end, text}` evidence spans |
| `source` | string | Provenance within the source pipeline |
| `retrieval_rank` | int or null | Rank among retrieved candidates (if applicable) |
| `gold_paper` | string | Source document id |
| `gold_chunk` | int | Source chunk index |
| `predicted_texts` | list[string] | Raw teacher outputs before alignment (ACL silver only) |
| `latency_s` | float | Teacher latency (ACL silver only) |
| `err` | string | Teacher error, if any (ACL silver only) |

### `encoder`

Pretokenized, ready for direct training with
[`Alibaba-NLP/gte-reranker-modernbert-base`](https://huggingface.co/Alibaba-NLP/gte-reranker-modernbert-base)
or any compatible ModernBERT checkpoint (the tokenizer is shared across the
ModernBERT family).

Fields: `input_ids`, `attention_mask`, `labels`.

Pretokenization settings:

| parameter | value |
|---|---|
| tokenizer | `Alibaba-NLP/gte-reranker-modernbert-base` |
| max_length | 8192 |
| doc_stride | 256 |
| truncation | `only_second` (question kept whole, chunk windowed) |
| label_scheme | binary (`0` = outside, `1` = evidence) |
| drop_unlabeled_positives | true |

If you want to train with a different tokenizer, rebuild from `canonical`:

```bash
python acl_verbatim/span_training/prepare_token_cls_dataset.py \
  --input-file <canonical_train.jsonl> \
  --output-file train.my_tokenizer.binary.jsonl \
  --tokenizer <your-tokenizer> \
  --label-scheme binary \
  --drop-unlabeled-positives
```

## Composition details

**RAGBench cap:** 15,000 rows per config for train, 2,000 per config for
validation, random seed 1337. This balances the 12 RAGBench configs so that
high-volume configs (tatqa, pubmedqa, finqa) do not dominate the mix. Without
capping, tatqa + pubmedqa alone would be ~70% of the training data.

**No RAGBench test split included.** The RAGBench test split is reserved for
downstream evaluation.

**Squeez:** all train and validation rows are included. The negative/positive
split (~2:1) is preserved as-is; negatives are important signal for teaching
the model when not to fire.

**ACL silver:** taken from the caption-preserving split released in
[`KRLabsOrg/acl-verbatim-spans`](https://huggingface.co/datasets/KRLabsOrg/acl-verbatim-spans).

## Intended use

Training a generic query-conditioned token classifier for evidence highlighting
across diverse RAG / extractive-QA use cases. The associated model is
[`KRLabsOrg/verbatim-rag-modern-bert-v2`](https://huggingface.co/KRLabsOrg/verbatim-rag-modern-bert-v2),
fine-tuned from `Alibaba-NLP/gte-reranker-modernbert-base` on the `encoder`
config above.

Evaluation against the human-annotated ACL gold benchmark is handled by the
sibling dataset
[`KRLabsOrg/acl-verbatim-spans`](https://huggingface.co/datasets/KRLabsOrg/acl-verbatim-spans)
(`canonical/test` split). For the full per-domain evaluation methodology
(RAGBench, Squeez, MultiSpanQA, plus baseline comparisons against Provence and
Zilliz), see
[`docs/GENERIC_EVAL.md`](https://github.com/KRLabsOrg/acl-verbatim/blob/main/docs/GENERIC_EVAL.md)
in the acl-verbatim repo.

## Limitations

- All labels are LLM-produced (Qwen for ACL, GPT-4o for RAGBench, silver-style
  GLiNER output for Squeez) — not strictly human-annotated.
- Evidence conventions vary across sources: a model trained on this mix will
  learn an average of sentence / paragraph / code-block scales rather than a
  single consistent convention.
- RAGBench domains are English only; Squeez is code + tool output; ACL is
  scientific prose. Transfer to other languages or domains (e.g. bilingual
  use-cases or spoken dialogue) is not evaluated.
- The GLiNER-style Squeez format uses a single entity type `RELEVANT`; span
  boundaries were produced by a GLiNER-trained model rather than human
  annotators.

## Licensing and attribution

Apache 2.0. All three source datasets are Apache 2.0:

- ACL silver: KRLabs Org — derived from the ACL Anthology corpus
  ([`KRLabsOrg/acl-anthology-md`](https://huggingface.co/datasets/KRLabsOrg/acl-anthology-md))
- RAGBench: Galileo Labs —
  [`galileo-ai/ragbench`](https://huggingface.co/datasets/galileo-ai/ragbench)
- Squeez: KRLabs Org —
  [`KRLabsOrg/tool-output-extraction-swebench-gliner`](https://huggingface.co/datasets/KRLabsOrg/tool-output-extraction-swebench-gliner)

## Reproducibility

All preparation scripts live in
[`KRLabsOrg/acl-verbatim`](https://github.com/KRLabsOrg/acl-verbatim):

- `scripts/experiments/prepare_ragbench_spans.py` — RAGBench → canonical spans
- `scripts/experiments/prepare_squeez_spans.py` — Squeez GLiNER format → canonical spans
- `scripts/publish/build_generic_spans_dataset.py` — caps + merges + pushes to HF
  (random seed 1337 for the RAGBench cap)

## Citation

TODO
