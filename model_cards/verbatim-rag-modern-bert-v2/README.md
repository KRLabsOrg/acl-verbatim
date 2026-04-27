---
license: apache-2.0
library_name: transformers
pipeline_tag: token-classification
base_model: Alibaba-NLP/gte-reranker-modernbert-base
language:
  - en
datasets:
  - KRLabsOrg/verbatim-spans
tags:
  - semantic-highlighting
  - extractive-qa
  - evidence-selection
  - modernbert
  - verbatim-rag
---

# Verbatim-RAG Extractor

[![ChiliGround Logo](https://github.com/KRLabsOrg/verbatim-rag/raw/main/assets/chiliground.png?raw=true)](https://github.com/KRLabsOrg/verbatim-rag)
_Chill, I Ground! 🌶️_

**Model Name:** verbatim-rag-modern-bert-v2
**Organization:** KRLabsOrg
**Github:** [https://github.com/KRLabsOrg/verbatim-rag](https://github.com/KRLabsOrg/verbatim-rag)

## Overview

The Verbatim-RAG Extractor is a query-conditioned token classifier that
highlights the verbatim spans of a passage that answer a question. It is the
encoder companion to [VerbatimRAG](https://github.com/KRLabsOrg/verbatim-rag)
and the successor to
[`verbatim-rag-modern-bert-v1`](https://huggingface.co/KRLabsOrg/verbatim-rag-modern-bert-v1).
Built on
[`Alibaba-NLP/gte-reranker-modernbert-base`](https://huggingface.co/Alibaba-NLP/gte-reranker-modernbert-base),
which provides the long ModernBERT context (up to 8192 tokens) and a
query-conditioned reranking prior on top of which span extraction is fine-tuned.

Most public evidence extractors (Provence, Zilliz Semantic-Highlight,
MultiSpanQA-trained models) are trained on Wikipedia-style prose QA only.
This model is trained on
[`KRLabsOrg/verbatim-spans`](https://huggingface.co/datasets/KRLabsOrg/verbatim-spans),
which adds financial tables, legal contracts, medical literature, product
manuals, and — uniquely among public extractors — coding-agent tool output
(`pytest` failures, `git diff` hunks, stack traces). The result is a single
150M-parameter encoder usable across the content shapes a real RAG or agent
pipeline tends to retrieve, not just article paragraphs.

For an ACL-Anthology-specialized variant, see
[`KRLabsOrg/acl-verbatim-modernbert`](https://huggingface.co/KRLabsOrg/acl-verbatim-modernbert).

## Model Details

* **Architecture:** ModernBERT (gte-reranker-modernbert-base) with 8192-token context
* **Task:** Token classification — binary evidence labels mapped to character spans
* **Training Dataset:** [`KRLabsOrg/verbatim-spans`](https://huggingface.co/datasets/KRLabsOrg/verbatim-spans) (multi-domain)
* **Language:** English
* **Parameters:** 150M

### Training data composition

| content shape | source |
|---|---|
| scientific paragraphs with citations | ACL silver |
| Wikipedia / general QA, multi-hop | RAGBench (HotpotQA, MS MARCO, ExpertQA, ...) |
| financial tables | RAGBench (TAT-QA, FinQA) |
| medical literature | RAGBench (PubMedQA, CovidQA) |
| legal contracts | RAGBench (CUAD) |
| product manuals | RAGBench (eManual, TechQA) |
| code, tool output, stack traces, logs | Squeez (SWE-bench tool outputs) |

## How It Works

A `(question, context)` pair is encoded as a single sequence; the model
predicts a per-token positive-class probability over the context tokens. Above
a threshold, contiguous positive runs are merged into character spans, with
post-processing (`min_span_chars`, `merge_gap_chars`) that removes
fragmentation artifacts. Long contexts are handled with sliding windows of
`max_length` tokens stepped by `doc_stride`, and spans are merged across
windows.

## Usage

```python
from transformers import AutoModel

model = AutoModel.from_pretrained(
    "KRLabsOrg/verbatim-rag-modern-bert-v2",
    trust_remote_code=True,
)

result = model.process(
    question="What is ModernBERT?",
    context=(
        "ModernBERT is a long-context encoder for NLP. "
        "It supports sequences up to 8192 tokens. "
        "Unlike earlier BERT variants, it uses rotary position embeddings."
    ),
    threshold=0.2,
)

for span in result["spans"]:
    print(f"[{span['score']:.2f}] {span['text']}")
```

`.process()` accepts: `question`, `context`, `threshold` (default `0.2`),
`max_length` (default `8192`), `doc_stride` (default `256`), `min_span_chars`
(default `30`), `merge_gap_chars` (default `20`), `return_sentence_metrics`
(default `False`). For short-answer benchmarks (file paths, table cells,
numbers), `threshold=0.1` and `min_span_chars=10` is the recall-tuned config
documented in Performance below.

The return shape is `{"spans": [{"start": int, "end": int, "text": str,
"score": float}, ...]}`, with `"sentences"` added when
`return_sentence_metrics=True`. Spans are character offsets into the input
`context` and are merged across sliding windows.

## Performance

Evaluated on the test splits of the three training sources, with the same
harness applied to baseline span extractors
([Provence](https://huggingface.co/naver/provence-reranker-debertav3-v1),
[Zilliz Semantic-Highlight](https://huggingface.co/zilliz/semantic-highlight-bilingual-v1))
and to the ACL-specialized variant. Word-level micro F1.

| domain | rows | this model | acl-specialized | provence | zilliz |
|---|---:|---:|---:|---:|---:|
| ACL Anthology gold | 47 | 0.495 | **0.562** | 0.480 | 0.322 |
| RAGBench (12 configs) | ~17k | **0.759** | 0.598 | 0.615 | 0.511 |
| Squeez tool-output | ~1k | **0.769** | 0.493 | 0.491 | 0.418 |

The ACL-specialized model wins on its home turf; the multi-domain extractor
wins everywhere else by 14–28 points and is 6.7 points behind on ACL. With
the recall-tuned config (`threshold=0.1`, `min_span_chars=10`), word-F1 on
ACL gold rises from 0.495 to 0.523, narrowing the gap to ~4 points.

### Squeez tool-output benchmark

Scored through the Squeez line-level harness (618 samples). Squeez and the
generative baselines emit text in line units, so they get full credit on
strict Span F1 by construction; the Verbatim-RAG Extractor emits character
spans that are mapped to lines, so Fuzzy F1 (≥50% character overlap) is the
metric where both model families are evaluated on the same footing.

| model | params | Fuzzy F1 | Span F1 | Partial Overlap | Empty Acc | Compression |
|---|---:|---:|---:|---:|---:|---:|
| Squeez-2B (fine-tuned) | 2B | 0.804 | 0.790 | 0.919 | 0.968 | 0.915 |
| Qwen 3.5 35B A3B (zero-shot) | 35B | 0.725 | 0.700 | 0.835 | 0.916 | 0.918 |
| Kimi K2 (zero-shot) | huge | 0.683 | 0.534 | 0.746 | 0.924 | 0.943 |
| **Verbatim-RAG Extractor** | **150M** | 0.646 | 0.515 | 0.820 | 0.898 | 0.917 |
| Qwen 3.5 2B (zero-shot) | 2B | 0.548 | 0.408 | 0.768 | 0.916 | 0.820 |

A 150M-parameter encoder lands within 4 Fuzzy-F1 points of zero-shot Kimi K2
and 16 of fine-tuned Squeez-2B. Compression and empty-prediction accuracy
match the larger generative models.

## Citing

```
TODO
```
