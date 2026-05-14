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

The goal is a lightweight extractor that can replace many LLM-based evidence
highlighting calls in production RAG systems: local, deterministic, cheap to
serve, and still competitive on span-overlap quality. In our ACL-Verbatim gold
benchmark, the ACL-specialized sibling model is on par with strong LLM
extractors by word-level F1, while this generic multi-domain model beats public
extractive baselines across ACL gold, RAGBench, Squeez, and QASPER.

You can use it as the extraction stage inside VerbatimRAG, or drop it into your
own RAG pipeline after retrieval/reranking to turn retrieved chunks into
grounded evidence spans before displaying them to users or passing them to a
generator.

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

### Use inside VerbatimRAG

```python
from verbatim_rag.core import VerbatimRAG
from verbatim_rag.index import VerbatimIndex
from verbatim_rag.extractors import ModelSpanExtractor
from verbatim_rag.vector_stores import LocalMilvusStore
from verbatim_rag.embedding_providers import SpladeProvider

# v2 is the default ModelSpanExtractor model, but passing it explicitly makes
# the dependency clear.
extractor = ModelSpanExtractor(
    model_path="KRLabsOrg/verbatim-rag-modern-bert-v2",
    threshold=0.2,
    min_span_chars=30,
    merge_gap_chars=20,
    device=None,  # auto-detects cuda, then mps, then cpu
)

sparse_provider = SpladeProvider(
    model_name="opensearch-project/opensearch-neural-sparse-encoding-doc-v2-distill",
    device="cuda",  # use "cpu" if no GPU is available
)

vector_store = LocalMilvusStore(
    db_path="./index.db",
    collection_name="verbatim_rag",
    enable_dense=False,
    enable_sparse=True,
)

# Assumes the index has already been populated with your documents.
index = VerbatimIndex(
    vector_store=vector_store,
    sparse_provider=sparse_provider,
)

rag = VerbatimRAG(
    index=index,
    extractor=extractor,
    k=5,
)

response = rag.query("Main findings of the paper?")
print(response.answer)
```

You can also use the model directly after your own retriever/reranker:

```python
from transformers import AutoModel

extractor = AutoModel.from_pretrained(
    "KRLabsOrg/verbatim-rag-modern-bert-v2",
    trust_remote_code=True,
)

question = "What evidence supports using DINOv2 as the visual backbone?"
context = (
    "We investigate different visual backbones for feature extraction. "
    "The results demonstrate DINOv2's effectiveness as a feature extractor "
    "for sign language translation."
)

result = extractor.process(question=question, context=context)

for span in result["spans"]:
    print(
        {
            "start": span["start"],
            "end": span["end"],
            "text": span["text"],
            "score": span["score"],
        }
        )
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

Evaluated with the shared span-extraction harness used by
[`acl-verbatim`](https://github.com/KRLabsOrg/acl-verbatim). The current
metric protocol scores every row in a slice: rows without gold spans are
negative examples, and false-positive extracted text lowers precision.

The table below compares this generic model to two public extractive baselines:
Zilliz Semantic Highlight and Provence. All systems are evaluated with the
same all-row scorer. Latency is intentionally omitted because runtime depends
on device, batching, and serving setup.

| dataset | system | Word-P | Word-R | Word-F1 | IoU@0.5 | AnyOverlap | OverPred |
|---|---|---:|---:|---:|---:|---:|---:|
| ACL gold | **verbatim-rag-modern-bert-v2** | **0.630** | 0.366 | **0.463** | **0.364** | 0.449 | 0.692 |
| ACL gold | Zilliz semantic-highlight | 0.470 | 0.221 | 0.301 | 0.113 | 0.513 | 1.500 |
| ACL gold | Provence | 0.276 | **0.457** | 0.344 | 0.153 | **0.718** | 3.013 |
| RAGBench | **verbatim-rag-modern-bert-v2** | 0.516 | **0.770** | **0.618** | 0.309 | **0.753** | 0.732 |
| RAGBench | Zilliz semantic-highlight | **0.573** | 0.362 | 0.443 | 0.316 | 0.358 | 0.581 |
| RAGBench | Provence | 0.430 | 0.547 | 0.481 | **0.317** | 0.547 | 0.922 |
| Squeez | **verbatim-rag-modern-bert-v2** | **0.506** | **0.700** | **0.588** | **0.511** | **0.809** | 1.572 |
| Squeez | Zilliz semantic-highlight | 0.184 | 0.352 | 0.242 | 0.098 | 0.658 | 3.866 |
| Squeez | Provence | 0.107 | 0.576 | 0.180 | 0.103 | 0.756 | 3.951 |
| QASPER | **verbatim-rag-modern-bert-v2** | **0.688** | 0.409 | **0.513** | **0.366** | 0.515 | 0.848 |
| QASPER | Zilliz semantic-highlight | 0.622 | 0.191 | 0.293 | 0.122 | 0.479 | 0.793 |
| QASPER | Provence | 0.522 | **0.435** | 0.474 | 0.285 | **0.737** | 1.413 |

The generic model achieves the best Word-F1 on all four evaluated slices
against the public extractor baselines, including QASPER, which is not part of
the training mix. This is the main result: a 150M-parameter local encoder can
act as a strong general-purpose evidence highlighter across scientific papers,
RAGBench QA domains, coding-agent tool output, and QASPER scientific QA. The
advantage is strongest on RAGBench and Squeez, matching the multi-domain
training mix. On ACL gold, the generic model is also stronger than the public
pruning/highlighting baselines, though the ACL-specialized model remains best
on its home domain. Provence is often stronger on recall-oriented metrics such
as AnyOverlap, but tends to over-predict substantially more; Zilliz is generally
more conservative and lower recall.

Evaluation commands and slice construction are documented in
[`docs/GENERIC_EVAL.md`](https://github.com/KRLabsOrg/acl-verbatim/blob/main/docs/GENERIC_EVAL.md).
The committed summary backing this table is
[`artifacts/eval/generic_extraction/summary.csv`](https://github.com/KRLabsOrg/acl-verbatim/blob/main/artifacts/eval/generic_extraction/summary.csv).

## Citing

```bibtex
@inproceedings{kovacs-etal-2025-kr,
    title = "{KR} Labs at {A}rch{EHR}-{QA} 2025: A Verbatim Approach for Evidence-Based Question Answering",
    author = "Kovacs, Adam and Schmitt, Paul and Recski, Gabor",
    editor = "Soni, Sarvesh and Demner-Fushman, Dina",
    booktitle = "Proceedings of the 24th Workshop on Biomedical Language Processing (Shared Tasks)",
    month = aug,
    year = "2025",
    address = "Vienna, Austria",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.bionlp-share.8/",
    pages = "69--74"
}
```
