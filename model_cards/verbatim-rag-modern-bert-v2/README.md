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
  - multi-domain
  - verbatim-rag
---

# Verbatim-RAG ModernBERT v2

[![ChiliGround Logo](https://github.com/KRLabsOrg/verbatim-rag/raw/main/assets/chiliground.png?raw=true)](https://github.com/KRLabsOrg/verbatim-rag)
_Chill, I Ground! 🌶️_

A query-conditioned token classifier that highlights the verbatim spans of a
passage that answer a question. The encoder companion to
[VerbatimRAG](https://github.com/KRLabsOrg/verbatim-rag), and successor to
[`KRLabsOrg/verbatim-rag-modern-bert-v1`](https://huggingface.co/KRLabsOrg/verbatim-rag-modern-bert-v1).

Input: `(question, context)` — output: character spans in `context` that
support the answer, with confidence scores. 8192-token context window. ~50 ms
per `(question, context)` pair on a single GPU.

For an **ACL-Anthology-specialized** variant (stronger on academic papers,
weaker elsewhere), see
[`KRLabsOrg/acl-verbatim-modernbert`](https://huggingface.co/KRLabsOrg/acl-verbatim-modernbert).

## What makes v2 different

Most public evidence-extraction models (Provence, Zilliz, OpenSearch
semantic-highlight, MultiSpanQA-trained baselines) are trained on
Wikipedia-style QA: clean prose, noun-phrase answers, sentence-grain
evidence. They do that well — and they struggle the moment your RAG context
contains anything *else*.

v2 was deliberately trained on a mix that covers the content shapes that
modern RAG and agent applications actually retrieve:

| content shape | source in training mix | what it teaches the model |
|---|---|---|
| scientific paragraphs with citations | ACL silver | long evidence spans, Author (Year) chains, mixed prose+formula |
| Wikipedia / general QA | RAGBench (HotpotQA, MS MARCO, ExpertQA, ...) | sentence-grain evidence, multi-hop |
| **financial tables** | RAGBench (TAT-QA, FinQA) | markdown table rows as evidence units |
| **medical literature** | RAGBench (PubMedQA, CovidQA) | dense technical prose |
| **legal contracts** | RAGBench (CUAD) | clause-grain extraction |
| **product manuals** | RAGBench (eManual, TechQA) | instructional prose, numbered steps |
| **code / tool output / stack traces / logs** | Squeez (SWE-bench tool outputs) | file paths, line numbers, error messages, log-line evidence |

The Squeez slice especially is novel — no other public evidence extractor we
know of trains on coding-agent tool output. That's why v2 hits 0.769 word-F1
on Squeez test vs ~0.49 for Provence and 0.42 for Zilliz: those models
literally have not seen evidence selection over `pytest` failures, `git diff`
hunks, or stack traces during training.

The result is a single 150M-parameter encoder you can drop into a RAG or
agent pipeline regardless of what the retrieval layer surfaces — markdown
articles, paper PDFs, CSV exports, log files, GitHub READMEs, JSON tool
responses. See [Evaluation](#evaluation) for the per-domain numbers.

## What's new vs v1

| | v1 | v2 |
|---|---|---|
| backbone | ModernBERT | gte-reranker-modernbert (query-conditioned prior) |
| training data | single-source | multi-domain (ACL silver + RAGBench + Squeez) |
| content shapes | prose only | prose + tables + code + logs + structured docs |
| context window | 2048 | 8192 |
| API | `.process()` returning sentences | `.process()` returning char spans + optional sentences |

## Quick Start

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

### Parameters

| arg | default | notes |
|---|---|---|
| `question` | — | Query string |
| `context` | — | Passage to search for supporting spans |
| `threshold` | `0.2` | Probability cutoff for marking a token as evidence. Lower for higher recall |
| `max_length` | `8192` | Max tokens per window |
| `doc_stride` | `256` | Overlap between sliding windows for long contexts |
| `min_span_chars` | `30` | Drop predicted spans shorter than this many characters |
| `merge_gap_chars` | `20` | Merge adjacent predicted spans separated by ≤ this many characters |
| `return_sentence_metrics` | `False` | Also return per-sentence mean evidence score |

`min_span_chars=30` is tuned for the multi-domain mix. RAGBench and Squeez
training data contains many short structured spans (table cells, log lines)
that produce noisy fragmentation without a higher minimum. Lower it for
short-answer benchmarks like MultiSpanQA.

### Return shape

```python
{
    "spans": [
        {"start": int, "end": int, "text": str, "score": float},
        ...
    ],
    "sentences": [...]  # only when return_sentence_metrics=True
}
```

Spans are character offsets into the input `context` and are merged across
sliding windows.

## Evaluation

Evaluated on the test splits of all three training sources, plus a baseline
comparison against [Provence](https://huggingface.co/naver/provence-reranker-debertav3-v1)
and [Zilliz Semantic-Highlight](https://huggingface.co/zilliz/semantic-highlight-bilingual-v1).
Same harness for every system; word-level micro F1 is the headline.

### Per-domain test results

| domain | rows | this model (v2) | acl-specialized | provence | zilliz |
|---|---:|---:|---:|---:|---:|
| ACL Anthology gold | 47 | 0.495 | **0.562** | 0.480 | 0.322 |
| RAGBench (12 configs) | ~17k | **0.759** | 0.598 | 0.615 | 0.511 |
| Squeez tool-output | ~1k | **0.769** | 0.493 | 0.491 | 0.418 |

Word-F1 (micro). The ACL-specialized model wins on its home turf; v2 wins
everywhere else by 14–28 points and is only 6.7 points behind the specialist
on ACL.

### Detailed numbers (this model)

| domain | word-P | word-R | word-F1 | IoU@0.5 F1 | recall@any-overlap | over-pred ratio |
|---|---:|---:|---:|---:|---:|---:|
| ACL gold | 0.728 | 0.375 | 0.495 | 0.382 | 0.449 | 0.679 |
| RAGBench test | 0.744 | 0.775 | 0.759 | 0.359 | 0.762 | 0.484 |
| Squeez test | 0.845 | 0.705 | 0.769 | 0.700 | 0.823 | 0.900 |

### How to read these numbers

- The model is **strongest on its in-distribution domains** (RAGBench and
  Squeez are part of the training mix; ACL is silver-only). These results
  are not a zero-shot generalization claim — they are "fits its training
  distribution well."
- **Provence and Zilliz were not trained on Squeez-style structured tool
  output.** Their numbers on the Squeez slice should be read as a scope
  comment rather than a capability comparison.
- The over-prediction ratio of 0.9 on Squeez means the model emits
  ~90% as many spans as there are gold spans — slightly aggressive on
  log-line text. Raise `threshold` or `min_span_chars` if you need higher
  precision in tool-output applications.

### Recall-tuned config for ACL-style academic text

Lowering `threshold` from 0.2 to 0.1 (and dropping `min_span_chars` to 10 to
keep short spans) trades 5pt of precision for 5pt of recall on ACL gold:

| config | word-P | word-R | word-F1 | recall@any-overlap |
|---|---:|---:|---:|---:|
| default (`threshold=0.2`, `min_span_chars=30`) | 0.728 | 0.375 | 0.495 | 0.449 |
| **recall-tuned** (`threshold=0.1`, `min_span_chars=10`) | 0.688 | 0.421 | **0.523** | 0.474 |

This narrows the gap to the ACL-specialized model (0.562 word-F1) to ~4pt.
Use the recall-tuned config when answering questions over scientific papers;
keep the default for tool-output and short-span QA where extra firings cost
more than they help.

Reproduce with the eval scripts in the
[acl-verbatim repo](https://github.com/KRLabsOrg/acl-verbatim) — see
[`docs/GENERIC_EVAL.md`](https://github.com/KRLabsOrg/acl-verbatim/blob/main/docs/GENERIC_EVAL.md)
for the full sweep.

## Training

| item | value |
|---|---|
| base model | `Alibaba-NLP/gte-reranker-modernbert-base` |
| dataset | `KRLabsOrg/verbatim-spans` (`encoder` config) |
| label scheme | binary (`0` outside, `1` evidence) |
| max_length | 8192 |
| doc_stride | 256 |

Reproduce with:

```bash
python acl_verbatim/span_training/train_token_cls.py \
  --hf-dataset KRLabsOrg/verbatim-spans \
  --hf-config encoder \
  --train-split train \
  --eval-split validation \
  --model Alibaba-NLP/gte-reranker-modernbert-base \
  --output-dir runs/models/verbatim-rag-modern-bert-v2 \
  --batch-size 8 \
  --lr 2e-5 \
  --epochs 5 \
  --label-scheme binary
```

## Intended Use

- Query-conditioned evidence highlighting over arbitrary passages
- Drop-in extractor for `verbatim-rag` outside the academic domain
- Re-ranking or filtering of retrieval outputs
- Evidence selection for tool-output pruning

## Limitations

- Test results on RAGBench and Squeez reflect in-distribution performance,
  not zero-shot generalization to unseen domains.
- Multi-domain training comes with a ~6.7-point F1 cost on ACL papers vs
  the ACL-specialized variant — pick the right model for your domain.
- Recall ceiling on ACL gold (~0.45 any-overlap) means the model sometimes
  predicts nothing on chunks that contain relevant evidence. For
  high-recall applications, lower `threshold` or combine with an LLM
  fallback.
- Tables, figures, and structured content are handled through their text
  representation; no structural awareness of tabular data.

## Citation

TODO
