---
license: apache-2.0
library_name: transformers
pipeline_tag: token-classification
base_model: Alibaba-NLP/gte-reranker-modernbert-base
language:
  - en
datasets:
  - KRLabsOrg/acl-verbatim-spans
tags:
  - semantic-highlighting
  - extractive-qa
  - evidence-selection
  - acl-anthology
  - modernbert
---

# ACL-Verbatim ModernBERT Highlighter

A query-conditioned token classifier that highlights supporting evidence spans
in scientific paper chunks. Fine-tuned from
[`Alibaba-NLP/gte-reranker-modernbert-base`](https://huggingface.co/Alibaba-NLP/gte-reranker-modernbert-base)
on silver spans from
[`KRLabsOrg/acl-verbatim-spans`](https://huggingface.co/datasets/KRLabsOrg/acl-verbatim-spans).

Input: `(question, context)` — output: character spans in `context` that
support the answer, with confidence scores.

The model uses the full 8192-token ModernBERT context, so long paper chunks
are handled without aggressive truncation. A 150M-parameter model that matches
the word-level F1 of 120B-parameter LLMs on this benchmark while running
~40 ms per (question, chunk) on GPU.

## Quick Start

```python
from transformers import AutoModel

model = AutoModel.from_pretrained(
    "KRLabsOrg/acl-verbatim-modernbert",
    trust_remote_code=True,
)

question = "What is ModernBERT?"
context = (
    "ModernBERT is a long-context encoder for NLP. "
    "It supports sequences up to 8192 tokens. "
    "Unlike earlier BERT variants, it uses rotary position embeddings."
)

result = model.process(
    question=question,
    context=context,
    threshold=0.2,
    return_sentence_metrics=True,
)

for span in result["spans"]:
    print(f"[{span['score']:.2f}] {span['text']}")
```

Example output:

```
[0.93] ModernBERT is a long-context encoder for NLP.
[0.87] It supports sequences up to 8192 tokens.
```

### Parameters

| arg | default | notes |
|---|---|---|
| `question` | — | Query string |
| `context` | — | Passage to search for supporting spans |
| `threshold` | `0.2` | Probability cutoff for marking a token as evidence. Use `0.2` for balanced F1, `0.5` for high precision |
| `max_length` | `8192` | Max tokens per window (ModernBERT supports 8192) |
| `doc_stride` | `256` | Overlap between windows for long contexts |
| `min_span_chars` | `10` | Drop predicted spans shorter than this many characters |
| `merge_gap_chars` | `20` | Merge adjacent predicted spans separated by ≤ this many characters |
| `return_sentence_metrics` | `False` | Also return per-sentence mean evidence score |

`min_span_chars` and `merge_gap_chars` together clean up token-level
fragmentation: without them, binary token labels often produce a "shotgun" of
3–10-character pseudo-spans that hurt span-level metrics. The defaults are what
we use in our evaluation.

### Return shape

```python
{
    "spans": [
        {"start": int, "end": int, "text": str, "score": float},
        ...
    ],
    "sentences": [  # only when return_sentence_metrics=True
        {"start": int, "end": int, "text": str, "score": float},
        ...
    ],
}
```

Spans are character offsets into the input `context`. They are merged across
sliding windows, so callers do not need to deduplicate.

## Raw Inference

If you prefer to skip the `.process()` helper and do the post-processing
yourself:

```python
from transformers import AutoTokenizer, AutoModelForTokenClassification

tokenizer = AutoTokenizer.from_pretrained("KRLabsOrg/acl-verbatim-modernbert")
model = AutoModelForTokenClassification.from_pretrained(
    "KRLabsOrg/acl-verbatim-modernbert"
)

enc = tokenizer(
    question, context,
    return_offsets_mapping=True,
    max_length=8192,
    truncation="only_second",
    return_tensors="pt",
)
logits = model(
    input_ids=enc["input_ids"], attention_mask=enc["attention_mask"]
).logits
labels = logits.argmax(dim=-1)
```

Label `0` is "outside", label `1` is "evidence" (binary scheme).

## Training

| item | value |
|---|---|
| base model | `Alibaba-NLP/gte-reranker-modernbert-base` |
| dataset | `KRLabsOrg/acl-verbatim-spans` (`encoder` config) |
| label scheme | binary (`0` outside, `1` evidence) |
| max_length | 8192 |
| doc_stride | 256 |
| batch size | 8 |
| learning rate | 2e-5 |
| epochs | 5 |
| best checkpoint | silver-dev F1 = 0.642 at epoch 3.21 |

We started from `Alibaba-NLP/gte-reranker-modernbert-base` rather than
vanilla `answerdotai/ModernBERT-base` because the reranker backbone has
already been post-trained on query/passage relevance — the semantic prior it
provides gives a large head start on query-conditioned span extraction. In
our ablations, this backbone swap alone improved gold word-F1 from 0.407 to
0.513 at argmax (and 0.449 to 0.563 with threshold tuning and post-processing).

Reproduce with:

```bash
python acl_verbatim/span_training/train_token_cls.py \
  --hf-dataset KRLabsOrg/acl-verbatim-spans \
  --hf-config encoder \
  --train-split train \
  --eval-split validation \
  --model Alibaba-NLP/gte-reranker-modernbert-base \
  --output-dir runs/models/acl-verbatim-modernbert \
  --batch-size 8 \
  --lr 2e-5 \
  --epochs 5 \
  --label-scheme binary
```

## Evaluation

Scored on the `canonical/test` split of `KRLabsOrg/acl-verbatim-spans`
(20 queries × 5 retrieved chunks, 47 relevant rows, 78 gold spans) with the
shared span metrics in `acl_verbatim.eval.span_metrics`.

### Headline numbers (balanced config: threshold=0.2 + merge)

| metric | value |
|---|---:|
| word-F1 (micro) | **0.563** |
| word precision | 0.738 |
| word recall | 0.454 |
| span F1 @ IoU 0.3 | 0.473 |
| span F1 @ IoU 0.5 | 0.427 |
| containment F1 @ 0.5 | 0.527 |
| containment F1 @ 0.8 | 0.343 |
| containment F1 @ 1.0 | 0.297 |
| gold-coverage recall @ 0.5 | 0.423 |
| gold-coverage recall @ 0.8 | 0.372 |
| recall @ any-overlap | 0.500 |
| over-prediction ratio | 0.679 |
| mean latency (GPU) | ~40 ms |

### How this compares

On the same benchmark and harness:

| system | word-F1 | IoU F1 @ 0.5 | any-overlap R |
|---|---:|---:|---:|
| **acl-verbatim-modernbert (this model)** | **0.563** | **0.427** | 0.500 |
| nemotron-120b-a12b | 0.561 | 0.437 | 0.654 |
| nemotron-120b-paragraph | 0.552 | 0.459 | 0.577 |
| qwen-3.6-paragraph (silver teacher) | 0.544 | 0.486 | 0.692 |
| mistral-small-2603 | 0.519 | 0.201 | 0.692 |
| glm-5 | 0.495 | 0.250 | 0.744 |
| qwen-3.6-default | 0.494 | 0.242 | 0.705 |
| provence-reranker-pruner | 0.480 | 0.276 | 0.718 |
| zilliz semantic-highlight | 0.217 | 0.088 | 0.321 |

This 150M-parameter model matches the word-F1 of the 120B nemotron (0.563 vs
0.561) and exceeds its silver teacher (0.544) by 1.9 points. LLMs retain an
advantage on any-overlap recall — they find more relevant passages across
chunks — but the student is competitive on token coverage where it fires.

### Threshold / post-processing ablation

| config | word-F1 | IoU F1 @ 0.5 | over-prediction |
|---|---:|---:|---:|
| argmax (no merge) | 0.513 | 0.250 | 1.564 |
| argmax + merge | 0.512 | 0.357 | 0.654 |
| threshold 0.3 (no merge) | 0.539 | 0.250 | 1.462 |
| **threshold 0.2 + merge** | **0.563** | **0.427** | **0.679** |

Lower thresholds boost recall; span merging + min-length filtering cleans up
fragmentation without hurting F1. The `threshold=0.2` + merge configuration
is the default in `model.process()`.

See the [`acl-verbatim`](https://github.com/KRLabsOrg/acl-verbatim) repo for
the full benchmark harness, LLM extractor scripts, and qualitative analysis.

## Intended Use

- Query-conditioned evidence highlighting over scientific text
- Re-ranking or filtering of retrieval outputs for extractive QA
- Dataset annotation assistance
- Fast local alternative to LLM extractors for evidence selection

## Limitations

- Trained on ACL Anthology markdown; transfer to other scientific domains
  (biomedical, legal, patents) is not evaluated.
- Silver supervision inherits noise from the LLM teacher and the retriever.
  Recall in particular reflects teacher behaviour: the model rarely extracts
  a passage the teacher would have skipped.
- The gold benchmark is small (20 queries, 47 relevant chunks, 78 gold spans)
  and single-annotator; confidence intervals on the headline numbers are wide.
- Tables and figures are represented through their caption text; the model
  has no structural awareness of tabular data.
- Any-overlap recall (0.500) lags frontier LLMs (0.65–0.74), meaning the
  model sometimes predicts nothing on chunks that contain relevant evidence.
  For high-recall applications, lower `threshold` further or combine with an
  LLM fallback.

## Citation

TODO
