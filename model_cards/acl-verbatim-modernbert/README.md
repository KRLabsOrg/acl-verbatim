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
are handled without aggressive truncation. On the current all-row ACL gold
benchmark, this 150M-parameter model achieves the best committed word-level F1
among the evaluated extractors.

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
provides gives a large head start on query-conditioned span extraction.

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
(20 queries × 5 retrieved chunks: 100 rows total, 47 relevant rows with 78
gold spans, and 53 irrelevant negative rows) with the shared span metrics in
`acl_verbatim.eval.span_metrics`. Irrelevant rows have empty gold spans, so
false-positive extracted text lowers precision.

### Headline numbers (balanced config: threshold=0.2 + merge)

| metric | value |
|---|---:|
| word-F1 (micro) | **0.536** |
| word precision | 0.654 |
| word recall | 0.454 |
| span F1 @ IoU 0.3 | 0.473 |
| span F1 @ IoU 0.5 | 0.389 |
| containment F1 @ 0.5 | 0.527 |
| containment F1 @ 0.8 | 0.343 |
| containment F1 @ 1.0 | 0.273 |
| gold-coverage recall @ 0.5 | 0.423 |
| gold-coverage recall @ 0.8 | 0.372 |
| recall @ any-overlap | 0.500 |
| over-prediction ratio | 0.846 |
| mean latency (local eval run) | 0.468 s |

### How this compares

On the same benchmark and harness:

| system | word-F1 | IoU F1 @ 0.5 | any-overlap R |
|---|---:|---:|---:|
| **acl-verbatim-modernbert (this model)** | **0.536** | 0.389 | 0.500 |
| glm-5 | 0.487 | 0.287 | 0.795 |
| mistral-small-2603 | 0.469 | 0.173 | 0.782 |
| qwen-3.6-paragraph | 0.467 | 0.424 | 0.692 |
| mistral-small-2603-paragraph | 0.466 | 0.346 | 0.795 |
| qwen-3.6-default | 0.427 | 0.234 | 0.641 |
| nemotron-120b-a12b | 0.409 | 0.302 | 0.667 |
| nemotron-120b-paragraph | 0.407 | **0.435** | 0.667 |
| provence-reranker-pruner | 0.344 | 0.153 | 0.718 |
| zilliz semantic-highlight | 0.301 | 0.113 | 0.513 |

These are all-row scores: irrelevant retrieved chunks are included as negative
examples and false-positive evidence on those rows lowers precision.

### Threshold / post-processing ablation

The released default is `threshold=0.2`, min-span length 10, and merge gap 20.
Additional threshold ablations can be regenerated with
`acl_verbatim/span_training/evaluate_token_cls.py`. The `threshold=0.2` +
merge configuration is the default in `model.process()`.

See the [`acl-verbatim`](https://github.com/KRLabsOrg/acl-verbatim) repo for
the full benchmark harness, LLM extractor scripts, and qualitative analysis.

## Intended Use

- Query-conditioned evidence highlighting over scientific text
- Re-ranking or filtering of retrieval outputs for extractive QA
- Dataset annotation assistance
- Local alternative to LLM extractors for evidence selection

## Limitations

- Trained on ACL Anthology markdown; transfer to other scientific domains
  (biomedical, legal, patents) is not evaluated.
- Silver supervision inherits noise from the LLM teacher and the retriever.
  Recall in particular reflects teacher behaviour: the model rarely extracts
  a passage the teacher would have skipped.
- The gold benchmark is small (20 queries, 100 query--chunk rows, 47 relevant
  chunks, 78 gold spans) and single-annotator; confidence intervals on the
  headline numbers are wide.
- Tables and figures are represented through their caption text; the model
  has no structural awareness of tabular data.
- Any-overlap recall (0.500) lags the LLM extractors rerun here, meaning the
  model sometimes predicts nothing on chunks that contain relevant evidence.
  For high-recall applications, lower `threshold` further or combine with an
  LLM fallback.

## Citation

Citation information will be added with the ACL-Verbatim paper release. For
now, please cite the model repository and dataset if you use this model.
