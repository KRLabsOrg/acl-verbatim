---
license: apache-2.0
library_name: transformers
pipeline_tag: token-classification
base_model: answerdotai/ModernBERT-base
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
[`answerdotai/ModernBERT-base`](https://huggingface.co/answerdotai/ModernBERT-base)
on silver spans from
[`KRLabsOrg/acl-verbatim-spans`](https://huggingface.co/datasets/KRLabsOrg/acl-verbatim-spans).

Input: `(question, context)` — output: character spans in `context` that
support the answer, with confidence scores.

The model uses the full 8192-token ModernBERT context, so long paper chunks
are handled without aggressive truncation.

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
    threshold=0.5,
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
| `threshold` | `0.5` | Probability cutoff for marking a token as evidence |
| `max_length` | `8192` | Max tokens per window (ModernBERT supports 8192) |
| `doc_stride` | `256` | Overlap between windows for long contexts |
| `return_sentence_metrics` | `False` | Also return per-sentence mean evidence score |

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

If you prefer to skip the `.process()` helper:

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
| base model | `answerdotai/ModernBERT-base` |
| dataset | `KRLabsOrg/acl-verbatim-spans` (`encoder` config) |
| label scheme | binary (`0` outside, `1` evidence) |
| max_length | 8192 |
| doc_stride | 256 |
| batch size | 8 |
| learning rate | 5e-5 |
| epochs | 3 |

Reproduce with:

```bash
python acl_verbatim/span_training/train_token_cls.py \
  --hf-dataset KRLabsOrg/acl-verbatim-spans \
  --hf-config encoder \
  --train-split train \
  --eval-split validation \
  --model answerdotai/ModernBERT-base \
  --output-dir runs/models/acl-verbatim-modernbert \
  --batch-size 8 \
  --lr 5e-5 \
  --epochs 3 \
  --label-scheme binary
```

## Evaluation

Scored on the `canonical/test` split of `KRLabsOrg/acl-verbatim-spans`
(20 queries × 5 retrieved chunks, 47 relevant rows, 78 gold spans) with the
shared span metrics in `acl_verbatim.eval.span_metrics`.

| metric | value |
|---|---:|
| word-F1 (micro) | TODO |
| span F1 @ IoU 0.5 | TODO |
| containment F1 @ 1.0 | TODO |
| gold-coverage recall @ 0.8 | TODO |
| recall @ any-overlap | TODO |

See the [`acl-verbatim`](https://github.com/KRLabsOrg/acl-verbatim) repo for the
full benchmark harness and comparison against LLM extractors and sentence-
selection baselines.

## Intended Use

- Query-conditioned evidence highlighting over scientific text
- Re-ranking or filtering of retrieval outputs for extractive QA
- Dataset annotation assistance

## Limitations

- Trained on ACL Anthology markdown; transfer to other scientific domains is
  not evaluated.
- Silver supervision inherits noise from the LLM teacher and the retriever.
- The gold benchmark is small (20 queries) and single-annotator.
- Tables and figures are represented through their caption text; the model has
  no structural awareness of tabular data.

## Citation

TODO
