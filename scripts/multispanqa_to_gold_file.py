"""Convert MultiSpanQA token-tagged JSON to gold-file JSONL.

MultiSpanQA ships as a list of {question, context, label} dicts where
`question` and `context` are token lists and `label` is a BIO tag list aligned
to `context`. We reconstruct text by space-joining tokens, derive char offsets
from the running position, and group BIO tags into evidence spans.

Used in: Zilliz semantic-highlight blog post; Provence; many evidence-selection
papers — direct comparison point for the model card.

Example:
    python scripts/multispanqa_to_gold_file.py \\
        /Users/adamkovacs/Downloads/MultiSpanQA_data/valid.json \\
        --output runs/eval/test_slices/multispanqa.gold.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def join_with_offsets(tokens: list[str]) -> tuple[str, list[tuple[int, int]]]:
    """Space-join tokens; return joined text + per-token (start, end) char offsets."""
    pieces: list[str] = []
    offsets: list[tuple[int, int]] = []
    pos = 0
    for i, tok in enumerate(tokens):
        if i > 0:
            pieces.append(" ")
            pos += 1
        offsets.append((pos, pos + len(tok)))
        pieces.append(tok)
        pos += len(tok)
    return "".join(pieces), offsets


def bio_to_spans(
    labels: list[str], offsets: list[tuple[int, int]], text: str
) -> list[dict]:
    """Group consecutive B/I tags into char-span dicts."""
    spans: list[dict] = []
    cur_start: int | None = None
    cur_end: int | None = None
    for label, (s, e) in zip(labels, offsets):
        if label == "B":
            if cur_start is not None:
                spans.append(
                    {
                        "start": cur_start,
                        "end": cur_end,
                        "text": text[cur_start:cur_end],
                    }
                )
            cur_start, cur_end = s, e
        elif label == "I" and cur_start is not None:
            cur_end = e
        else:
            if cur_start is not None:
                spans.append(
                    {
                        "start": cur_start,
                        "end": cur_end,
                        "text": text[cur_start:cur_end],
                    }
                )
                cur_start = cur_end = None
    if cur_start is not None:
        spans.append(
            {"start": cur_start, "end": cur_end, "text": text[cur_start:cur_end]}
        )
    return spans


def convert_example(ex: dict) -> dict | None:
    question_tokens = ex.get("question") or []
    context_tokens = ex.get("context") or []
    labels = ex.get("label") or []
    if not question_tokens or not context_tokens:
        return None
    question_text, _ = join_with_offsets(question_tokens)
    context_text, ctx_offsets = join_with_offsets(context_tokens)
    spans = bio_to_spans(labels, ctx_offsets, context_text) if labels else []
    is_relevant = bool(spans)
    qid = str(ex.get("id") or "")
    return {
        "query": question_text,
        "gold_paper": f"multispanqa/{qid}",
        "gold_chunk": 0,
        "results": [
            {
                "document_id": f"multispanqa/{qid}",
                "chunk_number": 0,
                "chunk": context_text,
                "relevance_label": "r" if is_relevant else "n",
                "gold_extraction": [s["text"] for s in spans] if is_relevant else [],
            }
        ],
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("input", type=Path, help="MultiSpanQA JSON (e.g. valid.json)")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()

    payload = json.loads(args.input.read_text())
    examples = (
        payload["data"] if isinstance(payload, dict) and "data" in payload else payload
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    n_in = n_out = n_rel = n_spans = 0
    with args.output.open("w") as fout:
        for ex in examples:
            n_in += 1
            converted = convert_example(ex)
            if converted is None:
                continue
            fout.write(json.dumps(converted) + "\n")
            n_out += 1
            for r in converted["results"]:
                if r["relevance_label"] == "r":
                    n_rel += 1
                    n_spans += len(r["gold_extraction"])
    print(
        f"read {n_in} examples -> wrote {n_out} ({n_rel} relevant, {n_spans} gold spans) "
        f"to {args.output}"
    )


if __name__ == "__main__":
    main()
