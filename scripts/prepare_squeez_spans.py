"""Convert the Squeez tool-output-extraction dataset to our span-pairs format.

Source: `KRLabsOrg/tool-output-extraction-swebench-gliner` — per-chunk evidence
for coding-agent tool outputs, formatted for GLiNER. Same extractive pattern
as our ACL / RAGBench spans.

Each source row has:
  - `input`:  "Query: <question>\\n\\nTool output:\\n<chunk>"
  - `output.entities.RELEVANT`: list of verbatim substrings of the chunk
  - `meta`:   { query, instance_id, tool_type, chunk_index, has_evidence, ... }

We drop the "Query: ... Tool output:\\n" prefix, treat the remainder as the
chunk, and locate each RELEVANT entity via substring search.

Example:
    python scripts/prepare_squeez_spans.py --output-file runs/squeez/train.jsonl --split train
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets import load_dataset


TOOL_OUTPUT_MARKER = "\n\nTool output:\n"


def extract_chunk(input_text: str) -> str | None:
    idx = input_text.find(TOOL_OUTPUT_MARKER)
    if idx == -1:
        return None
    return input_text[idx + len(TOOL_OUTPUT_MARKER) :]


def build_row(example: dict) -> dict | None:
    chunk = extract_chunk(example.get("input") or "")
    if chunk is None:
        return None
    meta = example.get("meta") or {}
    question = meta.get("query") or ""
    if not question:
        return None

    entities = ((example.get("output") or {}).get("entities") or {}).get("RELEVANT") or []

    spans: list[dict] = []
    cursor = 0
    for ent_text in entities:
        if not ent_text:
            continue
        idx = chunk.find(ent_text, cursor)
        if idx == -1:
            idx = chunk.find(ent_text)
            if idx == -1:
                continue
        spans.append({"start": idx, "end": idx + len(ent_text), "text": ent_text})
        cursor = idx + len(ent_text)

    has_spans = bool(spans)
    instance_id = meta.get("instance_id", "")
    chunk_index = int(meta.get("chunk_index", 0) or 0)
    tool_type = meta.get("tool_type", "")

    return {
        "question": question,
        "paper_id": f"squeez/{tool_type}/{instance_id}",
        "chunk_index": chunk_index,
        "chunk": chunk,
        "label": 1 if has_spans else 0,
        "answerable": has_spans,
        "spans": spans,
        "source": "gold",
        "retrieval_rank": chunk_index + 1,
        "gold_paper": f"squeez/{instance_id}",
        "gold_chunk": chunk_index,
        "predicted_texts": [],
        "latency_s": 0.0,
        "err": "",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-file", type=Path, required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Cap rows for smoke testing",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print stats without writing",
    )
    args = parser.parse_args()

    ds = load_dataset(
        "KRLabsOrg/tool-output-extraction-swebench-gliner", split=args.split
    )
    if args.limit:
        ds = ds.select(range(min(args.limit, len(ds))))

    rows: list[dict] = []
    positive = 0
    for example in ds:
        row = build_row(example)
        if row is None:
            continue
        rows.append(row)
        if row["label"] == 1:
            positive += 1

    print(f"total rows: {len(rows)}  positive: {positive} ({100*positive/max(1,len(rows)):.1f}%)")

    if args.dry_run:
        return

    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    with args.output_file.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    print(f"wrote {len(rows)} rows to {args.output_file}")


if __name__ == "__main__":
    main()
