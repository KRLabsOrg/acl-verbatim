"""Convert a span-pairs JSONL (output of prepare_ragbench_spans.py /
prepare_squeez_spans.py / silver pipeline) into gold-file JSONL — the format
that acl_verbatim/data/spans.py:load_gold_rows consumes.

Each input row {question, paper_id, chunk_index, chunk, label, spans, ...}
becomes one query_record with a single nested result.

Example:
    python scripts/experiments/prepare_ragbench_spans.py --split test \\
        --output-file runs/eval_data/ragbench_test.spans.jsonl
    python scripts/experiments/spans_jsonl_to_gold_file.py \\
        runs/eval_data/ragbench_test.spans.jsonl \\
        --output runs/eval/test_slices/ragbench.gold.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def convert_row(row: dict) -> dict:
    is_relevant = int(row.get("label", 0)) == 1
    spans = row.get("spans") or []
    paper_id = str(row.get("paper_id") or row.get("gold_paper") or "")
    chunk_index = int(row.get("chunk_index") or row.get("gold_chunk") or 0)
    return {
        "query": row["question"],
        "gold_paper": paper_id,
        "gold_chunk": chunk_index,
        "results": [
            {
                "document_id": paper_id,
                "chunk_number": chunk_index,
                "chunk": row["chunk"],
                "relevance_label": "r" if is_relevant else "n",
                "gold_extraction": ([s["text"] for s in spans] if is_relevant else []),
            }
        ],
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("input", type=Path, help="span-pairs JSONL")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument(
        "--relevant-only",
        action="store_true",
        help="Drop label=0 rows (eval already skips them; flag is for smaller files).",
    )
    args = p.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    n_in = n_out = n_rel = n_spans = 0
    with args.input.open() as fin, args.output.open("w") as fout:
        for line in fin:
            row = json.loads(line)
            n_in += 1
            if args.relevant_only and int(row.get("label", 0)) != 1:
                continue
            converted = convert_row(row)
            fout.write(json.dumps(converted) + "\n")
            n_out += 1
            for r in converted["results"]:
                if r["relevance_label"] == "r":
                    n_rel += 1
                    n_spans += len(r["gold_extraction"])
    print(
        f"read {n_in} rows -> wrote {n_out} ({n_rel} relevant, {n_spans} gold spans) "
        f"to {args.output}"
    )


if __name__ == "__main__":
    main()
