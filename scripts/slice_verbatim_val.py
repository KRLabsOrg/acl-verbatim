"""Slice the KRLabsOrg/verbatim-spans validation split by source_dataset
into gold-file-shaped JSONL files that the standard eval harness consumes.

Each output line follows the format that acl_verbatim/data/spans.py:load_gold_rows
expects: one query_record per row, with a single nested result wrapping the chunk.
"""

import argparse
import json
from pathlib import Path

from datasets import load_dataset


def normalize_spans(raw):
    if raw is None:
        return []
    if isinstance(raw, dict):
        starts = raw.get("start") or []
        ends = raw.get("end") or []
        texts = raw.get("text") or []
        return [
            {"start": int(s), "end": int(e), "text": str(t)}
            for s, e, t in zip(starts, ends, texts)
        ]
    out = []
    for sp in raw:
        out.append(
            {"start": int(sp["start"]), "end": int(sp["end"]), "text": str(sp["text"])}
        )
    return out


def row_to_query_record(row, fallback_index: int):
    spans = normalize_spans(row.get("spans"))
    paper_id = row.get("paper_id") or f"{row.get('source_dataset', 'src')}-{fallback_index}"
    chunk_index = row.get("chunk_index")
    if chunk_index is None:
        chunk_index = 0
    is_relevant = int(row.get("label", 0)) == 1
    return {
        "query": row["question"],
        "gold_paper": paper_id,
        "gold_chunk": int(chunk_index),
        "results": [
            {
                "document_id": str(paper_id),
                "chunk_number": int(chunk_index),
                "chunk": row["chunk"],
                "relevance_label": "r" if is_relevant else "n",
                "gold_extraction": [s["text"] for s in spans] if is_relevant else [],
            }
        ],
    }


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--hf-dataset", default="KRLabsOrg/verbatim-spans")
    p.add_argument("--hf-config", default="canonical")
    p.add_argument("--split", default="validation")
    p.add_argument("--output-dir", default="runs/eval/verbatim_val_slices")
    p.add_argument(
        "--sources",
        nargs="*",
        default=None,
        help="Optional subset of source_dataset values; default = all observed.",
    )
    p.add_argument(
        "--relevant-only",
        action="store_true",
        help="Drop rows with label=0. Eval scripts already skip them, so this is "
        "purely for smaller files.",
    )
    return p.parse_args()


def main():
    args = get_args()
    ds = load_dataset(args.hf_dataset, args.hf_config, split=args.split)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    by_source: dict[str, list[dict]] = {}
    for i, row in enumerate(ds):
        src = row.get("source_dataset") or "unknown"
        if args.sources and src not in args.sources:
            continue
        if args.relevant_only and int(row.get("label", 0)) != 1:
            continue
        record = row_to_query_record(row, fallback_index=i)
        by_source.setdefault(src, []).append(record)

    for src, records in by_source.items():
        out_path = out_dir / f"{src}.gold.jsonl"
        with out_path.open("w", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec) + "\n")
        n_rel = sum(
            1
            for rec in records
            for r in rec["results"]
            if r["relevance_label"] == "r"
        )
        n_spans = sum(
            len(r["gold_extraction"])
            for rec in records
            for r in rec["results"]
        )
        print(
            f"{src}: {len(records)} rows ({n_rel} relevant, {n_spans} gold spans) "
            f"-> {out_path}"
        )


if __name__ == "__main__":
    main()
