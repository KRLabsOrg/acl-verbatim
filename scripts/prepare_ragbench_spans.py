"""Convert RAGBench to the span-pairs JSONL format used by the silver pipeline.

RAGBench (`galileo-ai/ragbench`) provides GPT-4o-annotated, sentence-level
evidence keys per (question, document) pair. We treat each document as a chunk
and emit one row per (question, document) — with gold spans reconstructed from
`documents_sentences` and `all_relevant_sentence_keys`.

Output matches the schema produced by `annotate_spans_from_results_batched.py`,
so the downstream pipeline (`filter_silver_dataset.py` →
`prepare_token_cls_dataset.py` → `train_token_cls.py`) works unchanged.

Example:
    python scripts/prepare_ragbench_spans.py \\
        --output-file runs/ragbench/spans.jsonl \\
        --configs covidqa hotpotqa msmarco finqa pubmedqa \\
        --split train
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

from datasets import load_dataset


ALL_CONFIGS = (
    "covidqa", "cuad", "delucionqa", "emanual", "expertqa", "finqa",
    "hagrid", "hotpotqa", "msmarco", "pubmedqa", "tatqa", "techqa",
)
SENTENCE_KEY_RE = re.compile(r"^(\d+)([a-z]+)$")


def parse_key(key: str) -> tuple[int, str] | None:
    match = SENTENCE_KEY_RE.match(key)
    if not match:
        return None
    return int(match.group(1)), match.group(2)


def sentence_offsets(chunk: str, sentences: list[list[str]]) -> dict[str, tuple[int, int]]:
    """Map each sentence key to its character span in the original chunk."""
    offsets: dict[str, tuple[int, int]] = {}
    cursor = 0
    for entry in sentences:
        key, text = entry[0], entry[1]
        idx = chunk.find(text, cursor)
        if idx == -1:
            idx = chunk.find(text)
            if idx == -1:
                continue
        offsets[key] = (idx, idx + len(text))
        cursor = idx + len(text)
    return offsets


def build_row(
    question: str,
    document: str,
    doc_idx: int,
    doc_sentences: list[list[str]],
    relevant_keys: set[str],
    gold_paper: str,
    config_name: str,
) -> dict | None:
    if not document or not doc_sentences:
        return None
    offsets = sentence_offsets(document, doc_sentences)
    spans = []
    for key, _ in doc_sentences:
        parsed = parse_key(key)
        if parsed is None or parsed[0] != doc_idx:
            continue
        if key not in relevant_keys:
            continue
        if key not in offsets:
            continue
        start, end = offsets[key]
        spans.append({"start": start, "end": end, "text": document[start:end]})
    has_spans = bool(spans)
    return {
        "question": question,
        "paper_id": f"ragbench/{config_name}/{gold_paper}/doc{doc_idx}",
        "chunk_index": doc_idx,
        "chunk": document,
        "label": 1 if has_spans else 0,
        "answerable": has_spans,
        "spans": spans,
        "source": "gold",
        "retrieval_rank": doc_idx + 1,
        "gold_paper": f"ragbench/{config_name}/{gold_paper}",
        "gold_chunk": doc_idx,
        "predicted_texts": [],
        "latency_s": 0.0,
        "err": "",
    }


def iter_rows(configs, split, limit_per_config):
    for config_name in configs:
        ds = load_dataset("galileo-ai/ragbench", config_name, split=split)
        if limit_per_config:
            ds = ds.select(range(min(limit_per_config, len(ds))))
        for row in ds:
            relevant = set(row.get("all_relevant_sentence_keys") or [])
            documents = row.get("documents") or []
            doc_sentences_list = row.get("documents_sentences") or []
            question = row.get("question") or ""
            gold_paper = str(row.get("id", ""))
            seen = set()
            for doc_idx, (doc, doc_sents) in enumerate(zip(documents, doc_sentences_list)):
                if doc in seen:
                    continue
                seen.add(doc)
                out = build_row(
                    question=question,
                    document=doc,
                    doc_idx=doc_idx,
                    doc_sentences=doc_sents,
                    relevant_keys=relevant,
                    gold_paper=gold_paper,
                    config_name=config_name,
                )
                if out is not None:
                    yield config_name, out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-file", type=Path, required=True)
    parser.add_argument("--configs", nargs="*", default=list(ALL_CONFIGS))
    parser.add_argument("--split", default="train")
    parser.add_argument(
        "--limit-per-config", type=int, default=None,
        help="Cap rows per config for smoke testing",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print stats without writing",
    )
    args = parser.parse_args()

    unknown = set(args.configs) - set(ALL_CONFIGS)
    if unknown:
        raise SystemExit(f"unknown config(s): {sorted(unknown)}")

    stats: Counter[str] = Counter()
    positive_stats: Counter[str] = Counter()
    rows_buffer: list[dict] = []

    for config_name, row in iter_rows(args.configs, args.split, args.limit_per_config):
        stats[config_name] += 1
        if row["label"] == 1:
            positive_stats[config_name] += 1
        rows_buffer.append(row)

    print("=== per-config counts ===")
    for c in args.configs:
        total = stats[c]
        pos = positive_stats[c]
        print(f"  {c:<12} rows={total:>6} positive={pos:>6} ({100*pos/max(1,total):5.1f}%)")
    print(f"  {'TOTAL':<12} rows={sum(stats.values()):>6} positive={sum(positive_stats.values()):>6}")

    if args.dry_run:
        return

    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    with args.output_file.open("w") as f:
        for row in rows_buffer:
            f.write(json.dumps(row) + "\n")
    print(f"wrote {len(rows_buffer)} rows to {args.output_file}")


if __name__ == "__main__":
    main()
