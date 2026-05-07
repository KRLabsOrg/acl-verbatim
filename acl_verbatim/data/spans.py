"""Canonical schema + loader for ACL-Verbatim span data.

One row per (query, retrieved-chunk) pair. Shared by:
  - the extractor evaluation harness (reads gold rows)
  - the silver annotation pipeline (writes rows with `gold_spans` populated from LLM output)
  - the HF dataset builder (pushes rows as-is)
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, Iterator


@dataclass
class Span:
    start: int
    end: int
    text: str


@dataclass
class SpanRow:
    query: str
    paper_id: str
    chunk_index: int
    chunk: str
    relevance_label: str
    is_relevant: bool
    gold_spans: list[Span] = field(default_factory=list)
    gold_spans_raw: list[str] = field(default_factory=list)
    gold_paper_id: str | None = None
    gold_chunk_index: int | None = None
    retrieval_rank: int | None = None
    baseline_extraction: list[Span] = field(default_factory=list)
    source: str = "gold"

    def to_dict(self) -> dict:
        d = asdict(self)
        return d


_WS = re.compile(r"\s+")


def _align_span(chunk: str, span_text: str) -> Span | None:
    """Recover (start, end) offsets for a gold span given only its text.

    Strategies, in order: exact substring, case-insensitive substring,
    whitespace-normalized case-insensitive. The annotator occasionally
    uppercased spans when copy-pasting from the baseline extraction column —
    we treat casing as noise and always return the span in the chunk's
    original casing.
    """
    idx = chunk.find(span_text)
    if idx != -1:
        return Span(start=idx, end=idx + len(span_text), text=span_text)

    idx_ci = chunk.lower().find(span_text.lower())
    if idx_ci != -1:
        end_ci = idx_ci + len(span_text)
        return Span(start=idx_ci, end=end_ci, text=chunk[idx_ci:end_ci])

    # whitespace-tolerant, case-insensitive: collapse runs of whitespace in both sides
    chunk_norm = _WS.sub(" ", chunk).lower()
    span_norm = _WS.sub(" ", span_text).strip().lower()
    idx_norm = chunk_norm.find(span_norm)
    if idx_norm == -1:
        return None

    # map normalized index back to raw chunk by walking characters
    raw_i = 0
    norm_i = 0
    start_raw = None
    end_raw = None
    span_norm_end = idx_norm + len(span_norm)
    while raw_i < len(chunk) and norm_i <= span_norm_end:
        if norm_i == idx_norm and start_raw is None:
            start_raw = raw_i
        if norm_i == span_norm_end:
            end_raw = raw_i
            break
        c = chunk[raw_i]
        if c.isspace():
            if raw_i + 1 < len(chunk) and chunk[raw_i + 1].isspace():
                raw_i += 1
                continue
            norm_i += 1
        else:
            norm_i += 1
        raw_i += 1
    if start_raw is None:
        return None
    if end_raw is None:
        end_raw = len(chunk)
    return Span(start=start_raw, end=end_raw, text=chunk[start_raw:end_raw])


def load_gold_rows(path: Path | str) -> Iterator[SpanRow]:
    """Yield canonical SpanRow objects from the annotated results JSONL."""
    path = Path(path)
    with open(path) as f:
        for line in f:
            query_record = json.loads(line)
            query = query_record["query"]
            gold_paper_id = query_record.get("gold_paper")
            gold_chunk_index = query_record.get("gold_chunk")

            for rank, result in enumerate(query_record.get("results", []), start=1):
                chunk = result.get("chunk", "")
                relevance_label = result.get("relevance_label", "")
                is_relevant = relevance_label == "r"

                gold_raw = list(result.get("gold_extraction", []) or [])
                aligned = []
                for span_text in gold_raw:
                    span = _align_span(chunk, span_text)
                    if span is not None:
                        aligned.append(span)

                baseline = [
                    Span(start=int(e["start"]), end=int(e["end"]), text=e["text"])
                    for e in (result.get("extraction") or [])
                    if isinstance(e, dict) and {"start", "end", "text"} <= e.keys()
                ]

                yield SpanRow(
                    query=query,
                    paper_id=result.get("document_id", ""),
                    chunk_index=int(result.get("chunk_number", -1)),
                    chunk=chunk,
                    relevance_label=relevance_label,
                    is_relevant=is_relevant,
                    gold_spans=aligned,
                    gold_spans_raw=gold_raw,
                    gold_paper_id=gold_paper_id,
                    gold_chunk_index=int(gold_chunk_index)
                    if gold_chunk_index is not None
                    else None,
                    retrieval_rank=rank,
                    baseline_extraction=baseline,
                    source="gold",
                )


def summarize(rows: Iterable[SpanRow]) -> dict:
    rows = list(rows)
    n_rel = sum(1 for r in rows if r.is_relevant)
    n_with_spans = sum(1 for r in rows if r.gold_spans)
    n_gold_strings = sum(len(r.gold_spans_raw) for r in rows)
    n_aligned = sum(len(r.gold_spans) for r in rows)
    labels: dict[str, int] = {}
    for r in rows:
        labels[r.relevance_label] = labels.get(r.relevance_label, 0) + 1
    return {
        "total_rows": len(rows),
        "relevant": n_rel,
        "rows_with_aligned_spans": n_with_spans,
        "gold_strings_total": n_gold_strings,
        "gold_strings_aligned": n_aligned,
        "relevance_labels": labels,
    }
