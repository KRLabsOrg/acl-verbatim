"""Score our char-span predictions with Squeez's line-level metric harness.

Squeez reports span F1 / fuzzy F1 / exact match / ROUGE-L / compression at the
LINE level (set-overlap on \\n-split lines), not the word/char level our default
harness uses. To compare directly to Squeez's published numbers, we project our
char spans onto the lines they overlap and feed the resulting line lists into
Squeez's own scorer.

Requires the squeez project to be installed/importable. Default path assumes
~/projects/squeez; override with --squeez-path.

Example:
    # 1. Generate our predictions in the standard way
    python acl_verbatim/span_training/evaluate_token_cls.py \\
        --gold-file runs/eval/test_slices/squeez.gold.jsonl \\
        --model-dir runs/models/verbatim-generic-modernbert/ \\
        --threshold 0.2 --min-span-chars 30 --merge-gap-chars 20 \\
        --pred-file runs/eval/generic.squeez_test.preds.jsonl \\
        --output-file runs/eval/generic.squeez_test.json

    # 2. Score those predictions with Squeez's metrics
    python scripts/eval_squeez_metrics.py \\
        --gold-file runs/eval/test_slices/squeez.gold.jsonl \\
        --pred-file runs/eval/generic.squeez_test.preds.jsonl \\
        --output-file runs/eval/generic.squeez_test.squeez_metrics.json
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path


def lines_with_offsets(text: str) -> list[tuple[int, int, str]]:
    """Split text on \\n; return (start, end, line) per line including blanks."""
    out = []
    pos = 0
    for line in text.split("\n"):
        out.append((pos, pos + len(line), line))
        pos += len(line) + 1
    return out


def spans_to_lines(text: str, spans: list[dict]) -> list[str]:
    """Return the unique non-empty lines touched by any predicted span."""
    if not spans:
        return []
    line_records = lines_with_offsets(text)
    kept: list[str] = []
    seen: set[str] = set()
    for ls, le, line in line_records:
        for sp in spans:
            if not (sp["end"] <= ls or sp["start"] >= le):
                if line and line not in seen:
                    kept.append(line)
                    seen.add(line)
                break
    return kept


def gold_extraction_to_lines(gold_extraction: list[str]) -> list[str]:
    """Squeez gold spans are already line-level texts; flatten and dedupe."""
    out: list[str] = []
    seen: set[str] = set()
    for ent in gold_extraction:
        for line in ent.split("\n"):
            line = line.rstrip()
            if line and line not in seen:
                out.append(line)
                seen.add(line)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gold-file", type=Path, required=True)
    p.add_argument("--pred-file", type=Path, required=True)
    p.add_argument("--output-file", type=Path, default=None)
    p.add_argument(
        "--squeez-path",
        type=Path,
        default=Path.home() / "projects" / "squeez",
        help="Path to the squeez project (for importing its metric functions)",
    )
    args = p.parse_args()

    sys.path.insert(0, str(args.squeez_path))
    from squeez.training.evaluate import (
        compute_compression_ratio,
        compute_empty_accuracy,
        compute_fuzzy_span_metrics,
        compute_partial_overlap,
        compute_rouge_l,
        compute_span_metrics,
    )

    # Load gold: gold-file format with one query_record per line, single nested result
    gold_by_key: dict[tuple[str, str, int], dict] = {}
    with args.gold_file.open() as f:
        for line in f:
            rec = json.loads(line)
            for r in rec.get("results", []):
                if r.get("relevance_label") != "r":
                    continue
                key = (rec["query"], r.get("document_id", ""), int(r.get("chunk_number", 0)))
                gold_by_key[key] = {
                    "chunk": r["chunk"],
                    "gold_extraction": r.get("gold_extraction") or [],
                }

    # Load preds: JSONL with question, paper_id, chunk_index, pred_spans
    pred_by_key: dict[tuple[str, str, int], list[dict]] = {}
    with args.pred_file.open() as f:
        for line in f:
            pr = json.loads(line)
            key = (
                pr.get("question", ""),
                str(pr.get("paper_id", "")),
                int(pr.get("chunk_index", 0)),
            )
            pred_by_key[key] = pr.get("pred_spans") or []

    all_metrics: list[dict] = []
    n_missing_pred = 0
    for key, gold in gold_by_key.items():
        chunk = gold["chunk"]
        ref_lines = gold_extraction_to_lines(gold["gold_extraction"])
        spans = pred_by_key.get(key)
        if spans is None:
            n_missing_pred += 1
            spans = []
        pred_lines = spans_to_lines(chunk, spans)

        span = compute_span_metrics(pred_lines, ref_lines)
        fuzzy = compute_fuzzy_span_metrics(pred_lines, ref_lines, threshold=0.5)
        partial = compute_partial_overlap(pred_lines, ref_lines)
        empty = compute_empty_accuracy(pred_lines, ref_lines)
        pred_text = "\n".join(pred_lines)
        ref_text = "\n".join(ref_lines)
        rouge = compute_rouge_l(pred_text, ref_text)
        compression = compute_compression_ratio(chunk, pred_text)
        all_metrics.append(
            {
                "span_precision": span["precision"],
                "span_recall": span["recall"],
                "span_f1": span["f1"],
                "exact_match": span["exact_match"],
                "fuzzy_span_precision": fuzzy["precision"],
                "fuzzy_span_recall": fuzzy["recall"],
                "fuzzy_span_f1": fuzzy["f1"],
                "partial_overlap": partial,
                "empty_accuracy": empty["correct"],
                "empty_category": empty["category"],
                "rouge_l": rouge,
                "compression": compression,
            }
        )

    keys = [
        "span_precision", "span_recall", "span_f1", "exact_match",
        "fuzzy_span_precision", "fuzzy_span_recall", "fuzzy_span_f1",
        "partial_overlap", "empty_accuracy", "rouge_l", "compression",
    ]
    summary = {
        k: {
            "mean": round(statistics.mean(m[k] for m in all_metrics), 4),
            "median": round(statistics.median(m[k] for m in all_metrics), 4),
        }
        for k in keys
    }
    summary["num_samples"] = len(all_metrics)
    summary["num_missing_predictions"] = n_missing_pred

    print(json.dumps(summary, indent=2))
    if args.output_file:
        args.output_file.parent.mkdir(parents=True, exist_ok=True)
        args.output_file.write_text(json.dumps(summary, indent=2) + "\n")
        print(f"wrote {args.output_file}")


if __name__ == "__main__":
    main()
