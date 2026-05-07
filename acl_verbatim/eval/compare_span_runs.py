"""Compare multiple span-extraction runs in one table."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tabulate import tabulate

from acl_verbatim.data.spans import load_gold_rows
from acl_verbatim.eval.evaluate_predictions import load_predictions
from acl_verbatim.eval.span_metrics import evaluate_rows_against_predictions


def parse_run(spec: str) -> tuple[str, Path]:
    if "=" not in spec:
        raise argparse.ArgumentTypeError(
            f"Run spec must look like name=path, got: {spec}"
        )
    name, path = spec.split("=", 1)
    if not name:
        raise argparse.ArgumentTypeError(f"Missing run name in spec: {spec}")
    return name, Path(path)


def make_key(pred: dict) -> tuple[str, str, int]:
    query = pred.get("query", pred.get("question", ""))
    return query, pred.get("paper_id", ""), int(pred.get("chunk_index", -1))


def load_or_score(path: Path, rows) -> dict:
    text = path.read_text(encoding="utf-8")
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        payload = None

    if isinstance(payload, dict) and isinstance(payload.get("summary"), dict):
        return payload["summary"]

    predictions = load_predictions(path)
    pred_map = {make_key(pred): pred for pred in predictions}
    return evaluate_rows_against_predictions(rows, pred_map)["summary"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold-file", type=Path, required=True)
    parser.add_argument(
        "--run",
        action="append",
        type=parse_run,
        required=True,
        help="Run spec in the form name=path. May be repeated.",
    )
    args = parser.parse_args()

    rows = list(load_gold_rows(args.gold_file))
    table = []
    for name, path in args.run:
        summary = load_or_score(path, rows)
        table.append(
            [
                name,
                f"{summary['word_level']['micro_precision']:.3f}",
                f"{summary['word_level']['micro_recall']:.3f}",
                f"{summary['word_level']['micro_f1']:.3f}",
                f"{summary['span_level_iou']['0.5']['micro_f1']:.3f}",
                f"{summary['containment']['1.0']['micro_f1']:.3f}",
                f"{summary['gold_coverage_recall']['0.8']:.3f}",
                f"{summary['recall_any_overlap']:.3f}",
                f"{summary['over_prediction_ratio']:.3f}",
                f"{summary['mean_latency_s']:.2f}",
                str(summary["errors"]),
            ]
        )

    print(
        tabulate(
            table,
            headers=[
                "run",
                "word-P",
                "word-R",
                "word-F1",
                "IoU@0.5",
                "Cont@1.0",
                "GoldCov@0.8",
                "AnyOverlap",
                "OverPred",
                "latency",
                "errors",
            ],
        )
    )


if __name__ == "__main__":
    main()
