"""Score arbitrary span predictions against the local gold test set."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from acl_verbatim.data.spans import load_gold_rows
from acl_verbatim.eval.span_metrics import evaluate_rows_against_predictions


def load_predictions(path: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8")
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return [json.loads(line) for line in text.splitlines() if line.strip()]

    if isinstance(payload, list):
        return [p for p in payload if isinstance(p, dict)]
    if isinstance(payload, dict):
        if isinstance(payload.get("records"), list):
            return [p for p in payload["records"] if isinstance(p, dict)]
        return [payload]
    raise SystemExit(f"Unsupported prediction file format: {path}")


def make_key(pred: dict) -> tuple[str, str, int]:
    query = pred.get("query", pred.get("question", ""))
    return query, pred.get("paper_id", ""), int(pred.get("chunk_index", -1))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold-file", type=Path, required=True)
    parser.add_argument("--pred-file", type=Path, required=True)
    parser.add_argument(
        "--output-file", type=Path, default=None, help="Optional detailed output JSON"
    )
    args = parser.parse_args()

    rows = list(load_gold_rows(args.gold_file))
    pred_map = {make_key(pred): pred for pred in load_predictions(args.pred_file)}
    result = evaluate_rows_against_predictions(rows, pred_map)
    print(json.dumps(result["summary"], indent=2))

    if args.output_file:
        args.output_file.parent.mkdir(parents=True, exist_ok=True)
        with args.output_file.open("w") as f:
            json.dump(result, f, indent=2)
        print(f"wrote detailed evaluation to {args.output_file}")


if __name__ == "__main__":
    main()
