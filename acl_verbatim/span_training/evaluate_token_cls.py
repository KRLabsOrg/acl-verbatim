import argparse
import json
from pathlib import Path

from acl_verbatim.data.spans import load_gold_rows
from acl_verbatim.eval.span_metrics import evaluate_rows_against_predictions
from acl_verbatim.training.token_cls import predict_token_records


def get_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a token classification span extractor on the gold test set"
    )
    parser.add_argument(
        "--gold-file", required=True, help="Gold JSON/JSONL benchmark file"
    )
    parser.add_argument(
        "--model-dir",
        required=True,
        help="Trained token classification model directory",
    )
    parser.add_argument(
        "--output-file",
        default=None,
        help="Optional detailed JSON evaluation output",
    )
    parser.add_argument(
        "--pred-file",
        default=None,
        help="Optional JSONL file to write normalized predictions",
    )
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument(
        "--doc-stride",
        type=int,
        default=256,
        help="Stride for sliding windows over long chunks",
    )
    return parser.parse_args()


def main():
    args = get_args()
    gold_rows = list(load_gold_rows(Path(args.gold_file)))
    relevant_rows = [row for row in gold_rows if row.is_relevant]
    model_rows = [
        {
            "question": row.query,
            "paper_id": row.paper_id,
            "chunk_index": row.chunk_index,
            "chunk": row.chunk,
        }
        for row in relevant_rows
    ]

    predictions = predict_token_records(
        rows=model_rows,
        model_dir=args.model_dir,
        max_length=args.max_length,
        batch_size=args.batch_size,
        doc_stride=args.doc_stride,
    )
    pred_map = {
        (
            pred["question"],
            pred.get("paper_id", ""),
            int(pred.get("chunk_index", -1)),
        ): pred
        for pred in predictions
    }
    result = evaluate_rows_against_predictions(gold_rows, pred_map)
    print(json.dumps(result["summary"], indent=2))

    if args.pred_file:
        pred_path = Path(args.pred_file)
        pred_path.parent.mkdir(parents=True, exist_ok=True)
        with pred_path.open("w", encoding="utf-8") as f:
            for pred in predictions:
                f.write(json.dumps(pred) + "\n")
        print(f"wrote predictions to {pred_path}")

    if args.output_file:
        output_path = Path(args.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(f"wrote detailed evaluation to {output_path}")


if __name__ == "__main__":
    main()
