import argparse
import json
from pathlib import Path

from tqdm import tqdm
from transformers import AutoTokenizer
from acl_verbatim.core.jsonl import iter_jsonl
from acl_verbatim.core.rows import span_prediction_key


def get_args():
    parser = argparse.ArgumentParser(
        description="Evaluate span predictions with token-level F1"
    )
    parser.add_argument("--gold-file", required=True, help="Gold span pairs JSONL")
    parser.add_argument("--pred-file", required=True, help="Predictions JSONL")
    parser.add_argument(
        "--tokenizer",
        default="answerdotai/ModernBERT-base",
        help="HF tokenizer name",
    )
    parser.add_argument("--max-length", type=int, default=8192)
    parser.add_argument(
        "--missing-as-empty",
        action="store_true",
        help="Treat missing predictions as empty spans",
    )
    return parser.parse_args()


def make_key(row):
    return span_prediction_key(row)


def spans_to_labels(enc, spans):
    labels = []
    seq_ids = enc.sequence_ids()
    offsets = enc["offset_mapping"]
    for seq_id, (start, end) in zip(seq_ids, offsets):
        if seq_id != 1 or start == end:
            labels.append(None)
            continue
        label = 0
        for span in spans:
            s = int(span["start"])
            e = int(span["end"])
            if end <= s or start >= e:
                continue
            label = 1
            break
        labels.append(label)
    return labels


def main():
    args = get_args()
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)

    pred_map = {}
    for row in iter_jsonl(Path(args.pred_file)):
        pred_map[make_key(row)] = row.get("pred_spans", [])

    tp = fp = fn = 0
    total = 0
    for row in tqdm(list(iter_jsonl(Path(args.gold_file)))):
        question = row.get("question")
        chunk = row.get("chunk")
        if not question or chunk is None:
            continue
        key = make_key(row)
        if key not in pred_map:
            if not args.missing_as_empty:
                continue
            pred_spans = []
        else:
            pred_spans = pred_map[key]

        enc = tokenizer(
            question,
            chunk,
            return_offsets_mapping=True,
            max_length=args.max_length,
            truncation=True,
        )
        gold_labels = spans_to_labels(enc, row.get("spans", []))
        pred_labels = spans_to_labels(enc, pred_spans)

        for g, p in zip(gold_labels, pred_labels):
            if g is None:
                continue
            total += 1
            if g == 1 and p == 1:
                tp += 1
            elif g == 0 and p == 1:
                fp += 1
            elif g == 1 and p == 0:
                fn += 1

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (
        (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    )

    print(
        json.dumps(
            {
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "total_tokens": total,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
