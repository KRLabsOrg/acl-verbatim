import argparse
import json
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm

from acl_verbatim.data.spans import Span, SpanRow, load_gold_rows
from acl_verbatim.eval.span_metrics import evaluate_rows_against_predictions
from acl_verbatim.training.token_cls import (
    merge_char_spans,
    predict_token_records,
    spans_from_preds,
    tokenize_row_to_windows,
)


def get_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a token classification span extractor on the gold test set"
    )
    parser.add_argument(
        "--gold-file", default=None, help="Gold JSON/JSONL benchmark file"
    )
    parser.add_argument(
        "--hf-dataset",
        default=None,
        help="Optional HF dataset repo id, e.g. KRLabsOrg/acl-verbatim-spans",
    )
    parser.add_argument(
        "--hf-config",
        default="canonical",
        help="HF dataset config to load when --hf-dataset is set",
    )
    parser.add_argument(
        "--gold-split",
        default="test",
        help="HF split name containing gold rows when --hf-dataset is set",
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
    parser.add_argument("--max-length", type=int, default=8192)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument(
        "--doc-stride",
        type=int,
        default=256,
        help="Stride for sliding windows over long chunks",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help=(
            "Positive-class probability threshold for token decisions. "
            "If unset, uses argmax (standard). Lower values trade precision for recall."
        ),
    )
    parser.add_argument(
        "--min-span-chars",
        type=int,
        default=0,
        help="Drop predicted spans shorter than this many characters (post-processing).",
    )
    parser.add_argument(
        "--merge-gap-chars",
        type=int,
        default=0,
        help="Merge adjacent predicted spans separated by <= this many characters.",
    )
    return parser.parse_args()


def postprocess_spans(
    spans: list[dict], min_span_chars: int, merge_gap_chars: int
) -> list[dict]:
    """Drop tiny noise spans, then merge neighbours within a gap threshold."""
    if not spans:
        return spans
    kept = [s for s in spans if s["end"] - s["start"] >= min_span_chars]
    if not kept:
        return kept
    kept.sort(key=lambda s: (s["start"], s["end"]))
    merged = [dict(kept[0])]
    for sp in kept[1:]:
        last = merged[-1]
        if sp["start"] - last["end"] <= merge_gap_chars:
            last["end"] = max(last["end"], sp["end"])
        else:
            merged.append(dict(sp))
    return merged


def predict_with_threshold(
    rows: list[dict],
    model_dir: str,
    max_length: int,
    batch_size: int,
    doc_stride: int,
    threshold: float,
    min_span_chars: int = 0,
    merge_gap_chars: int = 0,
) -> list[dict]:
    """Inference path that applies a configurable positive-probability threshold.

    Kept local to the eval script — see project memory 'Keep eval-only knobs
    out of the training library'.
    """
    import torch
    from transformers import AutoModelForTokenClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
    model = AutoModelForTokenClassification.from_pretrained(model_dir)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    model.to(device).eval()

    predictions = []
    for row in tqdm(rows):
        question = row.get("question")
        chunk = row.get("chunk")
        if not question or chunk is None:
            continue

        enc = tokenize_row_to_windows(
            tokenizer, question, chunk, max_length, doc_stride
        )
        num_windows = len(enc["input_ids"])
        all_spans = []
        for start in range(0, num_windows, batch_size):
            end = min(num_windows, start + batch_size)
            batch = tokenizer.pad(
                {
                    "input_ids": enc["input_ids"][start:end],
                    "attention_mask": enc["attention_mask"][start:end],
                },
                padding=True,
                return_tensors="pt",
            )
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            with torch.no_grad():
                logits = model(
                    input_ids=input_ids, attention_mask=attention_mask
                ).logits
            positive = torch.softmax(logits, dim=-1)[..., 1:].sum(dim=-1)
            preds = (positive >= threshold).long().cpu().tolist()
            for offset, pred in enumerate(preds):
                window_idx = start + offset
                seq_ids = enc.sequence_ids(window_idx)
                offsets = enc["offset_mapping"][window_idx]
                all_spans.extend(spans_from_preds(chunk, offsets, seq_ids, pred))
        merged = merge_char_spans(all_spans)
        merged = postprocess_spans(merged, min_span_chars, merge_gap_chars)
        predictions.append(
            {
                "question": question,
                "paper_id": row.get("paper_id"),
                "chunk_index": row.get("chunk_index"),
                "pred_spans": [
                    {
                        "start": sp["start"],
                        "end": sp["end"],
                        "text": chunk[sp["start"] : sp["end"]],
                    }
                    for sp in merged
                ],
            }
        )
    return predictions


def load_gold_rows_from_hf(repo_id: str, config: str, split: str) -> list[SpanRow]:
    dataset = load_dataset(repo_id, config, split=split)
    rows: list[SpanRow] = []
    for row in dataset:
        spans_raw = row.get("spans") or {}
        if isinstance(spans_raw, dict):
            starts = spans_raw.get("start") or []
            ends = spans_raw.get("end") or []
            texts = spans_raw.get("text") or []
            spans = [
                Span(start=int(start), end=int(end), text=str(text))
                for start, end, text in zip(starts, ends, texts)
            ]
        else:
            spans = [
                Span(
                    start=int(span["start"]),
                    end=int(span["end"]),
                    text=str(span["text"]),
                )
                for span in spans_raw
            ]
        rows.append(
            SpanRow(
                query=str(row["question"]),
                paper_id=str(row["paper_id"]),
                chunk_index=int(row["chunk_index"]),
                chunk=str(row["chunk"]),
                relevance_label="r" if int(row.get("label", 0)) == 1 else "n",
                is_relevant=int(row.get("label", 0)) == 1,
                gold_spans=spans,
                gold_spans_raw=[span.text for span in spans],
                gold_paper_id=row.get("gold_paper"),
                gold_chunk_index=row.get("gold_chunk"),
                retrieval_rank=row.get("retrieval_rank"),
                source=str(row.get("source", "gold")),
            )
        )
    return rows


def main():
    args = get_args()
    if args.hf_dataset:
        if args.gold_file:
            raise SystemExit("Use either --gold-file or --hf-dataset, not both.")
        if args.hf_config == "encoder":
            raise SystemExit(
                "The 'encoder' config is pretokenized and has no raw chunks; "
                "evaluate against --hf-config canonical --gold-split test instead."
            )
        gold_rows = load_gold_rows_from_hf(
            repo_id=args.hf_dataset,
            config=args.hf_config,
            split=args.gold_split,
        )
    elif args.gold_file:
        gold_rows = list(load_gold_rows(Path(args.gold_file)))
    else:
        raise SystemExit("Provide --gold-file or use --hf-dataset with a gold split.")

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

    if (
        args.threshold is None
        and args.min_span_chars == 0
        and args.merge_gap_chars == 0
    ):
        predictions = predict_token_records(
            rows=model_rows,
            model_dir=args.model_dir,
            max_length=args.max_length,
            batch_size=args.batch_size,
            doc_stride=args.doc_stride,
        )
    else:
        predictions = predict_with_threshold(
            rows=model_rows,
            model_dir=args.model_dir,
            max_length=args.max_length,
            batch_size=args.batch_size,
            doc_stride=args.doc_stride,
            threshold=args.threshold if args.threshold is not None else 0.5,
            min_span_chars=args.min_span_chars,
            merge_gap_chars=args.merge_gap_chars,
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
