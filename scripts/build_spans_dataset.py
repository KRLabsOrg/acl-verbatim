"""Build and push the ACL-Verbatim span dataset to Hugging Face.

Configs currently supported:
  - canonical: silver train/dev rows plus gold test rows
  - encoder: token-classification-ready train/dev/test rows

Usage:
    # Dry run
    python scripts/build_spans_dataset.py \
        --silver-train runs/silver_qwen_2000_caption_ok/splits/train.jsonl \
        --silver-dev runs/silver_qwen_2000_caption_ok/splits/dev.jsonl \
        --gold-file 333_20260206_dense_top5_20260305.json \
        --encoder-train runs/silver_qwen_2000_caption_ok/token_cls/train.modernbert.binary.jsonl \
        --encoder-dev runs/silver_qwen_2000_caption_ok/token_cls/dev.modernbert.binary.jsonl \
        --dry-run

    # Push
    python scripts/build_spans_dataset.py \
        --silver-train runs/silver_qwen_2000_caption_ok/splits/train.jsonl \
        --silver-dev runs/silver_qwen_2000_caption_ok/splits/dev.jsonl \
        --gold-file 333_20260206_dense_top5_20260305.json \
        --encoder-train runs/silver_qwen_2000_caption_ok/token_cls/train.modernbert.binary.jsonl \
        --encoder-dev runs/silver_qwen_2000_caption_ok/token_cls/dev.modernbert.binary.jsonl \
        --repo-id KRLabsOrg/acl-verbatim-spans
"""

from __future__ import annotations

import argparse
from pathlib import Path

from datasets import Dataset, DatasetDict, Features, Sequence, Value
from huggingface_hub import HfApi

from acl_verbatim.core.jsonl import load_jsonl
from acl_verbatim.data.spans import load_gold_rows


SPAN_FEATURES = {
    "start": Value("int32"),
    "end": Value("int32"),
    "text": Value("string"),
}

CANONICAL_FEATURES = Features(
    {
        "question": Value("string"),
        "paper_id": Value("string"),
        "chunk_index": Value("int32"),
        "chunk": Value("string"),
        "label": Value("int32"),
        "answerable": Value("bool"),
        "spans": Sequence(SPAN_FEATURES),
        "source": Value("string"),
        "retrieval_rank": Value("int32"),
        "gold_paper": Value("string"),
        "gold_chunk": Value("int32"),
        "predicted_texts": Sequence(Value("string")),
        "latency_s": Value("float32"),
        "err": Value("string"),
    }
)

ENCODER_FEATURES = Features(
    {
        "input_ids": Sequence(Value("int32")),
        "attention_mask": Sequence(Value("int8")),
        "labels": Sequence(Value("int32")),
    }
)


def get_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--silver-train", type=Path, required=True)
    parser.add_argument("--silver-dev", type=Path, required=True)
    parser.add_argument("--gold-file", type=Path, required=True)
    parser.add_argument("--encoder-train", type=Path, default=None)
    parser.add_argument("--encoder-dev", type=Path, default=None)
    parser.add_argument(
        "--encoder-test",
        type=Path,
        default=None,
        help="Optional token-classification-ready gold/test split",
    )
    parser.add_argument("--repo-id", default="KRLabsOrg/acl-verbatim-spans")
    parser.add_argument("--max-shard-size", default="500MB")
    parser.add_argument(
        "--readme",
        type=Path,
        default=Path("dataset_cards/acl-verbatim-spans/README.md"),
        help="Dataset card README to upload",
    )
    parser.add_argument("--private", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def normalize_canonical_row(row: dict) -> dict:
    return {
        "question": str(row.get("question", "")),
        "paper_id": str(row.get("paper_id", "")),
        "chunk_index": int(row.get("chunk_index", -1)),
        "chunk": str(row.get("chunk", "")),
        "label": int(row.get("label", 0)),
        "answerable": bool(row.get("answerable", False)),
        "spans": [
            {
                "start": int(span["start"]),
                "end": int(span["end"]),
                "text": str(span.get("text", "")),
            }
            for span in (row.get("spans") or [])
            if isinstance(span, dict) and {"start", "end"} <= span.keys()
        ],
        "source": str(row.get("source", "")),
        "retrieval_rank": int(row.get("retrieval_rank", -1) or -1),
        "gold_paper": str(row.get("gold_paper", "")),
        "gold_chunk": int(row.get("gold_chunk", -1) or -1),
        "predicted_texts": [str(t) for t in (row.get("predicted_texts") or [])],
        "latency_s": float(row.get("latency_s", 0.0) or 0.0),
        "err": str(row.get("err", "") or ""),
    }


def gold_row_to_canonical(row) -> dict:
    return {
        "question": row.query,
        "paper_id": row.paper_id,
        "chunk_index": row.chunk_index,
        "chunk": row.chunk,
        "label": 1 if row.is_relevant else 0,
        "answerable": bool(row.is_relevant),
        "spans": [
            {"start": span.start, "end": span.end, "text": span.text}
            for span in row.gold_spans
        ],
        "source": "gold",
        "retrieval_rank": int(row.retrieval_rank or -1),
        "gold_paper": str(row.gold_paper_id or ""),
        "gold_chunk": int(row.gold_chunk_index or -1),
        "predicted_texts": [],
        "latency_s": 0.0,
        "err": "",
    }


def build_canonical_dataset(args) -> DatasetDict:
    train_rows = [normalize_canonical_row(row) for row in load_jsonl(args.silver_train)]
    dev_rows = [normalize_canonical_row(row) for row in load_jsonl(args.silver_dev)]
    test_rows = [gold_row_to_canonical(row) for row in load_gold_rows(args.gold_file)]
    return DatasetDict(
        {
            "train": Dataset.from_list(train_rows, features=CANONICAL_FEATURES),
            "validation": Dataset.from_list(dev_rows, features=CANONICAL_FEATURES),
            "test": Dataset.from_list(test_rows, features=CANONICAL_FEATURES),
        }
    )


def build_encoder_dataset(args) -> DatasetDict | None:
    if not args.encoder_train or not args.encoder_dev:
        return None

    splits = {
        "train": Dataset.from_list(load_jsonl(args.encoder_train), features=ENCODER_FEATURES),
        "validation": Dataset.from_list(load_jsonl(args.encoder_dev), features=ENCODER_FEATURES),
    }
    if args.encoder_test:
        splits["test"] = Dataset.from_list(
            load_jsonl(args.encoder_test), features=ENCODER_FEATURES
        )
    return DatasetDict(splits)


def push_dataset_dict(ds: DatasetDict, repo_id: str, config_name: str, max_shard_size: str):
    ds.push_to_hub(
        repo_id,
        config_name=config_name,
        max_shard_size=max_shard_size,
    )


def main():
    args = get_args()
    canonical = build_canonical_dataset(args)
    encoder = build_encoder_dataset(args)

    print(
        {
            "canonical": {split: len(ds) for split, ds in canonical.items()},
            "encoder": None
            if encoder is None
            else {split: len(ds) for split, ds in encoder.items()},
            "repo_id": args.repo_id,
        }
    )

    if args.dry_run:
        return

    api = HfApi()
    api.create_repo(
        repo_id=args.repo_id,
        repo_type="dataset",
        private=args.private,
        exist_ok=True,
    )
    push_dataset_dict(canonical, args.repo_id, "canonical", args.max_shard_size)
    if encoder is not None:
        push_dataset_dict(encoder, args.repo_id, "encoder", args.max_shard_size)
    if args.readme and args.readme.exists():
        api.upload_file(
            path_or_fileobj=str(args.readme),
            path_in_repo="README.md",
            repo_id=args.repo_id,
            repo_type="dataset",
        )


if __name__ == "__main__":
    main()
