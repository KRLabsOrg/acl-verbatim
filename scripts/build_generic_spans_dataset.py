"""Build and push `KRLabsOrg/verbatim-spans` — multi-domain extractive spans.

Combines three sources:
  - ACL silver (NLP papers, paragraph-scale, Qwen-paragraph teacher)
  - RAGBench capped (12 QA domains, sentence-scale, GPT-4o gold)
  - Squeez (tool-output extraction on SWE-bench, code-scale, GLiNER gold)

Configs pushed:
  - canonical: raw (question, chunk, spans) rows with a `source_dataset` field
  - encoder:   pretokenized ModernBERT 8192 rows (ready for train_token_cls.py)

Usage:
    # Dry-run (prints split sizes, skips upload)
    python scripts/build_generic_spans_dataset.py --dry-run

    # Push
    python scripts/build_generic_spans_dataset.py --repo-id KRLabsOrg/verbatim-spans
"""

from __future__ import annotations

import argparse
from pathlib import Path

from datasets import Dataset, DatasetDict, Features, Sequence, Value
from huggingface_hub import HfApi

from acl_verbatim.core.jsonl import iter_jsonl

ACL_TRAIN = Path("runs/silver_qwen_2000_caption_ok/splits/train.jsonl")
ACL_DEV = Path("runs/silver_qwen_2000_caption_ok/splits/dev.jsonl")
ACL_ENC_TRAIN = Path(
    "runs/silver_qwen_2000_caption_ok/token_cls/train.modernbert.binary.8k.jsonl"
)
ACL_ENC_DEV = Path(
    "runs/silver_qwen_2000_caption_ok/token_cls/dev.modernbert.binary.8k.jsonl"
)

RAGBENCH_TRAIN = Path("runs/ragbench/train.capped.jsonl")
RAGBENCH_DEV = Path("runs/ragbench/val.capped.jsonl")
RAGBENCH_ENC_TRAIN = Path("runs/ragbench/token_cls/train.modernbert.binary.8k.jsonl")
RAGBENCH_ENC_DEV = Path("runs/ragbench/token_cls/dev.modernbert.binary.8k.jsonl")

SQUEEZ_TRAIN = Path("runs/squeez/train.jsonl")
SQUEEZ_DEV = Path("runs/squeez/val.jsonl")
SQUEEZ_ENC_TRAIN = Path("runs/squeez/token_cls/train.modernbert.binary.8k.jsonl")
SQUEEZ_ENC_DEV = Path("runs/squeez/token_cls/dev.modernbert.binary.8k.jsonl")


SPAN_FEATURES = {
    "start": Value("int32"),
    "end": Value("int32"),
    "text": Value("string"),
}

CANONICAL_FEATURES = Features(
    {
        "source_dataset": Value("string"),
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


def normalize_canonical_row(row: dict, source: str) -> dict:
    return {
        "source_dataset": source,
        "question": str(row.get("question", "")),
        "paper_id": str(row.get("paper_id", "")),
        "chunk_index": int(row.get("chunk_index", -1) or -1),
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


def load_canonical(paths: list[tuple[str, Path]]) -> list[dict]:
    rows = []
    for source, path in paths:
        for row in iter_jsonl(path):
            rows.append(normalize_canonical_row(row, source))
    return rows


def load_encoder(paths: list[Path]) -> list[dict]:
    rows = []
    for path in paths:
        rows.extend(iter_jsonl(path))
    return rows


def build_canonical_dataset() -> DatasetDict:
    train_rows = load_canonical(
        [("acl", ACL_TRAIN), ("ragbench", RAGBENCH_TRAIN), ("squeez", SQUEEZ_TRAIN)]
    )
    dev_rows = load_canonical(
        [("acl", ACL_DEV), ("ragbench", RAGBENCH_DEV), ("squeez", SQUEEZ_DEV)]
    )
    return DatasetDict(
        {
            "train": Dataset.from_list(train_rows, features=CANONICAL_FEATURES),
            "validation": Dataset.from_list(dev_rows, features=CANONICAL_FEATURES),
        }
    )


def build_encoder_dataset() -> DatasetDict:
    train_rows = load_encoder([ACL_ENC_TRAIN, RAGBENCH_ENC_TRAIN, SQUEEZ_ENC_TRAIN])
    dev_rows = load_encoder([ACL_ENC_DEV, RAGBENCH_ENC_DEV, SQUEEZ_ENC_DEV])
    return DatasetDict(
        {
            "train": Dataset.from_list(train_rows, features=ENCODER_FEATURES),
            "validation": Dataset.from_list(dev_rows, features=ENCODER_FEATURES),
        }
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", default="KRLabsOrg/verbatim-spans")
    parser.add_argument("--max-shard-size", default="500MB")
    parser.add_argument(
        "--readme",
        type=Path,
        default=Path("dataset_cards/verbatim-spans/README.md"),
        help="Dataset card README to upload",
    )
    parser.add_argument("--private", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--skip-canonical", action="store_true", help="Only push the encoder config"
    )
    parser.add_argument(
        "--skip-encoder", action="store_true", help="Only push the canonical config"
    )
    args = parser.parse_args()

    canonical = None if args.skip_canonical else build_canonical_dataset()
    encoder = None if args.skip_encoder else build_encoder_dataset()

    print(
        {
            "canonical": None
            if canonical is None
            else {split: len(ds) for split, ds in canonical.items()},
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
    if canonical is not None:
        canonical.push_to_hub(
            args.repo_id, config_name="canonical", max_shard_size=args.max_shard_size
        )
    if encoder is not None:
        encoder.push_to_hub(
            args.repo_id, config_name="encoder", max_shard_size=args.max_shard_size
        )
    if args.readme and args.readme.exists():
        api.upload_file(
            path_or_fileobj=str(args.readme),
            path_in_repo="README.md",
            repo_id=args.repo_id,
            repo_type="dataset",
        )


if __name__ == "__main__":
    main()
