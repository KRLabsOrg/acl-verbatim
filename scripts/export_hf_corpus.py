"""Export the published ACL-Verbatim corpus dataset from Hugging Face to local files.

This script materializes:
  - paper_data.json  (JSONL metadata file expected by local scripts)
  - acl_md/*.md      (markdown files expected by indexing and QA-generation scripts)

Usage:
    python scripts/export_hf_corpus.py \
        --output-metadata-file paper_data.json \
        --output-md-dir acl_md

    # Smoke test on a small sample
    python scripts/export_hf_corpus.py \
        --output-metadata-file paper_data.json \
        --output-md-dir acl_md \
        --limit 100
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm


def get_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-id",
        default="KRLabsOrg/acl-anthology-md",
        help="HF dataset repo containing metadata and fulltext configs",
    )
    parser.add_argument(
        "--metadata-config",
        default="metadata",
        help="Metadata config name",
    )
    parser.add_argument(
        "--fulltext-config",
        default="fulltext",
        help="Fulltext config name",
    )
    parser.add_argument(
        "--output-metadata-file",
        type=Path,
        required=True,
        help="Local JSONL metadata output path, e.g. paper_data.json",
    )
    parser.add_argument(
        "--output-md-dir",
        type=Path,
        required=True,
        help="Local markdown directory output path, e.g. acl_md",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap for smoke testing",
    )
    parser.add_argument(
        "--only",
        choices=("metadata", "fulltext"),
        default=None,
        help="Export only one side of the corpus",
    )
    return parser.parse_args()


def export_metadata(
    repo_id: str, config_name: str, output_file: Path, limit: int | None
):
    ds = load_dataset(repo_id, config_name, split="train", streaming=True)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8") as f:
        for idx, row in enumerate(tqdm(ds, desc="metadata")):
            if limit is not None and idx >= limit:
                break
            f.write(json.dumps(dict(row), ensure_ascii=False) + "\n")


def export_fulltext(
    repo_id: str, config_name: str, output_dir: Path, limit: int | None
):
    ds = load_dataset(repo_id, config_name, split="train", streaming=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    for idx, row in enumerate(tqdm(ds, desc="fulltext")):
        if limit is not None and idx >= limit:
            break
        anthology_id = row["anthology_id"]
        markdown = row["markdown"]
        (output_dir / f"{anthology_id}.md").write_text(markdown, encoding="utf-8")


def main():
    args = get_args()
    if args.only != "fulltext":
        export_metadata(
            repo_id=args.repo_id,
            config_name=args.metadata_config,
            output_file=args.output_metadata_file,
            limit=args.limit,
        )
    if args.only != "metadata":
        export_fulltext(
            repo_id=args.repo_id,
            config_name=args.fulltext_config,
            output_dir=args.output_md_dir,
            limit=args.limit,
        )


if __name__ == "__main__":
    main()
