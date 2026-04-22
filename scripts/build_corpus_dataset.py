"""Build and push the ACL-Verbatim corpus dataset to Hugging Face.

Two configs in one repo:
  - metadata: one row per paper from paper_data.json (full bib record + has_markdown)
  - fulltext: one row per markdown file (anthology_id + markdown)

Join key: anthology_id (e.g. "2023.acl-long.42"), derived from the paper URL
and from the markdown filename stem.

Usage:
    # Smoke test locally (no upload):
    python scripts/build_corpus_dataset.py \
        --paper-data paper_data.json --md-dir acl_md --limit 100 --dry-run

    # Real push:
    python scripts/build_corpus_dataset.py \
        --paper-data paper_data.json --md-dir acl_md \
        --repo-id KRLabsOrg/acl-anthology-md
"""

import argparse
import json
from pathlib import Path

from datasets import Dataset, Features, Sequence, Value
from huggingface_hub import HfApi
from tqdm import tqdm


_PERSON = {
    "id": Value("string"),
    "first": Value("string"),
    "last": Value("string"),
    "full": Value("string"),
}

METADATA_FEATURES = Features(
    {
        "anthology_id": Value("string"),
        "paper_id": Value("string"),
        "bibkey": Value("string"),
        "bibtype": Value("string"),
        "ingest_date": Value("string"),
        "title": Value("string"),
        "title_html": Value("string"),
        "title_raw": Value("string"),
        "url": Value("string"),
        "pdf": Value("string"),
        "thumbnail": Value("string"),
        "doi": Value("string"),
        "citation": Value("string"),
        "citation_acl": Value("string"),
        "bibtex": Value("string"),
        "author": Sequence(_PERSON),
        "author_string": Value("string"),
        "editor": Sequence(_PERSON),
        "booktitle": Value("string"),
        "parent_volume_id": Value("string"),
        "year": Value("string"),
        "venue": Sequence(Value("string")),
        "pages": Value("string"),
        "page_first": Value("string"),
        "page_last": Value("string"),
        "abstract_html": Value("string"),
        "abstract_raw": Value("string"),
        "language": Value("string"),
        "attachment": Sequence(
            {
                "filename": Value("string"),
                "type": Value("string"),
                "url": Value("string"),
            }
        ),
        "has_markdown": Value("bool"),
    }
)

FULLTEXT_FEATURES = Features(
    {
        "anthology_id": Value("string"),
        "markdown": Value("string"),
    }
)


def anthology_id_from_url(url: str) -> str:
    return url.rstrip("/").split("/")[-1] if url else ""


def index_markdown_dir(md_dir: Path) -> dict[str, Path]:
    index: dict[str, Path] = {}
    for path in md_dir.rglob("*.md"):
        index.setdefault(path.stem, path)
    return index


def normalize_metadata(r: dict, has_markdown: bool) -> dict:
    anth_id = anthology_id_from_url(r.get("url", ""))
    venue = r.get("venue")
    if isinstance(venue, str):
        venue = [venue]
    out = {k: r.get(k) for k in METADATA_FEATURES}
    out["anthology_id"] = anth_id
    out["venue"] = venue
    out["has_markdown"] = has_markdown
    return out


def metadata_rows(paper_data_path: Path, md_index: dict[str, Path], limit: int | None):
    with open(paper_data_path) as f:
        for i, line in enumerate(tqdm(f, desc="metadata")):
            if limit is not None and i >= limit:
                break
            r = json.loads(line)
            anth_id = anthology_id_from_url(r.get("url", ""))
            yield normalize_metadata(r, has_markdown=anth_id in md_index)


def fulltext_rows(md_index: dict[str, Path], limit: int | None):
    items = sorted(md_index.items())
    if limit is not None:
        items = items[:limit]
    for anth_id, path in tqdm(items, desc="fulltext"):
        yield {
            "anthology_id": anth_id,
            "markdown": path.read_text(encoding="utf-8", errors="replace"),
        }


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--paper-data", type=Path, required=True)
    p.add_argument("--md-dir", type=Path, required=True)
    p.add_argument(
        "--repo-id",
        default=None,
        help="HF repo to push to (e.g. KRLabsOrg/acl-anthology-md)",
    )
    p.add_argument("--max-shard-size", default="500MB")
    p.add_argument("--limit", type=int, default=None, help="Cap rows for smoke testing")
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Build datasets in memory but skip upload",
    )
    p.add_argument("--only", choices=["metadata", "fulltext"], default=None)
    p.add_argument(
        "--readme",
        type=Path,
        default=None,
        help="Path to dataset card README.md to upload",
    )
    args = p.parse_args()

    print(f"indexing markdown under {args.md_dir} ...")
    md_index = index_markdown_dir(args.md_dir)
    print(f"found {len(md_index)} markdown files")

    if args.only != "fulltext":
        meta_ds = Dataset.from_generator(
            lambda: metadata_rows(args.paper_data, md_index, args.limit),
            features=METADATA_FEATURES,
        )
        print(f"metadata: {len(meta_ds)} rows, columns={meta_ds.column_names}")
        if args.repo_id and not args.dry_run:
            meta_ds.push_to_hub(
                args.repo_id, config_name="metadata", max_shard_size=args.max_shard_size
            )

    if args.only != "metadata":
        full_ds = Dataset.from_generator(
            lambda: fulltext_rows(md_index, args.limit),
            features=FULLTEXT_FEATURES,
        )
        print(f"fulltext: {len(full_ds)} rows")
        if args.repo_id and not args.dry_run:
            full_ds.push_to_hub(
                args.repo_id, config_name="fulltext", max_shard_size=args.max_shard_size
            )

    if args.readme and args.repo_id and not args.dry_run:
        print(f"uploading README from {args.readme}")
        HfApi().upload_file(
            path_or_fileobj=str(args.readme),
            path_in_repo="README.md",
            repo_id=args.repo_id,
            repo_type="dataset",
        )


if __name__ == "__main__":
    main()
