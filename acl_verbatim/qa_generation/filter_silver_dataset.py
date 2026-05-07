import argparse
import json
from pathlib import Path

from acl_verbatim.core.jsonl import load_jsonl, write_jsonl
from acl_verbatim.synthetic.filtering import SilverFilterConfig, filter_and_split_rows


def get_args():
    parser = argparse.ArgumentParser(
        description="Filter noisy silver span rows and create query-level splits"
    )
    parser.add_argument("--input-file", required=True, help="Raw silver JSONL")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--dev-fraction", type=float, default=0.1)
    parser.add_argument(
        "--min-span-words",
        type=int,
        default=6,
        help="Drop positive rows whose combined spans are shorter than this",
    )
    parser.add_argument(
        "--max-span-words",
        type=int,
        default=220,
        help="Drop positive rows whose combined spans are longer than this",
    )
    parser.add_argument(
        "--max-citation-density",
        type=float,
        default=0.12,
        help="Drop positive rows when citation-like matches per word exceed this",
    )
    parser.add_argument(
        "--max-year-density",
        type=float,
        default=0.08,
        help="Drop positive rows when year-like matches per word exceed this",
    )
    parser.add_argument(
        "--max-positive-rank",
        type=int,
        default=None,
        help="Optional retrieval rank cutoff for positive retrieved rows",
    )
    parser.add_argument(
        "--drop-caption-like",
        action="store_true",
        help="Drop caption/table/figure-like positives as a stricter filtering mode",
    )
    return parser.parse_args()


def main():
    args = get_args()
    rows = load_jsonl(Path(args.input_file))
    config = SilverFilterConfig(
        seed=args.seed,
        dev_fraction=args.dev_fraction,
        min_span_words=args.min_span_words,
        max_span_words=args.max_span_words,
        max_citation_density=args.max_citation_density,
        max_year_density=args.max_year_density,
        max_positive_rank=args.max_positive_rank,
        drop_caption_like=args.drop_caption_like,
    )
    result = filter_and_split_rows(rows, config)

    output_dir = Path(args.output_dir)
    splits_dir = output_dir / "splits"
    write_jsonl(splits_dir / "all_filtered.jsonl", result["all_filtered"])
    write_jsonl(splits_dir / "dropped_positives.jsonl", result["dropped"])
    write_jsonl(splits_dir / "train.jsonl", result["train"])
    write_jsonl(splits_dir / "dev.jsonl", result["dev"])
    (output_dir / "filter_summary.json").write_text(
        json.dumps(result["summary"], indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
