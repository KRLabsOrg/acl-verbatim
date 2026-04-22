import argparse

from acl_verbatim.training.token_cls import write_token_cls_dataset


def get_args():
    parser = argparse.ArgumentParser(
        description="Prepare token classification dataset from span pairs JSONL"
    )
    parser.add_argument("--input-file", required=True, help="Span pairs JSONL")
    parser.add_argument("--output-file", required=True, help="Output JSONL path")
    parser.add_argument(
        "--tokenizer",
        default="answerdotai/ModernBERT-base",
        help="HF tokenizer name",
    )
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument(
        "--doc-stride",
        type=int,
        default=256,
        help="Stride for sliding windows over long chunks",
    )
    parser.add_argument(
        "--drop-unlabeled-positives",
        action="store_true",
        help="Drop positives that end up with no labeled tokens",
    )
    parser.add_argument(
        "--label-scheme",
        choices=["bio", "binary"],
        default="binary",
        help="Token label scheme",
    )
    return parser.parse_args()


def main():
    args = get_args()
    write_token_cls_dataset(
        input_file=args.input_file,
        output_file=args.output_file,
        tokenizer_name=args.tokenizer,
        max_length=args.max_length,
        doc_stride=args.doc_stride,
        drop_unlabeled_positives=args.drop_unlabeled_positives,
        label_scheme=args.label_scheme,
    )


if __name__ == "__main__":
    main()
