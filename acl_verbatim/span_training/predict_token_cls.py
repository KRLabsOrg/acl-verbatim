import argparse

from acl_verbatim.training.token_cls import predict_token_spans


def get_args():
    parser = argparse.ArgumentParser(
        description="Run token classification model to predict spans"
    )
    parser.add_argument("--input-file", required=True, help="Span pairs JSONL")
    parser.add_argument("--output-file", required=True, help="Predictions JSONL")
    parser.add_argument(
        "--model-dir",
        required=True,
        help="Trained token classification model dir/name",
    )
    parser.add_argument("--max-length", type=int, default=8192)
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
    predict_token_spans(
        input_file=args.input_file,
        output_file=args.output_file,
        model_dir=args.model_dir,
        max_length=args.max_length,
        batch_size=args.batch_size,
        doc_stride=args.doc_stride,
    )


if __name__ == "__main__":
    main()
