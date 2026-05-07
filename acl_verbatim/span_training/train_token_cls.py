import argparse

from acl_verbatim.training.token_cls import train_token_classifier


def get_args():
    parser = argparse.ArgumentParser(
        description="Train a token classification model for span extraction"
    )
    parser.add_argument("--train-file", default=None, help="Train JSONL")
    parser.add_argument("--eval-file", default=None, help="Eval JSONL")
    parser.add_argument(
        "--hf-dataset",
        default=None,
        help="Optional HF dataset repo id, e.g. KRLabsOrg/acl-verbatim-spans",
    )
    parser.add_argument(
        "--hf-config",
        default="encoder",
        help="HF dataset config to load when --hf-dataset is set",
    )
    parser.add_argument(
        "--train-split",
        default="train",
        help="HF train split name when --hf-dataset is set",
    )
    parser.add_argument(
        "--eval-split",
        default="validation",
        help="HF eval split name when --hf-dataset is set",
    )
    parser.add_argument(
        "--model",
        default="answerdotai/ModernBERT-base",
        help="HF model name",
    )
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument(
        "--label-scheme",
        choices=["bio", "binary"],
        default="binary",
        help="Label scheme used in dataset",
    )
    return parser.parse_args()


def main():
    args = get_args()
    if args.hf_dataset:
        if args.train_file or args.eval_file:
            raise SystemExit(
                "Use either --hf-dataset or --train-file/--eval-file, not both."
            )
    elif not (args.train_file and args.eval_file):
        raise SystemExit(
            "Provide --train-file and --eval-file, or use --hf-dataset with splits."
        )

    train_token_classifier(
        train_file=args.train_file,
        eval_file=args.eval_file,
        hf_dataset=args.hf_dataset,
        hf_config=args.hf_config,
        train_split=args.train_split,
        eval_split=args.eval_split,
        model_name=args.model,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        lr=args.lr,
        epochs=args.epochs,
        seed=args.seed,
        label_scheme=args.label_scheme,
    )


if __name__ == "__main__":
    main()
