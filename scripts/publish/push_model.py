"""Inject custom code into a trained token-classifier dir and push to HF.

Takes a directory produced by `train_token_cls.py` (a standard HF
AutoModelForTokenClassification bundle), injects the `auto_map` entry that
makes `AutoModel.from_pretrained(..., trust_remote_code=True)` return the
`AclVerbatimHighlighter` class, copies the custom modeling file and the model
card into the dir, and uploads the whole thing to the hub.

Usage:
    python scripts/publish/push_model.py \\
        --model-dir runs/models/acl-verbatim-modernbert \\
        --repo-id KRLabsOrg/acl-verbatim-modernbert
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from huggingface_hub import HfApi

README_FILE = "README.md"


def get_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--private", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--custom-code-dir",
        type=Path,
        required=True,
        help="Directory containing the modeling .py and README.md to upload",
    )
    parser.add_argument(
        "--modeling-file",
        required=True,
        help="Modeling .py filename inside --custom-code-dir (e.g. modeling_verbatim_rag.py)",
    )
    parser.add_argument(
        "--custom-class",
        required=True,
        help="Dotted module.Class for auto_map (e.g. modeling_verbatim_rag.VerbatimRagHighlighter)",
    )
    return parser.parse_args()


def inject_auto_map(config_path: Path, custom_class: str) -> None:
    config = json.loads(config_path.read_text())
    auto_map = config.get("auto_map") or {}
    auto_map["AutoModel"] = custom_class
    auto_map["AutoModelForTokenClassification"] = custom_class
    config["auto_map"] = auto_map
    config_path.write_text(json.dumps(config, indent=2) + "\n")


def copy_custom_files(
    model_dir: Path, custom_code_dir: Path, modeling_file: str
) -> None:
    for name in (modeling_file, README_FILE):
        src = custom_code_dir / name
        if not src.exists():
            raise SystemExit(f"missing custom file: {src}")
        shutil.copy2(src, model_dir / name)


def main():
    args = get_args()
    model_dir = args.model_dir.resolve()
    config_path = model_dir / "config.json"
    if not config_path.exists():
        raise SystemExit(f"no config.json in {model_dir}")

    inject_auto_map(config_path, args.custom_class)
    copy_custom_files(model_dir, args.custom_code_dir, args.modeling_file)
    print(f"prepared {model_dir} for upload to {args.repo_id}")

    if args.dry_run:
        return

    api = HfApi()
    api.create_repo(
        repo_id=args.repo_id,
        repo_type="model",
        private=args.private,
        exist_ok=True,
    )
    api.upload_folder(
        folder_path=str(model_dir),
        repo_id=args.repo_id,
        repo_type="model",
        ignore_patterns=[
            "checkpoint-*/**",
            "runs/**",
            "training_args.bin",
            "optimizer.pt",
            "scheduler.pt",
            "rng_state*.pth",
            "trainer_state.json",
        ],
    )
    print(f"uploaded to https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()
