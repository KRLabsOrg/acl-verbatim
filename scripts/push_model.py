"""Inject custom code into a trained token-classifier dir and push to HF.

Takes a directory produced by `train_token_cls.py` (a standard HF
AutoModelForTokenClassification bundle), injects the `auto_map` entry that
makes `AutoModel.from_pretrained(..., trust_remote_code=True)` return the
`AclVerbatimHighlighter` class, copies the custom modeling file and the model
card into the dir, and uploads the whole thing to the hub.

Usage:
    python scripts/push_model.py \\
        --model-dir runs/models/acl-verbatim-modernbert \\
        --repo-id KRLabsOrg/acl-verbatim-modernbert
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from huggingface_hub import HfApi


CUSTOM_CODE_DIR = Path("model_cards/acl-verbatim-modernbert")
MODELING_FILE = "modeling_acl_verbatim.py"
README_FILE = "README.md"
CUSTOM_CLASS = "modeling_acl_verbatim.AclVerbatimHighlighter"


def get_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--private", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--custom-code-dir",
        type=Path,
        default=CUSTOM_CODE_DIR,
        help="Where to find modeling_acl_verbatim.py and README.md",
    )
    return parser.parse_args()


def inject_auto_map(config_path: Path) -> None:
    config = json.loads(config_path.read_text())
    auto_map = config.get("auto_map") or {}
    auto_map["AutoModel"] = CUSTOM_CLASS
    auto_map["AutoModelForTokenClassification"] = CUSTOM_CLASS
    config["auto_map"] = auto_map
    config_path.write_text(json.dumps(config, indent=2) + "\n")


def copy_custom_files(model_dir: Path, custom_code_dir: Path) -> None:
    for name in (MODELING_FILE, README_FILE):
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

    inject_auto_map(config_path)
    copy_custom_files(model_dir, args.custom_code_dir)
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
    )
    print(f"uploaded to https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()
