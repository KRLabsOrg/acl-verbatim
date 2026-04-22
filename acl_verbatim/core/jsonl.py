import json
from pathlib import Path
from typing import Iterable


def iter_jsonl(path: Path | str):
    with open(path) as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def load_jsonl(path: Path | str) -> list[dict]:
    return list(iter_jsonl(path))


def write_jsonl(path: Path | str, rows: Iterable[dict]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
