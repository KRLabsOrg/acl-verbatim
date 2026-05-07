import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any


def load_results(results_file: Path) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    with results_file.open() as f:
        for line in f:
            results.append(json.loads(line))
    return results


def save_results(results: Iterator[dict[str, Any]], output_file: Path) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w") as of:
        for res in results:
            of.write(json.dumps(res) + "\n")
