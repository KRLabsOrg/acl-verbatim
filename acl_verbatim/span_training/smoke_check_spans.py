import argparse
import json
from pathlib import Path


def get_args():
    parser = argparse.ArgumentParser(description="Quick span sanity checker")
    parser.add_argument("--input-file", required=True, help="Span pairs JSONL")
    parser.add_argument("--max", type=int, default=200, help="Max rows to check")
    return parser.parse_args()


def iter_jsonl(path: Path, max_rows: int):
    with open(path) as f:
        for i, line in enumerate(f):
            if i >= max_rows:
                break
            if line.strip():
                yield json.loads(line)


def main():
    args = get_args()
    total = 0
    bad = 0
    for row in iter_jsonl(Path(args.input_file), args.max):
        total += 1
        chunk = row.get("chunk", "")
        spans = row.get("spans", [])
        for s in spans:
            start = int(s.get("start", -1))
            end = int(s.get("end", -1))
            text = s.get("text", "")
            if start < 0 or end <= start or end > len(chunk):
                bad += 1
                break
            if chunk[start:end] != text:
                bad += 1
                break
    print(json.dumps({"checked": total, "bad_rows": bad}))


if __name__ == "__main__":
    main()
