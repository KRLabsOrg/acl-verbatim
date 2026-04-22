import argparse
import json
from pathlib import Path

from tqdm import tqdm


def get_args():
    parser = argparse.ArgumentParser(
        description="Prepare generative dataset for span extraction"
    )
    parser.add_argument("--input-file", required=True, help="Span pairs JSONL")
    parser.add_argument("--output-file", required=True, help="Output JSONL path")
    parser.add_argument(
        "--max-spans",
        type=int,
        default=3,
        help="Max spans to include in target output",
    )
    return parser.parse_args()


def make_prompt(question, chunk, max_spans):
    return (
        "You are extracting minimal evidence spans from a research excerpt.\n\n"
        "Return ONLY valid JSON with this schema:\n"
        '{ "answerable": true/false, "spans": [{"start": int, "end": int, "text": "..."}] }\n\n'
        f"Rules:\n- At most {max_spans} spans\n- Spans must be exact substrings of the excerpt\n"
        "- If not answerable, return answerable=false and spans=[]\n\n"
        f"Question: {question}\n\nExcerpt:\n{chunk}\n"
    )


def make_response(answerable, spans, max_spans):
    if not answerable:
        return json.dumps({"answerable": False, "spans": []})
    spans = spans[:max_spans]
    return json.dumps({"answerable": True, "spans": spans})


def iter_jsonl(path: Path):
    with open(path) as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def main():
    args = get_args()
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as of:
        for row in tqdm(list(iter_jsonl(Path(args.input_file)))):
            question = row.get("question")
            chunk = row.get("chunk")
            if not question or chunk is None:
                continue
            answerable = bool(row.get("answerable", False))
            spans = row.get("spans", [])
            prompt = make_prompt(question, chunk, args.max_spans)
            response = make_response(answerable, spans, args.max_spans)
            of.write(json.dumps({"prompt": prompt, "response": response}) + "\n")


if __name__ == "__main__":
    main()
