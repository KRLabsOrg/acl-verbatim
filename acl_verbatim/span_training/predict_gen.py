import argparse
import json
from pathlib import Path

from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def get_args():
    parser = argparse.ArgumentParser(
        description="Run generative model to predict spans"
    )
    parser.add_argument("--input-file", required=True, help="Span pairs JSONL")
    parser.add_argument("--output-file", required=True, help="Predictions JSONL")
    parser.add_argument(
        "--model-dir",
        required=True,
        help="Trained generative model dir/name",
    )
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    return parser.parse_args()


def iter_jsonl(path: Path):
    with open(path) as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def make_prompt(question, chunk, max_spans=3):
    return (
        "You are extracting minimal evidence spans from a research excerpt.\n\n"
        "Return ONLY valid JSON with this schema:\n"
        '{ "answerable": true/false, "spans": [{"start": int, "end": int, "text": "..."}] }\n\n'
        f"Rules:\n- At most {max_spans} spans\n- Spans must be exact substrings of the excerpt\n"
        "- If not answerable, return answerable=false and spans=[]\n\n"
        f"Question: {question}\n\nExcerpt:\n{chunk}\n"
    )


def safe_parse_json(text):
    try:
        return json.loads(text)
    except Exception:
        return None


def extract_first_json(text: str):
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    in_str = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_str = False
            continue
        else:
            if ch == '"':
                in_str = True
                continue
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return text[start : i + 1]
    return None


def main():
    args = get_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model_dir)
    model.eval()

    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as of:
        for row in tqdm(list(iter_jsonl(Path(args.input_file)))):
            question = row.get("question")
            chunk = row.get("chunk")
            if not question or chunk is None:
                continue
            prompt = make_prompt(question, chunk)
            enc = tokenizer(
                prompt, return_tensors="pt", truncation=True, max_length=args.max_length
            )
            with torch.no_grad():
                out = model.generate(
                    **enc,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                )
            full = tokenizer.decode(out[0], skip_special_tokens=True)
            pred = full[len(prompt) :]
            extracted = extract_first_json(pred) or ""
            data = safe_parse_json(extracted)
            spans = []
            if data and isinstance(data, dict):
                spans = data.get("spans", [])
            out_row = {
                "question": question,
                "paper_id": row.get("paper_id"),
                "chunk_index": row.get("chunk_index"),
                "pred_spans": spans,
                "raw_output": pred.strip(),
            }
            of.write(json.dumps(out_row) + "\n")


if __name__ == "__main__":
    import torch

    main()
