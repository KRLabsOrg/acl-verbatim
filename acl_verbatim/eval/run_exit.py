"""Run EXIT sentence selection on the gold set and emit normalized predictions."""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

from tqdm import tqdm

from acl_verbatim.data.spans import load_gold_rows

SENTENCE_RE = re.compile(r".+?(?:[.!?]+(?:\s+|$)|\n{2,}|$)", re.DOTALL)


def split_sentences(text: str) -> list[tuple[int, int, str]]:
    spans: list[tuple[int, int, str]] = []
    for match in SENTENCE_RE.finditer(text):
        raw = match.group(0)
        stripped = raw.strip()
        if not stripped:
            continue
        left_trim = len(raw) - len(raw.lstrip())
        right_trim = len(raw) - len(raw.rstrip())
        start = match.start() + left_trim
        end = match.end() - right_trim
        spans.append((start, end, text[start:end]))
    return spans


def merge_adjacent_spans(
    spans: list[tuple[int, int]], text: str
) -> list[tuple[int, int]]:
    if not spans:
        return []
    merged = [spans[0]]
    for start, end in spans[1:]:
        last_start, last_end = merged[-1]
        if text[last_end:start].strip() == "":
            merged[-1] = (last_start, end)
        else:
            merged.append((start, end))
    return merged


class ExitSentenceExtractor:
    def __init__(
        self,
        retriever_model: str,
        compression_model: str,
        device: str = "cpu",
        torch_dtype: str = "auto",
    ):
        try:
            import torch
            from peft import PeftModel
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise SystemExit(
                "EXIT dependencies missing. Install torch, transformers, peft, and accelerate."
            ) from exc

        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(retriever_model)
        model_kwargs = {"device_map": device}
        if torch_dtype == "float16":
            model_kwargs["torch_dtype"] = torch.float16
        elif torch_dtype == "bfloat16":
            model_kwargs["torch_dtype"] = torch.bfloat16
        self.base_model = AutoModelForCausalLM.from_pretrained(
            retriever_model, **model_kwargs
        )
        self.model = PeftModel.from_pretrained(self.base_model, compression_model)
        self.model.eval()
        self.device = self.model.device
        yes_ids = self.tokenizer.encode("Yes", add_special_tokens=False)
        no_ids = self.tokenizer.encode("No", add_special_tokens=False)
        if not yes_ids or not no_ids:
            raise SystemExit("Could not determine token ids for Yes/No.")
        self.yes_id = yes_ids[0]
        self.no_id = no_ids[0]

    def score_sentence(self, query: str, context: str, sentence: str) -> float:
        prompt = (
            "<start_of_turn>user\n"
            f"Query:\n{query}\n"
            f"Full context:\n{context}\n"
            f"Sentence:\n{sentence}\n"
            'Is this sentence useful in answering the query? Answer only "Yes" or "No".'
            "<end_of_turn>\n"
            "<start_of_turn>model\n"
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with self.torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits[0, -1, [self.yes_id, self.no_id]]
            return self.torch.softmax(logits, dim=0)[0].item()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold-file", type=Path, required=True)
    parser.add_argument("--output-file", type=Path, required=True)
    parser.add_argument("--retriever-model", default="google/gemma-2b-it")
    parser.add_argument("--compression-model", default="doubleyyh/exit-gemma-2b")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--torch-dtype", choices=("auto", "float16", "bfloat16"), default="auto"
    )
    args = parser.parse_args()

    extractor = ExitSentenceExtractor(
        retriever_model=args.retriever_model,
        compression_model=args.compression_model,
        device=args.device,
        torch_dtype=args.torch_dtype,
    )
    rows = [row for row in load_gold_rows(args.gold_file) if row.is_relevant]
    args.output_file.parent.mkdir(parents=True, exist_ok=True)

    with args.output_file.open("w") as f:
        for row in tqdm(rows, desc="exit"):
            t0 = time.perf_counter()
            try:
                selected = []
                scores = []
                for start, end, sentence in split_sentences(row.chunk):
                    score = extractor.score_sentence(row.query, row.chunk, sentence)
                    scores.append({"start": start, "end": end, "score": score})
                    if score >= args.threshold:
                        selected.append((start, end))
                selected = merge_adjacent_spans(selected, row.chunk)
                pred_spans = [
                    {"start": start, "end": end, "text": row.chunk[start:end]}
                    for start, end in selected
                ]
                error = None
            except Exception as exc:
                pred_spans = []
                scores = []
                error = f"{type(exc).__name__}: {exc}"
            record = {
                "query": row.query,
                "paper_id": row.paper_id,
                "chunk_index": row.chunk_index,
                "pred_spans": pred_spans,
                "sentence_scores": scores,
                "latency_s": time.perf_counter() - t0,
                "error": error,
                "extractor": "exit",
                "retriever_model": args.retriever_model,
                "compression_model": args.compression_model,
                "threshold": args.threshold,
            }
            f.write(json.dumps(record) + "\n")


if __name__ == "__main__":
    main()
