"""Run Provence context pruning on the gold set and emit normalized predictions."""

from __future__ import annotations

import argparse
import json
import re
import string
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


def normalize_text(text: str) -> str:
    text = text.lower()
    text = "".join(ch for ch in text if ch not in string.punctuation)
    return " ".join(text.split())


class ProvenceSentenceExtractor:
    def __init__(self, model_name: str, threshold: float, always_select_title: bool):
        import torch
        from transformers import AutoModel

        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        if torch.cuda.is_available():
            self.model.to("cuda")
        elif torch.backends.mps.is_available():
            self.model.to("mps")
        self.threshold = threshold
        self.always_select_title = always_select_title

    def extract_spans(
        self, query: str, context: str
    ) -> tuple[str, list[tuple[int, int]]]:
        result = self.model.process(
            query,
            context,
            title=None,
            threshold=self.threshold,
            always_select_title=self.always_select_title,
        )
        pruned_context = result["pruned_context"]
        if not isinstance(pruned_context, str) or not pruned_context.strip():
            return "", []

        normalized_pruned = normalize_text(pruned_context)
        selected: list[tuple[int, int]] = []
        for start, end, sentence in split_sentences(context):
            normalized_sentence = normalize_text(sentence)
            if normalized_sentence and normalized_sentence in normalized_pruned:
                selected.append((start, end))

        return pruned_context, merge_adjacent_spans(selected, context)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold-file", type=Path, required=True)
    parser.add_argument("--output-file", type=Path, required=True)
    parser.add_argument("--model-name", default="naver/provence-reranker-debertav3-v1")
    parser.add_argument("--threshold", type=float, default=0.1)
    parser.add_argument("--always-select-title", action="store_true")
    args = parser.parse_args()

    extractor = ProvenceSentenceExtractor(
        model_name=args.model_name,
        threshold=args.threshold,
        always_select_title=args.always_select_title,
    )
    rows = list(load_gold_rows(args.gold_file))
    args.output_file.parent.mkdir(parents=True, exist_ok=True)

    with args.output_file.open("w") as f:
        for row in tqdm(rows, desc="provence"):
            t0 = time.perf_counter()
            try:
                pruned_context, pred_spans = extractor.extract_spans(
                    row.query, row.chunk
                )
                predicted_texts = [row.chunk[start:end] for start, end in pred_spans]
                error = None
            except Exception as exc:
                pruned_context = ""
                predicted_texts = []
                pred_spans = []
                error = f"{type(exc).__name__}: {exc}"
            record = {
                "query": row.query,
                "paper_id": row.paper_id,
                "chunk_index": row.chunk_index,
                "predicted_texts": predicted_texts,
                "pred_spans": [
                    {"start": start, "end": end, "text": row.chunk[start:end]}
                    for start, end in pred_spans
                ],
                "pruned_context": pruned_context,
                "latency_s": time.perf_counter() - t0,
                "error": error,
                "model": args.model_name,
                "extractor": "provence",
                "threshold": args.threshold,
            }
            f.write(json.dumps(record) + "\n")


if __name__ == "__main__":
    main()
