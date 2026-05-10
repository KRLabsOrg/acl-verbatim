"""Run Zilliz semantic highlighting on the gold set and emit normalized predictions."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from types import SimpleNamespace

from tqdm import tqdm
from verbatim_rag.extractors import SemanticHighlightExtractor

from acl_verbatim.data.spans import load_gold_rows
from acl_verbatim.eval.span_metrics import align_predicted_texts


def resolve_device(device: str | None) -> str:
    if device:
        return device
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold-file", type=Path, required=True)
    parser.add_argument("--output-file", type=Path, required=True)
    parser.add_argument(
        "--model-name", default="zilliz/semantic-highlight-bilingual-v1"
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument(
        "--output-mode", choices=("sentences", "spans"), default="sentences"
    )
    parser.add_argument("--language", default="auto")
    args = parser.parse_args()

    device = resolve_device(args.device)
    extractor = SemanticHighlightExtractor(
        model_name=args.model_name,
        device=device,
        threshold=args.threshold,
        output_mode=args.output_mode,
        language=args.language,
    )
    if hasattr(extractor, "model"):
        extractor.model.to(device)
        extractor.model.eval()
    rows = list(load_gold_rows(args.gold_file))
    args.output_file.parent.mkdir(parents=True, exist_ok=True)

    with args.output_file.open("w") as f:
        for row in tqdm(rows, desc="highlighting"):
            t0 = time.perf_counter()
            try:
                result = extractor.extract_spans(
                    row.query, [SimpleNamespace(text=row.chunk)]
                )
                predicted_texts = result.get(row.chunk, [])
                pred_spans = align_predicted_texts(row.chunk, predicted_texts)
                error = None
            except Exception as exc:
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
                "latency_s": time.perf_counter() - t0,
                "error": error,
                "model": args.model_name,
                "extractor": "semantic_highlight",
                "output_mode": args.output_mode,
                "device": device,
            }
            f.write(json.dumps(record) + "\n")


if __name__ == "__main__":
    main()
