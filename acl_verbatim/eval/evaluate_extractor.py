"""Score a span extractor against the local gold test set.

Wires an `LLMSpanExtractor` over any OpenAI-compatible endpoint via the
standard trio of env vars:

  OPENAI_API_KEY   — the key for the chosen provider
  OPENAI_API_BASE  — endpoint (default https://api.openai.com/v1)
  OPENAI_MODEL     — model name (default gpt-4o-mini)

Switch provider by exporting different values; the script doesn't care. CLI
flags are thin overrides for one-off runs.

Metrics (all computed against the relevant chunks in the gold file):
  * Word-F1 (micro/macro) — token-set F1 on the union of predicted and gold
    spans at whitespace-word granularity. Headline metric, matches SQuAD /
    QuAC / LettuceDetect conventions.
  * Word-Recall (micro) — recall only; gold spans in this dataset are
    intentionally generous (see NOTES.md 2026.01.23), so recall is the axis
    that matters most for the downstream use case.
  * Span-F1 at IoU thresholds {0.3, 0.5, 0.7} — greedy bipartite matching
    using character-level intersection-over-union as the pair score. Captures
    "did you pick the same units as the annotator, modulo boundary drift."
  * Recall@any-overlap — fraction of gold spans touched by any predicted
    span. Cheap sanity signal.
  * Over-prediction ratio — mean |predicted| / |gold|. Captures the
    fragmentation pattern.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from types import SimpleNamespace

from tqdm import tqdm

from verbatim_core.extractors import LLMSpanExtractor
from verbatim_core.llm_client import LLMClient

from acl_verbatim.data.spans import SpanRow, load_gold_rows
from acl_verbatim.eval.span_metrics import (
    align_predicted_texts,
    evaluate_rows_against_predictions,
    make_prediction_key,
)


def build_extractor(args):
    api_key = os.environ.get("OPENAI_API_KEY", "")
    api_base = args.api_base or os.environ.get(
        "OPENAI_API_BASE", "https://api.openai.com/v1"
    )
    model = args.model or os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
    if not api_key:
        raise SystemExit("OPENAI_API_KEY not set. Export it before running.")
    client = LLMClient(
        model=model,
        temperature=args.temperature,
        api_base=api_base,
        api_key=api_key,
    )
    extraction_prompt = None
    if args.extraction_prompt_file:
        extraction_prompt = Path(args.extraction_prompt_file).read_text(
            encoding="utf-8"
        )
    extractor = LLMSpanExtractor(
        llm_client=client,
        model=model,
        extraction_mode="batch",
        batch_size=args.batch_size,
        span_match_mode="fuzzy",
        fuzzy_threshold=0.8,
        extraction_prompt=extraction_prompt,
    )
    return extractor, model, api_base


def evaluate(
    rows: list[SpanRow],
    extractor: LLMSpanExtractor,
    limit: int | None = None,
) -> dict:
    relevant = [r for r in rows if r.is_relevant]
    if limit is not None:
        relevant = relevant[:limit]

    # Group by query so each batched extractor call covers one question's chunks.
    # This preserves the original row order for the output records.
    by_query: dict[str, list[SpanRow]] = {}
    for row in relevant:
        by_query.setdefault(row.query, []).append(row)

    pred_map: dict[tuple[str, str, int], dict] = {}

    for query, group in tqdm(by_query.items(), desc="extracting"):
        stubs = [SimpleNamespace(text=row.chunk) for row in group]
        t0 = time.perf_counter()
        try:
            result = extractor.extract_spans(query, stubs)
            error = None
        except Exception as e:
            result = {}
            error = f"{type(e).__name__}: {e}"
        elapsed = time.perf_counter() - t0
        per_call_share = elapsed / max(1, len(group))
        for row in group:
            predicted_texts = result.get(row.chunk, []) if error is None else []
            pred_spans = align_predicted_texts(row.chunk, predicted_texts)
            pred_map[make_prediction_key(row.query, row.paper_id, row.chunk_index)] = {
                "query": row.query,
                "paper_id": row.paper_id,
                "chunk_index": row.chunk_index,
                "predicted_texts": predicted_texts,
                "pred_spans": [
                    {"start": start, "end": end, "text": row.chunk[start:end]}
                    for start, end in pred_spans
                ],
                "latency_s": per_call_share,
                "error": error,
            }

    return evaluate_rows_against_predictions(relevant, pred_map)


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--gold-file", type=Path, required=True)
    p.add_argument("--model", default=None, help="Override OPENAI_MODEL env var")
    p.add_argument("--api-base", default=None, help="Override OPENAI_API_BASE env var")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument(
        "--batch-size", type=int, default=5, help="Chunks per LLM call (batch mode)"
    )
    p.add_argument(
        "--extraction-prompt-file",
        default=None,
        help="Path to custom extraction prompt (Jinja2 with {{question}} and {{documents}})",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Cap to first N relevant rows (smoke test)",
    )
    p.add_argument(
        "--output-file", type=Path, default=None, help="Write per-row records here"
    )
    args = p.parse_args()

    rows = list(load_gold_rows(args.gold_file))
    extractor, model, api_base = build_extractor(args)
    print(f"model={model} api_base={api_base}")

    result = evaluate(rows, extractor, limit=args.limit)
    print(json.dumps(result["summary"], indent=2))

    if args.output_file:
        args.output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_file, "w") as f:
            json.dump(
                {"model": model, "api_base": api_base, **result},
                f,
                indent=2,
            )
        print(f"wrote per-row records to {args.output_file}")


if __name__ == "__main__":
    main()
