import argparse
import json
import multiprocessing as mp
import os
import queue
import time
from pathlib import Path
from types import SimpleNamespace

from tqdm import tqdm
from verbatim_core.extractors import LLMSpanExtractor
from verbatim_core.llm_client import LLMClient

from acl_verbatim.core.chunks import ChunkResolver
from acl_verbatim.core.jsonl import iter_jsonl
from acl_verbatim.eval.span_metrics import align_predicted_texts
from acl_verbatim.synthetic.annotation import build_retrieval_candidates, silver_record


def get_args():
    parser = argparse.ArgumentParser(
        description="Annotate spans from retrieval results using batched LLMSpanExtractor"
    )
    parser.add_argument(
        "--results-file",
        required=True,
        help="Path to retrieval results JSONL",
    )
    parser.add_argument(
        "--chunks-dir",
        help="Optional directory containing paper chunk JSONL files",
    )
    parser.add_argument("--output-file", required=True, help="Output JSONL path")
    parser.add_argument(
        "--model",
        default=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
        help="OpenAI-compatible model name",
    )
    parser.add_argument(
        "--api-base",
        default=os.environ.get("OPENAI_API_BASE", "https://api.openai.com/v1"),
        help="OpenAI-compatible API base",
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("OPENAI_API_KEY"),
        help="Optional API key for the endpoint",
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--batch-size", type=int, default=5, help="Documents per extractor request"
    )
    parser.add_argument(
        "--max-results-per-query",
        type=int,
        default=5,
        help="Max retrieved candidates to consider per query",
    )
    parser.add_argument(
        "--collection-name",
        required=True,
        help="Milvus collection name",
    )
    parser.add_argument(
        "--milvus-uri",
        required=True,
        help="Milvus URI for direct client",
    )
    parser.add_argument(
        "--milvus-token",
        default=os.environ.get("MILVUS_API_KEY"),
        help="Optional Milvus token / API key",
    )
    parser.add_argument(
        "--max-queries",
        type=int,
        default=None,
        help="Process only the first N queries from results file",
    )
    parser.add_argument(
        "--skip-missing",
        action="store_true",
        help="Skip rows when chunk text is missing",
    )
    parser.add_argument(
        "--flush",
        action="store_true",
        help="Flush output after each query",
    )
    parser.add_argument(
        "--extraction-prompt-file",
        default="acl_verbatim/prompts/extraction_paragraph.txt",
        help="Path to custom extraction prompt",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of worker processes over query groups",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from an existing output file by skipping completed queries",
    )
    return parser.parse_args()


def query_key_from_result_row(row: dict) -> tuple[str, str, int] | None:
    query = row.get("query")
    gold_paper = row.get("gold_paper")
    gold_chunk = row.get("gold_chunk")
    if not query or gold_paper is None or gold_chunk is None:
        return None
    return query, gold_paper, int(gold_chunk)


def query_key_from_output_row(row: dict) -> tuple[str, str, int] | None:
    query = row.get("question")
    gold_paper = row.get("gold_paper")
    gold_chunk = row.get("gold_chunk")
    if not query or gold_paper is None or gold_chunk is None:
        return None
    return query, gold_paper, int(gold_chunk)


def build_extractor(args):
    if not args.api_key:
        raise SystemExit("API key missing. Set OPENAI_API_KEY or pass --api-key.")
    llm_client = LLMClient(
        model=args.model,
        api_base=args.api_base,
        api_key=args.api_key,
        temperature=args.temperature,
    )
    prompt = None
    if args.extraction_prompt_file:
        prompt = Path(args.extraction_prompt_file).read_text(encoding="utf-8")
    return LLMSpanExtractor(
        llm_client=llm_client,
        model=args.model,
        extraction_mode="batch",
        batch_size=args.batch_size,
        span_match_mode="fuzzy",
        fuzzy_threshold=0.8,
        extraction_prompt=prompt,
    )


def process_result_row(row: dict, extractor, resolver, args) -> list[dict]:
    candidates = build_retrieval_candidates(
        row,
        resolver,
        args.max_results_per_query,
        skip_missing=args.skip_missing,
    )
    if not candidates:
        return []

    t0 = time.perf_counter()
    stubs = [SimpleNamespace(text=c["chunk"]) for c in candidates]
    try:
        extraction = extractor.extract_spans(candidates[0]["question"], stubs)
        err = None
    except Exception as exc:
        extraction = {}
        err = f"{type(exc).__name__}: {exc}"
    elapsed = time.perf_counter() - t0
    per_row_latency = elapsed / max(1, len(candidates))

    records = []
    for candidate in candidates:
        predicted_texts = extraction.get(candidate["chunk"], []) if err is None else []
        pred_spans = align_predicted_texts(candidate["chunk"], predicted_texts)
        answerable = bool(pred_spans)
        records.append(
            silver_record(
                candidate=candidate,
                spans=[
                    {
                        "start": start,
                        "end": end,
                        "text": candidate["chunk"][start:end],
                    }
                    for start, end in pred_spans
                ],
                answerable=answerable,
                predicted_texts=predicted_texts,
                latency_s=per_row_latency,
                err=err,
            )
        )
    return records


def load_pending_rows(args) -> list[dict]:
    rows = []
    for idx, row in enumerate(iter_jsonl(args.results_file)):
        if args.max_queries is not None and idx >= args.max_queries:
            break
        rows.append(row)

    if not args.resume or not Path(args.output_file).exists():
        return rows

    completed = set()
    for row in iter_jsonl(args.output_file):
        key = query_key_from_output_row(row)
        if key is not None:
            completed.add(key)

    pending = []
    for row in rows:
        key = query_key_from_result_row(row)
        if key is None or key not in completed:
            pending.append(row)
    return pending


def worker_main(task_queue, result_queue, config: dict):
    args = SimpleNamespace(**config)
    extractor = build_extractor(args)
    resolver = ChunkResolver(
        collection_name=args.collection_name,
        chunks_dir=args.chunks_dir,
        milvus_uri=args.milvus_uri,
        milvus_token=args.milvus_token,
    )

    while True:
        row = task_queue.get()
        if row is None:
            break
        key = query_key_from_result_row(row)
        try:
            records = process_result_row(row, extractor, resolver, args)
            result_queue.put(("ok", key, records))
        except Exception as exc:
            result_queue.put(("error", key, f"{type(exc).__name__}: {exc}"))


def run_parallel(rows: list[dict], args):
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_mode = "a" if args.resume and output_path.exists() else "w"

    ctx = mp.get_context("spawn")
    task_queue = ctx.Queue(maxsize=max(2 * args.workers, 8))
    result_queue = ctx.Queue()
    config = {
        "model": args.model,
        "api_base": args.api_base,
        "api_key": args.api_key,
        "temperature": args.temperature,
        "batch_size": args.batch_size,
        "max_results_per_query": args.max_results_per_query,
        "collection_name": args.collection_name,
        "chunks_dir": args.chunks_dir,
        "milvus_uri": args.milvus_uri,
        "milvus_token": args.milvus_token,
        "skip_missing": args.skip_missing,
        "extraction_prompt_file": args.extraction_prompt_file,
    }

    workers = [
        ctx.Process(target=worker_main, args=(task_queue, result_queue, config))
        for _ in range(args.workers)
    ]
    for proc in workers:
        proc.start()

    for row in rows:
        task_queue.put(row)
    for _ in workers:
        task_queue.put(None)

    errors = []
    with (
        output_path.open(write_mode) as of,
        tqdm(total=len(rows), desc="queries", unit="query") as pbar,
    ):
        completed = 0
        while completed < len(rows):
            try:
                status, key, payload = result_queue.get(timeout=1.0)
            except queue.Empty:
                continue
            if status == "ok":
                for record in payload:
                    of.write(json.dumps(record) + "\n")
                if args.flush:
                    of.flush()
            else:
                errors.append((key, payload))
            completed += 1
            pbar.update(1)

    for proc in workers:
        proc.join()

    if errors:
        sample = "; ".join(f"{key}: {msg}" for key, msg in errors[:3])
        raise SystemExit(
            f"parallel annotation had {len(errors)} query failures: {sample}"
        )


def run_sequential(rows: list[dict], args):
    extractor = build_extractor(args)
    resolver = ChunkResolver(
        collection_name=args.collection_name,
        chunks_dir=args.chunks_dir,
        milvus_uri=args.milvus_uri,
        milvus_token=args.milvus_token,
    )

    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_mode = "a" if args.resume and output_path.exists() else "w"

    with output_path.open(write_mode) as of:
        for row in tqdm(rows, desc="queries", unit="query"):
            for record in process_result_row(row, extractor, resolver, args):
                of.write(json.dumps(record) + "\n")
            if args.flush:
                of.flush()


def main():
    args = get_args()
    if args.workers < 1:
        raise SystemExit("--workers must be >= 1")

    rows = load_pending_rows(args)
    print(f"annotating {len(rows)} query groups", flush=True)
    if args.workers == 1:
        run_sequential(rows, args)
    else:
        run_parallel(rows, args)


if __name__ == "__main__":
    main()
