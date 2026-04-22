import csv
import json
from types import SimpleNamespace

from pymilvus import MilvusClient
from rapidfuzz import fuzz
from tqdm import tqdm

from verbatim_rag.extractors import LLMSpanExtractor

from acl_verbatim.eval.utils import get_chunk
from acl_verbatim.retrieval.runtime import get_extractor, get_llm_client


def get_extraction_results_for_query(
    data, extractor, k, client, fuzzy_threshold=0.9, partial_matches_writer=None
):
    chunks = [
        get_chunk(res["url"], res["chunk_number"], client)[0]
        for res in data["results"][:k]
    ]
    all_spans = extractor.extract_spans(
        data["query"], [SimpleNamespace(text=chunk) for chunk in chunks]
    )
    for i, res in enumerate(data["results"][:k]):
        res["extraction"] = []
        chunk = chunks[i]
        spans = all_spans.get(chunk)
        if spans is None:
            continue
        for span in spans:
            alignment = fuzz.partial_ratio_alignment(span, chunk)
            score = alignment.score / 100.0
            start = alignment.dest_start
            end = alignment.dest_end
            matched_text = chunk[start:end]
            if score < 1.0 and partial_matches_writer is not None:
                partial_matches_writer.writerow(
                    [f"{score:.4f}", span, matched_text, res["url"]]
                )
            if score < fuzzy_threshold:
                continue
            res["extraction"].append({"text": matched_text, "start": start, "end": end})

    return data


def iter_extraction_results(args):
    extractor = get_extractor(args)
    if extractor is None:
        extractor = LLMSpanExtractor(
            llm_client=get_llm_client(args.llm_response_log),
            extraction_mode="auto",
            max_display_spans=5,
            verify_spans=False,
        )

    milvus_kwargs = {"uri": args.cloud_uri}
    if args.milvus_token:
        milvus_kwargs["token"] = args.milvus_token
    client = MilvusClient(**milvus_kwargs)

    partial_matches_file = None
    partial_matches_writer = None
    if args.partial_matches_file:
        args.partial_matches_file.parent.mkdir(parents=True, exist_ok=True)
        partial_matches_file = open(args.partial_matches_file, "w", newline="")
        partial_matches_writer = csv.writer(partial_matches_file)

    try:
        with open(args.search_results_file) as f:
            for line in tqdm(f):
                yield get_extraction_results_for_query(
                    json.loads(line),
                    extractor,
                    args.k,
                    client,
                    partial_matches_writer=partial_matches_writer,
                )
    finally:
        if partial_matches_file:
            partial_matches_file.close()
