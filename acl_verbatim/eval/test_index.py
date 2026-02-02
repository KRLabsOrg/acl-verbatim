import argparse
import csv
import json
import logging
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Optional, Sequence

from pymilvus import MilvusClient
from rapidfuzz import fuzz
from tabulate import tabulate
from tqdm import tqdm

from verbatim_rag import VerbatimIndex, VerbatimRAG, BaseReranker
from verbatim_rag.embedding_providers import (
    SentenceTransformersProvider,
    # SpladeProvider,
)
from verbatim_rag.extractors import LLMSpanExtractor, SemanticHighlightExtractor
from verbatim_rag.vector_stores.base import SearchResult
from verbatim_rag.vector_stores import LocalMilvusStore, CloudMilvusStore
from verbatim_rag.core import LLMClient

from acl_verbatim.eval.utils import get_chunk

HYBRID_WEIGHTS = {"dense": 0.3, "full_text": 0.7}


class MyLLMClient(LLMClient):

    def _build_extraction_prompt(self, question: str, documents: Dict[str, str]) -> str:
        """Build the prompt for batch span extraction."""
        return f"""Extract EXACT verbatim text spans from multiple documents that answer the question.

# Rules
1. Extract **only** text that is relevant to the question
2. Include complete sentences or paragraphs to provide sufficient context
3. If the same information is stated multiple times, choose the best version
4. Never paraphrase, modify, or add to the original text
5. Preserve original wording, capitalization, and punctuation
6. Order spans within each document by relevance - MOST RELEVANT FIRST


# Output Format
Return a JSON object mapping document IDs to span arrays ordered by relevance:
{{
  "doc_0": ["most relevant span", "next most relevant span"],
  "doc_1": ["most relevant from doc 1"],
  "doc_2": []
}}

If no relevant information in a document, use empty array.

# Your Task
Question: {question}

Documents:
{json.dumps(documents, indent=2)}

Extract verbatim spans from each document:"""


@dataclass(frozen=True)
class TestIndexArgs:
    index_file: Optional[Path]
    collection_name: str
    search_type: str
    questions_dir: Optional[Path]
    search_results_file: Optional[Path]
    output_file: Optional[Path]
    query_field: Optional[str]
    k: int
    nprobe: int
    device: str
    retrieve_only: bool
    use_cloud: bool
    cloud_uri: Optional[str]
    rerank: bool
    extractor: str
    log_level: str
    partial_matches_file: Optional[Path]
    milvus_token: Optional[str]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Query / evaluate the ACL Anthology index"
    )

    index_group = parser.add_argument_group("Index")
    index_group.add_argument(
        "--collection-name", required=True, help="Milvus collection name"
    )
    index_group.add_argument(
        "--index-file", type=Path, help="Local Milvus DB path (local mode)"
    )

    search_group = parser.add_argument_group("Search")
    search_group.add_argument(
        "-s",
        "--search-type",
        default="auto",
        choices=("dense", "sparse", "hybrid", "full_text", "auto"),
        help='Search type (default: "auto")',
    )
    search_group.add_argument(
        "-k", type=int, default=5, help="Top-k to retrieve (default: 5)"
    )
    search_group.add_argument(
        "--nprobe", type=int, default=8, help="The number of cluster units to search."
    )
    search_group.add_argument(
        "--device",
        default="cpu",
        help="Embedding device (e.g. cpu, cuda) (default: cpu)",
    )
    search_group.add_argument(
        "-r",
        "--retrieve-only",
        action="store_true",
        help="Only test retrieval (skip LLM answer generation)",
    )

    mode_group = parser.add_argument_group("Mode")
    mode_group.add_argument(
        "--questions-dir", type=Path, help="Batch mode: directory of question files"
    )
    mode_group.add_argument(
        "--search-results-file",
        type=Path,
        help="For testing extraction only: file with search results",
    )
    mode_group.add_argument(
        "--output-file",
        type=Path,
        help="Batch mode: JSONL output file (created if missing)",
    )
    mode_group.add_argument(
        "-f",
        "--query-field",
        help="Batch mode: field name to use as query (e.g. question or question_en)",
    )

    store_group = parser.add_argument_group("Milvus Store")
    store_group.add_argument(
        "--use-cloud",
        action="store_true",
        help="Use CloudMilvusStore instead of LocalMilvusStore",
    )
    store_group.add_argument(
        "--cloud-uri",
        help="Cloud Milvus URI (e.g. http://localhost:19530) (required with --use-cloud)",
    )
    store_group.add_argument(
        "--milvus-token",
        help="Authentication token for Milvus connection",
    )

    rerank_group = parser.add_argument_group("Reranking")
    rerank_group.add_argument(
        "--rerank",
        action="store_true",
        help="Enable reranking of retrieved results",
    )

    extraction_group = parser.add_argument_group("Extraction")
    extraction_group.add_argument(
        "--extractor",
        default="LLM",
        choices=("LLM", "SHL"),
        help="Extractor to use (default: LLM)",
    )
    extraction_group.add_argument(
        "--partial-matches-file",
        type=Path,
        help="File to store partial matches (score < 1.0) in TSV format",
    )

    logging_group = parser.add_argument_group("Logging")
    logging_group.add_argument(
        "--log-level",
        default="INFO",
        choices=("CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"),
        help="Logging verbosity (default: INFO)",
    )

    return parser


def parse_args(argv: Optional[Sequence[str]] = None) -> TestIndexArgs:
    """Parse and validate CLI args."""
    parser = _build_parser()
    ns = parser.parse_args(argv)

    if ns.use_cloud and not ns.cloud_uri:
        parser.error("--cloud-uri is required when --use-cloud is specified")
    if not ns.use_cloud and not ns.index_file:
        parser.error("--index-file is required when --use-cloud is not specified")

    if ns.questions_dir or ns.search_results_file:
        if not ns.output_file:
            parser.error(
                "--output-file is required when --questions-dir or --search-results-file are specified"
            )
        if ns.search_results_file:
            if ns.questions_dir:
                parser.error(
                    "only one of --questions-dir and --search-results-file can be specified"
                )
            if ns.query_field:
                parser.error(
                    "only one of --query-field and --search-results-file can be specified"
                )
        elif not ns.query_field:
            parser.error("--query-field is required when --questions-dir is specified")

    return TestIndexArgs(
        index_file=ns.index_file,
        collection_name=ns.collection_name,
        search_type=ns.search_type,
        questions_dir=ns.questions_dir,
        search_results_file=ns.search_results_file,
        output_file=ns.output_file,
        query_field=ns.query_field,
        k=ns.k,
        nprobe=ns.nprobe,
        device=ns.device,
        retrieve_only=ns.retrieve_only,
        use_cloud=ns.use_cloud,
        cloud_uri=ns.cloud_uri,
        rerank=ns.rerank,
        extractor=ns.extractor,
        log_level=ns.log_level,
        partial_matches_file=ns.partial_matches_file,
        milvus_token=ns.milvus_token,
    )


def setup_logging(log_level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, log_level), format="%(levelname)s: %(message)s"
    )


def load_results(results_file: Path) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    with results_file.open() as f:
        for line in f:
            results.append(json.loads(line))
    return results


def save_results(results: list[dict[str, Any]], output_file: Path) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w") as of:
        for res in results:
            of.write(json.dumps(res) + "\n")


def build_reranker(args: TestIndexArgs) -> Optional[BaseReranker]:
    """Create a reranker instance from args (or None)."""
    if not args.rerank:
        return None

    from verbatim_rag import SentenceTransformersReranker

    return SentenceTransformersReranker(
        model="jinaai/jina-reranker-v2-base-multilingual",
        device=args.device,
        rerank_k=args.k,
        text_field="enhanced_text",
    )


def _query_index(
    index: VerbatimIndex,
    question: str,
    search_type: str,
    k: int,
    search_params: dict,
    hybrid_weights: dict,
    reranker: Optional[BaseReranker],
) -> list[SearchResult]:
    """Run retrieval and optionally rerank results."""
    retrieve_k = k
    if reranker is not None:
        retrieve_k = max(retrieve_k, int(getattr(reranker, "rerank_k", retrieve_k)))

    results = index.query(
        text=question,
        k=retrieve_k,
        rrf_k=60,
        hybrid_weights=hybrid_weights,
        search_params=search_params,
        filter=None,
    )
    if not reranker:
        return results[:k]
    try:
        return reranker.rerank(question, results)[:k]
    except Exception as exc:
        logging.warning(f"Reranker failed, using original order: {exc}")
        return results[:k]


def get_results_for_query(
    query: str,
    index: VerbatimIndex,
    paper_id: str,
    chunk_index: int,
    search_type: str,
    k: int,
    nprobe: int,
    reranker: Optional[BaseReranker],
    rag: Optional[VerbatimRAG],
) -> dict[str, Any]:
    """Run a single query and compute whether the gold paper/chunk is retrieved."""
    output = {
        "query": query,
        "gold_paper": paper_id,
        "gold_chunk": chunk_index,
        "results": [],
        "paper_found": False,
        "chunk_found": False,
        "corr_paper_rank": None,
        "corr_chunk_rank": None,
    }

    hybrid_weights = (
        HYBRID_WEIGHTS if search_type in ("auto", "hybrid") else {search_type: 1.0}
    )
    search_params = {"nprobe": nprobe}

    if rag is not None:
        rag_response, search_results = rag.query(
            query,
            k=k,
            hybrid_weights=hybrid_weights,
            search_params=search_params,
            return_search_results=True,
        )
    else:
        search_results = _query_index(
            index=index,
            question=query,
            hybrid_weights=hybrid_weights,
            search_type=search_type,
            k=k,
            search_params=search_params,
            reranker=reranker,
        )

    for i, res in enumerate(search_results):
        result = res.metadata
        result["extraction"] = (
            None
            if rag is None
            else [
                {"text": hl.text, "start": hl.start, "end": hl.end}
                for hl in rag_response.documents[i].highlights
            ]
        )

        output["results"].append(result)

    for i, res in enumerate(search_results):
        d = res.metadata
        if d["document_id"] == paper_id:
            if not output["paper_found"]:
                output["corr_paper_rank"] = i + 1
                output["paper_found"] = True
            if d["chunk_number"] == chunk_index:
                if not output["chunk_found"]:
                    output["corr_chunk_rank"] = i + 1
                    output["chunk_found"] = True
                    return output
    return output


def get_stats_from_results(results: list[dict[str, Any]]) -> Counter[str]:
    stats: Counter[str] = Counter()
    gold_papers, gold_chunks = set(), set()
    for res in results:
        gold_papers.add(res["gold_paper"])
        gold_chunks.add(f"{res['gold_paper']}_{res['gold_chunk']}")
        stats["queries"] += 1
        if res["paper_found"]:
            rank = res["corr_paper_rank"]
            stats[f"corr_paper@{rank}"] += 1
        if res["chunk_found"]:
            rank = res["corr_chunk_rank"]
            stats[f"corr_chunk@{rank}"] += 1

    stats["gold_papers"] = len(gold_papers)
    stats["gold_chunks"] = len(gold_chunks)

    return stats


def get_rag_results(
    index: VerbatimIndex,
    args: TestIndexArgs,
    reranker: Optional[BaseReranker],
    rag: Optional[VerbatimRAG],
) -> list[dict[str, Any]]:
    """Compute retrieval results for all questions under `questions_dir`."""
    results: list[dict[str, Any]] = []
    for file_path in tqdm(args.questions_dir.rglob("*")):
        paper_id = file_path.stem
        with open(file_path) as f:
            for line in f:
                chunk_data = json.loads(line)
                if not chunk_data["qa"]:
                    continue
                chunk_index = chunk_data["chunk_index"]
                for q in chunk_data["qa"]:
                    results.append(
                        get_results_for_query(
                            q[args.query_field],
                            index,
                            paper_id,
                            chunk_index,
                            args.search_type,
                            args.k,
                            args.nprobe,
                            reranker,
                            rag,
                        )
                    )

    return results


def get_extraction_results_for_query(
    data, extractor, client, fuzzy_threshold=0.9, partial_matches_writer=None
):
    chunks = [
        get_chunk(res["url"], res["chunk_number"], client)[0] for res in data["results"]
    ]
    all_spans = extractor.extract_spans(
        data["query"], [SimpleNamespace(text=chunk) for chunk in chunks]
    )
    for i, res in enumerate(data["results"]):
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
                partial_matches_writer.writerow([f"{score:.4f}", span, matched_text])
            if score < fuzzy_threshold:
                continue
            res["extraction"].append({"text": matched_text, "start": start, "end": end})

    return data


def get_extraction_results(args):
    """run extraction only on previously fetched search results, without using VerbatimRAG"""
    extractor = get_extractor(args)
    if extractor is None:
        # VerbatimRAG would initialize this by default
        extractor = LLMSpanExtractor(
            llm_client=get_llm_client(),
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
            return [
                get_extraction_results_for_query(
                    json.loads(line), extractor, client,
                    partial_matches_writer=partial_matches_writer
                )
                for line in tqdm(f)
            ]
    finally:
        if partial_matches_file:
            partial_matches_file.close()


def get_overall_stats(stats: Counter[str], args: TestIndexArgs) -> None:
    rows = []
    print(
        f"Results for {args.output_file=}, {args.query_field=}, {args.search_type=}, {HYBRID_WEIGHTS=}, {args.rerank=}"
    )
    print(f"Total queries: {stats['queries']}\n")
    for i in range(1, args.k + 1):
        stats[f"total_corr_paper@{i}"] = (
            stats[f"total_corr_paper@{i - 1}"] + stats[f"corr_paper@{i}"]
        )
        paper_recall_at_i = stats[f"total_corr_paper@{i}"] / stats["queries"]

        stats[f"total_corr_chunk@{i}"] = (
            stats[f"total_corr_chunk@{i - 1}"] + stats[f"corr_chunk@{i}"]
        )
        chunk_recall_at_i = stats[f"total_corr_chunk@{i}"] / stats["queries"]

        if i in (1, 3, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000):
            rows.append(
                [f"{i}", f"{paper_recall_at_i:.2%}", f"{chunk_recall_at_i:.2%}"]
            )

    print(tabulate(rows, headers=["k", "paper R @ k", "chunk R @ k"]))


def test_batch(args: TestIndexArgs) -> None:
    """Batch retrieval evaluation against a ground-truth questions directory."""
    if not args.output_file:
        raise ValueError("output_file is required for batch mode")
    if not args.questions_dir and not args.search_results_file:
        raise ValueError(
            "questions_dir or search_results_file is required for batch mode"
        )
    if args.questions_dir and not args.query_field:
        raise ValueError(
            "questions_dir and query_field are both required for batch mode"
        )

    if args.output_file.exists():
        print(f"output file {args.output_file} exists, will not run search")
        results = load_results(args.output_file)
    else:
        if args.questions_dir:
            index = get_index(args)
            reranker = build_reranker(args)
            rag = None if args.retrieve_only else get_rag(index, args, reranker)
            results = get_rag_results(index, args, reranker, rag)
        else:
            results = get_extraction_results(args)

        save_results(results, args.output_file)

    stats = get_stats_from_results(results)
    get_overall_stats(stats, args)


def test_interactive(args: TestIndexArgs) -> None:
    """Interactive querying for QA (`VerbatimRAG`) or retrieval-only."""
    index = get_index(args)
    reranker = build_reranker(args)
    rag = None if args.retrieve_only else get_rag(index, args, reranker)

    while True:
        test_query = input(">")
        logging.info(f"asking: {test_query}")
        if rag:
            response = rag.query(test_query)
            logging.info(f"answer: {response.answer}")
        else:
            results = _query_index(
                index=index,
                question=test_query,
                search_type=args.search_type,
                k=args.k,
                nprobe=args.nprobe,
                reranker=reranker,
            )
            for i, res in enumerate(results, start=1):
                d = res.metadata
                logging.info(
                    f"{i}. Paper: {d['document_id']}, Chunk: {d['chunk_number']}, Title: {d['title']}, URL: {d['url']}, Score: {res.score}"
                )


def get_index(args: TestIndexArgs) -> VerbatimIndex:
    """Create a `VerbatimIndex` configured for ACL Anthology search."""
    dense_provider = SentenceTransformersProvider(
        model_name="ibm-granite/granite-embedding-english-r2", device=args.device
    )
    # sparse_provider = SpladeProvider(
    #     model_name="opensearch-project/opensearch-neural-sparse-encoding-doc-v2-distill",
    #     device=args.device,
    # )

    # Create vector store
    if args.use_cloud:
        logging.info(f"Using CloudMilvusStore at {args.cloud_uri}")
        vector_store = CloudMilvusStore(
            uri=args.cloud_uri,
            collection_name=args.collection_name,
            enable_dense=True,
            enable_sparse=False,
            enable_full_text=True,
            dense_dim=dense_provider.get_dimension(),
            # sparse_dim=sparse_provider.get_dimension(),
            nlist=32768,
        )
    else:
        logging.info(f"Using LocalMilvusStore at {args.index_file}")
        vector_store = LocalMilvusStore(
            db_path=str(args.index_file),
            collection_name=args.collection_name,
            enable_dense=True,
            enable_sparse=False,
            dense_dim=dense_provider.get_dimension(),
            # sparse_dim=sparse_provider.get_dimension(),
            nlist=32768,
        )

    # Create index
    index = VerbatimIndex(
        vector_store=vector_store,
        dense_provider=dense_provider,
        # sparse_provider=sparse_provider,
    )

    return index


def get_llm_client():
    return MyLLMClient(
        model="moonshotai/kimi-k2-instruct-0905",
        api_base="https://api.groq.com/openai/v1/",
    )


def get_extractor(args):
    if args.extractor == "LLM":
        return None
    elif args.extractor == "SHL":
        return SemanticHighlightExtractor(output_mode="sentences")
    else:
        raise ValueError(f"unsupported extractor: {args.extractor}")


def get_rag(
    index: VerbatimIndex, args: TestIndexArgs, reranker: Optional[BaseReranker]
) -> VerbatimRAG:
    print("initializing RAG...")

    extractor = get_extractor(args)

    llm_client = get_llm_client()

    rag = VerbatimRAG(
        index,
        extractor=extractor,
        llm_client=llm_client,
        k=args.k,
        reranker=reranker,
        template_mode="static",
    )

    return rag


def main():
    args = parse_args()
    setup_logging(args.log_level)

    if args.questions_dir is not None or args.search_results_file is not None:
        test_batch(args)
        return
    test_interactive(args)


if __name__ == "__main__":
    main()
