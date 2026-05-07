import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

from acl_verbatim.retrieval.extraction import iter_extraction_results
from acl_verbatim.retrieval.io import load_results, save_results
from acl_verbatim.retrieval.runtime import (
    HYBRID_WEIGHTS,
    build_reranker,
    get_index,
    get_rag,
    iter_rag_results,
    query_index,
)
from acl_verbatim.retrieval.stats import get_stats_from_results, print_overall_stats


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
    llm_response_log: Optional[Path]
    hybrid_weights: dict


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
        help="File to store partial matches (score < 1.0) in CSV format",
    )
    extraction_group.add_argument(
        "--llm-response-log",
        type=Path,
        help="File to log raw LLM responses (JSON Lines format)",
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
        llm_response_log=ns.llm_response_log,
        hybrid_weights=HYBRID_WEIGHTS,
    )


def setup_logging(log_level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, log_level), format="%(levelname)s: %(message)s"
    )


def test_batch(args: TestIndexArgs) -> None:
    assert args.retrieve_only or args.k <= 10, (
        "will refuse to run extraction for more than 10 chunks per query in batch mode"
    )

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
    else:
        if args.questions_dir:
            index = get_index(args)
            reranker = build_reranker(args)
            rag = None if args.retrieve_only else get_rag(index, args, reranker)
            results_generator = iter_rag_results(index, args, reranker, rag)
        else:
            results_generator = iter_extraction_results(args)

        save_results(results_generator, args.output_file)

    results = load_results(args.output_file)
    stats = get_stats_from_results(results)
    print_overall_stats(stats, args)


def test_interactive(args: TestIndexArgs) -> None:
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
            results = query_index(
                index=index,
                question=test_query,
                search_type=args.search_type,
                k=args.k,
                search_params={"nprobe": args.nprobe},
                hybrid_weights=(
                    HYBRID_WEIGHTS
                    if args.search_type in ("auto", "hybrid")
                    else {args.search_type: 1.0}
                ),
                reranker=reranker,
            )
            for i, res in enumerate(results, start=1):
                d = res.metadata
                logging.info(
                    f"{i}. Paper: {d['document_id']}, Chunk: {d['chunk_number']}, "
                    f"Title: {d['title']}, URL: {d['url']}, Score: {res.score}"
                )


def main():
    args = parse_args()
    setup_logging(args.log_level)

    if args.questions_dir is not None or args.search_results_file is not None:
        test_batch(args)
        return
    test_interactive(args)


if __name__ == "__main__":
    main()
