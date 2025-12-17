import argparse
import json
import logging
import os
from collections import Counter
from pathlib import Path

from tabulate import tabulate
from tqdm import tqdm

from verbatim_rag import VerbatimIndex, VerbatimRAG
from verbatim_rag.embedding_providers import (
    SentenceTransformersProvider,
    SpladeProvider,
)
from verbatim_rag.vector_stores import LocalMilvusStore, CloudMilvusStore
from verbatim_rag.core import LLMClient


HYBRID_WEIGHTS = {"dense": 0.4, "sparse": 0.4, "full_text": 0.2}


def get_args():
    parser = argparse.ArgumentParser(description="Query ACL Anthology index")
    parser.add_argument("--index-file", help="File for storing index db (local mode)")
    parser.add_argument("--collection-name", required=True, help="Name of collection")
    parser.add_argument(
        "-s",
        "--search-type",
        required=True,
        help='Type of search ("dense", "sparse", "hybrid", "full_text", "auto")',
    )
    parser.add_argument("--questions-dir", help="Path to questions")
    parser.add_argument("--output-file", help="File for storing search results")
    parser.add_argument("-f", "--query-field", help="Field to use as search query")
    parser.add_argument("-k", type=int, default=5)
    parser.add_argument(
        "--device", required=True, help="Device to use for embedding (e.g. cpu or cuda)"
    )
    parser.add_argument(
        "-r",
        "--retrieve-only",
        action="store_true",
        help="Only test retrieval",
    )
    parser.add_argument(
        "--use-cloud",
        action="store_true",
        help="Use CloudMilvusStore instead of LocalMilvusStore",
    )
    parser.add_argument(
        "--cloud-uri",
        help="URI for cloud Milvus instance (e.g. http://localhost:19530)",
    )

    args = parser.parse_args()

    # Validate arguments
    if args.use_cloud:
        if not args.cloud_uri:
            parser.error("--cloud-uri is required when --use-cloud is specified")
    else:
        if not args.index_file:
            parser.error("--index-file is required when --use-cloud is not specified")

    return args


def load_results(results_file):
    results = []
    with open(results_file) as f:
        for line in f:
            results.append(json.loads(line))
    return results


def save_results(results, output_file):
    with open(output_file, "w") as of:
        for res in results:
            of.write(json.dumps(res) + "\n")


def get_results_for_query(query, index, paper_id, chunk_index, args):
    search_results = index.query(
        text=query,
        search_type=args.search_type,
        k=args.k,
        rrf_k=60,
        hybrid_weights=HYBRID_WEIGHTS,
        filter=None,
    )
    output = {
        "query": query,
        "gold_paper": paper_id,
        "gold_chunk": chunk_index,
        "results": [res.metadata for res in search_results],
        "paper_found": False,
        "chunk_found": False,
        "corr_paper_rank": None,
        "corr_chunk_rank": None,
    }
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


def get_stats_from_results(results):
    stats = Counter()
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


def get_results(index, rag, args):
    results = []
    for file_path in tqdm(Path(args.questions_dir).rglob("*")):
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
                            q[args.query_field], index, paper_id, chunk_index, args
                        )
                    )
    return results


def get_overall_stats(stats, args):

    rows = []
    print(
        f"Results for {args.output_file=}, {args.query_field=}, {args.search_type=}, {HYBRID_WEIGHTS=}"
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


def test_batch(args):
    if os.path.exists(args.output_file):
        print(f"output file {args.output_file} exists, will not run search")
        results = load_results(args.output_file)
    else:
        index = get_index(args)
        if args.retrieve_only:
            rag = None
        else:
            rag = get_rag(index, args)

        results = get_results(index, rag, args)
        save_results(results, args.output_file)

    stats = get_stats_from_results(results)
    get_overall_stats(stats, args)


def test_interactive(args):
    index = get_index(args)
    if args.retrieve_only:
        rag = None
    else:
        rag = get_rag(index, args)

    while True:
        test_query = input(">")
        logging.info(f"asking: {test_query}")
        if rag:
            response = rag.query(test_query)
            logging.info(f"answer: {response.answer}")
        else:
            results = index.query(
                text=test_query, k=args.k, rrf_k=60, hybrid_weights=None, filter=None
            )
            for i, res in enumerate(results[::-1]):
                d = res.metadata
                logging.info(
                    f"{i}. Paper: {d['document_id']}, Chunk: {d['chunk_number']}, Title: {d['title']}, URL: {d['url']}, Score: {res.score}"
                )


def get_index(args):
    dense_provider = SentenceTransformersProvider(
        model_name="ibm-granite/granite-embedding-english-r2", device=args.device
    )
    sparse_provider = SpladeProvider(
        model_name="opensearch-project/opensearch-neural-sparse-encoding-doc-v2-distill",
        device=args.device,
    )

    # Create vector store
    if args.use_cloud:
        logging.info(f"Using CloudMilvusStore at {args.cloud_uri}")
        vector_store = CloudMilvusStore(
            uri=args.cloud_uri,
            collection_name=args.collection_name,
            enable_dense=True,
            enable_sparse=True,
            enable_full_text=True,
            dense_dim=dense_provider.get_dimension(),
            sparse_dim=sparse_provider.get_dimension(),
            nlist=16384,
        )
    else:
        logging.info(f"Using LocalMilvusStore at {args.index_file}")
        vector_store = LocalMilvusStore(
            db_path=args.index_file,
            collection_name=args.collection_name,
            enable_dense=True,
            enable_sparse=True,
            dense_dim=dense_provider.get_dimension(),
            sparse_dim=sparse_provider.get_dimension(),
            nlist=16384,
        )

    # Create index
    index = VerbatimIndex(
        vector_store=vector_store,
        dense_provider=dense_provider,
        sparse_provider=sparse_provider,
    )

    return index


def get_rag(index, args):
    llm_client = LLMClient(
        model="moonshotai/kimi-k2-instruct-0905",
        api_base="https://api.groq.com/openai/v1/",
    )
    rag = VerbatimRAG(index, llm_client=llm_client, k=args.k)

    return rag


def main():
    args = get_args()

    if args.questions_dir:
        test_batch(args)
    else:
        test_interactive(args)


if __name__ == "__main__":
    main()
