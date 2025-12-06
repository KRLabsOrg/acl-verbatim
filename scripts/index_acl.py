import argparse
import json
import logging

from pathlib import Path
from tqdm import tqdm

from verbatim_rag import VerbatimIndex, VerbatimRAG
from verbatim_rag.embedding_providers import (
    SpladeProvider,
    SentenceTransformersProvider,
)
from verbatim_rag.core import LLMClient
from verbatim_rag.vector_stores import LocalMilvusStore, CloudMilvusStore
from verbatim_rag.schema import DocumentSchema
from verbatim_rag.chunker_providers import MarkdownChunkerProvider


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s : %(module)s (%(lineno)s) - %(levelname)s - %(message)s",
)


def index_acl(args):
    with open(args.metadata_file) as f:
        papers = {paper["url"].split("/")[-2]: paper for paper in json.load(f)}

    logging.info("loading documents...")
    documents = []
    for file_path in tqdm(Path(args.input_dir).rglob("*")):
        if file_path.suffix.lower() != ".md":
            logging.warning(f"skipping file because extension isn't md: {file_path}")
            continue
        paper_id = file_path.stem
        if paper_id not in papers:
            logging.warning(f"skipping paper not in metadata file: {paper_id}")
            continue

        content = file_path.read_text(encoding="utf-8")

        document = DocumentSchema(
            id=paper_id,
            content=content,
            title=papers[paper_id]["title"],
            url=papers[paper_id]["url"],
            authors=papers[paper_id].get("authors", []),
            year=papers[paper_id].get("year", None),
            publisher=papers[paper_id].get("publisher", None),
        )

        documents.append(document)

    logging.info("indexing documents...")

    chunker = MarkdownChunkerProvider(
        min_chunk_size=500,
        max_chunk_size=5000,
    )
    dense_provider = SentenceTransformersProvider(
        model_name="ibm-granite/granite-embedding-english-r2", device=args.device
    )
    sparse_provider = SpladeProvider(
        model_name="opensearch-project/opensearch-neural-sparse-encoding-doc-v2-distill",
        device=args.device,
    )

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

    index = VerbatimIndex(
        vector_store=vector_store,
        dense_provider=dense_provider,
        sparse_provider=sparse_provider,
        chunker_provider=chunker,
    )
    logging.info("chunking and indexing documents...")
    index.add_documents(documents)
    return index


def get_args():
    parser = argparse.ArgumentParser(description="Preprocess ACL Anthology papers")
    parser.add_argument(
        "--metadata-file", required=True, help="Path to paper metadata file"
    )
    parser.add_argument(
        "--input-dir", required=True, help="Directory for downloaded papers"
    )
    parser.add_argument("--index-file", help="File for storing index db (local mode)")
    parser.add_argument("--collection-name", required=True, help="Name of collection")
    parser.add_argument(
        "--device", required=True, help="Device to use for embedding (e.g. cpu or cuda)"
    )
    parser.add_argument("--dry-run", action="store_true", help="Dry run")
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


def main():
    args = get_args()
    # Check dependencies
    index = index_acl(args)

    llm_client = LLMClient(
        model="moonshotai/kimi-k2-instruct-0905",
        api_base="https://api.groq.com/openai/v1/",
    )

    rag = VerbatimRAG(index, llm_client=llm_client)
    test_query = "What is 4lang?"
    logging.info(f"asking: {test_query}")
    response = rag.query(test_query)
    logging.info(f"answer: {response.answer}")


if __name__ == "__main__":
    main()
