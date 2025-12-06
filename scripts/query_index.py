import argparse
import logging


from verbatim_rag import VerbatimIndex, VerbatimRAG
from verbatim_rag.embedding_providers import (
    SentenceTransformersProvider,
    SpladeProvider,
)
from verbatim_rag.vector_stores import LocalMilvusStore, CloudMilvusStore
from verbatim_rag.core import LLMClient


def get_args():
    parser = argparse.ArgumentParser(description="Query ACL Anthology index")
    parser.add_argument("--index-file", help="File for storing index db (local mode)")
    parser.add_argument("--collection-name", required=True, help="Name of collection")
    parser.add_argument(
        "--device", required=True, help="Device to use for embedding (e.g. cpu or cuda)"
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


def main():
    args = get_args()

    llm_client = LLMClient(
        model="moonshotai/kimi-k2-instruct-0905",
        api_base="https://api.groq.com/openai/v1/",
    )

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
    rag = VerbatimRAG(index, llm_client=llm_client)
    while True:
        test_query = input(">")
        logging.info(f"asking: {test_query}")
        response = rag.query(test_query)
        logging.info(f"answer: {response.answer}")


if __name__ == "__main__":
    main()
