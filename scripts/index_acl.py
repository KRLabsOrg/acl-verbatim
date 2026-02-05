import argparse
import json
import logging

from pathlib import Path
from tqdm import tqdm

from verbatim_rag import VerbatimIndex, VerbatimRAG
from verbatim_rag.embedding_providers import (
    SentenceTransformersProvider,
)
from verbatim_rag.core import LLMClient
from verbatim_rag.vector_stores import LocalMilvusStore, CloudMilvusStore
from verbatim_rag.schema import DocumentSchema
from verbatim_rag.chunker_providers import MarkdownChunkerProvider

from acl_verbatim.utils.preprocess import preprocess_markdown


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s : %(module)s (%(lineno)s) - %(levelname)s - %(message)s",
)


def load_papers_jsonl(path: str) -> dict:
    """Load JSONL format paper data.

    Each line is a JSON object. Extract ID from URL for backward compatibility
    with markdown filenames (e.g., 2025.acl-long.553.md).
    """
    papers = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            paper = json.loads(line)
            # Extract ID from URL: https://aclanthology.org/2025.acl-long.553/ -> 2025.acl-long.553
            paper_id = paper["url"].rstrip("/").split("/")[-1]
            papers[paper_id] = paper
    return papers


def extract_authors(paper: dict) -> list[str]:
    """Extract author full names from structured data.

    New format: [{"id": "...", "first": "...", "last": "...", "full": "Ashish Vaswani"}, ...]
    Falls back to author_string if structured data unavailable.
    """
    authors = paper.get("author", [])
    if isinstance(authors, list) and authors and isinstance(authors[0], dict):
        return [a["full"] for a in authors if a.get("full")]
    # Fallback to author_string if available
    author_string = paper.get("author_string")
    if author_string:
        return [name.strip() for name in author_string.split(",") if name.strip()]
    return []


def index_acl(args):
    # Load JSONL instead of JSON array
    logging.info(f"Loading paper metadata from {args.metadata_file}...")
    papers = load_papers_jsonl(args.metadata_file)
    logging.info(f"Loaded {len(papers)} papers from metadata")

    logging.info("Loading documents...")
    documents = []

    for file_path in tqdm(Path(args.input_dir).rglob("*")):
        if file_path.suffix.lower() != ".md":
            logging.warning(f"skipping file because extension isn't md: {file_path}")
            continue
        paper_id = file_path.stem
        if paper_id not in papers:
            logging.warning(f"skipping paper not in metadata file: {paper_id}")
            continue

        paper = papers[paper_id]
        content = file_path.read_text(encoding="utf-8")
        content = preprocess_markdown(content)

        # Extract year as int if valid
        year_str = paper.get("year", "")
        year = (
            int(year_str) if isinstance(year_str, str) and year_str.isdigit() else None
        )

        document = DocumentSchema(
            id=paper_id,
            content=content,
            title=paper["title"],
            url=paper["url"],
            authors=extract_authors(paper),  # Clean list from structured data
            year=year,
            venue=paper.get("venue"),  # Full venue name for facets
            booktitle=paper.get("booktitle"),
            publisher=paper.get("publisher"),
            bibtex=paper.get("bibtex"),  # Pre-generated BibTeX
            pdf_url=paper.get("pdf"),  # Direct PDF link
            doi=paper.get("doi"),
            pages=paper.get("pages"),
        )

        documents.append(document)

    logging.info(f"Found {len(documents)} documents to index")

    chunker = MarkdownChunkerProvider(
        min_chunk_size=500,
        max_chunk_size=5000,
    )
    dense_provider = SentenceTransformersProvider(
        model_name="ibm-granite/granite-embedding-english-r2", device=args.device
    )

    if args.use_cloud:
        logging.info(f"Using CloudMilvusStore at {args.cloud_uri}")
        vector_store = CloudMilvusStore(
            uri=args.cloud_uri,
            token=args.milvus_token,
            collection_name=args.collection_name,
            enable_dense=True,
            enable_sparse=False,
            enable_full_text=True,
            dense_dim=dense_provider.get_dimension(),
            nlist=32768,
        )
    else:
        logging.info(f"Using LocalMilvusStore at {args.index_file}")
        vector_store = LocalMilvusStore(
            db_path=args.index_file,
            collection_name=args.collection_name,
            enable_dense=True,
            enable_sparse=False,
            dense_dim=dense_provider.get_dimension(),
            nlist=32768,
        )

    index = VerbatimIndex(
        vector_store=vector_store,
        dense_provider=dense_provider,
        chunker_provider=chunker,
    )
    logging.info("Chunking and indexing documents...")
    index.add_documents(documents)

    return index


def get_args():
    parser = argparse.ArgumentParser(description="Preprocess ACL Anthology papers")
    parser.add_argument(
        "--metadata-file",
        required=True,
        help="Path to paper metadata file (JSONL format)",
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
    parser.add_argument(
        "--milvus-token",
        help="Authentication token for Milvus connection",
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
