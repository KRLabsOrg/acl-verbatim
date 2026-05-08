import json
import logging
import os
from pathlib import Path
from typing import Optional

from tqdm import tqdm
from verbatim_rag import BaseReranker, VerbatimIndex, VerbatimRAG
from verbatim_rag.core import LLMClient
from verbatim_rag.embedding_providers import SentenceTransformersProvider
from verbatim_rag.extractors import SemanticHighlightExtractor
from verbatim_rag.vector_stores import CloudMilvusStore, LocalMilvusStore
from verbatim_rag.vector_stores.base import SearchResult

HYBRID_WEIGHTS = {"dense": 0.5, "full_text": 0.5}


def build_reranker(args) -> Optional[BaseReranker]:
    if not args.rerank:
        return None

    from verbatim_rag import SentenceTransformersReranker

    return SentenceTransformersReranker(
        model="jinaai/jina-reranker-v2-base-multilingual",
        device=args.device,
        rerank_k=args.k,
        text_field="enhanced_text",
    )


def query_index(
    index: VerbatimIndex,
    question: str,
    search_type: str,
    k: int,
    search_params: dict,
    hybrid_weights: dict,
    reranker: Optional[BaseReranker],
) -> list[SearchResult]:
    retrieve_k = k
    if reranker is not None:
        retrieve_k = max(retrieve_k, int(getattr(reranker, "rerank_k", retrieve_k)))

    results = index.query(
        text=question,
        k=retrieve_k,
        rrf_k=60,
        hybrid_weights=hybrid_weights,
        search_type=search_type,
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
) -> dict:
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
        search_results = query_index(
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


def iter_rag_results(
    index: VerbatimIndex,
    args,
    reranker: Optional[BaseReranker],
    rag: Optional[VerbatimRAG],
):
    for file_path in tqdm(args.questions_dir.rglob("*")):
        paper_id = file_path.stem
        with open(file_path) as f:
            for line in f:
                chunk_data = json.loads(line)
                if not chunk_data["qa"]:
                    continue
                chunk_index = chunk_data["chunk_index"]
                for q in chunk_data["qa"]:
                    yield get_results_for_query(
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


def get_index(args) -> VerbatimIndex:
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
            db_path=str(args.index_file),
            collection_name=args.collection_name,
            enable_dense=True,
            enable_sparse=False,
            dense_dim=dense_provider.get_dimension(),
            nlist=32768,
        )

    return VerbatimIndex(
        vector_store=vector_store,
        dense_provider=dense_provider,
    )


def get_llm_client():
    print(os.environ.get("OPENAI_MODEL", "moonshotai/kimi-k2-instruct-0905"))
    print(os.environ.get("OPENAI_API_BASE", "https://api.groq.com/openai/v1/"))
    print(os.environ.get("OPENAI_API_KEY") )
    
    return LLMClient(
        model=os.environ.get("OPENAI_MODEL", "moonshotai/kimi-k2-instruct-0905"),
        api_base=os.environ.get("OPENAI_API_BASE", "https://api.groq.com/openai/v1/"),
        api_key=os.environ.get("OPENAI_API_KEY"),
    )


def get_extractor(args):
    if args.extractor == "LLM":
        return None
    if args.extractor == "SHL":
        return SemanticHighlightExtractor(output_mode="sentences")
    raise ValueError(f"unsupported extractor: {args.extractor}")


def get_rag(
    index: VerbatimIndex, args, reranker: Optional[BaseReranker]
) -> VerbatimRAG:
    print("initializing RAG...")
    extractor = get_extractor(args)
    llm_client = get_llm_client()
    return VerbatimRAG(
        index,
        extractor=extractor,
        llm_client=llm_client,
        k=args.k,
        reranker=reranker,
        template_mode="static",
    )
