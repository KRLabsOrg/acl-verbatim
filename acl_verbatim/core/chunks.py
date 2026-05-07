import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=256)
def load_paper_chunks(chunks_dir: str, paper_id: str):
    file_path = Path(chunks_dir) / f"{paper_id}.json"
    if not file_path.exists():
        return None
    mapping = {}
    with open(file_path) as f:
        for line in f:
            chunk = json.loads(line)
            mapping[chunk["chunk_index"]] = chunk.get("chunk", "")
    return mapping


def get_local_chunk_text(chunks_dir: str, paper_id: str, chunk_index: int):
    mapping = load_paper_chunks(chunks_dir, paper_id)
    if mapping is None:
        return None
    return mapping.get(chunk_index)


def build_milvus_client(milvus_uri: str, milvus_token: str | None = None):
    from pymilvus import MilvusClient

    kwargs = {"uri": milvus_uri}
    if milvus_token:
        kwargs["token"] = milvus_token
    return MilvusClient(**kwargs)


def fetch_chunk_from_milvus(
    client, collection_name: str, paper_id: str, chunk_index: int
):
    filter_expr = (
        f'document_id == "{paper_id}" and metadata["chunk_number"] == {chunk_index}'
    )
    results = client.query(
        collection_name=collection_name,
        filter=filter_expr,
        output_fields=["text", "metadata", "document_id"],
        limit=1,
    )
    if not results:
        filter_expr = (
            f'metadata["document_id"] == "{paper_id}" and '
            f'metadata["chunk_number"] == {chunk_index}'
        )
        results = client.query(
            collection_name=collection_name,
            filter=filter_expr,
            output_fields=["text", "metadata"],
            limit=1,
        )
    if not results:
        return None
    return results[0].get("text", "")


@dataclass
class ChunkResolver:
    collection_name: str
    chunks_dir: str | None = None
    milvus_uri: str | None = None
    milvus_token: str | None = None

    def __post_init__(self):
        self._milvus_client = None
        if self.milvus_uri:
            self._milvus_client = build_milvus_client(
                self.milvus_uri,
                self.milvus_token,
            )

    def get(self, paper_id: str, chunk_index: int):
        text = None
        if self.chunks_dir:
            text = get_local_chunk_text(self.chunks_dir, paper_id, chunk_index)
        if text is None and self._milvus_client is not None:
            text = fetch_chunk_from_milvus(
                self._milvus_client,
                self.collection_name,
                paper_id,
                chunk_index,
            )
        return text
