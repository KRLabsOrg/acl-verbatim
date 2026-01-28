def get_chunk(paper_url, chunk_no, client):
    docs = client.query(
        collection_name="acl",
        filter=f'metadata["url"] == \'{paper_url}\' AND metadata["chunk_number"] == {chunk_no}',
        output_fields=["text", "metadata"],
    )
    assert len(docs) == 1, f"retrieved zero or several chunks: {docs=}"
    return docs[0]["text"], docs[0]["metadata"]["title"]
