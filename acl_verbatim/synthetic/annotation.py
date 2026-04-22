import json


def span_annotation_prompt(question: str, chunk: str, max_spans: int) -> str:
    return f"""You are annotating minimal evidence spans in a research paper excerpt.

Return ONLY valid JSON with this schema:
{{
  "answerable": true/false,
  "spans": [
    {{"start": int, "end": int, "text": "exact substring from the excerpt"}}
  ]
}}

Rules:
- Spans must be minimal and directly support the answer.
- Spans must be substrings of the excerpt (use exact text).
- Return at most {max_spans} spans.
- If the question cannot be answered from the excerpt, set answerable=false and spans=[].

Question: {question}

Excerpt:
{chunk}
"""


def answerable_prompt(question: str, chunk: str) -> str:
    return f"""Answer ONLY with JSON: {{"answerable": true/false}}

Question: {question}

Excerpt:
{chunk}
"""


def normalize_spans(chunk: str, spans) -> list[dict]:
    normalized = []
    for span in spans:
        if not isinstance(span, dict):
            continue
        if "start" not in span or "end" not in span or "text" not in span:
            continue
        start = int(span["start"])
        end = int(span["end"])
        text = str(span["text"])
        if start < 0 or end <= start or end > len(chunk):
            if not text:
                continue
            idx = chunk.find(text)
            if idx == -1:
                continue
            start = idx
            end = idx + len(text)
        if chunk[start:end] != text:
            if not text:
                continue
            idx = chunk.find(text)
            if idx == -1:
                continue
            start = idx
            end = idx + len(text)
        normalized.append({"start": start, "end": end, "text": text})
    return normalized


def parse_annotation_response(chunk: str, response: str):
    try:
        data = json.loads(response)
    except Exception:
        return False, []
    answerable = bool(data.get("answerable", False))
    spans = normalize_spans(chunk, data.get("spans", []))
    if not answerable:
        return False, []
    if not spans:
        return True, []
    return True, spans


def annotate_pair(llm_client, question: str, chunk_text: str, max_spans: int):
    response = llm_client.complete(
        span_annotation_prompt(question, chunk_text, max_spans),
        json_mode=True,
    )
    return parse_annotation_response(chunk_text, response)


def is_answerable_with_llm(llm_client, question: str, chunk: str) -> bool:
    response = llm_client.complete(answerable_prompt(question, chunk), json_mode=True)
    try:
        data = json.loads(response)
    except Exception:
        return True
    return bool(data.get("answerable", True))


def build_retrieval_candidates(
    row: dict,
    resolver,
    max_results_per_query: int,
    skip_missing: bool = False,
):
    query = row.get("query")
    gold_paper = row.get("gold_paper")
    gold_chunk = row.get("gold_chunk")
    if not query or gold_paper is None or gold_chunk is None:
        return []

    candidates = []
    gold_text = resolver.get(gold_paper, gold_chunk)
    if gold_text is not None:
        candidates.append(
            {
                "question": query,
                "paper_id": gold_paper,
                "chunk_index": gold_chunk,
                "chunk": gold_text,
                "source": "gold",
                "retrieval_rank": None,
                "gold_paper": gold_paper,
                "gold_chunk": gold_chunk,
            }
        )

    results = row.get("results", [])[:max_results_per_query]
    for rank, res in enumerate(results, start=1):
        paper_id = res.get("document_id")
        chunk_index = res.get("chunk_number")
        if paper_id is None or chunk_index is None:
            continue
        if paper_id == gold_paper and chunk_index == gold_chunk:
            continue
        chunk_text = resolver.get(paper_id, chunk_index)
        if chunk_text is None:
            if skip_missing:
                continue
            chunk_text = ""
        candidates.append(
            {
                "question": query,
                "paper_id": paper_id,
                "chunk_index": chunk_index,
                "chunk": chunk_text,
                "source": "retrieved",
                "retrieval_rank": rank,
                "gold_paper": gold_paper,
                "gold_chunk": gold_chunk,
            }
        )
    return candidates


def silver_record(
    candidate: dict,
    spans: list[dict],
    answerable: bool,
    predicted_texts: list[str] | None = None,
    latency_s: float | None = None,
    err: str | None = None,
):
    return {
        "question": candidate["question"],
        "paper_id": candidate["paper_id"],
        "chunk_index": candidate["chunk_index"],
        "chunk": candidate["chunk"],
        "label": 1 if (candidate["source"] == "gold" or answerable) else 0,
        "answerable": answerable,
        "spans": spans,
        "source": candidate["source"],
        "retrieval_rank": candidate["retrieval_rank"],
        "gold_paper": candidate["gold_paper"],
        "gold_chunk": candidate["gold_chunk"],
        "predicted_texts": predicted_texts or [],
        "latency_s": latency_s,
        "err": err,
    }
