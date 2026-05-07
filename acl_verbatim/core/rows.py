def span_prediction_key(row: dict):
    return (row.get("question"), row.get("paper_id"), row.get("chunk_index"))
