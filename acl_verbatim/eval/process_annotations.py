import argparse
import csv
import json

from pymilvus import MilvusClient
from rapidfuzz import fuzz
from tqdm import tqdm

from acl_verbatim.eval.utils import get_chunk


def get_args():
    parser = argparse.ArgumentParser(description="Process annotated CSV back to JSON")
    parser.add_argument("--input-json", required=True, help="input JSONL file")
    parser.add_argument("--input-csv", required=True, help="annotated CSV file")
    parser.add_argument("--output-file", required=True, help="output JSONL file")
    parser.add_argument(
        "--cloud-uri",
        help="Cloud Milvus URI (e.g. http://localhost:19530)",
    )
    parser.add_argument(
        "--milvus-token",
        help="Authentication token for Milvus connection",
    )
    return parser.parse_args()


def read_jsonl(filepath):
    with open(filepath) as f:
        return [json.loads(line) for line in f]


def read_csv(filepath):
    """Read annotated CSV file.

    Expected columns: query, rank, title, url, chunk_number, chunk,
                      relevance_label, extraction_label, gold_extraction
    Empty rows separate different queries.
    """
    rows = []
    with open(filepath, newline="") as csvfile:
        reader = csv.reader(csvfile)
        for line_num, row in enumerate(reader, start=1):
            # Skip empty rows (used as separators between queries)
            if not row or not row[0].strip():
                continue
            if len(row) < 9:
                raise ValueError(f"Line {line_num}: expected 9 fields, got {len(row)}")
            rows.append(
                {
                    "query": row[0],
                    "rank": int(row[1]),
                    "title": row[2],
                    "url": row[3],
                    "chunk_number": int(row[4]),
                    "chunk": row[5],
                    "relevance_label": row[6],
                    "extraction_label": row[7],
                    "gold_extraction": [s for s in row[8].split("\n\n") if s.strip()],
                }
            )
    return rows


def _validate_and_fill(row, result):
    """Validate annotation fields and fill gold_extraction where applicable.

    Rules:
    - relevance_label empty or 'n': extraction_label and gold_extraction must be empty
    - relevance_label 'r':
        - extraction_label == 'c': gold_extraction must be empty; fill it from result['extraction']
        - extraction_label == 'p' or 'n': gold_extraction must be non-empty
        - any other extraction_label (including empty): raise ValueError
    - any other relevance_label: raise ValueError
    """
    relevance = row["relevance_label"]
    extraction_label = row["extraction_label"]
    gold = row["gold_extraction"]
    ctx = f"query={row['query']!r}, rank={row['rank']}, url={row['url']!r}"

    if "?" in relevance or "?" in extraction_label:
        row["gold_extraction"] = []
        return

    if relevance in ("", "n"):
        if extraction_label:
            raise ValueError(
                f"Expected empty extraction_label for relevance_label={relevance!r} ({ctx})"
            )
        if gold:
            raise ValueError(
                f"Expected empty gold_extraction for relevance_label={relevance!r} ({ctx})"
            )
    elif relevance == "r":
        if extraction_label == "c":
            if gold:
                raise ValueError(
                    f"Expected empty gold_extraction for extraction_label={extraction_label!r} ({ctx})"
                )
            row["gold_extraction"] = [hl["text"] for hl in result["extraction"]]
        elif extraction_label in ("p", "n"):
            if not gold:
                raise ValueError(
                    f"Expected non-empty gold_extraction for extraction_label={extraction_label!r} ({ctx})"
                )
        else:
            raise ValueError(
                f"Invalid extraction_label={extraction_label!r} for relevance_label='r' ({ctx})"
            )
    else:
        raise ValueError(f"Invalid relevance_label={relevance!r} ({ctx})")


def process_annotations(json_data, csv_rows):
    """Match CSV rows to JSON entries and add annotation data.

    Returns updated JSON data with annotations added.
    """
    # Build a lookup from (query, rank, url, chunk_number) to result object
    json_lookup = {}
    for entry in json_data:
        query = entry["query"]
        for rank, result in enumerate(entry["results"], start=1):
            key = (query, rank, result["url"], result["chunk_number"])
            json_lookup[key] = result

    # Process each CSV row, raising if no matching JSON entry is found
    for row in csv_rows:
        key = (row["query"], row["rank"], row["url"], row["chunk_number"])
        if key not in json_lookup:
            raise ValueError(
                f"CSV row cannot be matched to JSON: query={row['query']!r}, "
                f"rank={row['rank']}, url={row['url']!r}, chunk_number={row['chunk_number']}"
            )
        result = json_lookup[key]
        _validate_and_fill(row, result)
        result["relevance_label"] = row["relevance_label"]
        result["extraction_label"] = row["extraction_label"]
        result["gold_extraction"] = row["gold_extraction"]

    filtered = []
    for entry in json_data:
        annotated = [r for r in entry["results"] if r.get("relevance_label")]
        if annotated:
            entry["results"] = annotated
            filtered.append(entry)
    return filtered


def map_gold_extractions(json_data, client):
    """For each annotated result, retrieve the chunk text, fuzzy-match gold_extraction
    against it, and store the chunk text and match (text, start, end) on the result.
    """
    for entry in tqdm(json_data):
        for result in entry["results"]:
            chunk, _ = get_chunk(result["url"], result["chunk_number"], client)
            result["chunk"] = chunk
            gold_list = result.get("gold_extraction", [])
            if not gold_list:
                continue
            mapped = []
            for gold in gold_list:
                alignment = fuzz.partial_ratio_alignment(gold.lower(), chunk.lower())
                score = round(alignment.score / 100.0, 4)
                matched_text = chunk[alignment.dest_start : alignment.dest_end]
                if score < 0.9:
                    print(
                        f"WARNING: low fuzzy match score {score:.4f}\n"
                        f"  original: {gold!r}\n"
                        f"  matched:  {matched_text!r}"
                    )
                mapped.append(
                    {
                        "text": matched_text,
                        "start": alignment.dest_start,
                        "end": alignment.dest_end,
                        "score": score,
                    }
                )
            result["gold_extraction_mapped"] = mapped
    return json_data


def write_jsonl(data, filepath):
    with open(filepath, "w") as f:
        for entry in data:
            f.write(json.dumps(entry) + "\n")


def main():
    args = get_args()

    json_data = read_jsonl(args.input_json)
    csv_rows = read_csv(args.input_csv)

    updated_data = process_annotations(json_data, csv_rows)

    milvus_kwargs = {"uri": args.cloud_uri}
    if args.milvus_token:
        milvus_kwargs["token"] = args.milvus_token
    client = MilvusClient(**milvus_kwargs)

    updated_data = map_gold_extractions(updated_data, client)

    write_jsonl(updated_data, args.output_file)


if __name__ == "__main__":
    main()
