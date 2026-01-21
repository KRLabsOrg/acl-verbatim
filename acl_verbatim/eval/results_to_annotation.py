import argparse
import csv
import json

from pymilvus import MilvusClient
from tqdm import tqdm


def get_args():
    parser = argparse.ArgumentParser(description="Generate QA data")
    parser.add_argument("--input-file", required=True, help="input JSON file")
    parser.add_argument("--output-file", required=True, help="output CSV file")
    parser.add_argument("-k", type=int, default=5, help="Top-k to process (default: 5)")
    parser.add_argument(
        "--cloud-uri",
        help="Cloud Milvus URI (e.g. http://localhost:19530)",
    )
    return parser.parse_args()


def get_chunk(paper_url, chunk_no, chunks_dir, client):
    docs = client.query(
        collection_name="acl",
        filter=f'metadata["url"] == \'{paper_url}\' AND metadata["chunk_number"] == {chunk_no}',
        output_fields=["text", "metadata"],
    )
    assert len(docs) == 1, f"retrieved zero or several chunks: {docs=}"
    return docs[0]["text"], docs[0]["metadata"]["title"]


def results2csv(results, k, client):
    query = results["query"]
    rows = []
    for c, res in enumerate(results["results"]):
        if c + 1 > k:
            break
        chunk, title = get_chunk(res["url"], res["chunk_number"], k, client)
        for hl in res["extraction"]:
            i, j = hl["start"], hl["end"]
            assert chunk[i:j] == hl["text"], f"mismatch: {chunk[i:j]=}, {hl['text']=}"
            chunk = chunk[:i] + chunk[i:j].upper() + chunk[j:]

        rows.append([query, f"{c + 1}", title, res["url"], res["chunk_number"], chunk])
    return rows


def main():
    args = get_args()
    client = MilvusClient(uri=args.cloud_uri)
    with open(args.output_file, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        with open(args.input_file) as f:
            results = [json.loads(line) for line in f]
            for i, query_results in enumerate(tqdm(results)):
                query_rows = results2csv(query_results, args.k, client)
                for row in query_rows:
                    writer.writerow(row)
                writer.writerow([])
                # if i == 10:
                #     break  # for testing


if __name__ == "__main__":
    main()
