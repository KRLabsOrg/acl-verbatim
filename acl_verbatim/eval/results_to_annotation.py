import argparse
import csv
import json

from pymilvus import MilvusClient
from tqdm import tqdm

from acl_verbatim.eval.utils import get_chunk


def get_args():
    parser = argparse.ArgumentParser(description="Generate QA data")
    parser.add_argument("--input-file", required=True, help="input JSON file")
    parser.add_argument("--output-file", required=True, help="output CSV file")
    parser.add_argument("-k", type=int, default=5, help="Top-k to process (default: 5)")
    parser.add_argument(
        "--cloud-uri",
        help="Cloud Milvus URI (e.g. http://localhost:19530)",
    )
    parser.add_argument(
        "--milvus-token",
        help="Authentication token for Milvus connection",
    )
    return parser.parse_args()


def results2csv(results, k, client):
    query = results["query"]
    rows = []
    for c, res in enumerate(results["results"]):
        if c + 1 > k:
            break
        chunk, title = get_chunk(res["url"], res["chunk_number"], client)
        for hl in res["extraction"]:
            i, j = hl["start"], hl["end"]
            assert chunk[i:j].upper() == hl["text"].upper(), f"mismatch: {chunk[i:j]=}, {hl['text']=}"
            chunk = chunk[:i] + chunk[i:j].upper() + chunk[j:]

        rows.append([query, f"{c + 1}", title, res["url"], res["chunk_number"], chunk])
    return rows


def main():
    args = get_args()
    milvus_kwargs = {"uri": args.cloud_uri}
    if args.milvus_token:
        milvus_kwargs["token"] = args.milvus_token
    client = MilvusClient(**milvus_kwargs)
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
