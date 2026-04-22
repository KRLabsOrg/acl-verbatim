import argparse
import csv
import json

from pymilvus import MilvusClient
from tqdm import tqdm

from acl_verbatim.eval.utils import get_chunk


def get_args():
    parser = argparse.ArgumentParser(description="Generate QA data")
    parser.add_argument("--input-file", required=True, help="input JSON file")
    parser.add_argument("--output-file", required=True, help="output filename")
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


def _results2csv(results, k, client):
    query = results["query"]
    rows = []
    for c, res in enumerate(results["results"]):
        if c + 1 > k:
            break
        chunk, title = get_chunk(res["url"], res["chunk_number"], client)
        for hl in res["extraction"]:
            i, j = hl["start"], hl["end"]
            assert chunk[i:j].upper() == hl["text"].upper(), (
                f"mismatch: {chunk[i:j]=}, {hl['text']=}"
            )
            chunk = chunk[:i] + chunk[i:j].upper() + chunk[j:]

        rows.append([query, f"{c + 1}", title, res["url"], res["chunk_number"], chunk])
    return rows


def results2csv(results, client, args):
    with open(args.output_file, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        for i, query_results in enumerate(tqdm(results)):
            query_rows = _results2csv(query_results, args.k, client)
            for row in query_rows:
                writer.writerow(row)
            writer.writerow([])
            # if i == 10:
            #     break  # for testing


def _results2json(results, k, client):
    query = results["query"]
    for c, res in enumerate(results["results"]):
        if c + 1 > k:
            break
        chunk, title = get_chunk(res["url"], res["chunk_number"], client)
        text = f"{query} | {c + 1} | {title} | {chunk}"
        text = " ".join(line for line in (t.strip() for t in text.split("\n")) if line)
        entities = []
        for hl in res["extraction"]:
            i, j = hl["start"], hl["end"]
            assert chunk[i:j] == hl["text"], f"mismatch: {chunk[i:j]=}, {hl['text']=}"
            start = text.find(hl["text"])
            entities.append([start, start + len(hl["text"]), "EXT"])

        yield text, {"entities": entities}


def results2json(results, client, args):
    output = {
        "classes": "EXT",
        "annotations": [
            ann
            for query_results in tqdm(results[:5])
            for ann in _results2json(query_results, args.k, client)
        ],
    }
    text = "\n".join(ann[0] for ann in output["annotations"])
    with open(f"{args.output_file}.txt", "w") as outfile:
        outfile.write(text)
    with open(f"{args.output_file}.json", "w") as outfile:
        json.dump(output, outfile)


def main():
    args = get_args()
    milvus_kwargs = {"uri": args.cloud_uri}
    if args.milvus_token:
        milvus_kwargs["token"] = args.milvus_token
    client = MilvusClient(**milvus_kwargs)
    with open(args.input_file) as f:
        results = [json.loads(line) for line in f]
        results2csv(results, client, args)
        # results2json(results, client, args)


if __name__ == "__main__":
    main()
