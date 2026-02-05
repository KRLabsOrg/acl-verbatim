import argparse
import csv

from pymilvus import MilvusClient
from rapidfuzz import fuzz
from tqdm import tqdm

from acl_verbatim.eval.utils import get_paper_chunks
from acl_verbatim.utils.preprocess import preprocess_markdown


def get_args():
    parser = argparse.ArgumentParser(description="Generate QA data")
    parser.add_argument("--input-file", required=True, help="input JSON file")
    parser.add_argument("--output-file", required=True, help="output CSV file")
    parser.add_argument(
        "--cloud-uri",
        help="Cloud Milvus URI (e.g. http://localhost:19530)",
    )
    parser.add_argument(
        "--milvus-token",
        help="Authentication token for Milvus connection",
    )
    return parser.parse_args()


def check_extraction(span, url, client):
    try:
        chunks = get_paper_chunks(url, client)
    except:
        print(f"failed to get chunks for {url=}, skipping")
        return None
    best_alignment, best_chunk = None, None
    for chunk in chunks:
        alignment = fuzz.partial_ratio_alignment(span, chunk)
        if best_alignment is None or alignment.score > best_alignment.score:
            best_alignment = alignment
            best_chunk = chunk

    old_score = round(best_alignment.score / 100.0, 4)
    new_chunk = preprocess_markdown(best_chunk)
    new_span = preprocess_markdown(span)
    new_alignment = fuzz.partial_ratio_alignment(new_span, new_chunk)
    score = round(new_alignment.score / 100.0, 4)
    matched_text = new_chunk[new_alignment.dest_start : new_alignment.dest_end]
    if score == old_score:
        status = "unchanged"
    elif score > old_score:
        status = "better"
    else:
        status = "worse"
    return [status, f"{score:.4f}", f"{old_score:.4f}", new_span, matched_text, url]


def main():
    args = get_args()
    milvus_kwargs = {"uri": args.cloud_uri}
    if args.milvus_token:
        milvus_kwargs["token"] = args.milvus_token
    client = MilvusClient(**milvus_kwargs)
    with open(args.output_file, "w", newline="") as output_csvfile:
        writer = csv.writer(output_csvfile)
        with open(args.input_file, newline="") as input_csvfile:
            reader = csv.reader(input_csvfile)
            for i, row in tqdm(enumerate(reader)):
                _, span, __, url = row
                row = check_extraction(span, url, client)
                if row is not None:
                    writer.writerow(row)


if __name__ == "__main__":
    main()
