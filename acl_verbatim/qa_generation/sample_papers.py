import argparse
import json
import random


def _keep_paper(paper):
    if "language" in paper and paper["language"] != "English":
        return False
    if "author" not in paper or len(paper["author"]) == 0:
        # removing volumes
        return False
    if "aclanthology" not in paper["url"]:
        # removing papers not available from anthology (less than 5K out of 110K, mostly LREC)
        return False
    return True


def get_args():
    parser = argparse.ArgumentParser(description="Randomly choose papers")
    parser.add_argument("--input-file", required=True, help="Input file")
    parser.add_argument("--output-file", required=True, help="Output file")
    parser.add_argument("--n", type=int, required=True, help="Sample size")
    parser.add_argument("--seed", required=True, help="Random seed")

    return parser.parse_args()


def main():
    args = get_args()
    with open(args.input_file) as f:
        filtered_papers = [
            paper for paper in (json.loads(line) for line in f) if _keep_paper(paper)
        ]
    with open(args.output_file, "w") as of:
        random.seed(args.seed)
        print(
            f"randomly choosing {args.n} papers from {len(filtered_papers)}, random seed is {args.seed}"
        )
        sample = random.sample(filtered_papers, args.n)
        json.dump(sample, of)


if __name__ == "__main__":
    main()
