import argparse
import json
import random


def _keep_paper(paper):
    if paper["lang"] != "en":
        return False
    if not paper["authors"]:
        # removing volumes
        return False
    return True


def filter_papers(papers):
    return [paper for paper in papers if _keep_paper(paper)]


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
        data = json.load(f)
        filtered_papers = filter_papers(data)
    with open(args.output_file, "w") as of:
        random.seed(args.seed)
        print(
            f"randomly choosing {args.n} papers from {len(data)}, random seed is {args.seed}"
        )
        sample = random.sample(filtered_papers, args.n)
        json.dump(sample, of)


if __name__ == "__main__":
    main()
