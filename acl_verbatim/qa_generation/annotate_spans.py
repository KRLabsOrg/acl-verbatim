import argparse
import json
import random
from pathlib import Path
import os

from tqdm import tqdm

from verbatim_rag.llm_client import LLMClient
from acl_verbatim.core.jsonl import iter_jsonl
from acl_verbatim.synthetic.annotation import (
    annotate_pair,
    is_answerable_with_llm,
)


def get_args():
    parser = argparse.ArgumentParser(
        description="Annotate question-chunk pairs with evidence spans and negatives"
    )
    parser.add_argument("--input-dir", required=True, help="Path to questions dir")
    parser.add_argument("--output-dir", required=True, help="Output path")
    parser.add_argument(
        "--model",
        default=os.environ.get("OPENAI_MODEL", "gpt-5.2"),
        help="LLM model name",
    )
    parser.add_argument(
        "--api-base",
        default=os.environ.get("OPENAI_API_BASE", "https://api.openai.com/v1"),
        help="OpenAI-compatible API base",
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("OPENAI_API_KEY"),
        help="Optional API key for the endpoint",
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--max-spans", type=int, default=3, help="Max spans to return per question"
    )
    parser.add_argument(
        "--negatives-per-positive",
        type=int,
        default=1,
        help="Number of negative chunks per positive pair",
    )
    parser.add_argument(
        "--neg-scope",
        choices=["paper", "global"],
        default="paper",
        help="Sample negatives from same paper or globally",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1337,
        help="Random seed for negative sampling",
    )
    parser.add_argument(
        "--drop-empty",
        action="store_true",
        help="Drop positives when no span is returned",
    )
    parser.add_argument(
        "--verify-negatives",
        action="store_true",
        help="Use LLM to verify that negative chunks are not answerable",
    )
    return parser.parse_args()


def load_chunks(file_path):
    return list(iter_jsonl(file_path))


def build_global_index(files):
    global_chunks = []
    for file_path in files:
        paper_id = Path(file_path).stem
        chunks = load_chunks(file_path)
        for chunk in chunks:
            global_chunks.append((paper_id, chunk))
    return global_chunks


def main():
    args = get_args()

    llm_client = LLMClient(
        model=args.model,
        api_base=args.api_base,
        api_key=args.api_key,
        temperature=args.temperature,
    )
    rng = random.Random(args.seed)

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    files = list(input_dir.rglob("*.json"))
    global_pool = None
    if args.neg_scope == "global":
        global_pool = build_global_index(files)

    for file_path in tqdm(files):
        paper_id = file_path.stem
        output_file = output_dir / f"{paper_id}.jsonl"

        chunks = load_chunks(file_path)

        # Build candidate pool for negatives
        if args.neg_scope == "paper":
            neg_pool = [(paper_id, c) for c in chunks]
        else:
            neg_pool = global_pool

        with open(output_file, "w") as of:
            for chunk in chunks:
                if not chunk.get("qa"):
                    continue

                chunk_text = chunk.get("chunk", "")
                chunk_index = chunk.get("chunk_index")

                for qi, qa in enumerate(chunk["qa"]):
                    question = qa.get("question")
                    q_type = qa.get("q_type")
                    if not question:
                        continue

                    try:
                        answerable, spans = annotate_pair(
                            llm_client, question, chunk_text, args.max_spans
                        )
                    except Exception:
                        answerable, spans = True, []

                    if args.drop_empty and (not spans or not answerable):
                        continue

                    pos_record = {
                        "paper_id": paper_id,
                        "question_id": f"{chunk_index}.{qi}",
                        "question": question,
                        "q_type": q_type,
                        "chunk_index": chunk_index,
                        "chunk": chunk_text,
                        "label": 1,
                        "answerable": answerable,
                        "spans": spans,
                        "span_source": "llm",
                    }
                    of.write(json.dumps(pos_record) + "\n")

                    # Sample negatives
                    if args.negatives_per_positive > 0:
                        neg_candidates = [
                            (pid, c)
                            for (pid, c) in neg_pool
                            if not (
                                pid == paper_id and c.get("chunk_index") == chunk_index
                            )
                        ]
                        if not neg_candidates:
                            continue
                        needed = args.negatives_per_positive
                        attempts = 0
                        max_attempts = max(10, needed * 5)
                        while needed > 0 and attempts < max_attempts:
                            attempts += 1
                            neg_pid, neg_chunk = rng.choice(neg_candidates)
                            neg_text = neg_chunk.get("chunk", "")
                            if args.verify_negatives:
                                if is_answerable_with_llm(
                                    llm_client, question, neg_text
                                ):
                                    continue
                            neg_record = {
                                "paper_id": neg_pid,
                                "question_id": f"{chunk_index}.{qi}",
                                "question": question,
                                "q_type": q_type,
                                "chunk_index": neg_chunk.get("chunk_index"),
                                "chunk": neg_text,
                                "label": 0,
                                "answerable": False,
                                "spans": [],
                                "neg_reason": "random_chunk",
                                "positive_paper_id": paper_id,
                                "positive_chunk_index": chunk_index,
                            }
                            of.write(json.dumps(neg_record) + "\n")
                            needed -= 1


if __name__ == "__main__":
    main()
