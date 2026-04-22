import argparse
import json
from pathlib import Path
import os

from tqdm import tqdm

from verbatim_rag.llm_client import LLMClient


def get_args():
    parser = argparse.ArgumentParser(
        description="Generate semantic paraphrases of existing questions"
    )
    parser.add_argument(
        "--input-dir", required=True, help="Path to questions directory"
    )
    parser.add_argument("--output-dir", required=True, help="Output path")
    parser.add_argument(
        "--model",
        default=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
        help="OpenAI-compatible model name",
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
    return parser.parse_args()


def get_prompt(question):
    return f"""You are rewriting a technical research question to use different, more general terminology while preserving the meaning.

**Original question**: {question}

**Instructions**:
1. Replace specific method names (like "BERT", "FOFE", "BM25") with descriptions of what they do
2. Replace specific dataset names with descriptions of their type/domain
3. Replace technical jargon with plain language equivalents
4. Keep numbers and quantities if essential to the question
5. The rewritten question must have the SAME answer as the original
6. Do NOT start with "What is" if the original doesn't

**Example**:
- Original: "What is the STRM mechanism in BERT-based models?"
- Rewritten: "How do transformer models handle words not in their vocabulary?"

Return ONLY the rewritten question, nothing else.
"""


def main():
    args = get_args()

    llm_client = LLMClient(
        model=args.model,
        api_base=args.api_base,
        api_key=args.api_key,
        temperature=args.temperature,
    )

    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for file_path in tqdm(list(Path(args.input_dir).rglob("*.json"))):
        paper_id = file_path.stem
        output_file = output_path / f"{paper_id}.json"

        # Skip if already processed
        if output_file.exists():
            print(f"Skipping {paper_id}, already exists")
            continue

        with open(output_file, "w") as of:
            with open(file_path) as f:
                for line in f:
                    try:
                        chunk_data = json.loads(line)

                        # Process each question in the qa array
                        if chunk_data.get("qa"):
                            for q in chunk_data["qa"]:
                                if "question" in q and "semantic_question" not in q:
                                    prompt = get_prompt(q["question"])
                                    response = llm_client.complete(
                                        prompt, json_mode=False
                                    )
                                    q["semantic_question"] = response.strip()

                    except Exception as e:
                        chunk_data["err"] = f"{e}"
                        print(f"Error processing {paper_id}: {e}")

                    of.write(json.dumps(chunk_data) + "\n")


if __name__ == "__main__":
    main()
