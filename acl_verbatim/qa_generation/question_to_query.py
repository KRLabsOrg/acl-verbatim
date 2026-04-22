import argparse
import json
import traceback
from pathlib import Path
import os

from tqdm import tqdm

from verbatim_rag.llm_client import LLMClient


def get_args():
    parser = argparse.ArgumentParser(description="Generate QA data")
    parser.add_argument("--input-dir", required=True, help="path to classified chunks")
    parser.add_argument("--output-dir", required=True, help="output path")
    parser.add_argument(
        "--model",
        default=os.environ.get("OPENAI_MODEL", "moonshotai/kimi-k2-instruct-0905"),
        help="OpenAI-compatible model name",
    )
    parser.add_argument(
        "--api-base",
        default=os.environ.get("OPENAI_API_BASE", "https://api.groq.com/openai/v1/"),
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
    return f"""You are a researcher using a search engine to find information.
**Your question**: {question}

Please generate a search query that you would use to find the answer to this question.

**Instructions:**
1. Only return a search query without any other information.
2. The query should be short and simple, resembling what a user might type into a search engine.
3. The query does not need to be grammatical.
"""


# 4. The query should be equivalent to the original question.


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
    for file_path in tqdm(Path(args.input_dir).rglob("*")):
        paper_id = file_path.stem
        with open(output_path / f"{paper_id}.json", "w") as of:
            with open(file_path) as f:
                for line in f:
                    try:
                        chunk_data = json.loads(line)
                        if chunk_data.get("qa") is not None:
                            for q in chunk_data["qa"]:
                                prompt = get_prompt(q["question"])
                                q["query"] = llm_client.complete(
                                    prompt, json_mode=False
                                )
                    except Exception as e:
                        traceback.print_exc()
                        chunk_data["err"] = f"{e}"

                    of.write(json.dumps(chunk_data))
                    of.write("\n")
                    # break  # while we are testing
        # break  # while we are testing


if __name__ == "__main__":
    main()
