import argparse
import json
from pathlib import Path

from tqdm import tqdm

from verbatim_rag.llm_client import LLMClient

BASE = Path(__file__).parent
with open(BASE / "question_type.json") as f:
    QUESTION_TYPES = json.load(f)["QuestionTypes"]


def get_args():
    parser = argparse.ArgumentParser(description="Generate QA data")
    parser.add_argument("--input-dir", required=True, help="path to classified chunks")
    parser.add_argument("--output-dir", required=True, help="output path")
    return parser.parse_args()


def get_prompt(chunk, q_type):
    for qt in QUESTION_TYPES:
        if qt["Type"] == q_type:
            q_def = qt["Definition"]
            q_ex = qt["Examples"]
            break
    else:
        raise ValueError(f"invalid question type: {q_type}")

    return f"""You are a researcher asking questions aiming to find information in research papers.

**Content of paper**: {chunk}

Please generate one question that can be answered by the above text and which belongs to the question type below.

- **Question Type**: {q_type}
- **Question Description**: {q_def}
- **Question Example**: {q_ex}

**Instructions:**
1. Only return a question without any other information.
2. Use neutral terms like "a dataset," "data collection method," or "research approach," instead of specific references like "the study," "this dataset" or "dataset mentioned".
3. The question should be short and simple, resembling what a user might type into a search engine.
4. The question should be answerable based on the text above.

"""


def main():
    args = get_args()

    llm_client = LLMClient(
        model="moonshotai/kimi-k2-instruct-0905",
        api_base="https://api.groq.com/openai/v1/",
    )

    output_path = Path(args.output_dir)
    for file_path in tqdm(Path(args.input_dir).rglob("*")):
        paper_id = file_path.stem
        with open(output_path / f"{paper_id}.json", "w") as of:
            with open(file_path) as f:
                for line in f:
                    try:
                        chunk_data = json.loads(line)
                        if "q_types" in chunk_data:
                            chunk_data["paper_id"] = paper_id
                            chunk_data["qa"] = []
                            for q_type in chunk_data["q_types"]:
                                prompt = get_prompt(chunk_data["chunk"], q_type)
                                response = llm_client.complete(prompt, json_mode=False)
                                chunk_data["qa"].append(
                                    {"q_type": q_type, "question": response}
                                )
                    except Exception as e:
                        chunk_data["err"] = f"{e}"

                    of.write(json.dumps(chunk_data))
                    # break  # while we are testing
        # break  # while we are testing


if __name__ == "__main__":
    main()
