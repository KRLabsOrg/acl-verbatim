import argparse
import json
from pathlib import Path

from tqdm import tqdm

from verbatim_rag.chunker_providers import MarkdownChunkerProvider
from verbatim_rag.llm_client import LLMClient


# from https://github.com/MasterAI-EAM/SciQAG/blob/master/Inference.ipynb


def get_orig_prompt(paper, keywords):
    return f"Attached is a detailed scientific paper.\n\n{paper}\n\nYour task is to formulate 10 sophisticated Q&A pairs that delve into the underlying scientific principles and knowledge presented in this paper, focusing specifically on {keywords}. Steer clear of questions that are purely section-specific (e.g., 'What does Figure 5 represent?') or basic or definitional questions (e.g., 'What is XXX?'). Instead, focus on questions that require a deeper understanding of the subject matter, especially those relating to complex chemical compounds (like Al2O3, C2H5OH, TNT). Ensure diversity in your Q&A pairs, avoiding any duplication. Answers should be rich in detail, drawing on specific data, chemical properties, and contextual insights from the paper. Strive for clarity and depth in your responses, aiming to enhance the reader's comprehension of the intricate concepts discussed."


def get_prompt(chunk):
    return f"Attached is an excerpt from a scientific paper.\n\n{chunk}\n\nYour task is to formulate 3 Q&A pairs that are concerned with the facts presented in this text. The questions should be short, resembling what a user might type into a search engine. Steer clear of questions that are purely section-specific (e.g., 'What does Figure 5 represent?'). Ensure diversity in your Q&A pairs, avoiding any duplication. Return ONLY valid JSON - an array of objects, no markdown or explanations."


def get_args():
    parser = argparse.ArgumentParser(description="Generate QA data")
    parser.add_argument("--input-dir", required=True, help="path to MD papers")
    parser.add_argument("--output-dir", required=True, help="path to MD papers")
    return parser.parse_args()


def main():
    args = get_args()

    llm_client = LLMClient(
        model="moonshotai/kimi-k2-instruct-0905",
        api_base="https://api.groq.com/openai/v1/",
    )

    chunker = MarkdownChunkerProvider(
        min_chunk_size=500,
        max_chunk_size=5000,
    )

    output_path = Path(args.output_dir)
    for file_path in tqdm(Path(args.input_dir).rglob("*")):
        paper_id = file_path.stem
        content = file_path.read_text(encoding="utf-8")
        chunk_tuples = chunker.chunk(content)
        with open(output_path / f"{paper_id}.json", "w") as f:
            for i, (chunk, e_chunk) in enumerate(chunk_tuples):
                if i < 3:
                    continue
                out = {"chunk_index": i, "chunk": e_chunk, "qa": None}
                prompt = get_prompt(e_chunk)
                try:
                    response = llm_client.complete(prompt, json_mode=True)
                    out["qa"] = json.loads(response)
                except Exception as e:
                    out["err"] = f"{e}"
                f.write(json.dumps(out))
                # break  # while we are testing
        # break  # while we are testing


if __name__ == "__main__":
    main()
