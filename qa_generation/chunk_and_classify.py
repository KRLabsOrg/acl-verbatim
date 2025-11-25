import argparse
import json
from pathlib import Path

from tqdm import tqdm

from verbatim_rag.chunker_providers import MarkdownChunkerProvider
from verbatim_rag.llm_client import LLMClient


def get_prompt(chunk):
    return f"""You are a researcher generating questions and answers to find relevant information within a specific domain. Below are the potential question types. Choose the type that best fits the field information and the user's purpose.

1. **Verification**: Verification questions seek a simple 'yes' or 'no' answer to confirm specific details.
2. **Disjunctive**: Disjunctive questions present multiple options, asking the researcher to identify which one is applicable.
3. **Concept Completion**: Concept completion questions start with 'Who?', 'What?', 'When?', or 'Where?' to prompt the identification or completion of a specific term or defined element.
4. **Example**: Example questions ask for instances that illustrate a particular scientific concept.
5. **Feature Specification**: Feature specification questions inquire about the properties or characteristics of a concept, object, or phenomenon.
6. **Quantification**: Quantification questions seek numerical or measurable information.
7. **Definition**: Definition questions ask researchers to explain the meaning of a specific term or concept.
8. **Comparison**: Comparison questions require researchers to identify similarities and/or differences between two or more scientific resources or concepts.
9. **Interpretation**: Interpretation questions ask researchers to infer underlying rules of their observed data patterns.
10. **Causal Antecedent**: Causal antecedent questions inquire about the reasons or causes behind an event, trend.
11. **Causal Consequence**: Causal consequence questions ask about the outcomes or results that follow from a specific event, trend.
12. **Goal Orientation**: Goal orientation questions investigate the objectives or intentions behind the creation of a dataset, publication, or research project.
13. **Instrumental/Procedural**: Instrumental or procedural questions ask how to achieve certain goals.
14. **Enablement**: Enablement questions focus on identifying the resources or conditions that enable an agent to perform a specific action.
15. **Expectation**: Expectation questions inquire about anticipated outcomes or reasons why expected results did not occur.
16. **Judgmental**: Judgmental questions ask researchers to express their opinions or evaluations.
17. **Assertion**: Assertion questions make a statement indicating lack of knowledge or understanding.
18. **Request/Directive**: Request or directive questions involve asking researchers to perform specific tasks, such as summarizing information, analyzing data, or conducting searches.
        
Task: Based on the following text from a research paper, return the most appropriate 3 question types that could be answered by this text. Give me the name of each type and not other information. Return ONLY valid JSON - an array of objects, no markdown or explanations.

**Text:** {chunk}"""


def get_args():
    parser = argparse.ArgumentParser(
        description="Classify paper chunks based on question types"
    )
    parser.add_argument("--input-dir", required=True, help="path to MD papers")
    parser.add_argument("--output-dir", required=True, help="output path")
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
                out = {"chunk_index": i, "chunk": e_chunk, "qa": None}
                prompt = get_prompt(e_chunk)
                try:
                    response = llm_client.complete(prompt, json_mode=True)
                    out["q_types"] = []
                    for item in json.loads(response):
                        for key in ("type", "name"):
                            if key in item:
                                out["q_types"].append(item[key])
                                break
                        else:
                            print(
                                f"WARNING, no known key in response item: {item}, skipping"
                            )
                except Exception as e:
                    out["err"] = f"{e}"
                f.write(json.dumps(out) + "\n")
                # break  # while we are testing
        # break  # while we are testing


if __name__ == "__main__":
    main()
