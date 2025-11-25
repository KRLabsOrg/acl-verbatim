## QA data generation

- check [ScIRGen](https://arxiv.org/pdf/2506.11117) methodology
    - prompts and code are [here](https://github.com/ScIRGen/ScIRGen/blob/main/qa_pairs_generation.py)
    - use question type taxonomy, modify the original question types prompt to get types for chunks
    - then modify qa generation prompt and generate questions for chunks with given types
