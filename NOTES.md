### 2025.12.02
- to fix in benchmark generation
    - lang detection should run on chunks too, not just titles
    - filter strange papers (e.g. only an image)

### 2025.11.25
- check [ScIRGen](https://arxiv.org/pdf/2506.11117) methodology
    - prompts and code are [here](https://github.com/ScIRGen/ScIRGen/blob/main/qa_pairs_generation.py)
    - use question type taxonomy, modify the original question types prompt to get types for chunks
    - then modify qa generation prompt and generate questions for chunks with given types


### 2025. 11. 10
Finished processing all anthology papers, stats:
|   |   |
|---|---|
|total PDFs | 111 640 |
|total MDs created | 107 559 |
|skipped because not listed in metadata file | 4 037 |
|fails to convert | 63 |
|empty MDs (??) | 154 |

Out of the 63 PDFs that fail to convert, 47 will fail fast with an exception, the other 16 are listed
as `TO_SKIP` because they either cause segfaults (7 papers) or take a long time to fail (9 papers)

