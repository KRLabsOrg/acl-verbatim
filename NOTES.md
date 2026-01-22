### 2026.01.22
- for some very specific queries we get 4/5 bad results (`recursive embeddings time series forecasting without retraining`, `recursive embedding model predict future without retrain`)
- for a long but generic query the chunk and the extraction can be both relevant, but who knows if
  the result is useful (consider row 38: `BLEU perplexity model combination fixed weight
  interpolation smoothed clustering k=10`)
- choosing exact boundaries in running text seems arbitrary, treating it as the only acceptable ground truth would be wrong -> **ACTUALLY ANOTHER RANKING TASK**
- suggestion for annotating extraction spans:
    - split into sentences, list them with numbers
    - annotation is a list of numbers, specifying the subset to be highlighted
    - there should be a way to include tables, figures, and formulas
- maybe let this be the reranking in our pipeline?


### 2026.01.21
- annotating for extraction: how to choose ground truth span?
- no extraction can be a good outcome for irrelevant chunks
    - maybe even implicit signal to not show the chunk?
- so far (<20 lines) it seems that:
    - if extraction is present, it is nearly always good
    - if it is missing
        - often the chunk was not relevant
        - several times the chunk was relevant but it would really be problematic to find a span
            - In this case it would be great to fall back to sentence-level retrieval
        - only a few times could a good span be annotated

### 2026.01.13
- how could question generation include something specific to the paper? Consider the question
  "What is the auto-complete field used for in the search interface?" for a paper about INCEpTION
  (no mention of the tool in the question)


### 2025.12.15
- built larger sample (1000 papers) for benchmark generation
- will skip papers whose full text is not in the anthology. Based on URLs this appears to be less
  than 5K papers, and almost all of them are older LREC papers (2016 was the first LREC included in
  the anthology). Remaining papers are less than 300.

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

