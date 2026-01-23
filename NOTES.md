### 2026.01.23
- vector search (dense) results are much better than full text
- extraction now has some significant false negatives, i.e. good chunks with no extraction
    - we should wait and see whether discarding chunks without extraction is nevertheless viable or
      we need a fallback strategy (e.g. secondary index at the paragraph level) for choosing what to show from a chunk
- when annotating gold spans, I go for the longest continuous segment that is still relevant,
  potentially including a connecting sentence between two good sentences as opposed to splitting
  them up. I.e. I am looking for the best snippet as opposed to a precise list of relevant spans,
  because the latter would be much more subjective. This also allows me to mark a non-continuous
  system extraction as correct if all parts of it are relevant. The typical example is row 5, a
  more extreme example is row 15, both in [this document](https://docs.google.com/spreadsheets/d/1TpSsOljXsL-QsNlLTHWvLJ5HNHYi6ZgC2HmOI-uZTZw/edit?usp=sharing).
- there are also once again cases where a query is generic and so it is unclear which parts of a
  chunk should be extracted - but highlighting the whole chunk would be useless, since we cannot
  present that much text to the user. Here once again it could be good to have a secondary index
  that lets us "zoom in"
- this could also help with the annotation; the most time-consuming part of the manual annotation
  is the reading of long chunks to find the parts that might be relevant and need a closer look
- but the current LLM-based extraction can come up empty if the returned span was not found in the source
    - maybe first we should try an extraction method that reliably always find real spans, and redo
      the annotation - e.g. the semantic highlighting from ziliz
- when retrieval results are very bad, it is typically because they are missing a key aspect of the
  query, e.g.
    - 4/5 results for "maximum BLEU improvement clustering over baseline" are not about
  BLEU or even MT
    - 4/5 results for "BLEU perplexity model combination fixed weight interpolation smoothed clustering k=10" do not mention any kind of interpolation.
    - 4/5 results for "multimodal compact bilinear pooling vs weighted averaging visual text classification" do not mention weighted averages
  The BM25-based search rarely made such obvious errors, so clever ensembling of the two should have high potential (weighting or even just going with the vector search but discarding chunks that do not rank above some threshold with BM25)


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

