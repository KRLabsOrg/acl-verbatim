### 2026.03.06
- annotation for 100 rows done
    - json on gdrive, [333_20260206_dense_top5_20260305.json](https://drive.google.com/file/d/1BJObxAeZZnSgTOkE0Fsr1EsI71SAAn3O/view?usp=drive_link)
    - stats:
        - 100 chunks (20 queries, 5 chunks per query)
        - relevant: 47, not relevant: 44, cannot be annotated: 9
        - total annotated extractions: 78

### 2026.03.05

- new preprocessing done
    - loaded metadata for 120 034 papers, kept 114 567
    - stats on skipped papers: `[('lrec', 4351), ('no_pdf', 829), ('doi', 148), ('springer', 139)]`
    - (side note: total PDFs: 118375, stats on PDFs skipped: not in metadata file 1990, specified in `TO_SKIP`: 17, we don't know what the rest of the PDFs are but don't care much, metadata file is what matters)
    - total MDs created: 114 475
    - Docling errors: PDFs failed to convert: 49, unexpected error while opening the document: 11
    - 114 475 + 49 + 11 = 114 535, leaving 114 567 - 114 535 = 32 papers unaccounted for, we should
      look into this later just for a sanity check

- more annotation considerations
    - what to do about bibliography sections? I will mark them with ? (to indicate _won't
      annotate_), because even though it might make sense to annotate them for both relevance and
      extraction, this would require that I check the listed papers for relevance. If we care about
      paper-level relevance (as opposed to chunk-level relevance), there could be explicit ways (e.g.
      derive them from chunk relevance).
    - annotation can require considerable understanding of specific topics, consider e.g. the
      chunks retrieved for the query `parsing merge predicate sequence equivalence conditions`
    - The authors note that their ability to judge the relevance of chunks and spans for a given
      query strongly depends on their level of expertise in the subject matter. In particular, only
      when the query concerns a topic that the annotator is quite familiar with, is it possible to
      correctly annotate excerpts that appear very much on-topic but are actually misleading (`red
      herrings'). This means that our dataset cannot be considered a proper gold standard for
      rigorous evaluation. We
      nevertheless consider it valuable as a proxy for determining the perceived usefulness of our
      system's outputs for its users, who we expect will also search for information outside of their
      narrowest field of expertise. (add note that a lot of things in NLP are annotated for based
      on the layman's (subjective) understanding, e.g. hate speech).
    - a dimension we are not considering but usually informs literature research is (perceived) paper
      quality, but this is intentional, our tool is not intended to automate the critical judgements of researchers taken during literature research, it should only increase their efficiency in identifying potentially relevant excerpts


### 2026.03.04
- annotation considerations
    - we always annotate full paragraphs (many examples of why it doesn't make sense to think about
      more fine-grained boundaries)
    - we don't require a span to be relevant for all parts of a complex query, otherwise we would
      have e.g. almost nothing to annotate for _multi-label hate speech dataset multiple annotators features_ or _MSIT vs GPT-4 attribute extraction precision images bullet points titles_
    - but it's different when the whole chunk is not relevant for the whole query, only the more
      generic part, consider e.g. rows 39-40 vs. row 30
      [here](https://docs.google.com/spreadsheets/d/1t1bprbngBS4j44lI-HO1O_8hVHWdfLV96qQs_Qyd4MA/edit?usp=sharing)
    - another issue with our synthesis method is when a query isn't relevant for any other paper,
      e.g. because it is about a method presented by the paper that noone cited yet (MSIT)
    - if a table is relevant, we annotate its caption but not the table itself - this should capture the fact of the table
      being relevant and we can decide later how we want to represent tables in our data. Figures,
      images, and most formulas are not rendered in markdown, we do not make assumptions on whether
      they might be relevant. When this impacts our ability to annotate the whole chunk, we mark
      the row with `?` characters to indicate _won't annotate_. Rows marked with `?!` indicate that
      there is also some preprocessing issue that will require our attention.


### 2026.02.19

- created postprocessing script for annotations
    - chunks must have a binary relevance label (r)elevant or (n)ot relevant. Chunks marked as not relevant should not have further annotation.
    - relevant chunks have a ternary extraction label: (c)orrect, (p)artial, or (n)ot correct.
    - relevant chunks with partial or not correct extraction should have a `gold_extractions` field with
      empty lines separating multiple extractions
    - gold extractions are fuzzy matched against the chunk text to create
      `gold_extractions_mapped', complete with start and end positions


### 2026.02.06

- added preprocessing to indexing
    - runs html.unescape
    - replaces multiple spaces with single space between two non-whitespace characters
    - testing: checked effect on outcome of the `check_extraction` script that runs fuzzy matching
      and prints scores. In most cases match was 1.0 after the preprocessing, in all other cases
      there was still an improvement or no change (when the reason was something else).
- reindexed ACL, logs are on datalab
- generated new sample with the existing pipeline
    - started with 333 papers (aiming for 999 queries maximum), 310 could be matched to existing md
      files (this matching is still based on filenames vs URLs, needs to be investigated)
    - question type generation failed for three chunks, one contained only images, this I removed
      from the sample, the other two were just JSON validation issues that could be fixed
      by hand
    - from 3x310=930 question types we generated 906 questions. The rest were mostly invalid question types, plus in 3 cases it was temporary groq issues
- ran retrieval evaluation on new sample
    - for top 100 instead of top 500, because now we are using ziliz cloud and with 500 the
      MilvusClient raised an error: `grpc: received message larger than max (8511262 vs.
      4194304)`
    - see `EVAL.md` for results
- reran extraction
    - added some safety rules: extraction won't run for k>=10 in batch mode
    - capacity issues on groq lead to sporadic failures
        - usually fallback to individual extraction is enough
        - but when that fails too, extractor returns empty list, just as if the LLM had responded
          with no extractions, and there is no trace in the data or the logs where this happened,
          this should be fixed unless we are OK with it



### 2026.02.02

- Fuzzy matching for span extraction: accepts near-matches using rapidfuzz, logs partial matches to CSV
- Milvus cloud authentication support via `--milvus-token`

### 2026.01.29

- used Ziliz semantic highlighter for extraction, checked results on the same sample of 10x5
  chunks
    - generally worse than the LLM-based extractor
    - performs fewer extractions, left more chunks without any highlights
    - when it does highlight, it is usually as good as the LLM, just a bit worse at boundaries
    - maybe look into thresholds?
        - but only after we have automatic eval with gold data

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

