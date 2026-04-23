# acl-verbatim

Trustworthy question answering on the ACL Anthology using VerbatimRAG.


## Published artifacts

All reproducible artifacts are on Hugging Face:

- Corpus: [`KRLabsOrg/acl-anthology-md`](https://huggingface.co/datasets/KRLabsOrg/acl-anthology-md)
  - `metadata` config: paper metadata
  - `fulltext` config: docling-converted markdown
- Span dataset: [`KRLabsOrg/acl-verbatim-spans`](https://huggingface.co/datasets/KRLabsOrg/acl-verbatim-spans)
  - `canonical` config: silver train/dev rows plus the gold test benchmark
  - `encoder` config: pretokenized ModernBERT rows for direct token-classification training
- Trained model: [`KRLabsOrg/acl-verbatim-modernbert`](https://huggingface.co/KRLabsOrg/acl-verbatim-modernbert)
  - Loads via `AutoModel.from_pretrained(..., trust_remote_code=True)` and exposes a `.process()` API


## Using the trained model

[`KRLabsOrg/acl-verbatim-modernbert`](https://huggingface.co/KRLabsOrg/acl-verbatim-modernbert)
is a 150M-parameter token classifier that highlights supporting evidence spans in a paper chunk
given a query. On our gold benchmark it matches the word-level F1 of a 120B-parameter LLM
extractor (0.563 vs 0.561) while running in ~50 ms per (question, chunk) on a single GPU.

```python
from transformers import AutoModel

model = AutoModel.from_pretrained("KRLabsOrg/acl-verbatim-modernbert", trust_remote_code=True)
result = model.process(
    question="What is ModernBERT?",
    context="ModernBERT is a long-context encoder for NLP. It supports 8192 tokens.",
    threshold=0.2,
)
for span in result["spans"]:
    print(f"[{span['score']:.2f}] {span['text']}")
```

See the
[model card](https://huggingface.co/KRLabsOrg/acl-verbatim-modernbert) for the full `.process()`
parameter reference (threshold, min-span length, merge-gap, sentence-level scoring) and for
training details.


## Prerequisites

### Installation

```bash
pip install -e .
```

For the silver-label generation and token-classifier training:

```bash
pip install -e ".[training]"
```

For Hugging Face dataset tooling and semantic-highlighting baselines:

```bash
pip install -e ".[hf]"
```

### Downloading benchmark data

The gold extraction benchmark is kept in the repo as
[333_20260206_dense_top5_20260305.json](333_20260206_dense_top5_20260305.json) for local
reproducibility. It is the same split as `canonical/test` in
[`KRLabsOrg/acl-verbatim-spans`](https://huggingface.co/datasets/KRLabsOrg/acl-verbatim-spans).
For instructions on generating more synthetic queries, see the section on
[Generating synthetic evaluation data](#generating-synthetic-evaluation-data).

### Downloading markdown data

The markdown version of all papers is published as
[`KRLabsOrg/acl-anthology-md`](https://huggingface.co/datasets/KRLabsOrg/acl-anthology-md).
Materialize it as the local file layout the scripts expect:

```bash
python scripts/export_hf_corpus.py --output-metadata-file paper_data.jsonl --output-md-dir acl_md
```

This gives you `paper_data.jsonl` (JSONL metadata) and `acl_md/*.md` (markdown fulltext), sufficient
for indexing and all downstream experiments. For instructions on rebuilding this data locally, see
the sections on [Extracting paper metadata](#extracting-paper-metadata) and
[Obtaining and preprocessing PDFs](#obtaining-and-preprocessing-pdfs).

---

## Indexing

Chunk markdown files and build a Milvus vector index. `DEVICE` can be set to `cuda` for GPU and
`cpu` for CPU. Indexing to a file using `LocalMilvusStore` is also possible, but we recommend
`CloudMilvusStore`; for a locally hosted Milvus instance, `CLOUD_URI` can be set to e.g.
`http://localhost:19530`.

Indexing to `CloudMilvusStore`:

```bash
python scripts/index_acl.py --input-dir acl_md --metadata-file paper_data.jsonl --collection-name acl --device cuda --use-cloud --cloud-uri CLOUD_URI
```

Indexing to file using `LocalMilvusStore`:

```bash
python scripts/index_acl.py --input-dir acl_md --index-file acl.db --metadata-file paper_data.jsonl --collection-name acl --device DEVICE
```

---

## Testing the index interactively

Load a local index and try some queries interactively:

```bash
python acl_verbatim/eval/test_index.py --index-file acl.db --device DEVICE --collection-name acl
```

Using a cloud Milvus instance:

```bash
python acl_verbatim/eval/test_index.py --collection-name acl --device DEVICE --use-cloud --cloud-uri CLOUD_URI
```

Retrieval only (no LLM answer generation on every query):

```bash
python acl_verbatim/eval/test_index.py --collection-name acl --device DEVICE --use-cloud --cloud-uri CLOUD_URI -r
```

---

## Retrieval evaluation

Run queries against the index, store results, and print retrieval metrics. `SEARCH_TYPE` can be
`dense`, `sparse`, `hybrid`, `full_text`, or `auto` (default).

**CAUTION:** without the `-r` option the script will also run extraction on all retrieved chunks,
which is slow and LLM-costly.

```bash
python acl_verbatim/eval/test_index.py --collection-name acl --device cpu --use-cloud --cloud-uri CLOUD_URI -r -k 500 --questions-dir QUERIES_PATH --query-field query --output-file SEARCH_RESULTS_FILE -s SEARCH_TYPE | tee METRICS_FILE
```

Compare two sets of search results at the chunk level for the top-k results:

```bash
python acl_verbatim/eval/compare_results.py SEARCH_RESULTS_FILE_1 SEARCH_RESULTS_FILE_2 -k TOP_K_TO_COMPARE
```

---

## Extractor evaluation on the gold benchmark

The gold benchmark is 20 queries × 5 retrieved chunks (47 relevant rows, 78 gold spans). See
[acl_verbatim/eval/README.md](acl_verbatim/eval/README.md) for supported prediction formats.
Committed per-row predictions for every reported system live under
[artifacts/eval/gold_extraction/](artifacts/eval/gold_extraction/) and can be rescored at any time.

### LLM extractors

Uses any OpenAI-compatible endpoint via the standard trio of environment variables:

```bash
export OPENAI_API_BASE=https://api.openai.com/v1   # or any compatible endpoint
export OPENAI_API_KEY=sk-...
export OPENAI_MODEL=gpt-4o-mini

python acl_verbatim/eval/evaluate_extractor.py --gold-file 333_20260206_dense_top5_20260305.json --output-file <model>.gold_eval.json
```

Add `--extraction-prompt-file acl_verbatim/prompts/extraction_paragraph.txt` to reproduce the
paragraph-prompt variant used as our silver teacher.

### Encoder / pruning baselines

Provence:

```bash
python acl_verbatim/eval/run_provence.py --gold-file 333_20260206_dense_top5_20260305.json --output-file provence.jsonl
```

Zilliz Semantic Highlight (token-span mode at threshold 0.3, our best configuration):

```bash
python acl_verbatim/eval/run_semantic_highlight.py --gold-file 333_20260206_dense_top5_20260305.json --output-file zilliz_spans_03.jsonl --output-mode spans --threshold 0.3
```

### Student token classifier

See [Using the trained model](#using-the-trained-model) for the programmatic API. To reproduce
the paper's headline gold-benchmark numbers end-to-end:

```bash
python acl_verbatim/span_training/evaluate_token_cls.py --hf-dataset KRLabsOrg/acl-verbatim-spans --hf-config canonical --gold-split test --model-dir KRLabsOrg/acl-verbatim-modernbert --threshold 0.2 --min-span-chars 10 --merge-gap-chars 20 --output-file acl-verbatim-modernbert.gold_eval.json
```

### Score any predictions file or compare runs

Score a pre-computed predictions file:

```bash
python acl_verbatim/eval/evaluate_predictions.py --gold-file 333_20260206_dense_top5_20260305.json --pred-file <system>.jsonl --output-file <system>.gold_eval.json
```

Print a one-row-per-system comparison across any number of runs:

```bash
python acl_verbatim/eval/compare_span_runs.py --gold-file 333_20260206_dense_top5_20260305.json --run qwen=artifacts/eval/gold_extraction/qwen_paragraph_gold.json --run provence=artifacts/eval/gold_extraction/provence_gold.jsonl --run modernbert=artifacts/eval/gold_extraction/gte-reranker.thr_0.2_merged.json
```

---

## Training a student extractor

The published model was fine-tuned from `Alibaba-NLP/gte-reranker-modernbert-base` on the
pretokenized `encoder` split of `KRLabsOrg/acl-verbatim-spans`:

```bash
python acl_verbatim/span_training/train_token_cls.py --hf-dataset KRLabsOrg/acl-verbatim-spans --hf-config encoder --train-split train --eval-split validation --model Alibaba-NLP/gte-reranker-modernbert-base --output-dir acl-verbatim-modernbert --batch-size 8 --lr 2e-5 --epochs 5 --label-scheme binary
```

Publish a fine-tune of your own with the same `AutoModel.from_pretrained(..., trust_remote_code=True)`
API as our release:

```bash
python scripts/push_model.py --model-dir acl-verbatim-modernbert --repo-id <your-namespace>/acl-verbatim-modernbert
```

---

## Additional steps

### Extracting paper metadata

An up-to-date clone of
[acl-anthology](https://github.com/acl-org/acl-anthology) is necessary to obtain paper metadata.
Since `get_anthology_metadata.py` relies on Python code from `acl-anthology` that cannot be
installed as part of a package, the path to the repository must be passed via `--anthology-path`:

```bash
python scripts/get_anthology_metadata.py --anthology-path /path/to/acl-anthology --output-file paper_data.jsonl
```

### Obtaining and preprocessing PDFs

PDFs of ACL Anthology papers can be downloaded via the
[acl-anthology](https://github.com/acl-org/acl-anthology) repository, which provides detailed
instructions.

**NOTE:** the markdown version of all papers is sufficient to perform most experiments;
downloading PDFs is only required if you need to rerun the conversion step. We kindly ask that you
observe the ACL Anthology's request not to download large amounts of data unnecessarily. Without
their permissive policies this project would not have been possible.

PDFs can be converted to markdown via docling (batch sizes tested on a single A100 GPU):

```bash
python scripts/preprocess_acl.py --input-dir ../acl-anthology/build/anthology-files/pdf --output-dir acl_md --metadata-file paper_data.jsonl --doc-batch-size 512 --page-batch-size 1024
```

### Generating synthetic evaluation data

**Step 1** — Sample random papers:

```bash
python acl_verbatim/qa_generation/sample_papers.py --input-file paper_data.jsonl --output-file SAMPLE_PAPERS_FILE --n NO_OF_PAPERS_TO_SAMPLE --seed RANDOM_SEED
```

**Step 2** — Chunk papers and choose one random chunk per paper, classified by question type:

```bash
python acl_verbatim/qa_generation/chunk_and_classify.py --input-dir acl_md --output-dir CHUNKS_DIR --papers-file SAMPLE_PAPERS_FILE --n 1
```

**Step 3** — Generate questions for these chunks:

```bash
python acl_verbatim/qa_generation/gen_qa.py --input-dir CHUNKS_DIR --output-dir QUESTIONS_PATH
```

**Step 4** — Generate retrieval queries from questions:

```bash
python acl_verbatim/qa_generation/question_to_query.py --input-dir QUESTIONS_PATH --output-dir QUERIES_PATH
```

The file generated by Step 4 contains both questions and queries as well as chunk contents and
can therefore be used directly as input for all downstream evaluation and silver-generation steps.

### Running the full silver-supervision pipeline

The end-to-end silver pipeline (sample → retrieve → LLM-annotate → filter → prepare → train →
evaluate → publish) is documented separately in [docs/PIPELINE.md](docs/PIPELINE.md).


## License

Apache 2.0. See [LICENSE](LICENSE).

## Citation

TODO

## Acknowledgements

TODO
