---
license: cc-by-4.0
task_categories:
  - text-retrieval
  - question-answering
  - text-generation
language:
  - en
pretty_name: ACL Anthology Markdown Corpus
size_categories:
  - 100K<n<1M
modalities:
  - text
tags:
  - acl-anthology
  - scientific-papers
  - rag
  - retrieval
configs:
  - config_name: metadata
    data_files:
      - split: train
        path: metadata/train-*
  - config_name: fulltext
    data_files:
      - split: train
        path: fulltext/train-*
---

# ACL Anthology Markdown Corpus

A snapshot of the [ACL Anthology](https://aclanthology.org/) consisting of bibliographic metadata for **120,034** papers and full-text **markdown conversions of 114,484 papers** (≈95% of the catalogue, the remainder are frontmatter, abstract-only entries, or papers without an available PDF).

This corpus is the document collection used by [ACL-Verbatim](https://github.com/krlabsorg/acl-verbatim), a hallucination-free question-answering system for NLP research papers built on top of [VerbatimRAG](https://github.com/KRLabsOrg/verbatim-rag).

## Configurations

The dataset ships as **two configs** that share the join key `anthology_id` (e.g. `2023.acl-long.42`). Use only what you need — the metadata config is small, the fulltext config is large.

```python
from datasets import load_dataset

meta = load_dataset("KRLabsOrg/acl-anthology-md", "metadata", split="train")
full = load_dataset("KRLabsOrg/acl-anthology-md", "fulltext", split="train")

# Join on anthology_id when you want both:
by_id = {row["anthology_id"]: row for row in full}
for paper in meta.filter(lambda r: r["has_markdown"]):
    md = by_id[paper["anthology_id"]]["markdown"]
```

### `metadata` (1 row per paper)

All 120,034 papers from the ACL Anthology, including frontmatter and abstract-only entries.

| field | type | notes |
|-------|------|-------|
| `anthology_id` | string | e.g. `2023.acl-long.42`. Join key. |
| `paper_id` | string | Internal Anthology numeric id. |
| `bibkey`, `bibtype`, `bibtex` | string | BibTeX key, entry type, and full record. |
| `title`, `title_html`, `title_raw` | string | Cleaned, HTML, and raw forms of the title. |
| `author` | list&lt;struct&gt; | `{id, first, last, full}` per author. |
| `author_string`, `editor` | string / list | Author display string; editors for proceedings. |
| `url`, `pdf`, `thumbnail`, `doi` | string | Anthology page, PDF, thumbnail, DOI. |
| `citation`, `citation_acl` | string | Markdown and ACL-style citations. |
| `booktitle`, `parent_volume_id`, `year`, `venue` | string / list | Venue metadata. `venue` is a list of slugs (e.g. `["acl"]`). |
| `pages`, `page_first`, `page_last` | string | Pagination, when reported. |
| `abstract_html`, `abstract_raw` | string | Available for ~72k papers. |
| `language`, `attachment` | string / list | Non-English language flag; supplementary attachments. |
| `ingest_date` | string | ISO date the record entered the Anthology. |
| `has_markdown` | bool | Convenience flag — true iff `fulltext` contains this `anthology_id`. |

### `fulltext` (1 row per converted paper)

| field | type | notes |
|-------|------|-------|
| `anthology_id` | string | Join key against `metadata`. |
| `markdown` | string | Full paper body in markdown, produced by [docling](https://github.com/DS4SD/docling). |

## Corpus statistics

| | |
|---|---|
| Papers in metadata | 120,034 |
| Papers with full-text markdown | 114,484 (95.4%) |
| Year range | 1952 – 2026 |
| Distinct venues | 500 |
| Papers with abstract | 71,902 |
| Mean authors per paper | 3.7 |
| Total markdown size | 5.10 GB raw |
| Markdown per paper (median / p90 / p99) | 37 KB / 74 KB / 162 KB |

**Top venues by paper count:** `acl` (13,664), `emnlp` (11,525), `ws` (10,714), `findings` (10,519), `lrec` (9,105), `coling` (8,701), `naacl` (5,458), `ijcnlp` (3,871), `semeval` (3,330), `jeptalnrecital` (2,766).

**Recent years:** 2025 (14,577), 2024 (12,098), 2023 (9,032), 2022 (8,649), 2021 (7,148).

## How the corpus was built

1. **Metadata extraction.** Run against a local checkout of [`acl-org/acl-anthology`](https://github.com/acl-org/acl-anthology) using its `Anthology` Python API plus the repository's own `create_hugo_data.paper_to_dict` to flatten each paper to a JSON record. One JSONL row per paper. See `scripts/corpus/get_anthology_metadata.py` in the [acl-verbatim](https://github.com/krlabsorg/acl-verbatim) repository.

2. **PDF download.** PDFs are obtained via the `acl-anthology` repository's standard download tooling. Per the Anthology's request, we did not redistribute PDFs — only the docling-converted markdown is included here.

3. **PDF → markdown conversion.** Each PDF is converted with [docling](https://github.com/DS4SD/docling)'s `DocumentConverter` in batched mode (`doc_batch_size=512`, `page_batch_size=1024`), exporting via `document.export_to_markdown()`. Conversion was run on a single A100 GPU. A small allow-list of papers is skipped because they segfault docling or hang during conversion — these papers are present in `metadata` with `has_markdown=False`. See `scripts/corpus/preprocess_acl.py`.

4. **Dataset assembly.** `scripts/corpus/build_corpus_dataset.py` walks the markdown directory, joins each file to its metadata record by `anthology_id` (derived from the paper URL), normalizes the metadata schema, and writes both configs.

The conversion is automated and not manually validated — markdown quality varies with the underlying PDF (older scanned papers, complex tables, math-heavy papers may convert imperfectly). Treat the output as a strong starting point for chunking and retrieval, not as a verbatim transcription.

## Intended uses

- Retrieval-augmented generation over NLP research literature.
- Training and evaluating extractive QA / citation-grounding systems on scientific text.
- Bibliometric and meta-research studies of the NLP community.

## Out-of-scope / known limitations

- **OCR / layout artefacts.** Tables, figures, and equations are converted on a best-effort basis by docling and may be malformed.
- **Coverage gaps.** ~5% of papers lack markdown (frontmatter, withdrawn papers, conversion failures, an explicit skip list).
- **Language.** The catalogue is overwhelmingly English; ~5,000 papers carry an explicit non-English `language` tag (notably French, Chinese, German). No language detection is run on the markdown body.
- **Licensing per paper.** The Anthology aggregates papers under a mix of licenses. The aggregate metadata is released here under CC-BY-4.0; **individual paper texts retain the rights of their original authors and publishers**. Downstream users are responsible for verifying per-paper licensing for any redistribution.

## Citation

A paper describing ACL-Verbatim is in preparation; citation details will be added here once
available. In the meantime, please also cite the [ACL Anthology](https://aclanthology.org/).

## Acknowledgements

This work would not have been possible without the [ACL Anthology](https://aclanthology.org/) and the maintainers of the [acl-anthology](https://github.com/acl-org/acl-anthology) repository, whose tooling and permissive policies enabled both the metadata extraction and the PDF-to-markdown conversion at scale.
