# acl-verbatim

Q&A system based on papers in the ACL Anthology and the VerbatimRAG system

## Prerequisites

### Getting PDFs
PDFs are currently scraped via the Makefile of the acl-anthology repo

### Preprocessing

Sample command to preprocess all PDFs to MDs under a top-level directory:

```bash
python scripts/preprocess_acl.py --input-dir ../acl-anthology/build/anthology-files/pdf --output-dir acl_md --metadata-file papers.json --doc-batch-size 512 --page-batch-size 1024 &> acl_logs/20251103/202511030845.log
```

You can find everyhing we have on both `neptun` and `datalab` in my home dirs, under
`projects/verbatim-rag/acl_md`

### Indexing

Sample command to chunk and index all md files in a given directory (using a GPU):

```bash
time python scripts/index_acl.py --input-dir acl_md/acl --index-file acl.db --metadata-file papers.json --collection-name acl --device cuda &> acl_log/20251103_index_acl.log
```

Use cloud Milvus instance:
```bash
python scripts/index_acl.py --input-dir acl_md/ --metadata-file papers.json --collection-name acl --device cuda --use-cloud --cloud-uri http://localhost:19530
```

### Querying

Sample command for loading an index and trying some queries

```bash
python scripts/query_index.py --index-file acl.db --device cuda  --collection-name acl
```

Using cloud Milvus instance:
```bash
python scripts/query_index.py --collection-name acl --device cuda --use-cloud --cloud-uri http://localhost:19530
```

### QA benchmark generation
New, still needs fixes, see NOTES.md

Sample 33 random papers.
```bash
python acl-verbatim/qa_generation/sample_papers.py --input-file papers.json --output-file sample_data/random_papers_33.json --n 33 --seed 20251202
```

Chunk papers and choose one random chunk that is classified based on question type, generating
three question types for each chunk.
```bash
python qa_generation/chunk_and_classify.py --input-dir ../verbatim-rag/acl_md --output-dir sample_data/chunks --papers-file sample_data/random_papers_33.json --n 1
```

Generate questions for these chunks.
```bash
python acl-verbatim/qa_generation/gen_qa.py --input-dir sample_data/chunks --output-dir sample_data/questions
```




