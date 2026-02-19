### Create annotation spreadsheet from search results

Sample command:

```bash
python acl_verbatim/eval/results_to_annotation.py --input-file sample_data/333_20260206/rag_results_dense_n512_20260206.jsonl --output-file annotation/333_20260206_dense_top5_20260218.csv -k 5 --cloud-uri INDEX_ENDPOINT --milvus-token MILVUS_TOKEN
```


### Postprocess annotations

Sample command:

```bash
python acl_verbatim/eval/process_annotations.py --cloud-uri INDEX_ENDPOINT --milvus-token MILVUS_TOKEN  --input-csv annotation/test.csv --input-json sample_data/333_20260206/rag_results_dense_n512_20260206.jsonl --output-file annotation/test.json
```


