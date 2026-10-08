# `data/raw/` — intentionally empty

The raw public corpora are **not** committed. Re-create them with the released
downloader:

```bash
python tools/download_raw_data.py ds1     # MultiHop-RAG  -> data/raw/multihoprag/
python tools/download_raw_data.py ds2     # BeIR/nq       -> data/raw/nq/
python tools/download_raw_data.py ds3     # BeIR/trec-covid -> data/raw/trec-covid/
python tools/download_raw_data.py all
```

then run the preparation step:

```bash
python -m fedrevoke.data_prep --datasets ds1 ds2 ds3 --silos 5 10
```

## Building DS2 / DS3

The BeIR repositories on HuggingFace now ship sharded parquet and **no qrels**,
so the original downloader no longer works. Two scripts reproduce the layout
that `src/fedrevoke/data_prep.py` expects:

```bash
# 1. original BeIR zips (contain corpus.jsonl / queries.jsonl / qrels/test.tsv)
curl -L -o data/raw/nq.zip \
  https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/nq.zip
curl -L -o data/raw/trec-covid.zip \
  https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/trec-covid.zip

# 2. convert to data/raw/<name>/{corpus,queries}.parquet + qrels_test.tsv
python tools/beir_zip_to_parquet.py nq trec-covid

# 3. process (chunk / MinHash / silos / shadows / embeddings)
python -m fedrevoke.data_prep --dataset ds2
python -m fedrevoke.data_prep --dataset ds3

# 4. query embeddings (without these the runner falls back to a gold-centroid
#    proxy and the retrieval numbers are not comparable to the paper)
python tools/encode_queries_all.py nq trec-covid
```

`tools/download_raw_data.py` has been corrected to write `qrels_test.tsv`
(flat) to match `data_prep.load_qrels`; use it only if the HuggingFace layout
still has qrels.

DS1 (MultiHop-RAG) is unaffected: `data/processed/multihoprag/` is shipped
pre-built, so every DS1 result in the paper can be reproduced without any
download.
