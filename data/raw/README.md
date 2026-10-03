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

## Known issue in the released downloader (fix before relying on DS2 / DS3)

* `tools/download_raw_data.py` writes the BeIR relevance judgements to
  `data/raw/<ds>/qrels/test.tsv`, while `src/fedrevoke/data_prep.py`
  (`load_qrels`) reads `data/raw/<ds>/qrels_test.tsv`. One of the two paths
  has to change.
* The parquet file names requested by the downloader
  (`corpus.parquet`, `queries.parquet`) must be re-checked against the current
  HuggingFace layout of `BeIR/nq` and `BeIR/trec-covid`, which has been
  reorganised more than once.

DS1 (MultiHop-RAG) is unaffected: `data/processed/multihoprag/` is shipped
pre-built, so every DS1 result in the paper can be reproduced without any
download.
