# -*- coding: utf-8 -*-
"""Convert BeIR zip archives (corpus.jsonl/queries.jsonl/qrels/test.tsv) into
the flat layout expected by src/fedrevoke/data_prep.py:

    data/raw/<name>/corpus.parquet      columns: _id, title, text
    data/raw/<name>/queries.parquet     columns: _id, text
    data/raw/<name>/qrels_test.tsv      (flattened from qrels/test.tsv)

Streams the JSONL straight out of the zip so the 1.5 GB nq corpus never
touches disk twice.
"""
from __future__ import annotations

import io
import json
import sys
import zipfile
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

RAW = Path(__file__).resolve().parents[1] / "data" / "raw"


def convert(name: str) -> None:
    zpath = RAW / f"{name}.zip"
    out = RAW / name
    out.mkdir(parents=True, exist_ok=True)
    print(f"=== {name} ===", flush=True)

    with zipfile.ZipFile(zpath) as zf:
        names = zf.namelist()
        prefix = f"{name}/"

        # ---- qrels ----
        qrels_dest = out / "qrels_test.tsv"
        qrels_src = prefix + "qrels/test.tsv"
        with zf.open(qrels_src) as fh, open(qrels_dest, "wb") as oh:
            while True:
                chunk = fh.read(1 << 20)
                if not chunk:
                    break
                oh.write(chunk)
        print(f"  qrels_test.tsv  {qrels_dest.stat().st_size} bytes", flush=True)

        # ---- corpus -> parquet ----
        corpus_dest = out / "corpus.parquet"
        if not (corpus_dest.exists() and corpus_dest.stat().st_size > 0):
            writer = pq.ParquetWriter(
                str(corpus_dest),
                pa.schema([("_id", pa.string()), ("title", pa.string()),
                           ("text", pa.string())]),
            )
            batch_ids: list[str] = []
            batch_ttl: list[str] = []
            batch_txt: list[str] = []
            n = 0

            def flush() -> None:
                if not batch_ids:
                    return
                writer.write_table(
                    pa.table(
                        {"_id": pa.array(batch_ids, pa.string()),
                         "title": pa.array(batch_ttl, pa.string()),
                         "text": pa.array(batch_txt, pa.string())},
                    )
                )
                batch_ids.clear()
                batch_ttl.clear()
                batch_txt.clear()

            with zf.open(prefix + "corpus.jsonl") as fh:
                for raw in io.TextIOWrapper(fh, encoding="utf-8"):
                    raw = raw.strip()
                    if not raw:
                        continue
                    r = json.loads(raw)
                    batch_ids.append(str(r.get("_id", "")))
                    batch_ttl.append(str(r.get("title") or ""))
                    batch_txt.append(str(r.get("text") or ""))
                    n += 1
                    if len(batch_ids) >= 20000:
                        flush()
                        print(f"    corpus rows {n}", flush=True)
                flush()
            writer.close()
            print(f"  corpus.parquet  {n} rows, {corpus_dest.stat().st_size/1e6:.1f}MB", flush=True)
        else:
            print(f"  corpus.parquet  exists ({corpus_dest.stat().st_size/1e6:.1f}MB)", flush=True)

        # ---- queries -> parquet ----
        queries_dest = out / "queries.parquet"
        if not (queries_dest.exists() and queries_dest.stat().st_size > 0):
            ids: list[str] = []
            txts: list[str] = []
            with zf.open(prefix + "queries.jsonl") as fh:
                for raw in io.TextIOWrapper(fh, encoding="utf-8"):
                    raw = raw.strip()
                    if not raw:
                        continue
                    r = json.loads(raw)
                    ids.append(str(r.get("_id", "")))
                    txts.append(str(r.get("text") or r.get("query") or ""))
            pq.write_table(
                pa.table({"_id": pa.array(ids, pa.string()),
                          "text": pa.array(txts, pa.string())}),
                str(queries_dest),
            )
            print(f"  queries.parquet {len(ids)} rows, {queries_dest.stat().st_size/1e6:.2f}MB", flush=True)
        else:
            print(f"  queries.parquet exists", flush=True)


if __name__ == "__main__":
    which = sys.argv[1:] or ["nq", "trec-covid"]
    for w in which:
        convert(w)
    print("done")
