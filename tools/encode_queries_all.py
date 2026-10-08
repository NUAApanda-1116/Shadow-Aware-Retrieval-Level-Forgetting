# -*- coding: utf-8 -*-
"""Encode queries with the paper encoder for any processed dataset.

Writes data/processed/<name>/embeddings_queries_bge-small-en-v1.5.npy so that
run_experiments uses real query embeddings instead of the gold-centroid proxy.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
ENCODER = "BAAI/bge-small-en-v1.5"
OUT_NAME = "embeddings_queries_bge-small-en-v1.5.npy"


def encode_one(name: str, device: str | None = None) -> None:
    d = PROCESSED / name
    qpath = d / "queries.jsonl"
    if not qpath.exists():
        print(f"[skip] {name}: no queries.jsonl")
        return
    out = d / OUT_NAME
    if out.exists() and out.stat().st_size > 0:
        arr = np.load(str(out))
        print(f"[exists] {name}: {OUT_NAME} shape={arr.shape}")
        return

    queries = []
    with qpath.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                queries.append(json.loads(line))
    texts = [(q.get("query") or "").strip() for q in queries]
    print(f"[{name}] n_queries={len(texts)}", flush=True)

    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(ENCODER, device=device)
    t0 = time.time()
    vecs = model.encode(
        texts, batch_size=256, convert_to_numpy=True,
        normalize_embeddings=True, show_progress_bar=False,
    )
    arr = np.asarray(vecs, dtype=np.float32)
    np.save(str(out), arr)
    meta = {
        "encoder": ENCODER, "dim": int(arr.shape[1]), "n": int(arr.shape[0]),
        "normalized": True, "dtype": "float32",
        "row_order": "queries.jsonl 行序",
        "seconds": round(time.time() - t0, 2),
    }
    (d / "query_embedding_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[{name}] wrote {out.name} shape={arr.shape} "
          f"norm_mean={float(np.linalg.norm(arr,axis=1).mean()):.4f} ({meta['seconds']}s)",
          flush=True)


if __name__ == "__main__":
    which = sys.argv[1:] or ["multihoprag", "nq", "trec-covid"]
    for w in which:
        encode_one(w)
    print("done")
