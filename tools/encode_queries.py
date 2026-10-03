"""Encode queries with the paper encoder and save embeddings next to processed data."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed" / "multihoprag"
ENCODER = "BAAI/bge-small-en-v1.5"


def main() -> int:
    qpath = PROCESSED / "queries.jsonl"
    queries = []
    with qpath.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                queries.append(json.loads(line))
    texts = [(q.get("query") or "").strip() for q in queries]
    print(f"n_queries={len(texts)}")

    from sentence_transformers import SentenceTransformer

    st = SentenceTransformer(ENCODER, device="cuda")
    vecs = st.encode(
        texts,
        batch_size=256,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    vecs = np.asarray(vecs, dtype=np.float32)
    out = PROCESSED / "embeddings_queries_bge-small-en-v1.5.npy"
    np.save(out, vecs)
    print("saved", out, vecs.shape, vecs.dtype)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
