"""ANN-probe micro-benchmarks (paper Sec. 6.3).

Usage (from the repository root):

    python tools/probe_microbench.py

Builds an IndexIVFFlat over the ds2 (BeIR/nq) vectors (nlist=256, nprobe=16,
inner product) and reports the wall clock of a 128-query batch at 24 threads
versus 1 thread, and at k=50 versus k=150 (scan volume, not result size,
decides).
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
EMB = ROOT / "data" / "processed" / "nq" / "embeddings_bge-small-en-v1.5.npy"

import faiss  # noqa: E402


def main() -> None:
    vecs = np.load(EMB)
    if vecs.dtype != np.float32:
        vecs = vecs.astype(np.float32)
    n, d = vecs.shape

    faiss.omp_set_num_threads(24)
    quant = faiss.IndexFlatIP(d)
    index = faiss.IndexIVFFlat(quant, d, 256, faiss.METRIC_INNER_PRODUCT)
    index.train(vecs[:65536])
    index.add(vecs)
    index.nprobe = 16

    rng = np.random.default_rng(20260214)
    q = vecs[rng.choice(n, 128, replace=False)].copy()

    def bench(k: int, threads: int, repeats: int = 5) -> float:
        faiss.omp_set_num_threads(threads)
        index.search(q[:1], 1)
        t0 = time.perf_counter()
        for _ in range(repeats):
            index.search(q, k)
        return (time.perf_counter() - t0) / repeats

    t24 = bench(50, 24)
    t1 = bench(50, 1)
    tk150 = bench(150, 24)
    out = {
        "batch128_k50_24threads_s": round(t24, 6),
        "batch128_k50_1thread_s": round(t1, 6),
        "batch128_k150_24threads_s": round(tk150, 6),
        "per_query_24t_us": round(t24 / 128 * 1e6, 1),
        "per_query_1t_us": round(t1 / 128 * 1e6, 1),
        "speedup_threads": round(t1 / t24, 2),
    }
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
