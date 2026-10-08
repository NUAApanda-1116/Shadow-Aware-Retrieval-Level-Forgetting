"""Reproduce the IndexIDMap2 remove_ids pitfall (paper Sec. 4.2).

Usage:

    python tools/idmap2_pitfall.py

A 1,500-vector test index loses 30% of its vectors; compares the
approximate/exact top-10 Jaccard of (1) the intact index, (2) the
IndexIDMap2 + remove_ids shortcut (silently wrong), and (3) a bare
IndexIVFFlat with a hand-maintained id map and tombstones (the path the
pipeline uses).
"""

from __future__ import annotations

import json

import numpy as np

import faiss


def main() -> None:
    rng = np.random.default_rng(20260214)
    n, d, n_del, k = 1500, 384, 450, 10
    V = rng.standard_normal((n, d)).astype(np.float32)
    faiss.normalize_L2(V)
    q = V[rng.choice(n, 200, replace=False)].copy()
    nlist = max(1, n // 39)
    del_ids = np.array(sorted(rng.choice(n, n_del, replace=False)), dtype=np.int64)
    live_mask = np.ones(n, dtype=bool)
    live_mask[del_ids] = False
    live_ids = np.nonzero(live_mask)[0]

    exact_live = faiss.IndexIDMap(faiss.IndexFlatIP(d))
    exact_live.add_with_ids(np.ascontiguousarray(V[live_ids]), live_ids.astype(np.int64))

    def jaccard(index, exclude=None):
        D, I = index.search(q, k * 4)
        Dq, Iq = exact_live.search(q, k)
        out = []
        for r in range(q.shape[0]):
            a = set()
            for x in I[r]:
                if x < 0:
                    continue
                xi = int(x)
                if exclude and xi in exclude:
                    continue
                a.add(xi)
                if len(a) >= k:
                    break
            b = set(int(x) for x in Iq[r] if x >= 0)
            out.append(len(a & b) / max(1, len(a | b)))
        return float(np.mean(out))

    def build_ivf():
        quant = faiss.IndexFlatIP(d)
        idx = faiss.IndexIVFFlat(quant, d, nlist, faiss.METRIC_INNER_PRODUCT)
        idx.train(V)
        idx.nprobe = max(1, min(16, nlist))
        return idx

    exact_all = faiss.IndexIDMap(faiss.IndexFlatIP(d))
    exact_all.add_with_ids(V, np.arange(n, dtype=np.int64))
    b0 = build_ivf()
    b0.add(V)
    D, I = b0.search(q, k)
    Dq, Iq = exact_all.search(q, k)
    j_before = float(np.mean([
        len(set(int(x) for x in I[r] if x >= 0) & set(int(x) for x in Iq[r] if x >= 0)) /
        max(1, len(set(int(x) for x in I[r] if x >= 0) | set(int(x) for x in Iq[r] if x >= 0)))
        for r in range(q.shape[0])
    ]))

    wrapped = faiss.IndexIDMap2(build_ivf())
    wrapped.add_with_ids(V, np.arange(n, dtype=np.int64))
    sel = faiss.IDSelectorBatch(del_ids.size, faiss.swig_ptr(del_ids))
    wrapped.remove_ids(sel)
    j_idmap2 = jaccard(wrapped)

    bare = build_ivf()
    bare.add(V)
    j_own = jaccard(bare, exclude=set(int(x) for x in del_ids))

    print(json.dumps({
        "jaccard_before_delete": round(j_before, 4),
        "jaccard_idmap2_after_remove": round(j_idmap2, 4),
        "jaccard_own_map_tombstones": round(j_own, 4),
        "idmap2_ntotal_after_remove": int(wrapped.ntotal),
        "requested_deletes": int(n_del),
    }, indent=1))


if __name__ == "__main__":
    main()
