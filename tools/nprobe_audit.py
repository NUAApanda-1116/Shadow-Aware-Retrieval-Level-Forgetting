"""Closure-consistency audit for nprobe=8 versus 16 (paper Sec. 6.3).

Usage (from the repository root):

    python tools/nprobe_audit.py

Computes the shadow closure for the DS1 forget sets (r1/r5/r20) under every
detector-threshold cell with the deployed search budget (nprobe=16) and with
the cheaper budget (nprobe=8), and reports which closures differ and whether
each difference is a miss or an over-deletion.
"""

from __future__ import annotations

import json
import sys

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from fedrevoke.run_experiments import load_real_dataset, build_detector, forgot_doc_ids
from fedrevoke.index_core import ProvenanceIndex, VecMeta


def main() -> None:
    ds, _status = load_real_dataset("multihoprag")
    V = np.asarray(ds["vectors"], dtype=np.float32)
    corpus = list(ds["corpus"])
    metas = [
        VecMeta(
            pid=int(r.get("pid", i)),
            doc_id=str(r.get("doc_id", "d%06d" % i)),
            client_id=str(r.get("client_id", "c0")),
            topic=str(r.get("topic", "")),
            fingerprint=str(r.get("fingerprint", "")),
        )
        for i, r in enumerate(corpus)
    ]
    sigs = ds.get("signatures")

    def make_index(nprobe):
        idx = ProvenanceIndex(int(V.shape[1]), backend="faiss_ivf", nprobe=nprobe)
        idx.add(V, metas)
        return idx

    idx16 = make_index(16)
    idx8 = make_index(8)

    grid = [(tau, jac) for tau in (0.85, 0.90, 0.92, 0.95) for jac in (0.50, 0.70, 0.80, 0.90)]
    total = diff = miss_only = over = 0
    details = []
    for key, ratio in (("r1", 0.01), ("r5", 0.05), ("r20", 0.20)):
        docs = forgot_doc_ids(ds, key, ratio)
        seeds = set()
        for d in docs:
            seeds.update(int(i) for i in idx16.ids_for_doc(d))
        for tau, jac in grid:
            cfg = {"seed": 20260214, "shadow": {"sim_threshold": tau, "lsh_threshold": jac,
                                                "knn_k": 50, "cross_client_only": True}}
            c16 = set(int(x) for x in build_detector(cfg, sigs, sim_threshold=tau).closure(idx16, sorted(seeds)))
            c8 = set(int(x) for x in build_detector(cfg, sigs, sim_threshold=tau).closure(idx8, sorted(seeds)))
            total += 1
            if c8 != c16:
                diff += 1
                missing, extra = c16 - c8, c8 - c16
                if extra and not missing:
                    over += 1
                else:
                    miss_only += 1
                details.append({"key": key, "tau": tau, "jaccard": jac, "n16": len(c16),
                                "n8": len(c8), "missing": len(missing), "extra": len(extra)})
    print(json.dumps({
        "closures_compared": total,
        "identical": total - diff,
        "differing": diff,
        "differing_miss_only": miss_only,
        "differing_with_overdelete": over,
        "details": details,
    }, indent=1))


if __name__ == "__main__":
    main()
