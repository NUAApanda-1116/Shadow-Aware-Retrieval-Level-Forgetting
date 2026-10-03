"""Compare FedRevoke delete set vs oracle and diagnose rho_hat_ret."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fedrevoke.baselines import _exact_topk_pids, rho_hat_tv  # noqa: E402
from fedrevoke.run_experiments import (  # noqa: E402
    build_detector,
    build_index,
    build_oracle_index,
    build_repair,
    controlled_forget_set,
    load_real_dataset,
)
from fedrevoke.revoke import RevocationPipeline  # noqa: E402


def main() -> int:
    ds, status = load_real_dataset("ds1", limit_queries=500)
    print("qvec source:", status.get("query_vector_source"))
    doc_ids, cov = controlled_forget_set(ds, 0.05, 1.0, 20260214, "r5")
    print("forget docs", len(doc_ids), cov)

    shadow_map = ds.get("shadow_map") or {}
    oracle_docs = set(map(str, doc_ids))
    for d in doc_ids:
        oracle_docs.update(str(s) for s in (shadow_map.get(d) or []))
    print("oracle_docs", len(oracle_docs), "shadow extras", len(oracle_docs) - len(doc_ids))

    cfg = {
        "shadow": {"sim_threshold": 0.92, "lsh_threshold": 0.80, "knn_k": 50, "cross_client_only": True},
        "repair": {"n_anchors": 64, "recalibrate": True},
    }
    index = build_index(ds)
    oracle = build_oracle_index(ds, doc_ids)
    print("before alive", index.n_alive, "oracle alive", oracle.n_alive)

    detector = build_detector(cfg, ds.get("signatures"), knn_k=50, sim_threshold=0.92, vector_channel=True)
    repair = build_repair(cfg, enabled=True, recalibrate=None)
    pipe = RevocationPipeline(
        index=index, detector=detector, repair=repair, verifier=None,
        cert_dir=ROOT / "artifacts" / "certificates" / "diag",
        encoder=None, k=10, generator=None,
        forgotten_qa=[], seed=20260214, shadow_ratio=1.0, smoke=False, verbose=False,
        closed_book=False, client_id="c1",
    )
    res = pipe.revoke(doc_ids, query_sample=ds.get("qvecs")[:50] if ds.get("qvecs") is not None else None)
    print("n_deleted", len(res.deleted_ids))
    deleted_docs = set(res.certificate.get("deleted_doc_ids") or [])
    print("deleted docs", len(deleted_docs))
    extra = deleted_docs - oracle_docs
    missing = oracle_docs - deleted_docs
    print("extra vs oracle", len(extra), "missing vs oracle", len(missing))
    print("sample extra", list(extra)[:10])
    print("sample missing", list(missing)[:10])

    print("after alive", index.n_alive, "oracle alive", oracle.n_alive)

    q = np.asarray(ds["qvecs"][:20], dtype=np.float32)
    a = _exact_topk_pids(index, q, 10)
    o = _exact_topk_pids(oracle, q, 10)
    print("topk sizes after", [len(x) for x in a[:5]], "oracle", [len(x) for x in o[:5]])
    inter = [len(x & y) for x, y in zip(a, o)]
    print("intersections", inter)
    tvs = [1.0 - (len(x & y) / 10.0) for x, y in zip(a, o)]
    print("per-q TV", tvs)
    print("rho_hat_tv", rho_hat_tv(oracle, index, type("C", (), {"query_sample": q, "k": 10})()))

    # also compare live pid sets
    def live_pids(idx):
        alive = idx.alive_ids() if hasattr(idx, "alive_ids") else None
        if alive is None:
            alive = [i for i in range(idx._n_total) if i not in set(idx.deleted_ids())]
        return {int(idx.meta(int(i)).pid) for i in alive}

    pa, po = live_pids(index), live_pids(oracle)
    print("live pids after", len(pa), "oracle", len(po), "symdiff", len(pa ^ po))
    print("only after", len(pa - po), "only oracle", len(po - pa))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
