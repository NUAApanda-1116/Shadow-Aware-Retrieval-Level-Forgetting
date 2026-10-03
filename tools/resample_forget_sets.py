"""Regenerate forget_sets.json with single-silo sampling (A2 compliance).

Reads the existing corpus.jsonl (doc -> client_id) and rewrites forget_sets.json
so that every named document belongs to the requesting silo (largest silo by
document count). Nested r1 ⊂ r5 ⊂ r20, percentages still of the full corpus.

Usage:
    python tools/resample_forget_sets.py
    python tools/resample_forget_sets.py --dataset multihoprag
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fedrevoke import config as C  # noqa: E402

PROCESSED = ROOT / "data" / "processed"
DATASETS = ("multihoprag", "nq", "trec-covid")


def resample(ds_dir: Path, seed: int = int(C.SEED)) -> dict:
    corpus_path = ds_dir / "corpus.jsonl"
    if not corpus_path.exists():
        raise FileNotFoundError(corpus_path)
    doc_client: dict[str, str] = {}
    doc_chunks: dict[str, int] = defaultdict(int)
    with corpus_path.open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            did = str(r["doc_id"])
            if did.startswith("d"):
                doc_client[did] = str(r.get("client_id", "c0"))
            doc_chunks[did] += 1
    by_client: dict[str, list[str]] = defaultdict(list)
    for did, cid in doc_client.items():
        by_client[cid].append(did)
    requester = max(sorted(by_client), key=lambda c: (len(by_client[c]), c))
    pool = sorted(by_client[requester])
    n_docs = len(doc_client)
    want_r20 = max(1, int(round(C.FORGET_RATIOS["r20"] * n_docs)))
    if len(pool) < want_r20:
        raise ValueError(
            "%s: requester %s has %d docs < r20=%d" % (ds_dir.name, requester, len(pool), want_r20)
        )
    rng = np.random.default_rng(int(seed))
    perm = rng.permutation(len(pool))
    forget: dict[str, list[str]] = {}
    prev: list[str] = []
    for tag in ("r1", "r5", "r20"):
        want = max(1, int(round(C.FORGET_RATIOS[tag] * n_docs)))
        want = max(want, len(prev))
        ids = sorted(pool[int(i)] for i in perm[:want])
        forget[tag] = ids
        prev = ids
    out = dict(forget)
    fs_path = ds_dir / "forget_sets.json"
    fs_path.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    # keep dataset_stats.json in sync when present
    st_path = ds_dir / "dataset_stats.json"
    if st_path.exists():
        st = json.loads(st_path.read_text(encoding="utf-8"))
        st["forget_sets"] = {
            k: {"n_docs": len(v), "n_chunks": int(sum(doc_chunks.get(d, 0) for d in v))}
            for k, v in forget.items()
        }
        st["forget_sets"]["requester_client"] = requester
        st["forget_sets"]["requester_docs"] = len(pool)
        st_path.write_text(json.dumps(st, ensure_ascii=False, indent=1), encoding="utf-8")
    info = {
        "dataset": ds_dir.name,
        "n_docs": n_docs,
        "requester_client": requester,
        "requester_docs": len(pool),
        "sizes": {k: len(v) for k, v in forget.items()},
        "clients": {c: len(v) for c, v in sorted(by_client.items())},
    }
    return info


def main() -> int:
    ap = argparse.ArgumentParser(description="Resample forget sets as single-silo (A2)")
    ap.add_argument("--dataset", default=None, help="only this dataset name")
    args = ap.parse_args()
    names = (args.dataset,) if args.dataset else DATASETS
    for name in names:
        info = resample(PROCESSED / name)
        print(json.dumps(info, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
