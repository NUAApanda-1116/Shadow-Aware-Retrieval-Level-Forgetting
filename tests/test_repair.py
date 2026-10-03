"""tests/test_repair.py -- synthetic-data validation of AnchorRepair (no network, no external dependencies).

Controlled fixture (_RingIndex)
-------------------------
N=500 vectors evenly spaced on a circle (D=8, first two dims cos/sin); adjacency is the 2 nodes on
each side along the ring (M=4); retrieval is a "best-first traversal from a fixed entry point"
(visit budget = 2N, enough to walk an entire connected component).

* Intact index: the entry reaches the whole ring -> recall@10 == 1.0 for unrelated queries (gold = brute-force top-10 over alive vectors).
* Hard-delete 20% (delete 1 of every 5; deleted points and their edges are not traversable at search time):
  the ring is cut into small fragments of 4 alive nodes each, and the entry can reach only one of them
  -> recall collapses to near 0. This is the structural damage left by "naive deletion".
* AnchorRepair: for each deleted point, take its 8 nearest alive neighbors and connect them pairwise
  (a clique). Each deleted point has 4 alive neighbors on each side along the ring, so the clique
  necessarily contains a long edge that spans the deleted point (e.g. i-4 to i+4); the ring becomes
  connected again -> recall recovers to 1.0.

So the "recovery >= 90%" threshold in this file is not arbitrarily tuned: both the damage and the
recovery can be derived step by step.
"""

from __future__ import annotations

import heapq
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from fedrevoke.metrics import recall_at_k  # noqa: E402
from fedrevoke.repair import AnchorRepair, index_backend  # noqa: E402

K = 10
ROUND = 1e-9


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
@dataclass
class _Meta:
    doc_id: str
    text: str = ""


def _ring_vectors(n: int = 500, dim: int = 8) -> np.ndarray:
    """Points on a circle (D=8, first two dims cos/sin).

    A deterministic angular jitter of ±0.6 average spacing is superimposed: otherwise the cosines of
    the k-th neighbors to the left and right of a query point are exactly equal, the boundary ranks of
    the top-k are decided randomly by float32 noise, and recall cannot be reproduced stably. The jitter
    amplitude is far smaller than the gap between "4th nearest neighbor vs 5th nearest neighbor", so the
    geometric neighbor ordering still holds.
    """
    step = 2.0 * np.pi / float(n)
    jitter = np.random.default_rng(20260214).uniform(0.0, 0.6 * step, size=n)
    angles = 2.0 * np.pi * np.arange(n) / float(n) + jitter
    vectors = np.zeros((n, dim), dtype=np.float32)
    vectors[:, 0] = np.cos(angles)
    vectors[:, 1] = np.sin(angles)
    return vectors


def _bruteforce_topk(vectors: np.ndarray, queries: np.ndarray, k: int = K, allowed=None) -> list:
    pool_ids = list(range(len(vectors))) if allowed is None else sorted(int(i) for i in allowed)
    pool = np.asarray(vectors)[pool_ids]
    pool = pool / np.maximum(np.linalg.norm(pool, axis=1, keepdims=True), 1e-12)
    queries = np.asarray(queries, dtype=np.float32)
    queries = queries / np.maximum(np.linalg.norm(queries, axis=1, keepdims=True), 1e-12)
    sims = queries @ pool.T
    gold = []
    for row in sims:
        order = np.argsort(-row, kind="mergesort")[:k]
        gold.append([int(pool_ids[i]) for i in order.tolist()])
    return gold


def _mean_recall(retrieved_ids, gold, k: int = K) -> float:
    arr = np.asarray(retrieved_ids)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    return float(np.mean([recall_at_k(arr[i].tolist(), gold[i], k) for i in range(len(gold))]))


class _RingIndex:
    """Controlled worst-case graph index: a ring + hard deletion (deleted points are not traversable)."""

    backend = "hnswlib"

    def __init__(self, vectors: np.ndarray, links_half: int = 2, entry_count: int = 4):
        self._vecs = {i: np.asarray(v, dtype=np.float32) for i, v in enumerate(np.asarray(vectors))}
        self._alive = set(self._vecs)
        self._links = {i: set() for i in self._vecs}
        n = len(self._vecs)
        for i in range(n):
            for step in range(1, links_half + 1):
                self._links[i].add((i + step) % n)
                self._links[i].add((i - step) % n)
        self._entry_ids = list(range(min(entry_count, n)))
        self.visit_budget = 2 * n
        self.link_calls = 0

    # -- index primitives ---------------------------------------------------------- #
    def add_link(self, i, j):
        i, j = int(i), int(j)
        self._links[i].add(j)
        self._links[j].add(i)
        self.link_calls += 1

    def remove(self, ids):
        self._alive -= {int(i) for i in ids}

    def alive_ids(self):
        return np.asarray(sorted(self._alive), dtype=np.int64)

    def vector(self, internal_id):
        return self._vecs.get(int(internal_id))

    def meta(self, internal_id):
        return _Meta("d{0:05d}".format(int(internal_id)), "ring node {0}".format(int(internal_id)))

    def text_for_id(self, internal_id):
        return "ring node {0}".format(int(internal_id))

    def stats(self):
        return {
            "n_vectors": len(self._alive),
            "n_deleted": len(self._vecs) - len(self._alive),
            "backend": self.backend,
        }

    # -- retrieval -------------------------------------------------------------- #
    @staticmethod
    def _unit(vec):
        vec = np.asarray(vec, dtype=np.float32).reshape(-1)
        norm = float(np.linalg.norm(vec))
        return vec / norm if norm > 1e-12 else vec

    def _sim(self, unit_query, internal_id):
        return float(np.dot(unit_query, self._unit(self._vecs[internal_id])))

    def _traverse(self, unit_query) -> list:
        entries = [i for i in self._entry_ids if i in self._alive]
        if not entries:
            return []
        heap = [(-self._sim(unit_query, i), i) for i in entries]
        heapq.heapify(heap)
        seen = set(entries)
        visited: list = []
        while heap and len(visited) < self.visit_budget:
            _, node = heapq.heappop(heap)
            visited.append(node)
            for nxt in sorted(self._links.get(node, ())):
                if nxt in seen or nxt not in self._alive:
                    continue
                seen.add(nxt)
                heapq.heappush(heap, (-self._sim(unit_query, nxt), nxt))
        return visited

    def search(self, queries, k: int = K):
        arr = np.asarray(queries, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        k = int(k)
        scores = np.full((arr.shape[0], k), -1.0, dtype=np.float32)
        ids = np.full((arr.shape[0], k), -1, dtype=np.int64)
        for row, raw in enumerate(arr):
            unit = self._unit(raw)
            visited = self._traverse(unit)
            if not visited:
                continue
            ranked = sorted(visited, key=lambda i: (-self._sim(unit, i), i))[:k]
            for col, node in enumerate(ranked):
                scores[row, col] = self._sim(unit, node)
                ids[row, col] = node
        return scores, ids


class _SpyHnswIndex(_RingIndex):
    """Additionally provides a rebuild_local local-rebuild interface, verifying that the hnswlib branch really goes through the repair interface."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.repair_calls: list = []

    def rebuild_local(self, removed_ids, anchor_ids):
        self.repair_calls.append((list(removed_ids), list(anchor_ids)))


class _ListIndex:
    """faiss_ivf-style fake index: verifies the anchor replica + inverted-list recompute hooks."""

    backend = "faiss_ivf"

    def __init__(self, vectors, doc_ids):
        self._vecs = {i: np.asarray(v, dtype=np.float32) for i, v in enumerate(np.asarray(vectors))}
        self._metas = {i: _Meta(str(d)) for i, d in enumerate(doc_ids)}
        self._alive = set(self._vecs)
        self._next = len(self._vecs)
        self.added_metas: list = []
        self.rebuild_calls = 0

    def add(self, vectors, metas):
        new_ids = []
        for vec, meta in zip(np.asarray(vectors, dtype=np.float32), list(metas)):
            iid = self._next
            self._next += 1
            self._vecs[iid] = np.asarray(vec, dtype=np.float32)
            self._metas[iid] = meta
            self._alive.add(iid)
            self.added_metas.append(meta)
            new_ids.append(iid)
        return new_ids

    def remove(self, ids):
        self._alive -= {int(i) for i in ids}

    def alive_ids(self):
        return np.asarray(sorted(self._alive), dtype=np.int64)

    def vector(self, internal_id):
        return self._vecs.get(int(internal_id))

    def meta(self, internal_id):
        return self._metas.get(int(internal_id))

    def search(self, queries, k: int = K):
        arr = np.asarray(queries, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        alive = sorted(self._alive)
        pool = np.asarray([self._vecs[i] for i in alive], dtype=np.float32)
        pool = pool / np.maximum(np.linalg.norm(pool, axis=1, keepdims=True), 1e-12)
        q = arr / np.maximum(np.linalg.norm(arr, axis=1, keepdims=True), 1e-12)
        sims = q @ pool.T
        k = int(k)
        scores = np.full((arr.shape[0], k), -1.0, dtype=np.float32)
        ids = np.full((arr.shape[0], k), -1, dtype=np.int64)
        for row in range(sims.shape[0]):
            order = np.argsort(-sims[row], kind="mergesort")[:k]
            for col, pos in enumerate(order.tolist()):
                scores[row, col] = sims[row, pos]
                ids[row, col] = alive[pos]
        return scores, ids

    def rebuild_lists(self):
        self.rebuild_calls += 1

    def stats(self):
        return {"n_vectors": len(self._alive), "n_deleted": len(self._vecs) - len(self._alive), "backend": self.backend}


class _NoSearchIndex:
    """Has vector access but no search(): recall cannot be measured."""

    backend = "hnswlib"

    def __init__(self, vectors):
        self._vecs = {i: np.asarray(v, dtype=np.float32) for i, v in enumerate(np.asarray(vectors))}
        self._alive = set(self._vecs)

    def remove(self, ids):
        self._alive -= {int(i) for i in ids}

    def alive_ids(self):
        return np.asarray(sorted(self._alive), dtype=np.int64)

    def vector(self, internal_id):
        return self._vecs.get(int(internal_id))

    def meta(self, internal_id):
        return _Meta("d{0:05d}".format(int(internal_id)))


class _NoVectorIndex:
    """Has search() but no access to vectors: anchors degrade to random sampling and the recall reference is unavailable."""

    backend = "faiss_ivf"

    def __init__(self, vectors):
        self._matrix = np.asarray(vectors, dtype=np.float32)
        self._alive = set(range(self._matrix.shape[0]))

    def remove(self, ids):
        self._alive -= {int(i) for i in ids}

    def alive_ids(self):
        return np.asarray(sorted(self._alive), dtype=np.int64)

    def search(self, queries, k: int = K):
        arr = np.asarray(queries, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        alive = sorted(self._alive)
        pool = self._matrix[alive]
        pool = pool / np.maximum(np.linalg.norm(pool, axis=1, keepdims=True), 1e-12)
        q = arr / np.maximum(np.linalg.norm(arr, axis=1, keepdims=True), 1e-12)
        sims = q @ pool.T
        k = int(k)
        scores = np.full((arr.shape[0], k), -1.0, dtype=np.float32)
        ids = np.full((arr.shape[0], k), -1, dtype=np.int64)
        for row in range(sims.shape[0]):
            order = np.argsort(-sims[row], kind="mergesort")[:k]
            for col, pos in enumerate(order.tolist()):
                scores[row, col] = sims[row, pos]
                ids[row, col] = alive[pos]
        return scores, ids


def _query_vectors(vectors: np.ndarray, stride: int = 5, offset: int = 2) -> tuple:
    ids = [i for i in range(len(vectors)) if i % stride == offset]
    return np.asarray(vectors)[ids].copy(), ids


def _spaced_deletion(n: int, block: int = 2, period: int = 10) -> list:
    """Delete block consecutive points out of every period points (default 2/10 = exactly 20%).

    Note: this fixture's graph has ±1/±2 edges, so deleting a single isolated point is bypassed by the
    ±2 edge i-1 -> i+1; a run of >= 2 consecutive deleted points is required to cut the ring. This is
    also a controlled model of "revocations tend to happen in clusters".
    """
    return [i for i in range(n) if i % period < block]


def _jittered_deletion(n: int, target: int) -> list:
    """Jittered deletion pattern: runs of {2,3} consecutive deletions, gaps of {6..9} alive points, overall density about 20%."""
    picked: list = []
    pos = 0
    cycle = 0
    gaps = (8, 7, 9, 8, 6)
    while pos < n and len(picked) < target:
        run = 3 if cycle % 3 == 0 else 2
        for offset in range(run):
            if pos + offset < n and len(picked) < target:
                picked.append(pos + offset)
        pos += run + gaps[cycle % len(gaps)]
        cycle += 1
    return sorted(set(picked))


# --------------------------------------------------------------------------- #
# 1) Baseline: confirm the fixture is not a no-op
# --------------------------------------------------------------------------- #
def test_ring_fixture_intact_and_damaged_baseline():
    vectors = _ring_vectors()
    queries, _ = _query_vectors(vectors)
    k = K

    intact = _RingIndex(vectors)
    gold_all = _bruteforce_topk(vectors, queries, k)
    intact_recall = _mean_recall(intact.search(queries, k)[1], gold_all, k)
    assert intact_recall >= 0.99  # the entry reaches the whole ring -> exact top-k

    damaged = _RingIndex(vectors)
    removed = _spaced_deletion(len(vectors))
    assert len(removed) == len(vectors) // 5  # exactly 20%
    damaged.remove(removed)
    gold_alive = _bruteforce_topk(vectors, queries, k, allowed=damaged.alive_ids())
    damaged_recall = _mean_recall(damaged.search(queries, k)[1], gold_alive, k)
    assert damaged_recall < 0.5 * intact_recall  # naive deletion really does cause structural damage


# --------------------------------------------------------------------------- #
# 2) Main result: after deleting 20%, unrelated-query recall recovers to >= 90%
# --------------------------------------------------------------------------- #
def test_repair_restores_unrelated_recall_after_20pct_deletion():
    vectors = _ring_vectors()
    queries, _ = _query_vectors(vectors)
    removed = _spaced_deletion(len(vectors))
    assert len(removed) == len(vectors) // 5

    intact = _RingIndex(vectors)
    gold_all = _bruteforce_topk(vectors, queries, K)
    intact_recall = _mean_recall(intact.search(queries, K)[1], gold_all, K)

    index = _RingIndex(vectors)
    index.remove(removed)
    repair = AnchorRepair(n_anchors=512, recalibrate=True, seed=20260214, n_neighbors=8, k=K)
    result = repair.repair(index, removed, queries)

    # The four fields required by INTERFACES.md section 4 must be present
    for key in ("n_reconnected", "score_shift", "before_recall", "after_recall"):
        assert key in result

    assert result["recall_mode"] == "exact_alive"
    assert index.link_calls > 0 and result["n_reconnected"] > 0
    assert result["before_recall"] < 0.5 * intact_recall       # there really is damage before repair
    assert result["after_recall"] >= 0.90                      # absolute level >= 90%
    assert result["after_recall"] >= 0.90 * intact_recall      # relative to the intact index >= 90%
    gap = intact_recall - result["before_recall"]
    recovery = (result["after_recall"] - result["before_recall"]) / max(gap, ROUND)
    assert recovery >= 0.90                                    # recovers >= 90% of the lost points


def test_repair_recovers_jittered_deletion_pattern():
    vectors = _ring_vectors()
    queries, _ = _query_vectors(vectors)
    removed = _jittered_deletion(len(vectors), target=len(vectors) // 5)
    assert 0.15 * len(vectors) <= len(removed) <= 0.25 * len(vectors)

    intact = _RingIndex(vectors)
    gold_all = _bruteforce_topk(vectors, queries, K)
    intact_recall = _mean_recall(intact.search(queries, K)[1], gold_all, K)

    index = _RingIndex(vectors)
    index.remove(removed)
    repair = AnchorRepair(n_anchors=512, recalibrate=False, seed=20260214, n_neighbors=8, k=K)
    result = repair.repair(index, removed, queries)

    assert result["after_recall"] >= 0.90
    assert result["after_recall"] >= 0.90 * intact_recall
    assert result["after_recall"] >= result["before_recall"] - ROUND


# --------------------------------------------------------------------------- #
# 3) Backend branch contracts
# --------------------------------------------------------------------------- #
def test_faiss_backend_adds_only_alive_anchor_replicas_and_rebuilds_lists():
    vectors = _ring_vectors(n=200)
    doc_ids = ["d{0:04d}".format(i) for i in range(len(vectors))]
    index = _ListIndex(vectors, doc_ids)
    queries, _ = _query_vectors(vectors)
    removed = _spaced_deletion(len(vectors))
    index.remove(removed)
    alive_docs = {index.meta(int(i)).doc_id for i in index.alive_ids()}
    removed_docs = {index.meta(int(i)).doc_id for i in removed}

    repair = AnchorRepair(n_anchors=64, recalibrate=False, seed=20260214, n_neighbors=6, k=K)
    result = repair.repair(index, removed, queries)

    assert index_backend(index) == "faiss_ivf"
    assert result["repair_mode"].startswith("anchor_replica")
    assert result["n_anchor_replicas"] > 0
    assert "rebuild_lists" in result["rebuild_hooks"]
    assert index.rebuild_calls == 1
    # Scientific correctness: anchor replicas may only copy "alive" vectors and must never resurrect deleted documents
    assert len(index.added_metas) == result["n_anchor_replicas"]
    added_docs = {m.doc_id for m in index.added_metas}
    assert added_docs <= alive_docs
    assert not (added_docs & removed_docs)


def test_hnsw_backend_uses_local_repair_hook():
    vectors = _ring_vectors(n=300)
    queries, _ = _query_vectors(vectors)
    removed = _spaced_deletion(len(vectors))
    index = _SpyHnswIndex(vectors)
    index.remove(removed)

    repair = AnchorRepair(n_anchors=128, recalibrate=False, seed=20260214, n_neighbors=8, k=K)
    result = repair.repair(index, removed, queries)

    assert result["repair_mode"].endswith("hnsw_local_repair")
    assert "rebuild_local" in result["repair_hooks"]
    assert index.repair_calls, "rebuild_local must be called"
    called_removed, called_anchors = index.repair_calls[-1]
    assert set(called_removed) == set(removed)
    alive = set(int(i) for i in index.alive_ids())
    assert set(called_anchors) <= alive
    assert result["after_recall"] >= 0.90


# --------------------------------------------------------------------------- #
# 4) Score-quantile recalibration
# --------------------------------------------------------------------------- #
def test_recalibration_aligns_score_distributions_and_installs_map():
    vectors = _ring_vectors()
    queries, _ = _query_vectors(vectors)
    removed = _spaced_deletion(len(vectors))
    index = _RingIndex(vectors)
    index.remove(removed)

    repair = AnchorRepair(n_anchors=512, recalibrate=True, seed=20260214, n_neighbors=8, k=K)
    result = repair.repair(index, removed, queries)

    assert result["recalibrated"] is True
    assert result["calibration"] is not None and result["calibration"]["n_knots"] > 1
    assert result["score_shift"] > 0.0            # repair changed the score distribution -> a correction was applied
    assert result["score_residual"] <= result["score_shift"] + 1e-6
    assert getattr(index, "score_calibrator", None) is repair.calibration_
    scores, ids = repair.calibrated_search(index, queries, K)
    assert scores.shape == ids.shape == (len(queries), K)


def test_recalibrate_false_keeps_raw_scores():
    vectors = _ring_vectors(n=300)
    queries, _ = _query_vectors(vectors)
    removed = _spaced_deletion(len(vectors))
    index = _RingIndex(vectors)
    index.remove(removed)

    repair = AnchorRepair(n_anchors=128, recalibrate=False, seed=20260214, n_neighbors=8, k=K)
    result = repair.repair(index, removed, queries)

    assert result["recalibrated"] is False
    assert result["score_shift"] == 0.0
    assert result["calibration"] is None
    assert repair.calibration_ is None
    assert not hasattr(index, "score_calibrator")


# --------------------------------------------------------------------------- #
# 5) Edge cases and degradation
# --------------------------------------------------------------------------- #
def test_no_removed_ids_is_noop():
    vectors = _ring_vectors(n=200)
    queries, _ = _query_vectors(vectors)
    index = _RingIndex(vectors)
    repair = AnchorRepair(n_anchors=64, recalibrate=True, seed=20260214, k=K)
    result = repair.repair(index, [], queries)

    assert result["n_removed"] == 0
    assert result["n_reconnected"] == 0
    assert result["repair_mode"] == "noop"
    assert index.link_calls == 0
    assert result["before_recall"] == result["after_recall"]
    assert result["before_recall"] >= 0.99


def test_empty_query_sample_is_safe():
    vectors = _ring_vectors(n=200)
    index = _RingIndex(vectors)
    removed = _spaced_deletion(len(vectors))
    index.remove(removed)
    repair = AnchorRepair(n_anchors=64, recalibrate=True, seed=20260214, k=K)
    result = repair.repair(index, removed, np.zeros((0, 0), dtype=np.float32))
    assert result["before_recall"] == 0.0 and result["after_recall"] == 0.0
    assert result["score_shift"] == 0.0
    assert result["recall_mode"] == "unavailable"


def test_missing_search_reports_unavailable_and_strict_raises():
    vectors = _ring_vectors(n=200)
    queries, _ = _query_vectors(vectors)
    index = _NoSearchIndex(vectors)
    removed = _spaced_deletion(len(vectors))
    index.remove(removed)

    lenient = AnchorRepair(n_anchors=32, recalibrate=True, seed=20260214, k=K)
    result = lenient.repair(index, removed, queries)
    assert result["warnings"] and any("search" in w for w in result["warnings"])
    assert result["after_recall"] == 0.0

    strict = AnchorRepair(n_anchors=32, recalibrate=True, seed=20260214, k=K, strict=True)
    with pytest.raises(RuntimeError):
        strict.repair(index, removed, queries)


def test_missing_vectors_degrades_to_random_anchors():
    vectors = _ring_vectors(n=200)
    queries, _ = _query_vectors(vectors)
    index = _NoVectorIndex(vectors)
    removed = _spaced_deletion(len(vectors))
    index.remove(removed)

    repair = AnchorRepair(n_anchors=32, recalibrate=False, seed=20260214, k=K)
    result = repair.repair(index, removed, queries)
    assert result["recall_mode"] == "unavailable"
    assert result["anchor_mode"] in ("random", "none")
    assert result["repair_mode"] == "logical"
    assert result["before_recall"] == 0.0 and result["after_recall"] == 0.0


def test_gold_pids_override_is_used():
    vectors = _ring_vectors(n=300)
    queries, _ = _query_vectors(vectors)
    removed = _spaced_deletion(len(vectors))
    index = _RingIndex(vectors)
    index.remove(removed)
    gold = _bruteforce_topk(vectors, queries, K, allowed=index.alive_ids())

    repair = AnchorRepair(n_anchors=128, recalibrate=False, seed=20260214, n_neighbors=8, k=K)
    result = repair.repair(index, removed, queries, gold_pids=gold)
    assert result["recall_mode"] == "provided"
    assert result["after_recall"] >= 0.90
    assert result["before_recall"] >= 0.0


# --------------------------------------------------------------------------- #
# 6) Integration with a real ProvenanceIndex (auto-enabled when available; skipped otherwise)
# --------------------------------------------------------------------------- #
def test_repair_with_real_provenance_index_if_available():
    index_core = pytest.importorskip("fedrevoke.index_core")
    provenance_index = getattr(index_core, "ProvenanceIndex", None)
    vec_meta = getattr(index_core, "VecMeta", None)
    if provenance_index is None or vec_meta is None:
        pytest.skip("index_core has not yet implemented ProvenanceIndex/VecMeta")
    vectors = _ring_vectors(n=200)
    try:
        index = provenance_index(dim=vectors.shape[1], backend="faiss_ivf")
        metas = [vec_meta(pid=i, doc_id="d{0:04d}".format(i), client_id="c0", topic="t", fingerprint="f") for i in range(len(vectors))]
        index.add(vectors, metas)
    except Exception as exc:  # environment lacks faiss / backend unavailable
        pytest.skip("ProvenanceIndex cannot be constructed in the current environment: {0}".format(exc))
    removed = _spaced_deletion(len(vectors))
    index.remove(removed)
    queries, _ = _query_vectors(vectors)
    repair = AnchorRepair(n_anchors=64, recalibrate=True, seed=20260214, n_neighbors=6, k=K)
    result = repair.repair(index, removed, queries)
    assert set(["n_reconnected", "score_shift", "before_recall", "after_recall"]) <= set(result)


# --------------------------------------------------------------------------- #
# 7) ANN contract and performance regression for neighborhood search
# --------------------------------------------------------------------------- #
def _clustered_vectors(n: int, dim: int = 32, seed: int = 20260214, spread: float = 0.05) -> np.ndarray:
    """Deterministic clustered vectors: neighbors concentrate within the same cluster, so faiss IVF can produce high-recall ANN neighborhoods."""
    rng = np.random.default_rng(seed)
    n_clusters = max(8, n // 50)
    centers = rng.normal(size=(n_clusters, dim)).astype(np.float32)
    centers /= np.maximum(np.linalg.norm(centers, axis=1, keepdims=True), 1e-12)
    labels = rng.integers(0, n_clusters, size=n)
    return (centers[labels] + spread * rng.normal(size=(n, dim)).astype(np.float32)).astype(np.float32)


def _real_faiss_index(n: int, dim: int = 32, seed: int = 20260214):
    """Construct a real ProvenanceIndex(faiss_ivf); skip when faiss is missing or the backend degrades."""
    index_core = pytest.importorskip("fedrevoke.index_core")
    pytest.importorskip("faiss")
    index = index_core.ProvenanceIndex(dim=dim, backend="faiss_ivf")
    vectors = _clustered_vectors(n, dim=dim, seed=seed)
    metas = [
        index_core.VecMeta(pid=i, doc_id="d{0:06d}".format(i), client_id="c{0}".format(i % 5),
                           topic="t", fingerprint="f")
        for i in range(n)
    ]
    index.add(vectors, metas)
    if index_backend(index) != "faiss_ivf":  # cannot verify sublinearity when degraded to the numpy backend
        pytest.skip("faiss_ivf backend unavailable: {0}".format(getattr(index, "fallback_reason", "")))
    return index, vectors


def _timed_repair(n_total: int, n_removed: int = 100, n_queries: int = 20, repeats: int = 3):
    """Hold |removed| fixed, repeat timing on a given n_alive, and return (shortest duration, last result)."""
    seed = 20260214
    times = []
    result = None
    for _ in range(repeats):
        index, vectors = _real_faiss_index(n_total)
        rng = np.random.default_rng(seed + 1)
        removed = np.sort(rng.choice(n_total, size=n_removed, replace=False))
        index.remove([int(i) for i in removed.tolist()])
        queries = vectors[np.sort(rng.choice(n_total, size=n_queries, replace=False))].copy()
        repair = AnchorRepair(n_anchors=16, recalibrate=True, seed=seed, n_neighbors=8, k=K)
        t0 = time.perf_counter()
        result = repair.repair(index, removed, queries)
        times.append(time.perf_counter() - t0)
    return min(times), result


def test_repair_neighbor_search_uses_ann_not_linear_scan():
    """Real faiss IVF index: neighborhood search must go through the index's own ANN (validated by a small-sample exact audit)."""
    n_total = 2500
    index, vectors = _real_faiss_index(n_total)
    rng = np.random.default_rng(20260215)
    removed = np.sort(rng.choice(n_total, size=100, replace=False))
    index.remove([int(i) for i in removed.tolist()])
    queries = vectors[np.sort(rng.choice(n_total, size=20, replace=False))].copy()

    repair = AnchorRepair(n_anchors=16, recalibrate=True, seed=20260214, n_neighbors=8, k=K)
    result = repair.repair(index, removed, queries)

    assert result["neighbor_oracle"] == "ann_index_search"
    assert result["ann_audit_probes"] > 0
    assert result["ann_audit_overlap"] is not None and result["ann_audit_overlap"] >= 0.5
    assert result["anchor_mode"] == "removed_neighborhood"
    assert result["after_recall"] >= result["before_recall"] - ROUND
    # Index-size cost is observable: n_vectors_delta = number of anchor replicas (<= n_anchors)
    assert result["n_vectors_delta"] == result["n_anchor_replicas"] <= 16


def test_neighbor_oracle_audit_rejects_damaged_graph_search():
    """A damaged graph index can only return its own connected components -> the audit vetoes it and falls back to chunked exact kNN (quality first)."""
    vectors = _ring_vectors()
    queries, _ = _query_vectors(vectors)
    removed = _spaced_deletion(len(vectors))

    intact = _RingIndex(vectors)
    gold_all = _bruteforce_topk(vectors, queries, K)
    intact_recall = _mean_recall(intact.search(queries, K)[1], gold_all, K)

    index = _RingIndex(vectors)
    index.remove(removed)
    repair = AnchorRepair(n_anchors=32, recalibrate=False, seed=20260214, n_neighbors=8, k=K)
    result = repair.repair(index, removed, queries)

    # Ring-graph retrieval is cut into multiple connected components by the deleted points -> overlap with exact neighbors is extremely low -> the audit vetoes the ANN path
    assert result["neighbor_oracle"] == "exact_chunked"
    assert result["ann_audit_overlap"] is not None and result["ann_audit_overlap"] < 0.5
    # After the veto, the original repair strength must still be maintained
    assert result["after_recall"] >= 0.90
    assert result["after_recall"] >= 0.90 * intact_recall
    recovery = (result["after_recall"] - result["before_recall"]) / max(intact_recall - result["before_recall"], ROUND)
    assert recovery >= 0.90


def test_repair_time_does_not_scale_linearly_with_n_alive():
    """Performance regression: |removed| fixed at 100, n_alive scaled 4x (2500 -> 10000); the time increase must be < 2x.

    Reference magnitude (this machine): n_alive=2500 about 0.010s, n_alive=10000 about 0.014s (ratio about 1.4).
    After switching to an O(|removed| x n_alive x d) brute-force implementation, the ratio approaches 4 and the test fails immediately.
    """
    n_small, n_large = 2500, 10000
    t_small, r_small = _timed_repair(n_small)
    t_large, r_large = _timed_repair(n_large)

    for res in (r_small, r_large):
        assert res["n_removed"] == 100
        assert str(res["neighbor_oracle"]).startswith("ann_index_search"), res["neighbor_oracle"]

    ratio = t_large / max(t_small, 1e-9)
    assert ratio < 2.0, (
        "repair wall time grew by {0:.2f}x when n_alive was scaled 4x (t_small={1:.4f}s, t_large={2:.4f}s)".format(
            ratio, t_small, t_large
        )
    )
