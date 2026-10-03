"""Synthetic-data unit tests for ProvenanceIndex (INTERFACES.md §2).

Constraints: synthetic data only; no network, no reading data/, no dependency on the not-yet-landed config.py.
Run: cd fedrevoke; .venv/Scripts/python.exe -m pytest tests -q
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fedrevoke.index_core import (  # noqa: E402
    PAD_ID,
    ProvenanceIndex,
    VecMeta,
    available_backends,
    strip_padding,
)

SEED = 20260214
DIM = 32
N = 240
N_CLIENTS = 4
BACKENDS = available_backends()


# --------------------------------------------------------------------------------------
# Synthetic corpus
# --------------------------------------------------------------------------------------
def _corpus(n=N, dim=DIM, n_clients=N_CLIENTS, seed=SEED):
    """One Gaussian cluster per silo; guarantees different document vectors are all distinct."""
    rng = np.random.default_rng(seed)
    centers = rng.normal(size=(n_clients, dim)).astype(np.float32)
    vecs, metas = [], []
    for i in range(n):
        c = i % n_clients
        v = (0.8 * centers[c] + 0.6 * rng.normal(size=dim)).astype(np.float32)
        vecs.append(v)
        metas.append(
            VecMeta(
                pid=i,
                doc_id="d%06d" % i,
                client_id="c%d" % c,
                topic="t%d" % c,
                fingerprint="%064x" % i,
            )
        )
    return np.stack(vecs), metas, centers


def _make_index(backend, n=N, **kw):
    vecs, metas, centers = _corpus(n=n)
    index = ProvenanceIndex(DIM, backend=backend, **kw)
    ids = index.add(vecs, metas)
    return index, vecs, metas, ids, centers


def _assert_search_matches(idx_a, idx_b, queries, k=10, min_jaccard=0.9):
    """save/load round-trip: exact backends require element-wise identity; ANN backends require Jaccard overlap above threshold."""
    sa, ia = idx_a.search(queries, k=k)
    sb, ib = idx_b.search(queries, k=k)
    assert ia.shape == ib.shape == (len(queries), k), (ia.shape, ib.shape)
    if idx_a.backend == "numpy":
        assert np.array_equal(ia, ib)
        assert np.allclose(sa, sb, atol=1e-5)
        return
    overlaps = []
    for r in range(ia.shape[0]):
        a = set(int(x) for x in ia[r] if int(x) != PAD_ID)
        b = set(int(x) for x in ib[r] if int(x) != PAD_ID)
        if a or b:
            overlaps.append(len(a & b) / float(len(a | b)))
    assert overlaps, "no non-empty rows to compare"
    assert float(np.mean(overlaps)) >= min_jaccard, float(np.mean(overlaps))


# --------------------------------------------------------------------------------------
# 1. Add: stable ids, correct stats, traceable metadata
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize("backend", BACKENDS)
def test_add_returns_stable_ids(backend):
    index, vecs, metas, ids, _ = _make_index(backend)
    assert ids == list(range(N))
    assert index.n_vectors == N and index.n_alive == N and len(index) == N
    stats = index.stats()
    assert set(["n_vectors", "n_deleted", "backend", "memory_bytes"]).issubset(stats)
    assert stats["n_vectors"] == N and stats["n_deleted"] == 0
    assert stats["backend"] in BACKENDS and stats["memory_bytes"] > 0
    assert index.backend in available_backends()

    # Append a second batch: internal ids keep auto-incrementing
    more = index.add(vecs[:10], metas[:10])
    assert more == list(range(N, N + 10))
    assert index.n_vectors == N + 10

    # meta round-trip (including dict-form input)
    assert index.meta(0) == metas[0]
    assert index.meta(N) == VecMeta.from_dict({"pid": 0, "doc_id": "d000000", "client_id": "c0",
                                               "topic": "t0", "fingerprint": "%064x" % 0})
    with pytest.raises(KeyError):
        index.meta(-1)
    with pytest.raises(KeyError):
        index.meta(index.n_vectors)


@pytest.mark.parametrize("backend", BACKENDS)
def test_inverted_indices_and_boundaries(backend):
    index, vecs, metas, _, _ = _make_index(backend)
    assert index.ids_for_doc("d000004") == [4]
    assert index.ids_for_doc("no-such-doc") == []

    # When the same doc_id is split into multiple shards (cross-silo sharding), the inverted index must return all alive internal ids
    dup = index.add(
        np.stack([vecs[0], vecs[1]]),
        [
            VecMeta(pid=900000, doc_id="d000004", client_id="c7", topic="t0", fingerprint="ff" * 32),
            VecMeta(pid=900001, doc_id="d000004", client_id="c8", topic="t0", fingerprint="ee" * 32),
        ],
    )
    assert index.ids_for_doc("d000004") == [4, dup[0], dup[1]]
    assert index.ids_for_client("c7") == [dup[0]]
    assert dup == [N, N + 1]
    for c in range(N_CLIENTS):
        got = index.ids_for_client("c%d" % c)
        assert got == [i for i in range(N) if i % N_CLIENTS == c]
    assert index.ids_for_client("c99") == []
    assert sorted(index.clients()) == ["c%d" % c for c in range(N_CLIENTS)] + ["c7", "c8"]

    # Parameter validation
    with pytest.raises(ValueError):
        ProvenanceIndex(0)
    with pytest.raises(ValueError):
        ProvenanceIndex(DIM, backend="not-a-backend")
    with pytest.raises(ValueError):
        index.add(np.zeros((2, DIM + 1), dtype=np.float32), metas[:2])
    with pytest.raises(ValueError):
        index.add(np.zeros((2, DIM), dtype=np.float32), metas[:1])
    assert index.add(np.zeros((0, DIM), dtype=np.float32), []) == []


# --------------------------------------------------------------------------------------
# 2. Search: self-retrieval, shapes, cosine semantics
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize("backend", BACKENDS)
def test_search_self_retrieval(backend):
    index, vecs, _, _, _ = _make_index(backend)
    k = 10
    q = vecs[:20]
    scores, ids = index.search(q, k=k)
    assert scores.shape == (20, k) and ids.shape == (20, k)
    assert ids.dtype == np.int64 and scores.dtype == np.float32
    for row in range(20):
        assert int(ids[row, 0]) == row, (row, ids[row, :3])
        assert scores[row, 0] > 0.999
    # Scores are non-increasing, no -1 padding (enough alive entries)
    for row in range(20):
        assert np.all(np.diff(scores[row]) <= 1e-6)
        assert PAD_ID not in ids[row].tolist()
    # 1-D query
    s1, i1 = index.search(vecs[0], k=3)
    assert s1.shape == (1, 3) and int(i1[0, 0]) == 0
    # Dimension validation / k validation
    with pytest.raises(ValueError):
        index.search(np.zeros((1, DIM + 1), dtype=np.float32), k=3)
    with pytest.raises(ValueError):
        index.search(vecs[0], k=0)


# --------------------------------------------------------------------------------------
# 3. Remove: invisible, idempotent, stats stay in sync
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize("backend", BACKENDS)
def test_remove_hides_and_is_idempotent(backend):
    index, vecs, metas, _, _ = _make_index(backend)
    rng = np.random.default_rng(SEED)
    removed = sorted(rng.choice(N, size=72, replace=False).tolist())
    index.remove(removed)
    assert index.stats()["n_deleted"] == len(removed)
    assert index.n_alive == N - len(removed)
    assert len(index) == N - len(removed)
    alive = set(index.alive_ids().tolist())
    assert alive.isdisjoint(removed)
    assert sorted(alive) == [i for i in range(N) if i not in set(removed)]

    # Idempotent + out-of-range/unknown ids are safely ignored
    index.remove(removed)
    index.remove(removed[::-1])
    index.remove([-5, N + 100, 10 ** 9])
    index.remove([])
    assert index.stats()["n_deleted"] == len(removed)

    # Deleted documents are invisible in the inverted index, but metadata is retained (needed by M6 audit)
    for i in removed[:5]:
        assert index.ids_for_doc(metas[i].doc_id) == []
        assert index.meta(i) == metas[i]
        assert index.all_ids_for_doc(metas[i].doc_id) == [i]
        assert index.is_alive(i) is False
    for c in range(N_CLIENTS):
        assert set(index.ids_for_client("c%d" % c)).isdisjoint(removed)

    # search never returns deleted ids; querying with a deleted document's own vector does not hit it either
    queries = np.concatenate([vecs[removed[:40]], vecs[:40]], axis=0)
    scores, ids = index.search(queries, k=20)
    assert ids.shape == (80, 20)
    returned = set(int(x) for x in ids.ravel().tolist() if int(x) != PAD_ID)
    assert returned.isdisjoint(removed), sorted(returned & set(removed))[:5]
    assert returned <= alive
    for row in range(40):
        assert int(ids[row, 0]) != removed[row]


@pytest.mark.parametrize("backend", BACKENDS)
def test_remove_all_then_search(backend):
    index, vecs, _, _, _ = _make_index(backend, n=40)
    index.remove(list(range(40)))
    assert index.n_alive == 0 and index.stats()["n_deleted"] == 40
    assert index.alive_ids().size == 0
    assert index.deleted_ids().tolist() == list(range(40))
    scores, ids = index.search(vecs[:5], k=7)
    assert scores.shape == (5, 7) and ids.shape == (5, 7)
    assert (ids == PAD_ID).all()
    assert np.all(np.isneginf(scores))
    assert index.ids_for_client("c0") == []


# --------------------------------------------------------------------------------------
# 4. compact: preserve internal ids, reclaim ANN structures
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize("backend", BACKENDS)
def test_compact_preserves_ids(backend):
    index, vecs, _, _, _ = _make_index(backend)
    rng = np.random.default_rng(SEED + 1)
    removed = sorted(rng.choice(N, size=120, replace=False).tolist())
    index.remove(removed)
    before_alive = index.alive_ids().tolist()
    before_scores, before_ids = index.search(vecs[:30], k=15)
    n = index.compact()
    assert n == len(before_alive)
    assert index.alive_ids().tolist() == before_alive
    assert index.stats()["n_deleted"] == len(removed)
    assert index.stats()["n_compactions"] == 1
    after_scores, after_ids = index.search(vecs[:30], k=15)
    assert after_ids.shape == before_ids.shape
    for row in range(30):
        a = set(int(x) for x in before_ids[row] if int(x) != PAD_ID)
        b = set(int(x) for x in after_ids[row] if int(x) != PAD_ID)
        assert a == b or len(a & b) / float(max(1, len(a | b))) > 0.9
    returned = set(int(x) for x in after_ids.ravel().tolist() if int(x) != PAD_ID)
    assert returned.isdisjoint(removed)
    # compact after deleting everything must not crash
    index.remove(before_alive)
    assert index.compact() == 0
    assert index.search(vecs[:2], k=3)[1].tolist() == [[PAD_ID] * 3, [PAD_ID] * 3]


@pytest.mark.parametrize("backend", BACKENDS)
def test_maybe_compact_threshold(backend):
    index, _, _, _, _ = _make_index(backend, n=100, compact_ratio=0.1)
    assert index.maybe_compact() is False
    index.remove(list(range(20)))
    assert index.maybe_compact() is True
    assert index.stats()["n_compactions"] == 1


# --------------------------------------------------------------------------------------
# 5. Edges: k exceeds alive count, empty index, padding
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize("backend", BACKENDS)
def test_k_exceeds_alive_and_empty_index(backend):
    index = ProvenanceIndex(DIM, backend=backend)
    assert index.stats()["n_vectors"] == 0 and index.stats()["n_deleted"] == 0
    assert index.alive_ids().size == 0 and index.deleted_ids().size == 0
    assert index.ids_for_doc("x") == [] and index.ids_for_client("x") == []

    scores, ids = index.search(np.zeros((3, DIM), dtype=np.float32), k=5)
    assert scores.shape == (3, 5) and ids.shape == (3, 5)
    assert (ids == PAD_ID).all() and np.all(np.isneginf(scores))
    assert strip_padding(ids, scores) == [[], [], []]

    # 7 alive entries; k=5 (enough) and k=10 (not enough -> padding)
    vecs, metas, _ = _corpus(n=7)
    index.add(vecs, metas)
    s5, i5 = index.search(vecs[:2], k=5)
    assert (i5 != PAD_ID).all()
    s10, i10 = index.search(vecs[:2], k=10)
    assert i10.shape == (2, 10)
    for row in range(2):
        valid = i10[row][i10[row] != PAD_ID]
        assert valid.size == 7
        assert (i10[row, 7:] == PAD_ID).all()
        assert np.isneginf(s10[row, 7:]).all()
        assert np.all(np.diff(s10[row, :7]) <= 1e-6)
    rows = strip_padding(i10, s10)
    assert all(len(r) == 7 for r in rows)


# --------------------------------------------------------------------------------------
# 6. save / load round-trip
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize("backend", BACKENDS)
def test_save_load_roundtrip(backend, tmp_path):
    index, vecs, metas, _, _ = _make_index(backend)
    rng = np.random.default_rng(SEED + 2)
    removed = sorted(rng.choice(N, size=60, replace=False).tolist())
    index.remove(removed)

    path = tmp_path / ("idx_%s" % backend)
    index.save(str(path))
    assert (path / "vectors.npz").exists()
    assert (path / "meta.json").exists()
    assert (path / "manifest.json").exists()
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["n_total"] == N and manifest["n_deleted"] == len(removed)
    assert manifest["dim"] == DIM

    loaded = ProvenanceIndex.load(str(path))
    assert loaded is not index
    assert loaded.dim == DIM
    assert loaded.alive_ids().tolist() == index.alive_ids().tolist()
    assert loaded.stats()["n_vectors"] == index.stats()["n_vectors"]
    assert loaded.stats()["n_deleted"] == index.stats()["n_deleted"]
    assert loaded.stats()["backend"] == index.stats()["backend"]
    assert loaded.backend == index.backend
    for i in range(N):
        assert loaded.meta(i) == index.meta(i)
    for c in range(N_CLIENTS):
        assert loaded.ids_for_client("c%d" % c) == index.ids_for_client("c%d" % c)
    for m in metas[::7]:
        assert loaded.ids_for_doc(m.doc_id) == index.ids_for_doc(m.doc_id)

    queries = vecs[:40]
    _assert_search_matches(index, loaded, queries, k=10)
    # After load, behavior stays consistent: further deletes/adds are also consistent
    assert loaded.remove(removed) is None
    assert loaded.stats()["n_deleted"] == len(removed)
    new_id = loaded.add(vecs[:3], metas[:3])
    assert new_id == list(range(N, N + 3))
    # New shards of the same doc_id are appended at the end; whether the original shard is visible depends on whether it is in removed
    assert loaded.ids_for_doc(metas[0].doc_id) == ([0] if loaded.is_alive(0) else []) + [N]


@pytest.mark.parametrize("backend", BACKENDS)
def test_save_load_empty_index(backend, tmp_path):
    index = ProvenanceIndex(DIM, backend=backend)
    path = tmp_path / ("empty_%s" % backend)
    index.save(str(path))
    loaded = ProvenanceIndex.load(str(path))
    assert loaded.stats()["n_vectors"] == 0 and loaded.stats()["n_deleted"] == 0
    assert loaded.alive_ids().size == 0
    scores, ids = loaded.search(np.zeros((2, DIM), dtype=np.float32), k=4)
    assert (ids == PAD_ID).all() and np.all(np.isneginf(scores))


def test_load_missing_dir_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        ProvenanceIndex.load(str(tmp_path / "nope"))


# --------------------------------------------------------------------------------------
# 7. Real faiss IVF path (IVF only when n >= training threshold; skipped if faiss is missing)
# --------------------------------------------------------------------------------------
def test_faiss_ivf_training_remove_and_roundtrip(tmp_path):
    if "faiss_ivf" not in BACKENDS:
        pytest.skip("faiss unavailable")
    vecs, metas, _ = _corpus(n=1500, n_clients=5, seed=SEED + 11)
    index = ProvenanceIndex(DIM, backend="faiss_ivf")
    index.add(vecs, metas)
    assert index.backend == "faiss_ivf"
    assert index.stats()["fallback_reason"] is None
    assert getattr(index._ann, "_mode", None) == "ivf", "1500 entries should trigger the IVF training branch"
    # The flat -> IVF size upgrade does not count as compact (otherwise E5 cost stats would be polluted)
    assert index.stats()["n_compactions"] == 0

    # IVF is approximate retrieval: compare top-10 overlap with the exact backend
    exact = ProvenanceIndex(DIM, backend="numpy")
    exact.add(vecs, metas)
    _assert_search_matches(index, exact, vecs[:30], k=10, min_jaccard=0.85)

    # Never return deleted ids after removal (IVF remove_ids goes through IDSelectorBatch)
    rng = np.random.default_rng(SEED + 12)
    removed = sorted(rng.choice(1500, size=450, replace=False).tolist())
    index.remove(removed)
    assert index.stats()["n_deleted"] == len(removed)
    scores, ids = index.search(vecs[removed[:30]], k=10)
    returned = set(int(x) for x in ids.ravel().tolist() if int(x) != PAD_ID)
    assert returned.isdisjoint(removed)
    assert (ids[row, 0] != removed[row] for row in range(30))

    # Native ann.faiss file round-trip (including tombstones)
    path = tmp_path / "ivf_idx"
    index.save(str(path))
    assert (path / "ann.faiss").exists()
    loaded = ProvenanceIndex.load(str(path))
    assert loaded.backend == "faiss_ivf"
    assert loaded.alive_ids().tolist() == index.alive_ids().tolist()
    _assert_search_matches(index, loaded, vecs[:30], k=10, min_jaccard=0.9)

    # Still IVF after compact, and behavior does not degrade
    index.compact()
    after = index.search(vecs[:30], k=10)[1]
    before = loaded.search(vecs[:30], k=10)[1]
    overlap = []
    for r in range(30):
        a = set(int(x) for x in after[r] if int(x) != PAD_ID)
        b = set(int(x) for x in before[r] if int(x) != PAD_ID)
        overlap.append(len(a & b) / float(max(1, len(a | b))))
    assert float(np.mean(overlap)) > 0.8


@pytest.mark.parametrize("backend", BACKENDS)
def test_save_load_after_add_remove_add(backend, tmp_path):
    index, vecs, metas, _, _ = _make_index(backend, n=80)
    index.remove(list(range(0, 80, 3)))
    extra = _corpus(n=20, seed=SEED + 9)[0]
    new_ids = index.add(extra, metas[:20])
    assert new_ids == list(range(80, 100))
    path = tmp_path / ("seq_%s" % backend)
    index.save(str(path))
    loaded = ProvenanceIndex.load(str(path))
    assert loaded.stats()["n_deleted"] == index.stats()["n_deleted"]
    assert loaded.alive_ids().tolist() == index.alive_ids().tolist()
    _assert_search_matches(index, loaded, np.concatenate([vecs[:20], extra[:10]], axis=0), k=8)
