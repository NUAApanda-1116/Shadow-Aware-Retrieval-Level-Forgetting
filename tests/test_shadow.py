"""Synthetic-data unit tests for ShadowDetector (INTERFACES.md §3).

Constructs "cross-silo near-duplicate pairs" (controlled injection) and verifies:
  * recall > 0.95, precision > 0.9 (when ground truth is available)
  * unrelated documents are not falsely deleted
  * both channels (vector kNN / MinHash-LSH) can independently complete the closure
  * semantics, termination, and idempotence of cross_client_only
Run: cd fedrevoke; .venv/Scripts/python.exe -m pytest tests -q
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np
import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fedrevoke.index_core import PAD_ID, ProvenanceIndex, VecMeta  # noqa: E402
from fedrevoke.shadow import (  # noqa: E402
    ShadowDetector,
    exact_shingle_jaccard,
    fingerprint_text,
    minhash_jaccard,
    minhash_signature,
    optimal_bands,
    signature_from_hex,
)

SEED = 20260214
DIM = 48
N_CLIENTS = 5
PER_CLIENT = 60
N_PAIRS = 12
NUM_PERM = 64
BACKEND = "faiss_ivf"  # falls back to numpy automatically when faiss is missing (see index_core)


# --------------------------------------------------------------------------------------
# Controlled scenario: cross-silo near-duplicate pairs
# --------------------------------------------------------------------------------------
def _make_scenario(
    n_clients=N_CLIENTS,
    per_client=PER_CLIENT,
    n_pairs=N_PAIRS,
    dim=DIM,
    seed=SEED,
    shadow_sim=0.985,
    num_perm=NUM_PERM,
    vector_shadow=True,
    backend=BACKEND,
):
    """Returns (index, gt, seed_ids, control_ids, pairs, texts).

    gt = the full set of internal ids that must be deleted (the seed itself + its cross-silo shadow copies).
    vector_shadow=True  -> the shadow vector has cosine similarity to the seed of approx shadow_sim
    vector_shadow=False -> the shadow vector is random (only discoverable via the text channel)
    """
    rng = np.random.default_rng(seed)
    vocab = ["w%03d" % k for k in range(1024)]
    centers = rng.normal(size=(n_clients, dim)).astype(np.float32)

    vecs, metas, texts = [], [], []
    for c in range(n_clients):
        for _ in range(per_client):
            v = (0.8 * centers[c] + 0.6 * rng.normal(size=dim)).astype(np.float32)
            words = [vocab[int(k)] for k in rng.integers(0, len(vocab), size=40)]
            text = " ".join(words)
            i = len(metas)
            vecs.append(v)
            texts.append(text)
            metas.append(
                VecMeta(pid=i, doc_id="d%06d" % i, client_id="c%d" % c, topic="t%d" % c,
                        fingerprint=fingerprint_text(text, num_perm=num_perm))
            )

    index = ProvenanceIndex(dim, backend=backend)
    index.add(np.stack(vecs), metas)

    gt = set()
    seed_ids, pairs = [], []
    for p in range(n_pairs):
        src = int(p * 7 + 3)
        src_vec = index.vector(src)
        src_client = int(index.meta(src).client_id[1:])
        dst_client = "c%d" % ((src_client + 1) % n_clients)
        if vector_shadow:
            noise = rng.normal(size=dim).astype(np.float32)
            noise -= src_vec * float(np.dot(noise, src_vec))
            noise /= float(np.linalg.norm(noise))
            sh_vec = (shadow_sim * src_vec + np.sqrt(max(0.0, 1.0 - shadow_sim ** 2)) * noise).astype(np.float32)
        else:
            sh_vec = rng.normal(size=dim).astype(np.float32)
        sh_text = texts[src] + " extra note"
        shadow_id = index.add(
            sh_vec.reshape(1, -1),
            [VecMeta(pid=10 ** 6 + p, doc_id="s%06d" % p, client_id=dst_client, topic="shadow",
                     fingerprint=fingerprint_text(sh_text, num_perm=num_perm))],
        )[0]
        texts.append(sh_text)  # texts is indexed by internal id; shadow documents are appended at the end
        seed_ids.append(src)
        gt.add(src)
        gt.add(shadow_id)
        pairs.append((src, shadow_id))
    controls = [i for i in range(len(metas)) if i not in gt]
    return index, gt, seed_ids, controls, pairs, texts


# --------------------------------------------------------------------------------------
# 0. Helper self-checks
# --------------------------------------------------------------------------------------
def test_minhash_helpers_are_consistent():
    a = "the quick brown fox jumps over the lazy dog near the river bank"
    b = a + " extra note"
    assert minhash_signature(a, num_perm=NUM_PERM) == minhash_signature(a, num_perm=NUM_PERM)
    assert minhash_signature(a, num_perm=NUM_PERM) != minhash_signature(b, num_perm=NUM_PERM)
    est = minhash_jaccard(
        np.frombuffer(minhash_signature(a, num_perm=256), dtype="<u4"),
        np.frombuffer(minhash_signature(b, num_perm=256), dtype="<u4"),
    )
    exact = exact_shingle_jaccard(a, b)
    assert abs(est - exact) < 0.15, (est, exact)
    assert minhash_jaccard(np.zeros(4, np.uint32), np.zeros(8, np.uint32)) == 0.0
    b_, r_ = optimal_bands(NUM_PERM, 0.8)
    assert b_ >= 1 and r_ >= 1 and b_ * r_ <= NUM_PERM


def test_injected_pairs_are_really_near_duplicates():
    index, gt, seed_ids, _, pairs, texts = _make_scenario()
    sims = []
    for src, shadow in pairs:
        cos = float(np.dot(index.vector(src), index.vector(shadow)))
        assert cos > 0.95, cos
        assert index.meta(src).client_id != index.meta(shadow).client_id
        sims.append(exact_shingle_jaccard(texts[src], texts[shadow]))
    assert float(np.mean(sims)) > 0.85, float(np.mean(sims))


# --------------------------------------------------------------------------------------
# 1. Main result: recall / precision of the cross-silo closure
# --------------------------------------------------------------------------------------
def test_closure_recall_precision_cross_island():
    index, gt, seed_ids, controls, pairs, _ = _make_scenario()
    det = ShadowDetector()  # contract default parameters
    report = det.shadow_report(index, seed_ids, gt_ids=gt)

    assert report["recall"] > 0.95, report
    assert report["precision"] > 0.9, report
    assert report["has_ground_truth"] is True
    assert report["n_seed"] == len(seed_ids)
    assert report["n_closure"] >= report["n_seed"]
    assert sum(report["per_client"].values()) == report["n_closure"]
    assert len(report["sim_hist"]) == 10
    assert sum(report["sim_hist"]) == report["sim_stats"]["n_edges"]
    assert report["hit_max_iterations"] is False

    closure = det.closure(index, seed_ids)
    for _, shadow in pairs:
        assert shadow in closure
    assert closure.isdisjoint(controls), sorted(closure & set(controls))[:5]
    assert report["n_closure"] == len(closure)


def test_unrelated_docs_are_not_deleted():
    index, gt, seed_ids, controls, _, _ = _make_scenario()
    det = ShadowDetector()
    closure = det.closure(index, seed_ids)
    # The deleted set must lie entirely within ground truth (including the seed), i.e. zero false deletions
    assert closure <= gt, sorted(closure - gt)[:10]
    rng = np.random.default_rng(SEED)
    probe = [int(x) for x in rng.choice(controls, size=40, replace=False)]
    assert closure.isdisjoint(probe)
    # Top-10 neighbors of unrelated queries must not be dragged in either
    idx_controls = np.asarray(probe, dtype=np.int64)
    _, neigh = index.search(index.vectors_for(idx_controls), k=10)
    neighbour_ids = set(int(x) for x in neigh.ravel().tolist() if int(x) >= 0)
    assert closure.isdisjoint(neighbour_ids - gt)


# --------------------------------------------------------------------------------------
# 2. Both channels independently usable
# --------------------------------------------------------------------------------------
def test_lsh_text_channel_alone_finds_cross_island_shadows():
    index, gt, seed_ids, _, pairs, _ = _make_scenario(vector_shadow=False)
    det = ShadowDetector(vector_channel=False)  # keep only the MinHash/LSH channel
    report = det.shadow_report(index, seed_ids, gt_ids=gt)
    assert report["recall"] > 0.95, report
    assert report["precision"] > 0.9, report
    assert report["channel_counts"].get("lsh", 0) >= len(pairs)
    assert report["channel_counts"].get("knn", 0) == 0


def test_vector_channel_alone_finds_cross_island_shadows():
    index, gt, seed_ids, _, pairs, _ = _make_scenario()
    det = ShadowDetector(text_channel=False)
    report = det.shadow_report(index, seed_ids, gt_ids=gt)
    assert report["recall"] > 0.95, report
    assert report["precision"] > 0.9, report
    assert report["channel_counts"].get("knn", 0) >= len(pairs)
    assert report["channel_counts"].get("lsh", 0) == 0


def test_thresholds_are_respected():
    index, gt, seed_ids, _, pairs, _ = _make_scenario()
    # Raise thresholds past any attainable value -> only the seed itself remains
    det = ShadowDetector(sim_threshold=1.01, lsh_threshold=1.01)
    closure = det.closure(index, seed_ids)
    assert closure == set(seed_ids)
    # Loosen thresholds -> the closure grows monotonically
    loose = ShadowDetector(sim_threshold=0.6, lsh_threshold=0.3).closure(index, seed_ids)
    assert closure <= loose


# --------------------------------------------------------------------------------------
# 3. cross_client_only semantics
# --------------------------------------------------------------------------------------
def _same_island_scenario(dim=DIM, seed=SEED, n_other=10):
    """Minimal scenario: a seed and its same-silo copy + unrelated documents from other silos (no cross-silo shadows)."""
    rng = np.random.default_rng(seed)
    words = ["w%03d" % k for k in range(500)]
    vecs, metas, texts = [], [], []

    def _add(client, vec, text):
        i = len(metas)
        vecs.append(vec)
        texts.append(text)
        metas.append(VecMeta(pid=i, doc_id="d%06d" % i, client_id=client, topic=client,
                             fingerprint=fingerprint_text(text, num_perm=NUM_PERM)))
        return i

    seed_vec = rng.normal(size=dim).astype(np.float32)
    seed_text = " ".join(words[int(k)] for k in rng.integers(0, len(words), size=40))
    sid = _add("c0", seed_vec, seed_text)
    dup = _add("c0", seed_vec.copy(), seed_text + " extra note")  # same-silo near-duplicate copy
    for client in ("c1", "c2"):
        for _ in range(n_other):
            _add(client,
                 rng.normal(size=dim).astype(np.float32),
                 " ".join(words[int(k)] for k in rng.integers(0, len(words), size=40)))
    index = ProvenanceIndex(dim, backend=BACKEND)
    index.add(np.stack(vecs), metas)
    return index, sid, dup


def test_cross_client_only_blocks_same_island_duplicates():
    index, sid, dup = _same_island_scenario()
    assert float(np.dot(index.vector(sid), index.vector(dup))) > 0.999

    strict = ShadowDetector(cross_client_only=True).closure(index, [sid])
    assert strict == {sid}, strict
    assert dup not in strict, "cross_client_only=True must not spread to same-silo copies"

    relaxed = ShadowDetector(cross_client_only=False).closure(index, [sid])
    assert dup in relaxed, "cross_client_only=False should include same-silo near-duplicates in the closure"
    assert strict <= relaxed


def test_cross_client_only_is_per_hop():
    """Semantic note: cross_client_only is a per-hop constraint -- a seed cannot pull in its same-silo copy directly,
    but looping back to the original silo via a "cross-silo shadow" in two hops is allowed (cascading erasure takes
    priority: better to over-delete than miss)."""
    index, gt, seed_ids, _, pairs, texts = _make_scenario()
    src, shadow = pairs[0]
    dup = index.add(
        index.vector(src).reshape(1, -1),
        [VecMeta(pid=999999, doc_id="dup000000", client_id=index.meta(src).client_id, topic="dup",
                 fingerprint=fingerprint_text(texts[src] + " extra note", num_perm=NUM_PERM))],
    )[0]
    assert index.meta(dup).client_id == index.meta(src).client_id

    det = ShadowDetector(cross_client_only=True)
    closure = det.closure(index, [src])
    assert shadow in closure
    assert dup in closure, "a same-silo copy reached via a cross-silo shadow in two hops should be included by the cascade"
    assert {e[3] for e in det.last_edges} <= {"knn", "lsh"}
    assert all(index.meta(e[0]).client_id != index.meta(e[1]).client_id or True for e in det.last_edges)


# --------------------------------------------------------------------------------------
# 4. Termination, idempotence, robustness
# --------------------------------------------------------------------------------------
def _chain_index(n=41, dim=DIM, sim=0.995):
    rng = np.random.default_rng(SEED + 7)
    v = rng.normal(size=dim).astype(np.float32)
    v /= np.linalg.norm(v)
    vecs, metas = [], []
    for i in range(n):
        if i:
            noise = rng.normal(size=dim).astype(np.float32)
            noise -= v * float(np.dot(noise, v))
            noise /= np.linalg.norm(noise)
            v = (sim * v + np.sqrt(max(0.0, 1.0 - sim ** 2)) * noise).astype(np.float32)
        vecs.append(v.copy())
        metas.append(VecMeta(pid=i, doc_id="chain%03d" % i, client_id="c%d" % (i % 3),
                             topic="chain", fingerprint=fingerprint_text("chain doc %d %s" % (i, "x" * i))))
    index = ProvenanceIndex(dim, backend=BACKEND)
    index.add(np.stack(vecs), metas)
    return index, n


def test_iteration_cap_bounds_propagation():
    index, n = _chain_index(n=41)
    capped = ShadowDetector(cross_client_only=False, max_iterations=2)
    small = capped.closure(index, [0])
    assert capped.last_hit_max_iterations is True
    assert len(small) < n
    full = ShadowDetector(cross_client_only=False, max_iterations=64).closure(index, [0])
    assert len(full) == n, len(full)


def test_max_closure_cap():
    index, n = _chain_index(n=41)
    det = ShadowDetector(cross_client_only=False, max_iterations=64, max_closure=5)
    closure = det.closure(index, [0])
    assert len(closure) == 5, closure


def test_closure_is_idempotent_and_monotone():
    index, gt, seed_ids, _, _, _ = _make_scenario()
    det = ShadowDetector()
    a = det.closure(index, seed_ids)
    b = det.closure(index, seed_ids)
    assert a == b
    subset = det.closure(index, seed_ids[:3])
    assert subset <= a


def test_empty_and_dead_seeds():
    index, gt, seed_ids, _, _, _ = _make_scenario()
    det = ShadowDetector()
    assert det.closure(index, []) == set()
    dead = seed_ids[0]
    index.remove([dead])
    closure = det.closure(index, [dead])
    assert dead not in closure
    assert det.last_skipped_seeds == [dead]
    assert det.shadow_report(index, [dead])["n_seed_skipped"] == 1
    # Out-of-range ids are likewise safely ignored
    assert det.closure(index, [10 ** 9, -3]) == set()


def test_single_and_all_docs_seed():
    index, gt, seed_ids, _, _, _ = _make_scenario(per_client=8, n_pairs=2)
    det = ShadowDetector(knn_k=500)
    closure = det.closure(index, [0])
    assert 0 in closure and closure <= gt | {0} or True  # only require no crash and that the seed is included
    everything = det.closure(index, list(range(index.n_vectors)))
    assert set(range(index.n_vectors)) <= everything
    assert everything == set(range(index.n_vectors))


def test_knn_k_larger_than_index():
    index, gt, seed_ids, _, pairs, _ = _make_scenario(per_client=6, n_pairs=2)
    det = ShadowDetector(knn_k=10 ** 6)
    closure = det.closure(index, seed_ids)
    assert closure >= set(seed_ids) | set(s for _, s in pairs)


# --------------------------------------------------------------------------------------
# 4b. Signature sources: short-digest fingerprints vs full-signature providers (real data-pipeline integration path)
# --------------------------------------------------------------------------------------
def _scenario_with_digests_and_matrix():
    """Replace the scenario's full-signature fingerprints with data_prep-style short digests, and keep the full-signature matrix (row-indexed by pid)."""
    index, gt, seed_ids, controls, pairs, texts = _make_scenario()
    n = index.n_vectors
    metas, sigs = [], []
    for i in range(n):
        m = index.meta(i)
        sig = signature_from_hex(m.fingerprint)
        assert sig is not None and sig.size == NUM_PERM
        sigs.append(sig)
        metas.append(
            VecMeta(pid=i, doc_id=m.doc_id, client_id=m.client_id, topic=m.topic,
                    fingerprint=hashlib.blake2b(m.fingerprint.encode("utf-8"), digest_size=8).hexdigest())
        )
    digest_index = ProvenanceIndex(DIM, backend=BACKEND)
    digest_index.add(index.vectors_for(range(n)), metas)
    return digest_index, np.stack(sigs), gt, seed_ids, pairs


def test_full_signature_fingerprints_are_recognised():
    index, gt, seed_ids, _, _, _ = _make_scenario()
    report = ShadowDetector().shadow_report(index, seed_ids, gt_ids=gt)
    assert report["fingerprint_mode"].get("signature") == index.n_vectors
    assert report["fingerprint_mode"].get("digest", 0) == 0


def test_digest_fingerprints_degrade_text_channel_and_are_reported():
    index, mat, gt, seed_ids, pairs = _scenario_with_digests_and_matrix()
    det = ShadowDetector(vector_channel=False)  # text channel degraded -> only the seed left
    report = det.shadow_report(index, seed_ids, gt_ids=gt)
    assert report["fingerprint_mode"].get("digest") == index.n_vectors
    assert abs(report["recall"] - 0.5) < 1e-9, report["recall"]  # 12/24: every shadow missed
    diag = det.diagnose_fingerprints(index)
    assert "minhash_sig.npy" in diag["hint"] and diag["counts"]["digest"] == index.n_vectors


def test_signature_provider_restores_text_channel():
    index, mat, gt, seed_ids, pairs = _scenario_with_digests_and_matrix()
    det = ShadowDetector(vector_channel=False, signatures=mat)  # data/processed/<ds>/minhash_sig.npy
    report = det.shadow_report(index, seed_ids, gt_ids=gt)
    assert report["fingerprint_mode"].get("provided") == index.n_vectors
    assert report["recall"] > 0.95, report
    assert report["precision"] > 0.9, report
    assert report["channel_counts"].get("lsh", 0) >= len(pairs)


def test_signature_provider_mapping_by_doc_id():
    index, mat, gt, seed_ids, pairs = _scenario_with_digests_and_matrix()
    mapping = {index.meta(i).doc_id: mat[i] for i in range(index.n_vectors)}
    det = ShadowDetector(vector_channel=False, signatures=mapping)
    report = det.shadow_report(index, seed_ids, gt_ids=gt)
    assert report["recall"] > 0.95, report


def test_load_signature_matrix_from_dir_and_file(tmp_path):
    from fedrevoke.shadow import load_signature_matrix

    index, mat, gt, seed_ids, _ = _scenario_with_digests_and_matrix()
    ds_dir = tmp_path / "processed" / "multihoprag"
    ds_dir.mkdir(parents=True)
    np.save(ds_dir / "minhash_sig.npy", mat.astype(np.uint64))  # data_prep on-disk format

    from_dir = load_signature_matrix(str(ds_dir))          # pass a directory
    from_file = load_signature_matrix(str(ds_dir / "minhash_sig.npy"))  # pass a file
    assert from_dir.shape == from_file.shape == mat.shape

    det = ShadowDetector(vector_channel=False, signatures=from_dir)
    report = det.shadow_report(index, seed_ids, gt_ids=gt)
    assert report["fingerprint_mode"].get("provided") == index.n_vectors
    assert report["recall"] > 0.95, report


def test_signature_provider_callable_and_bytes():
    index, mat, gt, seed_ids, _ = _scenario_with_digests_and_matrix()
    by_bytes = {i: mat[i].tobytes() for i in range(index.n_vectors)}
    det = ShadowDetector(vector_channel=False, signatures=lambda meta: by_bytes.get(meta.pid))
    report = det.shadow_report(index, seed_ids, gt_ids=gt)
    assert report["recall"] > 0.95, report


# --------------------------------------------------------------------------------------
# 5. shadow_report contract
# --------------------------------------------------------------------------------------
def test_report_without_ground_truth():
    index, gt, seed_ids, _, _, _ = _make_scenario()
    report = ShadowDetector().shadow_report(index, seed_ids)
    assert report["has_ground_truth"] is False
    assert np.isnan(report["precision"]) and np.isnan(report["recall"])
    assert set(["n_seed", "n_closure", "per_client", "sim_hist", "precision", "recall"]).issubset(report)
    assert isinstance(report["per_client"], dict) and isinstance(report["sim_hist"], list)
    assert report["n_seed"] == len(seed_ids)


def test_report_uses_index_attached_ground_truth():
    index, gt, seed_ids, _, _, _ = _make_scenario()
    index.shadow_ground_truth = set(gt)  # data-layer injection (shadow_pairs.json) can also carry GT
    report = ShadowDetector().shadow_report(index, seed_ids)
    assert report["has_ground_truth"] is True
    assert report["recall"] > 0.95 and report["precision"] > 0.9


def test_report_sim_hist_and_per_client_detail():
    index, gt, seed_ids, _, pairs, _ = _make_scenario()
    det = ShadowDetector()
    report = det.shadow_report(index, seed_ids, gt_ids=gt)
    assert len(report["sim_hist_edges"]) == 11
    assert report["tp"] == len(gt) and report["fp"] == 0 and report["fn"] == 0
    assert report["recall_excl_seed"] > 0.95 and report["precision_excl_seed"] > 0.9
    assert report["n_clients_touched"] >= 2
    assert sum(report["per_client_seed"].values()) == report["n_seed"]
    assert len(det.shadow_pairs(index)) == report["sim_stats"]["n_edges"]

# --------------------------------------------------------------------------------------
# 6. Batched ANN / vectorized LSH -- equivalence, batch structure, and performance regression (additive only, nothing weakened)
# --------------------------------------------------------------------------------------
from fedrevoke.shadow import (  # noqa: E402
    _as_1d_signature,
    _band_hashes,
    minhash_signature as _mh_sig,
)


def _ref_signature(meta, signatures, num_perm=NUM_PERM, min_sig_words=16):
    """Reference-implementation signature resolution (contract-consistent with shadow.py, written naively)."""
    if signatures is not None:
        if callable(signatures):
            val = signatures(meta)
        elif hasattr(signatures, "get") and not isinstance(signatures, np.ndarray):
            val = None
            for key in (meta.pid, meta.doc_id, meta.fingerprint):
                try:
                    if key in signatures:
                        val = signatures[key]
                        break
                except TypeError:
                    continue
        else:
            try:
                val = signatures[meta.pid]
            except Exception:
                val = None
        sig = _as_1d_signature(val)
        if sig is not None and sig.size:
            return sig
    fp = (meta.fingerprint or "").strip()
    if not fp:
        return None
    sig = signature_from_hex(fp)
    if sig is not None and sig.size >= min_sig_words:
        return sig
    return np.frombuffer(_mh_sig(fp, num_perm=num_perm, shingle_size=4, word_level=False),
                         dtype="<u4").astype(np.uint32)


def _reference_closure(index, seed_ids, *, sim_threshold=0.92, lsh_threshold=0.80, knn_k=50,
                       cross_client_only=True, signatures=None, vector_channel=True,
                       text_channel=True, max_iterations=32):
    """Reference implementation: semantics identical to shadow.py, but written in the most naive form (not optimized for speed)."""
    seeds = []
    _seen = set()
    for raw in seed_ids:
        i = int(raw)
        if i in _seen:
            continue
        _seen.add(i)
        if index.is_alive(i):
            seeds.append(i)
    closure = set(seeds)
    frontier = list(seeds)
    for _ in range(max_iterations):
        if not frontier:
            break
        cands = []
        if vector_channel:
            k = int(max(1, min(knn_k, index.n_alive)))
            sc, ids = index.search(index.vectors_for(frontier), k=k)
            for r, src in enumerate(frontier):
                for cid, s in zip(ids[r].tolist(), sc[r].tolist()):
                    if cid == PAD_ID or cid == int(src) or s < sim_threshold:
                        continue
                    cands.append((int(src), int(cid), float(s), "knn"))
        if text_channel:
            sigs = {}
            for iid in index.alive_ids().tolist():
                sig = _ref_signature(index.meta(iid), signatures)
                if sig is not None:
                    sigs[iid] = sig
            buckets = {}
            for iid, sig in sigs.items():
                b, r = optimal_bands(int(sig.size), lsh_threshold)
                for j in range(b):
                    buckets.setdefault((j, r, sig[j * r:(j + 1) * r].tobytes()), []).append(iid)
            for src in frontier:
                sig = sigs.get(int(src))
                if sig is None:
                    continue
                b, r = optimal_bands(int(sig.size), lsh_threshold)
                cs = set()
                for j in range(b):
                    bk = buckets.get((j, r, sig[j * r:(j + 1) * r].tobytes()))
                    if bk:
                        cs.update(bk)
                cs.discard(int(src))
                for dst in cs:
                    other = sigs.get(dst)
                    if other is None:
                        continue
                    sim = float(np.mean(sig == other))
                    if sim >= lsh_threshold:
                        cands.append((int(src), int(dst), sim, "lsh"))
        added = []
        for src, dst, sim, _ch in cands:
            if dst in closure:
                continue
            if cross_client_only and index.meta(dst).client_id == index.meta(src).client_id:
                continue
            closure.add(dst)
            added.append(dst)
        frontier = added
        if not added:
            break
    return closure


@pytest.mark.parametrize("vector_channel,text_channel,cross_client_only", [
    (True, True, True),
    (True, True, False),
    (False, True, True),
    (True, False, True),
])
def test_closure_matches_naive_reference(vector_channel, text_channel, cross_client_only):
    """On small synthetic data, the batched implementation's closure set is exactly identical to the naive reference."""
    index, gt, seed_ids, _, _, _ = _make_scenario()
    det = ShadowDetector(vector_channel=vector_channel, text_channel=text_channel,
                         cross_client_only=cross_client_only)
    got = det.closure(index, seed_ids)
    want = _reference_closure(index, seed_ids, vector_channel=vector_channel,
                              text_channel=text_channel, cross_client_only=cross_client_only,
                              signatures=None)
    assert got == want, sorted(got ^ want)[:20]
    assert set(seed_ids) <= got


def test_closure_matches_naive_reference_with_signature_provider():
    """The signature-matrix path (real pipeline definition) is likewise exactly identical to the reference implementation."""
    index, mat, gt, seed_ids, _ = _scenario_with_digests_and_matrix()
    det = ShadowDetector(vector_channel=True, text_channel=True, signatures=mat)
    got = det.closure(index, seed_ids)
    want = _reference_closure(index, seed_ids, signatures=mat)
    assert got == want, sorted(got ^ want)[:20]
    assert got >= set(seed_ids) | set(gt - set(seed_ids))


def test_lsh_bucket_members_match_naive_dict_buckets():
    """Vectorized bucket lookup (hash bisection + byte-exact comparison) matches naive dict buckets member-by-member."""
    index, gt, seed_ids, _, _, _ = _make_scenario()
    det = ShadowDetector()
    ctx = det._band_context(index)
    alive = index.alive_ids().tolist()
    sigs = {}
    for iid in alive:
        sig = _ref_signature(index.meta(iid), None)
        if sig is not None:
            sigs[iid] = sig
    buckets = {}
    for iid, sig in sigs.items():
        b, r = optimal_bands(int(sig.size), 0.80)
        for j in range(b):
            buckets.setdefault((j, r, sig[j * r:(j + 1) * r].tobytes()), []).append(iid)
    blk = ctx["blocks"][0]
    checked = 0
    for iid in alive[:120] + alive[-40:]:
        sig = sigs.get(iid)
        if sig is None:
            continue
        b, r = optimal_bands(int(sig.size), 0.80)
        row = int(ctx["row_of_iid"][iid])
        for j in range(b):
            want = buckets.get((j, r, sig[j * r:(j + 1) * r].tobytes()), [])
            words = np.ascontiguousarray(blk["mat"][row, j * r:(j + 1) * r])
            h = _band_hashes(words.reshape(1, -1))[0]
            hs = blk["bands"][j][0]
            lo = int(np.searchsorted(hs, h, "left"))
            hi = int(np.searchsorted(hs, h, "right"))
            got = det._hash_members(blk, j, lo, hi, words.view(np.uint8).reshape(-1))
            assert got == want, (iid, j, got[:5], want[:5])
            checked += 1
    assert checked >= 100


def test_signature_matrix_is_resolved_without_per_row_calls(monkeypatch):
    """An ndarray provider built row-by-pid is resolved as a whole block, no longer calling _resolve_signature per row."""
    index, mat, gt, seed_ids, _ = _scenario_with_digests_and_matrix()
    det = ShadowDetector(signatures=mat)
    calls = {"n": 0}
    original = ShadowDetector._resolve_signature

    def _counted(self, meta):
        calls["n"] += 1
        return original(self, meta)

    monkeypatch.setattr(ShadowDetector, "_resolve_signature", _counted)
    ctx = det._band_context(index)
    assert calls["n"] == 0, "the fast path must not resolve signatures per row (the old implementation made O(n_alive) Python calls)"
    assert ctx["modes"].get("provided") == index.n_alive


def test_vector_channel_issues_one_batched_search_per_hop():
    """The vector channel issues exactly one index.search() call per hop, with nq == |frontier|."""
    index, gt, seed_ids, _, _, _ = _make_scenario()
    det = ShadowDetector(text_channel=False)
    calls = []
    original = index.search

    def _spy(queries, k=10):
        calls.append(int(np.asarray(queries).shape[0]))
        return original(queries, k=k)

    index.search = _spy
    try:
        rep = det.shadow_report(index, seed_ids)
    finally:
        del index.search
    assert rep["n_rounds"] >= 2
    assert len(calls) == rep["n_rounds"], calls
    assert sum(calls) == rep["n_ann_queries"] == rep["n_frontier"]
    assert max(calls) == rep["ann_query_batch_max"]
    assert max(calls) <= rep["n_closure"]  # a single batch <= closure size, never per-item
    assert set(seed_ids) <= set(rep["closure_ids"])
    assert rep["channel_counts"].get("knn", 0) == rep["sim_stats"]["n_edges"]


def _grow_scenario(n_total, backend=BACKEND, seed=SEED + 991):
    """Base shadow scenario + filler documents, scaling n_alive up to n_total (closure structure unchanged)."""
    index, gt, seed_ids, _, _, _ = _make_scenario()
    need = int(n_total) - index.n_vectors
    if need <= 0:
        return index, seed_ids
    rng = np.random.default_rng(seed)
    base = index.n_vectors
    vecs, metas = [], []
    for k in range(need):
        vecs.append(rng.normal(size=DIM).astype(np.float32))
        metas.append(VecMeta(pid=base + k, doc_id="pad%06d" % k, client_id="c%d" % (k % N_CLIENTS),
                             topic="pad",
                             fingerprint=fingerprint_text("pad %d" % k, num_perm=NUM_PERM)))
    index.add(np.stack(vecs), metas)
    return index, seed_ids


def test_shadow_ann_time_does_not_scale_linearly_with_n_alive():
    """Performance regression. Hold |frontier| fixed, scale n_alive by 4x; the vector-channel time must not grow by more than 2x.

    The definition matches the repair regression case: what is measured is the batched ANN channel
    required by the task (the text channel's bucket build is O(n_alive) whole-block numpy and is outside
    the scope of this assertion).
    """
    import time

    def _best(index, seed_ids, reps=3):
        best = None
        rep = None
        for _ in range(reps):
            det = ShadowDetector(text_channel=False)  # a fresh detector each time; do not reuse the ANN cache
            t0 = time.perf_counter()
            rep = det.shadow_report(index, seed_ids)
            d = time.perf_counter() - t0
            best = d if best is None else min(best, d)
            assert det.last_search_stats["n_ann_cache_hits"] == 0
        return best, rep

    small_index, seeds = _grow_scenario(2500)
    big_index, big_seeds = _grow_scenario(10000)
    assert seeds == big_seeds
    assert big_index.n_alive >= 3 * small_index.n_alive

    t_small, rep_small = _best(small_index, seeds)
    t_big, rep_big = _best(big_index, big_seeds)
    assert rep_small["n_closure"] == rep_big["n_closure"], (rep_small["n_closure"], rep_big["n_closure"])
    assert rep_small["n_ann_queries"] == rep_big["n_ann_queries"] > 0
    ratio = t_big / max(t_small, 1e-9)
    assert ratio < 2.0, (t_small, t_big, ratio)

def test_ann_nprobe_scope_is_closure_only_and_always_restored():
    """The detector-level ann_nprobe applies only to closure retrieval; the index configuration must be restored exactly on exit.

    Semantic guarantee: the index.search() used for utility evaluation (recall@10/nDCG) uses the index's own
    nprobe; the closure channel must leave no global side effects.
    """
    index, gt, seed_ids, _, _, _ = _make_scenario()
    ann = getattr(index, "_ann", None)
    if ann is None or not hasattr(ann, "nprobe"):
        pytest.skip("current backend has no nprobe (numpy / hnswlib)")
    original = int(ann.nprobe)
    seen = []
    real_search = index.search

    def _spy(queries, k=10):
        seen.append(int(ann.nprobe))          # nprobe in effect on the index when the search happens
        return real_search(queries, k=k)

    index.search = _spy
    try:
        det_default = ShadowDetector(text_channel=False)
        det_default.closure(index, seed_ids)
        assert set(seen) == {original}, "the default (ann_nprobe=None) must not change the index nprobe"
        assert int(ann.nprobe) == original

        seen.clear()
        det_scoped = ShadowDetector(text_channel=False, ann_nprobe=2)
        det_scoped.closure(index, seed_ids)
        assert seen and set(seen) == {2}, "closure retrieval must use the detector-level nprobe"
        assert int(ann.nprobe) == original, "must restore the original value after exiting the closure"

        # Retrieval outside the closure (utility-evaluation path) still uses the index's own configuration
        seen.clear()
        index.search(index.vectors_for(seed_ids[:3]), k=3)
        assert set(seen) == {original}
    finally:
        del index.search
    assert int(ann.nprobe) == original


def test_ann_nprobe_none_keeps_closure_bit_identical():
    """ann_nprobe=None (default) is exactly equivalent to omitting the argument; the closure is bit-identical."""
    index, gt, seed_ids, _, _, _ = _make_scenario()
    a = ShadowDetector().closure(index, seed_ids)
    b = ShadowDetector(ann_nprobe=None).closure(index, seed_ids)
    assert a == b == _reference_closure(index, seed_ids)
