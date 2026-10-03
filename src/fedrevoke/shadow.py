"""fedrevoke.shadow — ShadowDetector: cascading erasure closure for cross-silo semantic shadows.

Contract (INTERFACES.md Section 3, signatures must not change):

    class ShadowDetector:
        __init__(sim_threshold=0.92, lsh_threshold=0.80, knn_k=50,
                 cross_client_only=True, seed=20260214)
        closure(index: ProvenanceIndex, seed_ids: Sequence[int]) -> set[int]
        shadow_report(index, seed_ids) -> dict
        # {"n_seed","n_closure","per_client":{...},"sim_hist":[...],"precision":float,"recall":float}

Scientific positioning
---------------------------
Naive shard deletion only removes the hit shards in the local silo; **semantic shadows**
contributed by other silos (near-duplicate rewrites, translations, concatenations) still
make the revoked knowledge retrievable. This module computes the "internal ids that must
be deleted together" as an **erasure closure**:

    Channel A: vector kNN graph (neighbor edges with cosine similarity >= sim_threshold)
    Channel B: MinHash + LSH text fingerprints (near-duplicate edges with Jaccard >= lsh_threshold)

Both channels expand hop by hop along the frontier (cascade), and every hop is subject to
cross_client_only:
    cross_client_only=True  -> only **cross-silo** edges are allowed (same-silo near-duplicates are covered by regular deletion via ids_for_client)
    cross_client_only=False -> same-silo near-duplicate edges are also included in the closure (for upper-bound reference / ablation)

Termination guarantee: visited (here the closure set itself) + max_iterations (default 32) + optional max_closure.
The closure is idempotent and monotone for the same input (it does not change with repeated calls).

Fingerprint / signature sources (two routes, by priority)
--------------------------------
1. **signatures= provider** (recommended for real data pipelines):
   ShadowDetector(signatures=sig_matrix), where sig_matrix can be
     - np.ndarray with shape (n_chunks, num_perm), row-indexed by VecMeta.pid (= the
       data/processed/<ds>/minhash_sig.npy written by data_prep, row order aligned with the pid in corpus.jsonl);
     - Mapping keyed by one of pid / doc_id / fingerprint;
     - callable(meta) -> signature (1-D ndarray / bytes / hex str).
   You can load it directly with load_signature_matrix("data/processed/<ds>").
2. **VecMeta.fingerprint**: if it is the hex of a full MinHash signature (>= min_sig_words 32-bit words,
   default 16), it is parsed directly; if it is a **short digest** (e.g. sha1/sha256, or data_prep's
   blake2b(signature, digest_size=8) 16-char hex), it degrades to
   "run 4-gram MinHash on the fingerprint string itself" — only exactly identical fingerprints
   (== exactly identical text) will be judged as shadows by the text channel. In that case
   shadow_report's fingerprint_mode faithfully reports {"digest": N}, indicating that the text
   channel has degraded to exact-match semantics and route 1 should be used instead.

This file provides fingerprint_text() for implementations that write full-signature hex fingerprints.
"""

from __future__ import annotations

import hashlib
import re
import warnings
from collections import Counter
from contextlib import contextmanager
from functools import lru_cache
from time import perf_counter
from typing import Any, Mapping, Sequence

import numpy as np

from .index_core import PAD_ID, ProvenanceIndex, VecMeta

try:  # project-wide unified random seed (config.py; fall back to the contract constant if missing)
    from .config import SEED as _CONFIG_SEED  # type: ignore
except Exception:  # pragma: no cover - fallback when config is not in place yet
    _CONFIG_SEED = 20260214

SEED: int = int(_CONFIG_SEED)

DEFAULT_NUM_PERM = 64
MIN_SIG_WORDS = 16  # a "signature" with fewer than 16 32-bit words is treated as a digest (digest fallback path)
_UINT32_MAX = int(np.iinfo(np.uint32).max)
_WORD_RE = re.compile(r"[0-9a-z]+")

__all__ = [
    "ShadowDetector",
    "minhash_signature",
    "fingerprint_text",
    "signature_from_hex",
    "minhash_jaccard",
    "optimal_bands",
    "token_shingles",
    "exact_shingle_jaccard",
    "load_signature_matrix",
    "MIN_SIG_WORDS",
    "SEED",
]


def _as_1d_signature(val):
    """Normalize a 1-D ndarray / bytes / hex string / sequence into a 1-D ndarray; return None on failure."""
    if val is None:
        return None
    if isinstance(val, np.ndarray):
        a = np.ascontiguousarray(val)
        return a.reshape(-1) if a.ndim != 1 else a
    if isinstance(val, (bytes, bytearray, memoryview)):
        raw = bytes(val)
        if len(raw) % 4 == 0 and len(raw) >= 4:
            return np.frombuffer(raw, dtype="<u4").astype(np.uint32)
        return None
    if isinstance(val, str):
        return signature_from_hex(val)
    try:
        return np.asarray(val).reshape(-1)
    except Exception:
        return None


def load_signature_matrix(path, filename: str = "minhash_sig.npy"):
    """Load the MinHash signature matrix written by data_prep (row order aligned with the pid in corpus.jsonl).

    path can be the data/processed/<ds> directory, or the .npy file itself.
    """
    import os

    p = str(path)
    if os.path.isdir(p):
        p = os.path.join(p, filename)
    arr = np.load(p, allow_pickle=False)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    return np.ascontiguousarray(arr)


# --------------------------------------------------------------------------------------
# MinHash / LSH basics (no third-party dependencies, only numpy + hashlib)
# --------------------------------------------------------------------------------------
def token_shingles(text: str, shingle_size: int = 4, word_level: bool = True) -> list:
    """Split text into shingles: by default 4-grams of lowercase words; character 4-grams when word_level=False."""
    if not text:
        return []
    size = max(1, int(shingle_size))
    if word_level:
        toks = _WORD_RE.findall(text.lower())
        if len(toks) <= size:
            return [" ".join(toks)] if toks else []
        return [" ".join(toks[i : i + size]) for i in range(len(toks) - size + 1)]
    s = text.lower()
    if len(s) <= size:
        return [s]
    return [s[i : i + size] for i in range(len(s) - size + 1)]


def exact_shingle_jaccard(text_a: str, text_b: str, shingle_size: int = 4, word_level: bool = True) -> float:
    """True shingle Jaccard (for evaluating MinHash estimation error), O(n)."""
    a = set(token_shingles(text_a, shingle_size, word_level))
    b = set(token_shingles(text_b, shingle_size, word_level))
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / float(len(a | b))


@lru_cache(maxsize=64)
def _permutations(num_perm: int, seed: int):
    """(a, b) permutation parameters; a is odd so h -> a*h+b remains a bijection under uint64 wraparound (a valid MinHash family)."""
    rng = np.random.default_rng(int(seed) ^ 0x5EED5EED)
    a = rng.integers(1, 1 << 62, size=int(num_perm), dtype=np.uint64) | np.uint64(1)
    b = rng.integers(0, 1 << 62, size=int(num_perm), dtype=np.uint64)
    return a, b


def _shingle_hashes(shingles: Sequence[str]) -> np.ndarray:
    """Stable 64-bit hashes (blake2b, consistent across processes/platforms, not Python's built-in hash)."""
    if not shingles:
        return np.zeros(0, dtype=np.uint64)
    out = np.empty(len(shingles), dtype=np.uint64)
    for i, s in enumerate(shingles):
        out[i] = int.from_bytes(hashlib.blake2b(s.encode("utf-8"), digest_size=8).digest(), "little")
    return out


def minhash_signature(
    text: str,
    num_perm: int = DEFAULT_NUM_PERM,
    seed: int = SEED,
    shingle_size: int = 4,
    word_level: bool = True,
) -> bytes:
    """Return the MinHash signature (num_perm uint32 little-endian bytes). Empty text -> all 1s (by convention)."""
    num_perm = max(1, int(num_perm))
    hashes = _shingle_hashes(token_shingles(text, shingle_size, word_level))
    if hashes.size == 0:
        return np.full(num_perm, _UINT32_MAX, dtype=np.uint32).tobytes()
    a, b = _permutations(num_perm, int(seed))
    best = np.full(num_perm, np.iinfo(np.uint64).max, dtype=np.uint64)
    step = 4096
    for start in range(0, hashes.size, step):
        chunk = hashes[start : start + step]
        perm = a[:, None] * chunk[None, :] + b[:, None]  # uint64 wraparound; a odd => bijection
        np.minimum(best, perm.min(axis=1), out=best)
    return (best >> np.uint64(32)).astype(np.uint32).tobytes()


def fingerprint_text(text: str, num_perm: int = DEFAULT_NUM_PERM, seed: int = SEED,
                     shingle_size: int = 4, word_level: bool = True) -> str:
    """For data_prep: convert text into a hex string that can be written directly to VecMeta.fingerprint."""
    return minhash_signature(text, num_perm, seed, shingle_size, word_level).hex()


def signature_from_hex(fp: str) -> np.ndarray | None:
    """Parse a hex fingerprint into a uint32 signature array; return None if it is not a signature."""
    s = (fp or "").strip()
    if len(s) < 16 or len(s) % 2:
        return None
    try:
        raw = bytes.fromhex(s)
    except ValueError:
        return None
    if len(raw) % 4 or len(raw) < 8:
        return None
    return np.frombuffer(raw, dtype="<u4").astype(np.uint32)


def minhash_jaccard(sig_a: np.ndarray, sig_b: np.ndarray) -> float:
    """MinHash Jaccard estimate; different lengths (not comparable) or empty -> 0.0."""
    if sig_a is None or sig_b is None:
        return 0.0
    a = np.asarray(sig_a)
    b = np.asarray(sig_b)
    if a.size == 0 or a.size != b.size:
        return 0.0
    return float(np.mean(a == b))


@lru_cache(maxsize=512)
def optimal_bands(num_perm: int, threshold: float):
    """Choose (bands, rows) so the LSH S-curve P(s)=1-(1-s^r)^b best approximates the threshold step function.

    The objective is the mean squared error over 101 grid points on [0,1]; b*r <= num_perm.
    """
    num_perm = int(num_perm)
    threshold = float(threshold)
    if num_perm < 2:
        return (1, 1)
    grid = np.linspace(0.0, 1.0, 101)
    target = (grid >= threshold).astype(np.float64)
    best_err, best_b, best_r = None, 1, 1
    for r in range(1, num_perm + 1):
        b = num_perm // r
        if b < 1:
            break
        curve = 1.0 - np.power(1.0 - np.power(grid, r), b)
        err = float(np.mean((curve - target) ** 2))
        if best_err is None or err < best_err - 1e-12:
            best_err, best_b, best_r = err, b, r
    return (best_b, best_r)


# --------------------------------------------------------------------------------------
# Channel B vectorized primitives: band-key hashing
# --------------------------------------------------------------------------------------
# Bucket membership is **always** decided by exact byte-wise comparison of band keys;
# the 64-bit hashes below are only used to sort buckets, turning "which rows does a band
# key fall into" into numpy argsort + searchsorted. Hash collisions only widen the binary
# search interval and add a few exact byte comparisons; they never change any closure
# result (see _lsh_neighbors).
_BAND_HASH_SEED = 0x243F6A8885A308D3
_FNV_PRIME = np.uint64(0x100000001B3)
_FNV_SHIFT = np.uint64(29)

# Upper bound on ANN neighbor cache entries (invalidate the whole cache beyond it, to avoid unbounded memory growth in long-running processes)
DEFAULT_ANN_CACHE_MAX = 200_000


def _band_hashes(words: np.ndarray) -> np.ndarray:
    """Mix an (n, r) band-word matrix into (n,) uint64 hashes (FNV-1a + xor-shift finalizer).

    Vectorized column by column, computing all n rows at once; r is much smaller than the
    signature length, so the whole block costs O(n x r) numpy operations.
    """
    if words.ndim != 2 or words.shape[1] == 0:
        return np.zeros(words.shape[0], dtype=np.uint64)
    h = np.full(words.shape[0], _BAND_HASH_SEED, dtype=np.uint64)
    for j in range(int(words.shape[1])):
        h ^= words[:, j].astype(np.uint64, copy=False)
        h *= _FNV_PRIME
        h ^= h >> _FNV_SHIFT
    return h


def _key_bytes(mat: np.ndarray, start: int, stop: int) -> np.ndarray:
    """Take columns [start:stop) of an (n, L) signature matrix as an (n, (stop-start)*itemsize) uint8 key."""
    band = np.ascontiguousarray(mat[:, start:stop])
    return band.view(np.uint8).reshape(band.shape[0], -1)


def _normalize_gt(gt: Any):
    """Normalize various ground-truth forms (set / list / {client: [...]} / {"injected": [...]}) into set[int]."""
    if gt is None:
        return None
    if isinstance(gt, Mapping):
        items = []
        for v in gt.values():
            if isinstance(v, (set, list, tuple, np.ndarray)):
                items.extend(list(v))
            else:
                items.append(v)
        gt = items
    out = set()
    for v in gt:
        if isinstance(v, (int, np.integer)):
            out.add(int(v))
        elif isinstance(v, (set, list, tuple, np.ndarray)):
            for x in v:
                if isinstance(x, Mapping):
                    for k in ("internal_id", "id", "index"):
                        if k in x:
                            out.add(int(x[k]))
                else:
                    out.add(int(x))
        elif isinstance(v, Mapping):
            for k in ("internal_id", "id", "index"):
                if k in v:
                    out.add(int(v[k]))
    return out


# --------------------------------------------------------------------------------------
# ShadowDetector
# --------------------------------------------------------------------------------------
class ShadowDetector:
    """Cascading erasure-closure detector (vector kNN channel + MinHash/LSH text channel)."""

    def __init__(
        self,
        sim_threshold: float = 0.92,
        lsh_threshold: float = 0.80,
        knn_k: int = 50,
        cross_client_only: bool = True,
        seed: int = SEED,
        *,
        max_iterations: int = 32,
        max_closure: int | None = None,
        minhash_perm: int = DEFAULT_NUM_PERM,
        shingle_size: int = 4,
        word_level: bool = True,
        ground_truth: Any = None,
        vector_channel: bool = True,
        text_channel: bool = True,
        lsh_max_docs: int = 500_000,
        signatures: Any = None,
        min_sig_words: int = MIN_SIG_WORDS,
        ann_cache: bool = True,
        ann_cache_max: int = DEFAULT_ANN_CACHE_MAX,
        ann_nprobe: int | None = None,
        lsh_flat_limit: int = 20_000_000,
        verbose: bool = False,
    ) -> None:
        self.sim_threshold = float(sim_threshold)
        self.lsh_threshold = float(lsh_threshold)
        self.knn_k = int(knn_k)
        self.cross_client_only = bool(cross_client_only)
        self.seed = int(seed)
        self.max_iterations = max(1, int(max_iterations))
        self.max_closure = int(max_closure) if max_closure else None
        self.minhash_perm = max(8, int(minhash_perm))
        self.shingle_size = max(1, int(shingle_size))
        self.word_level = bool(word_level)
        self.ground_truth = _normalize_gt(ground_truth)
        self.vector_channel = bool(vector_channel)
        self.text_channel = bool(text_channel)
        self.lsh_max_docs = int(lsh_max_docs)
        self.signatures = signatures  # route 1: external signature provider (ndarray / Mapping / callable)
        self.min_sig_words = max(1, int(min_sig_words))
        self.ann_cache = bool(ann_cache)
        self.ann_cache_max = max(0, int(ann_cache_max))
        # Closure-retrieval-specific nprobe (detector-level parameter): None = follow the index's own config (default, bit-identical).
        # It only applies to the single batch index.search() inside _vector_neighbors and is restored on exit,
        # so retrieval used for utility evaluation (recall@10/nDCG) is completely unaffected.
        #
        # WARNING - empirical finding (2026-10-02 A/B over 2,284 cases):
        #   In a full A/B over 2,284 cases, nprobe=8 vs nprobe=16 produced different closure sets in 13 cases
        #   (and only misses, never extras), including ds2 r20 under the main configuration (3 ids lost)
        #   and all tau in {0.85, 0.90} levels of the E7 grid.
        #   -> Keep the default None (nprobe=16). **Do not** set it to 8 without redoing the full
        #   equivalence verification.
        self.ann_nprobe = int(ann_nprobe) if ann_nprobe else None
        self.lsh_flat_limit = max(1, int(lsh_flat_limit))
        self.verbose = bool(verbose)

        # Diagnostics from the most recent closure()
        self.last_edges = []  # type: list
        self.last_iterations = 0
        self.last_hit_max_iterations = False
        self.last_channel_counts = {}  # type: dict
        self.last_skipped_seeds = []  # type: list
        self.last_fingerprint_mode = {}  # type: dict
        self.last_timings = {}  # type: dict
        self.last_search_stats = {}  # type: dict

        self._band_cache = None  # type: dict | None
        self._ann_cache = None  # type: dict | None
        self._pid_cache = None  # type: tuple | None

    # ------------------------------------------------------------------ Public API
    def closure(self, index: ProvenanceIndex, seed_ids: Sequence[int]) -> set:
        """Return the "set of internal ids that must be deleted" (including the seeds themselves).

        Rules:
          * Only seeds that are still alive are accepted; deleted/out-of-range seeds are silently skipped (idempotent, safe to replay).
          * Both channels expand hop by hop; each hop checks cross_client_only.
          * Triple termination guarantee: visited + max_iterations + max_closure.
        """
        seeds = []
        seen = set()
        skipped = []
        for raw in seed_ids:
            try:
                i = int(raw)
            except (TypeError, ValueError):
                continue
            if i in seen:
                continue
            seen.add(i)
            if index.is_alive(i):
                seeds.append(i)
            else:
                skipped.append(i)
        self.last_skipped_seeds = skipped
        self.last_edges = []
        self.last_iterations = 0
        self.last_hit_max_iterations = False
        self.last_channel_counts = {"knn": 0, "lsh": 0}
        self.last_timings = {"prepare_s": 0.0, "band_build_s": 0.0, "neighbor_s": 0.0,
                             "ann_s": 0.0, "lsh_s": 0.0, "total_s": 0.0}
        self.last_search_stats = {"n_ann_queries": 0, "n_ann_cache_hits": 0, "n_rounds": 0,
                                  "n_frontier": 0, "n_lsh_candidates": 0, "n_lsh_edges": 0,
                                  "ann_query_batch_max": 0, "ann_cache_entries": 0}
        t_start = perf_counter()
        if not seeds:
            self.last_timings["total_s"] = perf_counter() - t_start
            return set()

        t0 = perf_counter()
        band_ctx = self._band_context(index) if self.text_channel else None
        self.last_timings["band_build_s"] = perf_counter() - t0
        closure = set(seeds)
        frontier = list(seeds)
        for it in range(self.max_iterations):
            if not frontier:
                break
            self.last_iterations = it + 1
            self.last_search_stats["n_rounds"] = it + 1
            self.last_search_stats["n_frontier"] += len(frontier)
            t0 = perf_counter()
            candidates = []
            if self.vector_channel:
                candidates.extend(self._vector_neighbors(index, frontier))
            if band_ctx is not None:
                candidates.extend(self._lsh_neighbors(index, frontier, band_ctx))
            self.last_timings["neighbor_s"] += perf_counter() - t0

            added = []
            for src, dst, sim, channel in candidates:
                if dst in closure:
                    continue
                if self.cross_client_only:
                    if index.meta(dst).client_id == index.meta(src).client_id:
                        continue
                closure.add(dst)
                added.append(dst)
                self.last_edges.append((int(src), int(dst), float(sim), channel))
                self.last_channel_counts[channel] = self.last_channel_counts.get(channel, 0) + 1
                if self.max_closure is not None and len(closure) >= self.max_closure:
                    break
            frontier = added
            if self.max_closure is not None and len(closure) >= self.max_closure:
                break
            if not added:
                break
        else:
            self.last_hit_max_iterations = True

        self.last_search_stats["ann_cache_entries"] = (
            len(self._ann_cache.get("rows", {})) if isinstance(self._ann_cache, dict) else 0
        )
        self.last_timings["total_s"] = perf_counter() - t_start
        if self.verbose:
            print(
                "[shadow] seeds=%d closure=%d iters=%d knn=%d lsh=%d"
                % (
                    len(seeds),
                    len(closure),
                    self.last_iterations,
                    self.last_channel_counts.get("knn", 0),
                    self.last_channel_counts.get("lsh", 0),
                )
            )
        return closure

    def shadow_report(self, index: ProvenanceIndex, seed_ids: Sequence[int], gt_ids: Any = None) -> dict:
        """Contract fields: n_seed / n_closure / per_client / sim_hist / precision / recall.

        precision/recall are only defined when ground truth is given; otherwise they are NaN and
        has_ground_truth=False. Ground-truth resolution order:
        gt_ids argument > self.ground_truth > index.shadow_ground_truth > index.ground_truth.
        The ground-truth semantics is "the full set of internal ids that must be deleted"
        (including the seeds themselves and their true shadow copies).
        """
        closure = self.closure(index, seed_ids)
        seeds_alive = [int(i) for i in dict.fromkeys(int(s) for s in seed_ids) if index.is_alive(int(i))]

        per_client = Counter(index.meta(i).client_id for i in closure)
        per_client_seed = Counter(index.meta(i).client_id for i in seeds_alive)

        sims = [e[2] for e in self.last_edges]
        if sims:
            hist, edges = np.histogram(np.asarray(sims, dtype=np.float64), bins=10, range=(0.0, 1.0))
            sim_stats = {
                "sim_mean": float(np.mean(sims)),
                "sim_min": float(np.min(sims)),
                "sim_max": float(np.max(sims)),
                "n_edges": int(len(sims)),
            }
        else:
            hist = np.zeros(10, dtype=np.int64)
            edges = np.linspace(0.0, 1.0, 11)
            sim_stats = {"sim_mean": float("nan"), "sim_min": float("nan"), "sim_max": float("nan"), "n_edges": 0}

        gt = _normalize_gt(gt_ids)
        if gt is None:
            gt = self.ground_truth
        if gt is None:
            gt = _normalize_gt(getattr(index, "shadow_ground_truth", None))
        if gt is None:
            gt = _normalize_gt(getattr(index, "ground_truth", None))

        if gt is None:
            precision = recall = float("nan")
            tp = fp = fn = 0
            p_excl = r_excl = float("nan")
        else:
            tp = len(closure & gt)
            fp = len(closure - gt)
            fn = len(gt - closure)
            precision = tp / float(tp + fp) if (tp + fp) else 1.0
            recall = tp / float(tp + fn) if (tp + fn) else 1.0
            seed_set = set(seeds_alive)
            c2, g2 = closure - seed_set, gt - seed_set
            tp2 = len(c2 & g2)
            p_excl = tp2 / float(len(c2)) if c2 else 1.0
            r_excl = tp2 / float(len(g2)) if g2 else 1.0

        report = {
            # --- Contract fields ---
            "n_seed": len(seeds_alive),
            "n_closure": len(closure),
            "per_client": {k: int(v) for k, v in sorted(per_client.items())},
            "sim_hist": [int(x) for x in hist],
            "precision": float(precision),
            "recall": float(recall),
            # --- Extra diagnostic fields ---
            "has_ground_truth": gt is not None,
            "n_ground_truth": len(gt) if gt is not None else 0,
            "tp": int(tp),
            "fp": int(fp),
            "fn": int(fn),
            "precision_excl_seed": float(p_excl),
            "recall_excl_seed": float(r_excl),
            "per_client_seed": {k: int(v) for k, v in sorted(per_client_seed.items())},
            "n_clients_touched": len(per_client),
            "sim_hist_edges": [float(x) for x in edges],
            "sim_stats": sim_stats,
            "iterations": int(self.last_iterations),
            "hit_max_iterations": bool(self.last_hit_max_iterations),
            "channel_counts": dict(self.last_channel_counts),
            "fingerprint_mode": dict(self.last_fingerprint_mode),
            "n_seed_skipped": len(self.last_skipped_seeds),
            "closure_ids": sorted(int(i) for i in closure),
            # --- M2 phase diagnostics: locate time spent inside the closure and batch-search sizes ---
            "n_rounds": int(self.last_search_stats.get("n_rounds", self.last_iterations)),
            "n_frontier": int(self.last_search_stats.get("n_frontier", 0)),
            "n_ann_queries": int(self.last_search_stats.get("n_ann_queries", 0)),
            "n_ann_cache_hits": int(self.last_search_stats.get("n_ann_cache_hits", 0)),
            "ann_cache_entries": int(self.last_search_stats.get("ann_cache_entries", 0)),
            "ann_query_batch_max": int(self.last_search_stats.get("ann_query_batch_max", 0)),
            "n_lsh_candidates": int(self.last_search_stats.get("n_lsh_candidates", 0)),
            "m2_total_s": float(self.last_timings.get("total_s", 0.0)),
            "m2_prepare_s": float(self.last_timings.get("prepare_s", 0.0)),
            "m2_band_build_s": float(self.last_timings.get("band_build_s", 0.0)),
            "m2_neighbor_s": float(self.last_timings.get("neighbor_s", 0.0)),
            "m2_ann_s": float(self.last_timings.get("ann_s", 0.0)),
            "m2_lsh_s": float(self.last_timings.get("lsh_s", 0.0)),
        }
        return report

    def shadow_pairs(self, index: ProvenanceIndex) -> list:
        """Shadow edges hit by the most recent closure(): [(src, dst, sim, channel), ...]."""
        return list(self.last_edges)

    # ------------------------------------------------------------------ Channel A: vector kNN
    @contextmanager
    def _closure_nprobe(self, index: ProvenanceIndex):
        """Scope for the closure-retrieval-specific faiss nprobe (the original value is unconditionally restored on exit).

        * The scope covers **only** the single batch index.search() in the closure vector channel;
        * By default ann_nprobe=None -> the index configuration is left completely untouched (bit-identical to pre-optimization behavior);
        * It is a no-op when the backend has no nprobe (numpy / hnswlib).
        This is a detector-level parameter: it never changes global faiss state, nor the retrieval
        behavior of other callers.
        """
        ann = getattr(index, "_ann", None)
        if self.ann_nprobe is None or ann is None or not hasattr(ann, "nprobe"):
            yield
            return
        old = int(ann.nprobe)
        try:
            ann.nprobe = int(self.ann_nprobe)
            yield
        finally:
            ann.nprobe = old

    def _ann_store(self, index: ProvenanceIndex) -> dict:
        """Maintain the ANN neighbor cache by index state (repeated pivots under the same state are not re-queried).

        The key includes the effective nprobe, so switching ann_nprobe on the same detector cannot
        read stale results.
        """
        key = (self._cache_key(index), self.ann_nprobe)
        cache = self._ann_cache
        if cache is None or cache.get("key") != key:
            cache = {"key": key, "rows": {}}
            self._ann_cache = cache
        return cache

    def _vector_neighbors(self, index: ProvenanceIndex, frontier: Sequence[int]) -> list:
        """Channel A: the whole frontier is concatenated into one query matrix, calling index.search() **exactly once per hop**.

        Batching is done by faiss in C++ (including OpenMP parallelism); the Python side only does
        O(|frontier| x k) vectorized edge extraction and never constructs an explicit
        (|frontier| x n_alive) distance matrix.
        """
        out = []
        if not frontier:
            return out
        n_alive = index.n_alive
        if n_alive <= 1:
            return out
        k = int(max(1, min(self.knn_k, n_alive)))
        stats = self.last_search_stats
        fr = [int(i) for i in frontier]
        use_cache = bool(self.ann_cache) and self.ann_cache_max > 0
        cache = self._ann_store(index) if use_cache else None
        rows = cache["rows"] if cache is not None else {}
        if use_cache:
            todo = [i for i in fr if i not in rows]
        else:
            todo = fr
        hits = len(fr) - len(todo)
        stats["n_ann_cache_hits"] = stats.get("n_ann_cache_hits", 0) + hits

        fresh_ids = fresh_sc = None
        if todo:
            vecs = index.vectors_for(todo)
            t_ann = perf_counter()
            with self._closure_nprobe(index):   # takes effect only here, and always restores the original value
                fresh_sc, fresh_ids = index.search(vecs, k=k)
            self.last_timings["ann_s"] = self.last_timings.get("ann_s", 0.0) + (perf_counter() - t_ann)
            stats["n_ann_queries"] = stats.get("n_ann_queries", 0) + len(todo)
            stats["ann_query_batch_max"] = max(stats.get("ann_query_batch_max", 0), len(todo))
            fresh_ids = np.asarray(fresh_ids, dtype=np.int64)
            fresh_sc = np.asarray(fresh_sc, dtype=np.float32)
            if cache is not None:
                if len(rows) + len(todo) > self.ann_cache_max:
                    rows.clear()
                for a, pid in enumerate(todo):
                    rows[pid] = (fresh_ids[a], fresh_sc[a])

        if hits == 0 and todo:
            ids_mat, sc_mat = fresh_ids, fresh_sc
        else:
            width = fresh_ids.shape[1] if fresh_ids is not None else int(len(rows[fr[0]][0]))
            ids_mat = np.empty((len(fr), width), dtype=np.int64)
            sc_mat = np.empty((len(fr), width), dtype=np.float32)
            for r, pid in enumerate(fr):
                cached = rows[pid]
                ids_mat[r] = cached[0]
                sc_mat[r] = cached[1]

        src_col = np.asarray(fr, dtype=np.int64).reshape(-1, 1)
        keep = (ids_mat != PAD_ID) & (ids_mat != src_col) & ~(sc_mat < self.sim_threshold)
        rr, cc = np.nonzero(keep)
        if rr.size == 0:
            return out
        srcs = np.asarray(fr, dtype=np.int64)[rr]
        dsts = ids_mat[rr, cc]
        sims = sc_mat[rr, cc].astype(np.float64)
        return [(int(s), int(d), float(v), "knn")
                for s, d, v in zip(srcs.tolist(), dsts.tolist(), sims.tolist())]

    # ------------------------------------------------------------------ Channel B: MinHash / LSH
    def _provider_signature(self, meta: VecMeta):
        """Route 1: fetch the signature from the signatures= provider."""
        src = self.signatures
        if src is None:
            return None
        val = None
        if callable(src):
            val = src(meta)
        elif isinstance(src, Mapping):
            for key in (meta.pid, meta.doc_id, meta.fingerprint):
                try:
                    if key in src:
                        val = src[key]
                        break
                except TypeError:
                    continue
        else:
            try:
                val = src[meta.pid]
            except Exception:
                val = None
        return _as_1d_signature(val)

    def _resolve_signature(self, meta: VecMeta):
        """Return (signature | None, mode), where mode is one of {"provided","signature","digest","missing"}."""
        sig = self._provider_signature(meta)
        if sig is not None and sig.size:
            return sig, "provided"
        fp = (meta.fingerprint or "").strip()
        if not fp:
            return None, "missing"
        sig = signature_from_hex(fp)
        if sig is not None and sig.size >= self.min_sig_words:
            return sig, "signature"
        # Fallback path: the fingerprint is a short digest (sha1/sha256, or data_prep's blake2b 8-byte digest);
        # run 4-gram MinHash on the fingerprint string itself — only exactly identical text is judged a shadow.
        return (
            np.frombuffer(
                minhash_signature(
                    fp,
                    num_perm=self.minhash_perm,
                    seed=self.seed,
                    shingle_size=4,
                    word_level=False,
                ),
                dtype="<u4",
            ).astype(np.uint32),
            "digest",
        )

    def _meta_signature(self, meta: VecMeta) -> np.ndarray | None:
        return self._resolve_signature(meta)[0]

    def diagnose_fingerprints(self, index: ProvenanceIndex) -> dict:
        """Count signature sources of alive documents for data-pipeline self-checks (does not build the full bucket index)."""
        alive = index.alive_ids()
        blocks, modes, row_mode = self._signature_blocks(index, alive)
        counts = Counter(modes)
        if row_mode is None:
            # Fast path (signatures is an ndarray keyed by pid rows): all "provided"
            lens = [int(blocks[0][0])] * int(alive.size) if blocks else []
        else:
            lens = [int(mat.shape[1]) for (L, iids, mat) in blocks
                    for i in iids.tolist() if row_mode.get(int(i)) in ("provided", "signature")]
        hint = ""
        if counts.get("digest"):
            hint = (
                "%d/%d fingerprints are short digests: the text channel degrades to exact-match semantics. "
                "For real data use ShadowDetector(signatures=load_signature_matrix(data/processed/<ds>)) "
                "to pass in the full signature matrix minhash_sig.npy written by data_prep."
                % (counts["digest"], sum(counts.values()))
            )
        return {
            "counts": {k: int(v) for k, v in counts.items()},
            "n_alive": int(index.n_alive),
            "median_sig_len": int(np.median(lens)) if lens else 0,
            "hint": hint,
        }

    def _bands_for(self, sig_len: int):
        return optimal_bands(int(sig_len), float(self.lsh_threshold))

    def _cache_key(self, index: ProvenanceIndex):
        """Index-state fingerprint: any change that could alter retrieval results changes the key.

        Note: index.stats() is no longer called (it walks every meta to count bytes, about 0.09 s per call at DS2 scale).
        """
        return (
            id(index),
            int(index.n_vectors),
            int(index.n_alive),
            int(getattr(index, "_n_compactions", 0) or 0),
        )

    # ------------------------------------------------------------ Channel B construction (vectorized)
    def _alive_pids(self, index: ProvenanceIndex, alive: np.ndarray) -> np.ndarray:
        """VecMeta.pid values of alive ids (ascending); the full table is cached by (id(index), n_vectors)."""
        ck = (id(index), int(index.n_vectors))
        cached = self._pid_cache
        if cached is not None and cached[0] == ck:
            pids = cached[1]
        else:
            metas = index.metas_snapshot()
            pids = np.fromiter((int(m.pid) for m in metas), dtype=np.int64, count=len(metas))
            self._pid_cache = (ck, pids)
        return pids[alive]

    def _signature_blocks(self, index: ProvenanceIndex, alive: np.ndarray):
        """Resolve the MinHash signatures of alive ids into [(L, iids, mat)] blocks + mode counts.

        Fast path (the only path for real data pipelines): signatures is an (n_rows, L) ndarray keyed
        by VecMeta.pid rows — one fancy-index fetches the whole signature matrix block, replacing N
        per-id calls. Other cases (Mapping / callable / pid out of range) fall back per id to
        _resolve_signature, with mode decisions bit-identical to the old implementation.
        """
        modes = Counter()
        n = int(alive.size)
        if n == 0:
            return [], modes, None
        src = self.signatures
        fast = isinstance(src, np.ndarray) and src.ndim == 2 and int(src.shape[1]) >= 1
        if fast:
            try:
                pids = self._alive_pids(index, alive)
                fast = bool(pids.size) and int(pids.min()) >= 0 and int(pids.max()) < int(src.shape[0])
            except Exception:
                fast = False
        if fast:
            mat = np.ascontiguousarray(np.asarray(src)[pids])
            modes["provided"] += n
            return [(int(mat.shape[1]), np.asarray(alive, dtype=np.int64), mat)], modes, None

        groups = {}
        row_mode = {}
        for iid in np.asarray(alive, dtype=np.int64).tolist():
            sig, mode = self._resolve_signature(index.meta(iid))
            modes[mode] += 1
            row_mode[iid] = mode
            if sig is None or sig.size == 0:
                continue
            groups.setdefault(int(sig.size), []).append((iid, sig))
        blocks = []
        for L in sorted(groups):
            rows = groups[L]
            mat = np.empty((len(rows), L), dtype=np.asarray(rows[0][1]).dtype)
            for a, (_, sig) in enumerate(rows):
                mat[a] = sig
            blocks.append((L, np.asarray([r0[0] for r0 in rows], dtype=np.int64), mat))
        return blocks, modes, row_mode

    def _band_context(self, index: ProvenanceIndex):
        """Build (and cache) the corpus-wide MinHash/LSH bucket index (vectorized).

        ctx["blocks"][bi] = {"L","iids","mat","r","b","bands":[(hs, ks, order), ...]}:
          * iids ascending, mat is the (m, L) signature matrix in the same order (row i == iids[i]);
          * bands[j] is the bucket sort of band j: hs = 64-bit hashes of band keys (ascending),
            ks = key bytes in the same order (m, 4r) uint8, order = row indices in the same order.
        Bucket lookup = searchsorted(hs) + exact byte-wise comparison of ks; hash collisions only
        widen the binary-search interval and never change the membership set (the semantics of the
        old dict[(j, r, bytes)] implementation is fully preserved).
        """
        key = self._cache_key(index)
        if self._band_cache is not None and self._band_cache["key"] == key:
            return self._band_cache

        t0 = perf_counter()
        alive = index.alive_ids()
        n_total = int(index.n_vectors)
        if alive.size > self.lsh_max_docs:
            warnings.warn(
                "Number of alive documents %d exceeds lsh_max_docs=%d; skipping the text LSH channel this time (vector channel only)."
                % (alive.size, self.lsh_max_docs),
                RuntimeWarning,
                stacklevel=2,
            )
            ctx = {"key": key, "disabled": True, "modes": {}, "blocks": [],
                   "block_id": np.full(n_total, -1, dtype=np.int64),
                   "row_of_iid": np.full(n_total, -1, dtype=np.int64),
                   "n_alive": int(alive.size), "build_s": perf_counter() - t0}
            self._band_cache = ctx
            self.last_fingerprint_mode = {}
            return ctx

        blocks, modes, _row_mode = self._signature_blocks(index, alive)
        block_id = np.full(n_total, -1, dtype=np.int64)
        row_of_iid = np.full(n_total, -1, dtype=np.int64)
        prepared = []
        for L, iids, mat in blocks:
            if iids.size == 0:
                continue
            b, r = self._bands_for(int(L))
            bands = []
            for j in range(b):
                words = np.ascontiguousarray(mat[:, j * r : (j + 1) * r])
                kb = words.view(np.uint8).reshape(words.shape[0], -1)
                h = _band_hashes(words)
                order = np.argsort(h, kind="stable")
                bands.append((h[order], np.ascontiguousarray(kb[order]), order))
            bi = len(prepared)
            prepared.append({"L": int(L), "iids": iids, "mat": mat, "r": int(r), "b": int(b),
                             "bands": bands})
            block_id[iids] = bi
            row_of_iid[iids] = np.arange(iids.size, dtype=np.int64)
        ctx = {"key": key, "disabled": False, "modes": {k: int(v) for k, v in modes.items()},
               "blocks": prepared, "block_id": block_id, "row_of_iid": row_of_iid,
               "n_alive": int(alive.size), "build_s": perf_counter() - t0}
        self._band_cache = ctx
        self.last_fingerprint_mode = ctx["modes"]
        return ctx

    # ------------------------------------------------------------ Channel B bucket lookup (batch)
    @staticmethod
    def _hash_members(blk, j, lo, hi, qkey):
        """Get members byte-equal to qkey within the hash interval [lo, hi) (list of internal ids in ascending order)."""
        _, ks, order = blk["bands"][j]
        if hi <= lo:
            return []
        sel = np.all(ks[lo:hi] == qkey[None, :], axis=1)
        if not sel.any():
            return []
        return blk["iids"][order[lo:hi][sel]].tolist()

    def _accumulate_band_candidates(self, blk, pairs, cands) -> None:
        """Accumulate same-bucket members of (src, mat_row) across all bands into cands[src].

        Insertion order matches the old implementation (bands ascending; within a band, members by
        ascending internal id), so downstream set iteration order and last_edges order stay consistent.
        """
        if not pairs:
            return
        mat = blk["mat"]
        iids = blk["iids"]
        r = int(blk["r"])
        srcs = [p[0] for p in pairs]
        rows = np.asarray([p[1] for p in pairs], dtype=np.int64)
        for j in range(int(blk["b"])):
            hs, ks, order = blk["bands"][j]
            words = np.ascontiguousarray(mat[rows, j * r : (j + 1) * r])
            kb = words.view(np.uint8).reshape(words.shape[0], -1)
            h = _band_hashes(words)
            lo = np.searchsorted(hs, h, "left")
            hi = np.searchsorted(hs, h, "right")
            cnt = hi - lo
            total = int(cnt.sum())
            if total <= 0:
                continue
            if total > self.lsh_flat_limit:  # degradation guard: handle per src exactly when a bucket is abnormally huge
                for a, s in enumerate(srcs):
                    cands[s].update(self._hash_members(blk, j, int(lo[a]), int(hi[a]), kb[a]))
                continue
            cnt64 = cnt.astype(np.int64)
            starts = np.repeat(np.cumsum(cnt64) - cnt64, cnt64)
            flat = np.arange(total, dtype=np.int64) - starts + np.repeat(lo, cnt64)
            qidx = np.repeat(np.arange(len(srcs), dtype=np.int64), cnt64)
            hit = np.all(ks[flat] == np.repeat(kb, cnt64, axis=0), axis=1)
            if not hit.any():
                continue
            members = iids[order[flat[hit]]].tolist()
            for a, m in zip(qidx[hit].tolist(), members):
                cands[srcs[a]].add(int(m))

    def _accumulate_key_candidates(self, index: ProvenanceIndex, src: int, ctx, out) -> None:
        """Fallback path: src is not in any signature block (e.g. the signature provider was replaced after construction)."""
        sig = self._meta_signature(index.meta(int(src)))
        if sig is None or sig.size == 0:
            return
        b, r = self._bands_for(int(sig.size))
        for blk in ctx["blocks"]:
            if int(blk["L"]) != int(sig.size):
                continue
            for j in range(b):
                words = np.ascontiguousarray(sig[j * r : (j + 1) * r])
                h = _band_hashes(words.reshape(1, -1))[0]
                hs = blk["bands"][j][0]
                lo = int(np.searchsorted(hs, h, "left"))
                hi = int(np.searchsorted(hs, h, "right"))
                out.update(self._hash_members(blk, j, lo, hi,
                                              words.view(np.uint8).reshape(-1)))
            return

    def _lsh_neighbors(self, index: ProvenanceIndex, frontier: Sequence[int], ctx: Mapping) -> list:
        """Channel B: the whole frontier looks up buckets for all bands at once (vectorized), then MinHash Jaccard is computed exactly per candidate."""
        out = []
        if ctx.get("disabled") or not frontier:
            return out
        t0 = perf_counter()
        blocks = ctx["blocks"]
        block_id = ctx["block_id"]
        row_of_iid = ctx["row_of_iid"]
        n_total = int(block_id.size)
        fr = [int(i) for i in frontier]
        grouped = {}
        fallback = []
        for s in fr:
            bi = int(block_id[s]) if 0 <= s < n_total else -1
            if bi >= 0:
                grouped.setdefault(bi, []).append((s, int(row_of_iid[s])))
            else:
                fallback.append(s)

        cands = {s: set() for s in fr}
        for bi in sorted(grouped):
            self._accumulate_band_candidates(blocks[bi], grouped[bi], cands)
        for s in fallback:
            self._accumulate_key_candidates(index, s, ctx, cands[s])
        if fallback:
            grouped[None] = [(s, -1) for s in fallback]

        src_ctx = {}
        for bi, pairs in grouped.items():
            for s, row in pairs:
                src_ctx[s] = (bi, row)

        self.last_search_stats["n_lsh_candidates"] = (
            self.last_search_stats.get("n_lsh_candidates", 0) + sum(len(v) for v in cands.values())
        )
        for s in fr:
            cl = cands[s]
            cl.discard(s)
            if not cl:
                continue
            bi, row = src_ctx[s]
            sig = None
            if bi is None:
                sig = self._meta_signature(index.meta(s))
                if sig is None:
                    continue
                blk = None
                for cand_blk in blocks:
                    if int(cand_blk["L"]) == int(sig.size):
                        blk = cand_blk
                        break
                if blk is None:
                    continue
            else:
                blk = blocks[bi]
            dst_list = list(cl)
            mat = blk["mat"]
            drows = np.asarray([int(row_of_iid[d]) for d in dst_list], dtype=np.int64)
            left = np.asarray(sig) if sig is not None else mat[row]
            sims = np.mean(left[None, :] == mat[drows], axis=1)
            for a, dst in enumerate(dst_list):
                sim = float(sims[a])
                if sim >= self.lsh_threshold:
                    out.append((int(s), int(dst), sim, "lsh"))
        self.last_timings["lsh_s"] = self.last_timings.get("lsh_s", 0.0) + (perf_counter() - t0)
        return out
