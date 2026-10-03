"""fedrevoke.index_core — provenance-aware vector index (ProvenanceIndex).

Contract (INTERFACES.md Section 2, signatures must not change):

    @dataclass(frozen=True)
    class VecMeta:
        pid: int; doc_id: str; client_id: str; topic: str; fingerprint: str

    class ProvenanceIndex:
        __init__(dim, backend="faiss_ivf", **kw)
        add(vectors, metas) -> list[int]
        remove(internal_ids) -> None                  # idempotent
        search(queries, k=10) -> (scores, internal_ids)
        meta(internal_id) -> VecMeta
        ids_for_doc(doc_id) -> list[int]
        ids_for_client(client_id) -> list[int]
        alive_ids() -> np.ndarray
        save(path) -> None
        load(path) -> ProvenanceIndex                 # classmethod
        stats() -> {"n_vectors","n_deleted","backend","memory_bytes"}

Design notes
------------
1. **Stable internal ids**: internal ids are auto-increment integers 0..n_total-1;
   once assigned they never change (compact does not renumber).
   Reason: the delete_set from the M2 shadow closure, the removed_ids from M4
   repair, and the M6 audit credentials must all reference the same id space.
2. **Dual backends + fallback**: faiss_ivf (primary) and hnswlib (for
   connectivity experiments); when both are missing, automatically fall back to
   numpy (exact brute-force inner-product retrieval with chunked matmul). The
   fallback reason is recorded in stats()["fallback_reason"].
3. **tombstone + compact**: remove() only applies tombstones (idempotent, no
   renumbering); compact() rebuilds the ANN structure from alive vectors
   (reclaims memory, restores retrieval quality) while internal ids stay unchanged.
4. **search never returns deleted ids**: ANN candidates are first filtered by
   tombstones; if fewer than k remain (common at high deletion ratios), the
   short rows get one exact backfill pass (ANN backends only), so top-k
   semantics stay complete.
5. **Cosine similarity**: all vectors are L2-normalized at add time, so inner
   product retrieval equals cosine. search scores are cosine similarities.
6. **Padding when k exceeds the alive count**: the returned arrays always have
   shape (nq, k); missing slots are filled with PAD_ID = -1 and -inf scores
   (downstream code can ignore -1 directly, or convert with strip_padding()).
7. **Persistence is defined by npz+json**: vectors.npz + meta.json + manifest.json
   are the authoritative ground truth; native ANN files (ann.faiss / ann.hnsw) are
   only accelerators. If they are missing or inconsistent with the tombstone set,
   the ANN structure is rebuilt from vectors; behavior matches the pre-save state
   (search results after load are consistent; see tests/test_index_core.py).
"""

from __future__ import annotations

import json
import time
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

try:  # project-wide unified random seed (config.py; fall back to the contract constant if missing)
    from .config import SEED as _CONFIG_SEED  # type: ignore
except Exception:  # pragma: no cover - fallback when config is not in place yet
    _CONFIG_SEED = 20260214

SEED: int = int(_CONFIG_SEED)

INDEX_FORMAT = "fedrevoke.provenance_index.v1"
PAD_ID = -1
PAD_SCORE = float("-inf")

__all__ = [
    "VecMeta",
    "ProvenanceIndex",
    "available_backends",
    "strip_padding",
    "INDEX_FORMAT",
    "PAD_ID",
    "PAD_SCORE",
    "SEED",
]


# --------------------------------------------------------------------------------------
# VecMeta
# --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class VecMeta:
    """Provenance metadata for one vector (shard); fields align with data/processed/<ds>/corpus.jsonl."""

    pid: int
    doc_id: str
    client_id: str
    topic: str
    fingerprint: str

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "VecMeta":
        return cls(
            pid=int(d["pid"]),
            doc_id=str(d["doc_id"]),
            client_id=str(d["client_id"]),
            topic=str(d.get("topic", "")),
            fingerprint=str(d.get("fingerprint", "")),
        )


# --------------------------------------------------------------------------------------
# Small utilities
# --------------------------------------------------------------------------------------
def _as_2d_float32(x: Any) -> np.ndarray:
    a = np.asarray(x, dtype=np.float32)
    if a.ndim == 1:
        a = a.reshape(1, -1)
    if a.ndim != 2:
        raise ValueError("expected 1-D or 2-D array, got shape %r" % (a.shape,))
    return np.ascontiguousarray(a)


def _l2_normalize(x: np.ndarray) -> np.ndarray:
    """Row-wise L2 normalization (prerequisite for cosine retrieval). Zero vectors stay zero vectors."""
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    np.maximum(norms, 1e-12, out=norms)
    return (x / norms).astype(np.float32, copy=False)


def strip_padding(ids: np.ndarray, scores: np.ndarray) -> list:
    """Convert search's padded (ids, scores) result into per-row [(id, score), ...] (dropping -1 padding)."""
    ids = np.asarray(ids)
    scores = np.asarray(scores)
    out = []
    for row_ids, row_scores in zip(ids, scores):
        out.append([(int(i), float(s)) for i, s in zip(row_ids, row_scores) if int(i) != PAD_ID])
    return out


# --------------------------------------------------------------------------------------
# Optional dependency probing
# --------------------------------------------------------------------------------------
def _import_faiss():
    try:
        import faiss  # type: ignore

        return faiss
    except Exception:
        return None


def _import_hnswlib():
    try:
        import hnswlib  # type: ignore

        return hnswlib
    except Exception:
        return None


def available_backends() -> list:
    """List of backend names actually usable in the current interpreter (for test parametrization)."""
    names = []
    if _import_faiss() is not None:
        names.append("faiss_ivf")
    if _import_hnswlib() is not None:
        names.append("hnswlib")
    names.append("numpy")
    return names


# --------------------------------------------------------------------------------------
# ANN backends (internal implementations; vector ground truth is always held by ProvenanceIndex)
# --------------------------------------------------------------------------------------
class _BackendBase:
    name = "base"
    exact = False

    def add(self, ids: np.ndarray, vecs: np.ndarray) -> None:
        raise NotImplementedError

    def remove(self, ids: np.ndarray) -> None:
        raise NotImplementedError

    def rebuild(self, ids: np.ndarray, vecs: np.ndarray) -> None:
        raise NotImplementedError

    def search(self, queries: np.ndarray, k: int):
        raise NotImplementedError

    def wants_upgrade(self) -> bool:
        return False

    @property
    def needs_rebuild(self) -> bool:
        """Set when the backend considers its state untrustworthy; ProvenanceIndex then triggers compact()."""
        return bool(getattr(self, "_needs_rebuild", False))

    def memory_bytes(self) -> int:
        return 0

    def save(self, path: Path) -> bool:
        return False

    def load(self, path: Path, n_expected: int) -> bool:
        return False


class _NumpyBackend(_BackendBase):
    """Exact brute-force inner-product retrieval (chunked matmul); also the fallback backend when faiss/hnswlib are missing."""

    name = "numpy"
    exact = True

    def __init__(self, dim: int, **kw: Any) -> None:
        self.dim = int(dim)
        self.chunk_bytes = int(kw.pop("chunk_bytes", 128 * 1024 * 1024))
        self._ids = np.zeros(0, dtype=np.int64)
        self._mat = np.zeros((0, self.dim), dtype=np.float32)

    def add(self, ids: np.ndarray, vecs: np.ndarray) -> None:
        if len(ids) == 0:
            return
        self._ids = np.concatenate([self._ids, np.asarray(ids, dtype=np.int64)])
        self._mat = np.concatenate([self._mat, np.asarray(vecs, dtype=np.float32)], axis=0)

    def remove(self, ids: np.ndarray) -> None:
        if len(ids) == 0 or self._ids.size == 0:
            return
        keep = ~np.isin(self._ids, np.asarray(ids, dtype=np.int64))
        self._ids = self._ids[keep]
        self._mat = self._mat[keep]

    def rebuild(self, ids: np.ndarray, vecs: np.ndarray) -> None:
        self._ids = np.array(ids, dtype=np.int64, copy=True)
        self._mat = np.array(vecs, dtype=np.float32, copy=True)

    def search(self, queries: np.ndarray, k: int):
        nq = int(queries.shape[0])
        n = int(self._ids.size)
        kk = int(min(k, n))
        scores = np.full((nq, kk), -np.inf, dtype=np.float32)
        ids = np.full((nq, kk), PAD_ID, dtype=np.int64)
        if nq == 0 or n == 0 or kk <= 0:
            return scores, ids
        chunk = max(1, min(nq, self.chunk_bytes // max(1, n * 4)))
        for start in range(0, nq, chunk):
            block = queries[start : start + chunk]
            sims = block @ self._mat.T
            if kk < n:
                part = np.argpartition(-sims, kk - 1, axis=1)[:, :kk]
                part_scores = np.take_along_axis(sims, part, axis=1)
                order = np.argsort(-part_scores, axis=1, kind="stable")
                top = np.take_along_axis(part, order, axis=1)
            else:
                top = np.argsort(-sims, axis=1, kind="stable")
            scores[start : start + chunk] = np.take_along_axis(sims, top, axis=1)
            ids[start : start + chunk] = self._ids[top]
        return scores, ids

    def memory_bytes(self) -> int:
        return int(self._mat.nbytes + self._ids.nbytes)


class _FaissIVFBackend(_BackendBase):
    """faiss-cpu: IndexFlatIP at small scale, automatically upgraded to IndexIVFFlat once large enough (inner product = cosine)."""

    name = "faiss_ivf"
    exact = False

    def __init__(self, dim: int, **kw: Any) -> None:
        faiss = _import_faiss()
        if faiss is None:
            raise RuntimeError("faiss unavailable")
        self._faiss = faiss
        self.dim = int(dim)
        self.nlist = int(kw.pop("nlist", 256))
        self.nprobe = int(kw.pop("nprobe", min(16, max(1, self.nlist))))
        self._ivf_min = int(kw.pop("ivf_min_train", max(64, 4 * min(self.nlist, 4096))))
        self._mode = "flat"
        self._nlist = 0
        self._n_added = 0
        self._needs_rebuild = False
        self._index = None
        self._inner = None
        self._build(np.zeros(0, dtype=np.int64), np.zeros((0, self.dim), np.float32))

    # -- Internal ----------------------------------------------------------
    def _build(self, ids: np.ndarray, vecs: np.ndarray) -> None:
        """Build the underlying structure.

        Important (learned the hard way): **IVF must not be wrapped in IndexIDMap2**.
        IndexIDMap2.remove_ids confuses the selector's positional index with the
        user ids stored in the sub-index; after deleting 30% on IVF, retrieval
        results are almost entirely wrong (measured: top-10 Jaccard vs exact
        retrieval dropped from 0.85 to 0.03). IndexIVFFlat itself supports
        add_with_ids/remove_ids, so use it directly; IndexFlatIP does not support
        custom ids, which is the only case that needs an IndexIDMap2 wrapper
        (there position and id correspond one-to-one, so remove_ids semantics
        are correct).
        """
        faiss = self._faiss
        n = int(len(ids))
        ids = np.asarray(ids, dtype=np.int64)
        nlist = int(min(self.nlist, max(1, n // 39)))
        if nlist >= 2 and n >= self._ivf_min:
            index = faiss.IndexIVFFlat(
                faiss.IndexFlatIP(self.dim), self.dim, nlist, faiss.METRIC_INNER_PRODUCT
            )
            index.train(np.ascontiguousarray(vecs, dtype=np.float32))
            index.nprobe = min(self.nprobe, nlist)
            if n:
                index.add_with_ids(np.ascontiguousarray(vecs, dtype=np.float32), ids)
            self._mode, self._nlist = "ivf", nlist
        else:
            index = faiss.IndexIDMap2(faiss.IndexFlatIP(self.dim))
            if n:
                index.add_with_ids(np.ascontiguousarray(vecs, dtype=np.float32), ids)
            self._mode, self._nlist = "flat", 0
        self._index = index
        self._inner = index
        self._n_added = n
        self._needs_rebuild = False

    # -- Maintenance -------------------------------------------------------
    def add(self, ids: np.ndarray, vecs: np.ndarray) -> None:
        if len(ids) == 0:
            return
        self._index.add_with_ids(np.asarray(vecs, np.float32), np.asarray(ids, dtype=np.int64))
        self._n_added += int(len(ids))

    def remove(self, ids: np.ndarray) -> None:
        ids = np.ascontiguousarray(np.asarray(ids, dtype=np.int64))
        if ids.size == 0:
            return
        try:
            sel = self._faiss.IDSelectorBatch(ids.size, self._faiss.swig_ptr(ids))
            removed = int(self._index.remove_ids(sel))
        except Exception:
            self._needs_rebuild = True
            return
        self._n_added = max(0, self._n_added - max(removed, 0))
        if removed != int(ids.size):
            self._needs_rebuild = True

    def rebuild(self, ids: np.ndarray, vecs: np.ndarray) -> None:
        self._build(np.asarray(ids, dtype=np.int64), np.asarray(vecs, dtype=np.float32))

    def wants_upgrade(self) -> bool:
        return self._mode == "flat" and self._n_added >= self._ivf_min and self.nlist >= 2

    # -- Retrieval ---------------------------------------------------------
    def search(self, queries: np.ndarray, k: int):
        nq = int(queries.shape[0])
        if self._n_added <= 0 or nq == 0 or int(k) <= 0:
            return (
                np.full((nq, 0), -np.inf, dtype=np.float32),
                np.full((nq, 0), PAD_ID, dtype=np.int64),
            )
        if self._inner is not None and hasattr(self._inner, "nprobe") and self._nlist:
            self._inner.nprobe = min(max(1, self.nprobe), self._nlist)
        scores, ids = self._index.search(queries, int(k))
        return scores.astype(np.float32, copy=False), ids.astype(np.int64, copy=False)

    def memory_bytes(self) -> int:
        return int(self._n_added * (4 * self.dim + 8))

    # -- Persistence -------------------------------------------------------
    def save(self, path: Path) -> bool:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            self._faiss.write_index(self._index, str(path))
            return True
        except Exception:
            return False

    def load(self, path: Path, n_expected: int) -> bool:
        try:
            index = self._faiss.read_index(str(path))
            if int(getattr(index, "d", -1)) != self.dim:
                return False
            if int(index.ntotal) != int(n_expected):
                return False
            self._index = index
            self._n_added = int(index.ntotal)
            # No extra unwrap when it is already IVF; IndexIDMap2(Flat) has no nlist on the inner index
            nlist = int(getattr(index, "nlist", 0) or 0)
            if nlist > 0:
                self._inner = index
                self._mode, self._nlist = "ivf", nlist
            else:
                inner = getattr(index, "index", None)
                self._inner = inner
                nlist = int(getattr(inner, "nlist", 0) or 0)
                self._mode, self._nlist = ("ivf" if nlist > 1 else "flat"), nlist
            if self._inner is not None and hasattr(self._inner, "nprobe") and self._nlist:
                self._inner.nprobe = min(max(1, self.nprobe), self._nlist)
            self._needs_rebuild = False
            return True
        except Exception:
            return False


class _HnswBackend(_BackendBase):
    """hnswlib (cosine space). Deletion uses mark_deleted (equivalent to a tombstone)."""

    name = "hnswlib"
    exact = False

    def __init__(self, dim: int, **kw: Any) -> None:
        hnswlib = _import_hnswlib()
        if hnswlib is None:
            raise RuntimeError("hnswlib unavailable")
        self._hnswlib = hnswlib
        self.dim = int(dim)
        self.M = int(kw.pop("M", 16))
        self.ef_construction = int(kw.pop("ef_construction", 200))
        self.ef_search = int(kw.pop("ef_search", 64))
        cap = int(kw.pop("max_elements", 4096))
        self._cap = max(1024, cap)
        self._index = hnswlib.Index(space="cosine", dim=self.dim)
        self._index.init_index(max_elements=self._cap, ef_construction=self.ef_construction, M=self.M)
        self._index.set_ef(max(self.ef_search, self.M))
        self._n_added = 0
        self._n_deleted = 0
        self._needs_rebuild = False

    # -- Internal ----------------------------------------------------------
    def _ensure_capacity(self, need: int) -> None:
        if need <= self._cap:
            return
        new_cap = int(max(need, self._cap * 2))
        self._index.resize_index(new_cap)
        self._cap = new_cap

    @property
    def _n_alive(self) -> int:
        return max(0, self._n_added - self._n_deleted)

    # -- Maintenance -------------------------------------------------------
    def add(self, ids: np.ndarray, vecs: np.ndarray) -> None:
        if len(ids) == 0:
            return
        self._ensure_capacity(self._n_added + int(len(ids)))
        self._index.add_items(np.asarray(vecs, np.float32), np.asarray(ids, dtype=np.int64))
        self._n_added += int(len(ids))

    def remove(self, ids: np.ndarray) -> None:
        for i in np.asarray(ids, dtype=np.int64).tolist():
            try:
                self._index.mark_deleted(int(i))
                self._n_deleted += 1
            except Exception:
                continue

    def rebuild(self, ids: np.ndarray, vecs: np.ndarray) -> None:
        hnswlib = self._hnswlib
        ids = np.asarray(ids, dtype=np.int64)
        self._index = hnswlib.Index(space="cosine", dim=self.dim)
        self._cap = max(1024, int(len(ids)) * 2)
        self._index.init_index(max_elements=self._cap, ef_construction=self.ef_construction, M=self.M)
        self._index.set_ef(max(self.ef_search, self.M))
        self._n_added, self._n_deleted = 0, 0
        if len(ids):
            self._index.add_items(np.asarray(vecs, np.float32), ids)
            self._n_added = int(len(ids))
        self._needs_rebuild = False

    # -- Retrieval ---------------------------------------------------------
    def search(self, queries: np.ndarray, k: int):
        nq = int(queries.shape[0])
        kk = int(min(k, self._n_alive))
        if nq == 0 or kk <= 0:
            return (
                np.full((nq, 0), -np.inf, dtype=np.float32),
                np.full((nq, 0), PAD_ID, dtype=np.int64),
            )
        try:
            self._index.set_ef(max(self.ef_search, kk, self.M))
            labels, dists = self._index.knn_query(queries, k=kk)
            labels = np.asarray(labels, dtype=np.int64).reshape(nq, -1)
            dists = np.asarray(dists, dtype=np.float32).reshape(nq, -1)
        except Exception:
            # hnswlib may raise when ef/M are too small or the graph is disconnected;
            # return empty candidates and let the upper layer do exact backfill.
            return (
                np.full((nq, 0), -np.inf, dtype=np.float32),
                np.full((nq, 0), PAD_ID, dtype=np.int64),
            )
        return (1.0 - dists).astype(np.float32, copy=False), labels

    def memory_bytes(self) -> int:
        return int(self._n_added * (4 * self.dim + 8 * self.M))

    # -- Persistence -------------------------------------------------------
    def save(self, path: Path) -> bool:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            self._index.save_index(str(path))
            return True
        except Exception:
            return False

    def load(self, path: Path, n_expected: int) -> bool:
        try:
            index = self._hnswlib.Index(space="cosine", dim=self.dim)
            cap = max(1024, int(n_expected) * 2)
            index.load_index(str(path), max_elements=cap)
            index.set_ef(max(self.ef_search, self.M))
            self._index = index
            self._cap = cap
            # If hnswlib did not persist delete_list, tombstoned elements look alive after reload;
            # ProvenanceIndex.search filters tombstones at the result layer, so correctness is unaffected.
            self._n_added = int(n_expected)
            self._n_deleted = 0
            self._needs_rebuild = False
            return True
        except Exception:
            return False


_ALIASES = {
    "faiss": "faiss_ivf",
    "faiss_ivf": "faiss_ivf",
    "faiss-ivf": "faiss_ivf",
    "ivf": "faiss_ivf",
    "hnsw": "hnswlib",
    "hnswlib": "hnswlib",
    "numpy": "numpy",
    "exact": "numpy",
    "flat": "numpy",
    "bruteforce": "numpy",
    "brute_force": "numpy",
}


def _make_backend(backend: str, dim: int, **kw: Any):
    """Construct a backend; fall back to numpy when faiss/hnswlib are unavailable, reporting the reason."""
    key = _ALIASES.get(str(backend).strip().lower())
    if key is None:
        raise ValueError("unknown backend %r; expected one of %r" % (backend, sorted(set(_ALIASES))))
    if key == "faiss_ivf":
        try:
            return _FaissIVFBackend(dim, **kw), None
        except Exception as exc:  # pragma: no cover - depends on environment
            return _NumpyBackend(dim, **kw), "faiss_ivf unavailable, falling back to numpy: %s" % (exc,)
    if key == "hnswlib":
        try:
            return _HnswBackend(dim, **kw), None
        except Exception as exc:  # pragma: no cover - depends on environment
            return _NumpyBackend(dim, **kw), "hnswlib unavailable, falling back to numpy: %s" % (exc,)
    return _NumpyBackend(dim, **kw), None


# --------------------------------------------------------------------------------------
# ProvenanceIndex
# --------------------------------------------------------------------------------------
class ProvenanceIndex:
    """Vector index with provenance metadata and tombstone deletion (faiss_ivf primary / hnswlib / numpy fallback)."""

    def __init__(self, dim: int, backend: str = "faiss_ivf", **kw: Any) -> None:
        self.dim = int(dim)
        if self.dim <= 0:
            raise ValueError("dim must be positive, got %r" % (dim,))
        self.backend_requested = str(backend)
        self.auto_compact = bool(kw.pop("auto_compact", False))
        self.compact_ratio = float(kw.pop("compact_ratio", 0.25))
        self.oversample = max(1, int(kw.pop("oversample", 3)))
        self.backfill_chunk = max(1, int(kw.pop("backfill_chunk", 64)))
        self.kwargs = dict(kw)

        self._vectors = np.zeros((0, self.dim), dtype=np.float32)  # capacity array; logical length is _n_total
        self._n_total = 0
        self._metas = []  # type: list
        self._deleted = set()  # type: set
        self._doc_index = {}  # type: dict
        self._client_index = {}  # type: dict
        self._alive_cache = None  # type: tuple | None
        self._n_compactions = 0

        self._ann, self.fallback_reason = _make_backend(self.backend_requested, self.dim, **self.kwargs)
        if self.fallback_reason:
            warnings.warn(self.fallback_reason, RuntimeWarning, stacklevel=2)

    # ------------------------------------------------------------------ Basic properties
    def __repr__(self) -> str:
        return "ProvenanceIndex(dim=%d, backend=%r, n_vectors=%d, n_deleted=%d)" % (
            self.dim,
            self.backend,
            self._n_total,
            len(self._deleted),
        )

    def __len__(self) -> int:
        """Number of alive vectors."""
        return self.n_alive

    @property
    def backend(self) -> str:
        """Name of the backend actually in effect (numpy after fallback)."""
        return self._ann.name

    @property
    def n_vectors(self) -> int:
        """Total number of vectors accepted (including deleted tombstones)."""
        return self._n_total

    @property
    def n_alive(self) -> int:
        return self._n_total - len(self._deleted)

    # ------------------------------------------------------------------ Writes
    def _ensure_capacity(self, extra: int) -> None:
        need = self._n_total + int(extra)
        cap = int(self._vectors.shape[0])
        if need <= cap:
            return
        new_cap = int(max(need, max(64, cap * 2)))
        buf = np.zeros((new_cap, self.dim), dtype=np.float32)
        buf[: self._n_total] = self._vectors[: self._n_total]
        self._vectors = buf

    def add(self, vectors: Any, metas: Sequence) -> list:
        """Append vectors and return the assigned internal ids (consecutively increasing, permanently stable)."""
        vecs = _as_2d_float32(vectors)
        if vecs.shape[1] != self.dim:
            raise ValueError("vector dim %d != index dim %d" % (vecs.shape[1], self.dim))
        metas = list(metas)
        if len(metas) != vecs.shape[0]:
            raise ValueError("len(metas)=%d != n_vectors=%d" % (len(metas), vecs.shape[0]))
        if vecs.shape[0] == 0:
            return []
        vecs = _l2_normalize(vecs)

        start = self._n_total
        n = int(vecs.shape[0])
        self._ensure_capacity(n)
        self._vectors[start : start + n] = vecs

        for offset, m in enumerate(metas):
            meta = m if isinstance(m, VecMeta) else VecMeta.from_dict(m)
            iid = start + offset
            self._metas.append(meta)
            self._doc_index.setdefault(meta.doc_id, set()).add(iid)
            self._client_index.setdefault(meta.client_id, set()).add(iid)

        new_ids = np.arange(start, start + n, dtype=np.int64)
        self._n_total += n
        self._ann.add(new_ids, vecs)
        self._alive_cache = None
        if self._ann.wants_upgrade():
            self._rebuild_ann()  # scale upgrade flat -> IVF; not counted in n_compactions
        return new_ids.tolist()

    def remove(self, internal_ids: Sequence) -> None:
        """Tombstone delete; idempotent (repeated deletes, nonexistent ids, and out-of-range ids are all safely ignored)."""
        fresh = []
        seen = set()
        for raw in internal_ids:
            try:
                i = int(raw)
            except (TypeError, ValueError):
                continue
            if i in seen or i < 0 or i >= self._n_total or i in self._deleted:
                continue
            seen.add(i)
            fresh.append(i)
        if not fresh:
            return
        self._deleted.update(fresh)
        self._ann.remove(np.asarray(sorted(fresh), dtype=np.int64))
        self._alive_cache = None
        if self._ann.needs_rebuild:
            self.compact()
        if self.auto_compact:
            self.maybe_compact()

    def _rebuild_ann(self) -> int:
        """Internal: rebuild the ANN structure from alive vectors (not counted). Returns the number of alive vectors used."""
        ids = self.alive_ids()
        vecs = (
            np.ascontiguousarray(self._vectors[ids], dtype=np.float32)
            if ids.size
            else np.zeros((0, self.dim), dtype=np.float32)
        )
        self._ann.rebuild(ids.astype(np.int64, copy=False), vecs)
        self._alive_cache = None
        return int(ids.size)

    def compact(self) -> int:
        """Rebuild the ANN structure from alive vectors (reclaim memory/quality held by tombstones); internal ids stay unchanged.

        Returns the number of alive vectors used for the rebuild. stats()["n_compactions"]
        only counts compactions triggered explicitly / by threshold / by failure recovery,
        not the flat -> IVF scale-up rebuild.
        """
        n = self._rebuild_ann()
        self._n_compactions += 1
        return n

    def maybe_compact(self) -> bool:
        """Run compact when the tombstone ratio reaches compact_ratio; returns whether it actually ran."""
        if self._n_total == 0:
            return False
        if len(self._deleted) >= max(1, int(self.compact_ratio * self._n_total)):
            self.compact()
            return True
        return False

    # ------------------------------------------------------------------ Retrieval
    def _alive_mask(self) -> np.ndarray:
        mask = np.ones(self._n_total, dtype=bool)
        if self._deleted:
            mask[np.fromiter(self._deleted, dtype=np.int64, count=len(self._deleted))] = False
        return mask

    def _alive_arrays(self):
        """(alive_ids, alive_matrix); used for exact backfill when ANN candidates are eaten by tombstones."""
        if self._alive_cache is None:
            ids = self.alive_ids()
            mat = (
                np.ascontiguousarray(self._vectors[ids], dtype=np.float32)
                if ids.size
                else np.zeros((0, self.dim), dtype=np.float32)
            )
            self._alive_cache = (ids, mat)
        return self._alive_cache

    def search(self, queries: Any, k: int = 10):
        """Cosine top-k retrieval, returning (scores, internal_ids) with shape always (nq, k).

        Guarantees no deleted internal id is ever returned; positions short of k
        are padded with (PAD_SCORE, PAD_ID).
        """
        k = int(k)
        if k <= 0:
            raise ValueError("k must be >= 1, got %r" % (k,))
        q = _as_2d_float32(queries)
        if q.shape[1] != self.dim:
            raise ValueError("query dim %d != index dim %d" % (q.shape[1], self.dim))
        q = _l2_normalize(q)
        nq = int(q.shape[0])
        scores = np.full((nq, k), PAD_SCORE, dtype=np.float32)
        ids = np.full((nq, k), PAD_ID, dtype=np.int64)
        n_alive = self.n_alive
        if nq == 0 or n_alive == 0:
            return scores, ids

        kk = int(min(k, n_alive))
        fetch = int(min(self._n_total, max(kk * self.oversample, kk)))
        raw_scores, raw_ids = self._ann.search(q, fetch)
        raw_ids = np.asarray(raw_ids, dtype=np.int64)
        raw_scores = np.asarray(raw_scores, dtype=np.float32)
        mask = self._alive_mask()

        kept_ids = []
        kept_scores = []
        short_rows = []
        for row in range(nq):
            if row < raw_ids.shape[0]:
                row_ids = raw_ids[row]
                row_sc = raw_scores[row]
            else:
                row_ids = np.zeros(0, dtype=np.int64)
                row_sc = np.zeros(0, dtype=np.float32)
            if row_ids.size == 0:
                kept_ids.append(row_ids[:0])
                kept_scores.append(row_sc[:0])
                if not self._ann.exact:
                    short_rows.append(row)
                continue
            ok = (row_ids >= 0) & (row_ids < self._n_total)
            if ok.any():
                ok = ok & mask[np.where(ok, row_ids, 0)]
            sel_ids = row_ids[ok][:kk]
            sel_sc = row_sc[ok][:kk]
            kept_ids.append(sel_ids)
            kept_scores.append(sel_sc)
            if sel_ids.size < kk and not self._ann.exact:
                short_rows.append(row)

        # When ANN candidates are eaten by tombstones, exact-backfill the short rows (chunked to bound peak memory)
        if short_rows:
            a_ids, a_mat = self._alive_arrays()
            if a_ids.size:
                step = max(1, self.backfill_chunk)
                for start in range(0, len(short_rows), step):
                    rows = short_rows[start : start + step]
                    sims = q[rows] @ a_mat.T  # (nb, n_alive)
                    for local, row in enumerate(rows):
                        have = kept_ids[row]
                        need = kk - int(have.size)
                        if need <= 0:
                            continue
                        cand_ids, cand_sc = a_ids, sims[local]
                        if have.size:
                            keep = ~np.isin(cand_ids, have)
                            cand_ids, cand_sc = cand_ids[keep], cand_sc[keep]
                        if cand_ids.size == 0:
                            continue
                        if need < cand_ids.size:
                            part = np.argpartition(-cand_sc, need - 1)[:need]
                            part = part[np.argsort(-cand_sc[part], kind="stable")]
                        else:
                            part = np.argsort(-cand_sc, kind="stable")
                        all_ids = np.concatenate([have, cand_ids[part]])
                        all_sc = np.concatenate([kept_scores[row], cand_sc[part]])
                        order = np.argsort(-all_sc, kind="stable")[:kk]
                        kept_ids[row] = all_ids[order]
                        kept_scores[row] = all_sc[order]

        for row in range(nq):
            m = int(kept_ids[row].size)
            if m:
                ids[row, :m] = kept_ids[row][:kk]
                scores[row, :m] = kept_scores[row][:kk]
        return scores, ids

    # ------------------------------------------------------------------ Queries
    def meta(self, internal_id: int) -> VecMeta:
        """Return the metadata for an internal id. Metadata of deleted ids is retained (needed by M6 audits), but is invisible to retrieval."""
        i = int(internal_id)
        if i < 0 or i >= self._n_total:
            raise KeyError("internal id %r out of range [0, %d)" % (internal_id, self._n_total))
        return self._metas[i]

    def is_alive(self, internal_id: int) -> bool:
        i = int(internal_id)
        return 0 <= i < self._n_total and i not in self._deleted

    def vector(self, internal_id: int) -> np.ndarray:
        """Fetch a single normalized vector (a copy)."""
        self.meta(internal_id)  # range check
        return np.array(self._vectors[int(internal_id)], dtype=np.float32, copy=True)

    def vectors_for(self, internal_ids: Sequence) -> np.ndarray:
        """Fetch vectors in batch (deleted ids can also be fetched, for audit/repair)."""
        idx = np.asarray([int(i) for i in internal_ids], dtype=np.int64)
        if idx.size == 0:
            return np.zeros((0, self.dim), dtype=np.float32)
        if idx.min() < 0 or idx.max() >= self._n_total:
            raise KeyError("internal id out of range")
        return np.ascontiguousarray(self._vectors[idx], dtype=np.float32)

    def alive_matrix(self):
        """(alive_ids, alive_vectors) — for kNN graph construction/repair."""
        return self._alive_arrays()

    def ids_for_doc(self, doc_id: str) -> list:
        """Alive internal ids for a doc_id (ascending). Deleted ids are not returned."""
        return sorted(i for i in self._doc_index.get(str(doc_id), ()) if i not in self._deleted)

    def ids_for_client(self, client_id: str) -> list:
        """Alive internal ids for a client_id (silo) (ascending)."""
        return sorted(i for i in self._client_index.get(str(client_id), ()) if i not in self._deleted)

    def all_ids_for_doc(self, doc_id: str) -> list:
        """doc_id -> internal ids including deleted ones (for revocation credentials / audits)."""
        return sorted(self._doc_index.get(str(doc_id), ()))

    def all_ids_for_client(self, client_id: str) -> list:
        return sorted(self._client_index.get(str(client_id), ()))

    def alive_ids(self) -> np.ndarray:
        """All alive internal ids (ascending int64)."""
        if self._n_total == 0:
            return np.zeros(0, dtype=np.int64)
        if not self._deleted:
            return np.arange(self._n_total, dtype=np.int64)
        return np.flatnonzero(self._alive_mask()).astype(np.int64)

    def deleted_ids(self) -> np.ndarray:
        """All tombstoned internal ids (ascending int64)."""
        return np.fromiter(sorted(self._deleted), dtype=np.int64, count=len(self._deleted))

    def clients(self) -> list:
        return sorted(self._client_index)

    def doc_ids(self) -> list:
        return sorted(self._doc_index)

    def metas_snapshot(self) -> list:
        return list(self._metas)

    # ------------------------------------------------------------------ Stats
    def stats(self) -> dict:
        """Contract fields n_vectors / n_deleted / backend / memory_bytes plus extra diagnostic fields."""
        meta_bytes = 0
        for m in self._metas:
            meta_bytes += 200 + len(m.doc_id) + len(m.client_id) + len(m.topic) + len(m.fingerprint)
        mem = int(self._vectors.nbytes + self._ann.memory_bytes() + meta_bytes)
        return {
            "n_vectors": int(self._n_total),
            "n_deleted": int(len(self._deleted)),
            "backend": self.backend,
            "memory_bytes": mem,
            # --- Extra fields (diagnostics beyond the contract) ---
            "n_alive": int(self.n_alive),
            "dim": int(self.dim),
            "backend_requested": self.backend_requested,
            "fallback_reason": self.fallback_reason,
            "n_compactions": int(self._n_compactions),
            "n_docs": len(self._doc_index),
            "n_clients": len(self._client_index),
        }

    # ------------------------------------------------------------------ Persistence
    def save(self, path: str) -> None:
        """Save to a directory: manifest.json + meta.json + vectors.npz (+ optional native ANN file)."""
        root = Path(path)
        root.mkdir(parents=True, exist_ok=True)
        np.savez(
            root / "vectors.npz",
            vectors=self._vectors[: self._n_total],
            deleted=self.deleted_ids(),
            dim=np.int64(self.dim),
            n_total=np.int64(self._n_total),
        )
        (root / "meta.json").write_text(
            json.dumps({"metas": [m.to_dict() for m in self._metas]}, ensure_ascii=False),
            encoding="utf-8",
        )

        ann_file = root / ("ann.faiss" if self.backend == "faiss_ivf" else "ann.hnsw")
        ann_name = ann_file.name if self._ann.save(ann_file) else ""
        if not ann_name and ann_file.exists():
            ann_file.unlink()

        manifest = {
            "format": INDEX_FORMAT,
            "dim": int(self.dim),
            "backend": self.backend,
            "backend_requested": self.backend_requested,
            "kwargs": self.kwargs,
            "n_total": int(self._n_total),
            "n_deleted": int(len(self._deleted)),
            "n_alive": int(self.n_alive),
            "ann_file": ann_name,
            "saved_unix": float(time.time()),
        }
        (root / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    @classmethod
    def load(cls, path: str) -> "ProvenanceIndex":
        """Load from a directory; behavior matches the pre-save state (rebuild from alive vectors if the native ANN file is missing/inconsistent)."""
        root = Path(path)
        manifest_path = root / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError("%s not found" % (manifest_path,))
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("format") != INDEX_FORMAT:
            raise ValueError("unsupported index format: %r" % (manifest.get("format"),))

        npz = np.load(root / "vectors.npz", allow_pickle=False)
        vectors = np.asarray(npz["vectors"], dtype=np.float32)
        deleted = np.asarray(npz["deleted"], dtype=np.int64)
        meta_blob = json.loads((root / "meta.json").read_text(encoding="utf-8"))
        metas = [VecMeta.from_dict(d) for d in meta_blob["metas"]]

        dim = int(manifest["dim"])
        if vectors.shape[1] != dim:
            raise ValueError("saved vectors dim %d != manifest dim %d" % (vectors.shape[1], dim))
        if vectors.shape[0] != len(metas):
            raise ValueError("vectors(%d) != metas(%d)" % (vectors.shape[0], len(metas)))

        obj = cls(dim, backend=str(manifest.get("backend", "faiss_ivf")), **manifest.get("kwargs", {}))

        # --- Restore vectors and metadata ---
        obj._ensure_capacity(int(vectors.shape[0]))
        obj._vectors[: vectors.shape[0]] = vectors
        obj._n_total = int(vectors.shape[0])
        obj._metas = metas
        obj._deleted = set(int(i) for i in deleted.tolist() if 0 <= int(i) < obj._n_total)
        obj._doc_index = {}
        obj._client_index = {}
        for i, m in enumerate(obj._metas):
            obj._doc_index.setdefault(m.doc_id, set()).add(i)
            obj._client_index.setdefault(m.client_id, set()).add(i)
        obj._alive_cache = None

        # --- Restore ANN: prefer the native file; otherwise rebuild from alive vectors (results match the pre-save state) ---
        ann_name = str(manifest.get("ann_file") or "")
        loaded = False
        if ann_name:
            ann_path = root / ann_name
            if ann_path.exists():
                loaded = bool(obj._ann.load(ann_path, obj.n_alive))
        if not loaded:
            ids = obj.alive_ids()
            obj._ann.rebuild(
                ids.astype(np.int64, copy=False),
                np.ascontiguousarray(obj._vectors[ids], dtype=np.float32)
                if ids.size
                else np.zeros((0, obj.dim), dtype=np.float32),
            )
        return obj
