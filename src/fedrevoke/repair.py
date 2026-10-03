"""repair.py -- impact-aware repair: anchor local reconnection + similarity-score quantile calibration.

Corresponds to INTERFACES.md Section 4::

    class AnchorRepair:
        def __init__(self, n_anchors: int = 512, recalibrate: bool = True,
                     seed: int = 20260214): ...
        def repair(self, index, removed_ids, query_sample) -> dict
        # {"n_reconnected", "score_shift", "before_recall", "after_recall"}

    Parameter names/order/return fields match the contract exactly; only the **default value**
    of n_anchors is changed to 64 (consistent with the cost-experiment configuration in
    run_experiments' cost mode; empirically the default 512 drops recall because anchor copies
    crowd out the top-k). Behavior is unchanged
    when the caller passes the argument explicitly.

Algorithm (three steps)
------------
(a) Select anchors: prefer the nearest neighbors of each deleted vector among the **alive vectors**
    (removed_neighborhood, at most n_anchors, order-preserving dedup); when the deleted vectors are
    unavailable, fall back to using the neighborhood of query_sample as anchors (query_neighborhood);
    when neither is available, sample alive vectors uniformly with a fixed seed (random).

Neighborhood search is **ANN-first**, with complexity O(|removed| x k x log n) rather than
O(|removed| x n_alive x d) (optimization, replacing the old brute-force path that recomputed
_normalize_rows(pool) + pool @ q per deleted vector):

    1) ann_index_search          reuse the index's own search() (faiss IVF / hnswlib / numpy backend);
    2) exact_chunked             chunked (argpartition) exact kNN when the index search() is
                                 untrustworthy or unavailable, peak memory O(knn_chunk x n_alive),
                                 allowed only when |removed| x n_alive <= exact_knn_budget
                                 (a configurable tiny-scale fallback);
    3) temp_ann_faiss            build a one-off temporary ANN from **alive vectors** when over budget
                                 (the index itself is not modified);
    4) exact_chunked_over_budget explicit-warning fallback when even the temporary ANN cannot be built
                                 (correctness first).

The code no longer materializes an explicit (n_removed x n_alive) distance matrix: the ANN path
never builds one, and the exact fallback builds (knn_chunk x n_alive) intermediate blocks chunked
by knn_chunk.

The trustworthiness of the ANN oracle is gated by a **small-sample exact audit**: randomly sample
ann_audit_probes query vectors, compute ground truth with chunked exact kNN, and when the mean
overlap with the ANN results is below ann_audit_min_overlap, judge that search() as an unreliable
neighborhood oracle (e.g. a damaged graph index can only return its own connected component) and
automatically fall back to the exact / temporary-ANN path.
(b) Rebuild local connectivity:
    * Graph backends (object provides add_link/link/connect/add_edge): connect the alive neighbors
      of each deleted point pairwise, restoring the bridging role the deleted point used to play;
      n_reconnected records the number of newly added edges.
    * faiss_ivf backend: re-add anchor copies (reusing the vectors and meta of alive vectors;
      never resurrect deleted knowledge), then call inverted-list recomputation hooks such as
      rebuild_lists / recompute_lists / reassign_lists / refine_lists / rebalance_lists /
      rebuild_quantizer / repair_backend; n_reconnected records the number of copies actually added.
    * hnswlib backend: prefer index.repair(removed_ids) / repair_local / rebuild_local for local
      graph reconstruction; on failure fall back to anchor copies or pure edge linking.
(c) Score calibration (recalibrate=True): fit a monotone quantile mapping (QuantileCalibrator) on
    the similarity distributions of query_sample **before** and **after** repair, aligning the
    after-repair distribution to the before-repair one, and install the mapping back into the index
    (set_score_calibrator / score_calibrator attribute, installed when available) for later retrieval.
    score_shift     = mean displacement applied by calibration, mean|calibrated - raw|
    score_residual  = residual vs the target distribution after calibration, mean|calibrated_after - before|

Semantics of before_recall / after_recall
-----------------------------------
Because the interface does not pass gold, the default reference is **brute-force exact top-k over
alive vectors** (recall_mode = "exact_alive"): it contains no deleted knowledge, so it precisely
measures the effect of "deletion + repair" on the structural connectivity for unrelated queries.
If the caller holds true gold, it can be overridden via the optional gold_pids argument
(recall_mode = "provided"); when neither the index nor the caller can provide a reference,
recall_mode is recorded as "unavailable" and 0.0 is returned (raises when strict=True).

Duck-typed attributes the index must expose (ProvenanceIndex must provide one of the vector access paths):
    alive_ids()                        # already in INTERFACES Section 2
    vector(iid) / get_vector(iid) / reconstruct(iid)   # recommended
    vectors                            # or an internal matrix (row number == internal id)
    meta(iid) / text_for_id(iid)       # optional, for copying meta into anchor copies and generation-side verification
"""

from __future__ import annotations

import itertools
import time
import warnings
from dataclasses import dataclass
from typing import Any, Iterable, Optional, Sequence

import numpy as np

from .metrics import recall_at_k

try:  # pin the source of randomness; fall back to the seed in INTERFACES.md Section 1 when config.py is not ready
    from .config import SEED as _SEED  # type: ignore
except Exception:  # pragma: no cover
    _SEED = 20260214

__all__ = [
    "AnchorRepair",
    "QuantileCalibrator",
    "CalibrationMap",
    "index_backend",
    "index_alive_ids",
    "index_vector",
    "index_all_vectors",
    "index_meta",
    "index_text",
]

_EPS = 1e-12
_VECTOR_GETTERS = ("get_vector", "vector", "reconstruct", "get_embedding", "embedding")
_VECTOR_MATRIX_ATTRS = ("vectors", "_vectors", "vectors_", "xb", "embeddings", "vecs")
_TEXT_GETTERS = ("text_for_id", "get_text", "text", "doc_text", "raw_text")
_BACKEND_ATTRS = ("backend", "index_type", "kind")
_LIST_REBUILD_HOOKS = (
    "rebuild_lists",
    "recompute_lists",
    "reassign_lists",
    "refine_lists",
    "rebalance_lists",
    "rebuild_quantizer",
    "retrain_lists",
    "repair_backend",
)
_LINK_HOOKS = ("add_link", "link", "connect", "add_edge", "link_ids")


# --------------------------------------------------------------------------- #
# Duck-typed index utilities (shared by repair / verify)
# --------------------------------------------------------------------------- #
def _unique_ints(values: Any) -> list[int]:
    out: list[int] = []
    seen: set[int] = set()
    if values is None:
        return out
    if isinstance(values, (int, np.integer)):
        return [int(values)]
    try:
        seq: Iterable[Any] = list(values)
    except TypeError:
        seq = [values]
    for v in seq:
        try:
            iv = int(v)
        except (TypeError, ValueError):
            continue
        if iv in seen:
            continue
        seen.add(iv)
        out.append(iv)
    return out


def index_backend(index: Any) -> str:
    """Return the backend name (lowercase): prefer backend / index_type / kind attributes, then stats(), then the class name."""
    for attr in _BACKEND_ATTRS:
        val = getattr(index, attr, None)
        if isinstance(val, str) and val.strip():
            return val.strip().lower()
    stats_fn = getattr(index, "stats", None)
    if callable(stats_fn):
        try:
            stats = stats_fn()
            if isinstance(stats, dict):
                val = stats.get("backend")
                if isinstance(val, str) and val.strip():
                    return val.strip().lower()
        except Exception:
            pass
    return type(index).__name__.lower()


def index_alive_ids(index: Any) -> np.ndarray:
    """Alive internal ids (int64 1-D array). When alive_ids() is missing, infer from stats()["n_vectors"]."""
    fn = getattr(index, "alive_ids", None)
    if callable(fn):
        try:
            arr = np.asarray(fn())
            if arr.size:
                return arr.astype(np.int64).ravel()
            return np.zeros(0, dtype=np.int64)
        except Exception:
            pass
    for attr in ("_alive_ids", "alive", "_alive"):
        val = getattr(index, attr, None)
        if val is not None:
            try:
                return np.asarray(sorted(int(v) for v in val), dtype=np.int64)
            except Exception:
                continue
    stats_fn = getattr(index, "stats", None)
    if callable(stats_fn):
        try:
            stats = stats_fn()
            n = int(stats.get("n_vectors", 0))
            if n > 0:
                warnings.warn("index does not provide alive_ids(); inferring internal ids as 0..n_vectors-1")
                return np.arange(n, dtype=np.int64)
        except Exception:
            pass
    return np.zeros(0, dtype=np.int64)


def _vector_matrix(index: Any) -> Optional[np.ndarray]:
    for attr in _VECTOR_MATRIX_ATTRS:
        val = getattr(index, attr, None)
        if val is None:
            continue
        try:
            arr = np.asarray(val, dtype=np.float32)
        except Exception:
            continue
        if arr.ndim == 2 and arr.shape[0] > 0 and arr.shape[1] > 0:
            return arr
    return None


def index_vector(index: Any, internal_id: int) -> Optional[np.ndarray]:
    """Fetch the vector of a single internal id; return None if unavailable (no exception)."""
    iid = int(internal_id)
    for name in _VECTOR_GETTERS:
        fn = getattr(index, name, None)
        if not callable(fn):
            continue
        try:
            vec = fn(iid)
        except Exception:
            continue
        if vec is None:
            continue
        try:
            arr = np.asarray(vec, dtype=np.float32).ravel()
        except Exception:
            continue
        if arr.size:
            return arr
    mat = _vector_matrix(index)
    if mat is not None and 0 <= iid < mat.shape[0]:
        return mat[iid]
    return None


def _bulk_vectors(index: Any, id_list: list[int]) -> Optional[np.ndarray]:
    """Fetch all vectors in a single call (avoid per-id Python loops).

    Priority: vectors_for(ids) batch interface -> alive_matrix() (reused directly when the request
    is exactly all alive ids) -> internal matrix by row-number / compact-position indexing. The
    result of any path is exactly consistent with the per-id getters (same ids, same order);
    on failure return None and fall through to the next path.
    """
    if not id_list:
        return None
    batch = getattr(index, "vectors_for", None)
    if callable(batch):
        try:
            arr = np.asarray(batch(list(id_list)), dtype=np.float32)
            if arr.ndim == 2 and arr.shape[0] == len(id_list):
                return arr
        except Exception:
            pass
    alive_matrix = getattr(index, "alive_matrix", None)
    if callable(alive_matrix):
        try:
            out = alive_matrix()
            alive_ids = np.asarray(out[0], dtype=np.int64).ravel()
            mat = np.asarray(out[1], dtype=np.float32)
            if (
                alive_ids.size == len(id_list)
                and mat.ndim == 2
                and mat.shape[0] == len(id_list)
                and np.array_equal(alive_ids, np.asarray(id_list, dtype=np.int64))
            ):
                return mat
        except Exception:
            pass
    mat = _vector_matrix(index)
    if mat is not None:
        arr = np.asarray(id_list, dtype=np.int64)
        if int(arr.min()) >= 0 and int(arr.max()) < mat.shape[0]:
            return mat[arr]
        alive = index_alive_ids(index)
        if alive.size == mat.shape[0]:
            pos = {int(a): k for k, a in enumerate(alive.tolist())}
            if all(int(i) in pos for i in id_list):
                return mat[[pos[int(i)] for i in id_list]]
    return None


def index_all_vectors(index: Any, ids: Optional[Sequence[int]] = None) -> Optional[np.ndarray]:
    """Fetch vectors in batch, returning a (len(ids), dim) float32 matrix; return None when no path is available.

    Vector-access priority: batch interfaces vectors_for/alive_matrix -> internal matrix by row number
    -> compact index (row number == alive_ids order) -> per-id getters (the final compatibility path).
    """
    if ids is None:
        mat = _vector_matrix(index)
        return mat
    id_list = _unique_ints(ids)
    if not id_list:
        return np.zeros((0, 0), dtype=np.float32)
    fast = _bulk_vectors(index, id_list)
    if fast is not None:
        return fast
    has_getter = any(callable(getattr(index, name, None)) for name in _VECTOR_GETTERS)
    if has_getter:
        rows: list[np.ndarray] = []
        ok = True
        for iid in id_list:
            vec = index_vector(index, iid)
            if vec is None:
                ok = False
                break
            rows.append(vec)
        if ok and rows:
            try:
                return np.vstack(rows).astype(np.float32)
            except Exception:
                return None
    mat = _vector_matrix(index)
    if mat is not None:
        arr = np.asarray(id_list, dtype=np.int64)
        if int(arr.min()) >= 0 and int(arr.max()) < mat.shape[0]:
            return mat[arr]
        alive = index_alive_ids(index)
        if alive.size == mat.shape[0]:
            pos = {int(a): k for k, a in enumerate(alive.tolist())}
            if all(int(i) in pos for i in id_list):
                return mat[[pos[int(i)] for i in id_list]]
    return None


def index_meta(index: Any, internal_id: int) -> Any:
    """Fetch VecMeta; return None if unavailable."""
    fn = getattr(index, "meta", None)
    if callable(fn):
        try:
            return fn(int(internal_id))
        except Exception:
            return None
    return None


def index_text(index: Any, internal_id: int) -> Optional[str]:
    """Fetch the source text for an internal id; return None if unavailable."""
    for name in _TEXT_GETTERS:
        fn = getattr(index, name, None)
        if not callable(fn):
            continue
        try:
            val = fn(int(internal_id))
        except Exception:
            continue
        if isinstance(val, str) and val:
            return val
    return None


def _as_query_matrix(query_sample: Any) -> np.ndarray:
    if query_sample is None:
        return np.zeros((0, 0), dtype=np.float32)
    try:
        arr = np.asarray(query_sample, dtype=np.float32)
    except Exception:
        return np.zeros((0, 0), dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.ndim != 2:
        return np.zeros((0, 0), dtype=np.float32)
    return arr


def _normalize_rows(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms = np.where(norms <= _EPS, 1.0, norms)
    return mat / norms


def _topn_by_cosine(pool: np.ndarray, vec: np.ndarray, n: int) -> list[int]:
    """Exact top-n for a single vector (kept compatibility path; the main path is now ANN / chunked _topn_positions)."""
    if pool.size == 0 or n <= 0:
        return []
    q = vec.reshape(-1).astype(np.float32)
    qn = float(np.linalg.norm(q))
    if qn <= _EPS:
        return []
    pool_n = _normalize_rows(pool)
    sims = pool_n @ (q / qn)
    n = min(int(n), int(sims.size))
    if n <= 0:
        return []
    idx = np.argpartition(-sims, n - 1)[:n]
    idx = idx[np.argsort(-sims[idx], kind="mergesort")]
    return [int(i) for i in idx.tolist()]


def _topn_positions(pool_norm: np.ndarray, queries_norm: np.ndarray, n: int, chunk: int = 64) -> list[list[int]]:
    """Chunked exact top-n (returns pool row indices).

    The peak intermediate matrix is only (<=chunk, n_alive); **no explicit (n_queries x n_alive)
    distance matrix is constructed**; used only for tiny-scale fallback / ANN audit probes; the
    main path goes through ANN.
    """
    nq = int(queries_norm.shape[0]) if queries_norm.ndim == 2 else 0
    if nq == 0 or pool_norm.size == 0 or n <= 0:
        return [[] for _ in range(max(nq, 0))]
    n = int(min(int(n), int(pool_norm.shape[0])))
    step = max(1, int(chunk))
    out: list[list[int]] = []
    for start in range(0, nq, step):
        block = queries_norm[start : start + step]
        sims = block @ pool_norm.T
        if n < sims.shape[1]:
            part = np.argpartition(-sims, n - 1, axis=1)[:, :n]
            part_scores = np.take_along_axis(sims, part, axis=1)
            order = np.argsort(-part_scores, axis=1, kind="stable")
            top = np.take_along_axis(part, order, axis=1)
        else:
            top = np.argsort(-sims, axis=1, kind="stable")
        out.extend([int(v) for v in row.tolist()] for row in top)
    return out


class _TempAnn:
    """One-off temporary ANN (built from **alive vectors**, without modifying the index itself).

    Enabled only when the index's own search() is untrustworthy/unavailable and the exact fallback
    exceeds exact_knn_budget; returns pool row indices (consistent with faiss IndexFlatIP/IndexIVFFlat
    retrieval semantics). When the backend is unavailable construction fails (returns None) and the
    caller continues to degrade.
    """

    def __init__(self, pool_norm: np.ndarray, *, nlist_cap: int = 4096, train_rows: int = 65536) -> None:
        self._index = None
        self.kind = ""
        n, d = (int(pool_norm.shape[0]), int(pool_norm.shape[1])) if pool_norm.ndim == 2 else (0, 0)
        if n <= 0 or d <= 0:
            return
        try:
            import faiss  # type: ignore
        except Exception:
            return
        data = np.ascontiguousarray(pool_norm, dtype=np.float32)
        try:
            nlist = int(min(max(1, nlist_cap), max(1, int(np.sqrt(max(1, n))))))
            if n >= 2048 and nlist >= 2:
                index = faiss.IndexIVFFlat(faiss.IndexFlatIP(d), d, nlist, faiss.METRIC_INNER_PRODUCT)
                step = max(1, n // max(1, int(train_rows)))
                index.train(np.ascontiguousarray(data[::step]))
                index.add(data)
                index.nprobe = int(max(1, min(nlist, nlist // 16 or 1)))
                self.kind = "temp_ann_faiss_ivf"
            else:
                index = faiss.IndexFlatIP(d)
                index.add(data)
                self.kind = "temp_ann_faiss_flat"
            self._index = index
        except Exception:
            self._index = None
            self.kind = ""

    @property
    def available(self) -> bool:
        return self._index is not None

    def search(self, queries_norm: np.ndarray, k: int) -> Optional[np.ndarray]:
        """Return (nq, k) pool row indices; return None when unavailable."""
        if self._index is None:
            return None
        nq = int(queries_norm.shape[0]) if queries_norm.ndim == 2 else 0
        ntotal = int(getattr(self._index, "ntotal", 0))
        kk = int(min(max(1, int(k)), ntotal))
        if nq == 0 or kk <= 0:
            return np.zeros((max(nq, 0), 0), dtype=np.int64)
        try:
            _, ids = self._index.search(np.ascontiguousarray(queries_norm, dtype=np.float32), kk)
        except Exception:
            return None
        return np.asarray(ids, dtype=np.int64)


# Neighborhood oracle names (result["neighbor_oracle"])
ORACLE_ANN = "ann_index_search"
ORACLE_ANN_UNAUDITED = "ann_index_search_unaudited"
ORACLE_EXACT = "exact_chunked"
ORACLE_EXACT_OVER_BUDGET = "exact_chunked_over_budget"
ORACLE_TEMP_ANN = "temp_ann_faiss"
ORACLE_NONE = "none"


# --------------------------------------------------------------------------- #
# Quantile calibration
# --------------------------------------------------------------------------- #
@dataclass
class CalibrationMap:
    """Monotone quantile mapping: raw(after) scores -> target(before) score distribution.

    knots_x is strictly increasing (duplicate quantile points get a tiny perturbation) and knots_y
    is non-decreasing, so the mapping is monotone and can be extrapolated with np.interp
    (clamped to endpoint values beyond the endpoints).
    """

    knots_x: np.ndarray
    knots_y: np.ndarray
    n_pairs: int = 0
    n_quantiles: int = 0

    def apply(self, scores: Any) -> np.ndarray:
        arr = np.asarray(scores, dtype=np.float64)
        if self.knots_x.size == 0:
            return arr
        return np.interp(arr, self.knots_x, self.knots_y)

    def __call__(self, scores: Any) -> np.ndarray:
        return self.apply(scores)

    @property
    def mean_shift(self) -> float:
        if self.knots_x.size == 0:
            return 0.0
        return float(np.mean(np.abs(self.knots_y - self.knots_x)))

    def as_dict(self) -> dict:
        if self.knots_x.size == 0:
            return {"n_knots": 0, "n_pairs": int(self.n_pairs), "mean_shift": 0.0}
        return {
            "n_knots": int(self.knots_x.size),
            "n_pairs": int(self.n_pairs),
            "n_quantiles": int(self.n_quantiles),
            "x_min": float(self.knots_x.min()),
            "x_max": float(self.knots_x.max()),
            "y_min": float(self.knots_y.min()),
            "y_max": float(self.knots_y.max()),
            "mean_shift": self.mean_shift,
        }


class QuantileCalibrator:
    """Quantile aligner that monotonically maps one set of scores (after repair) onto another (before repair)."""

    def __init__(self, n_quantiles: int = 64) -> None:
        self.n_quantiles = max(2, int(n_quantiles))
        self.map_: Optional[CalibrationMap] = None

    def fit(self, target_scores: Any, source_scores: Any) -> CalibrationMap:
        """target = pre-repair distribution (alignment target), source = post-repair distribution (to be calibrated)."""
        target = np.asarray(target_scores, dtype=np.float64).ravel()
        source = np.asarray(source_scores, dtype=np.float64).ravel()
        target = target[np.isfinite(target)]
        source = source[np.isfinite(source)]
        if target.size == 0 or source.size == 0:
            self.map_ = CalibrationMap(
                knots_x=np.zeros(0, dtype=np.float64),
                knots_y=np.zeros(0, dtype=np.float64),
                n_pairs=0,
                n_quantiles=self.n_quantiles,
            )
            return self.map_
        qs = np.linspace(0.0, 1.0, self.n_quantiles)
        xs = np.quantile(source, qs)
        ys = np.quantile(target, qs)
        xs = np.maximum.accumulate(xs)
        ys = np.maximum.accumulate(ys)
        # Drop duplicate knots_x values (np.interp requires xp to be increasing)
        span = float(xs[-1] - xs[0]) if xs.size > 1 else 0.0
        eps = max(span, 1.0) * 1e-9
        for i in range(1, xs.size):
            if xs[i] <= xs[i - 1]:
                xs[i] = xs[i - 1] + eps
        self.map_ = CalibrationMap(
            knots_x=xs, knots_y=ys, n_pairs=int(min(target.size, source.size)), n_quantiles=self.n_quantiles
        )
        return self.map_

    def transform(self, scores: Any) -> np.ndarray:
        if self.map_ is None:
            return np.asarray(scores, dtype=np.float64)
        return self.map_.apply(scores)


# --------------------------------------------------------------------------- #
# Main class
# --------------------------------------------------------------------------- #
class AnchorRepair:
    """Anchor local reconnection + score calibration (INTERFACES.md Section 4).

    Neighborhood search is ANN-first (see the module docstring): all newly added performance-related
    switches come with defaults and do not change the existing call style
    repair(index, removed_ids, query_sample).

    n_anchors
        Upper bound on anchors. Default 64 (was 512): the cost-experiment configuration
        (run_experiments' cost mode) uses 64, and anchor copies are written into the index
        (n_vectors_delta ~ min(n_anchors, |removed|)), so the default aligns with the experimental
        protocol and minimizes default index-size inflation.
    max_replicas
        Upper bound on physical anchor copies (None = min(n_anchors, |removed|)). Setting 0 gives a
        "zero new vectors" pure-relink mode (n_vectors_delta = 0).
    knn_chunk / exact_knn_budget
        Chunk size of the exact fallback and the "tiny-scale" budget (when |removed| x n_alive
        exceeds the budget, go to ANN / temporary ANN).
    ann_audit_probes / ann_audit_min_overlap / audit_probe_budget
        Small-sample exact audit of the ANN oracle: sample ann_audit_probes query vectors; when the
        mean overlap with chunked exact kNN is below ann_audit_min_overlap, that search() is deemed
        untrustworthy (degrade to exact / temporary ANN). audit_probe_budget bounds the cost of the
        audit itself (probes x n_alive).
    """

    def __init__(
        self,
        n_anchors: int = 64,
        recalibrate: bool = True,
        seed: int = _SEED,
        n_neighbors: int = 8,
        k: int = 10,
        n_quantiles: int = 64,
        max_replicas: Optional[int] = None,
        max_removed_probe: int = 4096,
        strict: bool = False,
        knn_chunk: int = 64,
        exact_knn_budget: int = 8_000_000,
        ann_audit_probes: int = 16,
        ann_audit_min_overlap: float = 0.5,
        audit_probe_budget: int = 4_000_000,
    ) -> None:
        self.n_anchors = max(0, int(n_anchors))
        self.recalibrate = bool(recalibrate)
        self.seed = int(seed)
        self.n_neighbors = max(1, int(n_neighbors))
        self.k = max(1, int(k))
        self.n_quantiles = max(2, int(n_quantiles))
        self.max_replicas = None if max_replicas is None else max(0, int(max_replicas))
        self.max_removed_probe = max(1, int(max_removed_probe))
        self.strict = bool(strict)
        self.knn_chunk = max(1, int(knn_chunk))
        self.exact_knn_budget = max(0, int(exact_knn_budget))
        self.ann_audit_probes = max(1, int(ann_audit_probes))
        self.ann_audit_min_overlap = float(min(1.0, max(0.0, ann_audit_min_overlap)))
        self.audit_probe_budget = max(0, int(audit_probe_budget))
        self.calibration_: Optional[CalibrationMap] = None
        self.last_result_: Optional[dict] = None

    # -- Main entry -------------------------------------------------------- #
    def repair(
        self,
        index: Any,
        removed_ids: Sequence[int],
        query_sample: Any,
        gold_pids: Optional[Sequence[Sequence[int]]] = None,
        k: Optional[int] = None,
    ) -> dict:
        """Run repair and return the dict specified in INTERFACES.md Section 4 (plus diagnostic fields).

        Parameters
        ----
        index : ProvenanceIndex (duck typing is enough, see the module docstring)
        removed_ids : internal ids that have already been deleted (the result of M3 cascading erasure)
        query_sample : query vectors used to measure unrelated-query recall and score distribution (nq, dim)
        gold_pids : optional, per-query true gold (if not given, use brute-force top-k over alive vectors as reference)
        k : optional, overrides the constructor's recall@k
        """
        kk = self.k if k is None else max(1, int(k))
        removed = _unique_ints(removed_ids)
        queries = _as_query_matrix(query_sample)
        rng = np.random.default_rng(self.seed)
        alive = index_alive_ids(index)
        backend = index_backend(index)
        stats_before = self._stats_snapshot(index)

        result: dict = {
            # ---- Fields specified by INTERFACES.md Section 4 ----
            "n_reconnected": 0,
            "score_shift": 0.0,
            "before_recall": 0.0,
            "after_recall": 0.0,
            # ---- Diagnostic fields (for ablation / cost analysis) ----
            "n_removed": len(removed),
            "n_alive": int(alive.size),
            "backend": backend,
            "anchor_mode": "none",
            "n_anchors": 0,
            "n_anchor_replicas": 0,
            "n_links_created": 0,
            "repair_hooks": [],
            "rebuild_hooks": [],
            "repair_mode": "none",
            "recalibrated": False,
            "score_residual": 0.0,
            "recall_mode": "unavailable",
            "calibration": None,
            "k": kk,
            "warnings": [],
            # ---- neighborhood oracle / index-size cost ----
            "neighbor_oracle": ORACLE_NONE,
            "ann_audit_overlap": None,
            "ann_audit_probes": 0,
            "n_neighbor_vectors": 0,
            "neighbor_seconds": 0.0,
            "n_vectors_before": (None if stats_before is None else int(stats_before.get("n_vectors", 0))),
            "n_vectors_after": None,
            "n_vectors_delta": 0,
            "bytes_written": 0,
        }
        warn = result["warnings"].append

        # 1) Pre-repair retrieval (must happen before any writes)
        before_scores, before_ids = self._search(index, queries, kk, warn)

        # 2) Reference gold (default: brute-force exact top-k over alive vectors, 0 deleted knowledge included)
        gold, recall_mode = self._resolve_gold(index, alive, queries, gold_pids, kk)
        result["recall_mode"] = recall_mode
        if gold is not None and before_ids is not None:
            result["before_recall"] = self._mean_recall(before_ids, gold, kk)

        if not removed:  # no deletions -> no repair needed (stay idempotent, do not write the index)
            result["after_recall"] = result["before_recall"]
            result["repair_mode"] = "noop"
            if self.strict and (recall_mode == "unavailable" or before_ids is None):
                raise RuntimeError("AnchorRepair(strict=True): cannot compute recall (index has no search/vector interface)")
            self.last_result_ = result
            return result

        # 3) Anchor neighborhoods: alive neighbors of each deleted point (ANN-first, O(|removed| x k x log n))
        t_neighbor = time.perf_counter()
        used_removed, neighbor_lists, anchor_mode, nb_diag = self._neighbor_map(index, removed, alive, queries)
        result["neighbor_seconds"] = float(time.perf_counter() - t_neighbor)
        anchors = self._select_anchors(index, alive, neighbor_lists, anchor_mode, rng)
        result["anchor_mode"] = anchor_mode
        result["n_anchors"] = int(len(anchors))
        result["neighbor_oracle"] = str(nb_diag.get("neighbor_oracle") or ORACLE_NONE)
        result["ann_audit_overlap"] = nb_diag.get("ann_audit_overlap")
        result["ann_audit_probes"] = int(nb_diag.get("ann_audit_probes", 0) or 0)
        result["n_neighbor_vectors"] = int(nb_diag.get("n_neighbor_vectors", 0) or 0)
        if not anchors and not neighbor_lists:
            warn("No anchors obtained (vectors inaccessible or the deletion set is empty); repair degrades to no-op")

        # 4) Local reconnection (faiss / hnswlib / graph-backend branches)
        repair_info = self._reconnect(index, anchors, neighbor_lists, used_removed, backend, warn)
        result["n_anchor_replicas"] = int(repair_info.get("n_replicas", 0))
        result["n_links_created"] = int(repair_info.get("n_links", 0))
        result["repair_hooks"] = list(repair_info.get("hooks", []))
        result["rebuild_hooks"] = list(repair_info.get("rebuild_hooks", []))
        result["repair_mode"] = str(repair_info.get("mode", "none"))
        result["n_reconnected"] = int(repair_info.get("n_reconnected", 0))

        # 4b) Index-size cost (volume of anchor copies written; n_vectors_delta is the paper's "work" measure)
        stats_after = self._stats_snapshot(index)
        if stats_before is not None and stats_after is not None:
            n_before = int(stats_before.get("n_vectors", 0))
            n_after = int(stats_after.get("n_vectors", 0))
            dim = int(stats_after.get("dim", getattr(index, "dim", 0)) or 0)
            result["n_vectors_before"] = n_before
            result["n_vectors_after"] = n_after
            result["n_vectors_delta"] = int(n_after - n_before)
            result["bytes_written"] = int(max(0, n_after - n_before) * max(0, dim) * 4)

        # 5) Post-repair retrieval
        after_scores, after_ids = self._search(index, queries, kk, warn)
        if gold is not None and after_ids is not None:
            result["after_recall"] = self._mean_recall(after_ids, gold, kk)

        # 6) Score quantile calibration
        flat_before = None if before_scores is None else np.asarray(before_scores, dtype=np.float64).ravel()
        flat_after = None if after_scores is None else np.asarray(after_scores, dtype=np.float64).ravel()
        if flat_before is not None and flat_after is not None and flat_before.size and flat_after.size:
            raw_gap = float(np.mean(np.abs(flat_after - flat_before))) if flat_after.size == flat_before.size else float(
                abs(float(np.mean(flat_after)) - float(np.mean(flat_before)))
            )
            result["score_residual"] = raw_gap
            if self.recalibrate:
                calibrator = QuantileCalibrator(self.n_quantiles)
                cmap = calibrator.fit(flat_before, flat_after)
                self.calibration_ = cmap
                calibrated = cmap.apply(flat_after)
                result["score_shift"] = float(np.mean(np.abs(calibrated - flat_after)))
                result["calibration"] = cmap.as_dict()
                result["recalibrated"] = bool(cmap.knots_x.size > 0)
                if cmap.knots_x.size:
                    target = flat_before if flat_before.size == flat_after.size else np.quantile(
                        flat_before, np.linspace(0.0, 1.0, flat_after.size)
                    )
                    result["score_residual"] = float(np.mean(np.abs(calibrated - target)))
                    self._install_calibrator(index, cmap, warn)

        if self.strict and (result["recall_mode"] == "unavailable" or before_ids is None):
            raise RuntimeError(
                "AnchorRepair(strict=True): cannot compute recall (the index provides no search()/vector access "
                "interface, and the caller did not pass gold_pids)"
            )
        self.last_result_ = result
        return result

    # -- Internal: retrieval ----------------------------------------------- #
    @staticmethod
    def _search(
        index: Any, queries: np.ndarray, k: int, warn: Any
    ) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        if queries.size == 0:
            return None, None
        fn = getattr(index, "search", None)
        if not callable(fn):
            warn("index does not provide search(); cannot measure recall / score distribution")
            return None, None
        try:
            out = fn(queries, k)
        except Exception as exc:
            warn("index.search failed: {0}: {1}".format(type(exc).__name__, exc))
            return None, None
        try:
            scores, ids = out[0], out[1]
            return np.asarray(scores, dtype=np.float64), np.asarray(ids)
        except Exception:
            warn("index.search return value is not a (scores, ids) pair")
            return None, None

    # -- Internal: reference gold ------------------------------------------ #
    def _resolve_gold(
        self,
        index: Any,
        alive: np.ndarray,
        queries: np.ndarray,
        gold_pids: Optional[Sequence[Sequence[int]]],
        k: int,
    ) -> tuple[Optional[list[list[int]]], str]:
        if gold_pids is not None:
            gold = [[int(v) for v in _unique_ints(row)] for row in gold_pids]
            return gold, "provided"
        if queries.size == 0 or alive.size == 0:
            return None, "unavailable"
        alive_vecs = index_all_vectors(index, alive.tolist())
        if alive_vecs is None or alive_vecs.shape[0] != alive.size:
            return None, "unavailable"
        pool = _normalize_rows(alive_vecs)
        qmat = _normalize_rows(queries.astype(np.float32))
        kk = min(int(k), int(alive.size))
        sims = qmat @ pool.T
        gold: list[list[int]] = []
        for row in sims:
            idx = np.argpartition(-row, kk - 1)[:kk]
            idx = idx[np.argsort(-row[idx], kind="mergesort")]
            gold.append([int(alive[i]) for i in idx.tolist()])
        return gold, "exact_alive"

    @staticmethod
    def _mean_recall(retrieved: np.ndarray, gold: list[list[int]], k: int) -> float:
        if retrieved is None or not gold:
            return 0.0
        arr = np.asarray(retrieved)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        n = min(arr.shape[0], len(gold))
        if n <= 0:
            return 0.0
        vals = [recall_at_k(arr[i].tolist(), gold[i], k) for i in range(n)]
        return float(np.mean(vals)) if vals else 0.0

    # -- Internal: anchor neighborhoods (ANN-first) ------------------------ #
    @staticmethod
    def _stats_snapshot(index: Any) -> Optional[dict]:
        fn = getattr(index, "stats", None)
        if not callable(fn):
            return None
        try:
            stats = fn()
        except Exception:
            return None
        return dict(stats) if isinstance(stats, dict) else None

    def _ann_neighbor_ids(
        self, index: Any, alive_set: set, vecs: np.ndarray, m: int
    ) -> Optional[list[list[int]]]:
        """Reuse the index's own ANN search() for neighborhoods (O(|queries| x k x log n)); return None when unavailable."""
        fn = getattr(index, "search", None)
        if not callable(fn) or vecs.size == 0 or not alive_set:
            return None
        try:
            out = fn(vecs, int(m))
            ids = np.asarray(out[1])
        except Exception:
            return None
        if ids.ndim != 2 or ids.shape[0] != int(vecs.shape[0]) or ids.shape[1] == 0:
            return None
        lists: list[list[int]] = []
        for row in ids:
            keep: list[int] = []
            seen: set = set()
            for raw in np.asarray(row).ravel().tolist():
                iid = int(raw)
                if iid < 0 or iid in seen or iid not in alive_set:
                    continue
                seen.add(iid)
                keep.append(iid)
                if len(keep) >= m:
                    break
            lists.append(keep)
        if not any(lists):
            return None
        return lists

    def _audit_ann(
        self,
        pool_norm: np.ndarray,
        alive: np.ndarray,
        vecs: np.ndarray,
        ann_lists: list[list[int]],
        m: int,
    ) -> Optional[tuple]:
        """Small-sample exact audit of the ANN oracle: return (mean overlap, probe count); return None if auditing is impossible."""
        n = int(vecs.shape[0])
        n_alive = int(pool_norm.shape[0])
        if n <= 0 or n_alive <= 0 or not ann_lists:
            return None
        n_probe = min(int(self.ann_audit_probes), n)
        if self.audit_probe_budget > 0:
            n_probe = min(n_probe, int(self.audit_probe_budget // max(1, n_alive)))
        if n_probe <= 0:
            return None
        audit_rng = np.random.default_rng(int(self.seed) + 7919)  # independent RNG stream, does not perturb random anchors
        if n_probe < n:
            idx = np.sort(audit_rng.choice(n, size=n_probe, replace=False))
        else:
            idx = np.arange(n, dtype=np.int64)
        probes = _normalize_rows(np.asarray(vecs[idx], dtype=np.float32))
        exact = _topn_positions(pool_norm, probes, m, chunk=self.knn_chunk)
        overlaps: list = []
        for row, pos_list in zip(np.asarray(idx).ravel().tolist(), exact):
            gold = {int(alive[p]) for p in pos_list}
            got = {int(v) for v in ann_lists[int(row)]}
            if not gold or not got:
                continue
            overlaps.append(len(gold & got) / float(min(len(gold), len(got))))
        if not overlaps:
            return None
        return float(np.mean(overlaps)), int(len(overlaps))

    def _exact_neighbor_ids(
        self, pool_norm: np.ndarray, alive: np.ndarray, vecs: np.ndarray, m: int
    ) -> list[list[int]]:
        """Chunked exact kNN (only for tiny-scale fallback / audit probes; does not build the full distance matrix)."""
        pos = _topn_positions(
            pool_norm, _normalize_rows(np.asarray(vecs, dtype=np.float32)), m, chunk=self.knn_chunk
        )
        return [[int(alive[p]) for p in row] for row in pos]

    @staticmethod
    def _blank_zero_rows(lists: list[list[int]], zero_mask: np.ndarray) -> list[list[int]]:
        """Zero vectors have no direction and should not produce "nearest neighbors" (consistent with the original _topn_by_cosine semantics)."""
        if lists and zero_mask.size == len(lists) and bool(zero_mask.any()):
            return [[] if bool(zero_mask[i]) else lst for i, lst in enumerate(lists)]
        return lists

    def _neighbor_lists(
        self, index: Any, pool_norm: np.ndarray, alive: np.ndarray, vecs: np.ndarray, m: int
    ) -> tuple[list[list[int]], str, dict]:
        """Return (list of alive-neighbor id lists per query vector, oracle name, diagnostics).

        Ladder: ANN (the index's own search, via small-sample exact audit) -> chunked exact (within budget)
        -> one-off temporary ANN (built from alive vectors) -> over-budget exact fallback.
        """
        diag: dict = {
            "ann_audit_overlap": None,
            "ann_audit_probes": 0,
            "n_neighbor_vectors": int(vecs.shape[0]),
        }
        empty: list[list[int]] = [[] for _ in range(int(vecs.shape[0]))]
        if int(vecs.shape[0]) == 0 or pool_norm.size == 0:
            return empty, ORACLE_NONE, diag
        zero = np.linalg.norm(np.asarray(vecs, dtype=np.float32), axis=1) <= _EPS
        pairs = int(vecs.shape[0]) * int(alive.size)
        alive_set = {int(a) for a in alive.tolist()}

        ann = self._ann_neighbor_ids(index, alive_set, vecs, m)
        if ann is not None:
            audit = self._audit_ann(pool_norm, alive, vecs, ann, m)
            if audit is None:
                diag["neighbor_oracle"] = ORACLE_ANN_UNAUDITED
                return self._blank_zero_rows(ann, zero), ORACLE_ANN_UNAUDITED, diag
            overlap, n_probe = audit
            diag["ann_audit_overlap"] = float(overlap)
            diag["ann_audit_probes"] = int(n_probe)
            if overlap >= self.ann_audit_min_overlap:
                return self._blank_zero_rows(ann, zero), ORACLE_ANN, diag

        if pairs <= self.exact_knn_budget:
            lists = self._exact_neighbor_ids(pool_norm, alive, vecs, m)
            return self._blank_zero_rows(lists, zero), ORACLE_EXACT, diag

        temp = _TempAnn(pool_norm)
        ids = temp.search(_normalize_rows(np.asarray(vecs, dtype=np.float32)), m) if temp.available else None
        if ids is not None:
            n_alive = int(alive.size)
            lists = [
                [int(alive[p]) for p in np.asarray(row).ravel().tolist() if 0 <= int(p) < n_alive]
                for row in ids
            ]
            return self._blank_zero_rows(lists, zero), (temp.kind or ORACLE_TEMP_ANN), diag

        # Fallback: correctness first (diagnostic fields explicitly mark the over-budget case)
        lists = self._exact_neighbor_ids(pool_norm, alive, vecs, m)
        return self._blank_zero_rows(lists, zero), ORACLE_EXACT_OVER_BUDGET, diag

    def _neighbor_map(
        self, index: Any, removed: list[int], alive: np.ndarray, queries: np.ndarray
    ) -> tuple[list[int], list[list[int]], str, dict]:
        """Return (deleted ids taking part in reconnection, alive-neighbor lists per deleted/query point, anchor_mode, diagnostics)."""
        diag: dict = {}
        if alive.size <= 1:
            return [], [], "none", diag
        alive_vecs = index_all_vectors(index, alive.tolist())
        if alive_vecs is None or alive_vecs.shape[0] != alive.size:
            return [], [], "random", diag
        # Normalize the alive pool only once (the old implementation recomputed it per deleted vector, the root cause of the 6.5 s cost)
        pool_norm = _normalize_rows(np.asarray(alive_vecs, dtype=np.float32))
        m = min(self.n_neighbors, max(1, int(alive.size) - 1))
        neighbor_lists: list[list[int]] = []
        used: list[int] = []
        if removed:
            probe_ids = removed[: self.max_removed_probe]
            removed_vecs = index_all_vectors(index, probe_ids)
            if removed_vecs is not None and removed_vecs.shape[0] > 0:
                lists, oracle, diag = self._neighbor_lists(index, pool_norm, alive, removed_vecs, m)
                diag["neighbor_oracle"] = oracle
                for rid, lst in zip(probe_ids[: len(lists)], lists):
                    if lst:
                        neighbor_lists.append(lst)
                        used.append(int(rid))
                if neighbor_lists:
                    return used, neighbor_lists, "removed_neighborhood", diag
        if queries.size:
            qvecs = np.asarray(queries[: self.max_removed_probe], dtype=np.float32)
            lists, oracle, qdiag = self._neighbor_lists(index, pool_norm, alive, qvecs, m)
            diag = qdiag
            diag["neighbor_oracle"] = oracle
            for lst in lists:
                if lst:
                    neighbor_lists.append(lst)
            if neighbor_lists:
                return [], neighbor_lists, "query_neighborhood", diag
        return [], [], "random", diag

    def _select_anchors(
        self,
        index: Any,
        alive: np.ndarray,
        neighbor_lists: list[list[int]],
        anchor_mode: str,
        rng: np.random.Generator,
    ) -> list[int]:
        if alive.size == 0 or self.n_anchors == 0:
            return []
        picked: list[int] = []
        seen: set[int] = set()
        for neighbors in neighbor_lists:
            for aid in neighbors:
                if aid in seen:
                    continue
                seen.add(aid)
                picked.append(int(aid))
                if len(picked) >= self.n_anchors:
                    return picked
        if picked:
            return picked
        n = min(self.n_anchors, int(alive.size))
        idx = rng.choice(int(alive.size), size=n, replace=False)
        return [int(alive[i]) for i in np.sort(idx).tolist()]

    # -- Internal: reconnection -------------------------------------------- #
    def _reconnect(
        self,
        index: Any,
        anchors: list[int],
        neighbor_lists: list[list[int]],
        used_removed: list[int],
        backend: str,
        warn: Any,
    ) -> dict:
        info: dict = {
            "mode": "none",
            "n_reconnected": 0,
            "n_links": 0,
            "n_replicas": 0,
            "hooks": [],
            "rebuild_hooks": [],
        }
        modes: list[str] = []

        # (b1) Graph backend: connect the alive neighbors of each deleted point pairwise
        pairs: list[tuple[int, int]] = []
        seen_pairs: set[tuple[int, int]] = set()
        for neighbors in neighbor_lists:
            if len(neighbors) < 2:
                continue
            for a, b in itertools.combinations(sorted(neighbors), 2):
                key = (a, b)
                if key in seen_pairs:
                    continue
                seen_pairs.add(key)
                pairs.append(key)
        if pairs:
            n_links, link_hook = self._link_pairs(index, pairs)
            if link_hook:
                info["n_links"] = int(n_links)
                info["hooks"].append(link_hook)
                modes.append("relink")

        # (b2) Backend-specific: faiss copies + list recomputation / hnswlib local rebuild
        if backend.startswith("faiss") or backend in ("ivf", "ivf_flat", "faiss_ivf"):
            n_replicas, added = self._add_anchor_replicas(index, anchors, used_removed, warn)
            info["n_replicas"] = int(n_replicas)
            if added:
                modes.append("anchor_replica")
            rebuild = self._call_rebuild_hooks(index, warn)
            info["rebuild_hooks"] = rebuild
        elif "hnsw" in backend:
            hook = self._call_local_repair(index, used_removed, anchors, warn)
            if hook:
                info["hooks"].append(hook)
                modes.append("hnsw_local_repair")
            else:
                n_replicas, added = self._add_anchor_replicas(index, anchors, used_removed, warn)
                info["n_replicas"] = int(n_replicas)
                if added:
                    modes.append("anchor_replica")
        else:
            # Unknown backend: add copies when add is available, otherwise only logical linking
            if callable(getattr(index, "add", None)):
                n_replicas, added = self._add_anchor_replicas(index, anchors, used_removed, warn)
                info["n_replicas"] = int(n_replicas)
                if added:
                    modes.append("anchor_replica")
            rebuild = self._call_rebuild_hooks(index, warn)
            info["rebuild_hooks"] = rebuild

        if not modes:
            warn("No usable reconnection interface found (add_link / repair / add all unavailable); doing logical repair only")
            info["mode"] = "logical"
        else:
            info["mode"] = "+".join(modes)
        info["n_reconnected"] = int(info["n_links"] + info["n_replicas"])
        return info

    def _link_pairs(self, index: Any, pairs: list[tuple[int, int]]) -> tuple[int, Optional[str]]:
        fn = None
        hook = None
        for name in _LINK_HOOKS:
            candidate = getattr(index, name, None)
            if callable(candidate):
                fn = candidate
                hook = name
                break
        if fn is None or not pairs:
            return 0, None
        made = 0
        for a, b in pairs:
            if a == b:
                continue
            try:
                fn(int(a), int(b))
                made += 1
            except Exception:
                continue
        return made, hook

    def _add_anchor_replicas(
        self, index: Any, anchors: list[int], used_removed: list[int], warn: Any
    ) -> tuple[int, bool]:
        add = getattr(index, "add", None)
        if not callable(add) or not anchors:
            return 0, False
        cap = self.max_replicas if self.max_replicas is not None else min(self.n_anchors, max(len(used_removed), 0))
        if cap <= 0:
            return 0, False
        selected = anchors[:cap]
        vecs = index_all_vectors(index, selected)
        if vecs is None or vecs.shape[0] == 0:
            warn("Could not fetch anchor vectors; skipping anchor copies")
            return 0, False
        metas = [index_meta(index, aid) for aid in selected[: vecs.shape[0]]]
        keep = [i for i, m in enumerate(metas) if m is not None]
        if not keep:
            warn("Anchor copies need meta(iid), which the index does not provide; skipping copies")
            return 0, False
        try:
            added = add(vecs[keep], [metas[i] for i in keep])
        except Exception as exc:
            warn("anchor-copy add() failed: {0}: {1}".format(type(exc).__name__, exc))
            return 0, False
        n = len(added) if added is not None else len(keep)
        return int(n), True

    @staticmethod
    def _call_rebuild_hooks(index: Any, warn: Any) -> list[str]:
        called: list[str] = []
        for name in _LIST_REBUILD_HOOKS:
            fn = getattr(index, name, None)
            if not callable(fn):
                continue
            try:
                fn()
                called.append(name)
            except Exception as exc:
                warn("{0}() call failed: {1}: {2}".format(name, type(exc).__name__, exc))
        return called

    @staticmethod
    def _call_local_repair(
        index: Any, removed: list[int], anchors: list[int], warn: Any
    ) -> Optional[str]:
        for name in ("repair", "repair_local", "rebuild_local", "rebuild_neighborhood"):
            fn = getattr(index, name, None)
            if not callable(fn):
                continue
            attempts = (
                lambda: fn(list(removed)),
                lambda: fn(list(removed), anchors),
                lambda: fn(list(removed), anchor_ids=anchors),
                lambda: fn(),
            )
            for attempt in attempts:
                try:
                    attempt()
                    return name
                except TypeError:
                    continue
                except Exception as exc:
                    warn("{0}() call failed: {1}: {2}".format(name, type(exc).__name__, exc))
                    break
        return None

    @staticmethod
    def _install_calibrator(index: Any, cmap: CalibrationMap, warn: Any) -> None:
        setter = getattr(index, "set_score_calibrator", None)
        if callable(setter):
            try:
                setter(cmap)
                return
            except Exception as exc:
                warn("set_score_calibrator failed: {0}".format(exc))
        try:
            setattr(index, "score_calibrator", cmap)
        except Exception:
            warn("The index object does not accept the score_calibrator attribute; the calibration map is kept only in AnchorRepair.calibration_")

    # -- Convenience interface --------------------------------------------- #
    def calibrated_search(self, index: Any, queries: Any, k: int = 10) -> tuple[np.ndarray, np.ndarray]:
        """Retrieve and pass the scores through the most recently fitted calibration map (returned as-is when uncalibrated)."""
        scores, ids = index.search(queries, k)
        if self.calibration_ is not None:
            scores = self.calibration_.apply(scores)
        return scores, ids
