"""metrics.py -- FedRevoke evaluation metrics (pure numpy, boundary-safe).

Strictly implements the interfaces in INTERFACES.md Section 7:

    recall_at_k(retrieved_pids, gold_pids, k=10) -> float
    ndcg_at_k(retrieved_pids, gold_pids, k=10) -> float
    exact_match(pred, golds) -> float
    f1_score_tokens(pred, golds) -> float
    mia_auc(pos_scores, neg_scores) -> float
    CostMeter  # context manager: wall_time / peak_vram_mb / bytes_transferred

Also provides helper functions not listed with signatures in INTERFACES.md:
faithfulness / fairness_stats / normalize_answer / tokenize / measure_bytes.

Unified conventions
-------------------
1. Retrieval metrics only accept sequences of pids (internal ids or doc pids);
   duplicates are removed by first occurrence, then truncated to the top k
   (so repeated returns cannot inflate hits).
2. All boundaries are safe: empty gold, empty retrieval results, k <= 0, and
   all-NaN inputs all return 0.0 without raising exceptions.
3. mia_auc returns 0.5 when either class is empty (no information).
4. Return values are all Python floats within [0, 1].
5. No dependency on scipy / sklearn / torch (torch is only used lazily and
   optionally inside CostMeter for VRAM sampling).
"""

from __future__ import annotations

import math
import re
import string
import time
from collections import Counter
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

__all__ = [
    "recall_at_k",
    "ndcg_at_k",
    "exact_match",
    "f1_score_tokens",
    "faithfulness",
    "evidence_units",
    "mia_auc",
    "fairness_stats",
    "normalize_answer",
    "tokenize",
    "measure_bytes",
    "CostMeter",
]

_EPS = 1e-12
_ARTICLES_RE = re.compile(r"\b(a|an|the)\b", re.UNICODE)
_PUNCT_TABLE = str.maketrans({ch: " " for ch in string.punctuation})
_SENT_RE = re.compile(r"[^.!?\n]+", re.UNICODE)


# --------------------------------------------------------------------------- #
# Text normalization (SQuAD v2 style)
# --------------------------------------------------------------------------- #
def normalize_answer(text: Any) -> str:
    """Lowercase, strip punctuation, remove articles, and collapse whitespace. Safe for None / non-strings."""
    if text is None:
        return ""
    try:
        s = str(text)
    except Exception:  # pragma: no cover - extreme input
        return ""
    s = s.lower().translate(_PUNCT_TABLE)
    s = _ARTICLES_RE.sub(" ", s)
    return " ".join(s.split())


def tokenize(text: Any) -> list[str]:
    """Split on whitespace after normalization; empty input returns an empty list."""
    s = normalize_answer(text)
    return s.split() if s else []


def _as_str_list(values: Any) -> list[str]:
    if values is None:
        return []
    if isinstance(values, str):
        return [values]
    try:
        out = []
        for v in values:
            if v is None:
                continue
            out.append(str(v))
        return out
    except TypeError:
        return [str(values)]


def _as_id_list(values: Any) -> list[int]:
    """Convert to an int list with order-preserving dedup; skip elements that cannot be converted (e.g. NaN / None)."""
    if values is None:
        return []
    if isinstance(values, np.ndarray):
        seq: Iterable[Any] = values.ravel().tolist()
    elif isinstance(values, (str, bytes)):
        seq = [values]
    else:
        try:
            seq = list(values)
        except TypeError:
            seq = [values]
    out: list[int] = []
    seen: set[int] = set()
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


def _finite_array(values: Any) -> np.ndarray:
    """Convert to a 1-D float64 array and drop NaN / inf; return an empty array if parsing fails."""
    if values is None:
        return np.zeros(0, dtype=np.float64)
    try:
        arr = np.asarray(values, dtype=np.float64).ravel()
    except (TypeError, ValueError):
        try:
            arr = np.asarray([float(v) for v in values], dtype=np.float64).ravel()
        except Exception:
            return np.zeros(0, dtype=np.float64)
    if arr.size == 0:
        return arr
    return arr[np.isfinite(arr)]


# --------------------------------------------------------------------------- #
# Retrieval metrics
# --------------------------------------------------------------------------- #
def recall_at_k(retrieved_pids: Sequence[int], gold_pids: Sequence[int], k: int = 10) -> float:
    """Recall@k = |unique(retrieved[:k]) intersect gold| / |unique(gold)|.

    - The retrieval list is deduplicated by first occurrence, then truncated
      (so duplicate entries cannot inflate hits).
    - Returns 0.0 when gold is empty, k <= 0, or the retrieval list is empty.
    - When k < |gold| the upper bound is k / |gold| (standard IR definition,
      no cap normalization).
    """
    try:
        kk = int(k)
    except (TypeError, ValueError):
        return 0.0
    if kk <= 0:
        return 0.0
    gold = set(_as_id_list(gold_pids))
    if not gold:
        return 0.0
    retrieved = _as_id_list(retrieved_pids)[:kk]
    if not retrieved:
        return 0.0
    hits = sum(1 for pid in retrieved if pid in gold)
    return float(min(1.0, hits / float(len(gold))))


def ndcg_at_k(retrieved_pids: Sequence[int], gold_pids: Sequence[int], k: int = 10) -> float:
    """Binary-relevance nDCG@k with discount 1 / log2(rank + 1); rank starts at 1."""
    try:
        kk = int(k)
    except (TypeError, ValueError):
        return 0.0
    if kk <= 0:
        return 0.0
    gold = set(_as_id_list(gold_pids))
    if not gold:
        return 0.0
    retrieved = _as_id_list(retrieved_pids)[:kk]
    if not retrieved:
        return 0.0
    dcg = 0.0
    for rank, pid in enumerate(retrieved, start=1):
        if pid in gold:
            dcg += 1.0 / math.log2(rank + 1.0)
    ideal_hits = min(len(gold), kk)
    idcg = sum(1.0 / math.log2(r + 1.0) for r in range(1, ideal_hits + 1))
    if idcg <= _EPS:
        return 0.0
    return float(min(1.0, dcg / idcg))


# --------------------------------------------------------------------------- #
# Generation metrics
# --------------------------------------------------------------------------- #
def exact_match(pred: str, golds: Sequence[str]) -> float:
    """Return 1.0 if the normalized prediction exactly matches any gold, else 0.0 (empty pred / empty golds returns 0.0)."""
    gold_list = _as_str_list(golds)
    pred_norm = normalize_answer(pred)
    if not gold_list or not pred_norm:
        return 0.0
    for g in gold_list:
        if pred_norm == normalize_answer(g):
            return 1.0
    return 0.0


def f1_score_tokens(pred: str, golds: Sequence[str]) -> float:
    """Token-level F1, taking the maximum over all golds (SQuAD v2 convention)."""
    gold_list = _as_str_list(golds)
    pred_tokens = tokenize(pred)
    if not gold_list or not pred_tokens:
        return 0.0
    pred_counts = Counter(pred_tokens)
    best = 0.0
    for g in gold_list:
        gold_tokens = tokenize(g)
        if not gold_tokens:
            continue
        overlap = sum((pred_counts & Counter(gold_tokens)).values())
        if overlap <= 0:
            continue
        precision = overlap / float(len(pred_tokens))
        recall = overlap / float(len(gold_tokens))
        if precision + recall <= _EPS:
            continue
        f1 = 2.0 * precision * recall / (precision + recall)
        best = max(best, f1)
    return float(min(1.0, best))


def evidence_units(evidence: Sequence[str]) -> list[str]:
    """Evidence units: split each evidence item into **sentences** (sentences with >=3 tokens); fall back to the whole item if no split is possible.

    Rationale for the revision: whole evidence spans (100+ tokens) dilute
    support so answer sentences can never reach the threshold.
    """
    units: list[str] = []
    for ev in _as_str_list(evidence):
        sents = [s for s in (m.group(0).strip() for m in _SENT_RE.finditer(str(ev or ""))) if len(tokenize(s)) >= 3]
        if sents:
            units.extend(sents)
        elif tokenize(ev):
            units.append(str(ev))
    return units


def faithfulness(pred: str, evidence: Sequence[str], threshold: float = 0.5) -> float:
    """Fraction of answer sentences supported by retrieved evidence (rule-based).

    Support definition::

        support(s) = max_u  |tok(s) intersect tok(u)| / |tok(s)|      u ranges over a single **evidence unit**

    i.e. what fraction of an answer sentence's tokens can be found in the same
    evidence sentence; support >= threshold counts as supported.

    Evidence units = sentence-level splits of the evidence (evidence_units),
    not whole long evidence spans: the old definition compared a ~15-token
    answer sentence against a 100+ token full evidence span with token-F1 >= 0.5
    (requiring about 2/3 overlap), which is **always 0** under "short answer +
    long evidence" (empirically measured).

    Boundaries: empty evidence returns 0.0; empty answer returns 0.0; answer
    sentences with <3 tokens are not counted in the denominator; if the answer
    contains no sentence with >=3 tokens, the whole answer is treated as one
    unit for the judgment (otherwise correct short answers like "March 1973"
    would be structurally scored 0).
    """
    evidence_list = _as_str_list(evidence)
    if not evidence_list:
        return 0.0
    pred_text = "" if pred is None else str(pred)
    sentences = [s for s in (m.group(0).strip() for m in _SENT_RE.finditer(pred_text)) if len(tokenize(s)) >= 3]
    if not sentences:
        stripped = pred_text.strip()
        if not tokenize(stripped):
            return 0.0
        sentences = [stripped]
    unit_sets = [set(tokenize(u)) for u in evidence_units(evidence_list)]
    unit_sets = [u for u in unit_sets if u]
    if not unit_sets:
        return 0.0
    supported = 0
    for sent in sentences:
        toks = set(tokenize(sent))
        if not toks:
            continue
        best = 0.0
        for unit in unit_sets:
            score = len(toks & unit) / float(len(toks))
            if score > best:
                best = score
                if best >= 1.0:
                    break
        if best >= float(threshold):
            supported += 1
    return float(supported / len(sentences))


# --------------------------------------------------------------------------- #
# Membership inference AUC
# --------------------------------------------------------------------------- #
def _rankdata_average(values: np.ndarray) -> np.ndarray:
    """Average ranks (1-based), averaging ties, to avoid depending on scipy.stats.rankdata."""
    n = int(values.size)
    ranks = np.empty(n, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    sorted_vals = values[order]
    i = 0
    while i < n:
        j = i + 1
        while j < n and sorted_vals[j] == sorted_vals[i]:
            j += 1
        avg_rank = 0.5 * ((i + 1) + j)  # average of rank interval [i+1, j]
        ranks[order[i:j]] = avg_rank
        i = j
    return ranks


def mia_auc(pos_scores: Any, neg_scores: Any) -> float:
    """Mann-Whitney U form of AUC = P(score_pos > score_neg) + 0.5 P(tie).

    - Returns 0.5 when either class is empty (no information).
    - NaN / inf are dropped automatically; ties use average ranks, so all-tied
      inputs give 0.5.
    """
    pos = _finite_array(pos_scores)
    neg = _finite_array(neg_scores)
    n_pos, n_neg = int(pos.size), int(neg.size)
    if n_pos == 0 or n_neg == 0:
        return 0.5
    all_scores = np.concatenate([pos, neg])
    ranks = _rankdata_average(all_scores)
    rank_sum_pos = float(ranks[:n_pos].sum())
    auc = (rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(min(1.0, max(0.0, auc)))


def fairness_stats(values: Any) -> dict:
    """Summary of min / max / mean / std across per-silo metrics."""
    if isinstance(values, Mapping):
        arr = _finite_array(list(values.values()))
    else:
        arr = _finite_array(values)
    if arr.size == 0:
        return {"min": 0.0, "max": 0.0, "mean": 0.0, "std": 0.0, "n": 0}
    return {
        "min": float(arr.min()),
        "max": float(arr.max()),
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=0)),
        "n": int(arr.size),
    }


# --------------------------------------------------------------------------- #
# Cost metering
# --------------------------------------------------------------------------- #
def measure_bytes(obj: Any) -> int:
    """Recursively count the bytes occupied by an object (numpy arrays by nbytes; containers by summing elements)."""
    if obj is None:
        return 0
    if isinstance(obj, np.ndarray):
        return int(obj.nbytes)
    if isinstance(obj, (bytes, bytearray, memoryview)):
        return len(obj)
    if isinstance(obj, str):
        return len(obj.encode("utf-8", errors="ignore"))
    if isinstance(obj, Mapping):
        return int(sum(measure_bytes(v) for v in obj.values()))
    if isinstance(obj, (list, tuple, set, frozenset)):
        return int(sum(measure_bytes(v) for v in obj))
    nbytes = getattr(obj, "nbytes", None)
    if isinstance(nbytes, int):
        return int(nbytes)
    return 0


class CostMeter:
    """Cost metering context manager (INTERFACES.md Section 7).

    Usage::

        with CostMeter("repair") as meter:
            meter.add_bytes(vectors.nbytes)
            meter.touch(len(ids))
            ...
        cost = meter.as_dict()   # wall_time / peak_vram_mb / bytes_transferred / ...

    - wall_time: uses time.perf_counter; synchronizes before exit in CUDA scenarios.
    - peak_vram_mb: torch.cuda.max_memory_allocated(); recorded as 0.0 when torch
      is unavailable / CUDA is absent, with vram_available set to False
      (no exception raised, so the CPU path can be reused normally).
    - bytes_transferred: must be reported explicitly by the caller (add_bytes / record),
      because generic code cannot intercept real faiss / socket I/O.
    - Nesting is supported: only the outermost enter resets peak VRAM statistics.
    """

    _active: int = 0

    def __init__(self, name: str = "cost", measure_vram: bool = True, sync_cuda: bool = True) -> None:
        self.name = str(name)
        self.measure_vram = bool(measure_vram)
        self.sync_cuda = bool(sync_cuda)
        self.wall_time: float = 0.0
        self.peak_vram_mb: float = 0.0
        self.bytes_transferred: int = 0
        self.n_vectors_touched: int = 0
        self.vram_available: bool = False
        self.entered: bool = False
        self._t0: float | None = None
        self._torch: Any = None

    # -- VRAM -------------------------------------------------------------- #
    def _lazy_torch(self) -> Any:
        if self._torch is None and self.measure_vram:
            try:
                import torch  # lazy import: CPU environments do not need torch

                self._torch = torch
            except Exception:
                self._torch = False
        return self._torch

    def _reset_vram(self) -> None:
        torch = self._lazy_torch()
        if not torch:
            return
        try:
            if torch.cuda.is_available():
                self.vram_available = True
                if self._active == 0:
                    torch.cuda.synchronize()
                    torch.cuda.reset_peak_memory_stats()
        except Exception:
            self.vram_available = False

    def _read_vram(self) -> None:
        if not self.vram_available:
            return
        torch = self._torch
        try:
            self.peak_vram_mb = float(torch.cuda.max_memory_allocated()) / (1024.0 ** 2)
        except Exception:
            self.peak_vram_mb = 0.0

    def _sync(self) -> None:
        torch = self._torch
        if not torch:
            return
        try:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        except Exception:
            pass

    # -- Context protocol -------------------------------------------------- #
    def __enter__(self) -> "CostMeter":
        self._reset_vram()
        self._t0 = time.perf_counter()
        self.entered = True
        CostMeter._active += 1
        return self

    def __exit__(self, *exc: Any) -> bool:
        if self.sync_cuda:
            self._sync()
        if self._t0 is not None:
            self.wall_time = float(time.perf_counter() - self._t0)
        self._read_vram()
        CostMeter._active = max(0, CostMeter._active - 1)
        self.entered = False
        return False

    # -- Manual accounting ------------------------------------------------- #
    def add_bytes(self, n: Any) -> int:
        """Accumulate transferred bytes; numpy arrays / containers are estimated via measure_bytes."""
        if isinstance(n, (int, np.integer)):
            add = int(n)
        else:
            add = measure_bytes(n)
        self.bytes_transferred += max(0, add)
        return self.bytes_transferred

    def record(self, *objs: Any) -> int:
        """Report object byte sizes in batch (equivalent to calling add_bytes on each)."""
        for obj in objs:
            self.add_bytes(obj)
        return self.bytes_transferred

    def touch(self, n: int = 1) -> int:
        """Accumulate the number of vectors touched (n_vectors_touched)."""
        try:
            self.n_vectors_touched += max(0, int(n))
        except (TypeError, ValueError):
            pass
        return self.n_vectors_touched

    # -- Output ------------------------------------------------------------ #
    def as_dict(self) -> dict:
        wall = float(self.wall_time)
        return {
            "name": self.name,
            "wall_time": wall,          # seconds; same value as wall_time_s, for convenient use across modules
            "wall_time_s": wall,
            "peak_vram_mb": float(self.peak_vram_mb),
            "vram_available": bool(self.vram_available),
            "bytes_transferred": int(self.bytes_transferred),
            "n_vectors_touched": int(self.n_vectors_touched),
        }

    def __repr__(self) -> str:  # pragma: no cover - debugging only
        return (
            "CostMeter(name={0!r}, wall_time={1:.4f}s, peak_vram_mb={2:.1f}, "
            "bytes_transferred={3}, n_vectors_touched={4})"
        ).format(
            self.name,
            self.wall_time,
            self.peak_vram_mb,
            self.bytes_transferred,
            self.n_vectors_touched,
        )
