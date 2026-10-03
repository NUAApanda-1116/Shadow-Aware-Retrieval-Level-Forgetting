"""verify.py -- forgetting verification protocol: MIA probe + generation-side elicitation test + forget report.

Corresponds to INTERFACES.md Section 6::

    @dataclass
    class ForgetReport:
        hit_rate: float; mia_auc: float; elicit_rate: float; rho_hat: float; n_probes: int

    class MIAProbe:
        def __init__(self, index, encoder, seed=20260214): ...
        def auc(self, forgotten_pids, control_pids) -> float: ...

    class ElicitationTest:
        def __init__(self, generator, index, k=5, max_new_tokens=200): ...
        def run(self, forgotten_qa: list[dict]) -> float   # elicit_rate

Metric semantics (describe them this way in the paper; do not mix them up)
------------------------------------
* membership signal  s(x) = maximum cosine similarity between x and the **alive vectors** in the index.
  - Document still in the index -> self-match -> s is about 1.0 (strong membership signal).
  - Document fully deleted -> only neighborhood similarity remains -> s drops clearly.
* MIAProbe.auc(forgotten, control) = P(s(forgotten) > s(control)):
  - When control is "other silo documents still in the index", AUC closer to 1 means more severe
    revoked-knowledge residue; close to 0 means the revoked document no longer looks like a member
    (this project's forgetting-quality reading, lower is better).
  - When control is "documents never indexed", ideal forgetting should return to 0.5 (indistinguishable).
  - distinguishability() = max(auc, 1-auc) is also provided; 0.5 means the two classes are completely
    indistinguishable.
* ElicitationTest.elicit_rate: feed revocation-related Q&A (including retrieved evidence) to the
  generator and judge the fraction of items that "reproduce the deleted knowledge". Clean deletion
  should be about 0; without deletion it should be clearly greater than 0.
  Criterion v2:
    Reference text R(q) = explicit short-answer spans U the question's gold short answers U **answer spans**
                          extracted from the evidence source text
                          (answer-sentence selection: the sentence with the strongest lexical overlap with
                          the query, cut into a window of <= span_max_tokens);
    Any one of three signals hitting counts as elicitation success --
      (1) containment: forward -- the normalized generated answer contains the reference span, or shares a
                         contiguous token run of length >= ngram_min with it;
                         backward (answer attribution) -- >= backward_ratio of the generated answer's content
                         tokens fall in the same revoked evidence
                         (short extractive answers like "Peggy Wood" hit via this branch);
      (2) short-F1   : token-F1 against **short references** >= f1_threshold;
      (3) keyword    : reference content-token coverage >= keyword_threshold;
    Uninformative references (single-token binary answers like yes/no/true) are no longer used as criteria;
    such items are removed from the denominator (the denominator of elicit_rate = number of informative
    items), so binary-answer noise does not dominate the reading.
* ForgetReport.rho_hat: empirical total-variation (TV) residue upper bound, default max(hit_rate, elicit_rate);
  rho_hat_from_scores() can also estimate TV directly from similarity histograms.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any, Optional, Sequence

import numpy as np

from .metrics import f1_score_tokens, mia_auc, normalize_answer, tokenize
from .repair import (
    index_alive_ids,
    index_all_vectors,
    index_meta,
    index_text,
    index_vector,
)

try:  # random seed consistent with INTERFACES.md Section 1
    from .config import SEED as _SEED  # type: ignore
except Exception:  # pragma: no cover
    _SEED = 20260214

__all__ = [
    "ForgetReport",
    "MIAProbe",
    "ElicitationTest",
    "retrieval_hit_rate",
    "forgotten_pid_coverage",
    "rho_hat_residual",
    "rho_hat_from_scores",
    "total_variation",
    "build_forget_report",
    "DEFAULT_PROMPT_TEMPLATE",
    # Criterion v2 helpers: answer-span extraction and containment checks
    "split_sentences",
    "content_tokens",
    "is_informative_reference",
    "extract_answer_span",
    "extract_answer_spans",
    "longest_common_token_run",
    "answer_contained",
]

_EPS = 1e-12
DEFAULT_PROMPT_TEMPLATE = "Context:\n{context}\nQuestion: {question}\nAnswer:"
_STOPWORDS = frozenset(
    """a an the of is are was were be been being in on at to for from by with without and or
    not no do does did this that these those it its as which who whom whose what when where
    why how than then there here also into over under about after before between during""".split()
)


# --------------------------------------------------------------------------- #
# Criterion v2: answer-span-level decisions
# --------------------------------------------------------------------------- #
_SENT_SPLIT_RE = re.compile(r"[^.!?\n]+", re.UNICODE)
# Binary/agreement answer words: alone they contain no "revoked knowledge" and cannot serve as criterion
# references (empirical lesson: binary-answer noise dominates the reading)
_BINARY_TOKENS = frozenset(
    """yes no true false agree disagree agreed aligned align consistent consistently inconsistent
    contradictory contradict contradicting neutral unknown unsure maybe probably correct incorrect
    supported refuted same different similar more less higher lower equal""".split()
)


def split_sentences(text: Any) -> list[str]:
    """Split into sentences by . ! ? and newlines (same convention as metrics._SENT_RE); empty input returns []."""
    return [m.group(0).strip() for m in _SENT_SPLIT_RE.finditer(str(text or "")) if m.group(0).strip()]


def content_tokens(text: Any) -> list[str]:
    """Content-token sequence: normalized tokens with stopwords and words of length <= 2 removed."""
    return [t for t in tokenize(text) if len(t) > 2 and t not in _STOPWORDS]


def is_informative_reference(text: Any, min_tokens: int = 2) -> bool:
    """Whether the reference text carries discriminative knowledge: >= min_tokens tokens and not a pure binary answer."""
    toks = tokenize(text)
    if len(toks) < max(1, int(min_tokens)):
        return False
    if toks and all(t in _BINARY_TOKENS for t in toks):
        return False
    return True


def extract_answer_spans(
    query: Any, passage: Any, max_tokens: int = 12, max_spans: int = 3
) -> list[str]:
    """Extract up to max_spans answer spans (rank sentences by lexical overlap with the query, then cut windows).

    Measurements: taking only the best sentence misses cases where "the answer sentence is not the most
    relevant sentence" (common in NQ); taking a few more sentences significantly increases criterion
    sensitivity on positive controls, and applies equally to revoked queries (the comparison stays fair).
    """
    q_tokens = set(content_tokens(query)) or set(tokenize(query))
    scored: list = []
    for sent in split_sentences(passage):
        toks = tokenize(sent)
        if not toks:
            continue
        overlap = sum(1 for t in toks if t in q_tokens)
        if overlap <= 0:
            continue
        scored.append((overlap / float(len(toks) ** 0.5), overlap, toks))
    if not scored:
        return []
    scored.sort(key=lambda x: (-x[0], -x[1]))
    n_max = max(1, int(max_tokens))
    out: list[str] = []
    for _score, _ov, toks in scored[: max(1, int(max_spans))]:
        if len(toks) <= n_max:
            span = " ".join(toks)
        else:
            best_win, best_win_score = toks[:n_max], -1
            for start in range(0, len(toks) - n_max + 1):
                win = toks[start : start + n_max]
                ov = sum(1 for t in win if t in q_tokens)
                if ov > best_win_score:
                    best_win_score, best_win = ov, win
            span = " ".join(best_win)
        if span and span not in out:
            out.append(span)
    return out


def extract_answer_span(query: Any, passage: Any, max_tokens: int = 12) -> str:
    """Extract an **answer span** (short reference) from an evidence passage.

    Rules: first do answer-sentence selection (the sentence with the strongest content-token overlap with the
    query), then take the contiguous max_tokens window with the most query overlap in that sentence; return the
    whole sentence when it is not longer than max_tokens. Return "" when no lexical overlap is found.
    """
    q_tokens = set(content_tokens(query)) or set(tokenize(query))
    best_tokens: list[str] = []
    best_score = 0.0
    for sent in split_sentences(passage):
        toks = tokenize(sent)
        if not toks:
            continue
        overlap = sum(1 for t in toks if t in q_tokens)
        if overlap <= 0:
            continue
        score = overlap / float(len(toks) ** 0.5)
        if score > best_score:
            best_score, best_tokens = score, toks
    if not best_tokens:
        return ""
    n_max = max(1, int(max_tokens))
    if len(best_tokens) <= n_max:
        return " ".join(best_tokens)
    best_win, best_win_score = best_tokens[:n_max], -1
    for start in range(0, len(best_tokens) - n_max + 1):
        win = best_tokens[start : start + n_max]
        ov = sum(1 for t in win if t in q_tokens)
        if ov > best_win_score:
            best_win_score, best_win = ov, win
    return " ".join(best_win)


def longest_common_token_run(a: Sequence[str], b: Sequence[str]) -> int:
    """Length of the longest common **contiguous** token run (O(len(a)*len(b)) DP with a rolling array)."""
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    best = 0
    for ta in a:
        cur = [0] * (len(b) + 1)
        for j, tb in enumerate(b, start=1):
            if ta == tb:
                cur[j] = prev[j - 1] + 1
                if cur[j] > best:
                    best = cur[j]
        prev = cur
    return int(best)


def answer_contained(
    pred_tokens: Sequence[str], span_tokens: Sequence[str], ngram_min: int = 6
) -> tuple[bool, int]:
    """Normalized containment: the whole span hits, or there is a common contiguous run of length >= ngram_min.

    Returns (whether it hit, length of the longest common contiguous run). Short spans (<= ngram_min tokens)
    require a full-span hit.
    """
    run = longest_common_token_run(list(pred_tokens or []), list(span_tokens or []))
    if run <= 0:
        return False, 0
    span_len = len(list(span_tokens or []))
    need = span_len if span_len <= int(ngram_min) else int(ngram_min)
    return bool(run >= need), run


# --------------------------------------------------------------------------- #
# Forget report
# --------------------------------------------------------------------------- #
@dataclass
class ForgetReport:
    """Revocation verification result (INTERFACES.md Section 6; field order is the paper table order)."""

    hit_rate: float
    mia_auc: float
    elicit_rate: float
    rho_hat: float
    n_probes: int

    def as_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "ForgetReport":
        return cls(
            hit_rate=float(data.get("hit_rate", 0.0)),
            mia_auc=float(data.get("mia_auc", 0.5)),
            elicit_rate=float(data.get("elicit_rate", 0.0)),
            rho_hat=float(data.get("rho_hat", 0.0)),
            n_probes=int(data.get("n_probes", 0)),
        )

    def __str__(self) -> str:  # pragma: no cover - for logging
        return (
            "ForgetReport(hit_rate={0:.3f}, mia_auc={1:.3f}, elicit_rate={2:.3f}, "
            "rho_hat={3:.3f}, n_probes={4})"
        ).format(self.hit_rate, self.mia_auc, self.elicit_rate, self.rho_hat, self.n_probes)


def rho_hat_residual(hit_rate: float, elicit_rate: float) -> float:
    """Empirical total-variation residue upper bound: rho_hat = max(retrieval residue hit_rate, generation residue elicit_rate)."""
    try:
        return float(max(0.0, min(1.0, max(float(hit_rate), float(elicit_rate)))))
    except (TypeError, ValueError):
        return 0.0


def total_variation(p: Any, q: Any, bins: int = 32, value_range: Optional[tuple] = None) -> float:
    """Total variation between two sample sets (or normalized histograms): 0.5 * sum|p - q|."""
    p_arr = np.asarray(p, dtype=np.float64).ravel()
    q_arr = np.asarray(q, dtype=np.float64).ravel()
    p_arr = p_arr[np.isfinite(p_arr)]
    q_arr = q_arr[np.isfinite(q_arr)]
    if p_arr.size == 0 or q_arr.size == 0:
        return 0.0
    if value_range is None:
        lo = float(min(p_arr.min(), q_arr.min()))
        hi = float(max(p_arr.max(), q_arr.max()))
        if hi <= lo:
            hi = lo + 1e-6
        value_range = (lo, hi)
    edges = np.linspace(value_range[0], value_range[1], int(bins) + 1)
    hp, _ = np.histogram(p_arr, bins=edges)
    hq, _ = np.histogram(q_arr, bins=edges)
    hp = hp / max(1, hp.sum())
    hq = hq / max(1, hq.sum())
    return float(0.5 * np.abs(hp - hq).sum())


def rho_hat_from_scores(before_scores: Any, after_scores: Any, bins: int = 32) -> float:
    """Estimate the TV distance from similarity score distributions."""
    return total_variation(before_scores, after_scores, bins=bins)


def build_forget_report(
    hit_rate: float,
    mia_auc_value: float,
    elicit_rate: float,
    rho_hat: Optional[float] = None,
    n_probes: int = 0,
) -> ForgetReport:
    """Uniformly construct a ForgetReport; rho_hat defaults to rho_hat_residual()."""
    hr = float(hit_rate)
    er = float(elicit_rate)
    rho = rho_hat_residual(hr, er) if rho_hat is None else float(rho_hat)
    return ForgetReport(
        hit_rate=hr,
        mia_auc=float(mia_auc_value),
        elicit_rate=er,
        rho_hat=rho,
        n_probes=int(n_probes),
    )


# --------------------------------------------------------------------------- #
# Retrieval-side hit rate
# --------------------------------------------------------------------------- #
def _normalize_ids(values: Any) -> list[int]:
    out: list[int] = []
    if values is None:
        return out
    arr = np.asarray(values).ravel() if isinstance(values, np.ndarray) else list(values)
    for v in arr:
        try:
            out.append(int(v))
        except (TypeError, ValueError):
            continue
    return out


def retrieval_hit_rate(
    index: Any, query_vectors: Any, forgotten_pids: Sequence[int], k: int = 10
) -> float:
    """Fraction of queries for which a deleted document enters the top-k (retrieval-side definition of ForgetReport.hit_rate)."""
    queries = np.asarray(query_vectors, dtype=np.float32)
    if queries.ndim == 1:
        queries = queries.reshape(1, -1)
    if queries.size == 0:
        return 0.0
    forgotten = set(_normalize_ids(forgotten_pids))
    if not forgotten:
        return 0.0
    try:
        _, ids = index.search(queries, int(k))
    except Exception:
        return 0.0
    ids = np.asarray(ids)
    if ids.ndim == 1:
        ids = ids.reshape(1, -1)
    hits = 0
    for row in ids:
        if any(int(v) in forgotten for v in row.tolist()):
            hits += 1
    return float(hits / max(1, ids.shape[0]))


def forgotten_pid_coverage(
    index: Any, query_vectors: Any, forgotten_pids: Sequence[int], k: int = 10
) -> float:
    """Fraction of "deleted documents" recalled by at least one query (a stricter residue reading)."""
    queries = np.asarray(query_vectors, dtype=np.float32)
    if queries.ndim == 1:
        queries = queries.reshape(1, -1)
    forgotten = set(_normalize_ids(forgotten_pids))
    if queries.size == 0 or not forgotten:
        return 0.0
    try:
        _, ids = index.search(queries, int(k))
    except Exception:
        return 0.0
    seen: set[int] = set()
    for row in np.asarray(ids):
        for v in np.asarray(row).ravel().tolist():
            try:
                iv = int(v)
            except (TypeError, ValueError):
                continue
            if iv in forgotten:
                seen.add(iv)
    return float(len(seen) / len(forgotten))


# --------------------------------------------------------------------------- #
# MIA probe
# --------------------------------------------------------------------------- #
class MIAProbe:
    """Membership inference probe: uses "maximum similarity to alive vectors" as the membership signal."""

    def __init__(
        self,
        index: Any,
        encoder: Any = None,
        seed: int = _SEED,
        k: int = 10,
        # Default True: consistent with the signal s(x)=max_{u in V\\{x}} <phi(p_x),phi(p_u)> in Eq.(17) of the paper.
        # False would count the probe itself into the neighbor pool, making alive samples always ~1.0 and
        # revoked samples capped at 0.98, so AUC degrades to anti-correlation (~0) and loses discriminative
        # power. For ablation only; do not make it the default.
        exclude_self: bool = True,
        max_probes: int = 4096,
    ) -> None:
        self.index = index
        self.encoder = encoder
        self.seed = int(seed)
        self.k = max(1, int(k))
        self.exclude_self = bool(exclude_self)
        self.max_probes = max(1, int(max_probes))
        self.n_skipped = 0
        self.last_scores_: dict[int, float] = {}

    # -- Vector access ----------------------------------------------------- #
    def _probe_vector(self, pid: int) -> Optional[np.ndarray]:
        vec = index_vector(self.index, int(pid))
        if vec is not None:
            return vec.astype(np.float32)
        text = index_text(self.index, int(pid))
        if text and self.encoder is not None:
            enc = _encode_texts(self.encoder, [text])
            if enc is not None and enc.shape[0] == 1:
                return enc[0].astype(np.float32)
        return None

    def _pool(self) -> Optional[tuple[np.ndarray, np.ndarray]]:
        alive = index_alive_ids(self.index)
        if alive.size == 0:
            return None
        pool = index_all_vectors(self.index, alive.tolist())
        if pool is None or pool.shape[0] != alive.size:
            return None
        return pool.astype(np.float32), alive.astype(np.int64)

    @staticmethod
    def _normalize(mat: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms = np.where(norms <= _EPS, 1.0, norms)
        return mat / norms

    # -- Membership signal ------------------------------------------------- #
    def membership_scores(self, pids: Sequence[int]) -> np.ndarray:
        """Compute the membership signal s(x) = max cosine similarity to alive vectors, per pid."""
        pid_list = _normalize_ids(pids)[: self.max_probes]
        pool_info = self._pool()
        if pool_info is None or not pid_list:
            self.n_skipped = len(pid_list)
            return np.zeros(0, dtype=np.float64)
        pool, alive = pool_info
        pool_n = self._normalize(pool)
        alive_set = set(int(a) for a in alive.tolist())
        scores: list[float] = []
        self.n_skipped = 0
        for pid in pid_list:
            vec = self._probe_vector(pid)
            if vec is None:
                self.n_skipped += 1
                continue
            v = vec.reshape(-1).astype(np.float32)
            norm = float(np.linalg.norm(v))
            if norm <= _EPS:
                self.n_skipped += 1
                continue
            sims = pool_n @ (v / norm)
            if self.exclude_self and pid in alive_set:
                pos = np.where(alive == pid)[0]
                if pos.size:
                    sims = np.delete(sims, pos[0])
            if sims.size == 0:
                self.n_skipped += 1
                continue
            scores.append(float(sims.max()))
            self.last_scores_[int(pid)] = float(sims.max())
        return np.asarray(scores, dtype=np.float64)

    def neighbor_scores(self, pids: Sequence[int]) -> np.ndarray:
        """Reconstruction-error variant of the signal: similarity to the nearest neighbor (self-match excluded); smaller means harder to reconstruct."""
        pid_list = _normalize_ids(pids)[: self.max_probes]
        pool_info = self._pool()
        if pool_info is None or not pid_list:
            return np.zeros(0, dtype=np.float64)
        pool, alive = pool_info
        pool_n = self._normalize(pool)
        alive_list = [int(a) for a in alive.tolist()]
        out: list[float] = []
        for pid in pid_list:
            vec = self._probe_vector(pid)
            if vec is None:
                continue
            v = vec.reshape(-1).astype(np.float32)
            norm = float(np.linalg.norm(v))
            if norm <= _EPS:
                continue
            sims = pool_n @ (v / norm)
            keep = np.array([i for i, a in enumerate(alive_list) if a != int(pid)], dtype=np.int64)
            if keep.size == 0:
                continue
            out.append(float(sims[keep].max()))
        return np.asarray(out, dtype=np.float64)

    # -- AUC --------------------------------------------------------------- #
    def auc(self, forgotten_pids: Sequence[int], control_pids: Sequence[int]) -> float:
        """AUC = P(s(forgotten) > s(control)); returns 0.5 when either set has no valid probes.

        Interpretation is in the module docstring: when control is alive documents, lower AUC means cleaner deletion.
        """
        pos = self.membership_scores(forgotten_pids)
        neg = self.membership_scores(control_pids)
        return float(mia_auc(pos, neg))

    def distinguishability(self, forgotten_pids: Sequence[int], control_pids: Sequence[int]) -> float:
        """Distinguishability max(auc, 1 - auc): 0.5 means the two classes are completely indistinguishable (ideal forgetting)."""
        value = self.auc(forgotten_pids, control_pids)
        return float(max(value, 1.0 - value))

    def report_scores(self, pids: Sequence[int]) -> list[dict]:
        """Per-probe diagnostics (for score histograms / appendix tables)."""
        pid_list = _normalize_ids(pids)
        scores = self.membership_scores(pid_list)
        return [{"pid": int(pid), "score": float(s)} for pid, s in zip(pid_list[: scores.size], scores)]

    def hit_rate(self, query_vectors: Any, forgotten_pids: Sequence[int], k: Optional[int] = None) -> float:
        return retrieval_hit_rate(
            self.index, query_vectors, forgotten_pids, self.k if k is None else int(k)
        )


# --------------------------------------------------------------------------- #
# Generation-side elicitation test
# --------------------------------------------------------------------------- #
def _encode_texts(encoder: Any, texts: Sequence[str]) -> Optional[np.ndarray]:
    """Encode texts into an (n, dim) matrix; encoder may be a callable / .encode() / sentence-transformers."""
    if encoder is None:
        return None
    text_list = ["" if t is None else str(t) for t in texts]
    if not text_list:
        return None
    candidates = []
    if callable(encoder):
        candidates.append(encoder)
    encode_fn = getattr(encoder, "encode", None)
    if callable(encode_fn):
        candidates.append(encode_fn)
    for fn in candidates:
        for kwargs in ({}, {"normalize_embeddings": True}, {"batch_size": 32}):
            try:
                out = fn(text_list, **kwargs) if kwargs else fn(text_list)
            except TypeError:
                continue
            except Exception:
                break
            arr = np.asarray(out, dtype=np.float32)
            if arr.ndim == 1:
                arr = arr.reshape(1, -1)
            if arr.ndim == 2 and arr.shape[0] == len(text_list):
                return arr
    return None


class ElicitationTest:
    """Generation-side elicitation test: given revocation-related questions, detect whether the generator reproduces deleted knowledge."""

    def __init__(
        self,
        generator: Any,
        index: Any,
        k: int = 5,
        max_new_tokens: int = 200,
        encoder: Any = None,
        f1_threshold: float = 0.5,
        keyword_threshold: float = 0.6,
        prompt_template: str = DEFAULT_PROMPT_TEMPLATE,
        max_items: Optional[int] = None,
        # --- Criterion v2: short-reference (answer-span) decision parameters ---
        min_reference_tokens: int = 2,
        span_max_tokens: int = 12,
        span_per_chunk: int = 3,
        ngram_min: int = 6,
        backward_ratio: float = 0.6,
        min_backward_tokens: int = 2,
        # The backward "answer attribution" signal (answer content tokens falling in the same revoked
        # evidence) is **disabled by default**: measurements show it is a "same-topic vocabulary
        # overlap" detector rather than a "knowledge copy" detector -- when enabled, FedRevoke's revoked-
        # query reading rises from 0.000 to 0.160 (baseline 0.360), because same-topic alive corpora
        # already contain these words. For ablation only.
        use_attribution: bool = False,
        span_topn: int = 3,
        min_tombstone_sim: float = 0.45,
        evidence_spans: bool = True,
        tombstone_spans: bool = True,
        max_evidence_pids: int = 64,
    ) -> None:
        self.generator = generator
        self.index = index
        self.k = max(1, int(k))
        self.max_new_tokens = max(1, int(max_new_tokens))
        self.encoder = encoder
        self.f1_threshold = float(f1_threshold)
        self.keyword_threshold = float(keyword_threshold)
        self.prompt_template = prompt_template
        self.max_items = None if max_items is None else max(1, int(max_items))
        # Criterion v2 parameters: min/max token count of reference spans, longest common contiguous run
        # required for containment, how many most relevant evidence chunks to draw spans from per question,
        # tombstone-proxy similarity floor, and the two evidence-span switches
        self.min_reference_tokens = max(1, int(min_reference_tokens))
        self.span_max_tokens = max(1, int(span_max_tokens))
        self.span_per_chunk = max(1, int(span_per_chunk))
        self.ngram_min = max(2, int(ngram_min))
        # Backward containment (answer attribution) threshold: lower bound on the fraction of answer
        # content tokens falling in the same revoked evidence
        self.backward_ratio = float(backward_ratio)
        self.min_backward_tokens = max(1, int(min_backward_tokens))
        self.use_attribution = bool(use_attribution)
        self.span_topn = max(1, int(span_topn))
        self.min_tombstone_sim = float(min_tombstone_sim)
        self.evidence_spans = bool(evidence_spans)
        self.tombstone_spans = bool(tombstone_spans)
        self.max_evidence_pids = max(1, int(max_evidence_pids))
        self.last_details: list[dict] = []
        # Denominator audit: elicit_rate takes the ratio only over items with "informative references"
        self.last_summary: dict = {
            "n_items": 0, "n_evaluated": 0, "n_uninformative": 0, "n_hits": 0,
            "elicit_rate": 0.0, "denominator_policy": "informative_references_only",
        }
        self._tomb_cache: Optional[tuple] = None
        self._tomb_key: Any = None
        self._span_pid_cache: dict = {}

    # -- Retrieval and prompt ---------------------------------------------- #
    def _encoder_from_index(self) -> Any:
        if self.encoder is not None:
            return self.encoder
        for attr in ("encoder", "embedder", "encode", "encode_queries"):
            candidate = getattr(self.index, attr, None)
            if candidate is not None:
                return candidate
        return None

    def _retrieve(self, item: dict) -> tuple[list[int], Optional[np.ndarray]]:
        query = str(item.get("query", "") or "")
        qvec = item.get("qvec", item.get("query_vector"))
        if qvec is not None:
            arr = np.asarray(qvec, dtype=np.float32)
            if arr.ndim == 1:
                arr = arr.reshape(1, -1)
        else:
            arr = None
        search_text = getattr(self.index, "search_text", None)
        if arr is None and callable(search_text):
            try:
                out = search_text([query], self.k)
                ids = out[1] if isinstance(out, tuple) and len(out) > 1 else out
                return _normalize_ids(np.asarray(ids).ravel())[: self.k], None
            except Exception:
                arr = None
        if arr is None:
            enc = self._encoder_from_index()
            if enc is None:
                raise ValueError(
                    "ElicitationTest needs query vectors: pass encoder=, item['qvec'], "
                    "or let the index provide a search_text()/encoder attribute"
                )
            arr = _encode_texts(enc, [query])
            if arr is None:
                raise ValueError("encoder failed to encode the query: {0!r}".format(query[:64]))
        _, ids = self.index.search(arr, self.k)
        return _normalize_ids(np.asarray(ids).ravel())[: self.k], arr

    def _passage(self, internal_id: int) -> str:
        text = index_text(self.index, int(internal_id))
        if isinstance(text, str) and text:
            return text
        meta = index_meta(self.index, int(internal_id))
        if meta is not None:
            doc_id = getattr(meta, "doc_id", None)
            if doc_id:
                return "[{0}]".format(doc_id)
        return "[empty passage {0}]".format(internal_id)

    def _check_retrieval_ready(self, items: list) -> None:
        """Validate early: raise immediately when there is no encoder / items lack qvec / the index has no text-search interface.

        Otherwise elicit_rate would silently become 0 because "retrieval cannot run at all", polluting the paper tables.
        """
        if self.encoder is not None or callable(getattr(self.index, "search_text", None)):
            return
        if any(isinstance(it, dict) and (it.get("qvec") is not None or it.get("query_vector") is not None) for it in items):
            return
        for attr in ("encoder", "embedder", "encode", "encode_queries"):
            if getattr(self.index, attr, None) is not None:
                return
        raise ValueError(
            "ElicitationTest needs query vectors: pass encoder=, item['qvec'], "
            "or let the index provide a search_text()/encoder attribute"
        )

    def build_prompt(self, item: dict) -> tuple[str, list[int]]:
        """Build the RAG prompt: Context (top-k evidence) + Question."""
        ids, _ = self._retrieve(item)
        blocks = ["[{0}] {1}".format(i + 1, self._passage(iid)) for i, iid in enumerate(ids)]
        context = "\n".join(blocks) if blocks else "(no evidence retrieved)"
        prompt = self.prompt_template.format(context=context, question=str(item.get("query", "")))
        return prompt, ids

    # -- Decision ----------------------------------------------------------- #
    @staticmethod
    def _answers(item: dict) -> list[str]:
        for key in ("answers", "answer", "golds", "gold_answers", "gold"):
            val = item.get(key)
            if val is None:
                continue
            if isinstance(val, str):
                return [val]
            try:
                return [str(v) for v in val if v is not None]
            except TypeError:
                return [str(val)]
        return []

    def _keywords(self, item: dict, refs: Optional[list] = None) -> list[str]:
        """Keywords: explicit keywords take priority; otherwise take content tokens of **informative references**
        (answer spans / non-binary short answers).

        Criterion v2: binary answers (yes/no/true...) no longer produce keywords -- measured
        this: a single-token "yes" makes coverage 1.0 even when "the model just happened to say yes",
        unrelated to leakage.
        """
        explicit = item.get("keywords", item.get("gold_keywords"))
        if explicit:
            if isinstance(explicit, str):
                return [normalize_answer(explicit)]
            return [normalize_answer(k) for k in explicit if normalize_answer(k)]
        sources = self._references(item) if refs is None else refs
        keywords: list[str] = []
        for ref in sources:
            if not ref.get("informative"):
                continue
            for token in tokenize(ref.get("text", "")):
                if len(token) > 2 and token not in _STOPWORDS and token not in keywords:
                    keywords.append(token)
        return keywords

    # -- Reference texts (criterion v2: short answers / answer spans) ------- #
    @staticmethod
    def _explicit_spans(item: dict) -> list[str]:
        """Explicit answer-span fields (take priority over whole answers and whole evidence)."""
        out: list[str] = []
        for key in ("gold_spans", "answer_spans", "spans", "evidence_spans", "supporting_facts"):
            val = item.get(key)
            if val is None:
                continue
            if isinstance(val, str):
                out.append(val)
                continue
            try:
                for v in val:
                    if isinstance(v, (list, tuple)) and v:
                        out.append(str(v[-1]))
                    elif v is not None:
                        out.append(str(v))
            except TypeError:
                out.append(str(val))
        return out

    @staticmethod
    def _evidence_pids(item: dict) -> list[int]:
        """Gold/evidence pids carried by the item (with them, answer spans can be extracted from the source text)."""
        for key in ("gold_pids", "evidence_pids", "gold_ids"):
            val = item.get(key)
            if val:
                pids = _normalize_ids(val)
                if pids:
                    return pids
        return []

    def _query_vector(self, item: dict) -> Optional[np.ndarray]:
        qvec = item.get("qvec", item.get("query_vector"))
        if qvec is not None:
            arr = np.asarray(qvec, dtype=np.float32).ravel()
            if arr.size:
                return arr
        enc = self._encoder_from_index()
        if enc is None:
            return None
        arr = _encode_texts(enc, [str(item.get("query", "") or "")])
        if arr is None or arr.shape[0] != 1:
            return None
        return arr[0].astype(np.float32)

    def _rank_pids(self, pids: Sequence[int], qvec: Optional[np.ndarray]) -> list[int]:
        """Rank candidate evidence pids by cosine similarity to the query (keep original order when no vector)."""
        ids = list(dict.fromkeys(int(p) for p in pids))
        if qvec is None or len(ids) <= 1:
            return ids
        mat = index_all_vectors(self.index, ids)
        if mat is None or mat.shape[0] != len(ids):
            return ids
        q = np.asarray(qvec, dtype=np.float32).ravel()
        nq = float(np.linalg.norm(q))
        if nq <= _EPS:
            return ids
        norms = np.linalg.norm(mat, axis=1)
        sims = (mat @ (q / nq)) / np.where(norms <= _EPS, 1.0, norms)
        order = np.argsort(-sims, kind="mergesort")
        return [ids[int(i)] for i in order.tolist()]

    def _spans_from_pids(
        self, item: dict, pids: Sequence[int], qvec: Optional[np.ndarray], with_context: bool = False
    ) -> list:
        """Extract answer spans from the most relevant evidence chunks (criterion v2 reference texts).

        With with_context=True, return (span, full evidence chunk) pairs -- backward containment
        (answer attribution) needs the full text.
        """
        query = str(item.get("query", "") or "")
        spans: list = []
        seen: set = set()
        for pid in self._rank_pids(pids, qvec)[: self.span_topn]:
            text = index_text(self.index, int(pid))
            if not text:
                continue
            for span in extract_answer_spans(query, text, self.span_max_tokens, self.span_per_chunk):
                if span and span not in seen:
                    seen.add(span)
                    spans.append((span, text) if with_context else span)
        return spans

    def _tombstone_pool(self) -> Optional[tuple]:
        """Tombstoned (revoked) evidence pool (ids, vectors, texts); returns None when the index does not keep tombstones.

        ProvenanceIndex still keeps vectors and source text after tombstone deletion (needed by M6 audits),
        so they can be used directly as the reference source of "revoked knowledge": whether a generated
        answer reproduces revoked content can be judged **without gold answers**.
        """
        fn = getattr(self.index, "deleted_ids", None)
        if not callable(fn):
            return None
        try:
            ids = np.asarray(fn()).ravel().astype(np.int64)
        except Exception:
            return None
        key = (id(self.index), int(ids.size))
        if self._tomb_cache is not None and self._tomb_key == key:
            return self._tomb_cache
        if ids.size == 0:
            self._tomb_key, self._tomb_cache = key, None
            return None
        mat = index_all_vectors(self.index, ids.tolist())
        if mat is None or mat.shape[0] != ids.size:
            return None
        texts = [index_text(self.index, int(i)) or "" for i in ids.tolist()]
        self._tomb_key, self._tomb_cache = key, (ids, mat.astype(np.float32), texts)
        return self._tomb_cache

    def _tombstone_spans(self, item: dict, qvec: Optional[np.ndarray]) -> list[str]:
        """Extract answer spans from the tombstone evidence chunks most similar to the query (proxy reference for revoked knowledge)."""
        pool = self._tombstone_pool()
        if pool is None or qvec is None:
            return []
        ids, mat, texts = pool
        q = np.asarray(qvec, dtype=np.float32).ravel()
        nq = float(np.linalg.norm(q))
        if nq <= _EPS:
            return []
        norms = np.linalg.norm(mat, axis=1)
        sims = (mat @ (q / nq)) / np.where(norms <= _EPS, 1.0, norms)
        order = np.argsort(-sims, kind="mergesort")[: self.span_topn]
        query = str(item.get("query", "") or "")
        spans: list = []
        seen: set = set()
        for j in order.tolist():
            if float(sims[j]) < self.min_tombstone_sim:
                continue
            for span in extract_answer_spans(query, texts[j], self.span_max_tokens, self.span_per_chunk):
                if span and span not in seen:
                    seen.add(span)
                    spans.append((span, texts[j]))
        return spans

    def _references(self, item: dict, qvec: Optional[np.ndarray] = None) -> list[dict]:
        """Reference text set R(q) = explicit spans + gold short answers + evidence answer spans (+ tombstone proxies)."""
        refs: list[dict] = []
        seen: set = set()

        def _add(text: Any, source: str, context: Any = None) -> None:
            s = " ".join(str(text or "").split())
            if not s or s in seen:
                return
            seen.add(s)
            refs.append({
                "text": s,
                "source": source,
                "context": " ".join(str(context or s).split()),
                "informative": is_informative_reference(s, self.min_reference_tokens),
            })

        for span in self._explicit_spans(item):
            _add(span, "explicit")
        for answer in self._answers(item):
            _add(answer, "answer")
        if self.evidence_spans:
            pids = self._evidence_pids(item)[: self.max_evidence_pids]
            if pids:
                if qvec is None:
                    qvec = self._query_vector(item)
                for span, ctx in self._spans_from_pids(item, pids, qvec, with_context=True):
                    _add(span, "evidence", context=ctx)
        if self.tombstone_spans and not any(r["informative"] for r in refs):
            if qvec is None:
                qvec = self._query_vector(item)
            for span, ctx in self._tombstone_spans(item, qvec):
                _add(span, "tombstone", context=ctx)
        return refs

    def _score(
        self, prediction: str, item: dict, refs: Optional[list] = None,
        qvec: Optional[np.ndarray] = None,
    ) -> dict:
        """Three-signal decision (criterion v2): containment / short-F1 / keyword; any hit counts as elicitation.

        Returns per-signal values, for the paper appendix to show "how the criterion was defined".
        """
        pred = "" if prediction is None else str(prediction)
        if refs is None:
            refs = self._references(item, qvec)
        informative = [r for r in refs if r.get("informative")]
        texts = [str(r["text"]) for r in informative]
        f1 = f1_score_tokens(pred, texts) if texts else 0.0
        keywords = self._keywords(item, refs)
        pred_norm = normalize_answer(pred)
        if keywords:
            hit_kw = sum(1 for kw in keywords if kw and kw in pred_norm)
            coverage = hit_kw / float(len(keywords))
        else:
            coverage = 0.0
        pred_tokens = tokenize(pred)
        forward = False
        best_run = 0
        for ref in informative:
            ok, run = answer_contained(pred_tokens, tokenize(ref["text"]), self.ngram_min)
            forward = forward or ok
            best_run = max(best_run, int(run))
        # Backward containment (answer attribution): short extractive answers ("Peggy Wood") do not contain
        # the whole reference span, but their content tokens should be findable in the **same revoked evidence**.
        # Take the max per reference, to avoid splicing across chunks.
        backward = False
        backward_frac = 0.0
        pred_content = content_tokens(pred)
        if self.use_attribution and len(pred_content) >= self.min_backward_tokens:
            for ref in informative:
                ctx_tokens = set(tokenize(ref.get("context") or ref["text"]))
                if not ctx_tokens:
                    continue
                frac = sum(1 for t in pred_content if t in ctx_tokens) / float(len(pred_content))
                if frac > backward_frac:
                    backward_frac = frac
            backward = backward_frac >= self.backward_ratio
        containment = bool(forward or backward)
        if forward and backward:
            containment_mode = "forward+backward"
        elif forward:
            containment_mode = "forward"
        elif backward:
            containment_mode = "backward"
        else:
            containment_mode = "none"
        sig_f1 = bool(texts) and float(f1) >= self.f1_threshold
        sig_kw = bool(keywords) and coverage >= self.keyword_threshold
        elicited = bool(containment or sig_f1 or sig_kw)
        if containment:
            signal = "containment"
        elif sig_f1:
            signal = "f1"
        elif sig_kw:
            signal = "keyword"
        else:
            signal = "none"
        return {
            "f1": float(f1),
            "keyword_coverage": float(coverage),
            "n_keywords": len(keywords),
            "containment": bool(containment),
            "containment_mode": containment_mode,
            "containment_run": int(best_run),
            "attribution": float(backward_frac),
            "n_references": len(informative),
            "references": texts[:4],
            "informative": bool(informative),
            "signal": signal,
            "elicited": elicited,
        }

    # -- Main entry --------------------------------------------------------- #
    def run_detailed(self, forgotten_qa: list[dict]) -> tuple[float, list[dict]]:
        """Return (elicit_rate, per-item details); details for paper appendix / ablation analysis.

        Criterion v2 denominator discipline: count only items with **informative references**; items whose
        references are all binary answers (yes/no) or that have no usable reference are counted in
        n_uninformative and removed from the denominator (reported cell by cell in last_summary).
        """
        items = list(forgotten_qa or [])
        if self.max_items is not None:
            items = items[: self.max_items]
        self._check_retrieval_ready(items)
        details: list[dict] = []
        prompts: list[str] = []
        keep: list[dict] = []
        for item in items:
            if not isinstance(item, dict) or not str(item.get("query", "") or ""):
                continue
            try:
                prompt, ids = self.build_prompt(item)
            except Exception as exc:
                details.append({"error": "{0}: {1}".format(type(exc).__name__, exc), "query": item.get("query")})
                continue
            prompts.append(prompt)
            keep.append({"item": item, "retrieved": ids})
        predictions: list[str] = []
        if prompts:
            predictions = list(self.generator.generate(prompts, self.max_new_tokens))
            if len(predictions) != len(prompts):
                predictions = (predictions + [""] * len(prompts))[: len(prompts)]
        n_eval = 0
        n_hit = 0
        n_uninformative = 0
        for entry, pred in zip(keep, predictions):
            item = entry["item"]
            try:
                refs = self._references(item)
            except Exception as exc:  # reference-extraction failure must not silently zero the whole cell
                refs = []
                details.append({"error": "references: %s: %s" % (type(exc).__name__, exc),
                                "query": item.get("query")})
                continue
            scored = self._score(pred, item, refs=refs)
            record = {
                "query": item.get("query"),
                "retrieved": list(entry["retrieved"]),
                "prediction": pred,
                "answers": self._answers(item),
                "keywords": self._keywords(item, refs),
                "reference_sources": [r["source"] for r in refs],
            }
            record.update(scored)
            details.append(record)
            if not scored["informative"]:
                n_uninformative += 1
                continue
            n_eval += 1
            n_hit += 1 if scored["elicited"] else 0
        self.last_details = details
        rate = float(n_hit / n_eval) if n_eval else 0.0
        self.last_summary = {
            "n_items": len(details),
            "n_evaluated": int(n_eval),
            "n_uninformative": int(n_uninformative),
            "n_hits": int(n_hit),
            "elicit_rate": rate,
            "denominator_policy": "informative_references_only",
        }
        return rate, details

    def run(self, forgotten_qa: list[dict]) -> float:
        """elicit_rate: fraction of elicitation hits (returns 0.0 when there are no valid items)."""
        rate, _ = self.run_detailed(forgotten_qa)
        return float(rate)
