"""baselines.py — unified interface for 5 baselines (INTERFACES.md Section 9).

Contract::

    class Baseline(Protocol):
        name: str
        def run(self, index, doc_ids, **kw) -> dict
        # uniformly returns {"forget": {...}, "utility": {...}, "cost": {...}}

Baseline list
------------------------
| Class | Name | Behavior | Training |
|---|---|---|---|
| FullRebuild  | full_rebuild  | Fully rebuild the ANN structure after deleting all requested doc shards (correctness upper bound, cost reference) | none |
| NaiveDelete  | naive_delete  | Delete only the hit shards in the local silo (cross-silo semantic shadows survive) — counterexample to Claim 1 | none |
| SISA         | sisa          | Sharded deletion + rebuild only the affected shards | none (rebuild) |
| LoRAFinetune | lora_finetune | LoRA-style low-rank forgetting finetune of the encoder, then re-encode + rebuild | light |
| TDSCAdapter  | tdsc_adapter  | wang2024whenmachine behavioral forgetting adapted to federated: rewrite the knowledge base (text + vector decorrelation), no document deletion, no model change | none |

Environment degradation (must be stated in the report)
----------------------------
* The local .venv has no transformers / peft / sentence-transformers installed, and bge-small
  weights are not local: real LoRA finetuning cannot be executed. LoRAFinetune therefore degrades
  to an "encoder finetune head only": train a LoRA-style low-rank adapter dW = A*B in representation
  space (torch, CPU or GPU), with objective = retention term (unrelated documents/queries keep
  self-similarity) + separation term (forgotten documents move away from their surviving shadow
  copies); after training, re-encode the whole corpus with the adapter and rebuild the index.
  The degradation is recorded in result["meta"]["degraded"] and degradation_reason; once the
  dependencies are in place (peft + local bge-small), the real LoRA branch is used
  (_real_lora_available()).
* The index backend is a single ANN (no faiss/hnswlib on this machine; index_core automatically
  falls back to numpy): SISA's "shard rebuild" is accounted by the vector volume of the affected
  shards (optimistic lower bound), while the real wall-clock time is a full-corpus compact; this
  difference is recorded in cost["cost_model"].

Unified kw conventions (run_experiments and unit tests both pass arguments this way)
------------------------------------------------
    query_sample      : (nq, dim) query vectors
    gold_pids         : list[list[int]] internal-id gold aligned with query_sample
    qa_items          : list[dict] generation-side elicitation test items (query/answers/qvec)
    generator         : generator (MockGenerator / HFGenerator), may be None
    shadow_surrogates : {doc_id: [surrogate_doc_id, ...]} controlled cross-silo shadow mapping
    client_id         : requester silo (used by NaiveDelete); defaults to the seed-majority silo
    k                 : retrieval depth (default 10)
    seed              : random seed
    in_place          : when True, mutate the passed index directly (default False: clone first, so methods stay comparable)
    text_store        : {pid: text} writable text store (used by TDSCAdapter to rewrite the knowledge base)
"""

from __future__ import annotations

import importlib.util
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Optional, Sequence

import numpy as np

from .metrics import (
    CostMeter,
    exact_match,
    f1_score_tokens,
    fairness_stats,
    faithfulness,
    ndcg_at_k,
    recall_at_k,
)
from .repair import index_all_vectors, index_meta
from .verify import ElicitationTest, MIAProbe, retrieval_hit_rate, rho_hat_residual

try:
    from .config import DIM as _DIM, SEED as _CONFIG_SEED  # type: ignore
except Exception:  # pragma: no cover
    _DIM, _CONFIG_SEED = 384, 20260214

SEED: int = int(_CONFIG_SEED)
_EPS = 1e-12
NAN_F = float("nan")

# Closed-book (empty-context) template: the criterion exactly matches the retrieval tier; only no evidence is given
CLOSED_BOOK_TEMPLATE = "Context:\n(no evidence retrieved)\nQuestion: {question}\nAnswer:"

__all__ = [
    "Baseline",
    "FullRebuild",
    "NaiveDelete",
    "SISA",
    "LoRAFinetune",
    "TDSCAdapter",
    "RandomReplica",
    "FullRebuildReencode",
    "BASELINES",
    "build_baseline",
    "available_baselines",
    "clone_index",
    "EvalContext",
    "build_eval_context",
    "evaluate_utility",
    "evaluate_forget",
    "evaluate_delta_utility",
    "unrel_query_indices",
    "subset_context",
    "rho_hat_tv",
    "surrogate_ids",
    "residual_stats",
    "SEED",
]


# ======================================================================================
# General utilities
# ======================================================================================
def _unique_ints(values: Any) -> list:
    out, seen = [], set()
    if values is None:
        return out
    if isinstance(values, (int, np.integer)):
        return [int(values)]
    for v in values:
        try:
            i = int(v)
        except (TypeError, ValueError):
            continue
        if i in seen:
            continue
        seen.add(i)
        out.append(i)
    return out


def _as_query_matrix(values: Any) -> np.ndarray:
    if values is None:
        return np.zeros((0, 0), dtype=np.float32)
    arr = np.asarray(values, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    return arr if arr.ndim == 2 else np.zeros((0, 0), dtype=np.float32)


def _normalize_rows(mat: np.ndarray) -> np.ndarray:
    if mat.size == 0:
        return mat
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms = np.where(norms <= _EPS, 1.0, norms)
    return (mat / norms).astype(np.float32)


_CLONE_ATTRS = ("text_for_id", "get_text", "text", "encoder", "embedder", "search_text", "score_calibrator")


def clone_index(index: Any, backend: Optional[str] = None) -> Any:
    """Clone the index: internal ids and tombstone state are kept exactly consistent (full add first, then remove by tombstones).

    This way every baseline starts from exactly the same initial state, and the doc_id -> internal id mapping is unchanged.
    """
    stats = index.stats() if callable(getattr(index, "stats", None)) else {}
    n_total = int(stats.get("n_vectors", 0))
    dim = int(getattr(index, "dim", stats.get("dim", _DIM)) or _DIM)
    be = backend or str(getattr(index, "backend", "numpy") or "numpy")
    new = type(index)(dim, backend=be)
    if n_total:
        ids = list(range(n_total))
        vecs = index_all_vectors(index, ids)
        metas = list(index.metas_snapshot())
        if vecs is None or len(metas) != n_total:
            raise RuntimeError("clone_index: cannot fetch full vectors/metadata; index type %r does not support cloning" % type(index).__name__)
        new.add(vecs, metas)
        try:
            deleted = _unique_ints(index.deleted_ids())
        except Exception:
            deleted = []
        if deleted:
            new.remove(deleted)
    for attr in _CLONE_ATTRS:
        if hasattr(index, attr):
            try:
                setattr(new, attr, getattr(index, attr))
            except Exception:
                pass
    return new


def _resolve_doc_ids(index: Any, doc_ids: Sequence[str]) -> dict:
    """{doc_id: [alive internal ids]}."""
    out: dict = {}
    fn = getattr(index, "ids_for_doc", None)
    for d in doc_ids or ():
        did = str(d)
        ids = []
        if callable(fn):
            try:
                ids = _unique_ints(fn(did))
            except Exception:
                ids = []
        out[did] = ids
    return out


def _client_of(index: Any, internal_ids: Sequence[int]) -> Optional[str]:
    counts: dict = {}
    for i in internal_ids:
        try:
            cid = str(index.meta(int(i)).client_id)
        except Exception:
            continue
        counts[cid] = counts.get(cid, 0) + 1
    if not counts:
        return None
    return max(counts.items(), key=lambda kv: (kv[1], kv[0]))[0]


def _rewrite_vectors(index: Any, ids: Sequence[int], vecs: np.ndarray) -> int:
    """Rewrite vectors in place (used by TDSCAdapter / encoder-adapter re-encoding); returns the number of successes."""
    ids = _unique_ints(ids)
    if not ids or vecs is None or len(vecs) != len(ids):
        return 0
    vecs = _normalize_rows(np.asarray(vecs, dtype=np.float32))
    mat = getattr(index, "_vectors", None)
    if isinstance(mat, np.ndarray) and mat.ndim == 2 and int(max(ids)) < mat.shape[0]:
        mat[np.asarray(ids, dtype=np.int64)] = vecs
        try:
            index._alive_cache = None
        except Exception:
            pass
        return len(ids)
    setter = getattr(index, "set_vector", None)
    if callable(setter):
        n = 0
        for i, v in zip(ids, vecs):
            try:
                setter(int(i), v)
                n += 1
            except Exception:
                continue
        return n
    return 0


def _compact(index: Any) -> None:
    fn = getattr(index, "compact", None)
    if callable(fn):
        try:
            fn()
        except Exception:
            pass


# ======================================================================================
# Evaluation context (all methods share the same metric conventions)
# ======================================================================================
@dataclass
class EvalContext:
    query_sample: np.ndarray = field(default_factory=lambda: np.zeros((0, 0), dtype=np.float32))
    gold_pids: list = field(default_factory=list)
    qa_items: list = field(default_factory=list)
    generator: Any = None
    k: int = 10
    seed: int = SEED
    shadow_surrogates: dict = field(default_factory=dict)
    control_pids: Optional[list] = None
    text_store: Any = None
    families: list = field(default_factory=list)      # per-query source label (official / synthetic_title / ...)
    primary_metric: str = "recall10"                  # datasets with very large gold sets (e.g. DS3) should use ndcg10 instead
    closed_book: bool = False                         # additionally run one closed-book (empty-context) generation, giving the elicit floor

    @property
    def n_queries(self) -> int:
        return int(self.query_sample.shape[0]) if self.query_sample.size else 0


def build_eval_context(index: Any, doc_ids: Sequence[str] = (), **kw: Any) -> EvalContext:
    """Build an evaluation context from kw (when gold is missing, sample randomly from the alive pool so all methods share the same convention)."""
    ctx = EvalContext(
        query_sample=_as_query_matrix(kw.get("query_sample")),
        gold_pids=[_unique_ints(row) for row in (kw.get("gold_pids") or [])],
        qa_items=list(kw.get("qa_items") or []),
        generator=kw.get("generator"),
        k=max(1, int(kw.get("k", 10) or 10)),
        seed=int(kw.get("seed", SEED) or SEED),
        shadow_surrogates=dict(kw.get("shadow_surrogates") or {}),
        control_pids=list(kw["control_pids"]) if kw.get("control_pids") else None,
        text_store=kw.get("text_store"),
        families=list(kw.get("query_families") or []),
        primary_metric=str(kw.get("primary_metric") or "recall10"),
        closed_book=bool(kw.get("closed_book", False)),
    )
    if not ctx.gold_pids and ctx.n_queries:
        try:
            alive = np.asarray(index.alive_ids()).ravel()
        except Exception:
            alive = np.zeros(0, dtype=np.int64)
        if alive.size:
            rng = np.random.default_rng(ctx.seed)
            pool = [int(a) for a in alive.tolist()]
            kk = min(ctx.k, len(pool))
            ctx.gold_pids = [
                [pool[int(j)] for j in np.sort(rng.choice(len(pool), size=kk, replace=False))]
                for _ in range(ctx.n_queries)
            ]
    return ctx


def surrogate_ids(index: Any, doc_ids: Sequence[str], ctx: EvalContext) -> dict:
    """{revoked doc_id: [alive shadow-surrogate internal ids]} (for retrieval-side residue readings)."""
    out: dict = {}
    for did in doc_ids or ():
        did = str(did)
        surrogates = ctx.shadow_surrogates.get(did) or []
        if isinstance(surrogates, str):
            surrogates = [surrogates]
        ids: list = []
        for s in surrogates:
            ids.extend(_resolve_doc_ids(index, [str(s)]).get(str(s), []))
        if ids:
            out[did] = sorted(set(ids))
    return out


def residual_stats(index: Any, doc_ids: Sequence[str], ctx: EvalContext) -> dict:
    """Fraction of revoked documents that "still have surviving cross-silo shadow copies" (E6's main metric; deterministic, no query noise).

    residual_doc_rate       = n_surviving / n_forgotten_docs (all revoked documents)
    residual_doc_rate_cond  = n_surviving / n_shadowed_forgotten (only shadowed revoked documents)
    """
    smap = ctx.shadow_surrogates or {}
    n_shadowed = 0
    n_surv = 0
    total_alive_surrogates = 0
    for did in doc_ids or ():
        surs = smap.get(str(did)) or []
        if isinstance(surs, str):
            surs = [surs]
        if not surs:
            continue
        n_shadowed += 1
        alive = 0
        for s in surs:
            try:
                alive += len(index.ids_for_doc(str(s)))
            except Exception:
                continue
        total_alive_surrogates += int(alive)
        if alive:
            n_surv += 1
    n_docs = len(list(doc_ids or ()))
    return {
        "residual_doc_rate": float(n_surv / max(1, n_docs)),
        "residual_doc_rate_cond": float(n_surv / n_shadowed) if n_shadowed else NAN_F,
        "n_shadowed_forgotten": int(n_shadowed),
        "n_forgotten_docs": int(n_docs),
        "n_surviving_surrogate_docs": int(n_surv),
        "n_surviving_surrogates": int(total_alive_surrogates),
    }


def _passage_text(index: Any, internal_id: int, ctx: EvalContext) -> str:
    if ctx.text_store is not None:
        try:
            val = ctx.text_store.get(int(internal_id))
            if isinstance(val, str) and val:
                return val
        except Exception:
            pass
    fn = getattr(index, "text_for_id", None)
    if callable(fn):
        try:
            val = fn(int(internal_id))
            if isinstance(val, str) and val:
                return val
        except Exception:
            pass
    meta = index_meta(index, internal_id)
    did = getattr(meta, "doc_id", None)
    return "[%s]" % did if did else "[empty passage %d]" % internal_id


def evaluate_utility(ctx: EvalContext, index: Any, generator: Any = None, max_new_tokens: int = 64) -> dict:
    """Retrieval and generation-side utility: recall@k / nDCG@k / EM / F1 (+ fairness stats of per-silo recall)."""
    out = {
        "recall10": 0.0,
        "ndcg10": 0.0,
        "em": 0.0,
        "f1": 0.0,
        "n_queries": ctx.n_queries,
        "per_client_recall": {},
        "recall_min": 0.0,
        "recall_std": 0.0,
    }
    if index is None or not ctx.n_queries:
        return out
    try:
        _, ids = index.search(ctx.query_sample, ctx.k)
    except Exception as exc:
        out["error"] = "%s: %s" % (type(exc).__name__, exc)
        return out
    ids = np.asarray(ids)
    recalls, ndcgs, per_client = [], [], {}
    for row, gold in zip(ids.tolist(), ctx.gold_pids):
        retrieved = [int(v) for v in row if int(v) >= 0]
        recalls.append(recall_at_k(retrieved, gold, ctx.k))
        ndcgs.append(ndcg_at_k(retrieved, gold, ctx.k))
        cid = None
        try:
            cid = str(index.meta(retrieved[0]).client_id) if retrieved else None
        except Exception:
            cid = None
        if cid:
            per_client.setdefault(cid, []).append(recalls[-1])
    out["recall10"] = float(np.mean(recalls)) if recalls else 0.0
    out["ndcg10"] = float(np.mean(ndcgs)) if ndcgs else 0.0
    out["per_client_recall"] = {k: float(np.mean(v)) for k, v in sorted(per_client.items())}
    fair = fairness_stats(list(out["per_client_recall"].values())) if out["per_client_recall"] else {}
    out["recall_min"] = float(fair.get("min", 0.0))
    out["recall_std"] = float(fair.get("std", 0.0))

    # ---- Query-family split (DS3: 50 official vs 450 synthetic queries must be reported separately) ----
    fams: Dict[str, Dict[str, float]] = {}
    if ctx.families and len(ctx.families) >= len(recalls):
        buckets: Dict[str, List[int]] = {}
        for i, lab in enumerate(ctx.families[: len(recalls)]):
            buckets.setdefault(str(lab), []).append(i)
        for lab, idxs in sorted(buckets.items()):
            fams[lab] = {
                "n": len(idxs),
                "recall10": float(np.mean([recalls[i] for i in idxs])),
                "ndcg10": float(np.mean([ndcgs[i] for i in idxs])),
            }
    out["families"] = fams
    out["primary_metric"] = ctx.primary_metric
    out["primary_value"] = float(out.get(ctx.primary_metric, 0.0) or 0.0)

    gen = generator if generator is not None else ctx.generator
    if gen is not None and ctx.qa_items:
        ems, f1s, faiths = [], [], []
        # Critical fix (misaligned-pairing bug): each qa_item must use the retrieval results of
        # **its own query**. The old implementation used zip(qa_items, ids), pairing qa_item[k] with the
        # top-k of query_sample[k]; but qa_items were selected from queries "whose gold hits a revoked
        # document", so their indices are unrelated to query_sample -> context mismatch -> em/f1/faithfulness
        # structurally 0. Now the row is taken via item["row"].
        for pos, item in enumerate(ctx.qa_items):
            qrow = None
            r = item.get("row")
            if isinstance(r, int) and 0 <= r < len(ids):
                qrow = ids[r]
            elif pos < len(ids):
                qrow = ids[pos]  # fallback: keep the old behavior when there is no row field (and make it visible in meta)
            if qrow is None:
                continue
            row = np.asarray(qrow).ravel().tolist()
            passages = [_passage_text(index, int(v), ctx) for v in row if int(v) >= 0]
            prompt = "Context:\n%s\nQuestion: %s\nAnswer:" % (
                "\n".join("[%d] %s" % (n + 1, p) for n, p in enumerate(passages)),
                str(item.get("query", "")),
            )
            try:
                pred = gen.generate([prompt], max_new_tokens)[0]
            except Exception:
                pred = ""
            golds = [str(a) for a in (item.get("answers") or [])]
            ems.append(exact_match(pred, golds))
            f1s.append(f1_score_tokens(pred, golds))
            faiths.append(faithfulness(pred, passages) if passages else 0.0)
        out["em"] = float(np.mean(ems)) if ems else 0.0
        out["f1"] = float(np.mean(f1s)) if f1s else 0.0
        out["faithfulness"] = float(np.mean(faiths)) if faiths else NAN_F
        out["faithfulness_measured"] = bool(faiths)
        out["n_qa_items"] = int(len(ctx.qa_items))
        out["qa_pairing"] = "row-indexed" if any(isinstance(it.get("row"), int) for it in ctx.qa_items) else "positional-fallback"
    return out


def subset_context(ctx: EvalContext, idx: Sequence[int]) -> EvalContext:
    """Build a new evaluation context from a subset of query indices (for Q_unrel)."""
    idx = [int(i) for i in idx]
    q = ctx.query_sample[idx] if ctx.n_queries else ctx.query_sample
    gold = [ctx.gold_pids[i] for i in idx if i < len(ctx.gold_pids)]
    # qa_items are remapped by "original query index row" to new indices within the subset; items outside the subset are dropped
    newpos = {int(orig): new for new, orig in enumerate(idx)}
    qa = []
    for it in ctx.qa_items:
        r = it.get("row")
        if r is None:
            continue
        if int(r) in newpos:
            it2 = dict(it)
            it2["row"] = int(newpos[int(r)])
            qa.append(it2)
    fams = [ctx.families[i] for i in idx if i < len(ctx.families)]
    return EvalContext(
        query_sample=q, gold_pids=gold, qa_items=qa, generator=ctx.generator, k=ctx.k,
        seed=ctx.seed, shadow_surrogates=ctx.shadow_surrogates, control_pids=ctx.control_pids,
        text_store=ctx.text_store, families=fams, primary_metric=ctx.primary_metric,
        closed_book=ctx.closed_book,
    )


def pid_to_internal(index: Any) -> Dict[int, int]:
    """{pid: internal_id} (to put gold_pids and the deletion closure in the same coordinate system)."""
    out: Dict[int, int] = {}
    snap = getattr(index, "metas_snapshot", None)
    if callable(snap):
        try:
            for i, m in enumerate(snap()):
                out[int(getattr(m, "pid", i))] = i
        except Exception:
            pass
    return out


def unrel_query_indices(index: Any, ctx: EvalContext, deleted_ids: Sequence[int]) -> List[int]:
    """Q_unrel = { q : gold_pids(q) intersect deletion_closure = empty } (INTERFACES evaluation convention)."""
    deleted = {int(i) for i in (deleted_ids or [])}
    if not deleted:
        return list(range(len(ctx.gold_pids)))
    p2i = pid_to_internal(index)
    out: List[int] = []
    for qi, gold in enumerate(ctx.gold_pids):
        g = {int(p) for p in (gold or [])}
        if p2i:
            g = {int(p2i[p]) for p in g if p in p2i}
        if not (g & deleted):
            out.append(qi)
    return out


def evaluate_delta_utility(
    index_before: Any,
    index_after: Any,
    ctx: EvalContext,
    deleted_ids: Sequence[int],
    min_n: int = 50,
    generator: Any = None,
) -> Dict[str, Any]:
    """Delta_util: evaluate utility only on Q_unrel (gold disjoint from the deletion closure), and report before/after differences.

    Also reports |Q_unrel| and a reliability flag (when |Q_unrel| < min_n, reliable=False; the caller/paper must not directly cite that cell).
    """
    out: Dict[str, Any] = {
        "n_unrel": 0, "unrel_reliable": False,
        "recall10_unrel": NAN_F, "ndcg10_unrel": NAN_F,
        "recall10_unrel_before": NAN_F, "ndcg10_unrel_before": NAN_F,
        "delta_recall10_unrel": NAN_F, "delta_ndcg10_unrel": NAN_F,
    }
    if index_after is None or not ctx.n_queries:
        return out
    idx = unrel_query_indices(index_after, ctx, deleted_ids)
    out["n_unrel"] = len(idx)
    if not idx:
        return out
    out["unrel_reliable"] = bool(len(idx) >= int(min_n))
    sub = subset_context(ctx, idx)
    after = evaluate_utility(sub, index_after, generator=generator)
    out["recall10_unrel"] = float(after.get("recall10", NAN_F))
    out["ndcg10_unrel"] = float(after.get("ndcg10", NAN_F))
    if index_before is not None:
        before = evaluate_utility(sub, index_before, generator=generator)
        out["recall10_unrel_before"] = float(before.get("recall10", NAN_F))
        out["ndcg10_unrel_before"] = float(before.get("ndcg10", NAN_F))
        out["delta_recall10_unrel"] = float(out["recall10_unrel"] - out["recall10_unrel_before"])
        out["delta_ndcg10_unrel"] = float(out["ndcg10_unrel"] - out["ndcg10_unrel_before"])
    return out


def _row_pids(index: Any, ids_row: Any) -> set:
    """Map a row of internal ids to stable pids (comparable across rebuilt indices)."""
    out: set = set()
    for x in np.asarray(ids_row).ravel():
        xi = int(x)
        if xi < 0:
            continue
        try:
            out.add(int(index.meta(xi).pid))
        except Exception:
            continue
    return out


def _exact_topk_pids(index: Any, queries: Any, k: int) -> List[set]:
    """Run exact top-k over the index's alive vectors and return the pid set per query.

    Exact k-NN is used instead of ANN results so rho_hat_ret reflects only content differences
    of the "alive vector set", unpolluted by IVF approximation error.
    """
    q = np.asarray(queries, dtype=np.float32)
    if q.ndim == 1:
        q = q.reshape(1, -1)
    try:
        ids_all = index.ids_all() if hasattr(index, "ids_all") else None
    except Exception:
        ids_all = None
    if ids_all is None:
        try:
            ids_all = [i for i in range(index._n_total) if i not in set(index.deleted_ids())]
        except Exception:
            return [set() for _ in range(q.shape[0])]
    alive = [int(i) for i in ids_all]
    if not alive:
        return [set() for _ in range(q.shape[0])]
    try:
        mat = np.ascontiguousarray(index._vectors[alive], dtype=np.float32)
    except Exception:
        return [set() for _ in range(q.shape[0])]
    # After L2 normalization, inner product = cosine
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    mat = mat / norms
    qn = np.linalg.norm(q, axis=1, keepdims=True)
    qn[qn == 0] = 1.0
    q = q / qn
    sims = q @ mat.T  # (nq, n_alive)
    kk = int(min(k, len(alive)))
    top_idx = np.argpartition(-sims, kk - 1, axis=1)[:, :kk]
    out: List[set] = []
    for row in top_idx:
        pids = set()
        for j in row:
            try:
                pids.add(int(index.meta(alive[int(j)]).pid))
            except Exception:
                continue
        out.append(pids)
    return out


def rho_hat_tv(index_oracle: Any, index_after: Any, ctx: EvalContext) -> float:
    """rho_hat_ret (paper Eq. 18): max_q [1 - |R_k(q;I') intersect R_k(q;I_empty)| / k].

    I_empty is the oracle index where "the revoked knowledge was never indexed" (fully rebuilt after
    removing seeds and their injected shadows). Both sides use exact k-NN and intersect by pid, so ANN
    approximation error does not pollute the content-level comparison.
    """
    if index_oracle is None or index_after is None or ctx.query_sample is None:
        return NAN_F
    try:
        lists_a = _exact_topk_pids(index_after, ctx.query_sample, ctx.k)
        lists_o = _exact_topk_pids(index_oracle, ctx.query_sample, ctx.k)
    except Exception:
        return NAN_F
    if not lists_a or not lists_o:
        return NAN_F
    k = max(1, int(ctx.k))
    worst = 0.0
    for a, o in zip(lists_a, lists_o):
        if not a and not o:
            continue
        tv = 1.0 - (len(a & o) / float(k))
        if tv > worst:
            worst = tv
    return float(worst)


def evaluate_forget(
    ctx: EvalContext,
    index: Any,
    forgotten_ids: Sequence[int],
    forgotten_doc_ids: Sequence[str] = (),
    generator: Any = None,
    elicit_k: int = 5,
) -> dict:
    """Forgetting quality: hit_rate / surrogate_hit_rate / mia_auc / elicit_rate / rho_hat.

    Note: index_core guarantees search() never returns deleted ids, so hit_rate is structurally 0 for
    "deletion-style" methods; what distinguishes methods is surrogate_hit_rate (can shadows still be
    recalled) and elicit_rate (can the generation side still elicit revoked knowledge).
    """
    forgotten = _unique_ints(forgotten_ids)
    out = {
        "hit_rate": 0.0,
        "surrogate_hit_rate": 0.0,
        "residual_doc_rate": NAN_F,
        "residual_doc_rate_cond": NAN_F,
        "n_shadowed_forgotten": 0,
        "n_forgotten_docs": len(list(forgotten_doc_ids or ())),
        "n_surrogates": 0,
        "mia_auc": 0.5,
        "elicit_rate": 0.0,
        "elicit_measured": False,
        "rho_hat": 0.0,
        "n_probes": len(forgotten),
    }
    if index is None:
        return out
    if ctx.n_queries and forgotten:
        try:
            out["hit_rate"] = float(retrieval_hit_rate(index, ctx.query_sample, forgotten, ctx.k))
        except Exception:
            pass

    sur = surrogate_ids(index, forgotten_doc_ids, ctx)
    flat = sorted({i for ids in sur.values() for i in ids})
    out["n_surrogates"] = len(flat)
    if ctx.n_queries and flat:
        try:
            out["surrogate_hit_rate"] = float(retrieval_hit_rate(index, ctx.query_sample, flat, ctx.k))
        except Exception:
            pass

    control = ctx.control_pids
    if control is None:
        try:
            alive = [int(a) for a in np.asarray(index.alive_ids()).ravel().tolist()]
        except Exception:
            alive = []
        banned = set(forgotten) | set(flat)
        pool = [a for a in alive if a not in banned]
        if pool:
            rng = np.random.default_rng(ctx.seed)
            n = min(len(pool), 512)
            control = [pool[int(j)] for j in np.sort(rng.choice(len(pool), size=n, replace=False))]
    if forgotten and control:
        try:
            probe = MIAProbe(index, None, seed=ctx.seed, k=ctx.k, exclude_self=True)
            out["mia_auc"] = float(probe.auc(forgotten, control))
        except Exception:
            pass

    out.update(residual_stats(index, forgotten_doc_ids, ctx))

    gen = generator if generator is not None else ctx.generator
    if gen is not None and ctx.qa_items:
        try:
            test = ElicitationTest(gen, index, k=elicit_k, max_new_tokens=64)
            rate, details = test.run_detailed(ctx.qa_items)
            out["elicit_rate"] = float(rate)
            out["elicit_measured"] = True
            # Per-item hit flags (same convention and item order as FedRevoke -> paired McNemar)
            out["elicit_flags"] = [1 if d.get("elicited") else 0 for d in details if "elicited" in d]
            if ctx.closed_book:
                # Closed-book floor: same items, same criterion, but the prompt contains no retrieved evidence
                cb = ElicitationTest(gen, index, k=elicit_k, max_new_tokens=64,
                                     prompt_template=CLOSED_BOOK_TEMPLATE)
                cb_rate, cb_details = cb.run_detailed(ctx.qa_items)
                out["elicit_rate_closed_book"] = float(cb_rate)
                out["elicit_flags_closed_book"] = [1 if d.get("elicited") else 0
                                                   for d in cb_details if "elicited" in d]
        except Exception as exc:
            out["elicit_error"] = "%s: %s" % (type(exc).__name__, exc)
    out["rho_hat"] = float(rho_hat_residual(out["hit_rate"], out["elicit_rate"]))
    return out


# ======================================================================================
# Baseline base class
# ======================================================================================
class Baseline:
    """Common baseline skeleton: clone index -> run method body -> unified evaluation -> unified cost accounting."""

    name: str = "baseline"
    display_name: str = "Baseline"

    # ---- Subclass implementation ------------------------------------------ #
    def _run(self, index: Any, ctx: EvalContext, info: dict, meter: CostMeter) -> dict:
        raise NotImplementedError

    # ---- Common entry ------------------------------------------------------ #
    def run(self, index: Any, doc_ids: Sequence[str], **kw: Any) -> dict:
        doc_ids = [str(d) for d in (doc_ids or []) if d is not None and str(d) != ""]
        ctx = build_eval_context(index, doc_ids, **kw)
        index_before = kw.get("index_before") or index  # unmutated reference index (for Q_unrel / Delta_util / rho_hat)
        in_place = bool(kw.get("in_place", False))
        t0 = time.perf_counter()
        try:
            work = index if in_place else clone_index(index)
        except Exception as exc:
            work = index
            kw = dict(kw)
            kw["clone_error"] = "%s: %s" % (type(exc).__name__, exc)

        seed_map = _resolve_doc_ids(work, doc_ids)
        seed_ids = sorted({i for ids in seed_map.values() for i in ids})
        requester = kw.get("client_id") or _client_of(work, seed_ids)
        info = {
            "method": self.name,
            "doc_ids": doc_ids,
            "seed_ids": seed_ids,
            "seed_map": {k: list(v) for k, v in seed_map.items()},
            "client_id": requester,
            "n_seed": len(seed_ids),
            # INTERFACES Section 11: near-duplicate-detection baselines must get the same signature matrix as the method
            "signatures": kw.get("signatures"),
            "dataset_name": kw.get("dataset_name"),
            "n_docs_full": kw.get("n_docs_full"),
            "lsh_threshold": float(kw.get("lsh_threshold", 0.80) or 0.80),
            "sim_threshold": float(kw.get("sim_threshold", 0.92) or 0.92),
            "knn_k": int(kw.get("knn_k", 50) or 50),
        }

        meter = CostMeter(self.name, measure_vram=True)
        error = ""
        with meter:
            try:
                detail = self._run(work, ctx, info, meter) or {}
            except Exception as exc:
                detail = {}
                error = "%s: %s" % (type(exc).__name__, exc)
        info.update(detail.get("meta", {}))
        if error:
            info["error"] = error

        forgotten_ids = _unique_ints(detail.get("removed_ids", seed_ids))
        forgotten_docs = sorted({str(work.meta(i).doc_id) for i in forgotten_ids}) if forgotten_ids else []
        utility = evaluate_utility(
            ctx, work, generator=kw.get("generator"),
            max_new_tokens=int(kw.get("max_new_tokens", 64) or 64),
        )
        # Delta_util is evaluated only on Q_unrel (the abstract's promise of "unrelated-query Recall@10 drop < 2%")
        try:
            prev_deleted = set(_unique_ints(index_before.deleted_ids())) if index_before is not None else set()
        except Exception:
            prev_deleted = set()
        try:
            now_deleted = set(_unique_ints(work.deleted_ids()))
        except Exception:
            now_deleted = set()
        deleted_by_run = sorted(now_deleted - prev_deleted)
        utility.update(evaluate_delta_utility(index_before, work, ctx, deleted_by_run,
                                              generator=kw.get("generator")))
        utility["rho_hat_tv"] = rho_hat_tv(kw.get("index_oracle") or index_before, work, ctx)
        utility["n_deleted_by_run"] = len(deleted_by_run)
        # Note: residue metrics (residual_doc_rate / surrogate_hit_rate) must be counted over the
        # **request set**; otherwise methods like NaiveDelete that "only delete local shards" would
        # underestimate residue because undeleted documents are excluded.
        forget = evaluate_forget(
            ctx, work, forgotten_ids, doc_ids, generator=kw.get("generator"),
            elicit_k=int(kw.get("elicit_k", 5) or 5),
        )
        # Unified convention: rho_hat = max(rho_hat_ret, elicit); conservative bound = max(hit, elicit)
        try:
            _er = float(forget.get("elicit_rate", 0.0) or 0.0)
        except Exception:
            _er = 0.0
        try:
            _hr = float(forget.get("hit_rate", 0.0) or 0.0)
        except Exception:
            _hr = 0.0
        _rret = float(utility.get("rho_hat_tv", NAN_F) or NAN_F)
        forget["rho_hat_ret"] = _rret
        if _rret == _rret and _er == _er:
            forget["rho_hat"] = float(max(_rret, _er))
        forget["rho_hat_bound"] = float(max(_hr, _er))
        forget["n_requested_docs"] = len(doc_ids)
        forget["n_removed_docs"] = len(forgotten_docs)

        cost = meter.as_dict()
        cost.update(
            {
                "reindex_seconds": float(detail.get("reindex_seconds", meter.wall_time)),
                "total_seconds": float(time.perf_counter() - t0),
                "bytes_transferred": int(detail.get("bytes_transferred", meter.bytes_transferred)),
                "n_vectors_touched": int(detail.get("n_vectors_touched", meter.n_vectors_touched)),
                "n_deleted": int(len(forgotten_ids)),
                "n_alive_after": int(work.stats().get("n_alive", 0)) if callable(getattr(work, "stats", None)) else 0,
                "cost_model": detail.get("cost_model", "measured"),
            }
        )

        return {
            "method": self.name,
            "display_name": self.display_name,
            "forget": forget,
            "utility": utility,
            "cost": cost,
            "meta": info,
            "index": work,  # convenient for callers to inspect further (ignored when writing CSV)
        }


# ======================================================================================
# B1 FullRebuild — correctness upper bound / cost reference
# ======================================================================================
class FullRebuild(Baseline):
    """Fully rebuild the index structure after deleting all requested shards (no shadow closure)."""

    name = "full_rebuild"
    display_name = "Full Rebuild"

    def _run(self, index: Any, ctx: EvalContext, info: dict, meter: CostMeter) -> dict:
        seed_ids = info["seed_ids"]
        n_before = int(index.stats().get("n_vectors", 0))
        dim = int(getattr(index, "dim", _DIM))
        t0 = time.perf_counter()
        if seed_ids:
            index.remove(seed_ids)
        _compact(index)
        elapsed = time.perf_counter() - t0
        n_alive = int(index.stats().get("n_alive", 0))
        meter.touch(n_alive)
        return {
            "removed_ids": seed_ids,
            "reindex_seconds": float(elapsed),
            "bytes_transferred": int(n_alive * dim * 4),
            "n_vectors_touched": int(n_alive),
            "cost_model": "full reindex: n_alive x dim x 4 bytes",
            "meta": {"n_vectors_before": n_before, "rebuild_scope": "all"},
        }


# ======================================================================================
# B2 NaiveDelete — delete only local-silo hit shards (counterexample to Claim 1)
# ======================================================================================
class NaiveDelete(Baseline):
    """Delete only shards inside the requester silo; semantic shadows in other silos stay alive."""

    name = "naive_delete"
    display_name = "Naive Delete"

    def _run(self, index: Any, ctx: EvalContext, info: dict, meter: CostMeter) -> dict:
        seed_ids = info["seed_ids"]
        client = info.get("client_id")
        local, remote = [], []
        for i in seed_ids:
            try:
                cid = str(index.meta(i).client_id)
            except Exception:
                cid = None
            (local if (client is None or cid == client) else remote).append(i)
        t0 = time.perf_counter()
        if local:
            index.remove(local)
        elapsed = time.perf_counter() - t0
        dim = int(getattr(index, "dim", _DIM))
        meter.touch(len(local))
        return {
            "removed_ids": local,
            "reindex_seconds": float(elapsed),
            "bytes_transferred": int(len(local) * dim * 4),
            "n_vectors_touched": int(len(local)),
            "cost_model": "local shard delete only",
            "meta": {
                "requester_client": client,
                "n_local_deleted": len(local),
                "n_remote_left_alive": len(remote),
                "remote_ids": remote[:64],
            },
        }


# ======================================================================================
# B3 SISA — sharded deletion + rebuild only affected shards
# ======================================================================================
class SISA(Baseline):
    """Shard by silo (or hash); after deleting the requested shards, rebuild only the affected shards."""

    name = "sisa"
    display_name = "SISA Shard"

    def __init__(self, n_shards: Optional[int] = None, shard_by: str = "client") -> None:
        self.n_shards = None if n_shards is None else max(1, int(n_shards))
        self.shard_by = str(shard_by)

    def _shard_of(self, index: Any, ids: Sequence[int], n_shards: int) -> dict:
        out: dict = {}
        for i in ids:
            if self.shard_by == "client":
                try:
                    key = str(index.meta(int(i)).client_id)
                except Exception:
                    key = "shard0"
                out[int(i)] = key
            else:
                out[int(i)] = "shard%d" % (int(i) % max(1, n_shards))
        return out

    def _run(self, index: Any, ctx: EvalContext, info: dict, meter: CostMeter) -> dict:
        seed_ids = info["seed_ids"]
        alive = _unique_ints(index.alive_ids())
        clients = set()
        for i in alive[:2000]:
            try:
                clients.add(str(index.meta(i).client_id))
            except Exception:
                continue
        n_shards = self.n_shards or max(1, len(clients))
        shard_of = self._shard_of(index, alive, n_shards)
        touched = set(shard_of.get(i, "shard0") for i in seed_ids)
        t0 = time.perf_counter()
        if seed_ids:
            index.remove(seed_ids)
        _compact(index)  # single ANN backend: actually rebuilds the whole corpus (cost accounted by affected shards)
        elapsed = time.perf_counter() - t0
        dim = int(getattr(index, "dim", _DIM))
        n_shard_vectors = int(sum(1 for i in alive if shard_of.get(i) in touched))
        meter.touch(n_shard_vectors)
        return {
            "removed_ids": seed_ids,
            "reindex_seconds": float(elapsed),
            "bytes_transferred": int(n_shard_vectors * dim * 4),
            "n_vectors_touched": int(n_shard_vectors),
            "cost_model": "shard reindex: sum(|shard|) x dim x 4 bytes (affected shards)",
            "meta": {
                "n_shards": int(len(set(shard_of.values()))),
                "touched_shards": sorted(touched)[:32],
                "shard_by": self.shard_by,
                "n_shard_vectors": n_shard_vectors,
                "note": "single ANN backend (no faiss/hnswlib); wall-clock time is a full-corpus compact; transfer/touch volumes are accounted per shard",
            },
        }


# ======================================================================================
# B4 LoRAFinetune — encoder forgetting finetune (degrades to a finetune head when dependencies are missing)
# ======================================================================================
def _real_lora_available() -> tuple:
    """Probe the real LoRA finetuning path (transformers + peft + loadable bge-small weights).

    Weight sources: local directory data/raw/models/bge-small-en-v1.5, or the HF cache (local_files_only).
    """
    missing = [m for m in ("transformers", "peft") if importlib.util.find_spec(m) is None]
    if missing:
        return False, "missing dependencies: %s" % ",".join(missing)
    try:
        import torch  # noqa: F401
        from transformers import AutoModel, AutoTokenizer
    except Exception as exc:
        return False, "transformers/torch unavailable: %s" % exc
    local = Path(__file__).resolve().parents[2] / "data" / "raw" / "models" / "bge-small-en-v1.5"
    candidates = [str(local)] if local.exists() else []
    candidates.append("BAAI/bge-small-en-v1.5")
    last = ""
    for cand in candidates:
        try:
            AutoTokenizer.from_pretrained(cand, local_files_only=True)
            AutoModel.from_pretrained(cand, local_files_only=True)
            return True, "ok:%s" % cand
        except Exception as exc:
            last = "%s: %s: %s" % (cand, type(exc).__name__, exc)
    return False, "weights unavailable (local_files_only): %s" % last


class LoRAFinetune(Baseline):
    """LoRA forgetting finetune of bge-small; degrades to a representation-space LoRA-style finetune head when dependencies are missing.

    Adapter dW = A*B (rank r), training only A/B (encoder weights frozen);
    objective = retention term (1 - cos(f(x), x)) + lambda * separation term (cos(f(x_f), f(x_s))).
    """

    name = "lora_finetune"
    display_name = "LoRA Finetune"

    def __init__(
        self,
        rank: int = 8,
        steps: int = 200,
        lr: float = 1e-4,   # standard LoRA learning rate; the early default 5e-2 collapses representations (see the hyperparameter sweep notes)
        lam_sep: float = 2.0,
        max_train: int = 512,
        device: Optional[str] = None,
    ) -> None:
        self.rank = max(1, int(rank))
        self.steps = max(1, int(steps))
        self.lr = float(lr)
        self.lam_sep = float(lam_sep)
        self.max_train = max(8, int(max_train))
        self.device = device

    # -- Train the adapter -------------------------------------------------- #
    def _train_adapter(self, index: Any, forgotten: Sequence[int], ctx: EvalContext):
        import torch

        # Pin all sources of randomness (PLAN/INTERFACES Section 0: unseeded randomness is forbidden) — including the torch global RNG
        torch.manual_seed(int(ctx.seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(ctx.seed))
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        dev = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        rng = np.random.default_rng(ctx.seed)
        alive = _unique_ints(index.alive_ids())
        forgotten_set = set(int(i) for i in forgotten)
        retained = [i for i in alive if i not in forgotten_set]
        if len(retained) > self.max_train:
            retained = [retained[int(j)] for j in np.sort(rng.choice(len(retained), self.max_train, replace=False))]
        V = index_all_vectors(index, retained)
        if V is None or V.shape[0] == 0:
            raise RuntimeError("LoRAFinetune: cannot fetch training vectors")
        X = torch.tensor(np.asarray(V, dtype=np.float32), device=dev)

        F = index_all_vectors(index, list(forgotten_set))
        pairs = []
        if F is not None and F.shape[0] and V.shape[0]:
            sims = np.asarray(V, dtype=np.float32) @ np.asarray(F, dtype=np.float32).T
            for k in range(min(F.shape[0], 64)):
                pairs.append((k, int(np.argmax(sims[:, k]))))
        XF = torch.tensor(np.asarray(F, dtype=np.float32), device=dev) if (F is not None and F.shape[0]) else None

        dim = int(X.shape[1])
        A = torch.zeros((dim, self.rank), device=dev, requires_grad=True)
        B = torch.zeros((self.rank, dim), device=dev, requires_grad=True)
        torch.nn.init.normal_(A, std=1e-3)
        opt = torch.optim.Adam([A, B], lr=self.lr)

        def cos(a, b):
            return torch.nn.functional.cosine_similarity(a, b, dim=-1, eps=1e-8)

        t0 = time.perf_counter()
        loss = loss_ret = torch.zeros((), device=dev)
        loss_sep = torch.zeros((), device=dev)
        for _ in range(self.steps):
            opt.zero_grad(set_to_none=True)
            out = X + (X @ A) @ B
            loss_ret = (1.0 - cos(out, X)).mean()
            loss_sep = torch.zeros((), device=dev)
            if XF is not None and pairs:
                oF = XF + (XF @ A) @ B
                src = oF[[p[0] for p in pairs]]
                dst = out[[p[1] for p in pairs]]
                loss_sep = cos(src, dst).mean()
            loss = loss_ret + self.lam_sep * loss_sep
            loss.backward()
            opt.step()
        secs = time.perf_counter() - t0
        with torch.no_grad():
            W = (torch.eye(dim, device=dev) + A @ B).detach().cpu().numpy().astype(np.float32)
        stats = {
            "train_seconds": float(secs),
            "final_loss": float(loss.detach().cpu().item()),
            "final_ret": float(loss_ret.detach().cpu().item()),
            "final_sep": float(loss_sep.detach().cpu().item()) if pairs else 0.0,
            "n_train": int(X.shape[0]),
            "n_pairs": int(len(pairs)),
            "rank": int(self.rank),
            "device": str(dev),
        }
        return W, stats

    # -- Real LoRA (peft + transformers + bge-small) ------------------------- #
    def _train_real_lora(self, index: Any, forgotten: Sequence[int], ctx: EvalContext):
        """Attach LoRA to bge-small attention projections and run forgetting finetune; returns (model, tokenizer, stats).

        Objective = retention term (keep cosine similarity with the frozen encoder's embeddings) +
        lambda * separation term (lower the cosine between forgotten documents and their surviving shadows).
        """
        import torch
        from transformers import AutoModel, AutoTokenizer
        from peft import LoraConfig, get_peft_model

        ok, reason = _real_lora_available()
        if not ok:
            raise RuntimeError(reason)
        model_id = reason.split("ok:", 1)[1] if reason.startswith("ok:") else "BAAI/bge-small-en-v1.5"
        torch.manual_seed(int(ctx.seed))
        dev = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        tok = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
        base = AutoModel.from_pretrained(model_id, local_files_only=True).to(dev)
        base.eval()
        for p in base.parameters():
            p.requires_grad_(False)

        alive = _unique_ints(index.alive_ids())
        forgot = _unique_ints(forgotten)
        fset = set(forgot)
        retained = [i for i in alive if i not in fset]
        rng = np.random.default_rng(int(ctx.seed))
        if len(retained) > self.max_train:
            retained = [retained[int(j)] for j in np.sort(rng.choice(len(retained), self.max_train, replace=False))]
        # Shadow pairs: (forgotten document, its surviving neighbor)
        F = index_all_vectors(index, forgot)
        pairs = []
        if F is not None and F.shape[0] and alive:
            A = index_all_vectors(index, alive)
            if A is not None and A.shape[0]:
                sims = _normalize_rows(np.asarray(A, dtype=np.float32)) @ _normalize_rows(
                    np.asarray(F, dtype=np.float32)).T
                for col in range(min(F.shape[0], 32)):
                    row = int(np.argmax(sims[:, col]))
                    if str(index.meta(alive[row]).client_id) != str(index.meta(forgot[col]).client_id):
                        pairs.append((int(forgot[col]), int(alive[row])))

        def texts_of(ids):
            out = []
            for i in ids:
                t = None
                fn = getattr(index, "text_for_id", None)
                if callable(fn):
                    try:
                        t = fn(int(i))
                    except Exception:
                        t = None
                out.append(str(t) if isinstance(t, str) and t else "[%s]" % getattr(index.meta(i), "doc_id", i))
            return out

        def encode(texts, model_):
            """Note: must not be called under no_grad (grad_fn is needed during training); call sites for frozen models wrap no_grad themselves."""
            enc = tok(list(texts), padding=True, truncation=True, max_length=256, return_tensors="pt").to(dev)
            out = model_(**enc).last_hidden_state[:, 0]
            return torch.nn.functional.normalize(out, dim=-1)

        ret_texts = texts_of(retained)
        with torch.no_grad():
            base_ret = encode(ret_texts, base)
        pair_texts = [(texts_of([a])[0], texts_of([b])[0]) for a, b in pairs[:32]]

        cfg = LoraConfig(r=int(self.rank), lora_alpha=int(self.rank) * 2, lora_dropout=0.0,
                         bias="none", target_modules=["query", "key", "value"],
                         task_type="FEATURE_EXTRACTION")
        model = get_peft_model(base, cfg).to(dev)
        for name, p in model.named_parameters():
            if "lora_" in name:
                p.requires_grad_(True)
        n_trainable = int(sum(1 for p in model.parameters() if p.requires_grad))
        if n_trainable == 0:
            raise RuntimeError("LoRA injected no trainable parameters (target_modules mismatch)")
        model.train()
        opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=self.lr)
        t0 = time.perf_counter()
        loss_val = 0.0
        for step in range(int(self.steps)):
            opt.zero_grad(set_to_none=True)
            if step % 2 == 0 or not pair_texts:
                idx = np.arange(len(ret_texts))
                if len(idx) > 32:
                    idx = np.sort(rng.choice(len(idx), 32, replace=False))
                cur = encode([ret_texts[int(i)] for i in idx], model)
                loss = (1.0 - torch.nn.functional.cosine_similarity(cur, base_ret[idx], dim=-1)).mean()
            else:
                a = torch.stack([encode([p[0]], model)[0] for p in pair_texts[:8]])
                b = torch.stack([encode([p[1]], model)[0] for p in pair_texts[:8]])
                loss = torch.nn.functional.cosine_similarity(a, b, dim=-1).mean()
            loss.backward()
            opt.step()
            loss_val = float(loss.detach().cpu().item())
        secs = time.perf_counter() - t0
        model.eval()
        stats = {"train_seconds": float(secs), "final_loss": loss_val,
                 "n_train": int(len(retained)),          # contract key (consistent with the linear-adapter branch)
                 "n_retained": int(len(retained)),
                 "n_pairs": len(pair_texts), "rank": int(self.rank), "device": str(dev),
                 "model_id": model_id, "final_ret": loss_val, "n_trainable_tensors": n_trainable}
        return model, tok, stats

    def _run(self, index: Any, ctx: EvalContext, info: dict, meter: CostMeter) -> dict:
        seed_ids = info["seed_ids"]
        ok, reason = _real_lora_available()
        degradations = [] if ok else [reason]
        t0 = time.perf_counter()

        # ---- Prefer real LoRA; fall back to the representation-space low-rank adapter on failure ----
        real_model = real_tok = None
        train_stats: dict = {}
        if ok and info.get("real_lora", True):
            try:
                real_model, real_tok, train_stats = self._train_real_lora(index, seed_ids, ctx)
                train_stats["mode"] = "real_lora"
            except Exception as exc:
                degradations.append("real LoRA failed, falling back to the representation-space adapter: %s: %s" % (type(exc).__name__, exc))
                real_model = real_tok = None
        W = None
        if real_model is None:
            try:
                W, train_stats = self._train_adapter(index, seed_ids, ctx)
                train_stats["mode"] = "linear_adapter"
            except Exception as exc:
                degradations.append("low-rank adapter training failed, degrading to direct deletion: %s: %s" % (type(exc).__name__, exc))

        alive = _unique_ints(index.alive_ids())
        n_reencoded = 0
        if real_model is not None and real_tok is not None and alive:
            try:
                import torch

                texts = []
                for i in alive:
                    t = None
                    fn = getattr(index, "text_for_id", None)
                    if callable(fn):
                        t = fn(int(i))
                    texts.append(str(t) if isinstance(t, str) and t else "[%s]" % index.meta(i).doc_id)
                dev = next(real_model.parameters()).device
                embs = []
                with torch.no_grad():
                    for s in range(0, len(texts), 64):
                        enc = real_tok(texts[s:s + 64], padding=True, truncation=True, max_length=256,
                                       return_tensors="pt").to(dev)
                        out = real_model(**enc).last_hidden_state[:, 0]
                        embs.append(torch.nn.functional.normalize(out, dim=-1).cpu().numpy().astype(np.float32))
                newV = np.vstack(embs) if embs else None
                if newV is not None and newV.shape[0] == len(alive):
                    idx_dim = int(getattr(index, "dim", 0) or 0)
                    if idx_dim and int(newV.shape[1]) != idx_dim:
                        # Explicit dimension check (latent defect found during verification):
                        # when the encoder dim differs from the index dim, skip re-encoding, but this does
                        # **not** count as a training failure and does not change lora_mode.
                        train_stats["reencode_skipped_reason"] = (
                            "encoder_dim=%d != index_dim=%d: skipping re-encode (LoRA training itself succeeded)"
                            % (int(newV.shape[1]), idx_dim))
                    else:
                        n_reencoded = _rewrite_vectors(index, alive, newV)
                        train_stats["reencoded"] = int(n_reencoded)
                elif newV is not None:
                    train_stats["reencode_skipped_reason"] = (
                        "encoded row count %d != alive vector count %d: skipping re-encode" % (int(newV.shape[0]), len(alive)))
            except Exception as exc:
                degradations.append("re-encode failed: %s: %s" % (type(exc).__name__, exc))
        elif W is not None and alive:
            V = index_all_vectors(index, alive)
            if V is not None and V.shape[0]:
                newV = _normalize_rows(np.asarray(V, dtype=np.float32) @ W.T)
                n_reencoded = _rewrite_vectors(index, alive, newV)
        # Contract: when training succeeded but re-encoding was skipped, faithfully record degraded (keep lora_mode=real_lora)
        skip_reason = str(train_stats.get("reencode_skipped_reason") or "")
        if skip_reason:
            degradations.append("re-encode not executed: " + skip_reason)

        if seed_ids:
            index.remove(seed_ids)  # explicitly delete the requested shards (the rest is suppressed by the adapter)
        _compact(index)
        elapsed = time.perf_counter() - t0
        dim = int(getattr(index, "dim", _DIM))
        n_touched = int(n_reencoded + len(seed_ids))
        meter.touch(n_touched)

        meta = {
            "degraded": bool(degradations),
            "degradation_reason": " / ".join(degradations) if degradations else "",
            "real_lora_available": bool(ok),
            "lora_mode": str(train_stats.get("mode", "none")),
            "n_reencoded": int(n_reencoded),
            "adapter": train_stats,
            "note": "real_lora=LoRA finetune of bge-small attention projections then re-encode; linear_adapter=representation-space low-rank adapter (fallback)",
        }
        return {
            "removed_ids": seed_ids,
            "reindex_seconds": float(elapsed),
            "bytes_transferred": int((n_reencoded + len(seed_ids)) * dim * 4),
            "n_vectors_touched": n_touched,
            "cost_model": "encoder finetune + full re-encode",
            "meta": meta,
        }


# ======================================================================================
# B5 TDSCAdapter — rewrite the knowledge base instead of the model (federated adaptation of wang2024whenmachine)
# ======================================================================================
class TDSCAdapter(Baseline):
    """Retrieval-side adaptation of behavioral forgetting: do not change the model; rewrite the requested knowledge and its shadow copies in the knowledge base.

    * Locate: requested shards + cross-silo shadow copies found by ShadowDetector (vector kNN + MinHash/LSH, same signature matrix)
    * Rewrite: text -> neutralized statement (requires text_store); vectors -> remove the forgetting-direction component and re-normalize
    * Delete no documents (utility preserved); only rebuild the ANN structure
    """

    name = "tdsc_adapter"
    display_name = "TDSC Adapter (KB rewrite)"

    def __init__(self, sim_threshold: float = 0.92, lsh_threshold: float = 0.80, knn_k: int = 50,
                 max_rewrite: int = 4096, redaction: Optional[str] = None,
                 cross_client_only: bool = True, use_detector: bool = True) -> None:
        self.sim_threshold = float(sim_threshold)
        self.lsh_threshold = float(lsh_threshold)
        self.knn_k = int(knn_k)
        self.max_rewrite = int(max_rewrite)
        self.cross_client_only = bool(cross_client_only)
        # use_detector=False: fall back to the original single-corpus form (in-silo vector kNN, no signature matrix, no cross-silo closure).
        # This switch only serves the control experiment tdsc_adapter_nodetector.
        self.use_detector = bool(use_detector)
        self.redaction = redaction or "[REDACTED] This passage has been withdrawn per a data-revocation request."
        self.last_detector_stats: Dict[str, Any] = {}

    def _shadow_ids(self, index: Any, forgotten: Sequence[int], info: Mapping[str, Any]) -> list:
        """Cross-silo near-duplicate copies: the same ShadowDetector as FedRevoke (same signature matrix + same thresholds).

        Evaluation protocol: baselines must receive the same information as the method, otherwise the E1 comparison does not hold.
        On the degraded path (no signature matrix) the vector channel is still usable, but detector channel stats are recorded in meta.
        """
        forgotten = _unique_ints(forgotten)
        if not forgotten:
            return []
        if not self.use_detector:
            return self._local_shadow_ids(index, forgotten)
        from .shadow import ShadowDetector

        det = ShadowDetector(
            sim_threshold=float(info.get("sim_threshold", self.sim_threshold) or self.sim_threshold),
            lsh_threshold=float(info.get("lsh_threshold", self.lsh_threshold) or self.lsh_threshold),
            knn_k=int(info.get("knn_k", self.knn_k) or self.knn_k),
            cross_client_only=self.cross_client_only,
            signatures=info.get("signatures"),
        )
        try:
            closure = det.closure(index, forgotten)
        except Exception:
            closure = set()
        self.last_detector_stats = {
            "channel_counts": dict(det.last_channel_counts),
            "n_edges": len(det.last_edges),
            "has_signatures": info.get("signatures") is not None,
        }
        return sorted(int(i) for i in closure if int(i) not in set(forgotten))[: self.max_rewrite]

    def _local_shadow_ids(self, index: Any, forgotten: Sequence[int]) -> list:
        """Original single-corpus form (control experiment tdsc_adapter_nodetector).

        Only in-silo vector-kNN near-duplicate rewriting: no signature matrix (no MinHash/LSH text channel),
        and no cross-silo closure expansion. This is exactly the information boundary of TDSC behavioral
        forgetting in the single-corpus setting, used to peel "our shadow-closure detector" out of B5's effect.
        """
        n_alive = int(getattr(index, "n_alive", 0) or 0)
        if n_alive <= 1:
            return []
        k = int(max(1, min(self.knn_k, n_alive)))
        vecs = index_all_vectors(index, forgotten)
        if vecs is None or not len(vecs):
            return []
        try:
            scores, ids = index.search(np.asarray(vecs, dtype=np.float32), k=k)
        except Exception:
            return []
        seeds = set(int(i) for i in forgotten)
        clients: Dict[int, Any] = {}
        out = set()
        n_edges = 0
        for r, src in enumerate(forgotten):
            src = int(src)
            if src not in clients:
                clients[src] = getattr(index.meta(src), "client_id", None)
            c = clients[src]
            for s, dst in zip(np.asarray(scores[r]).tolist(), np.asarray(ids[r]).tolist()):
                d = int(dst)
                if d < 0 or d in seeds or d in out:
                    continue
                if float(s) < self.sim_threshold:
                    continue
                if d not in clients:
                    clients[d] = getattr(index.meta(d), "client_id", None)
                if clients[d] != c:      # single-corpus form: cannot see copies in other silos
                    continue
                out.add(d)
                n_edges += 1
        self.last_detector_stats = {
            "channel_counts": {"knn": int(n_edges), "lsh": 0},
            "n_edges": int(n_edges),
            "has_signatures": False,
            "mode": "local_knn_only",
        }
        return sorted(out)[: self.max_rewrite]

    def _run(self, index: Any, ctx: EvalContext, info: dict, meter: CostMeter) -> dict:
        seed_ids = info["seed_ids"]
        t0 = time.perf_counter()
        shadow = self._shadow_ids(index, seed_ids, info)
        targets = sorted(set(seed_ids) | set(shadow))
        V = index_all_vectors(index, targets) if targets else None

        # ---- Vector rewrite: remove the forgetting subspace (normalized mean of forgotten-document directions) ----
        n_vec = 0
        if targets and V is not None and V.shape[0]:
            F = index_all_vectors(index, seed_ids) if seed_ids else None
            u = None
            if F is not None and F.shape[0]:
                mean = np.asarray(F, dtype=np.float32).mean(axis=0)
                nu = float(np.linalg.norm(mean))
                u = (mean / nu).astype(np.float32) if nu > _EPS else None
            newV = np.asarray(V, dtype=np.float32).copy()
            if u is not None:
                newV = newV - np.outer(newV @ u, u)
            n_vec = _rewrite_vectors(index, targets, _normalize_rows(newV))
            _compact(index)

        # ---- Text rewrite (knowledge-base level): replace the wording of revoked knowledge with a neutral statement ----
        n_text = 0
        store = ctx.text_store
        if store is not None:
            for i in targets:
                try:
                    store[int(i)] = self.redaction
                    n_text += 1
                except Exception:
                    continue
        elapsed = time.perf_counter() - t0
        dim = int(getattr(index, "dim", _DIM))
        meter.touch(len(targets))
        return {
            "removed_ids": [],  # behavioral forgetting: no documents deleted
            "reindex_seconds": float(elapsed),
            "bytes_transferred": int(len(targets) * dim * 4),
            "n_vectors_touched": int(len(targets)),
            "cost_model": "in-place KB rewrite (vectors + text)",
            "meta": {
                "n_rewritten_vectors": int(n_vec),
                "n_rewritten_texts": int(n_text),
                "n_shadow_rewritten": int(len(shadow)),
                "shadow_ids": shadow[:64],
                "detector": dict(self.last_detector_stats),
                "deletes_documents": False,
                "note": "federated adaptation of TDSC behavioral forgetting: rewrite the knowledge base (text+vectors), no document deletion, no model training",
            },
        }


class TDSCAdapterNoDetector(TDSCAdapter):
    """B5's original single-corpus form (control): no signature matrix, no cross-silo shadow closure.

    Evaluation protocol: B5 reuses FedRevoke's same ShadowDetector to wipe the retrieval channel, so the
    reading "with the detector taken away" must be reported to peel this paper's contribution out of B5's
    success. Apart from the detection scope, the remaining mechanisms (vector decorrelation rewrite + text
    neutralization, no document deletion) are identical.
    """

    name = "tdsc_adapter_nodetector"
    display_name = "TDSC Adapter (local kNN only, no shadow detector)"

    def __init__(self, **kw: Any) -> None:
        kw["use_detector"] = False
        super().__init__(**kw)


# ======================================================================================
# E5 convention supplement: full rebuild including "re-encoding" (proactive disclosure of baseline fairness)
# ======================================================================================
REENCODE_SECONDS_BY_DATASET = {  # remeasured full-corpus re-encoding (bge-small-en-v1.5, batch 256, fp16, cuda)
    "multihoprag": 6.01,    # 11,410 texts -> 1,898 text/s
    "nq": 73.73,            # 156,671 texts -> 2,125 text/s
    "trec-covid": 209.13,   # 386,596 texts -> 1,849 text/s
}


class FullRebuildReencode(FullRebuild):
    """**Upper-bound cost version** of full rebuild: additionally counts the time to "re-encode the entire corpus".

    Paper convention: the current full_rebuild uses cached embeddings (no re-encode), which is a
    baseline-**favorable** lower bound; this class provides the cost including encoding, for the second
    cost curve in E5.
    """

    name = "full_rebuild_reencode"
    display_name = "Full Rebuild (+re-encode)"

    def __init__(self, dataset_name: Optional[str] = None, n_docs_full: Optional[int] = None,
                 reencode_seconds: Optional[float] = None) -> None:
        self.dataset_name = dataset_name
        self.n_docs_full = n_docs_full
        self.reencode_seconds = reencode_seconds

    def _reencode_seconds(self, index: Any, info: Mapping[str, Any] | None = None) -> float:
        if self.reencode_seconds is not None:
            return float(self.reencode_seconds)
        info = info or {}
        name = str(self.dataset_name or info.get("dataset_name") or "")
        # In cost sweeps datasets are truncated to <name>_n<k>; match by prefix back to the original dataset's encoding throughput
        base = REENCODE_SECONDS_BY_DATASET.get(name)
        if base is None:
            base = next((v for k, v in REENCODE_SECONDS_BY_DATASET.items() if name.startswith(k)), 0.0)
        n_total = int(index.stats().get("n_vectors", 0)) or 1
        n_full = self.n_docs_full or info.get("n_docs_full")
        frac = 1.0 if not n_full else min(1.0, n_total / float(n_full))
        return float(base * frac)

    def _run(self, index: Any, ctx: EvalContext, info: dict, meter: CostMeter) -> dict:
        out = super()._run(index, ctx, info, meter)
        extra = self._reencode_seconds(index, info)
        out["reindex_seconds"] = float(out.get("reindex_seconds", 0.0)) + extra
        meta = dict(out.get("meta") or {})
        meta.update({
            "reencode_seconds": float(extra),
            "cost_model_note": "full rebuild + re-encode the whole corpus (the cached-embedding version is a comparable favorable lower bound)",
        })
        out["meta"] = meta
        out["cost_model"] = "full reindex + corpus re-encode"
        return out


# ======================================================================================
# Control group (not a paper baseline, but a causal control for repair effectiveness)
# ======================================================================================
class RandomReplica(Baseline):
    """Control: after deletion, add the same number of **random surviving vector copies** as AnchorRepair.

    Purpose: raise "repair effectiveness" from correlation to causation — if randomly inserting the same
    number of vectors cannot restore recall while AnchorRepair restores it to >= 0.9 of intact, then the
    recovery truly comes from anchor local reconnection rather than merely adding vectors.
    """

    name = "random_replica"
    display_name = "Random Replica (control)"

    def __init__(self, n_replicas: int = 512, seed: int = SEED) -> None:
        self.n_replicas = max(0, int(n_replicas))
        self.seed = int(seed)

    def _run(self, index: Any, ctx: EvalContext, info: dict, meter: CostMeter) -> dict:
        seed_ids = info["seed_ids"]
        n_before = int(index.stats().get("n_vectors", 0))
        t0 = time.perf_counter()
        if seed_ids:
            index.remove(seed_ids)
        alive = _unique_ints(index.alive_ids())
        rng = np.random.default_rng(self.seed)
        n = int(min(self.n_replicas, len(alive)))
        added: list = []
        if n > 0:
            pick = np.sort(rng.choice(len(alive), n, replace=False))
            ids = [alive[int(i)] for i in pick]
            vecs = index_all_vectors(index, ids)
            metas = [index_meta(index, i) for i in ids]
            keep = [k for k, m in enumerate(metas) if m is not None]
            if vecs is not None and keep:
                try:
                    added = list(index.add(np.asarray(vecs, dtype=np.float32)[keep],
                                           [metas[k] for k in keep]) or [])
                except Exception:
                    added = []
        _compact(index)
        elapsed = time.perf_counter() - t0
        n_after = int(index.stats().get("n_vectors", 0))
        dim = int(getattr(index, "dim", _DIM))
        meter.touch(len(added))
        return {
            "removed_ids": seed_ids,
            "reindex_seconds": float(elapsed),
            "bytes_transferred": int((len(added) + len(seed_ids)) * dim * 4),
            "n_vectors_touched": int(len(added) + len(seed_ids)),
            "cost_model": "control: delete + random replicas (no anchor repair)",
            "meta": {
                "n_replica_control": int(len(added)),
                "n_vectors_delta": int(n_after - n_before),
                "note": "control group: random replicas, no anchor local reconnection; used to test whether recall recovery comes from the repair strategy itself",
            },
        }


# ======================================================================================
# Registry
# ======================================================================================
BASELINES = {
    FullRebuild.name: FullRebuild,
    NaiveDelete.name: NaiveDelete,
    SISA.name: SISA,
    LoRAFinetune.name: LoRAFinetune,
    TDSCAdapter.name: TDSCAdapter,
    TDSCAdapterNoDetector.name: TDSCAdapterNoDetector,
    RandomReplica.name: RandomReplica,
    FullRebuildReencode.name: FullRebuildReencode,
}

_ALIASES = {
    "full_rebuild": "full_rebuild",
    "fullrebuild": "full_rebuild",
    "rebuild": "full_rebuild",
    "naive_delete": "naive_delete",
    "naivedelete": "naive_delete",
    "naive": "naive_delete",
    "sisa": "sisa",
    "sisa_shard": "sisa",
    "lora_finetune": "lora_finetune",
    "lora": "lora_finetune",
    "lorafinetune": "lora_finetune",
    "tdsc_adapter": "tdsc_adapter",
    "tdsc": "tdsc_adapter",
    "tdscadapter": "tdsc_adapter",
    "tdsc_adapter_nodetector": "tdsc_adapter_nodetector",
    "tdsc_nodetector": "tdsc_adapter_nodetector",
    "tdscadapterno": "tdsc_adapter_nodetector",
    "b5_nodetector": "tdsc_adapter_nodetector",
    "random_replica": "random_replica",
    "random": "random_replica",
    "control": "random_replica",
    "full_rebuild_reencode": "full_rebuild_reencode",
    "rebuild_reencode": "full_rebuild_reencode",
}


def available_baselines() -> list:
    return sorted(BASELINES)


def build_baseline(name: str, **kw: Any) -> Baseline:
    """Construct a baseline by name (case/hyphen insensitive)."""
    key = _ALIASES.get(str(name).strip().lower().replace("-", "_"))
    if key is None:
        raise ValueError("unknown baseline %r (options: %s)" % (name, ", ".join(available_baselines())))
    return BASELINES[key](**kw)
