"""revoke.py — M1–M6 revocation orchestration and hash-chain deletion certificates (INTERFACES.md Section 8).

Contract (signatures must not change)::

    @dataclass
    class RevocationResult:
        deleted_ids: list[int]; report: ForgetReport; cost: dict; certificate_path: str

    class RevocationPipeline:
        def __init__(self, index, detector, repair, verifier, cert_dir: Path, encoder=None) -> None
        def revoke(self, doc_ids: Sequence[str], query_sample=None) -> RevocationResult
        def certificate(self) -> dict   # {"request_id","timestamp","n_deleted","chain_sha256"}

Pipeline::

    Revocation request(doc_ids, client_id)
       +-[M1] provenance locating     index.ids_for_doc(doc_id)          -> seed_ids
       +-[M2] shadow closure          ShadowDetector.closure()           -> delete_set (+closure_stats)
       +-[M3] cascading erasure       ProvenanceIndex.remove(delete_set)
       +-[M4] impact-aware repair     AnchorRepair.repair(removed, query_sample)
       +-[M5] verification            MIAProbe / ElicitationTest          -> ForgetReport
       +-[M6] audit certificate       certificate() -> hash-chain JSON

Certificate (artifacts/certificates/*.json) fields include at least::

    request_id / timestamp / n_deleted / deleted_doc_ids / closure_stats
    prev_hash / chain_sha256

Chain definition: chain_sha256 = sha256(canonical_json(certificate without the chain_sha256 field));
prev_hash points to the chain_sha256 of the previous certificate (the first one is 64 zeros, GENESIS_HASH).
verify_certificate() can chain-verify a single certificate or an entire certificate directory
(including sequence-number continuity checks).

INTERFACES.md Section 11 (takes precedence over the default implementation)
--------------------------------------------------
* Short digest fingerprints (data_prep's blake2b 8-byte digest) are **strictly forbidden** for LSH
  near-duplicate estimation; assert_fingerprint_channel() calls detector.diagnose_fingerprints()
  after the index is built, and if the digest count is > 0 and that experiment's shadow_ratio > 0:
  full mode raises RuntimeError (with the hint), --smoke mode prints an explicit WARNING and continues.
* The signature-matrix source priority is implemented by the caller (run_experiments):
  load_bundle(ds)["signatures"] -> np.load(ds_dir/minhash_sig.npy) -> ShadowDetector(signatures=...).

Design trade-offs
--------
* Every step probes upstream modules for capabilities (hasattr + signature attempts); a missing
  module or an interface variant never crashes the whole pipeline — it is recorded in
  cost["stage_errors"] and the certificate's stage_errors instead.
* All statistics use "internal ids" (index_core guarantees that internal ids are permanently
  stable), so certificates can be re-checked independently of the index.
* Timestamps are used only for audit tracking and take part in no scientific measurement, so they
  do not affect result reproducibility.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

import numpy as np

from .metrics import CostMeter

try:  # unified random seed (config.py; fall back to the contract constant if missing)
    from .config import ARTIFACTS as _ARTIFACTS, SEED as _CONFIG_SEED  # type: ignore
except Exception:  # pragma: no cover - fallback when config is not in place yet
    _ARTIFACTS = Path(__file__).resolve().parents[2] / "artifacts"
    _CONFIG_SEED = 20260214

from .verify import (  # noqa: E402  (placed after config so the unified fallback above applies)
    ElicitationTest,
    ForgetReport,
    MIAProbe,
    build_forget_report,
    retrieval_hit_rate,
    rho_hat_residual,
)

SEED: int = int(_CONFIG_SEED)

# Closed-book (empty-context) template: the criterion matches the retrieval tier; only no evidence is given (elicit floor)
CLOSED_BOOK_TEMPLATE = "Context:\n(no evidence retrieved)\nQuestion: {question}\nAnswer:"

DEFAULT_CERT_DIR = Path(_ARTIFACTS) / "certificates"
CERT_VERSION = "fedrevoke.revocation.v1"
GENESIS_HASH = "0" * 64

__all__ = [
    "RevocationResult",
    "RevocationPipeline",
    "ForgetVerifier",
    "canonical_json",
    "certificate_digest",
    "verify_certificate",
    "load_certificate",
    "list_certificates",
    "make_request_id",
    "assert_fingerprint_channel",
    "load_signature_matrix_from_dir",
    "DEFAULT_CERT_DIR",
    "CERT_VERSION",
    "GENESIS_HASH",
    "SEED",
]

_EPS = 1e-12


# --------------------------------------------------------------------------------------
# Canonical hashing utilities (single source of truth for chain verification)
# --------------------------------------------------------------------------------------
def _json_default(obj: Any) -> Any:
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (set, frozenset)):
        return sorted(obj, key=lambda v: (str(type(v)), str(v)))
    if isinstance(obj, Path):
        return str(obj)
    return str(obj)


def canonical_json(obj: Any) -> str:
    """Canonical JSON text: sorted keys, compact separators, non-ASCII preserved as-is."""
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=_json_default)


def certificate_digest(payload: Mapping[str, Any]) -> str:
    """Certificate digest: sha256 hex of the canonical JSON after removing the chain_sha256 / _path fields."""
    body = {k: v for k, v in payload.items() if k not in ("chain_sha256", "_path")}
    return hashlib.sha256(canonical_json(body).encode("utf-8")).hexdigest()


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def make_request_id(doc_ids: Sequence[str], seed: int, t0: float) -> str:
    """Deterministic, readable request id: UTC timestamp + blake2b digest of the request content/seed/start time."""
    key = canonical_json({"docs": list(doc_ids)[:64], "n": len(doc_ids), "seed": int(seed), "t": round(t0, 6)})
    digest = hashlib.blake2b(key.encode("utf-8"), digest_size=6).hexdigest()
    return "rv-%s-%s" % (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S"), digest)


# --------------------------------------------------------------------------------------
# INTERFACES Section 11: fingerprint-channel fail-fast gate
# --------------------------------------------------------------------------------------
def load_signature_matrix_from_dir(ds_dir: Any, filename: str = "minhash_sig.npy"):
    """Load data/processed/<ds>/minhash_sig.npy (row order = corpus.jsonl row order = pid ascending)."""
    path = Path(ds_dir) / filename
    if not path.exists():
        return None
    arr = np.load(str(path), allow_pickle=False)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    return np.ascontiguousarray(arr)


def assert_fingerprint_channel(
    index: Any,
    detector: Any,
    shadow_ratio: float = 0.0,
    *,
    smoke: bool = False,
    minhash_perm: Optional[int] = None,
    context: str = "",
) -> dict:
    """INTERFACES.md Section 11 item 4: silent degradation of the text LSH channel is forbidden.

    When shadow_ratio > 0 and the detector reports a digest fingerprint count > 0:
      * smoke=False -> raise RuntimeError (the exception message carries the detector's hint)
      * smoke=True  -> print an explicit WARNING and continue (degraded=True in the returned dict)
    Returns {"checked","degraded","counts","n_alive","median_sig_len","hint","context"}.
    """
    info: dict = {
        "checked": False,
        "degraded": False,
        "counts": {},
        "n_alive": int(getattr(index, "n_alive", 0) or 0),
        "median_sig_len": 0,
        "hint": "",
        "context": str(context),
    }
    fn = getattr(detector, "diagnose_fingerprints", None)
    if detector is None or not callable(fn):
        info["hint"] = "detector does not provide diagnose_fingerprints(); cannot self-check fingerprint sources"
        return info
    try:
        diag = dict(fn(index))
    except Exception as exc:
        info["hint"] = "diagnose_fingerprints call failed: %s: %s" % (type(exc).__name__, exc)
        return info

    info["checked"] = True
    info["counts"] = {str(k): int(v) for k, v in (diag.get("counts") or {}).items()}
    info["n_alive"] = int(diag.get("n_alive", info["n_alive"]) or 0)
    info["median_sig_len"] = int(diag.get("median_sig_len", 0) or 0)
    info["hint"] = str(diag.get("hint", "") or "")

    if minhash_perm is not None and info["median_sig_len"]:
        if int(info["median_sig_len"]) != int(minhash_perm):
            info["hint"] = (
                info["hint"] + " / signature column count %d != minhash_perm %d" % (info["median_sig_len"], minhash_perm)
            ).strip(" /")

    n_digest = int(info["counts"].get("digest", 0))
    if float(shadow_ratio or 0.0) > 0.0 and n_digest > 0:
        info["degraded"] = True
        msg = (
            "[Fingerprint channel degraded] shadow_ratio=%.3f but %d/%d alive vectors use short-digest fingerprints: "
            "the LSH text channel degrades to exact match, and E1/E4/E7 conclusions will be invalid. %s"
            % (float(shadow_ratio), n_digest, int(info["n_alive"]), info["hint"])
        )
        if smoke:
            print("WARNING: " + msg, flush=True)
        else:
            raise RuntimeError(msg)
    return info


# --------------------------------------------------------------------------------------
# RevocationResult
# --------------------------------------------------------------------------------------
@dataclass
class RevocationResult:
    """Complete result of a single revocation request (INTERFACES.md Section 8)."""

    deleted_ids: list
    report: ForgetReport
    cost: dict
    certificate_path: str
    # --- Extra fields (do not change the semantics of contract fields) ---
    request_id: str = ""
    certificate: dict = field(default_factory=dict)
    closure_stats: dict = field(default_factory=dict)
    repair_stats: dict = field(default_factory=dict)
    seeds: list = field(default_factory=list)
    n_deleted_new: int = 0
    fingerprint_check: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "deleted_ids": [int(i) for i in self.deleted_ids],
            "n_deleted": len(self.deleted_ids),
            "n_deleted_new": int(self.n_deleted_new),
            "report": self.report.as_dict() if isinstance(self.report, ForgetReport) else dict(self.report),
            "cost": dict(self.cost),
            "certificate_path": str(self.certificate_path),
            "request_id": self.request_id,
            "closure_stats": dict(self.closure_stats),
            "repair_stats": dict(self.repair_stats),
            "seeds": [int(i) for i in self.seeds],
            "fingerprint_check": dict(self.fingerprint_check),
        }


# --------------------------------------------------------------------------------------
# Verifier (M5)
# --------------------------------------------------------------------------------------
class ForgetVerifier:
    """Default verifier that packages retrieval residue + MIAProbe + generation-side elicitation tests into a ForgetReport.

    Callers may also inject a custom verifier (see RevocationPipeline._run_verifier).
    """

    def __init__(
        self,
        encoder: Any = None,
        generator: Any = None,
        k: int = 10,
        forgotten_qa: Optional[Sequence[Mapping[str, Any]]] = None,
        control_pids: Optional[Sequence[int]] = None,
        seed: int = SEED,
        max_probes: int = 512,
        elicit: bool = True,
        # Aligned with the baseline evaluate_forget (the old 5/200 made FedRevoke over-report about 4 items on one side)
        k_elicit: int = 10,
        max_new_tokens: int = 64,
        closed_book: bool = False,   # additionally run one closed-book (empty-context) generation -> elicit floor value
        exclude_self: bool = True,
    ) -> None:
        self.encoder = encoder
        self.generator = generator
        self.k = max(1, int(k))
        self.forgotten_qa = list(forgotten_qa) if forgotten_qa else []
        self.control_pids = list(control_pids) if control_pids else None
        self.seed = int(seed)
        self.max_probes = max(1, int(max_probes))
        self.elicit = bool(elicit)
        self.k_elicit = max(1, int(k_elicit))
        self.max_new_tokens = max(1, int(max_new_tokens))
        self.closed_book = bool(closed_book)
        # Upstream verify.MIAProbe's exclude_self defaults to False: alive control samples would have
        # similarity 1.0 to themselves, making AUC always 0 (upstream issue U1).
        # Excluding self-matches is enabled by default here.
        self.exclude_self = bool(exclude_self)
        self.last_meta: dict = {}

    # -- Control group ----------------------------------------------------- #
    def _control(self, index: Any, forgotten: Sequence[int]) -> list:
        if self.control_pids:
            return [int(p) for p in self.control_pids][: self.max_probes]
        try:
            alive = np.asarray(index.alive_ids(), dtype=np.int64).ravel()
        except Exception:
            return []
        banned = set(int(p) for p in forgotten)
        pool = [int(a) for a in alive.tolist() if int(a) not in banned]
        if not pool:
            return []
        rng = np.random.default_rng(self.seed)
        n = min(len(pool), self.max_probes)
        idx = np.sort(rng.choice(len(pool), size=n, replace=False))
        return [pool[int(i)] for i in idx]

    # -- Main entry -------------------------------------------------------- #
    def verify(
        self,
        index: Any,
        forgotten_pids: Sequence[int],
        query_sample: Any = None,
        k: Optional[int] = None,
        forgotten_qa: Optional[Sequence[Mapping[str, Any]]] = None,
    ) -> ForgetReport:
        report, _ = self.verify_detailed(index, forgotten_pids, query_sample, k, forgotten_qa)
        return report

    def verify_detailed(
        self,
        index: Any,
        forgotten_pids: Sequence[int],
        query_sample: Any = None,
        k: Optional[int] = None,
        forgotten_qa: Optional[Sequence[Mapping[str, Any]]] = None,
    ) -> tuple:
        """Return (ForgetReport, meta); meta indicates whether each channel was actually measured."""
        kk = self.k if k is None else max(1, int(k))
        forgotten = [int(p) for p in forgotten_pids]
        meta: dict = {
            "k": kk,
            "n_forgotten": len(forgotten),
            "hit_rate_measured": False,
            "mia_measured": False,
            "elicit_measured": False,
            "errors": [],
        }

        hit_rate = 0.0
        if query_sample is not None and len(forgotten):
            try:
                hit_rate = float(retrieval_hit_rate(index, query_sample, forgotten, kk))
                meta["hit_rate_measured"] = True
            except Exception as exc:  # pragma: no cover - defensive
                meta["errors"].append("hit_rate: %s: %s" % (type(exc).__name__, exc))

        mia = 0.5
        control = self._control(index, forgotten)
        meta["n_control"] = len(control)
        if forgotten and control:
            try:
                probe = MIAProbe(index, self.encoder, seed=self.seed, k=kk, exclude_self=self.exclude_self)
                mia = float(probe.auc(forgotten, control))
                meta["mia_measured"] = True
            except Exception as exc:  # pragma: no cover - defensive
                meta["errors"].append("mia_auc: %s: %s" % (type(exc).__name__, exc))

        elicit = 0.0
        qa = list(forgotten_qa) if forgotten_qa else list(self.forgotten_qa)
        if self.elicit and self.generator is not None and qa:
            try:
                test = ElicitationTest(
                    self.generator, index, k=self.k_elicit, max_new_tokens=self.max_new_tokens
                )
                elicit, details = test.run_detailed(qa)
                elicit = float(elicit)
                meta["elicit_measured"] = True
                meta["n_elicit_items"] = len(qa)
                # Per-item hit flags (for paired McNemar; order = qa_items order, consistent with the baseline)
                meta["elicit_flags"] = [1 if d.get("elicited") else 0 for d in details if "elicited" in d]
                if self.closed_book:
                    cb = ElicitationTest(self.generator, index, k=self.k_elicit,
                                         max_new_tokens=self.max_new_tokens,
                                         prompt_template=CLOSED_BOOK_TEMPLATE)
                    cb_rate, cb_details = cb.run_detailed(qa)
                    meta["elicit_rate_closed_book"] = float(cb_rate)
                    meta["elicit_flags_closed_book"] = [1 if d.get("elicited") else 0
                                                        for d in cb_details if "elicited" in d]
            except Exception as exc:
                meta["errors"].append("elicit: %s: %s" % (type(exc).__name__, exc))

        report = build_forget_report(
            hit_rate=hit_rate,
            mia_auc_value=mia,
            elicit_rate=elicit,
            rho_hat=rho_hat_residual(hit_rate, elicit),
            n_probes=len(forgotten) + len(control),
        )
        self.last_meta = meta
        return report, meta


_VERIFIER_KEYS = frozenset(
    {"encoder", "generator", "k", "forgotten_qa", "control_pids", "seed", "max_probes", "elicit",
     "k_elicit", "max_new_tokens", "exclude_self", "closed_book"}
)


# --------------------------------------------------------------------------------------
# RevocationPipeline
# --------------------------------------------------------------------------------------
class RevocationPipeline:
    """M1–M6 revocation pipeline (INTERFACES.md Section 8)."""

    def __init__(
        self,
        index: Any,
        detector: Any = None,
        repair: Any = None,
        verifier: Any = None,
        cert_dir: Any = None,
        encoder: Any = None,
        *,
        k: int = 10,
        generator: Any = None,
        forgotten_qa: Optional[Sequence[Mapping[str, Any]]] = None,
        seed: int = SEED,
        shadow_ground_truth: Any = None,
        shadow_ratio: float = 0.0,
        smoke: bool = False,
        verbose: bool = False,
        strict: bool = False,
        client_id: Optional[str] = None,
        closed_book: bool = False,
    ) -> None:
        self.index = index
        self.detector = detector
        self.repair = repair
        self.verifier = verifier
        self.cert_dir = Path(cert_dir) if cert_dir is not None else DEFAULT_CERT_DIR
        self.encoder = encoder
        self.k = max(1, int(k))
        self.generator = generator
        self.forgotten_qa = list(forgotten_qa) if forgotten_qa else []
        self.seed = int(seed)
        self.shadow_ground_truth = shadow_ground_truth
        self.shadow_ratio = float(shadow_ratio or 0.0)
        self.smoke = bool(smoke)
        self.verbose = bool(verbose)
        self.strict = bool(strict)
        self.client_id = client_id
        self.closed_book = bool(closed_book)

        self.last_result: Optional[RevocationResult] = None
        self.last_certificate: Optional[dict] = None
        self.history: list = []  # certificates issued within this process
        self.fingerprint_check: dict = {}

    # ------------------------------------------------------------------ Utilities
    def _log(self, msg: str) -> None:
        if self.verbose:
            print("[revoke] " + str(msg), flush=True)

    def _default_verifier(self) -> ForgetVerifier:
        return ForgetVerifier(
            encoder=self.encoder,
            generator=self.generator,
            k=self.k,
            forgotten_qa=self.forgotten_qa,
            seed=self.seed,
            closed_book=self.closed_book,
        )

    @staticmethod
    def _unique_ints(values: Iterable[Any]) -> list:
        out, seen = [], set()
        for v in values or ():
            try:
                i = int(v)
            except (TypeError, ValueError):
                continue
            if i in seen:
                continue
            seen.add(i)
            out.append(i)
        return out

    # ------------------------------------------------------- Section 11 fingerprint-channel gate
    def check_fingerprints(self) -> dict:
        """Call once after the index is built (INTERFACES.md Section 11 item 4); the result is cached in self.fingerprint_check."""
        if self.fingerprint_check:
            return self.fingerprint_check
        minhash_perm = getattr(self.detector, "minhash_perm", None) if self.detector is not None else None
        self.fingerprint_check = assert_fingerprint_channel(
            self.index,
            self.detector,
            shadow_ratio=self.shadow_ratio,
            smoke=self.smoke,
            minhash_perm=minhash_perm,
            context="shadow_ratio=%.3f" % self.shadow_ratio,
        )
        return self.fingerprint_check

    # ------------------------------------------------------------------ M1 provenance locating
    def locate(self, doc_ids: Sequence[str]) -> tuple:
        """M1: resolve doc_ids into an internal-id seed set; returns (seeds, per_doc, missing).

        A2 operationalization: if the pipeline is bound to a client_id, documents belonging to
        other silos resolve to the empty set.
        """
        index = self.index
        seeds: list = []
        per_doc: dict = {}
        missing: list = []
        requester = self.client_id
        requested = [str(d) for d in (doc_ids or []) if d is not None and str(d) != ""]
        for doc_id in requested:
            ids = []
            fn = getattr(index, "ids_for_doc", None)
            if callable(fn):
                try:
                    ids = self._unique_ints(fn(doc_id))
                except Exception:
                    ids = []
            if requester is not None and ids:
                owner = None
                try:
                    owner = str(index.meta(ids[0]).client_id)
                except Exception:
                    owner = None
                if owner is not None and owner != str(requester):
                    ids = []
            per_doc[doc_id] = ids
            if not ids:
                missing.append(doc_id)
            seeds.extend(ids)
        return self._unique_ints(seeds), per_doc, missing

    # ------------------------------------------------------------------ M2 shadow closure
    def closure(self, seeds: Sequence[int]) -> tuple:
        """M2: shadow closure; returns (delete_set (sorted list, including seeds), closure_stats)."""
        seeds = self._unique_ints(seeds)
        stats: dict = {
            "n_seed": len(seeds),
            "n_closure": len(seeds),
            "per_client": {},
            "sim_hist": [],
            "precision": float("nan"),
            "recall": float("nan"),
            "channel_counts": {},
            "iterations": 0,
            "detector": None,
        }
        detector = self.detector
        if detector is None or not seeds:
            stats["detector"] = "none"
            return sorted(seeds), stats

        stats["detector"] = type(detector).__name__
        closure: Optional[set] = None
        report_fn = getattr(detector, "shadow_report", None)
        if callable(report_fn):
            try:
                kwargs = {}
                if self.shadow_ground_truth is not None:
                    kwargs["gt_ids"] = self.shadow_ground_truth
                rep = report_fn(self.index, seeds, **kwargs)
                if isinstance(rep, Mapping):
                    stats.update({k: v for k, v in rep.items() if k != "closure_ids"})
                    ids = rep.get("closure_ids")
                    if ids is not None:
                        closure = set(self._unique_ints(ids))
            except Exception as exc:
                stats["error"] = "%s: %s" % (type(exc).__name__, exc)
                if self.strict:
                    raise
        if closure is None:
            fn = getattr(detector, "closure", None)
            if callable(fn):
                try:
                    closure = set(self._unique_ints(fn(self.index, seeds)))
                except Exception as exc:
                    stats["error"] = "%s: %s" % (type(exc).__name__, exc)
                    if self.strict:
                        raise
                    closure = set(seeds)
            else:
                closure = set(seeds)

        closure.update(seeds)  # Section 11 semantics: the closure includes the seeds themselves
        delete_set = sorted(int(i) for i in closure)
        stats["n_closure"] = len(delete_set)
        if not stats.get("per_client"):
            counts: dict = {}
            for i in delete_set:
                try:
                    cid = str(self.index.meta(i).client_id)
                except Exception:
                    cid = "?"
                counts[cid] = counts.get(cid, 0) + 1
            stats["per_client"] = dict(sorted(counts.items()))
        return delete_set, stats

    # ------------------------------------------------------------------ M3 cascading erasure
    def erase(self, delete_set: Sequence[int]) -> dict:
        """M3: tombstone deletion; returns {n_deleted_new, n_deleted_total, n_vectors, n_alive}."""
        index = self.index
        before = int(index.stats().get("n_deleted", 0))
        ids = self._unique_ints(delete_set)
        fn = getattr(index, "remove", None)
        if callable(fn) and ids:
            fn(ids)
        after_stats = index.stats()
        after = int(after_stats.get("n_deleted", 0))
        return {
            "n_deleted_new": int(max(0, after - before)),
            "n_deleted_total": int(after),
            "n_vectors": int(after_stats.get("n_vectors", 0)),
            "n_alive": int(after_stats.get("n_alive", 0)),
        }

    # ------------------------------------------------------------------ M4 repair
    def run_repair(self, removed_ids: Sequence[int], query_sample: Any = None) -> dict:
        """M4: anchor local reconnection + score calibration; returns a placeholder dict when unavailable (no exception)."""
        repair = self.repair
        removed = self._unique_ints(removed_ids)
        if repair is None:
            return {"n_reconnected": 0, "score_shift": 0.0, "repaired": False, "reason": "no repair module"}
        fn = getattr(repair, "repair", None)
        if not callable(fn):
            return {"n_reconnected": 0, "score_shift": 0.0, "repaired": False, "reason": "repair module has no repair()"}
        try:
            out = fn(self.index, removed, query_sample)
            result = dict(out) if isinstance(out, Mapping) else {"result": out}
            result["repaired"] = True
            return result
        except Exception as exc:
            if self.strict:
                raise
            return {
                "n_reconnected": 0,
                "score_shift": 0.0,
                "repaired": False,
                "reason": "%s: %s" % (type(exc).__name__, exc),
            }

    # ------------------------------------------------------------------ M5 verification
    def _run_verifier(self, forgotten_pids: Sequence[int], query_sample: Any) -> tuple:
        forgotten = self._unique_ints(forgotten_pids)
        verifier = self.verifier if self.verifier is not None else self._default_verifier()

        if isinstance(verifier, Mapping):
            merged = dict(verifier)
            merged.setdefault("encoder", self.encoder)
            merged.setdefault("generator", self.generator)
            merged.setdefault("forgotten_qa", self.forgotten_qa)
            verifier = ForgetVerifier(**{k: v for k, v in merged.items() if k in _VERIFIER_KEYS})

        if hasattr(verifier, "verify_detailed"):
            try:
                report, meta = verifier.verify_detailed(self.index, forgotten, query_sample, self.k)
                return report, dict(meta)
            except Exception as exc:
                if self.strict:
                    raise
                return build_forget_report(0.0, 0.5, 0.0, 0.0, len(forgotten)), {
                    "errors": ["verify_detailed: %s: %s" % (type(exc).__name__, exc)]
                }
        if hasattr(verifier, "verify"):
            for attempt in (
                lambda: verifier.verify(self.index, forgotten, query_sample, self.k),
                lambda: verifier.verify(self.index, forgotten, query_sample),
                lambda: verifier.verify(index=self.index, forgotten_pids=forgotten, query_sample=query_sample),
            ):
                try:
                    out = attempt()
                except TypeError:
                    continue
                except Exception as exc:
                    if self.strict:
                        raise
                    break
                if isinstance(out, ForgetReport):
                    return out, {}
                if isinstance(out, Mapping):
                    return ForgetReport.from_dict(out), dict(out.get("meta", {}))
        if callable(verifier):
            try:
                out = verifier(self.index, forgotten, query_sample, self.k)
                if isinstance(out, ForgetReport):
                    return out, {}
                if isinstance(out, Mapping):
                    return ForgetReport.from_dict(out), dict(out.get("meta", {}))
            except Exception:
                if self.strict:
                    raise

        report, meta = self._default_verifier().verify_detailed(self.index, forgotten, query_sample, self.k)
        meta = dict(meta)
        meta.setdefault("errors", []).append("verifier not recognized; falling back to the default ForgetVerifier")
        return report, meta

    # ------------------------------------------------------------------ M6 certificate
    def _chain_head(self) -> tuple:
        """Return (prev_hash, next_seq): based on certificates already on disk (the chain can continue across processes)."""
        certs = list_certificates(self.cert_dir)
        if not certs:
            return GENESIS_HASH, 0
        certs.sort(key=lambda c: (int(c.get("seq", 0)), str(c.get("timestamp", ""))))
        head = certs[-1]
        return str(head.get("chain_sha256", GENESIS_HASH)), int(head.get("seq", len(certs) - 1)) + 1

    def write_certificate(self, payload: Mapping[str, Any]) -> tuple:
        """Write the certificate to cert_dir (atomic write); returns (full certificate dict, path)."""
        prev_hash, seq = self._chain_head()
        body = dict(payload)
        body["certificate_version"] = CERT_VERSION
        body["seq"] = int(seq)
        body["prev_hash"] = str(prev_hash)
        chain = certificate_digest(body)
        body["chain_sha256"] = chain

        self.cert_dir.mkdir(parents=True, exist_ok=True)
        safe_rid = "".join(ch if (ch.isalnum() or ch in "-_") else "_" for ch in str(body["request_id"]))
        path = self.cert_dir / ("%04d_%s.json" % (int(seq), safe_rid))
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(body, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")
        tmp.replace(path)

        self.last_certificate = body
        self.history.append(body)
        return body, path

    def certificate(self) -> dict:
        """The most recently issued certificate (contract fields request_id/timestamp/n_deleted/chain_sha256)."""
        if self.last_certificate is None:
            return {
                "request_id": None,
                "timestamp": None,
                "n_deleted": 0,
                "deleted_doc_ids": [],
                "closure_stats": {},
                "prev_hash": None,
                "chain_sha256": None,
            }
        return dict(self.last_certificate)

    # ------------------------------------------------------------------ Main entry
    def revoke(
        self,
        doc_ids: Sequence[str],
        query_sample: Any = None,
        *,
        forgotten_qa: Optional[Sequence[Mapping[str, Any]]] = None,
        control_pids: Optional[Sequence[int]] = None,
        extra: Optional[Mapping[str, Any]] = None,
    ) -> RevocationResult:
        """Run one complete revocation (M1->M6) and return a RevocationResult."""
        index = self.index
        t_start = time.perf_counter()
        requested = [str(d) for d in (doc_ids or []) if d is not None and str(d) != ""]
        request_id = make_request_id(requested, self.seed, t_start)
        stats_before = dict(index.stats())
        fp_check = self.check_fingerprints()  # Section 11 fail-fast (WARNING under smoke)

        meter = CostMeter("revoke", measure_vram=True)
        stage_errors: list = []
        with meter:
            # ---- M1 -------------------------------------------------------
            t0 = time.perf_counter()
            seeds, per_doc, missing = self.locate(requested)
            t_locate = time.perf_counter() - t0
            self._log("M1 seeds=%d (missing_docs=%d)" % (len(seeds), len(missing)))

            # ---- M2 -------------------------------------------------------
            t0 = time.perf_counter()
            delete_set, closure_stats = self.closure(seeds)
            t_closure = time.perf_counter() - t0
            closure_stats = dict(closure_stats)
            closure_stats["missing_docs"] = missing
            closure_stats["per_doc_seeds"] = {k: len(v) for k, v in per_doc.items()}
            closure_stats["fingerprint_mode"] = dict(fp_check.get("counts", {}))
            if not seeds:
                stage_errors.append("M1: the requested doc_ids do not exist in the index (no seeds; missing)")
            self._log("M2 closure=%d" % (len(delete_set),))

            # ---- M3 -------------------------------------------------------
            t0 = time.perf_counter()
            erase_stats = self.erase(delete_set)
            t_erase = time.perf_counter() - t0
            self._log("M3 deleted_new=%d total=%d" % (erase_stats["n_deleted_new"], erase_stats["n_deleted_total"]))

            # ---- M4 -------------------------------------------------------
            t0 = time.perf_counter()
            repair_stats = self.run_repair(delete_set, query_sample)
            t_repair = time.perf_counter() - t0
            if not repair_stats.get("repaired", False):
                stage_errors.append("M4: " + str(repair_stats.get("reason", "repair was not executed")))

            # ---- M5 -------------------------------------------------------
            t0 = time.perf_counter()
            forgotten_pids = self._unique_ints(delete_set)
            saved_control = None
            if control_pids is not None and isinstance(self.verifier, ForgetVerifier):
                saved_control = self.verifier.control_pids
                self.verifier.control_pids = list(control_pids)
            try:
                report, report_meta = self._run_verifier(forgotten_pids, query_sample)
            finally:
                if control_pids is not None and isinstance(self.verifier, ForgetVerifier) and saved_control is not None:
                    self.verifier.control_pids = saved_control
            if forgotten_qa is not None:
                probe = self._default_verifier()
                probe.forgotten_qa = list(forgotten_qa)
                report, report_meta = probe.verify_detailed(index, forgotten_pids, query_sample, self.k)
            t_verify = time.perf_counter() - t0
            self._log("M5 %s" % (report,))

            # ---- Cost accounting -----------------------------------------
            meter.touch(len(delete_set))
            if query_sample is not None:
                try:
                    meter.record(np.asarray(query_sample, dtype=np.float32))
                except Exception:
                    pass

        stats_after = dict(index.stats())
        total_time = time.perf_counter() - t_start
        dim = int(stats_before.get("dim", getattr(index, "dim", 0)) or 0)
        n_replicas = int(repair_stats.get("n_anchor_replicas", 0) or 0)
        bytes_transferred = int((len(delete_set) + n_replicas) * dim * 4)

        deleted_ids = sorted(int(i) for i in delete_set)
        try:
            deleted_docs = sorted({str(index.meta(i).doc_id) for i in deleted_ids})
        except Exception:
            deleted_docs = []

        cost = meter.as_dict()
        index_side = float(t_locate + t_closure + t_erase + t_repair)
        cost.update(
            {
                # Index-side cost (M1-M4: locating + closure + erasure + repair); M5 verification is evaluation and is not counted as reindexing cost
                "reindex_seconds": index_side,
                "pipeline_seconds": float(total_time),
                "total_seconds": float(total_time),
                "stage_seconds": {
                    "M1_locate": float(t_locate),
                    "M2_closure": float(t_closure),
                    "M3_erase": float(t_erase),
                    "M4_repair": float(t_repair),
                    "M5_verify": float(t_verify),
                },
                "bytes_transferred": bytes_transferred,
                "n_vectors_touched": int(len(delete_set) + n_replicas),
                "n_deleted": int(len(deleted_ids)),
                "n_alive": int(stats_after.get("n_alive", 0)),
                "stage_errors": stage_errors,
            }
        )

        certificate_payload = {
            "request_id": request_id,
            "timestamp": _utc_now_iso(),
            "n_deleted": int(len(deleted_ids)),
            "deleted_doc_ids": deleted_docs,
            "requested_doc_ids": requested,
            "deleted_ids": [int(i) for i in deleted_ids],
            "seed_ids": [int(i) for i in seeds],
            "closure_stats": closure_stats,
            "repair_stats": {k: v for k, v in repair_stats.items() if k != "calibration"},
            "verification": report.as_dict(),
            "verification_meta": dict(report_meta),
            "cost": {k: v for k, v in cost.items() if k != "stage_errors"},
            "stage_errors": list(stage_errors),
            "fingerprint_check": dict(fp_check),
            "index_stats_before": stats_before,
            "index_stats_after": stats_after,
            "client_id": self.client_id,
            "seed": int(self.seed),
            "k": int(self.k),
        }
        if extra:
            certificate_payload["extra"] = dict(extra)

        certificate, cert_path = self.write_certificate(certificate_payload)

        result = RevocationResult(
            deleted_ids=deleted_ids,
            report=report,
            cost=cost,
            certificate_path=str(cert_path),
            request_id=request_id,
            certificate=certificate,
            closure_stats=closure_stats,
            repair_stats=dict(repair_stats),
            seeds=[int(i) for i in seeds],
            n_deleted_new=int(erase_stats.get("n_deleted_new", 0)),
            fingerprint_check=dict(fp_check),
        )
        self.last_result = result
        return result


# --------------------------------------------------------------------------------------
# Certificate reading and chain verification
# --------------------------------------------------------------------------------------
def list_certificates(cert_dir: Any) -> list:
    """Read all certificates under a directory (sorted by filename; empty list if missing)."""
    root = Path(cert_dir)
    if not root.exists():
        return []
    files = [root] if root.is_file() else sorted(p for p in root.glob("*.json") if p.is_file())
    out = []
    for path in files:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if isinstance(data, dict) and "chain_sha256" in data:
            data.setdefault("_path", str(path))
            out.append(data)
    return out


def load_certificate(path: Any) -> dict:
    """Read a single certificate JSON."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def verify_certificate(path: Any, *, strict: bool = False) -> dict:
    """Verify the hash chain of deletion certificates.

    Parameters
    ----
    path : certificate file (.json) or certificate directory (artifacts/certificates/)
    strict : when True, require the first certificate to have seq=0 and prev_hash=GENESIS_HASH

    Returns
    ----
    {"valid", "n_certificates", "n_deleted_total", "errors", "head_sha256", "certificates", "path"}
    """
    p = Path(path)
    errors: list = []
    entries: list = []

    if p.is_file():
        try:
            entries = [(p, load_certificate(p))]
        except Exception as exc:
            return {"valid": False, "n_certificates": 0, "n_deleted_total": 0, "errors": ["read failed: %s" % exc],
                    "head_sha256": None, "certificates": [], "path": str(p)}
    elif p.is_dir():
        files = sorted(q for q in p.glob("*.json") if q.is_file())
        if not files:
            return {"valid": False, "n_certificates": 0, "n_deleted_total": 0,
                    "errors": ["no certificates in directory: %s" % p], "head_sha256": None,
                    "certificates": [], "path": str(p)}
        for q in files:
            try:
                entries.append((q, load_certificate(q)))
            except Exception as exc:
                errors.append("read failed %s: %s" % (q.name, exc))
    else:
        return {"valid": False, "n_certificates": 0, "n_deleted_total": 0, "errors": ["path does not exist: %s" % p],
                "head_sha256": None, "certificates": [], "path": str(p)}

    def _key(item):
        data = item[1] if isinstance(item[1], Mapping) else {}
        return (int(data.get("seq", 0)), str(data.get("timestamp", "")), str(item[0]))

    ordered = sorted(entries, key=_key)
    summaries: list = []
    for i, (q, data) in enumerate(ordered):
        if not isinstance(data, Mapping):
            errors.append("%s: not a JSON object" % q.name)
            continue
        stored = data.get("chain_sha256")
        missing = [f for f in ("request_id", "timestamp", "n_deleted", "prev_hash") if f not in data]
        if missing:
            errors.append("%s: missing contract field(s) %s" % (q.name, ",".join(missing)))
        recomputed = certificate_digest(data)
        if stored != recomputed:
            errors.append(
                "%s: chain_sha256 mismatch (stored %.12s / recomputed %.12s)" % (q.name, str(stored), recomputed)
            )
        if i > 0:
            prev = ordered[i - 1][1]
            expected = str(prev.get("chain_sha256"))
            if str(data.get("prev_hash")) != expected:
                errors.append(
                    "%s: prev_hash chain break (expected %.12s / actual %.12s)"
                    % (q.name, expected, str(data.get("prev_hash")))
                )
            if int(data.get("seq", 0)) <= int(prev.get("seq", 0)):
                errors.append("%s: seq not strictly increasing (%s <= %s)" % (q.name, data.get("seq"), prev.get("seq")))
        else:
            if strict and str(data.get("prev_hash")) != GENESIS_HASH:
                errors.append("%s: first certificate's prev_hash should be the genesis hash" % q.name)
            if strict and int(data.get("seq", 0)) != 0:
                errors.append("%s: first certificate's seq should be 0 (actual %s)" % (q.name, data.get("seq")))
        summaries.append(
            {
                "path": str(data.get("_path", q)),
                "seq": int(data.get("seq", 0)),
                "request_id": data.get("request_id"),
                "timestamp": data.get("timestamp"),
                "n_deleted": int(data.get("n_deleted", 0) or 0),
                "prev_hash": data.get("prev_hash"),
                "chain_sha256": stored,
            }
        )

    head = summaries[-1]["chain_sha256"] if summaries else None
    return {
        "valid": bool(not errors and summaries),
        "n_certificates": len(summaries),
        "n_deleted_total": int(sum(s["n_deleted"] for s in summaries)),
        "errors": errors,
        "head_sha256": head,
        "certificates": summaries,
        "path": str(p),
    }
