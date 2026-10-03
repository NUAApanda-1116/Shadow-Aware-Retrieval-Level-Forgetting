"""Synthetic-data unit tests for revoke.py (INTERFACES.md §8 M1–M6 + §11 fingerprint gate).

Constraints: synthetic data only; no network, no HF model loading, no writing to data/.
Run: cd fedrevoke; .venv/Scripts/python.exe -m pytest tests -q
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fedrevoke.generation import MockGenerator  # noqa: E402
from fedrevoke.index_core import ProvenanceIndex, VecMeta  # noqa: E402
from fedrevoke.repair import AnchorRepair  # noqa: E402
from fedrevoke.revoke import (  # noqa: E402
    GENESIS_HASH,
    RevocationPipeline,
    RevocationResult,
    assert_fingerprint_channel,
    verify_certificate,
)
from fedrevoke.run_experiments import synthesize_dataset  # noqa: E402
from fedrevoke.shadow import ShadowDetector  # noqa: E402
from fedrevoke.verify import ForgetReport  # noqa: E402

SEED = 20260214
DIM = 32
N_DOCS = 400
N_CLIENTS = 3
SHADOW_RATIO = 0.30

CERT_FIELDS = ("request_id", "timestamp", "n_deleted", "deleted_doc_ids", "closure_stats",
               "prev_hash", "chain_sha256")


# --------------------------------------------------------------------------------------
# Synthetic scenario
# --------------------------------------------------------------------------------------
def make_scenario(n_docs=N_DOCS, shadow_ratio=SHADOW_RATIO, dim=DIM, n_queries=20):
    """Synthetic "cross-silo shadow injection" corpus + index + detector (deterministic)."""
    ds = synthesize_dataset(
        n_docs=n_docs, n_clients=N_CLIENTS, n_queries=n_queries, shadow_ratio=shadow_ratio,
        dim=dim, seed=SEED, forget_ratios=[0.05], n_qa=3, shadow_sim=0.96,
    )
    index = ProvenanceIndex(dim, backend="numpy")
    metas = [
        VecMeta(pid=int(r["pid"]), doc_id=str(r["doc_id"]), client_id=str(r["client_id"]),
                topic=str(r["topic"]), fingerprint=str(r["fingerprint"]))
        for r in ds["corpus"]
    ]
    index.add(ds["vectors"], metas)
    texts = ds["texts"]
    index.text_for_id = lambda i: texts.get(int(i), "")
    detector = ShadowDetector(sim_threshold=0.92, lsh_threshold=0.80, knn_k=20,
                              cross_client_only=True, signatures=ds["signatures"], seed=SEED)
    shadowed = sorted(ds["shadow_map"])  # original docs that have cross-silo shadow copies
    return ds, index, detector, shadowed


def make_pipeline(index, detector, cert_dir, generator=None, shadow_ratio=SHADOW_RATIO, smoke=True):
    return RevocationPipeline(
        index=index, detector=detector, repair=AnchorRepair(n_anchors=16, seed=SEED),
        verifier=None, cert_dir=cert_dir, encoder=None, k=10,
        generator=generator or MockGenerator("MOCK"), shadow_ratio=shadow_ratio, smoke=smoke,
    )


# --------------------------------------------------------------------------------------
# M1–M6 full pipeline
# --------------------------------------------------------------------------------------
def test_revoke_full_flow_writes_chained_certificate(tmp_path):
    ds, index, detector, shadowed = make_scenario()
    qa = [{"query": "q", "answers": ["a"], "qvec": ds["vectors"][0]}]
    pipe = make_pipeline(index, detector, tmp_path / "certs", shadow_ratio=SHADOW_RATIO)
    doc = shadowed[0]
    seed_ids = index.ids_for_doc(doc)
    assert seed_ids, "seed document should exist in the index"

    res = pipe.revoke([doc], ds["qvecs"][:5], forgotten_qa=qa)

    # --- RevocationResult contract ---
    assert isinstance(res, RevocationResult)
    assert isinstance(res.report, ForgetReport)
    assert isinstance(res.cost, dict) and isinstance(res.deleted_ids, list)
    assert Path(res.certificate_path).exists()

    # --- M1/M2: closure includes the seed itself and hits cross-silo shadows ---
    assert set(seed_ids) <= set(res.deleted_ids)
    shadow_ids = [i for s in ds["shadow_map"][doc] for i in index.all_ids_for_doc(s)]
    assert shadow_ids, "the scenario should include shadow copies"
    assert set(shadow_ids) <= set(res.deleted_ids), "the shadow closure must cover the cross-silo copies"
    assert res.closure_stats["n_closure"] == len(res.deleted_ids)

    # --- M3: after tombstone deletion, search no longer returns deleted ids; stats stay in sync ---
    assert res.cost["n_deleted"] == len(res.deleted_ids)
    assert index.stats()["n_deleted"] == len(res.deleted_ids)
    _, got = index.search(ds["qvecs"][:5], 10)
    assert not (set(res.deleted_ids) & {int(v) for v in np.asarray(got).ravel().tolist()})
    for deleted in res.deleted_ids:
        assert index.ids_for_doc(index.meta(deleted).doc_id).count(deleted) == 0

    # --- M5: the verification report is written into the certificate ---
    cert = json.loads(Path(res.certificate_path).read_text(encoding="utf-8"))
    for field in CERT_FIELDS:
        assert field in cert, "certificate is missing contract field %s" % field
    assert cert["chain_sha256"] == res.certificate["chain_sha256"]
    assert cert["verification"]["hit_rate"] == pytest.approx(res.report.hit_rate)
    assert cert["verification_meta"]["elicit_measured"] is True
    assert cert["closure_stats"]["n_closure"] == len(res.deleted_ids)
    assert cert["deleted_doc_ids"], "the certificate must record the deleted doc_id"

    # --- M6: hash chain ---
    chk = verify_certificate(tmp_path / "certs", strict=True)
    assert chk["valid"] is True, chk["errors"]
    assert chk["n_certificates"] == 1
    assert chk["certificates"][0]["prev_hash"] == GENESIS_HASH


def test_certificate_chain_links_and_detects_tampering(tmp_path):
    ds, index, detector, shadowed = make_scenario()
    cert_dir = tmp_path / "certs"
    pipe = make_pipeline(index, detector, cert_dir)
    first = pipe.revoke([shadowed[0]], ds["qvecs"][:3])
    second = pipe.revoke([shadowed[1]], ds["qvecs"][:3])

    chk = verify_certificate(cert_dir, strict=True)
    assert chk["valid"] is True and chk["n_certificates"] == 2
    c0, c1 = chk["certificates"]
    assert c1["prev_hash"] == c0["chain_sha256"]
    assert c0["seq"] == 0 and c1["seq"] == 1
    assert chk["head_sha256"] == c1["chain_sha256"] == second.certificate["chain_sha256"]

    # Tampering with any field -> chain verification fails (hash mismatch)
    blob = json.loads(Path(first.certificate_path).read_text(encoding="utf-8"))
    blob["n_deleted"] = 999999
    Path(first.certificate_path).write_text(json.dumps(blob, ensure_ascii=False), encoding="utf-8")
    bad = verify_certificate(cert_dir)
    assert bad["valid"] is False
    assert any("chain_sha256" in e for e in bad["errors"])

    # Broken-chain detection: restore the self-hash but point prev_hash wrongly
    blob["n_deleted"] = 1
    from fedrevoke.revoke import certificate_digest
    blob.pop("chain_sha256")
    blob["prev_hash"] = "f" * 64
    blob["chain_sha256"] = certificate_digest(blob)
    Path(first.certificate_path).write_text(json.dumps(blob, ensure_ascii=False), encoding="utf-8")
    broken = verify_certificate(cert_dir)
    assert broken["valid"] is False
    assert any(("prev_hash" in e) or ("broken" in e.lower()) or ("chain" in e.lower()) for e in broken["errors"])


def test_revoke_is_idempotent_and_handles_missing_docs(tmp_path):
    ds, index, detector, shadowed = make_scenario()
    pipe = make_pipeline(index, detector, tmp_path / "certs")
    first = pipe.revoke([shadowed[0]], ds["qvecs"][:3])
    assert first.n_deleted_new > 0
    second = pipe.revoke([shadowed[0]], ds["qvecs"][:3])
    assert second.n_deleted_new == 0, "revoking the same document twice must not produce new deletions (idempotent)"
    assert second.seeds == []
    assert second.deleted_ids == []

    third = pipe.revoke(["d999999"], ds["qvecs"][:3])
    assert third.deleted_ids == []
    assert any(("not exist" in e.lower()) or ("not found" in e.lower()) or ("missing" in e.lower())
               for e in third.cost["stage_errors"])
    chk = verify_certificate(tmp_path / "certs")
    assert chk["valid"] is True and chk["n_certificates"] == 3


def test_certificate_before_revoke_is_empty_contract(tmp_path):
    ds, index, detector, shadowed = make_scenario()
    pipe = make_pipeline(index, detector, tmp_path / "certs")
    cert = pipe.certificate()
    for field in CERT_FIELDS:
        assert field in cert
    assert cert["n_deleted"] == 0 and cert["chain_sha256"] is None


# --------------------------------------------------------------------------------------
# INTERFACES §11: fingerprint-channel fail-fast
# --------------------------------------------------------------------------------------
def _short_fingerprint_index(n=60, dim=DIM):
    """Build an index that uses short-digest fingerprints (blake2b 8 bytes), mimicking the data_prep corpus.jsonl fields."""
    rng = np.random.default_rng(SEED)
    vecs = rng.normal(size=(n, dim)).astype(np.float32)
    metas = [
        VecMeta(pid=i, doc_id="d%05d" % i, client_id="c%d" % (i % N_CLIENTS), topic="t0",
                fingerprint="%016x" % i)  # 16 hex = 2 32-bit words < MIN_SIG_WORDS(16) -> digest
        for i in range(n)
    ]
    index = ProvenanceIndex(dim, backend="numpy")
    index.add(vecs, metas)
    return index


def test_fingerprint_gate_raises_in_full_mode_and_warns_in_smoke(capsys):
    index = _short_fingerprint_index()
    detector = ShadowDetector(sim_threshold=0.9, lsh_threshold=0.8, knn_k=5, signatures=None, seed=SEED)

    with pytest.raises(RuntimeError) as excinfo:
        assert_fingerprint_channel(index, detector, shadow_ratio=0.30, smoke=False)
    assert "fingerprint" in str(excinfo.value).lower() or "digest" in str(excinfo.value).lower()

    info = assert_fingerprint_channel(index, detector, shadow_ratio=0.30, smoke=True)
    assert info["degraded"] is True
    assert info["counts"].get("digest", 0) > 0
    assert "WARNING" in capsys.readouterr().out

    # No interception when shadow_ratio == 0 (no shadow closure involved)
    clean = assert_fingerprint_channel(index, detector, shadow_ratio=0.0, smoke=False)
    assert clean["degraded"] is False


def test_signature_matrix_is_reported_as_provided(tmp_path):
    ds, index, detector, shadowed = make_scenario()
    info = assert_fingerprint_channel(index, detector, shadow_ratio=0.30, smoke=False)
    assert info["degraded"] is False
    assert info["counts"].get("provided", 0) == index.n_alive
    assert info["median_sig_len"] == ds["signatures"].shape[1]


# --------------------------------------------------------------------------------------
# Cost and stage timings
# --------------------------------------------------------------------------------------
def test_cost_fields_and_stage_timings(tmp_path):
    ds, index, detector, shadowed = make_pipeline_scenario = make_scenario()
    pipe = make_pipeline(index, detector, tmp_path / "certs")
    res = pipe.revoke([shadowed[0]], ds["qvecs"][:10])
    for key in ("reindex_seconds", "pipeline_seconds", "bytes_transferred", "peak_vram_mb",
                "n_vectors_touched", "stage_seconds"):
        assert key in res.cost, "cost is missing %s" % key
    for stage in ("M1_locate", "M2_closure", "M3_erase", "M4_repair", "M5_verify"):
        assert stage in res.cost["stage_seconds"]
    assert res.cost["reindex_seconds"] > 0
    assert res.cost["n_vectors_touched"] >= len(res.deleted_ids)
    assert res.cost["stage_errors"] == []
    assert res.repair_stats.get("repaired") is True
