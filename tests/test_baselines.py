"""Synthetic-data unit tests for the 5 baselines in baselines.py (INTERFACES.md §9).

Coverage:
  * Unified interface consistency ({"forget","utility","cost"} + required keys + determinism)
  * Sanity check: FullRebuild utility is no worse than NaiveDelete, and cost (bytes / vectors touched) is strictly higher
  * Claim 1: NaiveDelete/FullRebuild preserve cross-silo shadows (residual_doc_rate_cond = 1.0)
  * TDSCAdapter: rewrites the knowledge base instead of deleting documents; LoRAFinetune: honestly reports degradation when dependencies are missing

Run: cd fedrevoke; .venv/Scripts/python.exe -m pytest tests -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fedrevoke.baselines import (  # noqa: E402
    BASELINES,
    FullRebuild,
    LoRAFinetune,
    NaiveDelete,
    SISA,
    TDSCAdapter,
    available_baselines,
    build_baseline,
    clone_index,
)
from fedrevoke.generation import MockGenerator  # noqa: E402
from fedrevoke.index_core import ProvenanceIndex, VecMeta  # noqa: E402
from fedrevoke.run_experiments import build_qa_items, synthesize_dataset  # noqa: E402

SEED = 20260214
DIM = 32
N_DOCS = 400
N_CLIENTS = 3
SHADOW_RATIO = 0.30
METHODS = ["full_rebuild", "naive_delete", "sisa", "lora_finetune", "tdsc_adapter"]


# --------------------------------------------------------------------------------------
def make_scene(n_docs=N_DOCS, shadow_ratio=SHADOW_RATIO, dim=DIM, n_queries=12):
    ds = synthesize_dataset(
        n_docs=n_docs, n_clients=N_CLIENTS, n_queries=n_queries, shadow_ratio=shadow_ratio,
        dim=dim, seed=SEED, forget_ratios=[0.05], n_qa=3, shadow_sim=0.96,
    )
    index = ProvenanceIndex(dim, backend="numpy")
    index.add(ds["vectors"], [
        VecMeta(pid=int(r["pid"]), doc_id=str(r["doc_id"]), client_id=str(r["client_id"]),
                topic=str(r["topic"]), fingerprint=str(r["fingerprint"]))
        for r in ds["corpus"]
    ])
    texts = ds["texts"]
    index.text_for_id = lambda i: texts.get(int(i), "")
    return ds, index


def make_kw(ds, doc_ids, n_gold_queries=8):
    """Evaluation kw: gold is deliberately drawn from alive documents unrelated to the revoked ones (utility sanity-check setting)."""
    revoked_pids = {int(p) for d in doc_ids for p in ds["doc_pids"].get(str(d), [])}
    alive_pool = [int(r["pid"]) for r in ds["corpus"]
                  if int(r["pid"]) not in revoked_pids
                  and str(r["doc_id"]) not in set(ds["shadow_docs"])]
    rng = np.random.default_rng(SEED)
    qvecs, gold = [], []
    for i in range(min(n_gold_queries, len(ds["qvecs"]))):
        pick = rng.choice(len(alive_pool), size=min(8, len(alive_pool)), replace=False)
        gold_pids = sorted(alive_pool[int(j)] for j in pick)
        gold.append(gold_pids)
        qvecs.append(ds["vectors"][gold_pids[0]])
    return dict(
        query_sample=np.stack(qvecs).astype(np.float32),
        gold_pids=gold,
        qa_items=build_qa_items(ds, doc_ids, max_items=3, seed=SEED),
        generator=MockGenerator("MOCK"),
        shadow_surrogates=ds["shadow_map"],
        k=10,
        seed=SEED,
        text_store={int(k): v for k, v in ds["texts"].items()},
    )


def shadowed_docs(ds, k=3):
    return sorted(ds["shadow_map"])[:k]


# --------------------------------------------------------------------------------------
# 1) Unified interface
# --------------------------------------------------------------------------------------
def test_registry_and_aliases():
    # 5 paper baselines + 1 repair causal control (random_replica)
    avail = set(available_baselines())
    assert set(METHODS) <= avail
    assert "random_replica" in avail
    for name in METHODS:
        assert build_baseline(name).name == name
    assert isinstance(build_baseline("full-rebuild"), FullRebuild)
    assert isinstance(build_baseline("SISA"), SISA)
    assert build_baseline("control").name == "random_replica"
    with pytest.raises(ValueError):
        build_baseline("no_such_baseline")


def test_random_replica_control_is_not_a_repair():
    """U3 causal control: the random-replica control arm's recall recovery should be significantly lower than AnchorRepair (fedrevoke)."""
    ds, index = make_scene()
    docs = shadowed_docs(ds, 3)
    kw = make_kw(ds, docs, n_gold_queries=10)
    out = build_baseline("random_replica", n_replicas=32).run(index, docs, **kw)
    assert out["cost"]["n_deleted"] >= 1
    assert out["meta"]["n_replica_control"] >= 1
    assert out["meta"]["n_vectors_delta"] >= out["meta"]["n_replica_control"] - len(docs)
    assert 0.0 <= out["utility"]["recall10"] <= 1.0


@pytest.mark.parametrize("method", METHODS)
def test_unified_interface_and_determinism(method):
    ds, index = make_scene()
    docs = shadowed_docs(ds, 2)
    kw = make_kw(ds, docs)
    n_alive_before = index.stats()["n_alive"]

    out1 = build_baseline(method).run(index, docs, **kw)
    out2 = build_baseline(method).run(index, docs, **kw)

    for key in ("forget", "utility", "cost", "method", "meta"):
        assert key in out1, "%s is missing unified return key %s" % (method, key)
    for key in ("hit_rate", "mia_auc", "elicit_rate", "rho_hat", "residual_doc_rate"):
        assert key in out1["forget"], "%s forget is missing %s" % (method, key)
    for key in ("recall10", "ndcg10", "em", "f1", "n_queries"):
        assert key in out1["utility"], "%s utility is missing %s" % (method, key)
    for key in ("reindex_seconds", "bytes_transferred", "peak_vram_mb", "n_vectors_touched"):
        assert key in out1["cost"], "%s cost is missing %s" % (method, key)
        assert float(out1["cost"][key]) >= 0.0

    assert out1["method"] == method
    assert out1["cost"]["n_deleted"] >= 0
    # Determinism: two runs on the same input yield identical metrics
    assert out1["utility"]["recall10"] == pytest.approx(out2["utility"]["recall10"])
    assert out1["forget"]["residual_doc_rate"] == pytest.approx(out2["forget"]["residual_doc_rate"])
    assert out1["cost"]["bytes_transferred"] == out2["cost"]["bytes_transferred"]
    # By default the passed-in index is not modified (clone semantics)
    assert index.stats()["n_alive"] == n_alive_before


# --------------------------------------------------------------------------------------
# 2) Sanity check: FullRebuild >= NaiveDelete (utility), cost strictly higher
# --------------------------------------------------------------------------------------
def test_full_rebuild_utility_not_worse_and_costs_more():
    ds, index = make_scene()
    docs = shadowed_docs(ds, 3)
    kw = make_kw(ds, docs, n_gold_queries=10)

    full = FullRebuild().run(index, docs, **kw)
    naive = NaiveDelete().run(index, docs, **kw)

    # Intuition: a full rebuild should not be worse than "delete only this silo" (no faiss on this machine; under the exact backend the two are usually equal)
    assert full["utility"]["recall10"] >= naive["utility"]["recall10"] - 0.05, (
        "FullRebuild recall=%.4f should be no lower than NaiveDelete recall=%.4f"
        % (full["utility"]["recall10"], naive["utility"]["recall10"])
    )
    # Cost: full rebuild's transferred bytes and vectors touched are strictly higher (by at least one order of magnitude)
    assert full["cost"]["bytes_transferred"] > naive["cost"]["bytes_transferred"] * 10
    assert full["cost"]["n_vectors_touched"] > naive["cost"]["n_vectors_touched"]
    # Deletion scope: full rebuild deletes all requested shards; NaiveDelete only deletes the local silo
    full_removed = len(full["meta"]["seed_ids"])
    naive_removed = naive["cost"]["n_deleted"]
    assert naive_removed <= full_removed
    assert naive["meta"]["n_remote_left_alive"] >= 0
    # SISA: deletion scope is the same as FullRebuild, but the accounted rebuild size is no larger than full
    sisa = SISA().run(index, docs, **kw)
    assert sisa["cost"]["bytes_transferred"] <= full["cost"]["bytes_transferred"]
    assert sisa["cost"]["n_vectors_touched"] <= full["cost"]["n_vectors_touched"]


# --------------------------------------------------------------------------------------
# 3) Claim 1: cross-silo shadows leave residue even after "clean" shard deletion
# --------------------------------------------------------------------------------------
def test_shadow_residual_survives_shard_deletion():
    ds, index = make_scene()
    docs = shadowed_docs(ds, 4)
    kw = make_kw(ds, docs)
    assert all(ds["shadow_map"].get(d) for d in docs), "the test scene must include cross-silo shadows"

    full = FullRebuild().run(index, docs, **kw)
    naive = NaiveDelete().run(index, docs, **kw)

    # Shadow copies belong to other doc_ids; deleting the original shard cannot touch them -- this is the experimental basis of Claim 1
    assert full["forget"]["residual_doc_rate_cond"] == pytest.approx(1.0)
    assert naive["forget"]["residual_doc_rate_cond"] == pytest.approx(1.0)
    assert full["forget"]["n_surviving_surrogate_docs"] == len(docs)


# --------------------------------------------------------------------------------------
# 4) TDSCAdapter: rewrites the knowledge base instead of deleting documents
# --------------------------------------------------------------------------------------
def test_tdsc_adapter_rewrites_instead_of_deleting():
    ds, index = make_scene()
    docs = shadowed_docs(ds, 3)
    kw = make_kw(ds, docs)
    store = kw["text_store"]
    before = {d: list(ds["shadow_map"][d]) for d in docs}
    n_alive_before = index.stats()["n_alive"]

    out = TDSCAdapter().run(index, docs, **kw)

    assert out["cost"]["n_deleted"] == 0
    assert out["meta"]["deletes_documents"] is False
    assert out["meta"]["n_shadow_rewritten"] >= 1
    assert out["meta"]["n_rewritten_texts"] >= 1
    # Documents remain in the index (utility preserved), but shadow texts have been rewritten
    assert out["index"].stats()["n_alive"] == n_alive_before
    shadow_doc = before[docs[0]][0]
    sid = out["index"].ids_for_doc(shadow_doc)
    assert sid, "TDSC must not delete shadow documents"
    assert store[sid[0]].startswith("[REDACTED]")
    # Cost is far below a full rebuild
    full = FullRebuild().run(index, docs, **kw)
    assert out["cost"]["bytes_transferred"] < full["cost"]["bytes_transferred"]


# --------------------------------------------------------------------------------------
# 5) LoRAFinetune: degrade and honestly report when dependencies are missing
# --------------------------------------------------------------------------------------
def test_lora_finetune_reports_degradation_and_trains_adapter(monkeypatch):
    """Must degrade and honestly report when dependencies are missing.

    The availability probe is explicitly monkeypatched so this case does **not** drift with the environment:
    once peft is installed in the venv and bge weights are cached, the original assertion
    `real_lora_available is False` would fail due to environment changes (testing the environment
    rather than the contract). The real branch is covered by
    test_lora_finetune_real_branch_when_dependencies_present.
    """
    import fedrevoke.baselines as _bl

    monkeypatch.setattr(_bl, "_real_lora_available",
                        lambda: (False, "monkeypatched: peft/local weights unavailable"))

    ds, index = make_scene()
    docs = shadowed_docs(ds, 2)
    kw = make_kw(ds, docs)
    out = LoRAFinetune(rank=4, steps=30, max_train=128).run(index, docs, **kw)

    assert out["meta"]["real_lora_available"] is False
    assert out["meta"]["degraded"] is True
    assert out["meta"]["degradation_reason"], "must state the degradation reason (missing transformers/peft or local weights)"
    assert out["meta"]["n_reencoded"] > 0
    adapter = out["meta"]["adapter"]
    assert adapter["rank"] == 4 and adapter["n_train"] > 0
    assert adapter["train_seconds"] >= 0.0
    # Still produces valid metrics after training
    assert 0.0 <= out["utility"]["recall10"] <= 1.0
    assert 0.0 <= out["forget"]["residual_doc_rate"] <= 1.0


def test_lora_finetune_real_branch_when_dependencies_present():
    """When dependencies are complete (peft + local encoder weights), take the real LoRA branch: no degradation, and an adapter is actually trained.

    This case guards B4 baseline credibility: the paper treats B4 as a real LoRA unlearning baseline,
    so it must be shown that it really trains an adapter rather than falling back to a
    representation-space approximation.
    """
    import pytest

    import fedrevoke.baselines as _bl

    ok, reason = _bl._real_lora_available()
    if not ok:
        pytest.skip("real LoRA unavailable: %s" % (reason,))

    ds, index = make_scene()
    docs = shadowed_docs(ds, 2)
    kw = make_kw(ds, docs)
    out = _bl.LoRAFinetune(rank=4, steps=2, max_train=8).run(index, docs, **kw)

    assert out["meta"]["real_lora_available"] is True
    # Real LoRA must actually train (this is the core assertion of B4 baseline credibility)
    assert out["meta"]["lora_mode"] == "real_lora"
    adapter = out["meta"]["adapter"]
    # The adapter field schema of the real LoRA branch differs from the degraded linear adapter (no n_train; has n_retained/n_trainable_tensors)
    assert adapter["rank"] == 4
    assert adapter.get("mode") == "real_lora"
    assert adapter.get("n_trainable_tensors", 0) > 0, "real LoRA must attach trainable tensors"
    assert adapter.get("n_retained", 0) > 0, "must actually train on retained samples"
    # Whether re-encoding is possible depends on whether the index dimension matches the encoder dimension (this case's synthetic scene is 32-d,
    # while bge-small outputs 384-d). The contract: on mismatch, must **honestly report the degradation reason** rather than silently fail on broadcasting.
    if int(getattr(index, "dim", 0)) == 384:
        assert out["meta"]["degraded"] is False
    else:
        assert out["meta"]["degraded"] is True
        assert "dim" in out["meta"]["degradation_reason"].lower() or \
               "shape" in out["meta"]["degradation_reason"].lower(), out["meta"]["degradation_reason"]
    assert 0.0 <= out["utility"]["recall10"] <= 1.0
    assert 0.0 <= out["forget"]["residual_doc_rate"] <= 1.0


# --------------------------------------------------------------------------------------
# 6) clone_index semantics (identical starting point for every method)
# --------------------------------------------------------------------------------------
def test_clone_index_preserves_ids_and_tombstones():
    ds, index = make_scene()
    docs = shadowed_docs(ds, 2)
    ids = [i for d in docs for i in index.ids_for_doc(d)][:5]
    index.remove(ids)
    clone = clone_index(index)
    assert clone.stats()["n_vectors"] == index.stats()["n_vectors"]
    assert sorted(clone.deleted_ids().tolist()) == sorted(index.deleted_ids().tolist())
    assert clone.ids_for_doc(docs[0]) == index.ids_for_doc(docs[0])
    s1, i1 = index.search(ds["qvecs"][:3], 5)
    s2, i2 = clone.search(ds["qvecs"][:3], 5)
    assert np.allclose(s1, s2, atol=1e-5)
    assert np.array_equal(i1, i2)
