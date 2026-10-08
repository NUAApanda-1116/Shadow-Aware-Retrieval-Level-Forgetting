"""run_experiments.py —— FedRevoke 实验总入口（INTERFACES.md §10）。

用法::

    python -m fedrevoke.run_experiments --config configs/e1_main.yaml [--smoke] [--limit N]

输出::

    artifacts/results/<exp>.csv          # 自解释列名（见 CSV_REQUIRED_COLUMNS）
    artifacts/results/<exp>_summary.json # 配置 + 聚合 + 环境/降级说明
    artifacts/figures/<exp>_e6_motivation.{pdf,png}   # 动机图：朴素删除残留 vs 影子比例
    artifacts/figures/<exp>_e5_pareto.{pdf,png}       # 代价 Pareto：残留泄漏 vs 重索引代价
    artifacts/logs/<exp>_run.json        # 运行日志（含 warning/degradation）

设计要点
--------
* **synthetic 模式（--smoke / dataset=synthetic）**：现场合成向量与文档，
  不联网、不加载任何 HF 模型、不读 data/；MockGenerator 做生成端评测；60 秒内跑完。
* **真实数据模式**：读 data/processed/<ds>/（corpus.jsonl / queries.jsonl / forget_sets.json /
  shadow_pairs.json / embeddings_*.npy / minhash_sig.npy）。数据缺失时**只跑 smoke**
  并把缺失清单写入 summary.json 的 data_status。
* **INTERFACES §11 指纹裁决**：凡 shadow_ratio > 0 的实验，用
  load_bundle(ds)["signatures"] → np.load(ds/minhash_sig.npy) 的顺序取签名矩阵，
  ShadowDetector(signatures=...) 注入；建索引后调用
  RevocationPipeline.check_fingerprints()（内部走 detector.diagnose_fingerprints）；
  full 模式 digest 指纹 > 0 直接 RuntimeError，--smoke 打印 WARNING 继续。
  基线 B2/B5 走同一签名矩阵，保证信息对等。
* **绘图**：本机 .venv 无 matplotlib（且禁止 pip install），
  因此内置一个无依赖绘图后端：同一份矢量场景同时输出
  * PDF（矢量线段 + PDF 标准字体 Helvetica 文本，可无损缩放，适合投稿）
  * PNG（300 dpi 栅格化 + 内置 5x7 位图字体）
  误差棒在有多 seed 时自动绘制。
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import sys
import time
import traceback
import warnings
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

try:  # 允许 "python -m fedrevoke.run_experiments"
    from . import config as C
except ImportError:  # pragma: no cover
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from fedrevoke import config as C

from .baselines import (
    BASELINES,
    EvalContext,
    available_baselines,
    build_baseline,
    build_eval_context,
    clone_index,
    evaluate_delta_utility,
    evaluate_forget,
    evaluate_utility,
    residual_stats,
    rho_hat_tv,
)
from .generation import MockGenerator, build_generator
from .index_core import ProvenanceIndex, VecMeta
from .repair import AnchorRepair
from .revoke import RevocationPipeline, assert_fingerprint_channel, verify_certificate
from .shadow import ShadowDetector, fingerprint_text, load_signature_matrix, minhash_signature

try:  # PyYAML 在 .venv 中可用；缺失时用极简 YAML 子集解析器兜底
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None

CFG_DIR = C.ROOT / "configs"
RESULTS_DIR = C.RESULTS
FIGURES_DIR = C.FIGURES
LOGS_DIR = C.LOGS
CERT_DIR = C.ARTIFACTS / "certificates"

# Tier B 的查询上限（唯一定义点）：生效值 = min(config.n_queries_eval, TIER_B_QUERY_CAP)。
# 见 apply_tier_overrides 的 docstring；论文 §Metrics 与 §Implementation 必须与此一致。
TIER_B_QUERY_CAP = 1000

CSV_REQUIRED_COLUMNS = [
    "exp", "dataset", "silos", "revocation_ratio", "shadow_ratio", "method",
    "hit_rate", "mia_auc", "elicit_rate", "recall10", "ndcg10", "em", "f1",
    "reindex_seconds", "bytes_transferred", "peak_vram_mb", "n_vectors_touched", "seed",
]
CSV_EXTRA_COLUMNS = [
    "rho_hat", "rho_hat_tv", "rho_hat_ret", "rho_hat_bound", "n_unrel", "unrel_reliable", "recall10_unrel", "recall10_unrel_before",
    "delta_recall10_unrel", "ndcg10_unrel", "delta_ndcg10_unrel", "utility_primary", "primary_value",
    "faithfulness", "n_official", "ndcg10_official", "recall10_official", "n_synthetic_q",
    "ndcg10_synthetic", "recall10_synthetic",
    "residual_doc_rate", "residual_doc_rate_cond", "n_shadowed_forgotten", "n_forgotten_docs",
    "self_probe_residual", "real_query_residual", "real_query_residual_k50",
    "retrieved_surrogate_fraction_realq", "revoked_evidence_residual", "unrel_query_residual",
    "n_self_probes", "n_real_queries", "n_revoked_evidence_queries",
    "n_unrel_queries", "retrieved_surrogate_fraction", "n_surviving_surrogate_docs",
    "detector_variant", "sim_threshold", "lsh_threshold", "n_pairs", "miss_rate", "mean_closure_size",
    "hit_seed", "hit_surrogate", "n_replica_control", "n_vectors_delta",
    "m2_closure_s", "m3_erase_s", "m4_repair_s", "m2_share", "m3_share",
    "n_qa_items", "elicit_flags", "gold_ref_scope",
    "elicit_rate_closed_book", "elicit_flags_closed_book",
    "surrogate_hit_rate", "n_surrogates", "n_queries", "n_deleted", "n_alive_after",
    "recall_min", "recall_std", "n_seed", "n_closure", "closure_precision", "closure_recall",
    "n_reconnected", "score_shift", "elicit_measured", "notes",
]
CSV_COLUMNS = CSV_REQUIRED_COLUMNS + CSV_EXTRA_COLUMNS

NAN = float("nan")


# ======================================================================================
# 0. 配置加载
# ======================================================================================
DEFAULT_CFG: Dict[str, Any] = {
    "exp": "e0_smoke",
    "mode": "smoke",
    "dataset": "synthetic",
    "datasets": None,
    "silos": 3,
    "silos_grid": None,
    "dirichlet_alpha": 0.5,
    "dirichlet_alpha_grid": None,
    "n_docs": 2000,
    "n_queries_eval": 50,
    "revocation_ratios": [0.05],
    "shadow_ratios": [0.10],
    "encoder": "synthetic",
    "generator": "mock",
    "generator_4bit": True,
    "k": 10,
    "method": "fedrevoke",
    "baselines": ["full_rebuild", "naive_delete"],
    "variants": None,
    "knn_grid": None,
    "sim_threshold_grid": None,
    "corpus_sizes": None,
    "shadow": {"sim_threshold": 0.92, "lsh_threshold": 0.80, "knn_k": 20, "cross_client_only": True},
    "repair": {"n_anchors": 64, "recalibrate": True},
    "seed": int(C.SEED),
    "out_dir": None,
    "seeds": None,
    "n_topics": 5,
}


def _parse_scalar(text: str) -> Any:
    """极简 YAML 标量解析（PyYAML 缺失时的兜底）。"""
    t = text.strip()
    if t in ("null", "~", ""):
        return None
    if t.lower() in ("true", "false"):
        return t.lower() == "true"
    if t.startswith("[") and t.endswith("]"):
        inner = t[1:-1].strip()
        return [] if not inner else [_parse_scalar(x) for x in inner.split(",")]
    if t.startswith("{") and t.endswith("}"):
        inner = t[1:-1].strip()
        out: Dict[str, Any] = {}
        if inner:
            for part in inner.split(","):
                k, _, v = part.partition(":")
                out[k.strip()] = _parse_scalar(v)
        return out
    try:
        return int(t)
    except ValueError:
        pass
    try:
        return float(t)
    except ValueError:
        pass
    return t.strip("'\"")


def _mini_yaml(text: str) -> Dict[str, Any]:
    """支持本项目 configs/*.yaml 的极小子集（顶层键 + 行内 dict/list + 注释）。"""
    cfg: Dict[str, Any] = {}
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].rstrip()
        if not line.strip() or line.lstrip() != line:
            continue
        key, sep, value = line.partition(":")
        if not sep:
            continue
        cfg[key.strip()] = _parse_scalar(value)
    return cfg


def load_config(path: Any) -> Dict[str, Any]:
    """读取 YAML 配置并与默认值合并。"""
    p = Path(path)
    if not p.is_absolute():
        p = (C.ROOT / p) if (C.ROOT / p).exists() else p
    text = p.read_text(encoding="utf-8")
    data = yaml.safe_load(text) if yaml is not None else _mini_yaml(text)
    cfg = dict(DEFAULT_CFG)
    cfg.update({k: v for k, v in (data or {}).items() if v is not None or k in ("generator", "reranker")})
    # 多数据集配置不应继承 DEFAULT_CFG 的 dataset=synthetic
    if (data or {}).get("datasets") and not (data or {}).get("dataset"):
        cfg["dataset"] = str(list(data["datasets"])[0])
    cfg["_config_path"] = str(p)
    return cfg


def apply_tier_overrides(cfg: Mapping[str, Any], tier: str, model: str, smoke: bool = False) -> Dict[str, Any]:
    """两档评测（父代理裁决）：Tier A 纯检索全网格；Tier B 含生成、缩减网格。

    Tier A: 不调生成器（elicit/EM/F1 留空并标 elicit_measured=False），网格可全跑。
    Tier B: 调 Qwen2.5-1.5B/7B-4bit；网格压缩为 1 个撤回比例 × 3 个影子比例，
    查询上限取 min(config.n_queries_eval, TIER_B_QUERY_CAP)。

    查询上限的唯一权威口径（论文 §Metrics/§Implementation 与此保持一致）：
    生效值 = min(config 的 n_queries_eval, TIER_B_QUERY_CAP)。
    e1_main*.yaml 给 500，e5_cross_dataset/e2/e3 给 300，未填时默认 TIER_B_QUERY_CAP。
    上限取 1000 而非 300：DS2/nq 每查询约 1 条 gold，rev=5% 时命中被撤文档的查询数
    ≈ n_q × 0.05，300 查询只给 ~15 条分母，低于评测协议要求的 ≥30；1000 查询给出 ~50 条，
    生成成本仍可控（只对 gold 命中闭包的查询调用生成器）。
    """
    out = dict(cfg)
    out["tier"] = str(tier)
    out["model"] = str(model)
    if str(tier) == "a":
        out["generator"] = "none" if not smoke else "mock"
    else:
        out["generator"] = C.GEN_ID_SMALL if str(model) == "1.5b" else C.GEN_ID
        rr = [float(x) for x in (out.get("revocation_ratios") or [0.05])]
        if not out.get("_rev_explicit"):
            out["revocation_ratios"] = [0.05 if 0.05 in rr else rr[0]]
        out["n_queries_eval"] = int(min(int(out.get("n_queries_eval") or TIER_B_QUERY_CAP),
                                        TIER_B_QUERY_CAP))
    return out


def apply_smoke_overrides(cfg: Mapping[str, Any]) -> Dict[str, Any]:
    """--smoke：MockGenerator + 合成小规模，保证 60 秒内跑完。"""
    out = dict(cfg)
    out["mode"] = "smoke"
    out["dataset"] = "synthetic"
    out["generator"] = "mock"
    out["encoder"] = "synthetic"
    out["n_queries_eval"] = int(min(int(out.get("n_queries_eval") or 50), 50))
    return out


# ======================================================================================
# 1. 合成数据（现场生成，不联网、不加载 HF 模型）
# ======================================================================================
_TOPIC_WORDS = [
    "quarterly revenue guidance", "clinical trial enrollment", "wildfire containment",
    "chip fabrication yield", "monetary policy stance", "vaccine efficacy cohort",
    "supply chain disruption", "earthquake magnitude report", "emission target review",
    "satellite launch window",
]
_FILLER = ["report", "analysis", "briefing", "summary", "dataset", "study", "memo", "digest"]


def synthesize_dataset(
    n_docs: int = 2000,
    n_clients: int = 3,
    n_queries: int = 50,
    shadow_ratio: float = 0.10,
    dim: int = 64,
    seed: int = int(C.SEED),
    n_topics: int = 5,
    num_perm: int = 64,
    shadow_sim: float = 0.96,
    alpha: float = 0.5,
    forget_ratios: Sequence[float] = (0.01, 0.05, 0.20),
    n_qa: int = 5,
) -> Dict[str, Any]:
    """现场合成"跨孤岛影子注入"语料（确定性；供 --smoke 与单测使用）。

    结构：
      * 每个文档 = 一个唯一事实（entity/attribute/value）+ 主题填充词
      * 孤岛按 Dirichlet(alpha) 倾斜分配 topic，再映射到 client
      * shadow_ratio 比例的文档在不同孤岛注入近重复副本（同事实、不同填充词、向量加噪）
      * 查询分两类：普通查询（gold = 同主题文档）与"被撤知识查询"（gold = 被撤文档）
    """
    rng = np.random.default_rng(int(seed))
    n_topics = max(1, int(n_topics))
    topic_of_doc = rng.integers(0, n_topics, size=n_docs)
    # Dirichlet 倾斜：每个孤岛对主题的偏好
    client_topic_p = rng.dirichlet([max(1e-3, float(alpha))] * n_topics, size=max(1, int(n_clients)))
    topic_center = rng.normal(size=(n_topics, dim)).astype(np.float32)

    entities = ["Acme", "Borealis", "Cygnus", "Delta", "Erebus", "Fulcrum", "Gemini", "Helios",
                "Ionic", "Juno", "Kestrel", "Lyra"]
    attributes = ["headcount", "budget", "latency", "yield", "dosage", "altitude", "capacity",
                  "throughput", "rating", "volume"]
    units = ["units", "ms", "mg", "km", "MW", "GB", "USD", "pct", "tn", "hrs"]

    corpus: List[Dict[str, Any]] = []
    vectors: List[np.ndarray] = []
    texts: Dict[int, str] = {}
    signatures: List[np.ndarray] = []

    def make_text(i: int) -> str:
        ent = entities[i % len(entities)]
        attr = attributes[(i // len(entities)) % len(attributes)]
        val = 1000 + i  # 唯一事实值：避免不同文档共享同一答案导致诱导测试失真
        unit = units[(i // (len(entities) * len(attributes))) % len(units)]
        topic = _TOPIC_WORDS[int(topic_of_doc[i]) % len(_TOPIC_WORDS)]
        filler = " ".join(_FILLER[(i + j) % len(_FILLER)] for j in range(3))
        return "The %s %s of %s is %d %s according to the latest %s on %s." % (
            ent, attr, ent.lower(), val, unit, filler, topic,
        )

    for i in range(n_docs):
        topic = int(topic_of_doc[i])
        vec = topic_center[topic] * 0.8 + rng.normal(scale=0.6, size=dim).astype(np.float32)
        vec = vec / max(1e-9, float(np.linalg.norm(vec)))
        text = make_text(i)
        pid = i
        corpus.append({
            "pid": pid, "doc_id": "d%06d" % i, "client_id": "c0",  # client 稍后按倾斜重排
            "topic": "t%d" % topic, "text": text, "n_tokens": len(text.split()),
            "fingerprint": fingerprint_text(text, num_perm, int(seed)),
        })
        vectors.append(vec.astype(np.float32))
        texts[pid] = text
        signatures.append(np.frombuffer(minhash_signature(text, num_perm, int(seed)), dtype="<u4"))

    # ---- 孤岛分配（Dirichlet 倾斜） ----
    client_of = np.zeros(n_docs, dtype=np.int64)
    for topic in range(n_topics):
        idx = np.flatnonzero(topic_of_doc == topic)
        if idx.size == 0:
            continue
        p = client_topic_p[:, topic]
        p = p / max(1e-12, float(p.sum()))
        client_of[idx] = rng.choice(int(n_clients), size=idx.size, p=p)
    for i in range(n_docs):
        corpus[i]["client_id"] = "c%d" % int(client_of[i])

    # ---- forget sets（doc 级，嵌套 r1 ⊂ r5 ⊂ r20；先算出来，让影子注入优先覆盖） ----
    forget_perm = rng.permutation(n_docs)
    forget: Dict[str, List[str]] = {}
    prev_ids: List[str] = []
    for ratio in sorted(forget_ratios):
        want = max(1, int(round(float(ratio) * n_docs)))
        want = max(want, len(prev_ids))
        ids = sorted("d%06d" % int(i) for i in forget_perm[:want].tolist())
        forget["r%d" % int(round(ratio * 100))] = ids
        prev_ids = ids
    # ---- 跨孤岛影子注入（在全体语料上随机注入：任取 5% 的 forget set 的
    #      "含影子文档比例" 期望即等于 shadow_ratio，与 DS1 实测口径一致(183/609)） ----
    injected: List[Dict[str, Any]] = []
    shadow_map: Dict[str, List[str]] = {}
    shadow_docs: List[str] = []
    n_shadow = int(round(float(shadow_ratio) * n_docs))
    if n_shadow > 0:
        targets = [int(i) for i in rng.permutation(n_docs)[:n_shadow].tolist()]
        for k, orig in enumerate(list(targets)):
            orig_client = int(client_of[orig])
            choices = [c for c in range(int(n_clients)) if c != orig_client]
            if not choices:
                continue
            shadow_client = int(choices[int(rng.integers(0, len(choices)))])
            pid = len(corpus)
            base = np.asarray(vectors[orig], dtype=np.float32)
            # 逐分量噪声：||n||^2 = 1/cos^2 - 1，sigma = ||n||/sqrt(dim)，使 cos(vec, base) 约等于 shadow_sim
            nn = math.sqrt(max(1e-9, 1.0 / max(1e-6, float(shadow_sim)) ** 2 - 1.0))
            noise = rng.normal(scale=nn / math.sqrt(dim), size=dim).astype(np.float32)
            vec = base + noise
            vec = vec / max(1e-9, float(np.linalg.norm(vec)))
            # 影子文本：同事实、不同填充词 → MinHash Jaccard ~0.8
            text = make_text(orig).replace("according to", "per").replace("latest", "newly released")
            doc_id = "s%06d" % k
            corpus.append({
                "pid": pid, "doc_id": doc_id, "client_id": "c%d" % shadow_client,
                "topic": "t%d" % int(topic_of_doc[orig]), "text": text, "n_tokens": len(text.split()),
                "fingerprint": fingerprint_text(text, num_perm, int(seed)),
            })
            vectors.append(vec)
            texts[pid] = text
            signatures.append(np.frombuffer(minhash_signature(text, num_perm, int(seed)), dtype="<u4"))
            injected.append({"orig": "d%06d" % orig, "shadow": doc_id, "client": "c%d" % shadow_client,
                             "sim": float(np.dot(vec, base))})
            shadow_map.setdefault("d%06d" % orig, []).append(doc_id)
            shadow_docs.append(doc_id)

    V = np.asarray(vectors, dtype=np.float32)
    S = np.stack(signatures).astype(np.uint64)
    n_total = len(corpus)

    # ---- 查询 ----
    queries: List[Dict[str, Any]] = []
    qa_items: List[Dict[str, Any]] = []
    for q in range(n_queries):
        anchor = int(rng.integers(0, n_docs))
        client = "c%d" % int(client_of[anchor])
        gold = [anchor] + [int(x) for x in np.flatnonzero(topic_of_doc == topic_of_doc[anchor])[:9].tolist()]
        gold = sorted(set(gold))
        qvec = V[anchor] * 0.9 + rng.normal(scale=0.25, size=dim).astype(np.float32)
        qvec = qvec / max(1e-9, float(np.linalg.norm(qvec)))
        queries.append({
            "qid": q, "client_id": client, "query": "What is reported in document d%06d?" % anchor,
            "answers": [corpus[anchor]["text"].split(" is ")[-1].split(" according")[0]],
            "gold_pids": gold, "qvec": qvec.astype(np.float32),
        })
    # 生成端诱导测试条目：针对被撤知识（seed 稍后确定，这里先给通用条目）
    for k in range(min(n_qa, n_shadow if n_shadow else n_qa)):
        anchor = int(injected[k]["orig"].lstrip("d")) if k < len(injected) else k % n_docs
        qa_items.append({
            "query": "What is reported in document d%06d?" % anchor,
            "answers": [corpus[anchor]["text"].split(" is ")[-1].split(" according")[0]],
            "qvec": V[anchor].astype(np.float32),
        })

    gold_pids = [q["gold_pids"] for q in queries]
    doc_pids: Dict[str, List[int]] = {}
    for row in corpus:
        doc_pids.setdefault(str(row["doc_id"]), []).append(int(row["pid"]))
    chunk_pairs = {}
    for row in injected:
        o, s = str(row["orig"]), str(row["shadow"])
        if doc_pids.get(o) and doc_pids.get(s):
            chunk_pairs[s] = (int(doc_pids[o][0]), int(doc_pids[s][0]))
    return {
        "name": "synthetic",
        "pid_row": {int(r["pid"]): i for i, r in enumerate(corpus)},
        "doc_pids": doc_pids,
        "shadow_chunk_pairs": chunk_pairs,
        "corpus": corpus,
        "vectors": V,
        "signatures": S,
        "texts": texts,
        "queries": queries,
        "qvecs": np.stack([q["qvec"] for q in queries]).astype(np.float32) if queries else np.zeros((0, dim), np.float32),
        "gold_pids": gold_pids,
        "qa_items": qa_items,
        "forget_sets": forget,
        "query_families": ["synthetic"] * len(queries),
        "gold_source": ["synthetic"] * len(queries),
        "primary_metric": "recall10",
        "shadow_map": shadow_map,
        "shadow_docs": sorted(shadow_docs),
        "shadow_pairs": {"injected": injected, "shadow_pid": {}, "meta": {"synthetic": True}},
        "client_of": client_of,
        "n_shadow": int(n_shadow),
        "meta": {"dim": int(dim), "num_perm": int(num_perm), "n_docs": n_total, "n_clients": int(n_clients)},
    }


# ======================================================================================
# 2. 真实数据加载（data/processed/<ds>/）
# ======================================================================================
REQUIRED_FILES = ["corpus.jsonl", "queries.jsonl", "forget_sets.json", "shadow_pairs.json"]
EMB_CANDIDATES = [C.EMB_FILENAME, "embeddings.npy"]
QRY_EMB_CANDIDATES = [
    "embeddings_queries_bge-small-en-v1.5.npy", "query_embeddings.npy",
    "embeddings_q_bge-small-en-v1.5.npy", "queries_bge-small-en-v1.5.npy",
]


def _read_jsonl(path: Path) -> List[dict]:
    out = []
    with Path(path).open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def data_status(ds_key: str) -> Dict[str, Any]:
    """检查某数据集的落盘完备度（供 summary.json / 报告使用）。"""
    name = C.DATASETS.get(ds_key, ds_key)
    d = Path(C.PROCESSED) / name
    status: Dict[str, Any] = {"ds_key": ds_key, "name": name, "dir": str(d), "exists": d.exists(), "files": {}}
    if not d.exists():
        status["missing"] = ["<dataset dir>"]
        return status
    missing = []
    for f in REQUIRED_FILES:
        ok = (d / f).exists()
        status["files"][f] = ok
        if not ok:
            missing.append(f)
    emb = next((f for f in EMB_CANDIDATES if (d / f).exists()), None)
    status["files"]["embeddings"] = bool(emb)
    status["embeddings_file"] = emb
    if not emb:
        missing.append(EMB_CANDIDATES[0])
    sig = (d / "minhash_sig.npy").exists()
    status["files"]["minhash_sig.npy"] = sig
    if not sig:
        missing.append("minhash_sig.npy")
    qemb = next((f for f in QRY_EMB_CANDIDATES if (d / f).exists()), None)
    qemb_path = str(d / qemb) if qemb else None
    if qemb is None:
        # 次选：SA5 用 bge-small 落盘到 artifacts/query_embeddings/（不改动 data/）
        qdir = Path(C.ARTIFACTS) / "query_embeddings"
        for f in QRY_EMB_CANDIDATES:
            cand = qdir / ("%s_%s" % (name, f))
            if cand.exists():
                qemb, qemb_path = f, str(cand)
                status["files"]["query_embeddings_source"] = "artifacts/query_embeddings"
                break
    status["files"]["query_embeddings"] = bool(qemb)
    status["query_embeddings_file"] = qemb
    status["query_embeddings_path"] = qemb_path
    if not qemb:
        missing.append(QRY_EMB_CANDIDATES[0] + "（queries 查询向量；data/processed 与 artifacts/query_embeddings 均无）")
    status["missing"] = missing
    return status


def load_real_dataset(ds_key: str, limit_queries: Optional[int] = None) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
    """读取真实数据集；不满足最低要求时返回 (None, status)。"""
    status = data_status(ds_key)
    name = status["name"]
    d = Path(status["dir"])
    if not status["exists"]:
        return None, status
    fatal = [m for m in status["missing"] if m in REQUIRED_FILES or m == EMB_CANDIDATES[0]]
    if fatal:
        return None, status

    corpus = _read_jsonl(d / "corpus.jsonl")
    queries = _read_jsonl(d / "queries.jsonl")
    forget_sets = json.loads((d / "forget_sets.json").read_text(encoding="utf-8"))
    shadow_pairs = json.loads((d / "shadow_pairs.json").read_text(encoding="utf-8"))
    emb_file = status["embeddings_file"]
    vectors = np.load(str(d / emb_file))
    pid_order = None
    if (d / C.EMB_PIDS_FILENAME).exists():
        pid_order = json.loads((d / C.EMB_PIDS_FILENAME).read_text(encoding="utf-8"))
    if pid_order is not None and len(pid_order) == vectors.shape[0] and pid_order != list(range(len(corpus))):
        order = np.argsort(np.asarray(pid_order, dtype=np.int64))
        vectors = vectors[order]

    signatures = None
    try:
        signatures = load_signature_matrix(d)
    except Exception as exc:
        status["signature_error"] = "%s: %s" % (type(exc).__name__, exc)

    qvecs = None
    if status.get("query_embeddings_path"):
        qvecs = np.load(str(status["query_embeddings_path"]))
        status["query_vector_source"] = "file:%s" % status["query_embeddings_path"]
    elif status.get("query_embeddings_file"):
        qvecs = np.load(str(d / status["query_embeddings_file"]))
        status["query_vector_source"] = "file:%s" % status["query_embeddings_file"]
    if limit_queries:
        queries = queries[: int(limit_queries)]
        if qvecs is not None:
            qvecs = qvecs[: int(limit_queries)]

    # --- 查询来源族 + 官方 strict gold（DS3：50 条官方 vs 450 条合成必须分开报） ---
    families: List[str] = []
    gold_effective: List[List[int]] = []
    gold_pooled: List[List[int]] = []
    gold_source: List[str] = []
    qmeta = {}
    qmeta_path = d / "qid_meta.json"
    if qmeta_path.exists():
        try:
            qmeta = json.loads(qmeta_path.read_text(encoding="utf-8"))
        except Exception as exc:
            status["qid_meta_error"] = "%s: %s" % (type(exc).__name__, exc)
    for i, q in enumerate(queries):
        m = qmeta.get(str(q.get("qid", i))) or {}
        fam = str(m.get("query_source") or ("official" if m.get("has_qrels") else "unknown"))
        families.append(fam)
        pooled = [int(p) for p in (q.get("gold_pids") or [])]
        strict = m.get("gold_pids_strict")
        if strict:
            gold_effective.append([int(p) for p in strict])
            gold_source.append("strict")
        else:
            gold_effective.append(list(pooled))
            gold_source.append("pooled")
        gold_pooled.append(list(pooled))
    try:
        mean_gold = float(np.mean([len(g) for g in gold_effective])) if gold_effective else 0.0
    except Exception:
        mean_gold = 0.0
    primary_metric = "ndcg10" if mean_gold > 20 else "recall10"
    status["query_families"] = {k: int(v) for k, v in
                                zip(*np.unique(np.asarray(families, dtype=object), return_counts=True))} if families else {}
    status["mean_gold_size"] = mean_gold
    status["primary_metric"] = primary_metric

    texts = {int(r["pid"]): r.get("text", "") for r in corpus}
    pid_to_row_all = {int(r.get("pid", i)): i for i, r in enumerate(corpus)}
    doc_pids: Dict[str, List[int]] = {}
    for i, r in enumerate(corpus):
        doc_pids.setdefault(str(r.get("doc_id")), []).append(int(r.get("pid", i)))
    # 影子-原件 chunk 配对（父代理的 self-query 探针口径）：shadow_pairs 提供 orig_pid/shadow_pid
    chunk_pairs: Dict[str, tuple] = {}
    _sp = shadow_pairs or {}
    _op, _shp = (_sp.get("orig_pid") or {}), (_sp.get("shadow_pid") or {})
    for k, v in _op.items():
        if k in _shp:
            try:
                chunk_pairs[str(k)] = (int(v), int(_shp[k]))
            except Exception:
                continue
    # --- 影子映射：优先 pid_meta.json（is_shadow/orig_doc_id），退回 shadow_pairs.injected ---
    shadow_map: Dict[str, List[str]] = {}
    pid_meta_path = d / "pid_meta.json"
    if pid_meta_path.exists():
        try:
            pid_meta = json.loads(pid_meta_path.read_text(encoding="utf-8"))
            for row in corpus:
                pm = pid_meta.get(str(row.get("pid")))
                if pm and pm.get("is_shadow") and pm.get("orig_doc_id"):
                    shadow_map.setdefault(str(pm["orig_doc_id"]), [])
                    did = str(row.get("doc_id"))
                    if did not in shadow_map[str(pm["orig_doc_id"])]:
                        shadow_map[str(pm["orig_doc_id"])].append(did)
            status["shadow_map_source"] = "pid_meta.json"
        except Exception as exc:
            status["shadow_map_error"] = "%s: %s" % (type(exc).__name__, exc)
    if not shadow_map:
        for row in (shadow_pairs.get("injected") or []):
            shadow_map.setdefault(str(row.get("orig")), []).append(str(row.get("shadow")))
        status["shadow_map_source"] = "shadow_pairs.json"
    status["n_shadowed_docs"] = len(shadow_map)

    # --- 查询向量：优先落盘文件；否则用 gold 证据质心代理（须在报告中标注） ---
    if qvecs is None:
        pid_to_row = {}
        if pid_order is not None and len(pid_order) == len(corpus):
            pid_to_row = {int(p): i for i, p in enumerate(pid_order)}
        else:
            pid_to_row = {int(r.get("pid", i)): i for i, r in enumerate(corpus)}
        dim = int(np.asarray(vectors).shape[1])
        built = np.zeros((len(queries), dim), dtype=np.float32)
        keep = []
        for i, q in enumerate(queries):
            rows = [pid_to_row[int(p)] for p in (q.get("gold_pids") or []) if int(p) in pid_to_row]
            if not rows:
                continue
            v = np.asarray(vectors)[rows].mean(axis=0)
            n = float(np.linalg.norm(v))
            built[i] = (v / n) if n > 1e-9 else v
            keep.append(i)
        if not keep:
            status["missing"] = list(status.get("missing", [])) + ["查询向量（且 gold_pids 为空，无法用质心代理）"]
            return None, status
        queries = [queries[i] for i in keep]
        qvecs = built[keep]
        status["query_vector_source"] = "gold_centroid_proxy（无查询编码器时的代理；绝对 recall 不可与论文口径直接比较）"
    return {
        "name": name,
        "corpus": corpus,
        "vectors": np.asarray(vectors, dtype=np.float32),
        "signatures": signatures,
        "texts": texts,
        "queries": queries,
        "qvecs": None if qvecs is None else np.asarray(qvecs, dtype=np.float32),
        "gold_pids": gold_effective,
        "gold_pids_pooled": gold_pooled,
        "gold_source": gold_source,
        "query_families": families,
        "primary_metric": primary_metric,
        "qa_items": [
            {"query": q.get("query", ""), "answers": list(q.get("answers") or []),
             "qvec": None if qvecs is None else np.asarray(qvecs[i], dtype=np.float32)}
            for i, q in enumerate(queries)
        ],
        "forget_sets": forget_sets,
        "shadow_map": shadow_map,
        "shadow_docs": sorted({s for vals in shadow_map.values() for s in vals}),
        "shadow_pairs": shadow_pairs,
        "pid_row": pid_to_row_all,
        "doc_pids": doc_pids,
        "shadow_chunk_pairs": chunk_pairs,
        "status": status,
        "meta": {"dim": int(np.asarray(vectors).shape[1]), "n_docs": len(corpus)},
    }, status


# ======================================================================================
# 3. 索引构建与单次方法评测
# ======================================================================================
def build_index(dataset: Mapping[str, Any], backend: str = "faiss_ivf") -> ProvenanceIndex:
    """由数据集构造 ProvenanceIndex，并挂上文本钩子供 verify/elicitation 取原文。"""
    corpus = list(dataset["corpus"])
    V = np.asarray(dataset["vectors"], dtype=np.float32)
    metas = [
        VecMeta(
            pid=int(r.get("pid", i)),
            doc_id=str(r.get("doc_id", "d%06d" % i)),
            client_id=str(r.get("client_id", "c0")),
            topic=str(r.get("topic", "")),
            fingerprint=str(r.get("fingerprint", "")),
        )
        for i, r in enumerate(corpus)
    ]
    index = ProvenanceIndex(int(V.shape[1]), backend=backend)
    index.add(V, metas)
    texts = dataset.get("texts") or {}
    if texts:
        index.text_for_id = lambda i, _t=texts: _t.get(int(i), "")
    return index


def build_oracle_index(dataset: Mapping[str, Any], doc_ids: Sequence[str],
                       backend: str = "faiss_ivf") -> ProvenanceIndex:
    """I∅（式 7）：删去被撤文档及其注入影子后全量重建。

    这是测量装置（式 18 的 oracle 侧），不计入任何方法的代价轴。
    影子取 shadow_map 中被撤文档的注入副本，即 Definition 2 在注入协议下的真值实例。
    """
    corpus = list(dataset["corpus"])
    V = np.asarray(dataset["vectors"], dtype=np.float32)
    shadow_map = dataset.get("shadow_map") or {}
    remove_docs = {str(d) for d in (doc_ids or []) if d}
    for d in list(remove_docs):
        remove_docs.update(str(s) for s in (shadow_map.get(d) or []))
    keep = [i for i, r in enumerate(corpus) if str(r.get("doc_id", "")) not in remove_docs]
    metas = [
        VecMeta(
            pid=int(corpus[i].get("pid", i)),
            doc_id=str(corpus[i].get("doc_id", "d%06d" % i)),
            client_id=str(corpus[i].get("client_id", "c0")),
            topic=str(corpus[i].get("topic", "")),
            fingerprint=str(corpus[i].get("fingerprint", "")),
        )
        for i in keep
    ]
    index = ProvenanceIndex(int(V.shape[1]), backend=backend)
    if keep:
        index.add(V[keep], metas)
    texts = dataset.get("texts") or {}
    if texts:
        index.text_for_id = lambda i, _t=texts: _t.get(int(i), "")
    return index


def forgot_doc_ids(dataset: Mapping[str, Any], ratio_key: str, ratio: float) -> List[str]:
    """按配置比例取 forget set（真实数据用 forget_sets.json 的 r1/r5/r20 键）。"""
    fs = dataset.get("forget_sets") or {}
    key = ratio_key if ratio_key in fs else None
    if key is None:
        for cand in ("r%d" % int(round(float(ratio) * 100)), "r5", "r1"):
            if cand in fs:
                key = cand
                break
    if key is None and fs:
        key = sorted(fs)[0]
    ids = list(fs.get(key, [])) if key else []
    return [str(x) for x in ids]


def query_residual_stats(index: Any, dataset: Mapping[str, Any], doc_ids: Sequence[str],
                         k: int = 10, max_probes: int = 64, seed: int = int(C.SEED)) -> Dict[str, Any]:
    """E6 三条曲线（父代理口径，分母固定、方法间可比）：

    * self_probe_residual  : 机制上界 —— 以"影子最相似的原始 chunk"嵌入作查询，
                             统计**含影子的被撤文档**中被其 self-probe 命中的比例（需仍有存活影子）。
    * real_query_residual  : 头条指标 —— MultiHop-RAG gold-evidence 查询里，
                             top-k 命中任一存活影子代理的查询比例。
    * unrel_query_residual : 对照 —— gold 与被撤文档无交集的无关查询的同一读数（应 ≈ 0）。
    """
    out: Dict[str, Any] = {
        "n_shadowed_revoked": 0, "n_surviving_surrogate_docs": 0, "n_surviving_surrogates": 0,
        "n_self_probes": 0, "self_probe_residual": NAN, "self_probe_hit_docs": 0,
        "n_real_queries": 0, "real_query_residual": NAN, "real_query_residual_k50": NAN,
        "retrieved_surrogate_fraction_realq": NAN,
        "n_revoked_evidence_queries": 0, "revoked_evidence_residual": NAN,
        "n_unrel_queries": 0, "unrel_query_residual": NAN,
        "retrieved_surrogate_fraction": NAN,
    }
    if index is None:
        return out
    smap = dataset.get("shadow_map") or {}
    V = dataset.get("vectors")
    pid_row = dataset.get("pid_row") or {}
    doc_pids = dataset.get("doc_pids") or {}
    pairs = dataset.get("shadow_chunk_pairs") or {}
    doc_ids = [str(d) for d in (doc_ids or [])]

    sur_alive: Dict[str, List[int]] = {}
    for d in doc_ids:
        ids: List[int] = []
        for s in (smap.get(d) or []):
            try:
                ids.extend(int(i) for i in index.ids_for_doc(str(s)))
            except Exception:
                continue
        sur_alive[d] = sorted(set(ids))
    shadowed = [d for d in doc_ids if smap.get(d)]
    out["n_shadowed_revoked"] = len(shadowed)
    out["n_surviving_surrogate_docs"] = int(sum(1 for d in shadowed if sur_alive[d]))
    all_sur = sorted({i for v in sur_alive.values() for i in v})
    out["n_surviving_surrogates"] = len(all_sur)

    rng = np.random.default_rng(int(seed))
    qvecs = dataset.get("qvecs")
    queries = list(dataset.get("queries") or [])
    revoked_pids = {p for d in doc_ids for p in (doc_pids.get(d) or [])}

    def _search_hits(vecs: np.ndarray) -> set:
        if vecs is None or len(vecs) == 0:
            return set()
        try:
            _, got = index.search(np.asarray(vecs, dtype=np.float32), int(k))
        except Exception:
            return set()
        got = np.asarray(got)
        return {int(v) for row in got.tolist() for v in row if int(v) >= 0}

    # ---- (1) self-probe：每个含影子的被撤文档取一个探针（影子最相似的原始 chunk） ----
    probes: List[np.ndarray] = []
    probe_docs: List[str] = []
    sel = shadowed if len(shadowed) <= max_probes else [shadowed[int(i)] for i in
                                                        np.sort(rng.choice(len(shadowed), max_probes, replace=False)).tolist()]
    for d in sel:
        pid = None
        for s in (smap.get(d) or []):
            pr = pairs.get(str(s))
            if pr:
                pid = int(pr[0])
                break
        if pid is None:
            pl = doc_pids.get(d) or []
            pid = int(pl[0]) if pl else None
        if pid is None or V is None:
            continue
        row = pid_row.get(int(pid))
        if row is None:
            continue
        probes.append(np.asarray(V)[int(row)].astype(np.float32))
        probe_docs.append(d)
    out["n_self_probes"] = len(probes)
    if probes:
        hit_ids = _search_hits(np.stack(probes))
        hit_docs = sum(1 for d, _ in zip(probe_docs, probes) if set(sur_alive.get(d) or []) & hit_ids)
        out["self_probe_hit_docs"] = int(hit_docs)
        out["self_probe_residual"] = float(hit_docs / max(1, len(probes)))
        if all_sur:
            out["retrieved_surrogate_fraction"] = float(len(set(all_sur) & hit_ids) / len(all_sur))

    # ---- (2)(3) 真实查询：gold 命中被撤文档 vs 无关查询 ----
    if qvecs is not None and queries:
        Q = np.asarray(qvecs)
        real_idx, unrel_idx = [], []
        for i, q in enumerate(queries):
            if i >= Q.shape[0]:
                break
            gold = {int(p) for p in (q.get("gold_pids") or [])}
            if gold & revoked_pids:
                if len(real_idx) < max_probes:
                    real_idx.append(i)
            elif len(unrel_idx) < max_probes:
                unrel_idx.append(i)
        if real_idx:
            out["n_real_queries"] = len(real_idx)
            sur_set = set(all_sur)
            if sur_set:
                Qr = Q[real_idx]
                for kk, key in ((int(k), "real_query_residual"),
                                (int(k) * 5, "real_query_residual_k50")):
                    try:
                        _, got = index.search(Qr, int(kk))
                    except Exception:
                        out[key] = NAN
                        continue
                    rows_ = np.asarray(got).tolist()
                    n_hit = sum(1 for row in rows_ if {int(v) for v in row if int(v) >= 0} & sur_set)
                    out[key] = float(n_hit / len(real_idx))
                try:
                    _, got5 = index.search(Qr, int(k) * 5)
                    hit_ids = {int(v) for row in np.asarray(got5).tolist() for v in row if int(v) >= 0}
                    out["retrieved_surrogate_fraction_realq"] = float(len(sur_set & hit_ids) / len(sur_set))
                except Exception:
                    out["retrieved_surrogate_fraction_realq"] = NAN
            else:
                out["real_query_residual"] = 0.0
                out["real_query_residual_k50"] = 0.0
                out["retrieved_surrogate_fraction_realq"] = 0.0
        # (2b) 真实查询受限版：查询向量只用"被撤文档的 gold chunk"质心（无查询编码器时的可用代理）
        restr_vecs, restr_idx = [], []
        if V is not None:
            for i in real_idx:
                gold = [int(p) for p in (queries[i].get("gold_pids") or [])]
                rows = [pid_row[int(p)] for p in gold if int(p) in revoked_pids and int(p) in pid_row]
                if not rows:
                    continue
                v = np.asarray(V)[rows].mean(axis=0)
                n = float(np.linalg.norm(v))
                if n <= 1e-9:
                    continue
                restr_vecs.append((v / n).astype(np.float32))
                restr_idx.append(i)
            if restr_vecs:
                out["n_revoked_evidence_queries"] = len(restr_vecs)
                if all_sur:
                    hit_ids2 = _search_hits(np.stack(restr_vecs))
                    out["revoked_evidence_residual"] = float(
                        sum(1 for i in restr_idx if (set(all_sur) & set(hit_ids2))) / len(restr_idx)
                    )
                else:
                    out["revoked_evidence_residual"] = 0.0

        if unrel_idx:
            out["n_unrel_queries"] = len(unrel_idx)
            sur_set = set(all_sur)
            if sur_set:
                try:
                    _, got = index.search(Q[unrel_idx], int(k))
                    rows_ = np.asarray(got).tolist()
                    n_hit = sum(1 for row in rows_ if {int(v) for v in row if int(v) >= 0} & sur_set)
                    out["unrel_query_residual"] = float(n_hit / len(unrel_idx))
                except Exception:
                    out["unrel_query_residual"] = NAN
            else:
                out["unrel_query_residual"] = 0.0
    return out


def build_qa_items(dataset: Mapping[str, Any], doc_ids: Sequence[str], max_items: int = 64,
                   seed: int = int(C.SEED)) -> List[Dict[str, Any]]:
    """构造"针对被撤知识"的生成端诱导条目（elicit_rate / EM / F1 / faithfulness 的口径保证）。

    每个条目携带：query / answers / qvec / **gold_pids**（供探针按 gold ∩ 闭包精确筛选）/
    **row**（该查询在 query_sample 中的下标，供评测端取到*正确*的检索结果，避免错位配对）。

    规则：只用 gold 命中被撤文档（优先含影子的）的查询；合成数据无命中时，
    用被撤文档自身的 chunk 质心作为代理查询向量（答案取该文档文本中的事实值）。
    """
    smap = dataset.get("shadow_map") or {}
    corpus = dataset.get("corpus") or []
    docset = {str(d) for d in doc_ids}
    # Q_R = gold 与闭包（种子 ∪ 影子）相交的查询；不能只取影子文档，
    # 否则有影子时会漏掉仅命中种子的查询（单孤岛请求下分母会塌到个位数）。
    shadow_docs = {str(s) for d in docset for s in (smap.get(d) or [])}
    prefer = docset | shadow_docs
    pids_by_doc: Dict[str, List[int]] = {}
    for i, row in enumerate(corpus):
        pids_by_doc.setdefault(str(row.get("doc_id")), []).append(int(row.get("pid", i)))
    pid_rows: Dict[int, int] = {}
    for i, row in enumerate(corpus):
        pid_rows[int(row.get("pid", i))] = i
    prefer_pids = {p for d in prefer for p in pids_by_doc.get(d, ())}
    qvecs = dataset.get("qvecs")
    items: List[Dict[str, Any]] = []
    for i, q in enumerate(dataset.get("queries") or []):
        golds = {int(p) for p in (q.get("gold_pids") or [])}
        if not (golds & prefer_pids):
            continue
        qv = None
        if qvecs is not None and i < int(np.asarray(qvecs).shape[0]):
            qv = np.asarray(qvecs[i], dtype=np.float32)
        items.append({
            "query": q.get("query", ""),
            "answers": list(q.get("answers") or []),
            "qvec": qv,
            "gold_pids": sorted(golds),          # 父代理/SA9 要求：供探针按 gold ∩ 闭包筛选
            "row": int(i),                        # 关键：与 query_sample 的正确对齐下标
            "query_source": (dataset.get("query_families") or [None] * (i + 1))[i]
                             if dataset.get("query_families") else None,
        })
        if len(items) >= int(max_items):
            return items
    if items:
        return items

    # --- 兜底：以被撤文档的 chunk 质心作为代理查询（仅合成/无查询命中时使用） ---
    V = dataset.get("vectors")
    texts = dataset.get("texts") or {}
    rng = np.random.default_rng(int(seed))
    pool = sorted(prefer)
    if pool and V is not None:
        order = [pool[int(j)] for j in np.sort(rng.choice(len(pool), size=min(len(pool), int(max_items) * 3), replace=False)).tolist()]
        for d in order:
            pids = [p for p in pids_by_doc.get(d, []) if p in pid_rows]
            if not pids:
                continue
            vec = np.asarray(V)[[pid_rows[p] for p in pids]].mean(axis=0)
            n = float(np.linalg.norm(vec))
            if n <= 1e-9:
                continue
            text = str(texts.get(pids[0], ""))
            answer = text.split(" is ")[-1].split(" according")[0] if " is " in text else text[:40]
            items.append({"query": "What is reported in document %s?" % d, "answers": [answer],
                          "qvec": (vec / n).astype(np.float32),
                          "gold_pids": [int(p) for p in pids], "row": None})
            if len(items) >= int(max_items):
                break
    return items


def requester_client_of(dataset: Mapping[str, Any]) -> str:
    """A2：请求者孤岛 = 文档数最多的孤岛（与 data_prep / resample_forget_sets 一致）。"""
    counts: Dict[str, int] = {}
    for r in (dataset.get("corpus") or []):
        did = str(r.get("doc_id", ""))
        if not did.startswith("d"):
            continue
        cid = str(r.get("client_id", "c0"))
        counts[cid] = counts.get(cid, 0) + 1
    if not counts:
        return "c0"
    return max(sorted(counts), key=lambda c: (counts[c], c))


def controlled_forget_set(dataset: Mapping[str, Any], ratio: float, coverage: Optional[float],
                          seed: int, ratio_key: str) -> Tuple[List[str], Dict[str, Any]]:
    """构造"含影子文档比例 = coverage"的 forget set（E6 专用；真实/合成统一口径）。

    真实数据里影子副本是固定注入的（DS1: 183/609 篇有影子），因此 E6 的自变量应当是
    **被撤文档中带影子的比例**，而不是语料整体的影子比例。coverage 为 None 时退回
    forget_sets.json 的自然切分（主实验用）。

    A2：候选池只含请求者孤岛自己的文档，D_R ⊆ D_{c_R}。
    """
    smap = dataset.get("shadow_map") or {}
    shadow_docs = {str(x) for x in (dataset.get("shadow_docs") or [])}
    requester = requester_client_of(dataset)
    doc_client = {
        str(r.get("doc_id")): str(r.get("client_id", "c0"))
        for r in (dataset.get("corpus") or [])
        if str(r.get("doc_id", "")).startswith("d")
    }
    all_docs = sorted({str(r.get("doc_id")) for r in (dataset.get("corpus") or [])})
    # 影子副本本身不是"原始知识"，不得作为被撤文档（真实/合成都一样）；
    # 且只允许请求者孤岛的文档（A2）。
    pool = [d for d in all_docs if d not in shadow_docs and doc_client.get(d, "c0") == requester]
    if coverage is None or not smap:
        ids = forgot_doc_ids(dataset, ratio_key, ratio)
        return ids, {"coverage_requested": coverage, "coverage_actual": float("nan"),
                     "n_shadowed_pool": len(smap), "source": "forget_sets",
                     "requester_client": requester}
    shadowed = [d for d in pool if smap.get(d)]
    plain = [d for d in pool if not smap.get(d)]
    n_corpus = len({str(r.get("doc_id")) for r in (dataset.get("corpus") or [])}) - len(shadow_docs)
    # 比例按语料级定义（与论文表口径一致），但只从请求者孤岛池中抽样（A2）。
    k = max(3, int(round(float(ratio) * max(n_corpus, len(pool)))))
    k = min(k, len(pool))
    rng = np.random.default_rng(int(seed))
    n_with = min(int(round(float(coverage) * k)), len(shadowed))
    n_plain = min(k - n_with, len(plain))
    ids: List[str] = []
    if n_with:
        ids += [shadowed[int(i)] for i in np.sort(rng.choice(len(shadowed), n_with, replace=False)).tolist()]
    if n_plain:
        ids += [plain[int(i)] for i in np.sort(rng.choice(len(plain), n_plain, replace=False)).tolist()]
    return sorted(set(ids)), {
        "coverage_requested": float(coverage),
        "coverage_actual": float(n_with / max(1, len(ids))),
        "n_shadowed_pool": len(shadowed),
        "n_plain_pool": len(plain),
        "source": "controlled",
        "requester_client": requester,
    }


def subset_dataset(dataset: Mapping[str, Any], n_docs: int, seed: int) -> Dict[str, Any]:
    """按 corpus 规模裁剪数据集（用于 E4 代价扫描）；保持 pid 连续重编号。"""
    n = min(int(n_docs), len(dataset["corpus"]))
    if n >= len(dataset["corpus"]):
        return dict(dataset)
    rng = np.random.default_rng(int(seed))
    keep = np.sort(rng.choice(len(dataset["corpus"]), size=n, replace=False))
    keep_set = set(int(i) for i in keep.tolist())
    remap = {int(old): new for new, old in enumerate(keep.tolist())}
    corpus, vectors = [], []
    for new, old in enumerate(keep.tolist()):
        row = dict(dataset["corpus"][int(old)])
        row["pid"] = new
        corpus.append(row)
        vectors.append(np.asarray(dataset["vectors"][int(old)], dtype=np.float32))
    texts = {}
    for new, old in enumerate(keep.tolist()):
        val = (dataset.get("texts") or {}).get(int(old))
        if val is not None:
            texts[new] = val
    queries, gold, keep_idx = [], [], []
    for qi, q in enumerate(dataset.get("queries") or []):
        g = [remap[int(p)] for p in (q.get("gold_pids") or []) if int(p) in keep_set]
        if not g:
            continue
        qq = dict(q)
        qq["gold_pids"] = g
        queries.append(qq)
        gold.append(g)
        keep_idx.append(qi)
    sigs = dataset.get("signatures")
    return {
        "name": dataset.get("name", "synthetic") + "_n%d" % n,
        "corpus": corpus,
        "vectors": np.asarray(vectors, dtype=np.float32),
        "signatures": None if sigs is None else np.asarray(sigs)[keep],
        "texts": texts,
        "queries": queries,
        "qvecs": (np.asarray(dataset["qvecs"])[keep_idx] if dataset.get("qvecs") is not None
                  and len(dataset["qvecs"]) >= (max(keep_idx) + 1 if keep_idx else 0) else dataset.get("qvecs")),
        "gold_pids": gold,
        "qa_items": [q for q in (dataset.get("qa_items") or []) if True][: len(queries)],
        # forget set 必须按"裁剪后仍存在的 doc_id"重建，否则代价扫描会退化为"无撤回"
        "forget_sets": {
            str(k): ([str(d) for d in (v or []) if str(d) in {str(r.get("doc_id")) for r in corpus}]
                     if isinstance(v, (list, tuple)) else v)
            for k, v in (dataset.get("forget_sets") or {}).items()
        },
        "shadow_map": {k: v for k, v in (dataset.get("shadow_map") or {}).items()},
        "shadow_pairs": dataset.get("shadow_pairs") or {},
        "meta": dict(dataset.get("meta") or {}),
        "query_families": [list(dataset.get("query_families") or [])[i] for i in keep_idx]
                           if dataset.get("query_families") and keep_idx else
                           list(dataset.get("query_families") or []),
        "primary_metric": dataset.get("primary_metric", "recall10"),
    }


@dataclass
class RunPoint:
    """一个实验配置点（实验格）。"""

    exp: str
    dataset: str
    silos: int
    revocation_ratio: float
    shadow_ratio: float
    ratio_key: str
    seed: int
    alpha: float = 0.5
    n_docs: Optional[int] = None
    variant: str = "full"
    knn_k: Optional[int] = None
    sim_threshold: Optional[float] = None
    methods: List[str] = field(default_factory=list)
    stage: str = "main"
    extra: Dict[str, Any] = field(default_factory=dict)

    def key(self) -> str:
        return "%s|%s|s%d|rev%.2f|sh%.2f|%s|knn%s|sim%s|a%.2f|n%s|sd%d" % (
            self.exp, self.dataset, self.silos, self.revocation_ratio, self.shadow_ratio, self.variant,
            self.knn_k, self.sim_threshold, self.alpha, self.n_docs, self.seed,
        )


def build_detector(
    cfg: Mapping[str, Any],
    signatures: Any,
    *,
    knn_k: Optional[int] = None,
    sim_threshold: Optional[float] = None,
    vector_channel: bool = True,
) -> ShadowDetector:
    """构造 ShadowDetector：凡启用影子闭包都把签名矩阵注入（INTERFACES §11 第 2/3 条）。"""
    scfg = dict(cfg.get("shadow") or {})
    return ShadowDetector(
        sim_threshold=float(sim_threshold if sim_threshold is not None else scfg.get("sim_threshold", 0.92)),
        lsh_threshold=float(scfg.get("lsh_threshold", 0.80)),
        knn_k=int(knn_k if knn_k is not None else scfg.get("knn_k", 50)),
        cross_client_only=bool(scfg.get("cross_client_only", True)),
        seed=int(cfg.get("seed", C.SEED)),
        signatures=signatures,
        vector_channel=bool(vector_channel),
    )


def build_repair(cfg: Mapping[str, Any], *, enabled: bool = True, recalibrate: Optional[bool] = None) -> Any:
    """构造 AnchorRepair。

    n_anchors 优先级：CLI/_n_anchors 覆盖 > 配置 > repair 模块默认值。
    注意（SA7/父代理实测）：n_anchors=512 会把 anchor 副本塞进 top-10、after_recall 掉到 0.896，
    而模块新默认 64 与之持平；因此当配置仍是遗留的 512 且未显式覆盖时，退回模块默认。
    """
    if not enabled:
        return None
    rcfg = dict(cfg.get("repair") or {})
    n_anchors = cfg.get("_n_anchors", rcfg.get("n_anchors"))
    if n_anchors is not None and int(n_anchors) == 512 and "_n_anchors" not in cfg:
        n_anchors = None  # 遗留配置 512 → 用模块默认（SA7 已把默认改为 64）
    kwargs = {}
    if n_anchors is not None:
        kwargs["n_anchors"] = int(n_anchors)
    if rcfg.get("max_replicas") is not None:
        kwargs["max_replicas"] = int(rcfg["max_replicas"])
    return AnchorRepair(
        **kwargs,
        recalibrate=bool(rcfg.get("recalibrate", True) if recalibrate is None else recalibrate),
        seed=int(cfg.get("seed", C.SEED)),
        k=int(cfg.get("k", 10)),
    )


def build_generator_for(cfg: Mapping[str, Any], smoke: bool) -> Any:
    """按配置构造生成器；smoke 或未安装 HF 依赖时使用 MockGenerator。"""
    spec = cfg.get("generator")
    if spec == "none":
        return None
    if smoke or spec in (None, "mock"):
        return MockGenerator("MOCK")
    try:
        return build_generator("hf", model_id=str(spec), load_in_4bit=bool(cfg.get("generator_4bit", True)),
                               device="cuda", max_batch=2, strict=False)
    except Exception as exc:
        warnings.warn("生成器构造失败，回退 MockGenerator：%s: %s" % (type(exc).__name__, exc))
        return MockGenerator("MOCK")


def evaluate_method(
    method: str,
    index: ProvenanceIndex,
    dataset: Mapping[str, Any],
    cfg: Mapping[str, Any],
    point: "RunPoint",
    doc_ids: Sequence[str],
    *,
    smoke: bool,
    generator: Any,
    text_store: Optional[Dict[int, str]] = None,
    notes: str = "",
    index_before: Any = None,
) -> Dict[str, Any]:
    """跑一个方法（fedrevoke 或基线）并返回统一结构的结果 dict。"""
    scfg = dict(cfg.get("shadow") or {})
    kw = dict(
        query_sample=dataset.get("qvecs"),
        gold_pids=dataset.get("gold_pids"),
        qa_items=build_qa_items(dataset, doc_ids, max_items=int(cfg.get("max_qa_items", 64)),
                                seed=int(point.seed)),
        generator=generator,
        k=int(cfg.get("k", 10)),
        seed=int(point.seed),
        shadow_surrogates=dataset.get("shadow_map") or {},
        text_store=text_store,
        max_new_tokens=48,
        elicit_k=int(cfg.get("k", 10)),
        # 信息公平性 + 查询族口径（父代理裁决 C/F）
        signatures=dataset.get("signatures"),
        sim_threshold=float(point.sim_threshold if point.sim_threshold is not None else scfg.get("sim_threshold", 0.92)),
        lsh_threshold=float(scfg.get("lsh_threshold", 0.80)),
        knn_k=int(point.knn_k if point.knn_k is not None else scfg.get("knn_k", 50)),
        query_families=dataset.get("query_families"),
        primary_metric=str(cfg.get("primary_metric") or dataset.get("primary_metric") or "recall10"),
        closed_book=bool(cfg.get("closed_book", False)),
        index_before=index_before,
        dataset_name=str(dataset.get("name") or ""),
        n_docs_full=int(dataset.get("meta", {}).get("n_docs") or len(dataset.get("corpus") or [])),
    )
    if method == str(cfg.get("method") or "fedrevoke") or method == "fedrevoke":
        detector = build_detector(
            cfg, dataset.get("signatures"),
            knn_k=point.knn_k, sim_threshold=point.sim_threshold,
            vector_channel=(point.variant != "no_shadow_closure"),
        )
        repair = build_repair(
            cfg,
            enabled=(point.variant not in ("no_repair",)),
            recalibrate=(False if point.variant == "no_calibration" else None),
        )
        pipe = RevocationPipeline(
            index=index, detector=detector if point.variant != "no_shadow_closure" else None,
            repair=repair, verifier=None, cert_dir=CERT_DIR,
            encoder=None, k=int(cfg.get("k", 10)), generator=generator,
            forgotten_qa=kw["qa_items"], seed=int(point.seed),
            shadow_ratio=float(point.shadow_ratio), smoke=bool(smoke), verbose=False,
            closed_book=bool(cfg.get("closed_book", False)),
            client_id=requester_client_of(dataset),
        )
        try:
            res = pipe.revoke(doc_ids, dataset.get("qvecs"))
        except RuntimeError as exc:
            # §11 fail-fast：full 模式下指纹退化直接失败（不静默出结果）
            raise
        ctx = build_eval_context(index, doc_ids, **kw)
        forget = res.report.as_dict()
        forget["surrogate_hit_rate"] = _surrogate_hit_rate(index, dataset, doc_ids, kw["k"])
        forget["n_surrogates"] = len(_surrogate_ids(index, dataset, doc_ids))
        _vmeta = res.certificate.get("verification_meta", {}) or {}
        forget["elicit_measured"] = bool(_vmeta.get("elicit_measured", False))
        # 逐条命中标记从凭证的 verification_meta 透传到行（配对 McNemar 用）
        forget["elicit_flags"] = list(_vmeta.get("elicit_flags") or [])
        forget["elicit_rate_closed_book"] = _vmeta.get("elicit_rate_closed_book", NAN)
        forget["elicit_flags_closed_book"] = list(_vmeta.get("elicit_flags_closed_book") or [])
        forget.update(residual_stats(index, doc_ids, ctx))
        forget.update(query_residual_stats(index, dataset, doc_ids, k=int(cfg.get("k", 10)),
                                           max_probes=int(cfg.get("max_query_probes", 512)),
                                           seed=int(point.seed)))
        utility = evaluate_utility(ctx, index, generator=generator)
        deleted_by_run = sorted(set(int(i) for i in index.deleted_ids()))
        utility.update(evaluate_delta_utility(index_before, index, ctx, deleted_by_run, generator=generator))
        # 式(18)：oracle 侧是"删去种子∪注入影子后重建"的 I∅（测量装置，不计费）
        index_oracle = kw.get("index_oracle") or build_oracle_index(dataset, doc_ids)
        utility["rho_hat_tv"] = rho_hat_tv(index_oracle, index, ctx)
        utility["n_deleted_by_run"] = len(deleted_by_run)
        # 统一口径：rho_hat = max(rho_hat_ret, elicit)；保守上界 = max(hit, elicit)
        _er = float(forget.get("elicit_rate", 0.0) or 0.0)
        _hr = float(forget.get("hit_rate", 0.0) or 0.0)
        _rret = float(utility.get("rho_hat_tv", NAN) or NAN)
        forget["rho_hat_ret"] = _rret
        if _rret == _rret and _er == _er:  # not NaN
            forget["rho_hat"] = float(max(_rret, _er))
        forget["rho_hat_bound"] = float(max(_hr, _er))
        cost = dict(res.cost)
        meta = {
            "n_seed": len(res.seeds),
            "n_closure": len(res.deleted_ids),
            "closure_precision": res.closure_stats.get("precision", NAN),
            "closure_recall": res.closure_stats.get("recall", NAN),
            "n_reconnected": res.repair_stats.get("n_reconnected", 0),
            "n_replica_control": int(res.repair_stats.get("n_anchor_replicas", 0) or 0),
            "n_vectors_delta": int(res.certificate.get("index_stats_after", {}).get("n_vectors", 0)
                                   - res.certificate.get("index_stats_before", {}).get("n_vectors", 0)),
            # M2/M3 耗时占比（父代理要求：在 DS2 规模上量出闭包/擦除是否成为新瓶颈）
            "m2_closure_s": float(res.cost.get("stage_seconds", {}).get("M2_closure", 0.0)),
            "m3_erase_s": float(res.cost.get("stage_seconds", {}).get("M3_erase", 0.0)),
            "m4_repair_s": float(res.cost.get("stage_seconds", {}).get("M4_repair", 0.0)),
            "m2_share": float(res.cost.get("stage_seconds", {}).get("M2_closure", 0.0) / max(1e-9, res.cost.get("reindex_seconds", 1.0))),
            "m3_share": float(res.cost.get("stage_seconds", {}).get("M3_erase", 0.0) / max(1e-9, res.cost.get("reindex_seconds", 1.0))),
            "score_shift": res.repair_stats.get("score_shift", 0.0),
            "certificate_path": res.certificate_path,
            "fingerprint_mode": res.fingerprint_check.get("counts", {}),
            "stage_errors": cost.get("stage_errors", []),
        }
        return {
            "method": "fedrevoke", "forget": forget, "utility": utility, "cost": cost, "meta": meta,
            "n_deleted": len(res.deleted_ids), "deleted_doc_ids": res.certificate.get("deleted_doc_ids", []),
        }

    base = build_baseline(method)
    kw = dict(kw)
    kw.setdefault("index_oracle", build_oracle_index(dataset, doc_ids))
    out = base.run(index, doc_ids, **kw)
    try:
        out["forget"].update(query_residual_stats(out.get("index", index), dataset, doc_ids,
                                                   k=int(cfg.get("k", 10)),
                                                   max_probes=int(cfg.get("max_query_probes", 512)),
                                                   seed=int(point.seed)))
    except Exception as exc:
        out["forget"]["query_residual_error"] = "%s: %s" % (type(exc).__name__, exc)
    meta = dict(out.get("meta") or {})
    meta["fingerprint_mode"] = _fingerprint_mode(dataset)
    out["meta"] = meta
    out["n_deleted"] = int(out["cost"].get("n_deleted", 0))
    out["deleted_doc_ids"] = []
    return out


def _fingerprint_mode(dataset: Mapping[str, Any]) -> Dict[str, int]:
    return {"provided": int(len(dataset.get("corpus") or []))} if dataset.get("signatures") is not None else {"digest": int(len(dataset.get("corpus") or []))}


def _surrogate_ids(index: ProvenanceIndex, dataset: Mapping[str, Any], doc_ids: Sequence[str]) -> List[int]:
    smap = dataset.get("shadow_map") or {}
    out: List[int] = []
    for d in doc_ids:
        for s in smap.get(str(d), []) or []:
            try:
                out.extend(int(i) for i in index.ids_for_doc(str(s)))
            except Exception:
                continue
    return sorted(set(out))


def _surrogate_hit_rate(index: ProvenanceIndex, dataset: Mapping[str, Any], doc_ids: Sequence[str], k: int) -> float:
    ids = _surrogate_ids(index, dataset, doc_ids)
    q = dataset.get("qvecs")
    if not ids or q is None or len(q) == 0:
        return NAN
    try:
        _, got = index.search(np.asarray(q, dtype=np.float32), int(k))
    except Exception:
        return NAN
    want = set(ids)
    hits = sum(1 for row in np.asarray(got).tolist() if any(int(v) in want for v in row))
    return float(hits / max(1, len(np.asarray(got))))



# ======================================================================================
# 4. 无依赖绘图后端（PDF 矢量 + PNG 300dpi；本机无 matplotlib 且禁止 pip install）
# ======================================================================================
_FONT_5X7: Dict[str, List[str]] = {
    "0": [".###.", "#...#", "#..##", "#.#.#", "##..#", "#...#", ".###."],
    "1": ["..#..", ".##..", "..#..", "..#..", "..#..", "..#..", ".###."],
    "2": [".###.", "#...#", "....#", "...#.", "..#..", ".#...", "#####"],
    "3": ["#####", "...#.", "..#..", "...#.", "....#", "#...#", ".###."],
    "4": ["...#.", "..##.", ".#.#.", "#..#.", "#####", "...#.", "...#."],
    "5": ["#####", "#....", "####.", "....#", "....#", "#...#", ".###."],
    "6": ["..##.", ".#...", "#....", "####.", "#...#", "#...#", ".###."],
    "7": ["#####", "....#", "...#.", "..#..", ".#...", ".#...", ".#..."],
    "8": [".###.", "#...#", "#...#", ".###.", "#...#", "#...#", ".###."],
    "9": [".###.", "#...#", "#...#", ".####", "....#", "...#.", ".##.."],
    "A": [".###.", "#...#", "#...#", "#####", "#...#", "#...#", "#...#"],
    "B": ["####.", "#...#", "#...#", "####.", "#...#", "#...#", "####."],
    "C": [".###.", "#...#", "#....", "#....", "#....", "#...#", ".###."],
    "D": ["###..", "#..#.", "#...#", "#...#", "#...#", "#..#.", "###.."],
    "E": ["#####", "#....", "#....", "####.", "#....", "#....", "#####"],
    "F": ["#####", "#....", "#....", "####.", "#....", "#....", "#...."],
    "G": [".###.", "#...#", "#....", "#.###", "#...#", "#...#", ".###."],
    "H": ["#...#", "#...#", "#...#", "#####", "#...#", "#...#", "#...#"],
    "I": [".###.", "..#..", "..#..", "..#..", "..#..", "..#..", ".###."],
    "J": ["..###", "...#.", "...#.", "...#.", "...#.", "#..#.", ".##.."],
    "K": ["#...#", "#..#.", "#.#..", "##...", "#.#..", "#..#.", "#...#"],
    "L": ["#....", "#....", "#....", "#....", "#....", "#....", "#####"],
    "M": ["#...#", "##.##", "#.#.#", "#.#.#", "#...#", "#...#", "#...#"],
    "N": ["#...#", "##..#", "#.#.#", "#..##", "#...#", "#...#", "#...#"],
    "O": [".###.", "#...#", "#...#", "#...#", "#...#", "#...#", ".###."],
    "P": ["####.", "#...#", "#...#", "####.", "#....", "#....", "#...."],
    "Q": [".###.", "#...#", "#...#", "#...#", "#.#.#", "#..#.", ".##.#"],
    "R": ["####.", "#...#", "#...#", "####.", "#.#..", "#..#.", "#...#"],
    "S": [".####", "#....", "#....", ".###.", "....#", "....#", "####."],
    "T": ["#####", "..#..", "..#..", "..#..", "..#..", "..#..", "..#.."],
    "U": ["#...#", "#...#", "#...#", "#...#", "#...#", "#...#", ".###."],
    "V": ["#...#", "#...#", "#...#", "#...#", "#...#", ".#.#.", "..#.."],
    "W": ["#...#", "#...#", "#...#", "#.#.#", "#.#.#", "##.##", "#...#"],
    "X": ["#...#", "#...#", ".#.#.", "..#..", ".#.#.", "#...#", "#...#"],
    "Y": ["#...#", "#...#", ".#.#.", "..#..", "..#..", "..#..", "..#.."],
    "Z": ["#####", "....#", "...#.", "..#..", ".#...", "#....", "#####"],
    ".": [".....", ".....", ".....", ".....", ".....", ".##..", ".##.."],
    ",": [".....", ".....", ".....", ".....", ".##..", ".##..", "#...."],
    "-": [".....", ".....", ".....", "#####", ".....", ".....", "....."],
    "_": [".....", ".....", ".....", ".....", ".....", ".....", "#####"],
    ":": [".....", ".##..", ".##..", ".....", ".##..", ".##..", "....."],
    "(": ["..##.", ".#...", "#....", "#....", "#....", ".#...", "..##."],
    ")": [".##..", "...#.", "....#", "....#", "....#", "...#.", ".##.."],
    "%": ["##..#", "##.#.", "..#..", ".#...", "#..##", "#..##", "....."],
    "+": [".....", "..#..", "..#..", "#####", "..#..", "..#..", "....."],
    "/": ["....#", "....#", "...#.", "..#..", ".#...", "#....", "#...."],
    "=": [".....", ".....", "#####", ".....", "#####", ".....", "....."],
    ">": ["#....", ".#...", "..#..", "...#.", "..#..", ".#...", "#...."],
    "<": ["....#", "...#.", "..#..", ".#...", "..#..", "...#.", "....#"],
    "*": [".....", "#.#.#", ".###.", "#####", ".###.", "#.#.#", "....."],
    "?": [".###.", "#...#", "....#", "...#.", "..#..", ".....", "..#.."],
    " ": [".....", ".....", ".....", ".....", ".....", ".....", "....."],
}


class Canvas:
    """极简矢量/栅格画布：坐标单位为设备像素（左上原点，y 向下）。"""

    def __init__(self, width_px: int, height_px: int, dpi: int = 300) -> None:
        self.w = int(width_px)
        self.h = int(height_px)
        self.dpi = int(dpi)
        self.ops: List[Tuple] = []

    # ---- 图元 ------------------------------------------------------------- #
    def line(self, x0, y0, x1, y1, width=1.0, color=(0, 0, 0)) -> None:
        self.ops.append(("line", (float(x0), float(y0), float(x1), float(y1)), float(width), tuple(color)))

    def polyline(self, pts, width=1.0, color=(0, 0, 0)) -> None:
        pts = [(float(x), float(y)) for x, y in pts]
        for a, b in zip(pts[:-1], pts[1:]):
            self.line(a[0], a[1], b[0], b[1], width, color)

    def marker(self, x, y, r=3.0, color=(0, 0, 0)) -> None:
        self.ops.append(("disc", (float(x), float(y)), float(r), tuple(color)))

    def rect(self, x, y, w, h, fill=None, stroke=None, width=1.0) -> None:
        self.ops.append(("rect", (float(x), float(y), float(w), float(h)), float(width), fill, stroke))

    def text(self, x, y, s, size_pt=8.0, color=(0, 0, 0), anchor="left", rotation: float = 0.0) -> None:
        self.ops.append(("text", (float(x), float(y)), str(s), float(size_pt), tuple(color),
                         str(anchor), float(rotation)))

    # ---- PNG -------------------------------------------------------------- #
    def _blank(self) -> np.ndarray:
        return np.full((self.h, self.w, 3), 255, dtype=np.uint8)

    def _stamp(self, img: np.ndarray, cx: float, cy: float, radius: float, color) -> None:
        r = max(0.5, float(radius))
        x0, x1 = int(max(0, math.floor(cx - r))), int(min(self.w - 1, math.ceil(cx + r)))
        y0, y1 = int(max(0, math.floor(cy - r))), int(min(self.h - 1, math.ceil(cy + r)))
        if x1 < x0 or y1 < y0:
            return
        ys, xs = np.mgrid[y0 : y1 + 1, x0 : x1 + 1]
        mask = (xs - cx) ** 2 + (ys - cy) ** 2 <= r * r
        img[y0 : y1 + 1, x0 : x1 + 1][mask] = np.asarray(color, dtype=np.uint8)

    # 位图字体宽高比修正：5x7 字体的字宽约为字号高的 0.85 倍，比 Helvetica(约 0.5) 宽，
    # 因此栅格化时按 _GLYPH_K 缩小有效字号，避免长标签横向溢出画布。
    _GLYPH_K = 0.62

    def _text_scale(self, size_pt: float) -> int:
        size_px = max(5.0, float(size_pt) * self.dpi / 72.0) * self._GLYPH_K
        return max(1, int(round(size_px / 7.0)))

    def _text_len(self, s: str, size_pt: float) -> float:
        return 6.0 * self._text_scale(size_pt) * len(str(s))

    def _draw_text(self, img: np.ndarray, x: float, y: float, s: str, size_pt: float, color, anchor: str,
                   rotation: float = 0.0) -> None:
        scale = self._text_scale(size_pt)
        advance = 6 * scale
        text = str(s).upper()
        length = advance * len(text)
        shift = -length / 2.0 if anchor == "center" else (-length if anchor == "right" else 0.0)
        theta = math.radians(float(rotation))
        ct, st = math.cos(theta), math.sin(theta)
        col = np.asarray(color, dtype=np.uint8)
        for ci, ch in enumerate(text):
            glyph = _FONT_5X7.get(ch) or _FONT_5X7.get("?")
            u0 = shift + ci * advance
            for row in range(7):
                for c in range(5):
                    if glyph[row][c] != "#":
                        continue
                    u = u0 + c * scale
                    v = -7 * scale / 2.0 + row * scale
                    px = x + u * ct - v * st
                    py = y + u * st + v * ct
                    px0, py0 = int(round(px)), int(round(py))
                    px1, py1 = min(self.w, px0 + scale), min(self.h, py0 + scale)
                    if 0 <= px0 < self.w and 0 <= py0 < self.h:
                        img[max(0, py0) : py1, max(0, px0) : px1] = col

    def render_png(self) -> np.ndarray:
        img = self._blank()
        for op in self.ops:
            kind = op[0]
            if kind == "line":
                (x0, y0, x1, y1), width, color = op[1], op[2], op[3]
                length = max(1.0, math.hypot(x1 - x0, y1 - y0))
                steps = int(length * 2) + 2
                for t in np.linspace(0.0, 1.0, steps):
                    self._stamp(img, x0 + (x1 - x0) * t, y0 + (y1 - y0) * t, max(0.5, width / 2.0), color)
            elif kind == "disc":
                (x, y), r, color = op[1], op[2], op[3]
                self._stamp(img, x, y, r, color)
            elif kind == "rect":
                (x, y, w, h), width, fill, stroke = op[1], op[2], op[3], op[4]
                if fill is not None:
                    x0, y0 = int(max(0, round(x))), int(max(0, round(y)))
                    x1, y1 = int(min(self.w, round(x + w))), int(min(self.h, round(y + h)))
                    if x1 > x0 and y1 > y0:
                        img[y0:y1, x0:x1] = np.asarray(fill, dtype=np.uint8)
                if stroke is not None:
                    self.line(x, y, x + w, y, width, stroke)
                    self.line(x + w, y, x + w, y + h, width, stroke)
                    self.line(x + w, y + h, x, y + h, width, stroke)
                    self.line(x, y + h, x, y, width, stroke)
            elif kind == "text":
                (x, y), s, size_pt, color, anchor = op[1], op[2], op[3], op[4], op[5]
                rot = op[6] if len(op) > 6 else 0.0
                self._draw_text(img, x, y, s, size_pt, color, anchor, rot)
        return img

    def to_png(self, path: Any) -> str:
        img = self.render_png()
        write_png(path, img)
        return str(path)

    # ---- PDF -------------------------------------------------------------- #
    def _pdf_ops(self) -> str:
        k = 72.0 / float(self.dpi)  # px -> pt
        def X(v):
            return v * k
        def Y(v):
            return (self.h - v) * k
        out: List[str] = []
        for op in self.ops:
            kind = op[0]
            if kind == "line":
                (x0, y0, x1, y1), width, color = op[1], op[2], op[3]
                r, g, b = [c / 255.0 for c in color]
                out.append("%.3f %.3f %.3f RG %.3f w %.3f %.3f m %.3f %.3f l S"
                           % (r, g, b, max(0.2, X(width)), X(x0), Y(y0), X(x1), Y(y1)))
            elif kind == "disc":
                (x, y), rad, color = op[1], op[2], op[3]
                r, g, b = [c / 255.0 for c in color]
                cx, cy, rr = X(x), Y(y), X(rad)
                c = 0.5523 * rr
                out.append("%.3f %.3f %.3f rg %.3f %.3f m %.3f %.3f %.3f %.3f %.3f %.3f c "
                           "%.3f %.3f %.3f %.3f %.3f %.3f c %.3f %.3f %.3f %.3f %.3f %.3f c "
                           "%.3f %.3f %.3f %.3f %.3f %.3f c f"
                           % (r, g, b, cx + rr, cy,
                              cx + rr, cy + c, cx + c, cy + rr, cx, cy + rr,
                              cx - c, cy + rr, cx - rr, cy + c, cx - rr, cy,
                              cx - rr, cy - c, cx - c, cy - rr, cx, cy - rr,
                              cx + c, cy - rr, cx + rr, cy - c, cx + rr, cy))
            elif kind == "rect":
                (x, y, w, h), width, fill, stroke = op[1], op[2], op[3], op[4]
                if fill is not None:
                    r, g, b = [c / 255.0 for c in fill]
                    out.append("%.3f %.3f %.3f rg %.3f %.3f %.3f %.3f re f" % (r, g, b, X(x), Y(y + h), X(w), X(h)))
                if stroke is not None:
                    r, g, b = [c / 255.0 for c in stroke]
                    out.append("%.3f %.3f %.3f RG %.3f w %.3f %.3f %.3f %.3f re S"
                               % (r, g, b, max(0.2, X(width)), X(x), Y(y + h), X(w), X(h)))
            elif kind == "text":
                (x, y), s, size_pt, color, anchor = op[1], op[2], op[3], op[4], op[5]
                rot = float(op[6]) if len(op) > 6 else 0.0
                r, g, b = [c / 255.0 for c in color]
                txt = "".join((ch if 32 <= ord(ch) < 127 else "?") for ch in str(s))
                txt = txt.replace("\\", "").replace("(", "[").replace(")", "]")
                size = max(4.0, float(size_pt))
                approx = 0.52 * size * len(txt)
                shift = -approx / 2.0 if anchor == "center" else (-approx if anchor == "right" else 0.0)
                theta = math.radians(rot)
                ct, st = math.cos(theta), math.sin(theta)
                px, py = X(x), Y(y)
                # 基线偏移（沿文本方向）+ -90 度时向上生长 => 屏幕 y 向上，PDF y 也向上，直接用 Tm
                ox = px + shift * ct
                oy = py - shift * st * (-1.0)
                out.append("BT /F1 %.2f Tf %.3f %.3f %.3f rg %.4f %.4f %.4f %.4f %.4f %.4f Tm (%s) Tj ET"
                           % (size, r, g, b, ct, -st, st, ct, ox, oy, txt))
        return "\n".join(out)

    def to_pdf(self, path: Any) -> str:
        page_w = self.w * 72.0 / float(self.dpi)
        page_h = self.h * 72.0 / float(self.dpi)
        content = self._pdf_ops().encode("latin-1", errors="replace")
        objs: List[bytes] = []
        objs.append(b"<< /Type /Catalog /Pages 2 0 R >>")
        objs.append(b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>")
        objs.append(("<< /Type /Page /Parent 2 0 R /MediaBox [0 0 %.2f %.2f] "
                     "/Resources << /Font << /F1 5 0 R >> >> /Contents 4 0 R >>" % (page_w, page_h)).encode("latin-1"))
        objs.append(b"<< /Length " + str(len(content)).encode() + b" >>\nstream\n" + content + b"\nendstream")
        objs.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>")

        blob = bytearray(b"%PDF-1.4\n")
        offsets = [0]
        for i, body in enumerate(objs, start=1):
            offsets.append(len(blob))
            blob += ("%d 0 obj\n" % i).encode("latin-1") + body + b"\nendobj\n"
        xref_at = len(blob)
        blob += ("xref\n0 %d\n" % (len(objs) + 1)).encode("latin-1")
        blob += b"0000000000 65535 f \n"
        for off in offsets[1:]:
            blob += ("%010d 00000 n \n" % off).encode("latin-1")
        blob += ("trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n"
                 % (len(objs) + 1, xref_at)).encode("latin-1")
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(bytes(blob))
        return str(p)


def write_png(path: Any, rgb: np.ndarray) -> str:
    """最小 PNG 编码器（8 位 RGB，无第三方依赖，确定性输出）。"""
    rgb = np.ascontiguousarray(rgb.astype(np.uint8))
    h, w, _ = rgb.shape
    raw = bytearray()
    for y in range(h):
        raw.append(0)
        raw += rgb[y].tobytes()

    def chunk(tag: bytes, data: bytes) -> bytes:
        return (len(data).to_bytes(4, "big") + tag + data
                + (zlib.crc32(tag + data) & 0xFFFFFFFF).to_bytes(4, "big"))

    ihdr = w.to_bytes(4, "big") + h.to_bytes(4, "big") + bytes([8, 2, 0, 0, 0])
    blob = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"IDAT", zlib.compress(bytes(raw), 6)) + chunk(b"IEND", b"")
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(blob)
    return str(p)


# ---- 图表绘制 -----------------------------------------------------------------------
_PALETTE = [
    (0, 90, 170), (200, 60, 40), (20, 140, 90), (140, 90, 170), (210, 140, 20), (80, 80, 80),
]


def _fit_size(cv: "Canvas", s: str, max_px: float, size_pt: float, min_pt: float = 4.0) -> float:
    """自动缩字号，保证文本宽度不超过给定像素宽（避免标题/轴标签被裁掉）。"""
    size = float(size_pt)
    while size > min_pt and cv._text_len(s, size) > max_px:
        size -= 0.5
    return size


def _nice_ticks(lo: float, hi: float, n: int = 5) -> List[float]:
    if not np.isfinite(lo) or not np.isfinite(hi):
        return [0.0, 1.0]
    if hi <= lo:
        hi = lo + 1.0
    span = hi - lo
    step = 10.0 ** math.floor(math.log10(span / max(1, n)))
    for mult in (1, 2, 2.5, 5, 10):
        if span / (step * mult) <= n:
            step *= mult
            break
    start = math.floor(lo / step) * step
    ticks = []
    v = start
    while v <= hi + step * 0.5:
        if v >= lo - step * 0.5:
            ticks.append(round(v, 10))
        v += step
    return ticks or [lo, hi]


def _log_ticks(lo: float, hi: float) -> List[float]:
    lo = max(1e-9, float(lo))
    hi = max(lo * 1.0001, float(hi))
    out = []
    e0, e1 = math.floor(math.log10(lo)), math.ceil(math.log10(hi))
    for e in range(int(e0), int(e1) + 1):
        for m in (1, 2, 5):
            v = m * (10.0 ** e)
            if lo * 0.999 <= v <= hi * 1.001:
                out.append(v)
    return out or [lo, hi]


def _fmt_tick(v: float) -> str:
    if v == 0:
        return "0"
    if abs(v) >= 1000 or (abs(v) < 0.01 and v != 0):
        return "%.0e" % v
    if abs(v) >= 10:
        return "%.0f" % v
    if abs(v) >= 1:
        return "%.1f" % v
    if abs(v) >= 0.1:
        return "%.2f" % v
    return "%.3f" % v


def _plot_xy_mpl(
    series: Sequence[Mapping[str, Any]],
    path_pdf: Any,
    path_png: Any,
    *,
    title: str = "",
    xlabel: str = "",
    ylabel: str = "",
    log_x: bool = False,
    figsize: Tuple[float, float] = (6.0, 3.8),
    dpi: int = 300,
) -> Dict[str, str]:
    """matplotlib 版本（本机 .venv 已装 matplotlib 3.11）——论文首选，矢量 PDF + 300 dpi PNG。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    colors = ["#1f4e9c", "#c23b22", "#128c5a", "#8c5aae", "#d28c14", "#505050", "#0f7f8c", "#8c1f4e"]
    for si, s in enumerate(series):
        xs = [float(v) for v in (s.get("x") or [])]
        ys = [float(v) for v in (s.get("y") or [])]
        if not xs:
            continue
        yerr = s.get("yerr")
        color = s.get("color") or colors[si % len(colors)]
        label = str(s.get("label", "series%d" % si))
        if yerr is not None and any(float(v or 0) > 0 for v in yerr):
            ax.errorbar(xs, ys, yerr=[float(v or 0) for v in yerr], color=color, marker="o",
                        markersize=4.0, linewidth=1.6, capsize=3, label=label)
        else:
            ax.plot(xs, ys, color=color, marker="o", markersize=4.0, linewidth=1.6, label=label)
    if log_x:
        ax.set_xscale("log")
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    if title:
        ax.set_title(title, fontsize=10)
    ax.grid(True, alpha=0.3, linewidth=0.6)
    ax.tick_params(labelsize=8)
    ax.legend(fontsize=7.5, framealpha=0.95, loc="best")
    fig.tight_layout()
    Path(path_pdf).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(path_pdf), dpi=dpi)
    fig.savefig(str(path_png), dpi=dpi)
    plt.close(fig)
    return {"pdf": str(path_pdf), "png": str(path_png), "backend": "matplotlib"}


def plot_xy(
    series: Sequence[Mapping[str, Any]],
    path_pdf: Any,
    path_png: Any,
    *,
    title: str = "",
    xlabel: str = "",
    ylabel: str = "",
    log_x: bool = False,
    figsize: Tuple[float, float] = (6.0, 3.8),
    dpi: int = 300,
) -> Dict[str, str]:
    """画多序列折线/散点图（带误差棒），同时输出 PDF 与 PNG。

    优先 matplotlib（矢量 PDF + 300 dpi PNG）；任何异常都回退到内置无依赖后端（PDF 矢量 + PNG 位图字体），
    因此即使换到没有 matplotlib 的解释器也能出图。

    series 元素：{"label","x","y","yerr"(可选),"xerr"(可选),"color"(可选),"marker"(可选)}
    """
    try:
        return _plot_xy_mpl(series, path_pdf, path_png, title=title, xlabel=xlabel, ylabel=ylabel,
                            log_x=log_x, figsize=figsize, dpi=dpi)
    except Exception as exc:  # pragma: no cover - 取决于环境
        print("    [plot] matplotlib 不可用（%s: %s），回退内置绘图后端" % (type(exc).__name__, exc))
    return _plot_xy_canvas(series, path_pdf, path_png, title=title, xlabel=xlabel, ylabel=ylabel,
                           log_x=log_x, figsize=figsize, dpi=dpi)


def _plot_xy_canvas(
    series: Sequence[Mapping[str, Any]],
    path_pdf: Any,
    path_png: Any,
    *,
    title: str = "",
    xlabel: str = "",
    ylabel: str = "",
    log_x: bool = False,
    figsize: Tuple[float, float] = (6.0, 3.8),
    dpi: int = 300,
) -> Dict[str, str]:
    """内置无依赖绘图后端（无 matplotlib 时的兜底；PDF 为矢量、PNG 为 300 dpi 位图字体）。"""
    W = int(round(figsize[0] * dpi))
    H = int(round(figsize[1] * dpi))
    cv = Canvas(W, H, dpi)
    m_left, m_right, m_top, m_bottom = 0.17 * W, 0.03 * W, 0.12 * H, 0.19 * H
    px0, px1 = m_left, W - m_right
    py0, py1 = H - m_bottom, m_top  # 绘图区上下（像素，y 向下）

    def tx(v):
        if log_x:
            v = max(1e-9, float(v))
            lo, hi = tx.lo, tx.hi
            return px0 + (math.log10(v) - math.log10(lo)) / max(1e-9, math.log10(hi) - math.log10(lo)) * (px1 - px0)
        return px0 + (float(v) - tx.lo) / max(1e-12, tx.hi - tx.lo) * (px1 - px0)

    xs = [float(v) for s in series for v in (s.get("x") or [])]
    ys = [float(v) for s in series for v in (s.get("y") or []) if np.isfinite(float(v))]
    if not xs:
        xs = [0.0, 1.0]
    if not ys:
        ys = [0.0, 1.0]
    ty_lo, ty_hi = min(ys), max(ys)
    if ty_hi <= ty_lo:
        ty_hi = ty_lo + max(1e-3, abs(ty_lo) * 0.1 + 1e-3)
    else:
        pad = 0.12 * (ty_hi - ty_lo)
        ty_lo, ty_hi = max(0.0, ty_lo - pad), ty_hi + pad
    tx.lo, tx.hi = (min(xs), max(xs)) if not log_x else (max(1e-6, min(xs)), max(xs))
    tx.lo = tx.lo if not log_x else max(1e-6, tx.lo)
    tx.hi = tx.hi if tx.hi > tx.lo else tx.lo + 1.0
    ty = type("T", (), {})()
    ty.lo, ty.hi = ty_lo, ty_hi

    def ty_map(v):
        return py0 + (float(v) - ty.lo) / max(1e-12, ty.hi - ty.lo) * (py1 - py0)

    # 背景与网格
    cv.rect(px0, py1, px1 - px0, py0 - py1, fill=None, stroke=(120, 120, 120), width=1.2)
    xticks = _log_ticks(tx.lo, tx.hi) if log_x else _nice_ticks(tx.lo, tx.hi, 5)
    yticks = _nice_ticks(ty.lo, ty.hi, 5)
    for t in yticks:
        if t < ty.lo or t > ty.hi:
            continue
        y = ty_map(t)
        cv.line(px0, y, px1, y, 0.6, (220, 220, 220))
        cv.text(px0 - 8, y, _fmt_tick(t), 8.0, (40, 40, 40), "right")
    for t in xticks:
        if t < tx.lo or t > tx.hi:
            continue
        x = tx(t) if not log_x else tx(t)
        cv.line(x, py0, x, py1, 0.6, (235, 235, 235))
        cv.text(x, py0 + 14, _fmt_tick(t), 8.0, (40, 40, 40), "center")
    # 轴标签（自动缩字号 + y 轴标签旋转 90 度）
    x_size = _fit_size(cv, xlabel, px1 - px0, 9.0)
    cv.text((px0 + px1) / 2.0, H - 24, xlabel, x_size, (20, 20, 20), "center")
    y_size = _fit_size(cv, ylabel, py0 - py1, 9.0)
    cv.text(30, (py0 + py1) / 2.0, ylabel, y_size, (20, 20, 20), "center", rotation=-90.0)
    if title:
        cv.text((px0 + px1) / 2.0, 26, title, _fit_size(cv, title, W - 40, 10.5), (10, 10, 10), "center")

    # 数据
    legend_items = []
    for si, s in enumerate(series):
        color = tuple(s.get("color") or _PALETTE[si % len(_PALETTE)])
        xs_s = [float(v) for v in (s.get("x") or [])]
        ys_s = [float(v) for v in (s.get("y") or [])]
        yerr = s.get("yerr") or [0.0] * len(ys_s)
        order = np.argsort(xs_s) if not log_x else np.argsort([math.log10(max(1e-9, v)) for v in xs_s])
        pts = []
        for i in order.tolist():
            if not np.isfinite(ys_s[i]):
                continue
            xx = tx(xs_s[i])
            yy = ty_map(ys_s[i])
            pts.append((xx, yy))
            e = float(yerr[i]) if i < len(yerr) and np.isfinite(float(yerr[i])) else 0.0
            if e > 0:
                ey0, ey1 = ty_map(max(ty.lo, ys_s[i] - e)), ty_map(min(ty.hi, ys_s[i] + e))
                cv.line(xx, ey0, xx, ey1, 1.4, color)
                cv.line(xx - 6, ey0, xx + 6, ey0, 1.4, color)
                cv.line(xx - 6, ey1, xx + 6, ey1, 1.4, color)
        if len(pts) > 1:
            cv.polyline(pts, 2.0, color)
        for xx, yy in pts:
            cv.marker(xx, yy, 5.0, color)
        legend_items.append((s.get("label", "series%d" % si), color))

    # 图例（左上角，白底方框，避免压住数据线）
    if legend_items:
        lsize = 7.5
        widest = max(cv._text_len(lbl, lsize) for lbl, _ in legend_items)
        box_w = widest + 72
        box_h = 20 * len(legend_items) + 12
        cv.rect(px0 + 10, py1 + 10, box_w, box_h, fill=(255, 255, 255), stroke=(170, 170, 170), width=0.8)
        lx = px0 + 22
        ly = py1 + 26
        for i, (label, color) in enumerate(legend_items):
            cv.line(lx, ly + i * 20, lx + 22, ly + i * 20, 2.2, color)
            cv.marker(lx + 11, ly + i * 20, 3.5, color)
            cv.text(lx + 30, ly + i * 20, label, lsize, (30, 30, 30), "left")

    Path(path_pdf).parent.mkdir(parents=True, exist_ok=True)
    cv.to_pdf(path_pdf)
    cv.to_png(path_png)
    return {"pdf": str(path_pdf), "png": str(path_png), "backend": "builtin"}



# ======================================================================================
# 5. 实验规划
# ======================================================================================
def _shadow_ratio_grid(cfg: Mapping[str, Any], smoke: bool, min_n: int = 3, with_full: bool = True) -> List[float]:
    """E6 的影子比例网格（= 被撤文档中含影子副本的比例）。

    with_full=True 时补 1.0（全体被撤文档都带影子）以便复现机制上界
    （父代理独立探针口径：朴素删除 183/183 = 1.000，闭包删除 0/183 = 0.000）。
    """
    ratios = [float(x) for x in (cfg.get("shadow_ratios") or [0.0])]
    if cfg.get("_shadow_explicit"):
        return sorted(set(ratios))  # CLI 显式指定覆盖率时不再并入默认网格（便于限定跑批规模）
    target = {0.0, 0.10, 0.30} | ({1.0} if with_full else set())
    if len(ratios) < min_n or with_full:
        ratios = sorted(set(ratios) | target)
    return sorted(ratios)


def _methods_of(cfg: Mapping[str, Any]) -> List[str]:
    out = [str(cfg.get("method") or "fedrevoke")]
    for m in cfg.get("baselines") or []:
        if str(m) not in out:
            out.append(str(m))
    return out


def plan_run_points(cfg: Mapping[str, Any], smoke: bool = False, limit: Optional[int] = None) -> List["RunPoint"]:
    """把 YAML 配置展开成实验格（RunPoint）列表。"""
    exp = str(cfg.get("exp") or "exp")
    methods = _methods_of(cfg)
    points: List[RunPoint] = []

    datasets = list(cfg.get("datasets") or [cfg.get("dataset") or "synthetic"])
    if smoke:
        datasets = ["synthetic"]
    silos_list = [int(x) for x in (cfg.get("silos_grid") or [cfg.get("silos") or 3])]
    alpha_list = [float(x) for x in (cfg.get("dirichlet_alpha_grid") or [cfg.get("dirichlet_alpha") or 0.5])]
    rev_ratios = [float(x) for x in (cfg.get("revocation_ratios") or [0.05])]
    shadow_ratios = [float(x) for x in (cfg.get("shadow_ratios") or [0.10])]
    variants = list(cfg.get("variants") or ["full"])
    seeds = [int(x) for x in (cfg.get("seeds") or [cfg.get("seed") or C.SEED])]

    # ---- stage: main（主实验 / 孤岛规模 / 跨数据集） ----
    for ds in datasets:
        for silos in silos_list:
            for alpha in alpha_list:
                for rev in rev_ratios:
                    for sh in shadow_ratios:
                        for variant in variants:
                            for seed in seeds:
                                points.append(RunPoint(
                                    exp=exp, dataset=str(ds), silos=silos, revocation_ratio=rev,
                                    shadow_ratio=sh, ratio_key="r%d" % int(round(rev * 100)),
                                    seed=seed, alpha=alpha, variant=variant, methods=list(methods),
                                    stage="main",
                                ))

    # ---- stage: e6（动机实验：朴素删除残留 vs 影子比例） ----
    # E6 需要足够的被撤文档才能给出有统计意义的真实查询读数；优先取 5% 档
    base_rev = 0.05 if any(abs(r - 0.05) < 1e-9 for r in rev_ratios) else (rev_ratios[0] if rev_ratios else 0.05)
    main_methods = [m for m in methods if m in ("naive_delete", "full_rebuild", "fedrevoke", "sisa")]
    for ds in datasets[:1]:
        for sh in _shadow_ratio_grid(cfg, smoke):
            for seed in seeds:
                points.append(RunPoint(
                    exp=exp, dataset=str(ds), silos=silos_list[0], revocation_ratio=base_rev,
                    shadow_ratio=sh, ratio_key="r%d" % int(round(base_rev * 100)), seed=seed,
                    alpha=alpha_list[0], variant="full",
                    methods=list(main_methods) or list(methods), stage="e6",
                    extra={"coverage": float(sh)},
                ))

    # ---- stage: ablation（消融：去掉影子闭包 / 修复 / 校准，kNN 与阈值扫描） ----
    if cfg.get("variants") or cfg.get("knn_grid") or cfg.get("sim_threshold_grid"):
        ab_rev, ab_sh = base_rev, (shadow_ratios[0] if shadow_ratios else 0.10)
        for variant in ["full", "no_shadow_closure", "no_repair", "no_calibration"]:
            points.append(RunPoint(
                exp=exp, dataset=datasets[0], silos=silos_list[0], revocation_ratio=ab_rev,
                shadow_ratio=ab_sh, ratio_key="r%d" % int(round(ab_rev * 100)), seed=seeds[0],
                variant=variant, methods=[str(cfg.get("method") or "fedrevoke")], stage="ablation",
            ))
        for knn in (cfg.get("knn_grid") or []):
            points.append(RunPoint(
                exp=exp, dataset=datasets[0], silos=silos_list[0], revocation_ratio=ab_rev,
                shadow_ratio=ab_sh, ratio_key="r%d" % int(round(ab_rev * 100)), seed=seeds[0],
                variant="full", knn_k=int(knn), methods=[str(cfg.get("method") or "fedrevoke")],
                stage="ablation",
            ))
        # U3 因果对照：同一次撤回下 FedRevoke(anchor 修复) vs 随机副本对照
        points.append(RunPoint(
            exp=exp, dataset=datasets[0], silos=silos_list[0], revocation_ratio=ab_rev,
            shadow_ratio=ab_sh, ratio_key="r%d" % int(round(ab_rev * 100)), seed=seeds[0],
            variant="repair_control",
            methods=[str(cfg.get("method") or "fedrevoke"), "random_replica"], stage="ablation",
        ))
        for sim in (cfg.get("sim_threshold_grid") or []):
            points.append(RunPoint(
                exp=exp, dataset=datasets[0], silos=silos_list[0], revocation_ratio=ab_rev,
                shadow_ratio=ab_sh, ratio_key="r%d" % int(round(ab_rev * 100)), seed=seeds[0],
                variant="full", sim_threshold=float(sim),
                methods=[str(cfg.get("method") or "fedrevoke")], stage="ablation",
            ))

    # ---- stage: e7（双通道覆盖曲线：仅向量 / 仅文本 / 双通道并集） ----
    if cfg.get("e7", True):
        sim_grid = [float(x) for x in (cfg.get("sim_threshold_grid") or [0.85, 0.90, 0.92, 0.95])]
        lsh_grid = [float(x) for x in (cfg.get("lsh_threshold_grid") or [0.50, 0.70, 0.80, 0.90])]
        if smoke:
            sim_grid = sim_grid[:2]
            lsh_grid = lsh_grid[:2]
        for tau in sim_grid:
            for jac in lsh_grid:
                points.append(RunPoint(
                    exp=exp, dataset=datasets[0], silos=silos_list[0], revocation_ratio=base_rev,
                    shadow_ratio=(shadow_ratios[0] if shadow_ratios else 0.10),
                    ratio_key="r%d" % int(round(base_rev * 100)), seed=seeds[0],
                    sim_threshold=tau, methods=["detect_vector_only", "detect_text_only", "detect_dual"],
                    stage="e7", extra={"tau": tau, "jaccard": jac, "sample_pairs": 30 if smoke else 200},
                ))

    # ---- stage: cost（代价扫描：语料规模 × 方法） ----
    if cfg.get("corpus_sizes"):
        cost_methods = [m for m in methods
                        if m in ("full_rebuild", "full_rebuild_reencode", "naive_delete", "sisa",
                                 "fedrevoke", "tdsc_adapter", "random_replica", "lora_finetune")]
        for n_docs in (cfg.get("corpus_sizes") or []):
            for sh in [shadow_ratios[0] if shadow_ratios else 0.10]:
                points.append(RunPoint(
                    exp=exp, dataset=datasets[0], silos=silos_list[0], revocation_ratio=base_rev,
                    shadow_ratio=sh, ratio_key="r%d" % int(round(base_rev * 100)), seed=seeds[0],
                    n_docs=int(n_docs), methods=list(cost_methods) or list(methods), stage="cost",
                ))

    stages = cfg.get("_stages")
    if stages:
        want = {str(s).strip() for s in stages if str(s).strip()}
        points = [p for p in points if p.stage in want]

    if smoke:
        # 冒烟：限制点数，保证 60 秒内跑完
        budget = int(limit) if limit else 14
        seen, kept = set(), []
        for p in points:
            k = p.key()
            if k in seen:
                continue
            seen.add(k)
            kept.append(p)
            if len(kept) >= budget:
                break
        points = kept
    elif limit:
        points = points[: int(limit)]
    return points


# ======================================================================================
# 6. 数据准备（合成 / 真实）
# ======================================================================================
def _synthetic_for_point(cfg: Mapping[str, Any], point: "RunPoint", cache: Dict[str, Any]) -> Dict[str, Any]:
    n_docs = int(point.n_docs or cfg.get("n_docs") or 2000)
    n_q = int(cfg.get("n_queries_eval") or 50)
    dim = int(cfg.get("dim") or 64)
    key = (n_docs, point.silos, round(point.shadow_ratio, 4), point.seed, dim, n_q, round(point.alpha, 4))
    if key not in cache:
        t0 = time.perf_counter()
        cache[key] = synthesize_dataset(
            n_docs=n_docs, n_clients=point.silos, n_queries=n_q, shadow_ratio=point.shadow_ratio,
            dim=dim, seed=point.seed, alpha=point.alpha,
            forget_ratios=[float(x) for x in (cfg.get("revocation_ratios") or [0.05])],
            n_topics=int(cfg.get("n_topics") or 5), n_qa=5,
        )
        cache[key]["_build_seconds"] = time.perf_counter() - t0
    return cache[key]


def _real_for_point(cfg: Mapping[str, Any], ds_key: str, cache: Dict[str, Any], limit_queries: Optional[int]) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
    if ds_key in cache:
        ds = cache[ds_key]
        return ds, (ds or {}).get("_status", {})
    ds, status = load_real_dataset(ds_key, limit_queries=limit_queries)
    if ds is not None:
        ds["_status"] = status
    cache[ds_key] = ds
    return ds, status


# ======================================================================================
# 7. 单点执行 → 行记录
# ======================================================================================
def _row_from_result(point: "RunPoint", cfg: Mapping[str, Any], method: str, res: Mapping[str, Any],
                     fp: Mapping[str, Any], stage: str, notes: str) -> Dict[str, Any]:
    forget = dict(res.get("forget") or {})
    utility = dict(res.get("utility") or {})
    cost = dict(res.get("cost") or {})
    meta = dict(res.get("meta") or {})
    row = {
        "exp": point.exp, "dataset": point.dataset, "silos": point.silos,
        "revocation_ratio": point.revocation_ratio, "shadow_ratio": point.shadow_ratio,
        "method": method,
        "hit_rate": forget.get("hit_rate", NAN),
        # 论文 Eq.(16) 命名：hit_seed = 被删 id 是否进 top-k（结构性恒 0，sanity）；
        # hit_surrogate = 闭包内影子代理是否仍可召回（有信息量的残留读数）
        "hit_seed": forget.get("hit_rate", NAN),
        "hit_surrogate": forget.get("surrogate_hit_rate", NAN),
        "mia_auc": forget.get("mia_auc", NAN),
        "elicit_rate": forget.get("elicit_rate", NAN),
        "recall10": utility.get("recall10", NAN),
        "ndcg10": utility.get("ndcg10", NAN),
        "em": utility.get("em", NAN),
        "f1": utility.get("f1", NAN),
        "reindex_seconds": cost.get("reindex_seconds", NAN),
        "bytes_transferred": cost.get("bytes_transferred", NAN),
        "peak_vram_mb": cost.get("peak_vram_mb", NAN),
        "n_vectors_touched": cost.get("n_vectors_touched", NAN),
        "seed": point.seed,
        "rho_hat": forget.get("rho_hat", NAN),
        "rho_hat_tv": utility.get("rho_hat_tv", NAN),
        "rho_hat_ret": forget.get("rho_hat_ret", utility.get("rho_hat_tv", NAN)),
        "rho_hat_bound": forget.get("rho_hat_bound", NAN),
        "n_unrel": utility.get("n_unrel", 0),
        "unrel_reliable": bool(utility.get("unrel_reliable", False)),
        "recall10_unrel": utility.get("recall10_unrel", NAN),
        "recall10_unrel_before": utility.get("recall10_unrel_before", NAN),
        "delta_recall10_unrel": utility.get("delta_recall10_unrel", NAN),
        "ndcg10_unrel": utility.get("ndcg10_unrel", NAN),
        "delta_ndcg10_unrel": utility.get("delta_ndcg10_unrel", NAN),
        "utility_primary": utility.get("primary_metric", ""),
        "primary_value": utility.get("primary_value", NAN),
        "faithfulness": utility.get("faithfulness", NAN),
        "n_official": int((utility.get("families", {}) or {}).get("official", {}).get("n", 0)),
        "ndcg10_official": (utility.get("families", {}) or {}).get("official", {}).get("ndcg10", NAN),
        "recall10_official": (utility.get("families", {}) or {}).get("official", {}).get("recall10", NAN),
        "n_synthetic_q": int(sum(v.get("n", 0) for k, v in (utility.get("families", {}) or {}).items()
                                 if str(k).startswith("synthetic"))),
        "ndcg10_synthetic": next((v.get("ndcg10") for k, v in sorted((utility.get("families", {}) or {}).items())
                                  if str(k).startswith("synthetic")), NAN),
        "recall10_synthetic": next((v.get("recall10") for k, v in sorted((utility.get("families", {}) or {}).items())
                                    if str(k).startswith("synthetic")), NAN),
        "residual_doc_rate": forget.get("residual_doc_rate", NAN),
        "residual_doc_rate_cond": forget.get("residual_doc_rate_cond", NAN),
        "self_probe_residual": forget.get("self_probe_residual", NAN),
        "real_query_residual": forget.get("real_query_residual", NAN),
        "real_query_residual_k50": forget.get("real_query_residual_k50", NAN),
        "retrieved_surrogate_fraction_realq": forget.get("retrieved_surrogate_fraction_realq", NAN),
        "revoked_evidence_residual": forget.get("revoked_evidence_residual", NAN),
        "n_revoked_evidence_queries": forget.get("n_revoked_evidence_queries", 0),
        "unrel_query_residual": forget.get("unrel_query_residual", NAN),
        "n_self_probes": forget.get("n_self_probes", 0),
        "n_real_queries": forget.get("n_real_queries", 0),
        "n_unrel_queries": forget.get("n_unrel_queries", 0),
        "retrieved_surrogate_fraction": forget.get("retrieved_surrogate_fraction", NAN),
        "n_surviving_surrogate_docs": forget.get("n_surviving_surrogate_docs", 0),
        "n_replica_control": meta.get("n_replica_control", 0),
        "n_vectors_delta": meta.get("n_vectors_delta", 0),
        "m2_closure_s": meta.get("m2_closure_s", NAN),
        "m3_erase_s": meta.get("m3_erase_s", NAN),
        "m4_repair_s": meta.get("m4_repair_s", NAN),
        "m2_share": meta.get("m2_share", NAN),
        "m3_share": meta.get("m3_share", NAN),
        "n_shadowed_forgotten": forget.get("n_shadowed_forgotten", 0),
        "n_forgotten_docs": forget.get("n_forgotten_docs", 0),
        "surrogate_hit_rate": forget.get("surrogate_hit_rate", NAN),
        "n_surrogates": forget.get("n_surrogates", 0),
        "n_queries": utility.get("n_queries", 0),
        "n_deleted": cost.get("n_deleted", meta.get("n_deleted", 0)),
        "n_alive_after": cost.get("n_alive_after", cost.get("n_alive", NAN)),
        "recall_min": utility.get("recall_min", NAN),
        "recall_std": utility.get("recall_std", NAN),
        "n_seed": meta.get("n_seed", 0),
        "n_closure": meta.get("n_closure", meta.get("n_deleted", 0)),
        "closure_precision": meta.get("closure_precision", NAN),
        "closure_recall": meta.get("closure_recall", NAN),
        "n_reconnected": meta.get("n_reconnected", 0),
        "score_shift": meta.get("score_shift", 0.0),
        "elicit_measured": bool(forget.get("elicit_measured", False)),
        # 逐条命中标记（"0/1" 串）：与基线同 item 顺序 → 可直接算配对 McNemar
        "elicit_flags": "".join(str(int(x)) for x in (forget.get("elicit_flags") or [])),
        "n_qa_items": int(len(forget.get("elicit_flags") or []) or utility.get("n_qa_items", 0) or 0),
        "gold_ref_scope": "gold_intersect_revoked",
        "elicit_rate_closed_book": forget.get("elicit_rate_closed_book", NAN),
        "elicit_flags_closed_book": "".join(str(int(x)) for x in (forget.get("elicit_flags_closed_book") or [])),
        "notes": "; ".join(x for x in [
            "stage=%s" % stage,
            "variant=%s" % point.variant,
            "alpha=%.2f" % point.alpha,
            "knn_k=%s" % point.knn_k,
            "sim_threshold=%s" % point.sim_threshold,
            "fingerprint=%s" % json.dumps(fp.get("counts", {}), ensure_ascii=False),
            notes,
        ] if x and not x.endswith("=None")),
    }
    if meta.get("degraded"):
        row["notes"] += "; degraded=%s" % meta.get("degradation_reason", "")
    if meta.get("error"):
        row["notes"] += "; error=%s" % meta.get("error")
    if forget.get("elicit_error"):
        row["notes"] += "; elicit_error=%s" % forget.get("elicit_error")
    return row


E7_CHANNELS = {
    "detect_vector_only": (True, False),
    "detect_text_only": (False, True),
    "detect_dual": (True, True),
}


def run_e7_point(point: "RunPoint", cfg: Mapping[str, Any], smoke: bool, caches: Dict[str, Any],
                 fp: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """E7：在真实注入影子对上比较 仅向量 / 仅文本 / 双通道 的漏检率与误删规模。

    分母固定为同一组注入影子对（跨孤岛、含中等相似困难样本），因此三种通道可直接比较。
    """
    if point.dataset == "synthetic":
        ds = _synthetic_for_point(cfg, point, caches["synth"])
    else:
        ds, _ = _real_for_point(cfg, point.dataset, caches["real"], cfg.get("n_queries_eval"))
    if ds is None:
        return []
    tau = float((point.extra or {}).get("tau", cfg.get("shadow", {}).get("sim_threshold", 0.92)))
    jac = float((point.extra or {}).get("jaccard", cfg.get("shadow", {}).get("lsh_threshold", 0.80)))
    max_pairs = int((point.extra or {}).get("sample_pairs", 150))
    smap = ds.get("shadow_map") or {}
    pairs = [(str(o), str(s)) for o, sv in sorted(smap.items()) for s in sv]
    if not pairs:
        return []
    rng = np.random.default_rng(int(point.seed))
    if len(pairs) > max_pairs:
        sel = np.sort(rng.choice(len(pairs), max_pairs, replace=False))
        pairs = [pairs[int(i)] for i in sel]
    index = build_index(ds)
    # 关键：seed 必须取"该影子副本对应的原始 chunk"（shadow_pairs 的 orig_pid），
    # 否则从文档的其它 chunk 出发根本扩散不到该影子 —— 会系统性高估漏检率。
    pid_to_iid = {}
    try:
        for i, m in enumerate(index.metas_snapshot()):
            pid_to_iid[int(m.pid)] = i
    except Exception:
        pass
    pair_chunks = ds.get("shadow_chunk_pairs") or {}
    rows: List[Dict[str, Any]] = []
    for variant, (use_vec, use_txt) in E7_CHANNELS.items():
        det = ShadowDetector(
            sim_threshold=tau, lsh_threshold=jac,
            knn_k=int((cfg.get("shadow") or {}).get("knn_k", 50)),
            cross_client_only=True, seed=int(point.seed),
            signatures=ds.get("signatures"), vector_channel=use_vec, text_channel=use_txt,
        )
        n_hit = 0
        closures: List[int] = []
        seeds_used = 0
        for orig, shadow in pairs:
            seed_ids: List[int] = []
            pr = pair_chunks.get(str(shadow))
            if pr is not None:
                iid = pid_to_iid.get(int(pr[0]))
                if iid is not None:
                    seed_ids = [int(iid)]
            if not seed_ids:
                seed_ids = index.ids_for_doc(orig)[:4]
            if not seed_ids:
                continue
            seeds_used += len(seed_ids)
            try:
                cl = det.closure(index, seed_ids)
            except Exception:
                cl = set()
            shadow_ids = set(index.all_ids_for_doc(shadow))
            if cl & shadow_ids:
                n_hit += 1
            closures.append(len(cl))
        n_pairs_eff = int(len(closures))
        recall = (n_hit / n_pairs_eff) if n_pairs_eff else NAN
        row = {c: "" for c in CSV_COLUMNS}
        row.update({
            "exp": point.exp, "dataset": point.dataset, "silos": point.silos,
            "revocation_ratio": point.revocation_ratio, "shadow_ratio": point.shadow_ratio,
            "method": variant, "seed": point.seed,
            "hit_rate": recall, "recall10": NAN, "ndcg10": NAN, "em": NAN, "f1": NAN,
            "mia_auc": NAN, "elicit_rate": NAN,
            "reindex_seconds": NAN, "bytes_transferred": NAN, "peak_vram_mb": NAN, "n_vectors_touched": NAN,
            "detector_variant": variant, "sim_threshold": tau, "lsh_threshold": jac,
            "n_pairs": n_pairs_eff, "miss_rate": (1.0 - recall) if n_pairs_eff else NAN,
            "mean_closure_size": float(np.mean(closures)) if closures else NAN,
            "closure_recall": recall,
            "n_seed": seeds_used,
            "notes": "stage=e7; tau=%.3f; jaccard=%.3f; channels=%s; fingerprint=%s"
                     % (tau, jac, "vector+text" if (use_vec and use_txt) else ("vector" if use_vec else "text"),
                        json.dumps(fp.get("counts", {}), ensure_ascii=False)),
        })
        rows.append(row)
    return rows


def run_point(point: "RunPoint", cfg: Mapping[str, Any], smoke: bool, caches: Dict[str, Any],
              generator: Any) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """执行一个实验格（所有方法）并返回 (rows, info)。"""
    info: Dict[str, Any] = {"point": point.key(), "stage": point.stage, "status": "ok", "seconds": 0.0}
    t0 = time.perf_counter()
    rows: List[Dict[str, Any]] = []
    if point.dataset == "synthetic":
        ds = _synthetic_for_point(cfg, point, caches["synth"])
        status = {"name": "synthetic", "missing": []}
    else:
        ds, status = _real_for_point(cfg, point.dataset, caches["real"], cfg.get("n_queries_eval"))
        if ds is not None:
            status = dict(ds.get("status") or status)
            caches.setdefault("status", {})[point.dataset] = status
        if ds is None:
            info.update(status="skipped", reason="真实数据缺失: %s" % ", ".join(status.get("missing", [])))
            return rows, info
    # E4/E5 代价扫描：按 point.n_docs 裁剪语料（subset_dataset 保持 pid 重编号与 gold 交集一致）
    if point.n_docs and int(point.n_docs) < len(ds.get("corpus") or []):
        ds = subset_dataset(ds, int(point.n_docs), point.seed)
        info["subset_from"] = point.n_docs
    info["dataset"] = ds.get("name")
    info["n_docs"] = len(ds.get("corpus") or [])

    coverage = (point.extra or {}).get("coverage") if point.stage == "e6" else None
    doc_ids, cov_info = controlled_forget_set(ds, point.revocation_ratio, coverage, point.seed, point.ratio_key)
    info["n_forget_docs"] = len(doc_ids)
    info["coverage"] = cov_info

    # --- INTERFACES §11：建索引后立即做指纹门禁（digest 且 shadow_ratio>0 时 full 模式抛错） ---
    probe_index = build_index(ds)
    index_before = build_index(ds)  # 未变异参照索引：Q_unrel / Δ_util / rho_hat_tv 用
    detector = build_detector(cfg, ds.get("signatures"), knn_k=point.knn_k,
                              sim_threshold=point.sim_threshold,
                              vector_channel=(point.variant != "no_shadow_closure"))
    fp = assert_fingerprint_channel(
        probe_index, detector, shadow_ratio=point.shadow_ratio, smoke=bool(smoke),
        minhash_perm=int(getattr(detector, "minhash_perm", 64) or 64),
        context="%s|%s" % (point.exp, point.dataset),
    )
    info["fingerprint"] = dict(fp)
    if point.stage == "e7":
        rows = run_e7_point(point, cfg, smoke, caches, fp)
        info["seconds"] = time.perf_counter() - t0
        info["n_rows"] = len(rows)
        return rows, info

    for method in point.methods:
        index = build_index(ds)
        store = {int(k): v for k, v in (ds.get("texts") or {}).items()}
        notes = ""
        try:
            res = evaluate_method(method, index, ds, cfg, point, doc_ids, smoke=smoke,
                                  generator=generator, text_store=store, notes=notes,
                                  index_before=index_before)
        except RuntimeError:
            raise  # §11 fail-fast 必须冒泡
        except Exception as exc:
            traceback.print_exc()
            info["status"] = "partial"
            res = {
                "method": method, "forget": {}, "utility": {}, "cost": {},
                "meta": {"error": "%s: %s" % (type(exc).__name__, exc)},
            }
        rows.append(_row_from_result(point, cfg, method, res, fp, point.stage, notes))
    info["seconds"] = time.perf_counter() - t0
    return rows, info


# ======================================================================================
# 8. 写出 CSV / summary / 图
# ======================================================================================
def _cell(v: Any) -> Any:
    if isinstance(v, (np.floating, float)):
        f = float(v)
        if math.isnan(f):
            return ""
        return round(f, 6)
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, bool):
        return int(v)
    return v


def write_rows_csv(rows: Sequence[Mapping[str, Any]], path: Any, columns: Sequence[str] = CSV_COLUMNS) -> str:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    lines = [",".join(columns)]
    for row in rows:
        cells = []
        for col in columns:
            v = _cell(row.get(col, ""))
            s = "" if v is None else str(v)
            if any(ch in s for ch in [",", '"', "\n"]):
                s = '"' + s.replace('"', '""') + '"'
            cells.append(s)
        lines.append(",".join(cells))
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(p)


def aggregate_rows(rows: Sequence[Mapping[str, Any]], keys: Sequence[str], metrics: Sequence[str]) -> List[Dict[str, Any]]:
    buckets: Dict[Tuple, List[Mapping[str, Any]]] = {}
    for r in rows:
        buckets.setdefault(tuple(r.get(k) for k in keys), []).append(r)
    out = []
    for key, group in sorted(buckets.items(), key=lambda kv: str(kv[0])):
        rec = {k: v for k, v in zip(keys, key)}
        rec["n"] = len(group)
        for m in metrics:
            vals = [float(g.get(m)) for g in group
                    if isinstance(g.get(m), (int, float, np.integer, np.floating)) and math.isfinite(float(g.get(m)))]
            rec[m + "_mean"] = float(np.mean(vals)) if vals else NAN
            rec[m + "_std"] = float(np.std(vals, ddof=0)) if len(vals) > 1 else 0.0
        out.append(rec)
    return out


def row_stage(row: Mapping[str, Any]) -> str:
    for part in str(row.get("notes", "")).split(";"):
        part = part.strip()
        if part.startswith("stage="):
            return part.split("=", 1)[1].strip()
    return "main"


def e6_monotonicity(rows: Sequence[Mapping[str, Any]], tol: float = 0.02) -> Dict[str, Any]:
    """E6 验收：朴素删除的残留应随"被撤文档含影子比例"单调上升，FedRevoke 应贴近 0。"""
    sub = [r for r in rows if row_stage(r) == "e6"]
    if not sub:
        return {"checked": False, "reason": "no e6 rows"}
    # 单调性口径：被撤文档中仍有存活影子副本的比例（父代理预测 0% -> 0，30% -> ~0.3）
    metric = "residual_doc_rate"
    out: Dict[str, Any] = {"checked": True, "metric": metric, "methods": {}}
    methods = []
    for r in sub:
        m = str(r.get("method"))
        if m not in methods:
            methods.append(m)
    for m in methods:
        pts: Dict[float, List[float]] = {}
        for r in sub:
            if str(r.get("method")) != m:
                continue
            v = r.get(metric)
            if not isinstance(v, (int, float, np.floating)) or not math.isfinite(float(v)):
                continue
            pts.setdefault(round(float(r.get("shadow_ratio", 0.0)), 4), []).append(float(v))
        xs = sorted(pts)
        ys = [float(np.mean(pts[x])) for x in xs]
        mono = all(ys[i + 1] >= ys[i] - float(tol) for i in range(len(ys) - 1))
        out["methods"][m] = {"shadow_ratio": xs, "residual": ys, "monotone": bool(mono),
                             "residual_at_max": float(ys[-1]) if ys else NAN}
    nd = out["methods"].get("naive_delete")
    fr = out["methods"].get("fedrevoke")
    out["naive_delete_monotone"] = bool(nd.get("monotone")) if nd else None
    if nd and fr and nd.get("shadow_ratio") and fr.get("shadow_ratio"):
        out["gap_at_max_shadow"] = float(nd["residual_at_max"] - fr["residual_at_max"])
    return out


def env_report() -> Dict[str, Any]:
    import platform
    import importlib.util

    def has(m: str) -> bool:
        try:
            return importlib.util.find_spec(m) is not None
        except Exception:
            return False

    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": has("pandas"),
        "yaml": has("yaml"),
        "torch": has("torch"),
        "faiss": has("faiss"),
        "hnswlib": has("hnswlib"),
        "transformers": has("transformers"),
        "peft": has("peft"),
        "matplotlib": has("matplotlib"),
        "matplotlib_note": "未安装 matplotlib（禁止 pip install）→ 使用内置无依赖 PDF/PNG 绘图后端",
    }


def _residual_metric(rows: Sequence[Mapping[str, Any]]) -> str:
    """残留泄漏的作图指标：优先确定性文档级残留，其次生成端诱导率。

    residual_doc_rate = 被撤文档中仍有存活跨孤岛影子副本的比例（E6 主指标）
    """
    has_res = any(isinstance(r.get("residual_doc_rate"), (int, float)) and math.isfinite(float(r.get("residual_doc_rate")))
                  for r in rows)
    if has_res:
        return "residual_doc_rate"
    if any(r.get("elicit_measured") for r in rows):
        return "elicit_rate"
    return "surrogate_hit_rate"


def figure_e6(rows: Sequence[Mapping[str, Any]], exp: str, out_dir: Any) -> Optional[Dict[str, str]]:
    """E6 动机图：朴素删除的残留泄漏 vs 影子比例。"""
    import numpy as _np

    sub = [r for r in rows if row_stage(r) == "e6"] or list(rows)
    if not sub:
        return None
    metric = _residual_metric(sub)
    methods = []
    for r in sub:
        m = str(r.get("method"))
        if m not in methods:
            methods.append(m)
    series = []
    for i, m in enumerate(methods):
        pts: Dict[float, List[float]] = {}
        for r in sub:
            if str(r.get("method")) != m:
                continue
            v = r.get(metric)
            if not isinstance(v, (int, float, _np.floating)) or not math.isfinite(float(v)):
                continue
            pts.setdefault(round(float(r.get("shadow_ratio", 0.0)), 4), []).append(float(v))
        if not pts:
            continue
        xs = sorted(pts)
        ys = [float(_np.mean(pts[x])) for x in xs]
        es = [float(_np.std(pts[x], ddof=0)) if len(pts[x]) > 1 else 0.0 for x in xs]
        series.append({"label": m, "x": xs, "y": ys, "yerr": es})
    if not series:
        return None
    return plot_xy(
        series,
        Path(out_dir) / ("%s_e6_motivation.pdf" % exp),
        Path(out_dir) / ("%s_e6_motivation.png" % exp),
        title="E6: residual leakage after shard deletion vs shadow ratio",
        xlabel="shadow ratio (fraction of corpus with cross-silo duplicates)",
        ylabel="%s" % metric,
    )


def figure_e6_query_families(rows: Sequence[Mapping[str, Any]], exp: str, out_dir: Any) -> Optional[Dict[str, str]]:
    """E6 第二条图：self-query 探针（上界） / real-query（头条） / 无关查询（对照）。"""
    import numpy as _np

    sub = [r for r in rows if row_stage(r) == "e6"] or list(rows)
    if not sub:
        return None
    families = [("real_query_residual", "real query @10 (headline)"),
                ("self_probe_residual", "self-probe (upper bound)"),
                ("revoked_evidence_residual", "revoked-evidence query (proxy)")]
    present = []
    for r in sub:
        m = str(r.get("method"))
        if m not in present:
            present.append(m)
    # 只画两条代表性曲线：无闭包基线（naive_delete/full_rebuild）与 FedRevoke
    methods = [m for m in ("naive_delete", "full_rebuild") if m in present][:1] + \
              [m for m in ("fedrevoke",) if m in present]
    series = []
    for fi, (col, flabel) in enumerate(families):
        if col == "unrel_query_residual":
            pts: Dict[float, List[float]] = {}
            for r in sub:
                v = r.get(col)
                if isinstance(v, (int, float, _np.floating)) and math.isfinite(float(v)):
                    pts.setdefault(round(float(r.get("shadow_ratio", 0.0)), 4), []).append(float(v))
            if pts:
                xs = sorted(pts)
                series.append({"label": flabel, "x": xs,
                               "y": [float(_np.mean(pts[x])) for x in xs],
                               "yerr": [float(_np.std(pts[x], ddof=0)) if len(pts[x]) > 1 else 0.0 for x in xs]})
            continue
        for m in methods:
            if m not in ("naive_delete", "full_rebuild", "fedrevoke", "sisa"):
                continue
            pts = {}
            for r in sub:
                if str(r.get("method")) != m:
                    continue
                v = r.get(col)
                if isinstance(v, (int, float, _np.floating)) and math.isfinite(float(v)):
                    pts.setdefault(round(float(r.get("shadow_ratio", 0.0)), 4), []).append(float(v))
            if not pts:
                continue
            xs = sorted(pts)
            series.append({"label": "%s · %s" % (m, flabel), "x": xs,
                           "y": [float(_np.mean(pts[x])) for x in xs],
                           "yerr": [float(_np.std(pts[x], ddof=0)) if len(pts[x]) > 1 else 0.0 for x in xs]})
    if not series:
        return None
    return plot_xy(
        series,
        Path(out_dir) / ("%s_e6_query_families.pdf" % exp),
        Path(out_dir) / ("%s_e6_query_families.png" % exp),
        title="E6: residual leakage by query family",
        xlabel="shadow ratio (fraction of corpus with cross-silo duplicates)",
        ylabel="residual fraction (lower = better)",
        figsize=(6.8, 4.2),
    )


def figure_e7_channels(rows: Sequence[Mapping[str, Any]], exp: str, out_dir: Any) -> Dict[str, str]:
    """E7：仅向量 / 仅文本 / 双通道并集 的漏检率曲线（对 τ 与 Jaccard 阈值各一张）。"""
    import numpy as _np

    sub = [r for r in rows if row_stage(r) == "e7"]
    out: Dict[str, str] = {}
    if not sub:
        return out
    variants = []
    for r in sub:
        v = str(r.get("detector_variant") or r.get("method"))
        if v not in variants:
            variants.append(v)
    labels = {
        "detect_vector_only": "vector only (kNN)",
        "detect_text_only": "text only (MinHash/LSH)",
        "detect_dual": "dual channel (union)",
    }

    def series_for(xcol: str, other: str):
        ser = []
        for v in variants:
            pts: Dict[float, List[float]] = {}
            for r in sub:
                if str(r.get("detector_variant") or r.get("method")) != v:
                    continue
                x, y = r.get(xcol), r.get("miss_rate")
                if not isinstance(x, (int, float, _np.floating)) or not isinstance(y, (int, float, _np.floating)):
                    continue
                if not (math.isfinite(float(x)) and math.isfinite(float(y))):
                    continue
                pts.setdefault(round(float(x), 4), []).append(float(y))
            if not pts:
                continue
            xs = sorted(pts)
            ser.append({"label": labels.get(v, v), "x": xs,
                        "y": [float(_np.mean(pts[x])) for x in xs],
                        "yerr": [float(_np.std(pts[x], ddof=0)) if len(pts[x]) > 1 else 0.0 for x in xs]})
        return ser

    s_tau = series_for("sim_threshold", "lsh_threshold")
    if s_tau:
        r1 = plot_xy(s_tau, Path(out_dir) / ("%s_e7_channels_tau.pdf" % exp),
                     Path(out_dir) / ("%s_e7_channels_tau.png" % exp),
                     title="E7: shadow-detection miss rate vs cosine threshold",
                     xlabel="cosine similarity threshold tau",
                     ylabel="miss rate (lower = better coverage)")
        out.update({("tau_" + k): v for k, v in r1.items()})
    s_j = series_for("lsh_threshold", "sim_threshold")
    if s_j:
        r2 = plot_xy(s_j, Path(out_dir) / ("%s_e7_channels_jaccard.pdf" % exp),
                     Path(out_dir) / ("%s_e7_channels_jaccard.png" % exp),
                     title="E7: shadow-detection miss rate vs MinHash Jaccard threshold",
                     xlabel="MinHash Jaccard threshold J",
                     ylabel="miss rate (lower = better coverage)")
        out.update({("jaccard_" + k): v for k, v in r2.items()})
    return out


def figure_pareto(rows: Sequence[Mapping[str, Any]], exp: str, out_dir: Any,
                  cost_col: str = "bytes_transferred") -> Optional[Dict[str, str]]:
    """E5 代价 Pareto 图：残留泄漏 vs 重索引代价（默认传输字节，对数轴）。"""
    import numpy as _np

    if not rows:
        return None
    metric = _residual_metric(rows)
    agg = aggregate_rows(rows, ["method"], [cost_col, metric])
    xs, ys, es, labels = [], [], [], []
    for rec in agg:
        x = rec.get(cost_col + "_mean")
        y = rec.get(metric + "_mean")
        if not (isinstance(x, float) and math.isfinite(x) and x > 0):
            continue
        if not (isinstance(y, float) and math.isfinite(y)):
            continue
        xs.append(max(1e-4, float(x)))
        ys.append(float(y))
        es.append(float(rec.get(metric + "_std", 0.0) or 0.0))
        labels.append(str(rec.get("method")))
    if not xs:
        return None
    series = [{"label": "methods (mean over runs)", "x": xs, "y": ys, "yerr": es}]
    xlabel = "bytes transferred (log scale)" if cost_col == "bytes_transferred" else "%s (log scale)" % cost_col
    res = plot_xy(
        series,
        Path(out_dir) / ("%s_e5_pareto.pdf" % exp),
        Path(out_dir) / ("%s_e5_pareto.png" % exp),
        title="E5 Cost Pareto: residual leakage vs reindex cost",
        xlabel=xlabel,
        ylabel="%s (lower = better forgetting)" % metric,
        log_x=True,
        figsize=(6.4, 4.0),
    )
    # 在 PDF/PNG 上补注方法名（用第二个序列的图例位置画出点标签）
    for label, x, y in zip(labels, xs, ys):
        print("    [pareto:%s] %-16s x=%.4g y=%.3f" % (cost_col, label, x, y))
    if cost_col == "bytes_transferred":
        try:
            res2 = figure_pareto(rows, exp + "_time", out_dir, cost_col="reindex_seconds")
            res["time_pdf"] = res2.get("pdf") if res2 else None
            res["time_png"] = res2.get("png") if res2 else None
        except Exception:
            pass
    return res


# ======================================================================================
# 9. 主流程
# ======================================================================================
def run_experiment(cfg: Mapping[str, Any], smoke: bool = False, limit: Optional[int] = None,
                   out_dir: Optional[Any] = None, make_figures: bool = True) -> Dict[str, Any]:
    exp = str(cfg.get("exp") or "exp")
    t_start = time.perf_counter()
    res_dir = Path(out_dir) if out_dir else (RESULTS_DIR / exp)
    res_dir.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    points = plan_run_points(cfg, smoke=smoke, limit=limit)
    print("[run_experiments] exp=%s mode=%s points=%d methods=%s" % (
        exp, "smoke" if smoke else cfg.get("mode"), len(points), ",".join(_methods_of(cfg))))
    generator = build_generator_for(cfg, smoke)
    caches: Dict[str, Any] = {"synth": {}, "real": {}}
    rows: List[Dict[str, Any]] = []
    infos: List[Dict[str, Any]] = []
    all_points = [p.key() for p in points]
    for i, point in enumerate(points, start=1):
        new_rows, info = run_point(point, cfg, smoke, caches, generator)
        rows.extend(new_rows)
        infos.append(info)
        print("[run_experiments] (%d/%d) %s -> %s rows=%.0f %.2fs" % (
            i, len(points), point.stage, info.get("status"), len(new_rows), info.get("seconds", 0.0)))
        sys.stdout.flush()

    elapsed = time.perf_counter() - t_start

    # ---- 真实数据可用性状态（报告/摘要用） ----
    ds_keys = sorted({str(p.dataset) for p in points if str(p.dataset) != "synthetic"})
    extra_keys = [] if smoke else sorted(set(C.DATASETS) - set(ds_keys))
    loaded_status = caches.get("status") or {}
    data_status_all = {}
    for k in (ds_keys + extra_keys):
        data_status_all[k] = dict(loaded_status.get(k) or data_status(k))

    # ---- CSV ----
    csv_paths = {}
    if rows:
        csv_paths["all"] = write_rows_csv(rows, RESULTS_DIR / ("%s.csv" % exp))
        csv_paths["rows"] = write_rows_csv(rows, res_dir / "rows.csv")
        for stage in sorted({row_stage(r) for r in rows}):
            sub = [r for r in rows if row_stage(r) == stage]
            csv_paths[stage] = write_rows_csv(sub, res_dir / ("%s.csv" % stage))
    else:
        csv_paths["all"] = write_rows_csv([], RESULTS_DIR / ("%s.csv" % exp))

    # ---- summary ----
    metric_cols = ["hit_rate", "mia_auc", "elicit_rate", "rho_hat", "surrogate_hit_rate",
                   "recall10", "ndcg10", "em", "f1", "reindex_seconds", "bytes_transferred",
                   "peak_vram_mb", "n_vectors_touched"]
    agg_method = aggregate_rows(rows, ["method"], metric_cols) if rows else []
    agg_shadow = aggregate_rows(rows, ["method", "shadow_ratio"], metric_cols) if rows else []
    degradations = sorted({str(r.get("notes")) for r in rows if "degraded=" in str(r.get("notes"))})
    warnings_list = [w for w in [i.get("reason") for i in infos if i.get("status") != "ok"] if w]

    missing_real = {
        k: v.get("missing", []) for k, v in data_status_all.items() if v.get("missing")
    }
    ds_status_map = {k: {kk: vv for kk, vv in v.items() if kk in ("name", "exists", "missing", "query_vector_source", "shadow_map_source", "n_shadowed_docs")}
                     for k, v in data_status_all.items()}
    summary = {
        "exp": exp,
        "config_path": cfg.get("_config_path"),
        "mode": "smoke" if smoke else cfg.get("mode"),
        "tier": cfg.get("tier", "b" if smoke else "a"),
        "model": cfg.get("model", "mock" if isinstance(generator, MockGenerator) else "n/a"),
        "model_id": getattr(generator, "model_id", None) or type(generator).__name__,
        "generator": type(generator).__name__,
        "generator_backend": (getattr(generator, "backend", None) or ("mock" if isinstance(generator, MockGenerator) else "none")),
        "n_queries_eval": int(cfg.get("n_queries_eval") or 0),
        "stages": sorted({row_stage(r) for r in rows}) if rows else [],
        "dataset_status": ds_status_map,
        "e6_monotonicity": e6_monotonicity(rows),
        "started_unix": time.time() - elapsed,
        "elapsed_seconds": round(elapsed, 3),
        "n_points": len(points),
        "n_rows": len(rows),
        "methods": _methods_of(cfg),
        "point_keys": all_points,
        "csv": csv_paths,
        "environment": env_report(),
        "data_status": data_status_all,
        "real_data_missing": missing_real,
        "degradations": degradations,
        "skipped": warnings_list,
        "aggregates_by_method": agg_method,
        "aggregates_by_method_shadow": agg_shadow,
        "stage_seconds": infos,
        "notes": [
            "smoke 模式为现场合成数据 + MockGenerator，规模小、绝对代价数值不代表真实规模",
            "hit_rate 结构性为 0（index_core 保证 search() 不返回已删除 id）；区分方法优劣的是 elicit_rate / surrogate_hit_rate",
            "INTERFACES §11：shadow_ratio>0 时签名矩阵必须注入；digest 指纹在 full 模式下 fail-fast",
        ],
    }

    # ---- 图 ----
    figures = {}
    if make_figures and rows:
        try:
            f6 = figure_e6(rows, exp, FIGURES_DIR)
            if f6:
                figures["e6_motivation"] = f6
        except Exception as exc:
            summary["notes"].append("E6 绘图失败: %s: %s" % (type(exc).__name__, exc))
        try:
            f6b = figure_e6_query_families(rows, exp, FIGURES_DIR)
            if f6b:
                figures["e6_query_families"] = f6b
        except Exception as exc:
            summary["notes"].append("E6 query-family 绘图失败: %s: %s" % (type(exc).__name__, exc))
        try:
            f7 = figure_e7_channels(rows, exp, FIGURES_DIR)
            if f7:
                figures["e7_channels"] = f7
        except Exception as exc:
            summary["notes"].append("E7 绘图失败: %s: %s" % (type(exc).__name__, exc))
        try:
            f5 = figure_pareto(rows, exp, FIGURES_DIR)
            if f5:
                figures["e5_pareto"] = f5
        except Exception as exc:
            summary["notes"].append("E5 绘图失败: %s: %s" % (type(exc).__name__, exc))
    summary["figures"] = figures

    # ---- rho_hat 双口径 + Q_unrel 可靠性（父代理裁决 B/E） ----
    rho_tv = [r.get("rho_hat_tv") for r in rows
              if isinstance(r.get("rho_hat_tv"), (int, float)) and math.isfinite(float(r.get("rho_hat_tv")))]
    rho_cons = [r.get("rho_hat") for r in rows
                if isinstance(r.get("rho_hat"), (int, float)) and math.isfinite(float(r.get("rho_hat")))]
    summary["rho_hat"] = {
        "rho_hat_tv_mean": float(np.mean(rho_tv)) if rho_tv else NAN,
        "rho_hat_conservative_mean": float(np.mean(rho_cons)) if rho_cons else NAN,
        "definition": "rho_hat_tv/rho_hat_ret = 式(18) max_q [1-|R_k(q;I')∩R_k(q;I∅)|/k]，"
                      "oracle 侧 I∅ = 删去种子∪注入影子后重建；"
                      "rho_hat = max(rho_hat_ret, elicit_rate)（Definition 1 操作化）；"
                      "rho_hat_conservative/rho_hat_bound = max(hit_rate, elicit_rate)（无 oracle 保守上界）",
        "used_in_results_section": "rho_hat_ret + rho_hat_bound",
    }
    unrel = [(r.get("method"), r.get("n_unrel"), r.get("unrel_reliable")) for r in rows if r.get("n_unrel")]
    summary["q_unrel"] = {
        "definition": "Q_unrel = { q : gold_pids(q) ∩ deletion_closure = ∅ }；Δ_util 只在该子集上评估",
        "cells": [{"method": m, "n_unrel": int(n), "reliable": bool(rel)} for m, n, rel in unrel],
        "min_n_threshold": 50,
        "unreliable_cells": [m for m, n, rel in unrel if not rel],
    }

    (res_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    (LOGS_DIR / ("%s_run.json" % exp)).write_text(
        json.dumps({"summary": summary, "points": infos}, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    return summary


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="FedRevoke 实验入口")
    ap.add_argument("--config", default=str(CFG_DIR / "e0_smoke.yaml"), help="YAML 实验配置路径")
    ap.add_argument("--smoke", action="store_true", help="冒烟模式：合成数据 + MockGenerator，60 秒内跑完")
    ap.add_argument("--limit", type=int, default=None, help="实验格（grid point）数量上限")
    ap.add_argument("--out-dir", default=None, help="结果目录（默认 artifacts/results/<exp>）")
    ap.add_argument("--no-figures", action="store_true", help="跳过绘图")
    ap.add_argument("--tier", choices=["a", "b"], default=None,
                    help="评测档：a=纯检索全网格（默认）；b=含生成（elicit/EM/F1）缩减网格")
    ap.add_argument("--model", choices=["1.5b", "7b"], default="1.5b",
                    help="Tier B 生成器：1.5b=Qwen2.5-1.5B-Instruct；7b=Qwen2.5-7B-Instruct（4bit）")
    ap.add_argument("--dataset", default=None, help="覆盖配置中的 dataset（如 ds1）")
    ap.add_argument("--exp", default=None, help="覆盖实验名（决定 results/figures 文件名）")
    ap.add_argument("--stages", default=None, help="只跑指定 stage，逗号分隔：main,e6,ablation,cost")
    ap.add_argument("--limit-queries", type=int, default=None, help="评测查询数上限（覆盖配置）")
    ap.add_argument("--corpus-sizes", default=None, help="覆盖 cost 阶段的语料规模列表，如 2500,5000,10000")
    ap.add_argument("--methods", default=None, help="覆盖方法列表（逗号分隔；fedrevoke 表示本方法）")
    ap.add_argument("--n-anchors", type=int, default=None, help="覆盖 repair 的 anchor 数（SA7 后建议 64）")
    ap.add_argument("--closed-book", action="store_true",
                    help="额外生成一次闭卷（空上下文）作答，输出 elicit 地板值（判据不变）")
    ap.add_argument("--rev-ratios", default=None, help="覆盖撤回比例列表，如 0.05")
    ap.add_argument("--shadow-ratios", default=None, help="覆盖影子比例列表，如 0.1")
    ap.add_argument("--list-baselines", action="store_true", help="列出可用基线后退出")
    args = ap.parse_args(argv)

    if args.list_baselines:
        print("baselines:", ", ".join(available_baselines()))
        return 0

    cfg = load_config(args.config)
    if args.smoke:
        cfg = apply_smoke_overrides(cfg)
    if args.exp:
        cfg["exp"] = args.exp
    if args.dataset:
        cfg["dataset"] = args.dataset
        cfg["datasets"] = None
    if args.limit_queries:
        cfg["n_queries_eval"] = int(args.limit_queries)
    if args.stages:
        cfg["_stages"] = [s for s in str(args.stages).split(",") if s.strip()]
    if args.corpus_sizes:
        cfg["corpus_sizes"] = [int(x) for x in str(args.corpus_sizes).split(",") if x.strip()]
    if args.methods:
        ms = [m.strip() for m in str(args.methods).split(",") if m.strip()]
        method = next((m for m in ms if m == "fedrevoke"), "fedrevoke")
        cfg["method"] = method
        cfg["baselines"] = [m for m in ms if m != "fedrevoke"]
    if args.n_anchors is not None:
        cfg["_n_anchors"] = int(args.n_anchors)
    if args.closed_book:
        cfg["closed_book"] = True
    if args.rev_ratios:
        cfg["revocation_ratios"] = [float(x) for x in str(args.rev_ratios).split(",") if x.strip()]
        cfg["_rev_explicit"] = True
    if args.shadow_ratios:
        cfg["shadow_ratios"] = [float(x) for x in str(args.shadow_ratios).split(",") if x.strip()]
        cfg["_shadow_explicit"] = True
    tier = args.tier or ("b" if args.smoke else "a")
    cfg = apply_tier_overrides(cfg, tier, args.model, smoke=bool(args.smoke))
    summary = run_experiment(cfg, smoke=bool(args.smoke), limit=args.limit,
                             out_dir=args.out_dir, make_figures=not args.no_figures)
    print("[run_experiments] done: rows=%d elapsed=%.2fs results=%s figures=%s" % (
        summary["n_rows"], summary["elapsed_seconds"],
        summary.get("csv", {}).get("all"), summary.get("figures")))
    if summary.get("real_data_missing"):
        print("[run_experiments] 真实数据缺口（只跑 smoke）:")
        for k, v in summary["real_data_missing"].items():
            print("   - %s: %s" % (k, ", ".join(v)))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


