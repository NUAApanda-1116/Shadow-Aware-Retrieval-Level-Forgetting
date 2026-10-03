"""FedRevoke global configuration (single entry point).

Strictly follows INTERFACES.md §1. If you need to add constants, append them;
do not modify the frozen names and values.
"""

from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------- Paths (frozen)
ROOT = Path(__file__).resolve().parents[2]
DATA, ARTIFACTS = ROOT / "data", ROOT / "artifacts"

# ---------------------------------------------------------------- Constants (frozen)
SEED = 20260214
ENCODER_ID = "BAAI/bge-small-en-v1.5"
DIM = 384
GEN_ID = "Qwen/Qwen2.5-7B-Instruct"
GEN_ID_SMALL = "Qwen/Qwen2.5-1.5B-Instruct"
DATASETS = {"ds1": "multihoprag", "ds2": "nq", "ds3": "trec-covid"}

# ------------------------------------------------- Additional constants (non-frozen contract)
RAW = DATA / "raw"
PROCESSED = DATA / "processed"
LOGS = ARTIFACTS / "logs"
REPORTS = ARTIFACTS / "reports"
RESULTS = ARTIFACTS / "results"
FIGURES = ARTIFACTS / "figures"

# Data pipeline default hyperparameters
CHUNK_WORDS = 120          # about 100-200 words/chunk
CHUNK_OVERLAP = 0.20       # 20% overlap
MINHASH_PERM = 64          # 64-bit MinHash fingerprint
SILO_SIZES = (5, 10)       # two silo-count settings
DIRICHLET_ALPHA = 0.5      # non-IID skew strength
FORGET_RATIOS = {"r1": 0.01, "r5": 0.05, "r20": 0.20}
SHADOW_RATIOS = {"r10": 0.10, "r30": 0.30}
SHADOW_SIM_FLOOR = 0.90    # minimum cosine similarity for injecting shadow copies (target ~0.94)

EMB_FILENAME = "embeddings_bge-small-en-v1.5.npy"
EMB_PIDS_FILENAME = "embedding_pids.json"

__all__ = [
    "ROOT", "DATA", "ARTIFACTS", "SEED", "ENCODER_ID", "DIM",
    "GEN_ID", "GEN_ID_SMALL", "DATASETS",
    "RAW", "PROCESSED", "LOGS", "REPORTS", "RESULTS", "FIGURES",
    "CHUNK_WORDS", "CHUNK_OVERLAP", "MINHASH_PERM", "SILO_SIZES",
    "DIRICHLET_ALPHA", "FORGET_RATIOS", "SHADOW_RATIOS", "SHADOW_SIM_FLOOR",
    "EMB_FILENAME", "EMB_PIDS_FILENAME",
]
