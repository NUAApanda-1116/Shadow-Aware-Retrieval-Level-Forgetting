# INTERFACES.md — module contracts (single source of truth for APIs)

## §0 Environment (read first)

- Project root: this repository root; all paths are relative to the repository root
- Install: `python -m pip install -e . --no-deps` (or `PYTHONPATH=src`)
- Virtualenv: `.venv` (`python -m venv .venv`)
- If Python 3.14 lacks wheels: use `uv` (`pip install uv`) with 3.12 — `uv venv --python 3.12 .venv312`
- Run tests: `python -m pytest tests -q`
- All randomness must use `config.SEED = 20260214`; bare `np.random` is forbidden.

## §1 config.py

~~~python
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
DATA, ARTIFACTS = ROOT/"data", ROOT/"artifacts"
SEED = 20260214
ENCODER_ID = "BAAI/bge-small-en-v1.5"
DIM = 384
GEN_ID = "Qwen/Qwen2.5-7B-Instruct"
GEN_ID_SMALL = "Qwen/Qwen2.5-1.5B-Instruct"
DATASETS = {"ds1": "multihoprag", "ds2": "nq", "ds3": "trec-covid"}
~~~

## §2 index_core.py

~~~python
@dataclass(frozen=True)
class VecMeta:
    pid: int; doc_id: str; client_id: str; topic: str; fingerprint: str

class ProvenanceIndex:
    def __init__(self, dim: int, backend: str = "faiss_ivf", **kw) -> None: ...
    def add(self, vectors, metas: list[VecMeta]) -> list[int]: ...        # returns internal ids
    def remove(self, internal_ids: Sequence[int]) -> None: ...            # idempotent
    def search(self, queries, k: int = 10) -> tuple[np.ndarray, np.ndarray]: ...  # (scores, internal_ids)
    def meta(self, internal_id: int) -> VecMeta: ...
    def ids_for_doc(self, doc_id: str) -> list[int]: ...
    def ids_for_client(self, client_id: str) -> list[int]: ...
    def alive_ids(self) -> np.ndarray: ...
    def save(self, path: str) -> None: ...
    @classmethod
    def load(cls, path: str) -> "ProvenanceIndex": ...
    def stats(self) -> dict: ...   # {"n_vectors","n_deleted","backend","memory_bytes"}
~~~

Constraints: `search` must not return deleted ids; after `remove`, `stats()["n_deleted"]` must stay in sync.

## §3 shadow.py

~~~python
class ShadowDetector:
    def __init__(self, sim_threshold: float = 0.92, lsh_threshold: float = 0.80,
                 knn_k: int = 50, cross_client_only: bool = True, seed: int = 20260214): ...
    def closure(self, index: ProvenanceIndex, seed_ids: Sequence[int]) -> set[int]: ...
    def shadow_report(self, index, seed_ids) -> dict: ...
    # {"n_seed","n_closure","per_client":{...},"sim_hist":[...],"precision":float,"recall":float}
~~~

Self-check: in `tests/test_shadow.py`, use the controlled injection set (`shadow_pairs.json`) and verify recall > 0.95.

## §4 repair.py

~~~python
class AnchorRepair:
    def __init__(self, n_anchors: int = 512, recalibrate: bool = True, seed: int = 20260214): ...
    def repair(self, index: ProvenanceIndex, removed_ids: Sequence[int],
               query_sample: np.ndarray) -> dict: ...
    # {"n_reconnected":int,"score_shift":float,"before_recall":float,"after_recall":float}
~~~

## §5 generation.py

~~~python
class Generator(Protocol):
    def generate(self, prompts: list[str], max_new_tokens: int = 200) -> list[str]: ...

class MockGenerator:      # deterministic stub for unit tests; no network
    def __init__(self, keyword: str = "MOCK"): ...
class HFGenerator:
    def __init__(self, model_id: str = GEN_ID, load_in_4bit: bool = True,
                 device: str = "cuda", max_batch: int = 4): ...
class GGUFGenerator:      # optional back end
    def __init__(self, repo_id: str, filename: str): ...
~~~

## §6 verify.py

~~~python
@dataclass
class ForgetReport:
    hit_rate: float; mia_auc: float; elicit_rate: float; rho_hat: float; n_probes: int

class MIAProbe:
    def __init__(self, index, encoder, seed: int = 20260214): ...
    def auc(self, forgotten_pids: Sequence[int], control_pids: Sequence[int]) -> float: ...

class ElicitationTest:
    def __init__(self, generator: Generator, index, k: int = 5, max_new_tokens: int = 200): ...
    def run(self, forgotten_qa: list[dict]) -> float: ...   # elicit_rate
~~~

## §7 metrics.py

~~~python
def recall_at_k(retrieved_pids, gold_pids, k=10) -> float
def ndcg_at_k(retrieved_pids, gold_pids, k=10) -> float
def exact_match(pred: str, golds: list[str]) -> float
def f1_score_tokens(pred: str, golds: list[str]) -> float
def mia_auc(pos_scores, neg_scores) -> float
class CostMeter:  # context manager: wall_time, peak_vram_mb, bytes_transferred
    def __enter__(self); def __exit__(self, *a); def as_dict(self) -> dict
~~~

## §8 revoke.py

~~~python
@dataclass
class RevocationResult:
    deleted_ids: list[int]; report: ForgetReport; cost: dict; certificate_path: str

class RevocationPipeline:
    def __init__(self, index, detector, repair, verifier, cert_dir: Path, encoder=None): ...
    def revoke(self, doc_ids: Sequence[str], query_sample=None) -> RevocationResult: ...
    def certificate(self) -> dict:   # {"request_id","timestamp","n_deleted","chain_sha256"}
~~~

## §9 baselines.py

~~~python
class Baseline(Protocol):
    name: str
    def run(self, index, doc_ids, **kw) -> dict: ...   # always returns {"forget":..., "utility":..., "cost":...}
class FullRebuild, NaiveDelete, SISA, LoRAFinetune, TDSCAdapter  # all implement Baseline
~~~

## §10 run_experiments.py

CLI: `python -m fedrevoke.run_experiments --config configs/e1_main.yaml [--smoke]`
- Each experiment writes `artifacts/results/<exp>.csv` and `artifacts/figures/<exp>_*.pdf`
- `--smoke` mode: MockGenerator + synthetic index + 50 queries, **finishes in under 60 s**

---

## §11 Appendix: fingerprint / MinHash channel ruling (overrides the default implementation)

**Problem**: in `data_prep.py`, `corpus.jsonl` stores `fingerprint = blake2b(signature, digest_size=8)` (16 hex chars).
That is a further lossy digest of the 64-permutation signature, so it **cannot estimate MinHash Jaccard**, and
`ShadowDetector`'s LSH text channel degrades to exact matching.

**Ruling: route (a) — do not change the on-disk data format; write the signature matrix separately and inject it explicitly in the pipeline.**

1. `data_prep.py` must write `minhash_sig.npy` (uint64, shape `(n_chunks, num_perm)`) and `minhash_meta.json`.
   Row order = `corpus.jsonl` row order = ascending `pid` (already implemented).
2. The `fingerprint` field in `corpus.jsonl` remains a 64-bit digest. Its uses are **limited** to:
   exact-duplicate detection, provenance display, and the public fingerprint in audit certificates.
   It is **forbidden** for LSH / near-duplicate Jaccard estimation.
3. Any experiment that enables shadow closure must construct
   `ShadowDetector(signatures=<signature matrix>)`, with source priority:
   (i) `data_prep.load_bundle(ds)["signatures"]` when that key exists; (ii) otherwise `np.load(ds_dir/minhash_sig.npy)`.
4. **Mandatory assertion (fail-fast; silent degradation is forbidden)**: after building the index, call
   `detector.diagnose_fingerprints(index)`. If `fingerprint_mode` reports digest counts > 0 and the
   experiment has `shadow_ratio > 0`: raise `RuntimeError` in full mode (with a hint); in `--smoke`
   mode print an explicit WARNING and continue.
5. Semantic contract (implemented and tested): `closure()` returns a set that **includes the seeds**;
   `cross_client_only` is a **per-hop** constraint (the second hop may return to the original silo;
   cascade takes priority); `search()` always returns shape `(nq, k)`, padding empty slots with
   `PAD_ID=-1` and `score=-inf`. Extra APIs available: `vector(i)` / `vectors_for(ids)` /
   `alive_matrix()` / `is_alive(i)` / `deleted_ids()` / `all_ids_for_doc()` / `compact()` /
   `maybe_compact()` / `n_alive` / `n_vectors`.
6. Confirmed impact: E1 (shadow 10/30%), E4 ablations (−shadow closure, LSH threshold sensitivity),
   E7 sensitivity analysis.
