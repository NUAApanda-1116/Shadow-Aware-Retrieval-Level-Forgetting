# FedRevoke

**Shadow-aware retrieval-level forgetting for cross-silo federated retrieval-augmented generation**


## 1. Environment

| Item | This machine (all numbers in the paper come from this machine) |
|---|---|
| GPU | NVIDIA RTX 5080 Laptop 16GB (sm_120, 15.89 GB available) |
| CPU / RAM | Intel Core Ultra 9 275HX (24 cores) / 63.5 GB |
| Python | 3.14.7 |
| Key libraries | torch 2.11.0+cu128, transformers 5.18.0, sentence-transformers 6.1.0, faiss-cpu 1.15.1, hnswlib 0.8.0, bitsandbytes 0.50.2 |

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
# This project uses a src/ layout; fedrevoke must be importable:
.venv\Scripts\python.exe -m pip install -e . --no-deps
```

> **sm_120 note**: torch must be a CUDA 12.8+ wheel
> (--index-url https://download.pytorch.org/whl/cu128). Older wheels report
> "no kernel image is available".
>
> **hnswlib note**: PyPI ships sdist only; Windows + Python ≥ 3.12 has no
> official wheel. The paper used a prebuilt conda-forge hnswlib.cp314-win_amd64.pyd
> dropped into site-packages. That step's script is not in this repository.
> Skip it if you do not need the HNSW back end (every paper table uses IVF-Flat).

---

## 2. Data

| Corpus | Role in the paper | Shipped here |
|---|---|---|
| **DS1** MultiHop-RAG (609 docs / 11,410 passages / 2,000 queries) | forgetting quality; Tables I, IV, V; Figs. 1–3 | Included: `data/processed/multihoprag/` |
| **DS2** BeIR/nq (100,639 docs / 126,478 passages / 1,000 queries) | scale and cost; Tables II–IV | No — rebuild with `tools/download_raw_data.py ds2` |
| **DS3** BeIR/trec-covid (166,890 docs / 336,529 passages / 500 queries) | out-of-domain robustness and cost | No — rebuild with `tools/download_raw_data.py ds3` |

`data/processed/multihoprag/` contains exactly what the paper says is released:
passage ids (`pid_meta.json`), silo assignments (`silo_assignments.json`),
injected shadow pairs with their similarity records (`shadow_pairs.json`),
forget sets (`forget_sets.json`), pre-computed embeddings, and the MinHash
signature matrix (`minhash_sig.npy`).

For the raw corpora see [data/raw/README.md](data/raw/README.md) (includes one
known path bug in the downloader).

---

## 3. Running experiments

```powershell
# smoke: synthetic data + MockGenerator, under 60 s, no model downloads
python -m fedrevoke.run_experiments --config configs/e0_smoke.yaml --smoke

# DS1 main grid (Tier A: retrieval only, no generator)
python -m fedrevoke.run_experiments --config configs/e1_main.yaml --tier a

# DS1 main grid + generation (Tier B, needs a GPU)
python -m fedrevoke.run_experiments --config configs/e1_main.yaml --tier b --model 1.5b --closed-book

# DS2 / DS3
python -m fedrevoke.run_experiments --config configs/e5_cross_dataset.yaml

# cost sweep
python -m fedrevoke.run_experiments --config configs/e4_cost.yaml
```

Results land in `artifacts/results/<exp>/` (`rows.csv` full dump + staged CSVs
`main.csv` / `e6.csv` + `summary.json`); run logs are in `artifacts/logs/`.
Audit credentials (deletion-certificate hash chain) are written to
`artifacts/certificates/`.

All randomness is fixed by `config.SEED = 20260214`; bare `np.random` is forbidden.

---



## 4. Unit tests

```powershell
python -m pytest tests -q      # 140 passed, ~32 s, synthetic data only, no network or GPU
```

---

## 5. Layout

```
src/fedrevoke/   data_prep, index_core, shadow, repair, revoke, generation,
                 verify, metrics, baselines, run_experiments
configs/         e0_smoke, e1_main, e1_main_r20full, e2_silos, e3_ablation,
                 e4_cost, e5_cross_dataset
tests/           pytest (synthetic data only, no network)
tools/           download_raw_data, encode_queries, resample_forget_sets,
                 make_paper_figures, diag_rho
data/processed/  released DS1 splits
data/raw/        empty (rebuild with tools/download_raw_data.py)
artifacts/       results / logs / certificates (figures land in artifacts/figures/)
```

---

## 7. Read this before reproducing paper tables

This repository **ships only part of the experimental results** (DS1 E1/E6 grid
+ smoke run). Data needed for Tables II–III–IV–V and Figs. 2–3 is not in the
repository. Item-by-item status:

| Paper item | Shipped artefact | Status |
|---|---|---|
| Table I (`tab:ds1`) — DS1, `r5` × 3 shadow coverages | `artifacts/results/e1_rho_tables/e6.csv` | Reproduces exactly (0.012821 / 0.027778 / 0.140–0.160, `n` = 78 / 72 / 50) |
| Fig. 2 (E6 motivation curve) | `e1_rho_tables/e6.csv`, `e1_rho_tables/paper_ds1_e6.json` | DS1 only; the caption refers to a median over three corpora |
| Deletion certificates (M6) | `artifacts/certificates/` (16 + 3 diagnostic chains) | All verify with `strict=True` |
| Table II (`tab:ds2`) — BeIR/nq | — | No processed DS2 data in this repo (`e1_rho_tables_run.json` records `ds2: exists=false`) |
| Table III (`tab:tierb`) — generation tier | — | Not shipped |
| Table IV (`tab:cost`) | — | No `e4_cost` run shipped |
| Fig. 3 (cost scaling) | — | `tools/make_paper_figures.py` finds no `cost.csv` |
| Table V (`tab:b5`) — B4 / B5 baselines | — | Shipped cells carry only `fedrevoke`, `full_rebuild`, `naive_delete`, `sisa` |
| §VI-D DS3 numbers, §VII ablations | — | Not shipped |

The smoke run under `artifacts/results/e0_smoke/` is a fixture, not evidence:
it uses synthetic data and a `MockGenerator`.
