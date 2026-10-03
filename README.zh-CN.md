# FedRevoke（中文说明）

**面向跨孤岛联邦检索增强生成的影子感知检索级遗忘**


## 1. 环境

| 项目 | 本机（论文中的所有数字都出自这台机器） |
|---|---|
| GPU | NVIDIA RTX 5080 Laptop 16GB (sm_120, 15.89 GB 可用) |
| CPU / RAM | Intel Core Ultra 9 275HX (24 核) / 63.5 GB |
| Python | 3.14.7 |
| 关键库 | torch 2.11.0+cu128, transformers 5.18.0, sentence-transformers 6.1.0, faiss-cpu 1.15.1, hnswlib 0.8.0, bitsandbytes 0.50.2 |

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
# 本项目是 src/ 布局，必须让 fedrevoke 可导入：
.venv\Scripts\python.exe -m pip install -e . --no-deps
```

> **sm_120 提醒**：torch 必须是 CUDA 12.8+ 的 wheel
> （--index-url https://download.pytorch.org/whl/cu128），旧 wheel 会报
> "no kernel image is available"。
>
> **hnswlib 提醒**：PyPI 只有 sdist，Windows + Python ≥ 3.12 没有官方轮子。
> 论文里用的办法是取出 conda-forge 的预编译 hnswlib.cp314-win_amd64.pyd
> 放进 site-packages。该步骤的脚本不在本仓库内，若你不需要 HNSW 后端
> （论文所有表格都是 IVF-Flat 后端）可以跳过。

---

## 2. 数据

| 语料 | 论文中的用途 | 本仓库是否附带 |
|---|---|---|
| **DS1** MultiHop-RAG（609 文档 / 11,410 段落 / 2,000 查询） | 遗忘质量；表 I、IV、V；图 1–3 | 已附带 `data/processed/multihoprag/` |
| **DS2** BeIR/nq（100,639 文档 / 126,478 段落 / 1,000 查询） | 规模与代价；表 II–IV | 否，用 `tools/download_raw_data.py ds2` 重建 |
| **DS3** BeIR/trec-covid（166,890 文档 / 336,529 段落 / 500 查询） | 跨域鲁棒性与代价 | 否，用 `tools/download_raw_data.py ds3` 重建 |

`data/processed/multihoprag/` 里就是论文承诺释放的那些文件：段落 id
（`pid_meta.json`）、孤岛划分（`silo_assignments.json`）、注入影子对及其
相似度记录（`shadow_pairs.json`）、forget set（`forget_sets.json`）、
预计算嵌入、以及 MinHash 签名矩阵（`minhash_sig.npy`）。

原始语料见 [data/raw/README.md](data/raw/README.md)（含下载脚本的一处已知路径 bug）。

---

## 3. 跑实验

```powershell
# 冒烟：合成数据 + MockGenerator，60 秒内跑完，不下载任何模型
python -m fedrevoke.run_experiments --config configs/e0_smoke.yaml --smoke

# DS1 主网格（Tier A：纯检索，不加载生成器）
python -m fedrevoke.run_experiments --config configs/e1_main.yaml --tier a

# DS1 主网格 + 生成端（Tier B，需要 GPU）
python -m fedrevoke.run_experiments --config configs/e1_main.yaml --tier b --model 1.5b --closed-book

# DS2 / DS3
python -m fedrevoke.run_experiments --config configs/e5_cross_dataset.yaml

# 代价扫描
python -m fedrevoke.run_experiments --config configs/e4_cost.yaml
```

结果写到 `artifacts/results/<exp>/`（`rows.csv` 全量 + `main.csv` / `e6.csv`
分阶段 CSV + `summary.json`），运行日志在 `artifacts/logs/`。
审计凭证（删除证书哈希链）写在 `artifacts/certificates/`。

全部随机性由 `config.SEED = 20260214` 固定，禁止裸 `np.random`。

---



## 4. 单元测试

```powershell
python -m pytest tests -q      # 140 passed，约 32 秒，纯合成数据，不需要网络和 GPU
```

---

## 5. 目录

```
src/fedrevoke/   data_prep, index_core, shadow, repair, revoke, generation,
                 verify, metrics, baselines, run_experiments
configs/         e0_smoke, e1_main, e1_main_r20full, e2_silos, e3_ablation,
                 e4_cost, e5_cross_dataset
tests/           pytest（仅合成数据，无需网络）
tools/           download_raw_data, encode_queries, resample_forget_sets,
                 make_paper_figures, diag_rho
data/processed/  已释放的 DS1 划分
data/raw/        空目录（用 tools/download_raw_data.py 重建）
artifacts/       results / logs / certificates（图在生成后落到 artifacts/figures/）
```

---

## 7. 复现论文表格前请先看

本仓库 **只附带了一部分实验结果**（DS1 的 E1/E6 网格 + 冒烟跑），
表 II、III、IV、V 与图 2、3 所需的数据不在仓库里。逐项对照如下：

| 论文条目 | 已附带产物 | 状态 |
|---|---|---|
| 表 I（`tab:ds1`）— DS1，`r5` × 3 档影子覆盖 | `artifacts/results/e1_rho_tables/e6.csv` | 可精确复现（0.012821 / 0.027778 / 0.140–0.160，`n` = 78 / 72 / 50） |
| 图 2（E6 动机曲线） | `e1_rho_tables/e6.csv`，`e1_rho_tables/paper_ds1_e6.json` | 仅 DS1；图注写的是三个语料的中位数 |
| 删除证书（M6） | `artifacts/certificates/`（16 条 + 3 条诊断链） | 均可通过 `strict=True` 校验 |
| 表 II（`tab:ds2`）— BeIR/nq | — | 本仓库无 DS2 处理后数据（`e1_rho_tables_run.json` 记录 `ds2: exists=false`） |
| 表 III（`tab:tierb`）— 生成端 | — | 未附带 |
| 表 IV（`tab:cost`） | — | 未附带 `e4_cost` 运行结果 |
| 图 3（代价扩展） | — | `tools/make_paper_figures.py` 找不到 `cost.csv` |
| 表 V（`tab:b5`）— B4 / B5 基线 | — | 已附带结果只含 `fedrevoke`、`full_rebuild`、`naive_delete`、`sisa` |
| §VI-D 的 DS3 数字、§VII 消融 | — | 未附带 |

`artifacts/results/e0_smoke/` 里的冒烟跑只是夹具，不是证据：用的是合成数据和 `MockGenerator`。
