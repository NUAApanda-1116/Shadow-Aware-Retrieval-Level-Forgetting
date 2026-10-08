r"""FedRevoke paper-grade figure generator (independent of run_experiments).

Usage: .venv\Scripts\python.exe tools\make_paper_figures.py
Output: artifacts/figures/paper_cost_scaling.{pdf,png}  paper_e6_motivation.{pdf,png}

Design notes:
  * The cost figure reports Claim 3 under three measures: write amplification (bytes) / wall-clock (cached embeddings) / wall-clock (including re-encoding, converted using measured encoding throughput).
  * One independent curve per method per figure (methods are no longer averaged into a single line).
  * Two-column journal layout: width 7.16in, font size 8, 300dpi, vector PDF.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "artifacts" / "results"
FIG = ROOT / "artifacts" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

# Measured encoding throughput (used to convert "rebuild including re-encoding")
ENC = {"ds1": (11410, 5.23), "ds2": (156670, 42.97), "ds3": (386596, 151.60)}
STYLE = {
    "fedrevoke": ("#1f4e9c", "o", "-", "FedRevoke (ours)"),
    "naive_delete": ("#2e8b57", "s", "-", "Naive shard deletion"),
    "full_rebuild": ("#c0392b", "^", "-", "Full rebuild (cached embeddings)"),
    "sisa": ("#8e44ad", "D", "-", "SISA-style shard retrain"),
    "tdsc_adapter": ("#e67e22", "v", "-", "TDSC adapter"),
    "lora_finetune": ("#7f8c8d", "P", "-", "LoRA fine-tuning"),
}
plt.rcParams.update({"font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
                     "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7,
                     "axes.grid": True, "grid.alpha": 0.3, "grid.linewidth": 0.4,
                     "figure.dpi": 300, "savefig.bbox": "tight", "savefig.pad_inches": 0.02})


def _corpus_size(g: pd.DataFrame) -> float:
    if "corpus_size" in g.columns and g["corpus_size"].notna().any():
        return float(g["corpus_size"].dropna().iloc[0])
    fr = g[g["method"] == "full_rebuild"]
    if len(fr) and fr["n_alive_after"].notna().any():
        return float(fr["n_alive_after"].dropna().iloc[0] + fr["n_deleted"].dropna().iloc[0])
    return float("nan")


def load_cost_points() -> pd.DataFrame:
    """Cost rows inside each (exp, dataset) are blocked in file order (each block = the methods at the same corpus size);
    the size is inferred from n_alive_after + n_deleted of the full_rebuild/sisa rows inside the block (the cost CSV has no corpus_size column)."""
    frames = []
    for csv in list(RES.glob("*/cost.csv")) + list(RES.glob("e*_cost.csv")):
        df = pd.read_csv(csv)
        if "method" in df.columns and len(df):
            df["exp"] = csv.parent.name if csv.parent != RES else csv.stem
            frames.append(df)
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    rows = []
    for (exp, ds), g in df.groupby(["exp", "dataset"], sort=False):
        g = g.reset_index(drop=True)
        sizes = []
        for _, r in g.iterrows():
            if pd.notna(r.get("n_alive_after")) and pd.notna(r.get("n_deleted")):
                s = float(r["n_alive_after"]) + float(r["n_deleted"])
                if s not in sizes:
                    sizes.append(s)
        if not sizes:
            continue
        sizes = sorted(sizes)
        n_blk = len(sizes)
        blk = max(1, len(g) // n_blk)
        for i, r in g.iterrows():
            size = sizes[min(i // blk, n_blk - 1)]
            rows.append({"exp": exp, "dataset": ds, "method": r["method"], "corpus": size,
                         "bytes": r["bytes_transferred"], "sec": r["reindex_seconds"],
                         "touched": r["n_vectors_touched"]})
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return (out.groupby(["exp", "dataset", "method", "corpus"], as_index=False)
               .agg(bytes=("bytes", "median"), sec=("sec", "median"), touched=("touched", "median"))
               .sort_values(["exp", "corpus"]))


def fig_cost(df: pd.DataFrame) -> None:
    if df.empty:
        print("cost: no data"); return
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.75))
    for ax, ycol, ylab in ((axes[0], "bytes", "Bytes written (log)"),
                           (axes[1], "sec", "Re-index wall clock [s] (log)")):
        for method, g in df.groupby("method"):
            color, marker, ls, label = STYLE.get(method, ("#333333", "x", "--", method))
            g = g.sort_values("corpus")
            ax.plot(g["corpus"], g[ycol], marker=marker, ls=ls, color=color, ms=3.2, lw=1.1, label=label)
        if ycol == "sec":
            fr = df[df["method"] == "full_rebuild"]
            for ds, g in fr.groupby("dataset"):
                if ds not in ENC: continue
                total, enc_s = ENC[ds]
                x = g.sort_values("corpus")
                y = x["sec"].values + enc_s * (x["corpus"].values / total)
                ax.plot(x["corpus"], y, ls=":", color="#c0392b", lw=1.1,
                        label=f"Full rebuild incl. re-encoding ({ds.upper()})")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("Corpus size (passages, log)")
        ax.set_ylabel(ylab)
    axes[0].legend(frameon=False, loc="upper left")
    axes[1].legend(frameon=False, loc="upper left")
    for ax, t in zip(axes, ["(a) Write amplification", "(b) Wall clock"]):
        ax.set_title(t, loc="left", fontsize=8, fontweight="bold")
    out = FIG / "paper_cost_scaling"
    fig.savefig(out.with_suffix(".pdf")); fig.savefig(out.with_suffix(".png"))
    plt.close(fig)
    print("wrote", out.with_suffix(".pdf").name)


def fig_e6() -> None:
    frames = []
    for csv in RES.glob("*/e6.csv"):
        df = pd.read_csv(csv)
        if "residual_doc_rate" in df.columns and len(df):
            df["exp"] = csv.parent.name; frames.append(df)
    if not frames:
        print("e6: no data"); return
    df = pd.concat(frames, ignore_index=True)
    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    for method, g in df.groupby("method"):
        color, marker, ls, label = STYLE.get(method, ("#333333", "x", "--", method))
        g = g.groupby("shadow_ratio")["residual_doc_rate"].median().reset_index().sort_values("shadow_ratio")
        ax.plot(g["shadow_ratio"], g["residual_doc_rate"], marker=marker, ls=ls,
                color=color, ms=3.2, lw=1.1, label=label)
    ax.set_xlabel("Shadow coverage $r_s$")
    ax.set_ylabel("Residual leakage (doc-level)")
    ax.legend(frameon=False, loc="upper left")
    out = FIG / "paper_e6_motivation"
    fig.savefig(out.with_suffix(".pdf")); fig.savefig(out.with_suffix(".png"))
    plt.close(fig)
    print("wrote", out.with_suffix(".pdf").name)


if __name__ == "__main__":
    df = load_cost_points()
    print(df.to_string() if len(df) else "no cost rows")
    fig_cost(df)
    fig_e6()
