# -*- coding: utf-8 -*-
"""Regenerate Fig.2 (e6 motivation) over the three corpora.

Uses exactly one e6.csv per corpus (DS1, DS2, DS3) — the runs with real query
embeddings — and plots the median residual_doc_rate per shadow coverage.
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
OUT = Path(__file__).resolve().parents[2] / "paper_fcs" / "figures"

# one authoritative run per corpus
SOURCES = {
    "DS1": RES / "rerun_table1" / "e6.csv",
    "DS2": RES / "rerun_table2_ds2_v2" / "e6.csv",
    "DS3": RES / "rerun_ds3" / "e6.csv",
}
STYLE = {
    "fedrevoke": ("#1f4e9c", "o", "-", "FedRevoke (ours)"),
    "naive_delete": ("#2e8b57", "s", "-", "Naive shard deletion"),
    "full_rebuild": ("#c0392b", "^", "-", "Full rebuild"),
    "sisa": ("#8e44ad", "D", "-", "SISA-style shard retrain"),
}
plt.rcParams.update({"font.size": 8, "axes.labelsize": 8, "legend.fontsize": 7,
                     "xtick.labelsize": 7, "ytick.labelsize": 7,
                     "axes.grid": True, "grid.alpha": 0.3, "grid.linewidth": 0.4,
                     "figure.dpi": 300, "savefig.bbox": "tight",
                     "savefig.pad_inches": 0.02})


def main() -> None:
    frames = []
    for name, p in SOURCES.items():
        if not p.exists():
            print(f"[missing] {name}: {p}")
            continue
        df = pd.read_csv(p)
        df["corpus"] = name
        frames.append(df)
        print(f"[ok] {name}: {len(df)} rows, shadow_ratio={sorted(df.shadow_ratio.unique())}")
    if len(frames) < 3:
        print("need all three corpora; aborting")
        return
    df = pd.concat(frames, ignore_index=True)

    print("\nmedian residual_doc_rate over corpora:")
    piv = df.groupby(["shadow_ratio", "method"])["residual_doc_rate"].median().unstack()
    print(piv.to_string())

    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    for method, g in df.groupby("method"):
        color, marker, ls, label = STYLE.get(method, ("#333333", "x", "--", method))
        med = (g.groupby("shadow_ratio")["residual_doc_rate"].median()
                 .reset_index().sort_values("shadow_ratio"))
        ax.plot(med["shadow_ratio"], med["residual_doc_rate"],
                marker=marker, ls=ls, color=color, ms=3.2, lw=1.1, label=label)
    ax.set_xlabel("Shadow coverage $r_s$")
    ax.set_ylabel("Residual leakage (doc-level)")
    ax.set_xticks([0.0, 0.1, 0.3, 1.0])
    ax.legend(frameon=False, loc="upper left")

    FIG.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    for dest in (FIG / "paper_e6_motivation", OUT / "fig2_e6_motivation"):
        fig.savefig(dest.with_suffix(".pdf"))
        fig.savefig(dest.with_suffix(".png"))
        print("wrote", dest.with_suffix(".pdf"))
    plt.close(fig)


if __name__ == "__main__":
    main()
