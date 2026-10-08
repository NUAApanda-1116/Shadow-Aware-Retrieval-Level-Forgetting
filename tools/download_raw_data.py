"""Download raw public corpora for FedRevoke data_prep."""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

RAW = Path(__file__).resolve().parents[1] / "data" / "raw"


def ensure_multihoprag() -> None:
    out = RAW / "multihoprag"
    out.mkdir(parents=True, exist_ok=True)
    from huggingface_hub import hf_hub_download

    for fn in ("corpus.json", "MultiHopRAG.json"):
        dest = out / fn
        if dest.exists() and dest.stat().st_size > 0:
            print(f"exists {dest} ({dest.stat().st_size})")
            continue
        p = hf_hub_download(
            repo_id="yixuantt/MultiHopRAG", filename=fn, repo_type="dataset"
        )
        shutil.copy2(p, dest)
        print(f"copied {fn} -> {dest} ({dest.stat().st_size})")


def ensure_beir(name: str, files: tuple[str, ...]) -> None:
    out = RAW / name
    out.mkdir(parents=True, exist_ok=True)
    from huggingface_hub import hf_hub_download

    repo = "BeIR/nq" if name == "nq" else "BeIR/trec-covid"
    for fn in files:
        dest = out / fn
        if dest.exists() and dest.stat().st_size > 0:
            print(f"exists {dest} ({dest.stat().st_size})")
            continue
        # BeIR keeps qrels under qrels/test.tsv in the HF repo; we flatten it
        # to qrels_test.tsv so it matches src/fedrevoke/data_prep.py.
        src_fn = "qrels/test.tsv" if fn == "qrels_test.tsv" else fn
        p = hf_hub_download(repo_id=repo, filename=src_fn, repo_type="dataset")
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, dest)
        print(f"copied {src_fn} -> {dest} ({dest.stat().st_size})")


def main() -> int:
    which = sys.argv[1] if len(sys.argv) > 1 else "ds1"
    if which in ("ds1", "all", "multihoprag"):
        ensure_multihoprag()
    if which in ("ds2", "all", "nq"):
        ensure_beir("nq", ("corpus.parquet", "queries.parquet", "qrels_test.tsv"))
    if which in ("ds3", "all", "trec-covid"):
        ensure_beir(
            "trec-covid",
            ("corpus.parquet", "queries.parquet", "qrels_test.tsv"),
        )
    print("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
