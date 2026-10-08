"""Encoding-throughput benchmark: remeasure full-corpus encoding time (paper Sec. 5.4).

Usage (from the repository root):

    python tools/encode_benchmark.py

Reports the whole-corpus encoding seconds and text/s per dataset at batch size
256, fp16, CUDA; one warm-up batch is excluded from the timing. These values
feed REENCODE_SECONDS_BY_DATASET in fedrevoke.baselines.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fedrevoke.data_prep import Encoder  # noqa: E402


def main() -> None:
    out = {}
    for ds in ["multihoprag", "nq", "trec-covid"]:
        texts = []
        with open(ROOT / "data" / "processed" / ds / "corpus.jsonl", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    texts.append(json.loads(line)["text"])
        enc = Encoder(batch_size=256, max_length=256, use_fp16=True)
        enc.encode(texts[:256], batch_size=256)   # warm-up
        enc._cache.clear()
        if enc.device == "cuda":
            enc.torch.cuda.synchronize()
        t0 = time.perf_counter()
        enc.encode(texts, batch_size=256)
        if enc.device == "cuda":
            enc.torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        out[ds] = {"n_texts": len(texts), "seconds": round(dt, 3),
                   "rate_per_s": round(len(texts) / dt, 1)}
        print(f"{ds}: {len(texts)} texts in {dt:.3f}s -> {len(texts)/dt:.1f} text/s", flush=True)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
