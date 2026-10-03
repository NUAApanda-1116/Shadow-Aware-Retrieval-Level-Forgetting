r"""FedRevoke data pipeline.

Pipeline: raw download -> chunking -> MinHash fingerprints -> silo split (Dirichlet)
-> forget sets -> shadow near-duplicate injection -> bge-small encoding to disk.

Usage (from the fedrevoke/ directory):
    .venv\Scripts\python.exe -m fedrevoke.data_prep --dataset all
    .venv\Scripts\python.exe -m fedrevoke.data_prep --dataset ds1 --smoke
    .venv\Scripts\python.exe -m fedrevoke.data_prep --dataset ds2 --embed-only

Outputs (data/processed/<name>/):
    corpus.jsonl / queries.jsonl / forget_sets.json / shadow_pairs.json
    embeddings_bge-small-en-v1.5.npy / embedding_pids.json
    plus sidecars: silo_assignments.json / pid_meta.json / qid_meta.json
                   minhash_sig.npy / minhash_meta.json / embedding_meta.json
                   dataset_stats.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

try:  # allow running "python data_prep.py" directly
    from . import config as C
except ImportError:  # pragma: no cover
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from fedrevoke import config as C


# =====================================================================
# 0. General utilities
# =====================================================================

def log(msg: str) -> None:
    print("[data_prep] " + msg, flush=True)


def seeded_rng(*parts: Any) -> np.random.Generator:
    """Deterministic RNG derived from strings/integers (bare np.random is forbidden)."""
    key = "|".join(str(p) for p in parts).encode("utf-8")
    digest = hashlib.blake2b(key, digest_size=8).digest()
    return np.random.default_rng((int.from_bytes(digest, "little") ^ C.SEED) & 0xFFFFFFFFFFFFFFFF)


def dump_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        json.dump(obj, fh, ensure_ascii=False, indent=1)
    tmp.replace(path)


def dump_jsonl(path: Path, rows: Iterable[dict]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    n = 0
    with tmp.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False))
            fh.write("\n")
            n += 1
    tmp.replace(path)
    return n


def load_jsonl(path: Path) -> list[dict]:
    with Path(path).open("r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


# =====================================================================
# 1. Raw data loading
# =====================================================================

@dataclass
class RawDoc:
    raw_id: str
    text: str
    title: str = ""
    topic: str = "unknown"
    url: str | None = None
    extra: dict = field(default_factory=dict)


@dataclass
class RawQuery:
    raw_id: str
    query: str
    answers: list[str] = field(default_factory=list)
    gold_refs: list[str] = field(default_factory=list)
    meta: dict = field(default_factory=dict)


def _pick(cols: Sequence[str], candidates: Sequence[str]) -> str | None:
    for c in candidates:
        if c in cols:
            return c
    return None


def read_parquet_rows(path: Path, columns: Sequence[str] | None = None,
                      keep: Any = None, batch_size: int = 50000) -> list[dict]:
    """Stream-read parquet; keep(row_index, row_dict) -> bool decides whether to keep a row."""
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(str(path))
    cols = list(columns) if columns else None
    out: list[dict] = []
    idx = 0
    for batch in pf.iter_batches(batch_size=batch_size, columns=cols):
        d = batch.to_pydict()
        n = len(next(iter(d.values()))) if d else 0
        for i in range(n):
            row = {k: v[i] for k, v in d.items()}
            if keep is None or keep(idx, row):
                out.append(row)
            idx += 1
    return out


def load_qrels(path: Path) -> dict[str, list[str]]:
    """BeIR qrels tsv -> {query_id: [corpus_id, ...]} (score>0)."""
    qrels: dict[str, list[str]] = defaultdict(list)
    with Path(path).open("r", encoding="utf-8") as fh:
        header = fh.readline()
        if "query-id" not in header and "query_id" not in header:
            fh.seek(0)
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            qid, cid, score = parts[0], parts[1], parts[2]
            try:
                if float(score) <= 0:
                    continue
            except ValueError:
                continue
            qrels[qid].append(cid)
    return dict(qrels)


# ---------------------------------------------------------------- DS1

def load_ds1_raw(max_queries: int | None = None) -> tuple[list[RawDoc], list[RawQuery], dict]:
    base = C.RAW / "multihoprag"
    corpus = json.loads((base / "corpus.json").read_text(encoding="utf-8"))
    queries = json.loads((base / "MultiHopRAG.json").read_text(encoding="utf-8"))

    docs: list[RawDoc] = []
    for i, a in enumerate(corpus):
        body = (a.get("body") or "").strip()
        if not body:
            continue
        docs.append(RawDoc(
            raw_id="mhr-%06d" % i,
            text=body,
            title=(a.get("title") or "").strip(),
            topic=((a.get("category") or "unknown").strip().lower() or "unknown"),
            url=a.get("url"),
            extra={"source": a.get("source"), "published_at": a.get("published_at"),
                   "author": a.get("author")},
        ))

    raw_qs: list[RawQuery] = []
    for i, q in enumerate(queries):
        ev = q.get("evidence_list") or []
        refs = [e.get("url") or e.get("title") for e in ev if (e.get("url") or e.get("title"))]
        raw_qs.append(RawQuery(
            raw_id="mhrq-%05d" % i,
            query=(q.get("query") or "").strip(),
            answers=[q["answer"]] if q.get("answer") else [],
            gold_refs=refs,
            meta={"question_type": q.get("question_type"),
                  "evidence_titles": [e.get("title") for e in ev],
                  "evidence_urls": [e.get("url") for e in ev]},
        ))

    info = {"source": "yixuantt/MultiHopRAG (corpus.json + MultiHopRAG.json)",
            "n_raw_docs": len(docs), "n_raw_queries": len(raw_qs)}
    return docs, raw_qs, info


# ---------------------------------------------------------------- DS2 / DS3 (BeIR)

def _beir_load(name: str, raw_dir: Path, *, max_corpus: int | None = None,
               gold_ids: set[str] | None = None, max_queries: int | None = None
               ) -> tuple[list[RawDoc], list[RawQuery], dict]:
    import pyarrow.parquet as pq

    cpath, qpath = raw_dir / "corpus.parquet", raw_dir / "queries.parquet"
    schema_cols = pq.ParquetFile(str(cpath)).schema_arrow.names
    id_col = _pick(schema_cols, ["_id", "id", "doc_id"]) or "_id"
    txt_col = _pick(schema_cols, ["text", "body", "passage"]) or "text"
    ttl_col = _pick(schema_cols, ["title"]) or "title"

    gold_ids = gold_ids or set()
    keep_cols = [c for c in [id_col, txt_col, ttl_col, "metadata"] if c in schema_cols]

    def keep(idx: int, row: dict) -> bool:
        if row.get(id_col) in gold_ids:
            return True  # ensure passages hit by qrels are always in the subset
        return max_corpus is None or idx < max_corpus

    rows = read_parquet_rows(cpath, columns=keep_cols, keep=keep)
    docs: list[RawDoc] = []
    for i, r in enumerate(rows):
        text = (r.get(txt_col) or "").strip()
        title = (r.get(ttl_col) or "").strip()
        if not text and not title:
            continue
        meta = r.get("metadata") or {}
        topic = "unknown"
        if isinstance(meta, dict):
            for key in ("category", "journal", "source", "venue"):
                if meta.get(key):
                    topic = str(meta[key]).strip().lower()
                    break
        docs.append(RawDoc(raw_id=str(r.get(id_col)), text=text, title=title,
                           topic=topic, extra={"metadata": meta if isinstance(meta, dict) else {}}))

    qschema = pq.ParquetFile(str(qpath)).schema_arrow.names
    qid_col = _pick(qschema, ["_id", "id", "query_id"]) or "_id"
    qtxt_col = _pick(qschema, ["text", "query"]) or "text"
    qrows = read_parquet_rows(qpath, columns=[c for c in [qid_col, qtxt_col] if c in qschema])
    qs: list[RawQuery] = []
    for r in qrows:
        qtext = (r.get(qtxt_col) or "").strip()
        if not qtext:
            continue
        qs.append(RawQuery(raw_id=str(r.get(qid_col)), query=qtext))
    if max_queries:
        qs = qs[:max_queries]
    info = {"source": "BeIR/" + name, "n_raw_docs": len(docs), "n_raw_queries": len(qs),
            "corpus_sampled_rows": len(rows)}
    return docs, qs, info


def load_ds2_raw(max_corpus: int | None, max_queries: int | None):
    qrels = load_qrels(C.RAW / "nq" / "qrels_test.tsv")
    gold = {cid for v in qrels.values() for cid in v}
    docs, qs, info = _beir_load("nq", C.RAW / "nq", max_corpus=max_corpus,
                                gold_ids=gold, max_queries=max_queries)
    for q in qs:
        q.gold_refs = list(qrels.get(q.raw_id, []))
        q.meta["has_qrels"] = q.raw_id in qrels
    info["n_qrels_queries"] = len(qrels)
    info["n_gold_ids_total"] = len(gold)
    return docs, qs, info


def load_ds3_raw(max_queries: int | None):
    qrels = load_qrels(C.RAW / "trec-covid" / "qrels_test.tsv")
    gold = {cid for v in qrels.values() for cid in v}
    docs, qs, info = _beir_load("trec-covid", C.RAW / "trec-covid", max_corpus=None,
                                gold_ids=gold, max_queries=max_queries)
    for q in qs:
        q.gold_refs = list(qrels.get(q.raw_id, []))
        q.meta["has_qrels"] = q.raw_id in qrels
        q.meta["query_source"] = "official"
    info["n_qrels_queries"] = len(qrels)
    info["n_gold_ids_total"] = len(gold)
    return docs, qs, info


# =====================================================================
# 2. Chunking
# =====================================================================

def chunk_text(text: str, chunk_words: int = C.CHUNK_WORDS,
               overlap: float = C.CHUNK_OVERLAP, min_words: int = 25) -> list[str]:
    """Split into chunks by whitespace words with 20% overlap; merge a too-short tail into the previous chunk."""
    words = re.sub(r"\s+", " ", text).strip().split()
    if not words:
        return []
    stride = max(1, int(round(chunk_words * (1.0 - overlap))))
    out: list[str] = []
    for start in range(0, len(words), stride):
        piece = words[start:start + chunk_words]
        if not piece:
            break
        if len(piece) < min_words and out:
            out[-1] = out[-1] + " " + " ".join(piece)
            break
        out.append(" ".join(piece))
        if start + chunk_words >= len(words):
            break
    return out


# =====================================================================
# 3. MinHash fingerprints (64-bit / 64 permutations)
# =====================================================================

class MinHasher:
    """64-permutation min-wise independent hash signature; fingerprint = 64-bit blake2b digest of the signature.

    backend="fast": numpy-vectorized (default; minutes for 360k chunks)
    backend="datasketch": official implementation (for small datasets / cross-validation)
    """

    def __init__(self, num_perm: int = C.MINHASH_PERM, seed: int = C.SEED,
                 backend: str = "fast") -> None:
        self.num_perm = int(num_perm)
        self.seed = int(seed)
        self.backend = backend
        rng = np.random.default_rng(self.seed)
        self._a = (rng.integers(1, 2 ** 63, size=self.num_perm, dtype=np.uint64) * 2 + 1)
        self._b = rng.integers(0, 2 ** 63, size=self.num_perm, dtype=np.uint64)
        self._tok_cache: dict[str, int] = {}

    def token_hash(self, token: str) -> int:
        h = self._tok_cache.get(token)
        if h is None:
            h = int.from_bytes(hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest(), "little")
            if len(self._tok_cache) < 5_000_000:
                self._tok_cache[token] = h
        return h

    def signature(self, text: str) -> np.ndarray:
        toks = {t for t in re.findall(r"[a-z0-9]+", text.lower()) if t}
        if not toks:
            return np.zeros(self.num_perm, dtype=np.uint64)
        hv = np.fromiter((self.token_hash(t) for t in toks), dtype=np.uint64, count=len(toks))
        vals = self._a[:, None] * hv[None, :] + self._b[:, None]  # uint64 auto-wraps modulo 2**64
        return vals.min(axis=1)

    @staticmethod
    def fingerprint(sig: np.ndarray) -> str:
        return hashlib.blake2b(np.ascontiguousarray(sig, dtype=np.uint64).tobytes(),
                               digest_size=8).hexdigest()

    @staticmethod
    def jaccard(sig_a: np.ndarray, sig_b: np.ndarray) -> float:
        return float(np.mean(sig_a == sig_b))

    @staticmethod
    def bands(sig: np.ndarray, n_bands: int = 16) -> list[str]:
        band_size = max(1, len(sig) // n_bands)
        return [hashlib.blake2b(sig[i * band_size:(i + 1) * band_size].tobytes(),
                                digest_size=8).hexdigest() for i in range(n_bands)]


# =====================================================================
# 4. Silo split (topic aggregation + Dirichlet(alpha) non-IID)
# =====================================================================

def assign_silos(doc_ids: Sequence[str], topics: Sequence[str], n_silos: int,
                 alpha: float = C.DIRICHLET_ALPHA, seed: int = C.SEED) -> dict[str, int]:
    rng = np.random.default_rng(seed + 1000 * n_silos)
    by_topic: dict[str, list[int]] = defaultdict(list)
    for i, t in enumerate(topics):
        by_topic[t].append(i)
    assign = np.full(len(doc_ids), -1, dtype=np.int64)
    for topic in sorted(by_topic):
        idxs = by_topic[topic]
        p = rng.dirichlet(np.full(n_silos, alpha, dtype=np.float64))
        draws = rng.choice(n_silos, size=len(idxs), p=p)
        for j, i in enumerate(idxs):
            assign[i] = int(draws[j])
    # Empty-silo fix: migrate one document from the largest silo so every client is non-empty
    for c in range(n_silos):
        if not np.any(assign == c):
            counts = Counter(assign.tolist())
            donor = max(counts, key=lambda k: counts[k])
            victims = np.flatnonzero(assign == donor)
            assign[int(victims[len(victims) // 2])] = c
    return {doc_ids[i]: int(assign[i]) for i in range(len(doc_ids))}


def derive_topics(texts: Sequence[str], n_topics: int = 20, seed: int = C.SEED) -> list[str]:
    """Derive topics with TF-IDF + MiniBatchKMeans (used when BeIR corpora have no topic field)."""
    from sklearn.cluster import MiniBatchKMeans
    from sklearn.feature_extraction.text import TfidfVectorizer

    vec = TfidfVectorizer(max_features=60000, min_df=2, sublinear_tf=True,
                          stop_words="english", dtype="float32")
    X = vec.fit_transform(texts)
    k = int(min(n_topics, max(2, X.shape[0] // 50)))
    km = MiniBatchKMeans(n_clusters=k, random_state=seed, batch_size=4096, n_init=3,
                         max_iter=100)
    labels = km.fit_predict(X)
    return ["t%02d" % int(l) for l in labels]


def silo_stats(assign: dict[str, int], n_silos: int, topics: dict[str, str]) -> dict:
    per_client = Counter(assign.values())
    grid: dict[str, Counter] = defaultdict(Counter)
    for d, c in assign.items():
        grid["c%d" % c][topics.get(d, "unknown")] += 1
    return {
        "n_silos": n_silos,
        "per_client_docs": {"c%d" % c: per_client.get(c, 0) for c in range(n_silos)},
        "topic_by_client": {k: dict(v) for k, v in sorted(grid.items())},
        "min_client_docs": min(per_client.values()) if per_client else 0,
        "max_client_docs": max(per_client.values()) if per_client else 0,
        "topic_entropy_mean": float(np.mean([
            -sum((n / sum(cnt.values())) * math.log(n / sum(cnt.values()) + 1e-12)
                 for n in cnt.values()) for cnt in grid.values()])) if grid else 0.0,
    }


# =====================================================================
# 5. Shadow injection (near-duplicate rewritten copies, cross-silo)
# =====================================================================

STOPWORDS = set("""a about above after again against all am an and any are aren't as at be because been
before being below between both but by can't cannot could couldn't did didn't do does doesn't doing don't
down during each few for from further had hadn't has hasn't have haven't having he he'd he'll he's her here
here's hers herself him himself his how how's i i'd i'll i'm i've if in into is isn't it it's its itself
let's me more most mustn't my myself no nor not of off on once only or other ought our ours ourselves out
over own same shan't she she'd she'll she's should shouldn't so some such than that that's the their theirs
them themselves then there there's these they they'd they'll they're they've this those through to too
under until up very was wasn't we we'd we'll we're we've were weren't what what's when when's where where's
which while who who's whom why why's with won't would wouldn't you you'd you'll you're you've your yours
yourself yourselves also however therefore moreover thus hence although though since while whereas""".split())

SYNONYMS = {
    "said": "stated", "stated": "said", "says": "notes", "according": "per", "report": "account",
    "reported": "indicated", "announced": "declared", "revealed": "disclosed", "showed": "demonstrated",
    "found": "discovered", "added": "noted", "told": "informed", "big": "large", "large": "substantial",
    "small": "minor", "new": "recent", "old": "former", "good": "strong", "bad": "poor", "high": "elevated",
    "low": "reduced", "increase": "rise", "increased": "rose", "decrease": "decline", "decreased": "fell",
    "growth": "expansion", "company": "firm", "companies": "firms", "people": "individuals",
    "person": "individual", "year": "twelve-month period", "years": "decades", "week": "seven-day span",
    "month": "thirty-day period", "day": "24-hour period", "study": "investigation", "research": "inquiry",
    "data": "figures", "money": "funds", "market": "marketplace", "price": "cost", "prices": "costs",
    "government": "administration", "official": "representative", "officials": "representatives",
    "million": "seven-figure sum", "billion": "ten-figure sum", "percent": "percentage points",
    "help": "assist", "helps": "assists", "use": "employ", "used": "employed", "make": "produce",
    "made": "produced", "get": "obtain", "got": "obtained", "give": "provide", "gave": "provided",
    "take": "assume", "took": "assumed", "want": "seek", "wanted": "sought", "need": "require",
    "needs": "requires", "show": "demonstrate", "shows": "demonstrates", "start": "begin",
    "started": "began", "end": "conclude", "ended": "concluded", "begin": "commence",
    "important": "significant", "major": "principal", "main": "primary", "problem": "issue",
    "problems": "issues", "issue": "matter", "result": "outcome", "results": "outcomes",
    "cause": "trigger", "caused": "triggered", "change": "shift", "changed": "shifted",
    "support": "backing", "supported": "backed", "against": "opposed to", "because": "owing to the fact that",
    "but": "yet", "also": "additionally", "very": "extremely", "many": "numerous", "much": "considerable",
    "most": "the majority of", "some": "several", "all": "every", "more": "additional",
    "first": "initial", "last": "final", "next": "subsequent", "newer": "more recent",
    "health": "well-being", "doctor": "physician", "doctors": "physicians", "patient": "case",
    "patients": "cases", "treatment": "therapy", "disease": "illness", "vaccine": "immunization",
    "virus": "pathogen", "cases": "instances", "deaths": "fatalities", "risk": "hazard",
    "team": "squad", "game": "match", "games": "matches", "player": "athlete", "players": "athletes",
    "coach": "manager", "season": "campaign", "win": "victory", "won": "secured a win",
    "lost": "suffered a defeat", "score": "tally", "points": "scores",
    "tech": "technology", "phone": "handset", "device": "gadget", "devices": "gadgets",
    "app": "application", "users": "subscribers", "user": "subscriber", "launch": "rollout",
    "launched": "rolled out", "release": "unveiling", "released": "unveiled",
}


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", re.sub(r"\s+", " ", text).strip()) if s.strip()]


def perturb_text(text: str, rng: np.random.Generator, *, n_syn: int = 4, n_del: int = 3,
                 n_ins: int = 1, n_swap: int = 1) -> str:
    """Light perturbation: sentence-order swaps + stopword insert/delete + synonym substitution (preserves meaning and length scale)."""
    sents = split_sentences(text)
    if len(sents) >= 2 and n_swap:
        for _ in range(n_swap):
            i = int(rng.integers(0, len(sents) - 1))
            sents[i], sents[i + 1] = sents[i + 1], sents[i]
    words = " ".join(sents).split()
    if not words:
        return text

    drop: set[int] = set()
    cand_del = [i for i, w in enumerate(words) if w.strip(".,;:!?()\"'").lower() in STOPWORDS]
    if cand_del and n_del:
        order = rng.permutation(len(cand_del))[:n_del]
        drop = {cand_del[int(k)] for k in order}

    cand_syn = [i for i, w in enumerate(words)
                if i not in drop and w.strip(".,;:!?()\"'").lower() in SYNONYMS]
    syn_map: dict[int, str] = {}
    if cand_syn and n_syn:
        order = rng.permutation(len(cand_syn))[:n_syn]
        for k in order:
            i = cand_syn[int(k)]
            raw = words[i]
            core = raw.strip(".,;:!?()\"'")
            repl = SYNONYMS[core.lower()]
            if core[:1].isupper():
                repl = repl[:1].upper() + repl[1:]
            syn_map[i] = raw.replace(core, repl, 1)

    keep = [syn_map.get(i, w) for i, w in enumerate(words) if i not in drop]
    if n_ins:
        fillers = ["also", "however", "in addition", "moreover", "meanwhile"]
        for _ in range(n_ins):
            pos = int(rng.integers(1, max(2, len(keep))))
            keep.insert(pos, fillers[int(rng.integers(0, len(fillers)))])
    out = " ".join(keep)
    return re.sub(r"\s+", " ", out).strip()


# =====================================================================
# 6. Encoder (sentence-transformers preferred, transformers fallback)
# =====================================================================

class Encoder:
    """bge-small: CLS pooling + L2 normalization, float32 output."""

    def __init__(self, model_id: str = C.ENCODER_ID, device: str | None = None,
                 batch_size: int = 256, max_length: int = 256, use_fp16: bool = True) -> None:
        import torch
        self.torch = torch
        self.model_id = model_id
        self.batch_size = batch_size
        self.max_length = max_length
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.backend = "sentence-transformers"
        self._cache: dict[str, np.ndarray] = {}
        try:
            from sentence_transformers import SentenceTransformer
            self.st = SentenceTransformer(model_id, device=self.device)
            self.st.max_seq_length = max_length
            if use_fp16 and self.device == "cuda":
                try:
                    self.st.half()
                except Exception:
                    pass
            self._dim = int(self.st.get_sentence_embedding_dimension())
        except Exception as exc:  # noqa: BLE001
            log("sentence-transformers unavailable (%s); falling back to transformers AutoModel" % exc)
            self.backend = "transformers"
            from transformers import AutoModel, AutoTokenizer
            self.tok = AutoTokenizer.from_pretrained(model_id)
            self.model = AutoModel.from_pretrained(model_id)
            self.model.eval().to(self.device)
            if use_fp16 and self.device == "cuda":
                self.model.half()
            self._dim = int(self.model.config.hidden_size)

    @property
    def dim(self) -> int:
        return self._dim

    def _encode_transformers(self, texts: list[str]) -> np.ndarray:
        torch = self.torch
        with torch.no_grad():
            batch = self.tok(texts, padding=True, truncation=True, max_length=self.max_length,
                             return_tensors="pt").to(self.device)
            out = self.model(**batch)
            last = out.last_hidden_state[:, 0]          # bge: CLS pooling
            last = torch.nn.functional.normalize(last.float(), p=2, dim=1)
        return last.cpu().numpy().astype(np.float32)

    def encode(self, texts: Sequence[str], batch_size: int | None = None,
               progress_every: int = 20000) -> np.ndarray:
        bs = batch_size or self.batch_size
        todo = [t for t in texts if t not in self._cache]
        out = np.zeros((len(texts), self._dim), dtype=np.float32)
        if todo:
            t0 = time.time()
            done = 0
            for i in range(0, len(todo), bs):
                chunk = todo[i:i + bs]
                if self.backend == "sentence-transformers":
                    vecs = self.st.encode(chunk, batch_size=bs, convert_to_numpy=True,
                                          normalize_embeddings=True, show_progress_bar=False)
                    vecs = np.asarray(vecs, dtype=np.float32)
                else:
                    vecs = self._encode_transformers(list(chunk))
                if len(self._cache) < 400000:
                    for t, v in zip(chunk, vecs):
                        self._cache[t] = v
                done += len(chunk)
                if progress_every and done % progress_every < bs:
                    rate = done / max(1e-6, time.time() - t0)
                    log("  encoding %d/%d  (%.0f text/s)" % (done, len(todo), rate))
            log("  encoded %d items in %.1fs" % (len(todo), time.time() - t0))
        for i, t in enumerate(texts):
            v = self._cache.get(t)
            if v is None:
                raise RuntimeError("embedding cache miss")
            out[i] = v
        return out


# =====================================================================
# 7. Main build flow
# =====================================================================

DS_LOADERS = {
    "ds1": "multihoprag",
    "ds2": "nq",
    "ds3": "trec-covid",
}


def stratified_pick(items: list[dict], n: int, key: str, seed: int) -> list[dict]:
    """Stratified sampling by meta[key] (input is a list of query dicts)."""
    if len(items) <= n:
        return list(items)
    groups: dict[Any, list[dict]] = defaultdict(list)
    for it in items:
        groups[(it.get("meta") or {}).get(key, "unknown")].append(it)
    rng = np.random.default_rng(seed)
    picked: list[dict] = []
    for g in sorted(groups, key=lambda k: str(k)):
        pool = groups[g]
        take = max(1, int(round(n * len(pool) / len(items))))
        take = min(take, len(pool))
        idx = rng.permutation(len(pool))[:take]
        picked.extend(pool[int(i)] for i in idx)
    if len(picked) > n:
        picked = picked[:n]
    while len(picked) < n:
        for it in items:
            if it not in picked:
                picked.append(it)
                if len(picked) == n:
                    break
    return picked


def build(ds_key: str, *, max_queries: int, max_orig_docs: int | None,
          n_queries_target: int | None = None, do_embed: bool = True,
          embed_only: bool = False, device: str | None = None,
          batch_size: int = 256, minhash_backend: str = "fast",
          out_root: Path = C.PROCESSED, smoke: bool = False,
          alpha: float = C.DIRICHLET_ALPHA) -> dict:
    name = DS_LOADERS[ds_key]
    out_dir = Path(out_root) / name
    out_dir.mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    stats: dict[str, Any] = {"dataset": name, "ds_key": ds_key, "seed": C.SEED}

    # ---------------------------------------------------------- 6.0 embed only
    if embed_only:
        corpus = load_jsonl(out_dir / "corpus.jsonl")
        log("%s: embed-only, loaded %d chunks" % (name, len(corpus)))
        enc = Encoder(device=device, batch_size=batch_size)
        vecs = enc.encode([r["text"] for r in corpus], batch_size=batch_size)
        np.save(out_dir / C.EMB_FILENAME, vecs.astype(np.float32))
        dump_json(out_dir / C.EMB_PIDS_FILENAME, [r["pid"] for r in corpus])
        dump_json(out_dir / "embedding_meta.json", {
            "encoder": C.ENCODER_ID, "dim": int(vecs.shape[1]), "n": int(vecs.shape[0]),
            "normalized": True, "dtype": "float32", "backend": enc.backend,
            "row_order": "corpus.jsonl row order / pid ascending"})
        return {"dataset": name, "embed_only": True, "n": len(corpus)}

    # ---------------------------------------------------------- 6.1 raw data
    t = time.time()
    if ds_key == "ds1":
        raw_docs, raw_qs, info = load_ds1_raw()
    elif ds_key == "ds2":
        raw_docs, raw_qs, info = load_ds2_raw(max_corpus=max_orig_docs or 100000, max_queries=None)
    else:
        raw_docs, raw_qs, info = load_ds3_raw(max_queries=None)
    if max_orig_docs and len(raw_docs) > max_orig_docs:
        raw_docs = raw_docs[:max_orig_docs]
    log("%s: raw docs %d, raw queries %d (%.1fs)" % (name, len(raw_docs), len(raw_qs), time.time() - t))
    stats["raw"] = info

    # ---------------------------------------------------------- 6.2 chunking
    t = time.time()
    docs: list[dict] = []          # raw documents (doc level)
    chunks: list[dict] = []        # chunk level
    n_title_only = 0
    for i, d in enumerate(raw_docs):
        doc_id = "d%06d" % i
        text = d.text
        if len(text.split()) < 25 and len((d.title or "").split()) >= 5:
            text = (d.title or "").strip()          # title-only entries (present in BeIR); fall back to the title
            n_title_only += 1
        pieces = chunk_text(text)
        if not pieces:
            continue
        docs.append({"doc_id": doc_id, "raw_id": d.raw_id, "topic": d.topic,
                     "title": d.title, "url": d.url, "n_chunks": len(pieces)})
        for j, p in enumerate(pieces):
            chunks.append({"doc_id": doc_id, "chunk_idx": j, "text": p,
                           "topic": d.topic, "raw_id": d.raw_id})
    for k, ch in enumerate(chunks):
        ch["pid"] = k
    doc_index = {d["doc_id"]: d for d in docs}
    log("%s: chunking done %d docs -> %d chunks (%.1fs)" % (name, len(docs), len(chunks), time.time() - t))
    stats["chunking"] = {"chunk_words": C.CHUNK_WORDS, "overlap": C.CHUNK_OVERLAP,
                         "n_raw_docs_with_chunks": len(docs), "n_title_only_docs": n_title_only,
                         "n_chunks": len(chunks),
                         "words_mean": float(np.mean([len(c["text"].split()) for c in chunks])) if chunks else 0.0,
                         "words_total": int(sum(len(c["text"].split()) for c in chunks))}

    # ---------------------------------------------------------- 6.3 queries and gold mapping
    if n_queries_target is None:
        n_queries_target = {"ds1": 2000, "ds2": 1000, "ds3": 500}[ds_key]
    if smoke:
        n_queries_target = min(n_queries_target, 50)

    ref_to_doc: dict[str, str] = {}
    for d in docs:
        for key in (d.get("url"), d.get("title"), d.get("raw_id")):
            if key:
                ref_to_doc.setdefault(str(key), d["doc_id"])

    pid_by_doc: dict[str, list[int]] = defaultdict(list)
    for c in chunks:
        pid_by_doc[c["doc_id"]].append(c["pid"])

    queries: list[dict] = []
    qmeta: dict[str, dict] = {}
    for q in raw_qs:
        gold_docs = []
        for ref in q.gold_refs:
            did = ref_to_doc.get(str(ref))
            if did and did not in gold_docs:
                gold_docs.append(did)
        gold_pids = sorted({p for did in gold_docs for p in pid_by_doc.get(did, [])})
        queries.append({"raw_id": q.raw_id, "query": q.query, "answers": q.answers,
                        "gold_docs": gold_docs, "gold_pids": gold_pids, "meta": dict(q.meta)})

    # DS3: only 50 official queries -> synthesize extra queries from document titles up to the target size (explicitly labeled)
    n_synth = 0
    if ds_key == "ds3" and len(queries) < n_queries_target:
        need = n_queries_target - len(queries)
        rng = seeded_rng("ds3-synth", C.SEED)
        cand = [d for d in docs if len(d.get("title") or "") >= 25 and d["n_chunks"] >= 1]
        order = rng.permutation(len(cand))[:need] if len(cand) >= need else np.arange(len(cand))
        for k in order:
            d = cand[int(k)]
            queries.append({"raw_id": "synth-%06d" % int(k), "query": d["title"],
                            "answers": [], "gold_docs": [d["doc_id"]],
                            "gold_pids": sorted(pid_by_doc[d["doc_id"]]),
                            "meta": {"question_type": "title_lookup", "query_source": "synthetic_title",
                                     "has_qrels": False}})
            n_synth += 1
        log("%s: %d official queries, added %d synthetic queries" % (name, len(queries) - n_synth, n_synth))

    with_gold = [q for q in queries if q["gold_pids"]]
    without_gold = [q for q in queries if not q["gold_pids"]]
    if ds_key == "ds1":
        picked = stratified_pick(with_gold, n_queries_target, "question_type", C.SEED)
        if len(picked) < n_queries_target:
            picked += without_gold[:n_queries_target - len(picked)]
    else:
        rng = np.random.default_rng(C.SEED)
        pool = list(with_gold)
        if len(pool) > n_queries_target:
            idx = rng.permutation(len(pool))[:n_queries_target]
            picked = [pool[int(i)] for i in idx]
        else:
            picked = pool + without_gold[:max(0, n_queries_target - len(pool))]
    picked = picked[:n_queries_target]
    log("%s: selected %d queries (%d with gold)" % (name, len(picked), sum(1 for q in picked if q["gold_pids"])))

    # ---------------------------------------------------------- 6.4 MinHash fingerprints
    t = time.time()
    mh = MinHasher(C.MINHASH_PERM, C.SEED, backend=minhash_backend)
    sigs = np.zeros((len(chunks), C.MINHASH_PERM), dtype=np.uint64)
    for i, c in enumerate(chunks):
        sig = mh.signature(c["text"])
        sigs[i] = sig
        c["fingerprint"] = MinHasher.fingerprint(sig)
        if (i + 1) % 50000 == 0:
            log("  minhash %d/%d" % (i + 1, len(chunks)))
    log("%s: MinHash fingerprints done (%.1fs)" % (name, time.time() - t))

    # ------------------------------------------------- 6.4b topic derivation (BeIR has no topic field)
    tc0 = Counter(d["topic"] for d in docs)
    dominant = (max(tc0.values()) / max(1, len(docs))) if tc0 else 1.0
    topic_source = "raw"
    if dominant > 0.5 and len(docs) >= 50:
        first_chunk: dict[str, str] = {}
        for c in chunks:
            first_chunk.setdefault(c["doc_id"], c["text"])
        texts = [(d["title"] or "") + ". " + first_chunk.get(d["doc_id"], "") for d in docs]
        labels = derive_topics(texts, n_topics=20, seed=C.SEED)
        for d, lab in zip(docs, labels):
            d["topic"] = lab
        topic_source = "tfidf_kmeans"
        log("  topic derivation (tfidf+kmeans20): original dominant topic share %.2f" % dominant)
    tc1 = Counter(d["topic"] for d in docs)
    stats["topics"] = {"source": topic_source, "n_topics": len(tc1),
                       "dominant_share_before": round(dominant, 3),
                       "dominant_share_after": round(max(tc1.values()) / max(1, len(docs)), 3),
                       "distribution": dict(tc1.most_common(25))}

    # ---------------------------------------------------------- 6.5 silo split
    doc_ids = [d["doc_id"] for d in docs]
    doc_topics = [d["topic"] for d in docs]
    silos: dict[str, dict[str, int]] = {}
    silo_stats_all: dict[str, dict] = {}
    for K in C.SILO_SIZES:
        a = assign_silos(doc_ids, doc_topics, K, alpha, C.SEED)
        silos["silo%d" % K] = a
        silo_stats_all["silo%d" % K] = silo_stats(a, K, {d["doc_id"]: d["topic"] for d in docs})
        log("  silo%d: per-silo doc counts %s" % (K, silo_stats_all["silo%d" % K]["per_client_docs"]))
    primary = silos["silo5"]
    stats["silos"] = silo_stats_all

    # ---------------------------------------------------------- 6.6 shadow injection (10%/30%, nested)
    t = time.time()
    rng = seeded_rng("shadow", C.SEED)
    perm = rng.permutation(len(docs))
    n30 = int(round(C.SHADOW_RATIOS["r30"] * len(docs)))
    n10 = int(round(C.SHADOW_RATIOS["r10"] * len(docs)))
    n30 = max(n30, n10)
    selected = [docs[int(i)] for i in perm[:n30]]
    r10_ids = {selected[i]["doc_id"] for i in range(min(n10, len(selected)))}
    log("%s: shadow injection selected %d docs (r10=%d, r30=%d)" % (name, len(selected), len(r10_ids), len(selected)))

    shadow_rows: list[dict] = []
    for j, d in enumerate(selected):
        pids = pid_by_doc[d["doc_id"]]
        if not pids:
            continue
        src_pid = int(pids[int(rng.integers(0, len(pids)))])
        src_text = chunks[src_pid]["text"]
        # Cross-silo: the shadow lands in a different silo from the original document
        orig_client = primary[d["doc_id"]]
        cand = [c for c in range(5) if c != orig_client]
        tgt = int(cand[int(rng.integers(0, len(cand)))])
        orig_client10 = silos["silo10"][d["doc_id"]]
        cand10 = [c for c in range(10) if c != orig_client10]
        tgt10 = int(cand10[int(rng.integers(0, len(cand10)))])
        shadow_rows.append({
            "shadow_doc_id": "s%06d" % j, "orig_doc_id": d["doc_id"], "orig_pid": src_pid,
            "text": perturb_text(src_text, rng), "src_text": src_text,
            "client": tgt, "client10": tgt10, "orig_client": orig_client, "topic": d["topic"],
            "ratio10": d["doc_id"] in r10_ids,
        })

    # Validate/calibrate shadow similarity with bge-small (retry with lighter perturbation when below the floor)
    enc = Encoder(device=device, batch_size=batch_size) if do_embed else None
    if enc is not None and shadow_rows:
        t2 = time.time()
        pair_texts = [r["src_text"] for r in shadow_rows] + [r["text"] for r in shadow_rows]
        vecs = enc.encode(pair_texts, batch_size=batch_size)
        n = len(shadow_rows)
        sims = np.sum(vecs[:n] * vecs[n:], axis=1)
        retry_idx = [i for i in range(n) if sims[i] < C.SHADOW_SIM_FLOOR]
        if retry_idx:
            log("  %d shadows below similarity floor %.2f; retrying with lighter perturbation" % (len(retry_idx), C.SHADOW_SIM_FLOOR))
            rng2 = seeded_rng("shadow-retry", C.SEED)
            lighter = [perturb_text(shadow_rows[i]["src_text"], rng2, n_syn=1, n_del=1, n_ins=0, n_swap=1)
                       for i in retry_idx]
            v2 = enc.encode(lighter, batch_size=batch_size)
            for k, i in enumerate(retry_idx):
                s2 = float(np.dot(vecs[i], v2[k]))
                if s2 > sims[i]:
                    sims[i] = s2
                    shadow_rows[i]["text"] = lighter[k]
        for i in range(n):
            shadow_rows[i]["sim"] = float(sims[i])
        log("  shadow similarity: mean=%.4f min=%.4f max=%.4f (%.1fs)" % (
            float(sims.mean()), float(sims.min()), float(sims.max()), time.time() - t2))

    weak = [r for r in shadow_rows if r.get("sim", 1.0) < C.SHADOW_SIM_FLOOR]
    stats["shadow"] = {
        "n_selected_docs": len(selected), "n_r10": len(r10_ids), "n_injected": len(shadow_rows),
        "ratio_doc_level": {"r10": C.SHADOW_RATIOS["r10"], "r30": C.SHADOW_RATIOS["r30"]},
        "sim_mean": float(np.mean([r["sim"] for r in shadow_rows if "sim" in r])) if any("sim" in r for r in shadow_rows) else None,
        "sim_min": float(np.min([r["sim"] for r in shadow_rows if "sim" in r])) if any("sim" in r for r in shadow_rows) else None,
        "n_below_floor": len(weak),
    }

    # Append shadow chunks to the corpus (new doc_id / new pid)
    body = list(chunks)
    shadow_sigs: list[np.ndarray] = []
    for r in shadow_rows:
        pid = len(body)
        sig = mh.signature(r["text"])
        shadow_sigs.append(sig)
        body.append({"doc_id": r["shadow_doc_id"], "chunk_idx": 0, "text": r["text"],
                     "topic": r["topic"], "raw_id": r["shadow_doc_id"], "pid": pid,
                     "fingerprint": MinHasher.fingerprint(sig),
                     "is_shadow": True, "orig_doc_id": r["orig_doc_id"]})
        r["shadow_pid"] = pid
    if shadow_sigs:
        sigs = np.vstack([sigs, np.stack(shadow_sigs)])
    silo10_of = dict(silos["silo10"])
    silo10_of.update({r["shadow_doc_id"]: r["client10"] for r in shadow_rows})
    log("%s: corpus total %d chunks (%d shadows) (%.1fs)" % (name, len(body), len(shadow_rows), time.time() - t))

    # ---------------------------------------------------------- 6.7 forget sets (doc level, nested, single silo)
    # A2: a request can only name documents of the requester's own silo, so D_R subset D_{c_R} and V_R all fall in the requester's shard.
    # The requester is the silo with the most documents (deterministic rule), to accommodate the corpus-level r20 ratio.
    doc_client5 = dict(silos["silo5"])
    by_client: dict[str, list[int]] = defaultdict(list)
    for i, d in enumerate(docs):
        by_client[str(doc_client5.get(d["doc_id"], "c0"))].append(i)
    requester = max(sorted(by_client), key=lambda c: (len(by_client[c]), c))
    pool = sorted(by_client[requester])
    if len(pool) < max(1, int(round(C.FORGET_RATIOS["r20"] * len(docs)))):
        raise ValueError(
            "requester silo %s has only %d docs, cannot hold r20=%d"
            % (requester, len(pool), int(round(C.FORGET_RATIOS["r20"] * len(docs))))
        )
    rngf = seeded_rng("forget", C.SEED)
    permf = rngf.permutation(len(pool))
    forget: dict[str, list[str]] = {}
    prev: list[str] = []
    for tag in ("r1", "r5", "r20"):
        want = max(1, int(round(C.FORGET_RATIOS[tag] * len(docs))))
        want = max(want, len(prev))
        want = min(want, len(pool))
        ids = [docs[pool[int(i)]]["doc_id"] for i in permf[:want]]
        forget[tag] = sorted(ids)
        prev = ids
    stats["forget_sets"] = {k: {"n_docs": len(v),
                                "n_chunks": int(sum(len(pid_by_doc[d]) for d in v))}
                            for k, v in forget.items()}
    stats["forget_sets"]["requester_client"] = requester
    stats["forget_sets"]["requester_docs"] = len(pool)

    # ---------------------------------------------------------- 6.8 write corpus/queries
    topic_by_doc = {d["doc_id"]: d["topic"] for d in docs}
    # Backfill shadow clients from shadow_rows
    shadow_client = {r["shadow_doc_id"]: r["client"] for r in shadow_rows}
    shadow_sim = {r["shadow_doc_id"]: r.get("sim") for r in shadow_rows}
    corpus_out = []
    pid_meta = {}
    for c in body:
        did = c["doc_id"]
        if c.get("is_shadow"):
            client = "c%d" % shadow_client[did]
            topic = c["topic"]
            n_chunks_of_doc = 1
        else:
            client = "c%d" % primary[did]
            topic = topic_by_doc[did]
            n_chunks_of_doc = doc_index[did]["n_chunks"]
        corpus_out.append({
            "pid": int(c["pid"]),
            "doc_id": did,
            "client_id": client,
            "topic": topic,
            "text": c["text"],
            "n_tokens": int(len(c["text"].split())),
            "fingerprint": c["fingerprint"],
        })
        pid_meta[str(c["pid"])] = {
            "doc_id": did, "chunk_idx": int(c["chunk_idx"]),
            "n_chunks_of_doc": int(n_chunks_of_doc),
            "client_id_silo10": "c%d" % silo10_of[did],
            "topic": topic, "is_shadow": bool(c.get("is_shadow")),
            "orig_doc_id": c.get("orig_doc_id"),
            "shadow_sim": shadow_sim.get(did) if c.get("is_shadow") else None,
            "n_words": int(len(c["text"].split())),
            "raw_id": c.get("raw_id"),
        }
    n_corpus = dump_jsonl(out_dir / "corpus.jsonl", corpus_out)

    # Assign queries to silos: the client with the most gold hits
    pid_to_client = {r["pid"]: r["client_id"] for r in corpus_out}
    query_out = []
    for i, q in enumerate(picked):
        if q["gold_pids"]:
            cnt = Counter(pid_to_client[p] for p in q["gold_pids"] if p in pid_to_client)
            client = cnt.most_common(1)[0][0] if cnt else "c0"
        else:
            client = "c%d" % int(np.random.default_rng(C.SEED + i).integers(0, 5))
        query_out.append({
            "qid": i,
            "client_id": client,
            "query": q["query"],
            "answers": q["answers"],
            "gold_pids": [int(p) for p in q["gold_pids"]],
        })
        qmeta[str(i)] = {
            "raw_id": q["raw_id"], "gold_doc_ids": q["gold_docs"],
            "n_gold_pids": len(q["gold_pids"]),
            "question_type": q["meta"].get("question_type"),
            "query_source": q["meta"].get("query_source", "official"),
            "has_qrels": bool(q["meta"].get("has_qrels", bool(q["gold_pids"]))),
            "answers": q["answers"],
        }
    n_queries = dump_jsonl(out_dir / "queries.jsonl", query_out)

    dump_json(out_dir / "forget_sets.json", forget)
    dump_json(out_dir / "shadow_pairs.json", {
        "meta": {"ratios": {"r10": C.SHADOW_RATIOS["r10"], "r30": C.SHADOW_RATIOS["r30"]},
                 "definition": "For each selected document, take one of its chunks, lightly perturb it, and place the near-duplicate copy into another silo",
                 "sim_metric": "bge-small-en-v1.5 CLS cosine similarity",
                 "nested": True,
                 "sim_floor": C.SHADOW_SIM_FLOOR},
        "injected": [{"orig": r["orig_doc_id"], "shadow": r["shadow_doc_id"],
                      "client": "c%d" % r["client"], "sim": round(float(r.get("sim", 0.0)), 6)}
                     for r in shadow_rows],
        "injected_r10": [{"orig": r["orig_doc_id"], "shadow": r["shadow_doc_id"],
                          "client": "c%d" % r["client"], "sim": round(float(r.get("sim", 0.0)), 6)}
                         for r in shadow_rows if r["ratio10"]],
        "injected_r30": [{"orig": r["orig_doc_id"], "shadow": r["shadow_doc_id"],
                          "client": "c%d" % r["client"], "sim": round(float(r.get("sim", 0.0)), 6)}
                         for r in shadow_rows],
        "shadow_pid": {r["shadow_doc_id"]: r["shadow_pid"] for r in shadow_rows},
        "client_silo10": {r["shadow_doc_id"]: "c%d" % r["client10"] for r in shadow_rows},
        "orig_pid": {r["shadow_doc_id"]: r["orig_pid"] for r in shadow_rows},
    })
    dump_json(out_dir / "silo_assignments.json", {
        "alpha": alpha, "seed": C.SEED, "primary": "silo5",
        "note": "client_id in corpus.jsonl is silo5; for silo10 see doc_to_client.silo10",
        "doc_to_client": silos,
        "stats": silo_stats_all,
        "query_client": {str(q["qid"]): q["client_id"] for q in query_out},
    })
    dump_json(out_dir / "pid_meta.json", pid_meta)
    dump_json(out_dir / "qid_meta.json", qmeta)
    np.save(out_dir / "minhash_sig.npy", sigs.astype(np.uint64))
    dump_json(out_dir / "minhash_meta.json", {
        "num_perm": C.MINHASH_PERM, "backend": minhash_backend, "seed": C.SEED,
        "fingerprint": "blake2b(signature, digest_size=8) -> 16 hex chars (64-bit)",
        "row_order": "corpus.jsonl row order / pid ascending"})

    stats["corpus"] = {"n_chunks": n_corpus, "n_shadow_chunks": len(shadow_rows),
                       "n_words_total": int(sum(r["n_tokens"] for r in corpus_out))}
    stats["queries"] = {
        "n_queries": n_queries,
        "n_with_gold": sum(1 for q in query_out if q["gold_pids"]),
        "n_synthetic": n_synth,
        "gold_pids_mean": float(np.mean([len(q["gold_pids"]) for q in query_out])) if query_out else 0.0,
        "per_client": dict(Counter(q["client_id"] for q in query_out)),
    }

    # ---------------------------------------------------------- 6.9 encoding
    if do_embed:
        t = time.time()
        log("%s: encoding %d items (backend=%s, device=%s)" % (name, len(corpus_out), enc.backend, enc.device))
        vecs = enc.encode([r["text"] for r in corpus_out], batch_size=batch_size)
        np.save(out_dir / C.EMB_FILENAME, vecs.astype(np.float32))
        dump_json(out_dir / C.EMB_PIDS_FILENAME, [r["pid"] for r in corpus_out])
        dump_json(out_dir / "embedding_meta.json", {
            "encoder": C.ENCODER_ID, "dim": int(vecs.shape[1]), "n": int(vecs.shape[0]),
            "normalized": True, "dtype": "float32", "backend": enc.backend,
            "device": enc.device, "row_order": "corpus.jsonl row order / pid ascending",
            "seconds": round(time.time() - t, 2)})
        stats["embedding"] = {"dim": int(vecs.shape[1]), "n": int(vecs.shape[0]),
                              "seconds": round(time.time() - t, 2),
                              "norm_check": float(np.mean(np.linalg.norm(vecs, axis=1)))}

    stats["elapsed_seconds"] = round(time.time() - t_start, 2)
    dump_json(out_dir / "dataset_stats.json", stats)
    log("%s: done in %.1fs -> %s" % (name, stats["elapsed_seconds"], out_dir))
    return stats


# =====================================================================
# 8. Read helpers (for other modules, read-only)
# =====================================================================

def resolve_dir(ds_key_or_dir: Any, root: Path = C.PROCESSED) -> Path:
    """Accept keys like "ds1"/"multihoprag", or a directory path directly."""
    p = Path(ds_key_or_dir)
    if p.is_dir():
        return p
    key = str(ds_key_or_dir)
    return Path(root) / DS_LOADERS.get(key, key)


def load_minhash_meta(ds_key_or_dir: Any, root: Path = C.PROCESSED) -> dict:
    """Read minhash_meta.json; return {} when it does not exist."""
    d = resolve_dir(ds_key_or_dir, root)
    p = d / "minhash_meta.json"
    if not p.exists():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))


def load_signature_matrix(ds_key_or_dir: Any, root: Path = C.PROCESSED) -> np.ndarray:
    """Read the 64-permutation MinHash signature matrix, shape (n_chunks, num_perm), dtype uint64.

    Row order matches corpus.jsonl / embedding_pids.json (pid ascending).
    Note: the fingerprint field in corpus.jsonl is the 64-bit blake2b digest of the signature and
    can only support exact-duplicate detection; estimating Jaccard requires the signature matrix
    returned by this function.
    """
    d = resolve_dir(ds_key_or_dir, root)
    path = d / "minhash_sig.npy"
    if not path.exists():
        raise FileNotFoundError("MinHash signature matrix not found: " + str(path))
    arr = np.load(path)
    if arr.ndim != 2:
        raise ValueError("minhash_sig.npy has abnormal dimensions: %r" % (arr.shape,))
    arr = np.ascontiguousarray(arr, dtype=np.uint64)
    meta = load_minhash_meta(d)
    num_perm = meta.get("num_perm")
    if num_perm is not None and int(num_perm) != int(arr.shape[1]):
        raise ValueError(
            "minhash_meta.json num_perm=%r does not match signature matrix column count %d" % (num_perm, arr.shape[1]))
    return arr


def load_bundle(ds_key: str, root: Path = C.PROCESSED) -> dict:
    """One-stop reader for all on-disk artifacts of a dataset.

    signatures / minhash_meta are None / {} when not yet generated, without raising (to avoid blocking smoke runs).
    """
    name = DS_LOADERS[ds_key] if ds_key in DS_LOADERS else ds_key
    d = Path(root) / name
    bundle = {
        "dir": d, "name": name,
        "corpus": load_jsonl(d / "corpus.jsonl"),
        "queries": load_jsonl(d / "queries.jsonl"),
        "forget_sets": json.loads((d / "forget_sets.json").read_text(encoding="utf-8")),
        "shadow_pairs": json.loads((d / "shadow_pairs.json").read_text(encoding="utf-8")),
        "embedding_meta": json.loads((d / "embedding_meta.json").read_text(encoding="utf-8")),
        "silo_assignments": json.loads((d / "silo_assignments.json").read_text(encoding="utf-8")),
        "stats": json.loads((d / "dataset_stats.json").read_text(encoding="utf-8")),
    }
    bundle["embeddings"] = np.load(d / C.EMB_FILENAME)
    bundle["embedding_pids"] = json.loads((d / C.EMB_PIDS_FILENAME).read_text(encoding="utf-8"))
    bundle["signatures"] = None
    bundle["minhash_meta"] = {}
    try:
        bundle["signatures"] = load_signature_matrix(d)
        bundle["minhash_meta"] = load_minhash_meta(d)
    except Exception as exc:  # noqa: BLE001  -- missing artifacts are allowed at the smoke stage
        log("load_bundle: signature matrix unavailable (%s)" % exc)
    return bundle


# =====================================================================
# 9. CLI
# =====================================================================

def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="FedRevoke data pipeline")
    ap.add_argument("--dataset", default="all", choices=["ds1", "ds2", "ds3", "all"])
    ap.add_argument("--out", default=str(C.PROCESSED))
    ap.add_argument("--device", default=None)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--max-orig-docs", type=int, default=None)
    ap.add_argument("--max-queries", type=int, default=None,
                    help="target query-count override (default ds1=2000 ds2=1000 ds3=500)")
    ap.add_argument("--no-embed", action="store_true")
    ap.add_argument("--embed-only", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--minhash-backend", default="fast", choices=["fast", "datasketch"])
    ap.add_argument("--alpha", type=float, default=C.DIRICHLET_ALPHA, help="Dirichlet non-IID skew strength")
    ap.add_argument("--check", action="store_true", help="only validate on-disk artifacts (including the signature matrix)")
    args = ap.parse_args(argv)

    if args.check:
        out = Path(args.out)
        report = {}
        for key, name in DS_LOADERS.items():
            d = out / name
            item: dict[str, Any] = {"dir": str(d), "exists": d.exists()}
            if d.exists():
                try:
                    sig = load_signature_matrix(d)
                    item["minhash_sig_shape"] = list(sig.shape)
                    item["minhash_sig_dtype"] = str(sig.dtype)
                except Exception as exc:  # noqa: BLE001
                    item["minhash_sig_error"] = str(exc)
                item["minhash_meta"] = load_minhash_meta(d)
                for fn in ("corpus.jsonl", "queries.jsonl", "forget_sets.json",
                           "shadow_pairs.json", C.EMB_FILENAME, C.EMB_PIDS_FILENAME):
                    item[fn] = (d / fn).exists()
            report[key] = item
        print(json.dumps(report, ensure_ascii=False, indent=1))
        return 0

    keys = ["ds1", "ds2", "ds3"] if args.dataset == "all" else [args.dataset]
    all_stats = {}
    for k in keys:
        all_stats[k] = build(k, max_queries=args.max_queries, max_orig_docs=args.max_orig_docs,
                             n_queries_target=args.max_queries, do_embed=not args.no_embed,
                             embed_only=args.embed_only, device=args.device,
                             batch_size=args.batch_size, minhash_backend=args.minhash_backend,
                             out_root=Path(args.out), smoke=args.smoke, alpha=args.alpha)
    dump_json(Path(args.out) / "all_stats.json", all_stats)
    print(json.dumps({k: {"corpus": v.get("corpus"), "queries": v.get("queries")}
                      for k, v in all_stats.items()}, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
