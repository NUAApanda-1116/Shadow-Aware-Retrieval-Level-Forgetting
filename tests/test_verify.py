"""tests/test_verify.py -- MIAProbe / ElicitationTest / ForgetReport checks (synthetic data, no network).

Core assertions (matching the task requirements):
* Use MockGenerator + a synthetic index to verify "cleanly deleted -> elicit_rate approx 0, not deleted -> elicit_rate > 0".
* The MIA probe cannot distinguish the two classes while documents are still in the index (AUC = 0.5); after deletion the membership signal collapses (AUC -> 0).
* Numeric semantics of ForgetReport / rho_hat / TV estimation.
"""

from __future__ import annotations

import os
import subprocess
import sys
import zlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from fedrevoke.generation import MockGenerator  # noqa: E402
from fedrevoke.metrics import evidence_units, faithfulness, tokenize  # noqa: E402
from fedrevoke.verify import (  # noqa: E402
    ElicitationTest,
    ForgetReport,
    MIAProbe,
    answer_contained,
    build_forget_report,
    extract_answer_span,
    forgotten_pid_coverage,
    is_informative_reference,
    longest_common_token_run,
    retrieval_hit_rate,
    rho_hat_from_scores,
    rho_hat_residual,
    total_variation,
)

DIM = 64


# --------------------------------------------------------------------------- #
# Synthetic index and encoder
# --------------------------------------------------------------------------- #
@dataclass
class _Meta:
    doc_id: str
    text: str = ""


def _token_hash(token: str) -> int:
    return zlib.crc32(token.encode("utf-8")) % DIM


class _HashEncoder:
    """Deterministic bag-of-words hash encoder (for unit tests; no network, no model)."""

    def __init__(self, dim: int = DIM):
        self.dim = dim

    @staticmethod
    def _clean(text: str) -> list:
        out = []
        for token in str(text).lower().split():
            token = "".join(ch for ch in token if ch.isalnum())
            if token:
                out.append(token)
        return out

    def encode(self, texts, **kwargs):
        if isinstance(texts, str):
            texts = [texts]
        mat = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for token in self._clean(text):
                mat[row, _token_hash(token)] += 1.0
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        return mat / np.maximum(norms, 1e-9)


class _ToyIndex:
    """Brute-force synthetic index: keeps deleted vectors (for MIA/repair use), search does not return deleted ids."""

    backend = "faiss_ivf"

    def __init__(self):
        self._vecs: dict = {}
        self._metas: dict = {}
        self._alive: set = set()
        self._next = 0

    def add(self, vectors, metas):
        new_ids = []
        for vec, meta in zip(np.asarray(vectors, dtype=np.float32), list(metas)):
            iid = self._next
            self._next += 1
            self._vecs[iid] = np.asarray(vec, dtype=np.float32)
            self._metas[iid] = meta
            self._alive.add(iid)
            new_ids.append(iid)
        return new_ids

    def remove(self, ids):
        self._alive -= {int(i) for i in ids}

    def alive_ids(self):
        return np.asarray(sorted(self._alive), dtype=np.int64)

    def vector(self, internal_id):
        return self._vecs.get(int(internal_id))

    def meta(self, internal_id):
        return self._metas.get(int(internal_id))

    def text_for_id(self, internal_id):
        meta = self._metas.get(int(internal_id))
        return meta.text if meta is not None else None

    def stats(self):
        return {"n_vectors": len(self._alive), "n_deleted": len(self._vecs) - len(self._alive), "backend": self.backend}

    def search(self, queries, k: int = 10):
        arr = np.asarray(queries, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        alive = sorted(self._alive)
        k = int(k)
        scores = np.full((arr.shape[0], k), -1.0, dtype=np.float32)
        ids = np.full((arr.shape[0], k), -1, dtype=np.int64)
        if not alive:
            return scores, ids
        pool = np.asarray([self._vecs[i] for i in alive], dtype=np.float32)
        pool = pool / np.maximum(np.linalg.norm(pool, axis=1, keepdims=True), 1e-9)
        q = arr / np.maximum(np.linalg.norm(arr, axis=1, keepdims=True), 1e-9)
        sims = q @ pool.T
        for row in range(sims.shape[0]):
            order = np.argsort(-sims[row], kind="mergesort")[:k]
            for col, pos in enumerate(order.tolist()):
                scores[row, col] = sims[row, pos]
                ids[row, col] = alive[pos]
        return scores, ids


SECRET_TEXTS = [
    "The Zephyr project uses quantum flux capacitors to stabilise the reactor core.",
    "Zephyr engineers recalibrated the quantum flux capacitor bank in March.",
    "A quantum flux capacitor fault triggered the Zephyr emergency shutdown sequence.",
    "Three quantum flux capacitors are wired in series inside the Zephyr reactor.",
    "The Zephyr manual warns that quantum flux capacitors must stay below three hundred kelvin.",
]
FILLER_TEXTS = [
    "The cafeteria serves lentil soup every Tuesday afternoon.",
    "Quarterly budget reports are archived in the basement storage room.",
    "The library extended its opening hours during the summer term.",
    "A new bus route connects the campus with the northern suburbs.",
    "Volunteers planted maple saplings along the riverbank last autumn.",
    "The museum acquired a collection of nineteenth century maps.",
]
SECRET_QUERIES = [
    "What does the Zephyr reactor use to stabilise the core?",
    "Who recalibrated the quantum flux capacitor bank?",
    "What triggered the Zephyr emergency shutdown?",
    "How many quantum flux capacitors are wired in series?",
    "What temperature must Zephyr quantum flux capacitors stay below?",
]


def _build_index(n_secret: int = 5, n_filler: int = 6):
    encoder = _HashEncoder()
    texts = (SECRET_TEXTS * 4)[:n_secret] + (FILLER_TEXTS * 4)[:n_filler]
    metas = [
        _Meta("d{0:04d}".format(i), text) for i, text in enumerate(texts)
    ]
    index = _ToyIndex()
    index.add(encoder.encode(texts), metas)
    secret_ids = list(range(n_secret))
    filler_ids = list(range(n_secret, n_secret + n_filler))
    return index, encoder, secret_ids, filler_ids


def _qa_items(secret_ids, n_items: int = 5):
    items = []
    for i in range(n_items):
        items.append(
            {
                "qid": i,
                "query": SECRET_QUERIES[i % len(SECRET_QUERIES)],
                "answers": ["quantum flux capacitors"],
                "keywords": ["zephyr", "capacitor"],
                "gold_pids": list(secret_ids),
            }
        )
    return items


# --------------------------------------------------------------------------- #
# MockGenerator
# --------------------------------------------------------------------------- #
def test_mock_generator_is_deterministic_and_extracts_context():
    prompt = "Context:\n[1] alpha beta gamma\nQuestion: what?\nAnswer:"
    first = MockGenerator("MOCK").generate([prompt], max_new_tokens=50)
    second = MockGenerator("MOCK").generate([prompt], max_new_tokens=50)
    assert first == second
    assert first[0].startswith("MOCK:")
    assert "alpha beta gamma" in first[0]
    assert "Question:" not in first[0]  # keep only the evidence span
    assert MockGenerator("").generate([prompt], 50)[0].startswith("[1] alpha beta gamma")


def test_mock_generator_respects_token_budget_and_records_history():
    gen = MockGenerator("M")
    out = gen.generate(["Context:\n" + "word " * 100 + "\nQuestion: q"], max_new_tokens=5)
    assert len(out[0]) <= 5 * 4 + len("M: ") + 2
    assert gen.history and gen.n_generated == 1
    assert gen.generate([], 10) == []


def test_generation_module_is_lazy_and_gguf_reports_importerror():
    snippet = (
        "import sys\n"
        "import fedrevoke.generation as g\n"
        "print('torch' in sys.modules, 'transformers' in sys.modules)\n"
        "gen = g.MockGenerator('MOCK')\n"
        "print(len(gen.generate(['Context:\\n[1] alpha\\nQuestion: q\\nAnswer:'], 8)))\n"
        "try:\n"
        "    g.GGUFGenerator('repo', 'file.gguf')\n"
        "    print('gguf-ok')\n"
        "except ImportError:\n"
        "    print('gguf-importerror')\n"
    )
    env = dict(os.environ)
    env["PYTHONPATH"] = str(_SRC) + os.pathsep + env.get("PYTHONPATH", "")
    proc = subprocess.run(
        [sys.executable, "-c", snippet], capture_output=True, text=True, env=env, timeout=300
    )
    assert proc.returncode == 0, proc.stderr
    lines = proc.stdout.strip().splitlines()
    assert lines[0] == "False False", lines
    assert lines[1] == "1"
    assert lines[2] in ("gguf-ok", "gguf-importerror")


# --------------------------------------------------------------------------- #
# MIAProbe
# --------------------------------------------------------------------------- #
def test_mia_auc_moves_from_chance_to_zero_after_deletion():
    """Degenerate regime of the self-match-**included** mode (exclude_self=False), kept for regression and ablation notes.

    In this mode a surviving sample's self-match is always 1.0, so before deletion both classes are equally
    saturated and dirty AUC can only hover near chance (measured 0.6167); all discriminative power comes
    from after deletion (measured clean=0.0000). Paper Eq.(17) uses the self-match-excluded definition;
    see the next case.
    """
    index, encoder, secret_ids, filler_ids = _build_index()
    probe = MIAProbe(index, encoder=encoder, exclude_self=False)

    dirty = probe.auc(secret_ids, filler_ids)
    # Both classes self-match to 1.0 (membership signal saturated) -> AUC near 0.5; float32 makes 1.0 jitter by ~1e-7,
    # so this uses an interval assertion; the real discriminative power is after deletion (clean -> 0).
    assert 0.3 <= dirty <= 0.7, dirty

    assert probe.membership_scores(secret_ids).min() > 0.999
    assert probe.n_skipped == 0

    index.remove(secret_ids)
    clean = probe.auc(secret_ids, filler_ids)
    assert clean <= 0.05                      # membership signal collapses after deletion
    assert clean < dirty
    assert probe.membership_scores(secret_ids).max() < 0.99
    assert probe.membership_scores(filler_ids).min() > 0.999
    assert probe.distinguishability(secret_ids, filler_ids) >= 0.99
    details = probe.report_scores(secret_ids)
    assert len(details) == len(secret_ids) and set(details[0]) == {"pid", "score"}


def test_mia_default_excludes_self_and_is_informative_before_deletion():
    """Paper Eq.(17) definition: s(x) = max over u != x of <phi(p_x), phi(p_u)>; self-matches must be excluded by default.

    Measured: dirty=0.9333 -> clean=0.2333. Membership is distinguishable before deletion; after deletion the
    signal collapses with the correct direction (below chance, i.e. revoked samples are harder to retrieve than
    surviving ones). This is the metric suitable for comparing methods. If self-matches were included by default,
    dirty would degrade to 0.6167 (near chance) and differences between methods would be drowned in noise.
    """
    import inspect

    assert inspect.signature(MIAProbe.__init__).parameters["exclude_self"].default is True

    index, encoder, secret_ids, filler_ids = _build_index()
    probe = MIAProbe(index, encoder=encoder)  # use the defaults
    dirty = probe.auc(secret_ids, filler_ids)
    assert dirty >= 0.85, dirty

    index.remove(secret_ids)
    clean = probe.auc(secret_ids, filler_ids)
    assert clean <= 0.35, clean
    assert clean < 0.5, clean
    assert dirty - clean >= 0.5, (dirty, clean)


def test_mia_neighbor_scores_and_missing_vectors():
    index, encoder, secret_ids, filler_ids = _build_index()
    probe = MIAProbe(index, encoder=encoder)
    neighbors = probe.neighbor_scores(secret_ids)
    assert neighbors.size == len(secret_ids)
    assert neighbors.max() < 0.999  # self-match excluded

    empty_index = _ToyIndex()
    empty_probe = MIAProbe(empty_index, encoder=encoder)
    assert empty_probe.membership_scores(secret_ids).size == 0
    assert empty_probe.auc(secret_ids, filler_ids) == pytest.approx(0.5)

    index.remove(secret_ids)
    probe_after = MIAProbe(index, encoder=encoder)
    assert probe_after.membership_scores(secret_ids).size == len(secret_ids)


# --------------------------------------------------------------------------- #
# ElicitationTest: cleanly deleted -> elicit_rate approx 0; not deleted -> > 0
# --------------------------------------------------------------------------- #
def test_elicitation_rate_drops_to_zero_after_deletion():
    index, encoder, secret_ids, filler_ids = _build_index()
    items = _qa_items(secret_ids)
    generator = MockGenerator("MOCK")
    test = ElicitationTest(generator, index, k=5, max_new_tokens=200, encoder=encoder)

    rate_before = test.run(items)
    assert rate_before > 0.5                       # not deleted: the generator copies the evidence -> reproduces revoked knowledge
    details_before = list(test.last_details)
    assert details_before and all("retrieved" in d for d in details_before)
    assert any(set(d["retrieved"]) & set(secret_ids) for d in details_before)

    index.remove(secret_ids)
    rate_after = test.run(items)
    assert rate_after <= 0.1                       # cleanly deleted: elicit_rate approx 0
    assert rate_after < rate_before
    for detail in test.last_details:
        assert not (set(detail["retrieved"]) & set(secret_ids))
        assert detail["keyword_coverage"] == 0.0


def test_elicitation_uses_f1_when_no_keywords_given():
    index, encoder, secret_ids, _ = _build_index()
    items = [{"query": SECRET_QUERIES[0], "answers": ["quantum flux capacitors"]}]
    test = ElicitationTest(MockGenerator(""), index, k=3, max_new_tokens=60, encoder=encoder,
                           f1_threshold=0.05, keyword_threshold=1.1)
    rate, details = test.run_detailed(items)
    assert rate == pytest.approx(1.0)
    assert details[0]["f1"] >= 0.05
    assert details[0]["n_keywords"] >= 1


def test_elicitation_requires_encoder_or_text_search():
    index, encoder, secret_ids, _ = _build_index()
    test = ElicitationTest(MockGenerator("MOCK"), index, k=3, encoder=None)
    with pytest.raises(ValueError):
        test.run(_qa_items(secret_ids)[:1])


def test_retrieval_hit_rate_and_coverage():
    index, encoder, secret_ids, _ = _build_index()
    queries = encoder.encode(SECRET_QUERIES)
    assert retrieval_hit_rate(index, queries, secret_ids, k=5) == pytest.approx(1.0)
    assert forgotten_pid_coverage(index, queries, secret_ids, k=5) == pytest.approx(1.0)
    index.remove(secret_ids)
    assert retrieval_hit_rate(index, queries, secret_ids, k=5) == pytest.approx(0.0)
    assert forgotten_pid_coverage(index, queries, secret_ids, k=5) == pytest.approx(0.0)
    assert retrieval_hit_rate(index, np.zeros((0, 0), dtype=np.float32), secret_ids, k=5) == 0.0


# --------------------------------------------------------------------------- #
# ForgetReport / rho_hat / TV
# --------------------------------------------------------------------------- #
def test_forget_report_construction_and_roundtrip():
    report = build_forget_report(hit_rate=0.4, mia_auc_value=0.5, elicit_rate=0.2, n_probes=25)
    assert isinstance(report, ForgetReport)
    assert report.rho_hat == pytest.approx(0.4)          # max(hit, elicit)
    assert report.n_probes == 25
    payload = report.as_dict()
    assert set(payload) == {"hit_rate", "mia_auc", "elicit_rate", "rho_hat", "n_probes"}
    assert ForgetReport.from_dict(payload) == report
    assert "ForgetReport(" in str(report)
    explicit = build_forget_report(0.0, 0.5, 0.0, rho_hat=0.123, n_probes=3)
    assert explicit.rho_hat == pytest.approx(0.123)


def test_rho_hat_and_total_variation():
    assert rho_hat_residual(0.3, 0.7) == pytest.approx(0.7)
    assert rho_hat_residual(-1.0, 2.0) == pytest.approx(1.0)
    same = np.linspace(0.0, 1.0, 200)
    assert total_variation(same, same) == pytest.approx(0.0)
    assert total_variation(np.zeros(50), np.ones(50)) == pytest.approx(1.0)
    assert rho_hat_from_scores(same, same) == pytest.approx(0.0)
    shifted = same + 5.0
    assert rho_hat_from_scores(same, shifted, bins=16) > 0.5
    assert total_variation([], same) == 0.0


# --------------------------------------------------------------------------- #
# Integration with a real ProvenanceIndex (auto-enabled when available; skipped otherwise)
# --------------------------------------------------------------------------- #
def test_verify_with_real_provenance_index_if_available():
    index_core = pytest.importorskip("fedrevoke.index_core")
    provenance_index = getattr(index_core, "ProvenanceIndex", None)
    vec_meta = getattr(index_core, "VecMeta", None)
    if provenance_index is None or vec_meta is None:
        pytest.skip("index_core has not yet implemented ProvenanceIndex/VecMeta")
    encoder = _HashEncoder()
    texts = SECRET_TEXTS + FILLER_TEXTS
    try:
        index = provenance_index(dim=DIM, backend="faiss_ivf")
        metas = [
            vec_meta(pid=i, doc_id="d{0:04d}".format(i), client_id="c0", topic="t", fingerprint="f")
            for i in range(len(texts))
        ]
        index.add(encoder.encode(texts), metas)
    except Exception as exc:
        pytest.skip("ProvenanceIndex cannot be constructed in the current environment: {0}".format(exc))
    secret_ids = list(range(len(SECRET_TEXTS)))
    filler_ids = list(range(len(SECRET_TEXTS), len(texts)))
    probe = MIAProbe(index, encoder=encoder)
    assert 0.0 <= probe.auc(secret_ids, filler_ids) <= 1.0
    test = ElicitationTest(MockGenerator("MOCK"), index, k=3, encoder=encoder)
    rate = test.run(_qa_items(secret_ids)[:2])
    assert 0.0 <= rate <= 1.0


# --------------------------------------------------------------------------- #
# Criterion v2: answer-span references + containment + denominator discipline (regression test for DS2 structural zeros)
# --------------------------------------------------------------------------- #
class _TombIndex(_ToyIndex):
    """Synthetic index with tombstones: vectors/raw text remain retrievable after remove (consistent with ProvenanceIndex semantics)."""

    def deleted_ids(self):
        return np.asarray(sorted(set(self._vecs) - self._alive), dtype=np.int64)


class _EchoGenerator:
    """Fixed-output generator (for positive/negative controls; no model or network needed)."""

    def __init__(self, text: str = ""):
        self.text = str(text)
        self.n_calls = 0

    def generate(self, prompts, max_new_tokens: int = 200):
        self.n_calls += 1
        return [self.text for _ in prompts]


def _build_tomb_index():
    encoder = _HashEncoder()
    texts = SECRET_TEXTS + FILLER_TEXTS
    index = _TombIndex()
    index.add(encoder.encode(texts), [_Meta("d{0:04d}".format(i), t) for i, t in enumerate(texts)])
    secret_ids = list(range(len(SECRET_TEXTS)))
    filler_ids = list(range(len(SECRET_TEXTS), len(texts)))
    return index, encoder, secret_ids, filler_ids


def test_answer_span_helpers():
    """Pure-function semantics of answer-span extraction / containment / whether a reference is informative."""
    span = extract_answer_span(SECRET_QUERIES[0], SECRET_TEXTS[0])
    assert "quantum flux capacitors" in span
    assert 0 < len(tokenize(span)) <= 12
    assert extract_answer_span("totally unrelated wording here", SECRET_TEXTS[0]) == ""
    hit, run = answer_contained(
        tokenize("the reactor uses quantum flux capacitors to stabilise"),
        tokenize("uses quantum flux capacitors to stabilise the reactor core"), 6)
    assert hit and run >= 6
    miss, _ = answer_contained(tokenize("nothing relevant at all"), tokenize("quantum flux capacitors"), 6)
    assert not miss
    assert longest_common_token_run(["a", "b", "c"], ["x", "b", "c", "y"]) == 2
    assert longest_common_token_run([], ["b"]) == 0
    assert is_informative_reference("quantum flux capacitors")
    assert not is_informative_reference("Yes")
    assert not is_informative_reference("no")
    assert not is_informative_reference("a")


def test_elicitation_v2_detects_leak_without_gold_answers():
    """Core regression: when gold answers are empty (DS2 measured 1000/1000 empty), the criterion must still be discriminative.

    The old definition (empty answers -> f1=0 and keywords=[]) **always judged "not elicited"** in this
    situation, regardless of whether retrieval leaked. v2 extracts answer spans from tombstone evidence,
    so it can distinguish "reproduced" from "not reproduced".
    """
    index, encoder, secret_ids, _ = _build_tomb_index()
    index.remove(secret_ids)  # revoked documents get tombstones: vectors and raw text kept, retrieval invisible
    qvec = encoder.encode([SECRET_QUERIES[0]])[0]
    items = [{"query": SECRET_QUERIES[0], "answers": [], "qvec": qvec}]
    clean = ElicitationTest(_EchoGenerator("The cafeteria serves lentil soup."), index, k=3, encoder=encoder)
    rate_clean, details_clean = clean.run_detailed(items)
    assert rate_clean == 0.0
    assert details_clean[0]["informative"] is True          # reference comes from tombstone -> informative
    assert clean.last_summary["n_evaluated"] == 1
    leak = ElicitationTest(_EchoGenerator(SECRET_TEXTS[0]), index, k=3, encoder=encoder)
    rate_leak, details_leak = leak.run_detailed(items)
    assert rate_leak == pytest.approx(1.0)
    assert details_leak[0]["containment"] is True
    assert details_leak[0]["signal"] == "containment"


def test_elicitation_v2_backward_attribution_for_short_answers():
    """Ablation switch: reverse "answer attribution" signal (off by default, see verify.py; enabled here to test its hit behavior)."""
    index, encoder, secret_ids, _ = _build_tomb_index()
    index.remove(secret_ids)
    qvec = encoder.encode([SECRET_QUERIES[0]])[0]
    items = [{"query": SECRET_QUERIES[0], "answers": [], "qvec": qvec}]
    off = ElicitationTest(_EchoGenerator("quantum flux capacitors"), index, k=3, encoder=encoder)
    rate_off, _ = off.run_detailed(items)
    assert rate_off == 0.0                                   # reverse signal off by default
    test = ElicitationTest(_EchoGenerator("quantum flux capacitors"), index, k=3, encoder=encoder,
                           use_attribution=True)
    rate, details = test.run_detailed(items)
    assert rate == pytest.approx(1.0)
    assert details[0]["containment_mode"] in ("backward", "forward+backward")
    assert details[0]["attribution"] >= 0.6


def test_elicitation_v2_denominator_excludes_binary_references():
    """Items whose references are only binary yes/no answers are excluded from the denominator (binary-answer noise dominates the reading)."""
    index, encoder, secret_ids, _ = _build_index()
    items = [{"query": SECRET_QUERIES[0], "answers": ["Yes"]},
             {"query": SECRET_QUERIES[1], "answers": ["no"]}]
    test = ElicitationTest(_EchoGenerator("Yes, that is right."), index, k=3, encoder=encoder,
                           tombstone_spans=False)
    rate, details = test.run_detailed(items)
    assert rate == 0.0
    assert test.last_summary["n_evaluated"] == 0
    assert test.last_summary["n_uninformative"] == 2
    assert all(d["informative"] is False for d in details)
    assert all(d["n_keywords"] == 0 for d in details)


def test_elicitation_v2_gold_pid_spans_work_on_live_index():
    """Unrevoked index + gold_pids: references are answer spans from gold evidence (positive-capability control definition)."""
    index, encoder, secret_ids, _ = _build_index()
    qvec = encoder.encode([SECRET_QUERIES[0]])[0]
    items = [{"query": SECRET_QUERIES[0], "answers": [], "gold_pids": list(secret_ids), "qvec": qvec}]
    test = ElicitationTest(_EchoGenerator(SECRET_TEXTS[0]), index, k=3, encoder=encoder)
    rate, details = test.run_detailed(items)
    assert rate == pytest.approx(1.0)
    assert details[0]["reference_sources"]
    assert all(src == "evidence" for src in details[0]["reference_sources"])


def test_faithfulness_short_answer_is_not_structurally_zero():
    """Short answers + long evidence are no longer structurally scored 0 (the old definition used sentence-level F1 against whole-paragraph evidence)."""
    evidence = ["The first nuclear power plant opened in Obninsk in 1954 after four years of construction. It fed electricity into the grid."]
    assert faithfulness("Obninsk, 1954.", evidence) == pytest.approx(1.0)
    assert faithfulness("Bananas grow in the tropics.", evidence) == pytest.approx(0.0)
    assert faithfulness("The cat sat on the mat.", ["cat sat on the mat"]) == pytest.approx(1.0)
    assert faithfulness("Dogs fly high above the clouds.", ["cat sat on the mat"]) == pytest.approx(0.0)
    assert faithfulness("", evidence) == 0.0
    assert faithfulness("Obninsk, 1954.", []) == 0.0
    assert evidence_units(evidence)

