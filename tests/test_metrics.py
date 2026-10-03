"""tests/test_metrics.py -- hand-computed checks for metrics.py (pure synthetic data, no network).

All expected values can be verified by hand:
* recall@2 / nDCG@2 computed item-by-item on small id lists;
* AUC with four analytically tractable constructions: 1.0 / 0.0 / 0.5 / 0.125;
* F1 computed by hand on token intersections after SQuAD-style normalization.
"""

from __future__ import annotations

import math
import sys
import time
from pathlib import Path

import numpy as np
import pytest

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from fedrevoke.metrics import (  # noqa: E402
    CostMeter,
    exact_match,
    f1_score_tokens,
    fairness_stats,
    faithfulness,
    measure_bytes,
    mia_auc,
    ndcg_at_k,
    normalize_answer,
    recall_at_k,
    tokenize,
)


# --------------------------------------------------------------------------- #
# Retrieval metrics
# --------------------------------------------------------------------------- #
def test_recall_at_k_hand_computed():
    # retrieved = [3, 1, 7], gold = [1, 2]
    assert recall_at_k([3, 1, 7], [1, 2], k=1) == pytest.approx(0.0)      # top1=3 misses
    assert recall_at_k([3, 1, 7], [1, 2], k=2) == pytest.approx(0.5)      # top2=[3,1] hits 1 of 2 gold
    assert recall_at_k([3, 1, 7], [1, 2], k=3) == pytest.approx(0.5)
    assert recall_at_k([3, 1, 7], [1, 2], k=100) == pytest.approx(0.5)    # safe when k > len


def test_recall_at_k_dedupes_repeated_ids():
    # the same hit repeated 7 times must not inflate recall
    assert recall_at_k([5, 5, 5, 5], [5, 6], k=3) == pytest.approx(0.5)
    assert recall_at_k([5, 5, 9], [5], k=2) == pytest.approx(1.0)


def test_recall_at_k_edge_cases():
    assert recall_at_k([], [], k=10) == 0.0
    assert recall_at_k([1, 2], [], k=10) == 0.0
    assert recall_at_k([], [1], k=10) == 0.0
    assert recall_at_k([1, 2], [1], k=0) == 0.0
    assert recall_at_k([1, 2], [1], k=-3) == 0.0
    assert recall_at_k(np.array([1.0, np.nan, 2.0]), [2], k=3) == pytest.approx(1.0)
    assert recall_at_k(None, [1], k=10) == 0.0
    assert recall_at_k([1], None, k=10) == 0.0


def test_ndcg_at_k_hand_computed():
    # retrieved=[1,2,3], gold={1,3}
    # DCG  = 1/log2(2) + 0 + 1/log2(4) = 1 + 0.5 = 1.5
    # IDCG = 1/log2(2) + 1/log2(3)     = 1.6309297535714575
    expected = 1.5 / (1.0 + 1.0 / math.log2(3.0))
    assert ndcg_at_k([1, 2, 3], [1, 3], k=3) == pytest.approx(expected, rel=1e-9)
    assert ndcg_at_k([1, 2, 3], [1, 3], k=3) == pytest.approx(0.9197207891, rel=1e-9)
    # perfect ranking -> 1.0
    assert ndcg_at_k([1, 3, 9], [1, 3], k=3) == pytest.approx(1.0)
    # hits at rank 2 -> discounted
    assert ndcg_at_k([9, 1, 3], [1, 3], k=3) == pytest.approx((1.0 / math.log2(3.0) + 1.0 / math.log2(4.0)) / expected_ideal(), rel=1e-9)
    # edge cases
    assert ndcg_at_k([1], [], k=5) == 0.0
    assert ndcg_at_k([2], [1], k=5) == 0.0
    assert ndcg_at_k([1], [1], k=0) == 0.0


def expected_ideal():
    return 1.0 + 1.0 / math.log2(3.0)


# --------------------------------------------------------------------------- #
# Generation metrics
# --------------------------------------------------------------------------- #
def test_normalize_and_tokenize():
    assert normalize_answer("The  CAT, sat!") == "cat sat"
    assert tokenize("A dog.") == ["dog"]
    assert normalize_answer(None) == ""
    assert tokenize("") == []


def test_exact_match():
    assert exact_match("The Cat.", ["the cat"]) == 1.0
    assert exact_match("the  cat", ["a cat"]) == 1.0
    assert exact_match("cat", ["dog"]) == 0.0
    assert exact_match("", ["cat"]) == 0.0
    assert exact_match("cat", []) == 0.0
    assert exact_match(None, ["cat"]) == 0.0


def test_f1_score_tokens_hand_computed():
    # pred tokens = [cat, sat] ; gold tokens = [cat, sat, on, mat]
    # P = 2/2 = 1.0, R = 2/4 = 0.5 -> F1 = 2/3
    assert f1_score_tokens("the cat sat", ["cat sat on the mat"]) == pytest.approx(2.0 / 3.0)
    assert f1_score_tokens("cat", ["cat"]) == pytest.approx(1.0)
    # take the max over multiple golds
    assert f1_score_tokens("blue car", ["a red car", "blue car"]) == pytest.approx(1.0)
    # P = 2/3, R = 1.0 -> F1 = 2 * (2/3) / (5/3) = 0.8
    assert f1_score_tokens("red car fast", ["red car"]) == pytest.approx(0.8)
    # edge cases
    assert f1_score_tokens("", ["cat"]) == 0.0
    assert f1_score_tokens("cat", []) == 0.0
    assert f1_score_tokens("dog", ["cat"]) == 0.0


def test_faithfulness_is_fraction_of_supported_sentences():
    evidence = ["cat sat on the mat"]
    assert faithfulness("The cat sat on the mat.", evidence) == pytest.approx(1.0)
    assert faithfulness("Dogs fly high above the clouds.", evidence) == pytest.approx(0.0)
    assert faithfulness("", evidence) == 0.0
    assert faithfulness("The cat sat on the mat.", []) == 0.0


# --------------------------------------------------------------------------- #
# MIA AUC
# --------------------------------------------------------------------------- #
def test_mia_auc_perfect_and_chance():
    assert mia_auc([0.9, 0.8], [0.1, 0.2]) == pytest.approx(1.0)
    assert mia_auc([0.1, 0.2], [0.9, 0.8]) == pytest.approx(0.0)
    assert mia_auc([0.5], [0.5]) == pytest.approx(0.5)          # all ties
    assert mia_auc([1.0, 2.0, 3.0], [4.0, 5.0, 6.0]) == pytest.approx(0.0)


def test_mia_auc_partial_ties():
    # pos=[1,2], neg=[2,3] -> 4 pairs: 0 + 0 + 0.5(tie) + 0 = 0.125
    assert mia_auc([1.0, 2.0], [2.0, 3.0]) == pytest.approx(0.125)


def test_mia_auc_edge_cases():
    assert mia_auc([], [1.0]) == pytest.approx(0.5)             # no information
    assert mia_auc([1.0], []) == pytest.approx(0.5)
    assert mia_auc([], []) == pytest.approx(0.5)
    assert mia_auc([np.nan, 0.9], [0.1]) == pytest.approx(1.0)  # NaN is dropped
    assert mia_auc([np.inf], [1.0]) == pytest.approx(0.5)       # empty after dropping inf


# --------------------------------------------------------------------------- #
# Fairness / cost
# --------------------------------------------------------------------------- #
def test_fairness_stats():
    stats = fairness_stats([0.5, 0.7, 0.9])
    assert stats["min"] == pytest.approx(0.5)
    assert stats["max"] == pytest.approx(0.9)
    assert stats["mean"] == pytest.approx(0.7)
    assert stats["std"] == pytest.approx(math.sqrt(0.02666666666666667))
    assert stats["n"] == 3
    empty = fairness_stats([])
    assert empty == {"min": 0.0, "max": 0.0, "mean": 0.0, "std": 0.0, "n": 0}
    per_client = fairness_stats({"c0": 0.8, "c1": 0.6})
    assert per_client["min"] == pytest.approx(0.6)


def test_measure_bytes():
    assert measure_bytes(np.zeros((2, 3), dtype=np.float32)) == 24
    assert measure_bytes(np.zeros((2, 3), dtype=np.float64)) == 48
    assert measure_bytes(b"abc") == 3
    assert measure_bytes([np.zeros(4, dtype=np.float32), b"ab"]) == 16 + 2
    assert measure_bytes(None) == 0


def test_cost_meter_records_wall_time_bytes_and_touch():
    with CostMeter("unit") as meter:
        time.sleep(0.002)
        meter.add_bytes(1024)
        meter.record(np.zeros((3, 3), dtype=np.float32))
        meter.touch(7)
        meter.touch()
    payload = meter.as_dict()
    assert payload["bytes_transferred"] == 1024 + 36
    assert payload["n_vectors_touched"] == 8
    assert payload["wall_time"] > 0.0
    assert payload["wall_time"] == payload["wall_time_s"]
    assert payload["peak_vram_mb"] >= 0.0
    assert isinstance(payload["vram_available"], bool)
    assert set(["wall_time", "wall_time_s", "peak_vram_mb", "bytes_transferred", "n_vectors_touched"]) <= set(payload)
    assert "CostMeter(" in repr(meter)


def test_cost_meter_nested_does_not_crash():
    outer = CostMeter("outer")
    with outer:
        with CostMeter("inner") as inner:
            inner.add_bytes(8)
    assert inner.bytes_transferred == 8
    assert outer.bytes_transferred == 0
    assert outer.wall_time >= inner.wall_time >= 0.0
