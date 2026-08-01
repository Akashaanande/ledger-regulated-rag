import math

import pytest

from ledger.eval.metrics import (
    mean_ndcg_at_k,
    mean_recall_at_k,
    mean_reciprocal_rank,
    ndcg_at_k,
    recall_at_k,
    reciprocal_rank,
)


def test_recall_at_k_partial_hit():
    retrieved = ["a", "b", "c", "d"]
    relevant = {"b", "z"}
    assert recall_at_k(retrieved, relevant, k=3) == pytest.approx(0.5)


def test_recall_at_k_all_hit():
    retrieved = ["a", "b", "c"]
    relevant = {"a", "b"}
    assert recall_at_k(retrieved, relevant, k=3) == pytest.approx(1.0)


def test_recall_at_k_empty_relevant_raises():
    with pytest.raises(ValueError, match="non-empty"):
        recall_at_k(["a"], set(), k=1)


def test_reciprocal_rank_first_position():
    assert reciprocal_rank(["a", "b"], {"a"}) == pytest.approx(1.0)


def test_reciprocal_rank_third_position():
    assert reciprocal_rank(["x", "y", "a"], {"a"}) == pytest.approx(1 / 3)


def test_reciprocal_rank_no_hit():
    assert reciprocal_rank(["x", "y"], {"a"}) == pytest.approx(0.0)


def test_ndcg_perfect_ranking_is_one():
    retrieved = ["a", "b", "c"]
    relevant = {"a", "b"}
    assert ndcg_at_k(retrieved, relevant, k=3) == pytest.approx(1.0)


def test_ndcg_worst_ranking_below_one():
    retrieved = ["c", "d", "a"]
    relevant = {"a"}
    expected = (1.0 / math.log2(4)) / (1.0 / math.log2(2))
    assert ndcg_at_k(retrieved, relevant, k=3) == pytest.approx(expected)


def test_ndcg_no_hits_is_zero():
    assert ndcg_at_k(["x", "y"], {"a"}, k=2) == pytest.approx(0.0)


def test_mean_helpers_average_across_queries():
    runs = [
        (["a", "b"], {"a"}),
        (["x", "y"], {"y"}),
    ]
    assert mean_recall_at_k(runs, k=1) == pytest.approx(0.5)
    assert mean_reciprocal_rank(runs) == pytest.approx((1.0 + 0.5) / 2)
    assert mean_ndcg_at_k(runs, k=1) == pytest.approx(0.5)


def test_mean_helpers_empty_runs_is_zero():
    assert mean_recall_at_k([], k=5) == 0.0
    assert mean_reciprocal_rank([]) == 0.0
    assert mean_ndcg_at_k([], k=5) == 0.0
