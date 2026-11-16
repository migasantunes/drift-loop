"""Metrics checked against values worked out by hand, plus a brute-force reference."""

from __future__ import annotations

import math

import numpy as np
import pytest

from driftloop.eval.metrics import coverage, hit_rate_at_k, ndcg_at_k, recall_at_k
from driftloop.models.base import PAD_ITEM

P = PAD_ITEM
# Discount of rank r (0-based) is 1 / log2(r + 2).
D0, D1, D2 = 1.0, 1.0 / math.log2(3), 1.0 / math.log2(4)


def _arr(*items: int) -> np.ndarray:
    return np.array(items, dtype=np.int64)


def test_hand_computed_mixed_users() -> None:
    recs = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.int64)
    truth = [_arr(2, 99), _arr(100), _arr(9, 7)]
    np.testing.assert_allclose(recall_at_k(recs, truth), [0.5, 0.0, 1.0])
    np.testing.assert_allclose(hit_rate_at_k(recs, truth), [1.0, 0.0, 1.0])
    # User 0: hit at rank 1, ideal is two hits at ranks 0 and 1.
    # User 2: hits at ranks 0 and 2, ideal ranks 0 and 1.
    expected = [D1 / (D0 + D1), 0.0, (D0 + D2) / (D0 + D1)]
    np.testing.assert_allclose(ndcg_at_k(recs, truth), expected)
    assert expected[0] == pytest.approx(0.3868528, abs=1e-6)


def test_perfect_and_zero_rankings() -> None:
    recs = np.array([[3, 1, 2], [4, 5, 6]], dtype=np.int64)
    truth = [_arr(1, 2, 3), _arr(7, 8)]
    np.testing.assert_allclose(recall_at_k(recs, truth), [1.0, 0.0])
    np.testing.assert_allclose(ndcg_at_k(recs, truth), [1.0, 0.0])
    np.testing.assert_allclose(hit_rate_at_k(recs, truth), [1.0, 0.0])


def test_more_relevant_items_than_k() -> None:
    recs = np.array([[1, 2], [9, 1]], dtype=np.int64)
    truth = [_arr(1, 2, 3, 4), _arr(1, 2, 3)]
    # Recall divides by |truth|, so it cannot reach 1 here.
    np.testing.assert_allclose(recall_at_k(recs, truth), [0.5, 1 / 3])
    # NDCG's ideal is min(k, |truth|) = 2 hits, so a fully relevant list scores 1.
    np.testing.assert_allclose(ndcg_at_k(recs, truth), [1.0, D1 / (D0 + D1)])


def test_padding_never_counts() -> None:
    recs = np.array([[5, P, P], [P, P, P], [P, P, P]], dtype=np.int64)
    truth = [_arr(5, 6), _arr(1), _arr(0)]
    np.testing.assert_allclose(recall_at_k(recs, truth), [0.5, 0.0, 0.0])
    np.testing.assert_allclose(ndcg_at_k(recs, truth), [D0 / (D0 + D1), 0.0, 0.0])
    np.testing.assert_allclose(hit_rate_at_k(recs, truth), [1.0, 0.0, 0.0])


def test_padded_row_with_single_truth_scores_one() -> None:
    recs = np.array([[5, P, P]], dtype=np.int64)
    np.testing.assert_allclose(ndcg_at_k(recs, [_arr(5)]), [1.0])


def test_duplicate_truth_ids_collapse() -> None:
    recs = np.array([[1, 3]], dtype=np.int64)
    truth = [_arr(1, 1, 2)]
    np.testing.assert_allclose(recall_at_k(recs, truth), [0.5])
    np.testing.assert_allclose(ndcg_at_k(recs, truth), [1.0 / (D0 + D1)])


def test_items_are_matched_per_user_only() -> None:
    # Item 2 is relevant for user 1, recommended to user 0: must not count.
    recs = np.array([[2], [3]], dtype=np.int64)
    truth = [_arr(1), _arr(2)]
    np.testing.assert_allclose(recall_at_k(recs, truth), [0.0, 0.0])


def test_returns_float64_per_user() -> None:
    recs = np.array([[1], [2]], dtype=np.int32)
    out = recall_at_k(recs, [_arr(1), _arr(1)])
    assert out.dtype == np.float64
    assert out.shape == (2,)


def test_no_users() -> None:
    recs = np.empty((0, 5), dtype=np.int64)
    for fn in (recall_at_k, ndcg_at_k, hit_rate_at_k):
        assert fn(recs, []).shape == (0,)


@pytest.mark.parametrize(
    ("recs", "truth", "match"),
    [
        (np.array([[1, 2]]), [_arr(1), _arr(2)], "2 users but recs has 1 rows"),
        (np.array([1, 2]), [_arr(1)], "2-D"),
        (np.empty((1, 0), dtype=np.int64), [_arr(1)], "k >= 1"),
        (np.array([[1, 2]]), [np.empty(0, dtype=np.int64)], "empty"),
        (np.array([[1, 2]]), [np.array([[1]])], "1-D"),
        (np.array([[1, 1]]), [_arr(1)], "duplicate"),
        (np.array([[1, -5]]), [_arr(1)], "negative ids"),
        (np.array([[1, 2]]), [_arr(-1)], "non-negative"),
        (np.array([[1.0, 2.0]]), [_arr(1)], "integer"),
    ],
)
def test_invalid_inputs_raise(recs: np.ndarray, truth: list[np.ndarray], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        recall_at_k(recs, truth)


def test_matches_brute_force_reference() -> None:
    rng = np.random.default_rng(0)
    n_users, k, n_items = 200, 10, 60
    recs = np.full((n_users, k), P, dtype=np.int64)
    truth = []
    for u in range(n_users):
        n_recs = int(rng.integers(0, k + 1))
        recs[u, :n_recs] = rng.choice(n_items, size=n_recs, replace=False)
        truth.append(rng.integers(0, n_items, size=int(rng.integers(1, 25))))

    exp_recall, exp_ndcg, exp_hit = [], [], []
    for u in range(n_users):
        rel = set(truth[u].tolist())
        gains = [1.0 if (i != P and i in rel) else 0.0 for i in recs[u].tolist()]
        dcg = sum(g / math.log2(r + 2) for r, g in enumerate(gains))
        idcg = sum(1.0 / math.log2(r + 2) for r in range(min(k, len(rel))))
        exp_recall.append(sum(gains) / len(rel))
        exp_ndcg.append(dcg / idcg)
        exp_hit.append(float(any(gains)))

    np.testing.assert_allclose(recall_at_k(recs, truth), exp_recall)
    np.testing.assert_allclose(ndcg_at_k(recs, truth), exp_ndcg)
    np.testing.assert_allclose(hit_rate_at_k(recs, truth), exp_hit)


def test_coverage_counts_distinct_non_pad_items() -> None:
    recs = np.array([[1, 2, P], [2, 3, P]], dtype=np.int64)
    assert coverage(recs, 10) == pytest.approx(0.3)
    assert coverage(np.full((2, 3), P, dtype=np.int64), 10) == 0.0


def test_coverage_rejects_bad_inputs() -> None:
    with pytest.raises(ValueError, match="n_items_available"):
        coverage(np.array([[1]]), 0)
    with pytest.raises(ValueError, match="2-D"):
        coverage(np.array([1, 2]), 5)
