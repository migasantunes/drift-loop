"""Ground truth construction and window scoring."""

from __future__ import annotations

import math
from typing import Self

import numpy as np
import polars as pl
import pytest

from driftloop.data.schema import empty_interactions, to_canonical
from driftloop.eval.protocol import WindowResult, build_ground_truth, evaluate_window
from driftloop.models.base import PAD_ITEM, Recommender, SeenItems


def _frame(rows: list[tuple[int, int, int]]) -> pl.DataFrame:
    if not rows:
        return empty_interactions()
    users, items, ts = zip(*rows, strict=True)
    return to_canonical(pl.DataFrame({"user_id": users, "item_id": items, "ts": ts}))


HISTORY = _frame([(1, 10, 0), (1, 11, 1), (2, 10, 2), (2, 12, 3)])
FUTURE = _frame(
    [
        (1, 10, 10),  # repeat for user 1: excluded
        (1, 13, 11),
        (1, 13, 12),  # duplicate in future: collapsed
        (1, 12, 13),  # user 2 saw 12, user 1 did not: new for user 1
        (2, 12, 14),  # user 2 only repeats: dropped
        (3, 10, 15),  # user absent from history: kept
    ]
)


class RecordingModel:
    """Returns fixed lists and records exactly what the harness passed in."""

    name = "recording"

    def __init__(self, lists: dict[int, list[int]]) -> None:
        self.trained_until: int | None = 5
        self.lists = lists
        self.calls: list[tuple[np.ndarray, int, SeenItems | None]] = []

    def fit(self, interactions: pl.DataFrame, *, until: int) -> Self:
        raise AssertionError("evaluate_window must not fit the model")

    def recommend(
        self, user_ids: np.ndarray, k: int, *, seen: SeenItems | None = None
    ) -> np.ndarray:
        self.calls.append((user_ids.copy(), k, seen))
        out = np.full((user_ids.size, k), PAD_ITEM, dtype=np.int64)
        for row, u in enumerate(user_ids.tolist()):
            items = self.lists.get(u, [])[:k]
            out[row, : len(items)] = items
        return out


def test_ground_truth_rules() -> None:
    users, truth = build_ground_truth(HISTORY, FUTURE)
    np.testing.assert_array_equal(users, [1, 3])
    assert len(truth) == 2
    np.testing.assert_array_equal(truth[0], [12, 13])
    np.testing.assert_array_equal(truth[1], [10])
    assert users.dtype == np.int64


def test_ground_truth_users_sorted() -> None:
    future = _frame([(9, 1, 10), (4, 2, 11), (7, 3, 12)])
    users, truth = build_ground_truth(HISTORY, future)
    np.testing.assert_array_equal(users, [4, 7, 9])
    np.testing.assert_array_equal(np.concatenate(truth), [2, 3, 1])


def test_ground_truth_empty_future() -> None:
    users, truth = build_ground_truth(HISTORY, empty_interactions())
    assert users.size == 0
    assert truth == []


def test_evaluate_window_scores_and_isolates_future() -> None:
    model = RecordingModel({1: [13, 99, 98], 3: [55, 10, 54]})
    assert isinstance(model, Recommender)
    result = evaluate_window(model, history=HISTORY, future=FUTURE, k=3)

    assert len(model.calls) == 1
    user_ids, k, seen = model.calls[0]
    np.testing.assert_array_equal(user_ids, [1, 3])
    assert k == 3
    # seen comes from history only: future items (13 for user 1, 10 for user 3) absent.
    assert isinstance(seen, SeenItems)
    assert sorted(seen.get(1).tolist()) == [10, 11]
    assert sorted(seen.get(2).tolist()) == [10, 12]
    assert seen.get(3).size == 0
    assert len(seen) == 2

    d1 = 1.0 / math.log2(3)
    # User 1: truth {12, 13}, hit at rank 0. User 3: truth {10}, hit at rank 1.
    assert result.k == 3
    assert result.n_users == 2
    assert result.recall == pytest.approx((0.5 + 1.0) / 2)
    assert result.hit_rate == pytest.approx(1.0)
    assert result.ndcg == pytest.approx((1.0 / (1.0 + d1) + d1) / 2)
    # Distinct recommended items {13, 99, 98, 55, 10, 54} over 3 items in history.
    assert result.coverage == pytest.approx(6 / 3)


def test_no_future_row_reaches_model() -> None:
    """A model fed future rows through ``seen`` would see a future-only item."""
    future = _frame([(1, 777, 20)])
    model = RecordingModel({1: [777]})
    result = evaluate_window(model, history=HISTORY, future=future, k=1)
    _, _, seen = model.calls[0]
    assert seen is not None
    for user in (1, 2, 3):
        assert 777 not in seen.get(user).tolist()
    assert result.recall == 1.0


def test_empty_future_gives_nan_without_calling_model() -> None:
    model = RecordingModel({})
    result = evaluate_window(model, history=HISTORY, future=empty_interactions(), k=5)
    assert model.calls == []
    assert result.n_users == 0
    assert result.k == 5
    for value in (result.recall, result.ndcg, result.hit_rate, result.coverage):
        assert math.isnan(value)


def test_only_repeats_gives_nan() -> None:
    future = _frame([(1, 10, 20), (2, 12, 21)])
    result = evaluate_window(RecordingModel({}), history=HISTORY, future=future, k=2)
    assert result.n_users == 0
    assert math.isnan(result.recall)


def test_empty_history_coverage_is_nan() -> None:
    future = _frame([(1, 5, 20)])
    result = evaluate_window(
        RecordingModel({1: [5]}), history=empty_interactions(), future=future, k=1
    )
    assert result.n_users == 1
    assert result.recall == 1.0
    assert math.isnan(result.coverage)


def test_wrong_shape_from_model_raises() -> None:
    class Short(RecordingModel):
        def recommend(
            self, user_ids: np.ndarray, k: int, *, seen: SeenItems | None = None
        ) -> np.ndarray:
            return np.zeros((user_ids.size, k - 1), dtype=np.int64)

    with pytest.raises(ValueError, match="shape"):
        evaluate_window(Short({}), history=HISTORY, future=FUTURE, k=3)


def test_invalid_k_raises() -> None:
    with pytest.raises(ValueError, match="k must be"):
        evaluate_window(RecordingModel({}), history=HISTORY, future=FUTURE, k=0)


def test_window_result_is_frozen() -> None:
    result = WindowResult(k=1, n_users=1, recall=1.0, ndcg=1.0, hit_rate=1.0, coverage=1.0)
    with pytest.raises(AttributeError):
        result.k = 2  # type: ignore[misc]
