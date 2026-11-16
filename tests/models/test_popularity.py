"""Popularity baseline: ranking, tie-break, seen filtering, padding, leakage guard."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from driftloop.data.schema import empty_interactions, to_canonical
from driftloop.models.base import PAD_ITEM, Recommender, SeenItems
from driftloop.models.popularity import PopularityRecommender

P = PAD_ITEM


def _frame(rows: list[tuple[int, int, int, float]]) -> pl.DataFrame:
    users, items, ts, weights = zip(*rows, strict=True)
    return to_canonical(
        pl.DataFrame({"user_id": users, "item_id": items, "ts": ts, "weight": weights})
    )


# Summed weights: item 30 -> 3.0, item 20 -> 2.0, item 10 -> 2.0, item 40 -> 0.5.
TRAIN = _frame(
    [
        (1, 30, 0, 1.0),
        (2, 30, 1, 1.0),
        (3, 30, 2, 1.0),
        (1, 20, 3, 2.0),
        (2, 10, 4, 1.0),
        (3, 10, 5, 1.0),
        (1, 40, 6, 0.5),
    ]
)


def _fitted() -> PopularityRecommender:
    return PopularityRecommender().fit(TRAIN, until=7)


def test_satisfies_protocol() -> None:
    model = PopularityRecommender()
    assert isinstance(model, Recommender)
    assert model.name == "popularity"
    assert model.trained_until is None


def test_orders_by_summed_weight_with_id_tie_break() -> None:
    model = _fitted()
    assert model.trained_until == 7
    recs = model.recommend(np.array([1]), 4)
    # 10 and 20 tie at 2.0; the smaller id wins. Weight, not count, decides 20 vs 40.
    np.testing.assert_array_equal(recs, [[30, 10, 20, 40]])
    assert recs.dtype == np.int64


def test_tie_break_is_independent_of_row_order() -> None:
    rows = [(1, 5, 0, 1.0), (1, 3, 1, 1.0), (1, 9, 2, 1.0)]
    a = PopularityRecommender().fit(_frame(rows), until=3).recommend(np.array([0]), 3)
    b = PopularityRecommender().fit(_frame(rows[::-1]), until=3).recommend(np.array([0]), 3)
    np.testing.assert_array_equal(a, [[3, 5, 9]])
    np.testing.assert_array_equal(a, b)


def test_seen_items_are_filtered_per_user() -> None:
    model = _fitted()
    seen = SeenItems({1: np.array([30, 20]), 2: np.array([10, 999])})
    recs = model.recommend(np.array([1, 2, 3]), 2, seen=seen)
    np.testing.assert_array_equal(recs, [[10, 40], [30, 20], [30, 10]])


def test_unknown_user_gets_global_list() -> None:
    model = _fitted()
    seen = SeenItems({1: np.array([30])})
    recs = model.recommend(np.array([12345]), 3, seen=seen)
    np.testing.assert_array_equal(recs, [[30, 10, 20]])


def test_pads_when_catalogue_runs_out() -> None:
    model = _fitted()
    np.testing.assert_array_equal(model.recommend(np.array([1]), 6), [[30, 10, 20, 40, P, P]])
    seen = SeenItems({1: np.array([30, 10, 20])})
    np.testing.assert_array_equal(model.recommend(np.array([1]), 3, seen=seen), [[40, P, P]])


def test_padding_not_leaked_between_users() -> None:
    model = _fitted()
    seen = SeenItems({1: np.array([30, 10, 20, 40])})
    recs = model.recommend(np.array([1, 2]), 2, seen=seen)
    np.testing.assert_array_equal(recs, [[P, P], [30, 10]])


def test_empty_training_data_pads_everything() -> None:
    model = PopularityRecommender().fit(empty_interactions(), until=0)
    np.testing.assert_array_equal(model.recommend(np.array([1, 2]), 2), [[P, P], [P, P]])


def test_no_users() -> None:
    assert _fitted().recommend(np.array([], dtype=np.int64), 3).shape == (0, 3)


@pytest.mark.parametrize("until", [6, 0])
def test_rejects_rows_at_or_after_until(until: int) -> None:
    model = PopularityRecommender()
    with pytest.raises(ValueError, match="until"):
        model.fit(TRAIN, until=until)
    assert model.trained_until is None


def test_recommend_before_fit_raises() -> None:
    with pytest.raises(RuntimeError):
        PopularityRecommender().recommend(np.array([1]), 3)


def test_invalid_k_raises() -> None:
    with pytest.raises(ValueError, match="k must be"):
        _fitted().recommend(np.array([1]), 0)


def test_refit_replaces_ranking() -> None:
    model = _fitted()
    model.fit(_frame([(1, 77, 10, 1.0)]), until=11)
    assert model.trained_until == 11
    np.testing.assert_array_equal(model.recommend(np.array([1]), 2), [[77, P]])
