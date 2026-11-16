"""Tests for the synthetic generator: contract, determinism and the drift dial.

The stationarity test matters most: downstream, ``drift_strength = 0`` is the
control experiment in which a frozen model must not decay, so the generator has to
really switch every source of drift off.
"""

from __future__ import annotations

from dataclasses import replace

import polars as pl
import pytest

from driftloop.data.schema import INTERACTION_SCHEMA, ITEM, TS, USER, validate_interactions
from driftloop.data.synthetic import SyntheticConfig, generate_interactions
from driftloop.replay.clock import DAY

SMALL = SyntheticConfig(
    n_users=400,
    n_new_users_per_day=2.0,
    n_items_initial=300,
    n_new_items_per_day=10.0,
    n_clusters=10,
    n_days=60,
    seed=0,
)
THIRD = SMALL.n_days * DAY // 3


def _max(df: pl.DataFrame | pl.Series, col: str = "") -> int:
    series = df if isinstance(df, pl.Series) else df.get_column(col)
    value = series.max()
    assert isinstance(value, int)
    return value


def _min(df: pl.DataFrame | pl.Series, col: str = "") -> int:
    series = df if isinstance(df, pl.Series) else df.get_column(col)
    value = series.min()
    assert isinstance(value, int)
    return value


def _thirds(df: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    return df.filter(pl.col(TS) < THIRD), df.filter(pl.col(TS) >= 2 * THIRD)


def _item_share(df: pl.DataFrame) -> pl.DataFrame:
    return df.group_by(ITEM).len().select(ITEM, (pl.col("len") / df.height).alias("p"))


def _total_variation(a: pl.DataFrame, b: pl.DataFrame) -> float:
    joined = _item_share(a).join(_item_share(b), on=ITEM, how="full", coalesce=True)
    diff = joined.fill_null(0.0).select((pl.col("p") - pl.col("p_right")).abs().sum() / 2)
    return float(diff.item())


def _late_unseen_share(df: pl.DataFrame) -> float:
    first, last = _thirds(df)
    seen = first.get_column(ITEM).unique().implode()
    return last.filter(~pl.col(ITEM).is_in(seen)).height / last.height


@pytest.fixture(scope="module")
def stationary() -> pl.DataFrame:
    return generate_interactions(replace(SMALL, drift_strength=0.0))


@pytest.fixture(scope="module")
def drifting() -> pl.DataFrame:
    return generate_interactions(SMALL)


def test_output_is_canonical(drifting: pl.DataFrame) -> None:
    validate_interactions(drifting)
    assert dict(drifting.schema) == INTERACTION_SCHEMA
    assert drifting.height > 0
    assert _min(drifting, TS) == pytest.approx(0, abs=DAY)
    assert _max(drifting, TS) < SMALL.n_days * DAY


def test_events_spread_within_each_day(drifting: pl.DataFrame) -> None:
    seconds_of_day = drifting.get_column(TS) % DAY
    # Not all at midnight: uniform within the day spans most of it.
    assert _min(seconds_of_day) < DAY // 10
    assert _max(seconds_of_day) > DAY - DAY // 10


def test_event_volume_matches_rate(stationary: pl.DataFrame) -> None:
    expected = SMALL.n_users * SMALL.n_days * SMALL.events_per_user_per_day
    assert stationary.height == pytest.approx(expected, rel=0.05)


def test_deterministic_given_seed() -> None:
    a = generate_interactions(SMALL)
    b = generate_interactions(SMALL)
    assert a.equals(b)


def test_different_seeds_differ(drifting: pl.DataFrame) -> None:
    other = generate_interactions(replace(SMALL, seed=1))
    assert not drifting.equals(other)


def test_stationary_has_no_new_users_or_items(stationary: pl.DataFrame) -> None:
    assert _max(stationary, USER) < SMALL.n_users
    assert _max(stationary, ITEM) < SMALL.n_items_initial


def test_stationary_items_all_appear_early(stationary: pl.DataFrame) -> None:
    first_day = stationary.group_by(ITEM).agg((pl.col(TS).min() // DAY).alias("day"))
    assert _max(first_day, "day") <= 7


def test_stationary_item_popularity_is_stable(stationary: pl.DataFrame) -> None:
    first, last = _thirds(stationary)
    # The residual is sampling noise from ~24k events per third over 300 items.
    assert _total_variation(first, last) < 0.1


def test_drift_moves_item_popularity(stationary: pl.DataFrame, drifting: pl.DataFrame) -> None:
    assert _total_variation(*_thirds(drifting)) > 3 * _total_variation(*_thirds(stationary))


def test_drift_brings_new_items_and_users(drifting: pl.DataFrame) -> None:
    assert _max(drifting, ITEM) >= SMALL.n_items_initial
    assert _max(drifting, USER) >= SMALL.n_users


def test_drift_raises_late_share_on_unseen_items(
    stationary: pl.DataFrame, drifting: pl.DataFrame
) -> None:
    assert _late_unseen_share(stationary) < 0.01
    assert _late_unseen_share(drifting) > 0.2


def test_drift_strength_is_monotone_in_unseen_share(drifting: pl.DataFrame) -> None:
    half = generate_interactions(replace(SMALL, drift_strength=0.5))
    assert _late_unseen_share(half) < _late_unseen_share(drifting)


def test_single_day_works() -> None:
    df = generate_interactions(replace(SMALL, n_days=1))
    assert _max(df, TS) < DAY
    assert _max(df, USER) < SMALL.n_users


@pytest.mark.parametrize(
    "overrides",
    [
        {"n_users": 0},
        {"n_items_initial": -1},
        {"n_clusters": 0},
        {"n_days": 0},
        {"n_items_initial": 5, "n_clusters": 10},
        {"events_per_user_per_day": 0.0},
        {"n_new_users_per_day": -1.0},
        {"n_new_items_per_day": -0.5},
        {"drift_strength": -0.1},
        {"drift_strength": float("nan")},
        {"preference_concentration": 0.0},
        {"freshness_decay_days": 0.0},
        {"freshness_boost": -1.0},
        {"trend_volatility": -0.1},
        {"taste_volatility": -0.1},
        {"item_popularity_exponent": -1.0},
    ],
)
def test_invalid_config_rejected(overrides: dict[str, float]) -> None:
    with pytest.raises(ValueError, match="must be"):
        SyntheticConfig(**overrides)  # type: ignore[arg-type]
