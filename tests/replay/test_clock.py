"""Tests for the time-slicing functions, where the no-leakage rule lives."""

from __future__ import annotations

import itertools

import polars as pl
import pytest

from driftloop.data.schema import TS, to_canonical
from driftloop.replay.clock import Window, history_before, make_windows, window_slice


def _events(ts: list[int]) -> pl.DataFrame:
    n = len(ts)
    return to_canonical(pl.DataFrame({"user_id": [0] * n, "item_id": list(range(n)), TS: ts}))


def test_windows_are_half_open_and_consecutive() -> None:
    windows = make_windows(0, 99, warmup=40, step=20)
    assert [(w.index, w.start, w.end) for w in windows] == [(0, 40, 60), (1, 60, 80), (2, 80, 100)]
    for prev, nxt in itertools.pairwise(windows):
        assert prev.end == nxt.start
    assert all(w.length == 20 for w in windows)


def test_trailing_partial_window_is_dropped() -> None:
    windows = make_windows(0, 97, warmup=40, step=20)
    assert [w.start for w in windows] == [40, 60]


def test_inclusive_t_max_edge() -> None:
    # t_max is the last event's timestamp, so [80, 100) is complete when t_max == 99
    # and incomplete when t_max == 98.
    assert make_windows(0, 99, warmup=40, step=20)[-1].end == 100
    assert make_windows(0, 98, warmup=40, step=20)[-1].end == 80


def test_windows_respect_t_min_offset() -> None:
    windows = make_windows(1000, 1099, warmup=40, step=20)
    assert windows[0].start == 1040
    assert windows[-1].end == 1100


def test_no_windows_when_warmup_covers_everything() -> None:
    assert make_windows(0, 50, warmup=100, step=10) == []


@pytest.mark.parametrize(
    ("t_min", "t_max", "warmup", "step"),
    [(0, 10, 0, 1), (0, 10, 1, 0), (0, 10, -1, 1), (10, 0, 1, 1)],
)
def test_make_windows_rejects_invalid_arguments(
    t_min: int, t_max: int, warmup: int, step: int
) -> None:
    with pytest.raises(ValueError, match="must be"):
        make_windows(t_min, t_max, warmup=warmup, step=step)


def test_empty_window_rejected() -> None:
    with pytest.raises(ValueError, match="empty"):
        Window(index=0, start=5, end=5)


def test_history_before_excludes_row_exactly_at_t() -> None:
    df = _events([0, 9, 10, 10, 11])
    out = history_before(df, 10)
    assert out.get_column(TS).to_list() == [0, 9]


@pytest.mark.parametrize("t", [0, 1, 5, 10, 11, 12, 100])
def test_history_before_never_returns_future(t: int) -> None:
    df = _events([0, 1, 5, 5, 10, 11])
    out = history_before(df, t)
    assert (out.get_column(TS) < t).all()
    # Nothing visible is lost either: every row with ts < t is returned.
    assert out.height == df.filter(pl.col(TS) < t).height


def test_history_before_lookback_bounds() -> None:
    df = _events([0, 4, 5, 6, 9, 10])
    # lookback 5 at t=10 keeps [5, 10): lower bound inclusive, upper exclusive.
    assert history_before(df, 10, lookback=5).get_column(TS).to_list() == [5, 6, 9]


@pytest.mark.parametrize("lookback", [0, -3])
def test_history_before_rejects_non_positive_lookback(lookback: int) -> None:
    with pytest.raises(ValueError, match="lookback"):
        history_before(_events([0]), 1, lookback=lookback)


def test_window_slice_bounds() -> None:
    df = _events([9, 10, 15, 19, 20, 21])
    out = window_slice(df, Window(index=0, start=10, end=20))
    assert out.get_column(TS).to_list() == [10, 15, 19]


def test_windows_partition_the_replay_period() -> None:
    df = _events(list(range(0, 100, 3)))
    windows = make_windows(0, 99, warmup=40, step=20)
    total = sum(window_slice(df, w).height for w in windows)
    assert total == df.filter(pl.col(TS) >= 40).height
    for w in windows:
        assert history_before(df, w.start).height + window_slice(df, w).height == (
            history_before(df, w.end).height
        )
