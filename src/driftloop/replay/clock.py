"""Time handling for the replay harness.

The whole project rests on one rule: **at simulated time t, nothing with
ts >= t exists.** Every function that hands data to a model or to a drift
statistic goes through :func:`history_before` or :func:`window_slice`, so the
rule is enforced in one place and tested once.

Time is integer seconds since the start of the dataset. Intervals are half-open:
``[start, end)``.
"""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl

from driftloop.data.schema import TS

DAY = 86_400
WEEK = 7 * DAY


@dataclass(frozen=True, slots=True)
class Window:
    """One evaluation window ``[start, end)`` of the replay."""

    index: int
    start: int
    end: int

    def __post_init__(self) -> None:
        if self.end <= self.start:
            raise ValueError(f"window {self.index} is empty: [{self.start}, {self.end})")

    @property
    def length(self) -> int:
        return self.end - self.start


def make_windows(t_min: int, t_max: int, *, warmup: int, step: int) -> list[Window]:
    """Cut ``[t_min, t_max]`` into consecutive evaluation windows of length ``step``.

    The first ``warmup`` seconds are reserved as initial training history, so the
    first window starts at ``t_min + warmup``. A trailing partial window is dropped:
    comparing a half-length window with full ones would distort the metric curve.

    ``t_max`` is the timestamp of the last event (inclusive).
    """
    if warmup <= 0 or step <= 0:
        raise ValueError("warmup and step must be positive")
    if t_max < t_min:
        raise ValueError("t_max must be >= t_min")
    windows: list[Window] = []
    start = t_min + warmup
    # t_max is inclusive, so a window [start, end) is complete when end - 1 <= t_max.
    while start + step - 1 <= t_max:
        windows.append(Window(index=len(windows), start=start, end=start + step))
        start += step
    return windows


def history_before(
    interactions: pl.DataFrame, t: int, *, lookback: int | None = None
) -> pl.DataFrame:
    """Everything that is visible at simulated time ``t``: rows with ``ts < t``.

    With ``lookback``, only the most recent ``lookback`` seconds are kept
    (``t - lookback <= ts < t``), which is how a rolling training window is built.
    """
    cond = pl.col(TS) < t
    if lookback is not None:
        if lookback <= 0:
            raise ValueError("lookback must be positive")
        cond = cond & (pl.col(TS) >= t - lookback)
    return interactions.filter(cond)


def window_slice(interactions: pl.DataFrame, window: Window) -> pl.DataFrame:
    """Rows that fall inside ``window``: ``start <= ts < end``."""
    return interactions.filter((pl.col(TS) >= window.start) & (pl.col(TS) < window.end))
