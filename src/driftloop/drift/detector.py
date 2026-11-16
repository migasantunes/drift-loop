"""Threshold detector: turns drift statistics into a retrain-or-not verdict.

The detector is deliberately dumb: fixed thresholds on four interpretable signals.
Thresholds are configuration, tuned on an early part of the replay and then frozen
(see docs/EVALUATION.md), so every firing can be explained by naming the signal and
the value that crossed its line.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, fields

import numpy as np
import polars as pl

from driftloop.drift import stats
from driftloop.drift.report import DriftReport


@dataclass(frozen=True)
class DriftThresholds:
    """Per-signal thresholds; ``None`` disables a signal (it is still computed and recorded).

    Field names are the signal names used in ``DriftReport.signals``, and the field
    order is the order in which fired signals are listed in ``DriftReport.reasons``.
    """

    popularity_psi: float | None = 0.2
    popularity_jsd: float | None = None
    new_item_share: float | None = 0.15
    new_user_share: float | None = None

    def __post_init__(self) -> None:
        # A NaN threshold would silently never fire; reject it rather than let a typo
        # in a config file disable a signal without anyone noticing.
        for f in fields(self):
            value = getattr(self, f.name)
            if value is not None and math.isnan(value):
                raise ValueError(f"threshold {f.name} must be a number or None, got NaN")


#: Signal names in the stable order used for ``signals`` and ``reasons``.
SIGNAL_NAMES: tuple[str, ...] = tuple(f.name for f in fields(DriftThresholds))


class DriftDetector:
    """Compare a reference period with a current one and apply thresholds."""

    def __init__(self, thresholds: DriftThresholds | None = None, *, n_buckets: int = 10) -> None:
        if n_buckets < 1:
            raise ValueError(f"n_buckets must be >= 1, got {n_buckets}")
        self.thresholds = thresholds if thresholds is not None else DriftThresholds()
        self.n_buckets = n_buckets

    def compare(self, reference: pl.DataFrame, current: pl.DataFrame) -> DriftReport:
        """Compute all four signals and report which ones crossed their threshold.

        All signals are computed even when disabled, so the recorded results show how
        a different threshold choice would have behaved. A signal fires only when its
        value is strictly greater than its threshold; NaN (empty input) never fires.
        The bucket shares are computed once and shared by PSI and JSD, both over
        reference-known items only; if every current event is on a new item they are
        NaN and only ``new_item_share`` can fire.
        """
        ref_shares, cur_shares = stats.known_item_shares(
            *stats.popularity_bucket_shares(reference, current, n_buckets=self.n_buckets)
        )
        undefined = bool(np.isnan(ref_shares).any())
        signals: dict[str, float] = {
            "popularity_psi": math.nan if undefined else stats.psi(ref_shares, cur_shares),
            "popularity_jsd": stats.jensen_shannon(ref_shares, cur_shares),
            "new_item_share": stats.new_item_share(reference, current),
            "new_user_share": stats.new_user_share(reference, current),
        }
        reasons = tuple(
            name
            for name in SIGNAL_NAMES
            if (threshold := getattr(self.thresholds, name)) is not None
            and not math.isnan(signals[name])
            and signals[name] > threshold
        )
        return DriftReport(signals=signals, triggered=bool(reasons), reasons=reasons)
