"""Leakage canaries for the replay loop: the most important tests in the repository.

The rule (docs/EVALUATION.md): at simulated time t nothing with ts >= t exists. Each
test here targets a different way the loop could break it, and each was checked to
fail when the corresponding leak was introduced on purpose:

- ``test_every_fit_*``: a spy model sees exactly what the loop trains on. Catches
  training past the window start (for example up to the window end).
- ``test_future_independence``: perturbing the data from some boundary onwards must
  not change any result row of a window that ends at or before it. The perturbation
  inside the last compared window repeats already-seen (user, item) pairs, which
  leaves that window's ground truth and seen items unchanged but would move any
  popularity count or drift statistic that read the window early.
- ``test_rows_match_independent_recomputation``: rebuilds selected rows from
  ``replay.clock`` slices by hand. Catches the evaluation or the drift report being
  fed the wrong slice (for example seen items that include the scored window).
"""

from __future__ import annotations

import math
from typing import Self

import numpy as np
import polars as pl
import pytest
from polars.testing import assert_frame_equal

from driftloop.config import ExperimentConfig, load_interactions
from driftloop.data.schema import ITEM, TS, USER, WEIGHT, to_canonical, validate_interactions
from driftloop.drift.detector import SIGNAL_NAMES, DriftDetector
from driftloop.eval.protocol import evaluate_window
from driftloop.loop.simulate import run_simulation
from driftloop.models.popularity import PopularityRecommender
from driftloop.replay.clock import Window, history_before, make_windows, window_slice


def _config(policies: list[str]) -> ExperimentConfig:
    return ExperimentConfig.model_validate(
        {
            "name": "leakage",
            "data": {
                "source": "synthetic",
                "synthetic": {
                    "n_users": 200,
                    "n_new_users_per_day": 2.0,
                    "n_items_initial": 400,
                    "n_new_items_per_day": 8.0,
                    "n_clusters": 8,
                    "n_days": 63,
                    "events_per_user_per_day": 2.0,
                    "drift_strength": 0.5,
                    "seed": 11,
                },
            },
            "replay": {"warmup_days": 21, "step_days": 7, "lookback_days": 14},
            "eval": {"k": 10},
            "policies": policies,
            # Low thresholds so on_drift actually retrains in this short run.
            "drift": {
                "n_buckets": 5,
                "thresholds": {"popularity_psi": 0.01, "new_item_share": 0.08},
            },
        }
    )


ALL_POLICIES = ["never", "always", "fixed:2", "on_drift"]


@pytest.fixture(scope="module")
def cfg() -> ExperimentConfig:
    return _config(ALL_POLICIES)


@pytest.fixture(scope="module")
def data(cfg: ExperimentConfig) -> pl.DataFrame:
    return load_interactions(cfg)


@pytest.fixture(scope="module")
def windows(data: pl.DataFrame, cfg: ExperimentConfig) -> list[Window]:
    ts = data.get_column(TS)
    t_min, t_max = ts.min(), ts.max()
    assert isinstance(t_min, int)
    assert isinstance(t_max, int)
    out = make_windows(t_min, t_max, warmup=cfg.replay.warmup, step=cfg.replay.step)
    assert len(out) >= 4, "the canaries need a few windows on each side of the boundary"
    return out


class SpyModel(PopularityRecommender):
    """Popularity model that checks and records what it is trained on."""

    def __init__(self, log: list[dict[str, int]]) -> None:
        super().__init__()
        self._log = log

    def fit(self, interactions: pl.DataFrame, *, until: int) -> Self:
        ts = interactions.get_column(TS)
        # Asserted here, not only afterwards, so the failure points at the offending fit.
        assert interactions.height > 0
        ts_max, ts_min = ts.max(), ts.min()
        assert isinstance(ts_max, int)
        assert isinstance(ts_min, int)
        assert ts_max < until, f"fit saw ts={ts_max} >= until={until}"
        self._log.append({"until": until, "ts_min": ts_min, "ts_max": ts_max, "id": id(self)})
        return super().fit(interactions, until=until)


def _run_with_spy(
    data: pl.DataFrame, cfg: ExperimentConfig
) -> tuple[pl.DataFrame, list[dict[str, int]]]:
    log: list[dict[str, int]] = []
    result = run_simulation(data, cfg, model_factory=lambda: SpyModel(log))
    return result.rows, log


def test_every_fit_is_before_its_cutoff_and_cutoffs_are_window_starts(
    data: pl.DataFrame, cfg: ExperimentConfig, windows: list[Window]
) -> None:
    rows, log = _run_with_spy(data, cfg)
    starts = {w.start for w in windows}
    lookback = cfg.replay.lookback
    assert lookback is not None
    assert len(log) == 1 + int(rows.get_column("retrained").sum())
    for fit in log:
        assert fit["until"] in starts
        assert fit["ts_min"] >= fit["until"] - lookback
    # Every retrain builds a new model object; none is refitted in place.
    assert len({fit["id"] for fit in log}) == len(log)


def test_every_fit_cutoff_is_the_start_of_the_window_it_serves(
    data: pl.DataFrame, windows: list[Window]
) -> None:
    """Link each fit to its row: the always policy must train at each window's start."""
    rows, log = _run_with_spy(data, _config(["always"]))
    retrain_starts = rows.filter(pl.col("retrained")).sort("window").get_column("start").to_list()
    assert log[0]["until"] == windows[0].start  # the shared warm-up model
    assert [fit["until"] for fit in log[1:]] == retrain_starts
    assert retrain_starts == [w.start for w in windows[1:]]


def _perturb(data: pl.DataFrame, inject: Window, boundary: int) -> pl.DataFrame:
    """Data identical to ``data`` before ``inject.start``, and different from then on.

    Inside ``inject``: about 2000 extra events on a few rarely played items, each a
    (user, item) pair already present before ``inject.start``. Those are repeats, so
    the window's ground truth (new-to-user items), its scored users and the seen
    items at its start are unchanged, while item counts in the window change a lot.

    From ``boundary`` on: user ids and item ids are shuffled among the rows.
    """
    rng = np.random.default_rng(0)
    before = history_before(data, inject.start)
    counts = before.group_by(ITEM).len().sort(["len", ITEM])
    rare_items = counts.filter(pl.col("len") >= 2).head(5).get_column(ITEM)
    pairs = before.filter(pl.col(ITEM).is_in(rare_items.implode())).select(USER, ITEM).unique()
    copies = 2000 // pairs.height + 1
    injected = pairs.select(pl.all().repeat_by(copies).explode()).with_columns(
        pl.Series(TS, rng.integers(inject.start, inject.end, size=pairs.height * copies)),
        pl.lit(1.0, dtype=pl.Float32).alias(WEIGHT),
    )

    late = data.filter(pl.col(TS) >= boundary)
    scrambled = late.with_columns(
        pl.Series(USER, rng.permutation(late.get_column(USER).to_numpy())),
        pl.Series(ITEM, rng.permutation(late.get_column(ITEM).to_numpy())),
    )
    early = data.filter(pl.col(TS) < boundary)
    return to_canonical(pl.concat([early, injected.select(early.columns), scrambled]))


def test_future_independence(
    data: pl.DataFrame, cfg: ExperimentConfig, windows: list[Window]
) -> None:
    inject = windows[2]
    boundary = inject.end
    perturbed = validate_interactions(_perturb(data, inject, boundary))

    base = run_simulation(data, cfg).rows.drop("fit_seconds")
    other = run_simulation(perturbed, cfg).rows.drop("fit_seconds")
    assert base.height == other.height

    def upto(rows: pl.DataFrame) -> pl.DataFrame:
        return rows.filter(pl.col("end") <= boundary).sort("policy", "window")

    assert upto(base).get_column("window").max() == inject.index
    assert_frame_equal(upto(base), upto(other))

    # Not vacuous: the perturbation changes results after the boundary...
    after = pl.col("start") >= boundary
    assert not base.filter(after).equals(other.filter(after))
    # ...and a model that trained on the injected window would rank differently.
    lookback = cfg.replay.lookback
    at_end = [
        PopularityRecommender()
        .fit(history_before(d, boundary, lookback=lookback), until=boundary)
        .recommend(np.array([0]), cfg.eval.k)
        for d in (data, perturbed)
    ]
    assert not np.array_equal(at_end[0], at_end[1])


def test_rows_match_independent_recomputation(
    data: pl.DataFrame, cfg: ExperimentConfig, windows: list[Window]
) -> None:
    """Rebuild the never and always rows from clock slices, outside the loop."""
    rows = run_simulation(data, cfg).rows
    lookback = cfg.replay.lookback
    k = cfg.eval.k
    detector = DriftDetector(cfg.drift.thresholds.to_thresholds(), n_buckets=cfg.drift.n_buckets)
    t0 = windows[0].start
    frozen = PopularityRecommender().fit(history_before(data, t0, lookback=lookback), until=t0)

    for w in windows:
        visible = history_before(data, w.start)
        future = window_slice(data, w)
        fresh = PopularityRecommender().fit(
            history_before(data, w.start, lookback=lookback), until=w.start
        )
        for policy, model, ref_until in (
            ("never", frozen, t0),
            ("always", fresh, windows[w.index - 1].start if w.index else t0),
        ):
            row = rows.filter((pl.col("policy") == policy) & (pl.col("window") == w.index))
            expected = evaluate_window(model, history=visible, future=future, k=k)
            assert row.item(0, "n_users") == expected.n_users
            for metric in ("recall", "ndcg", "hit_rate", "coverage"):
                assert row.item(0, metric) == pytest.approx(getattr(expected, metric))
            if w.index == 0:
                continue
            report = detector.compare(
                history_before(data, ref_until, lookback=lookback),
                window_slice(data, windows[w.index - 1]),
            )
            assert row.item(0, "drift_triggered") == report.triggered
            for name in SIGNAL_NAMES:
                value = row.item(0, name)
                if math.isnan(report.signals[name]):
                    assert value is None or math.isnan(value)
                else:
                    assert value == pytest.approx(report.signals[name])


def test_simulation_is_deterministic(data: pl.DataFrame, cfg: ExperimentConfig) -> None:
    a = run_simulation(data, cfg).rows.drop("fit_seconds")
    b = run_simulation(data, cfg).rows.drop("fit_seconds")
    assert_frame_equal(a, b)
