"""Loop semantics, control experiments and the summary arithmetic of the replay."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import polars as pl
import pytest
import yaml

from driftloop.config import ExperimentConfig, load_config, load_interactions
from driftloop.data.schema import to_canonical
from driftloop.drift.detector import SIGNAL_NAMES
from driftloop.loop.simulate import (
    ROW_SCHEMA,
    SimulationResult,
    run_simulation,
    summarize,
    write_run,
)
from driftloop.replay.clock import DAY

CONFIGS = Path(__file__).resolve().parents[2] / "configs"


def _config(**overrides: Any) -> ExperimentConfig:
    """The small test config with top-level sections replaced or merged."""
    raw = yaml.safe_load((CONFIGS / "synthetic-small.yaml").read_text())
    for key, value in overrides.items():
        if isinstance(value, dict):
            raw[key] = {**raw.get(key, {}), **value}
        else:
            raw[key] = value
    return ExperimentConfig.model_validate(raw)


@pytest.fixture(scope="module")
def small() -> tuple[ExperimentConfig, pl.DataFrame, SimulationResult]:
    cfg = _config(policies=["never", "always", "fixed:2", "fixed:3", "on_drift"])
    data = load_interactions(cfg)
    return cfg, data, run_simulation(data, cfg)


def _policy_rows(result: SimulationResult, policy: str) -> pl.DataFrame:
    return result.rows.filter(pl.col("policy") == policy).sort("window")


# --- loop semantics ------------------------------------------------------------------


def test_row_layout(small: tuple[ExperimentConfig, pl.DataFrame, SimulationResult]) -> None:
    cfg, _, result = small
    rows = result.rows
    assert dict(rows.schema) == ROW_SCHEMA
    n_windows = result.summary["n_windows"]
    assert rows.height == len(cfg.policies) * n_windows == result.summary["n_rows"]
    assert rows.get_column("scored").to_list() == [
        w >= cfg.replay.tuning_windows for w in rows.get_column("window").to_list()
    ]
    assert (rows.get_column("k") == cfg.eval.k).all()


def test_never_never_retrains(
    small: tuple[ExperimentConfig, pl.DataFrame, SimulationResult],
) -> None:
    rows = _policy_rows(small[2], "never")
    assert not rows.get_column("retrained").any()
    assert rows.get_column("model_age_windows").to_list() == rows.get_column("window").to_list()
    assert (rows.get_column("fit_seconds") == 0.0).all()


def test_always_retrains_before_every_window_but_the_first(
    small: tuple[ExperimentConfig, pl.DataFrame, SimulationResult],
) -> None:
    rows = _policy_rows(small[2], "always")
    # Window 0 would refit the warm-up model on identical data: a no-op, skipped.
    assert rows.get_column("retrained").to_list() == [False] + [True] * (rows.height - 1)
    assert (rows.get_column("model_age_windows") == 0).all()
    assert (rows.filter(pl.col("retrained")).get_column("fit_seconds") > 0).all()


@pytest.mark.parametrize("every", [2, 3])
def test_fixed_interval_cadence(
    small: tuple[ExperimentConfig, pl.DataFrame, SimulationResult], every: int
) -> None:
    rows = _policy_rows(small[2], f"fixed:{every}")
    windows = rows.get_column("window").to_list()
    assert rows.get_column("retrained").to_list() == [w > 0 and w % every == 0 for w in windows]
    assert rows.get_column("model_age_windows").to_list() == [w % every for w in windows]


def test_on_drift_retrains_exactly_when_the_report_triggers(
    small: tuple[ExperimentConfig, pl.DataFrame, SimulationResult],
) -> None:
    rows = _policy_rows(small[2], "on_drift")
    triggered = rows.get_column("drift_triggered").fill_null(False)
    assert rows.get_column("retrained").to_list() == triggered.to_list()
    assert triggered.any(), "the small config should make on_drift fire at least once"
    reasons = rows.filter(pl.col("drift_triggered")).get_column("drift_reasons")
    assert all(r for r in reasons.to_list())


def test_model_age_resets_on_retrain_and_grows_otherwise(
    small: tuple[ExperimentConfig, pl.DataFrame, SimulationResult],
) -> None:
    for policy in small[0].policies:
        rows = _policy_rows(small[2], policy)
        ages = rows.get_column("model_age_windows").to_list()
        retrained = rows.get_column("retrained").to_list()
        assert ages[0] == 0
        for i in range(1, len(ages)):
            assert ages[i] == (0 if retrained[i] else ages[i - 1] + 1), policy


def test_first_window_has_no_drift_report(
    small: tuple[ExperimentConfig, pl.DataFrame, SimulationResult],
) -> None:
    rows = small[2].rows
    drift_cols = ["drift_triggered", "drift_reasons", *SIGNAL_NAMES]
    first = rows.filter(pl.col("window") == 0).select(drift_cols)
    assert all(first.get_column(c).null_count() == first.height for c in drift_cols)
    later = rows.filter(pl.col("window") > 0)
    assert later.get_column("drift_triggered").null_count() == 0
    assert later.get_column("new_item_share").null_count() == 0


def test_policies_start_from_the_same_model(
    small: tuple[ExperimentConfig, pl.DataFrame, SimulationResult],
) -> None:
    first = small[2].rows.filter(pl.col("window") == 0)
    assert first.get_column("recall").n_unique() == 1


def test_summary_metadata(small: tuple[ExperimentConfig, pl.DataFrame, SimulationResult]) -> None:
    cfg, data, result = small
    s = result.summary
    assert s["name"] == cfg.name
    assert s["n_interactions"] == data.height
    assert s["n_scored_windows"] == s["n_windows"] - cfg.replay.tuning_windows
    assert set(s["policies"]) == set(cfg.policies)
    for stats in s["policies"].values():
        assert stats["windows"] == s["n_scored_windows"]


def test_too_few_windows_is_a_clear_error() -> None:
    cfg = _config(replay={"warmup_days": 49, "step_days": 7})
    with pytest.raises(ValueError, match="at least two windows"):
        run_simulation(load_interactions(cfg), cfg)


def test_tuning_windows_must_leave_scored_windows() -> None:
    cfg = _config(replay={"tuning_windows": 5})
    with pytest.raises(ValueError, match="no scored windows"):
        run_simulation(load_interactions(cfg), cfg)


def _gapped_interactions() -> pl.DataFrame:
    """Daily events on days 0-27 and 42-55, nothing on days 28-41."""
    days = [d for d in range(56) if d < 28 or d >= 42]
    return to_canonical(
        pl.DataFrame(
            {
                "user_id": [d % 3 for d in days],
                "item_id": [d % 5 for d in days],
                "ts": [d * DAY for d in days],
                "weight": [1.0] * len(days),
            }
        )
    )


def test_empty_training_set_is_an_error() -> None:
    # Windows start on days 28, 35, 42, 49. The warm-up fit at day 28 sees days 21-27;
    # the retrain at day 35 would see days 28-34, which are empty.
    cfg = _config(
        replay={"warmup_days": 28, "step_days": 7, "lookback_days": 7, "tuning_windows": 0},
        policies=["always"],
    )
    with pytest.raises(ValueError, match=rf"t={35 * DAY} with lookback_days=7"):
        run_simulation(_gapped_interactions(), cfg)


def test_lookback_none_trains_on_all_history() -> None:
    cfg = _config(replay={"lookback_days": None}, policies=["never", "always"])
    result = run_simulation(load_interactions(cfg), cfg)
    assert result.rows.height == 2 * result.summary["n_windows"]


# --- control experiments --------------------------------------------------------------


def _control(drift: float) -> SimulationResult:
    cfg = ExperimentConfig.model_validate(
        {
            "name": f"control-{drift}",
            "data": {
                "source": "synthetic",
                "synthetic": {
                    "n_users": 400,
                    "n_items_initial": 800,
                    "n_new_items_per_day": 30.0,
                    "n_new_users_per_day": 4.0,
                    "n_clusters": 10,
                    "n_days": 91,
                    "drift_strength": drift,
                    "seed": 0,
                },
            },
            "replay": {"warmup_days": 28, "step_days": 7, "lookback_days": 28},
            "eval": {"k": 20},
            "policies": ["never", "always", "on_drift"],
            "drift": {"thresholds": {"popularity_psi": None, "new_item_share": 0.2}},
        }
    )
    return run_simulation(load_interactions(cfg), cfg)


def test_control_without_drift_retraining_does_not_matter() -> None:
    """Stationary data: a frozen model is as good as a fresh one.

    Observed gap across seeds 0-4 is at most 1.6% of never's recall; 5% leaves room.
    """
    s = _control(0.0).summary["policies"]
    never, always = s["never"]["mean_recall"], s["always"]["mean_recall"]
    assert abs(always - never) < 0.05 * never
    assert s["on_drift"]["retrains"] == 0


def test_control_with_drift_retraining_pays_off() -> None:
    """Drifting data: always beats never clearly over the later windows.

    Observed later-window ratio across seeds 0-4 is at least 1.35; 1.2 leaves room.
    on_drift retrained 4-5 of 7 times across those seeds.
    """
    result = _control(0.5)
    late = (
        result.rows.filter(pl.col("window") >= 4)
        .group_by("policy")
        .agg(pl.col("recall").mean())
        .rows_by_key("policy", unique=True)
    )
    assert late["always"][0] > 1.2 * late["never"][0]
    s = result.summary["policies"]
    assert 1 <= s["on_drift"]["retrains"] < s["always"]["retrains"]


# --- summary arithmetic -------------------------------------------------------------


def _rows(
    spec: dict[str, list[tuple[float, bool]]], fit_seconds: dict[str, float] | None = None
) -> pl.DataFrame:
    """Rows from {policy: [(recall, retrained) per window]}; other metrics mirror recall.

    ``fit_seconds`` gives each policy's duration per retrain (default 0.5).
    """
    durations = fit_seconds or {}
    records = [
        {
            "policy": policy,
            "window": w,
            "recall": recall,
            "ndcg": recall,
            "hit_rate": recall,
            "coverage": recall,
            "retrained": retrained,
            "fit_seconds": durations.get(policy, 0.5) if retrained else 0.0,
        }
        for policy, values in spec.items()
        for w, (recall, retrained) in enumerate(values)
    ]
    return pl.DataFrame(records)


def test_summary_hand_checked() -> None:
    nan = math.nan
    rows = _rows(
        {
            # Window 0 is a tuning window: its huge recall and its retrain must not count.
            "never": [(0.9, False), (0.1, False), (0.1, False)],
            "always": [(0.9, True), (0.3, True), (0.3, True)],
            # NaN (no evaluable users) is skipped, not averaged as zero.
            "fixed:2": [(0.9, False), (0.2, True), (nan, False)],
        },
        # fixed:2's retrain is 5x faster than always's, so a seconds-based cost share
        # would be 0.1 / 1.0 = 0.1, not the count-based 1 / 2 = 0.5 asserted below.
        fit_seconds={"always": 0.5, "fixed:2": 0.1},
    )
    s = summarize(rows, tuning_windows=1)["policies"]
    assert s["never"]["mean_recall"] == pytest.approx(0.1)
    assert s["always"]["mean_recall"] == pytest.approx(0.3)
    assert s["fixed:2"]["mean_recall"] == pytest.approx(0.2)
    assert s["always"]["retrains"] == 2
    assert s["always"]["retrain_seconds"] == pytest.approx(1.0)
    assert s["fixed:2"]["retrain_seconds"] == pytest.approx(0.1)
    assert s["fixed:2"]["quality_kept"] == pytest.approx(0.5)
    assert s["fixed:2"]["cost_share"] == pytest.approx(0.5)
    assert s["never"]["quality_kept"] == pytest.approx(0.0)
    assert s["always"]["quality_kept"] == pytest.approx(1.0)
    assert s["always"]["cost_share"] == pytest.approx(1.0)


def test_summary_zero_denominators_give_none() -> None:
    rows = _rows(
        {
            "never": [(0.2, False), (0.2, False)],
            "always": [(0.2, False), (0.2, False)],
            "on_drift": [(0.25, False), (0.25, False)],
        }
    )
    s = summarize(rows, tuning_windows=0)["policies"]
    for stats in s.values():
        assert stats["quality_kept"] is None
        assert stats["cost_share"] is None


def test_summary_without_both_bounds_gives_none() -> None:
    rows = _rows({"always": [(0.3, True)], "fixed:2": [(0.2, False)]})
    s = summarize(rows, tuning_windows=0)["policies"]
    assert s["fixed:2"]["quality_kept"] is None
    assert s["fixed:2"]["cost_share"] is None


def test_summary_all_nan_mean_is_nan() -> None:
    rows = _rows({"never": [(math.nan, False)]})
    assert math.isnan(summarize(rows, tuning_windows=0)["policies"]["never"]["mean_recall"])


# --- run directory ------------------------------------------------------------------


def test_write_run(
    small: tuple[ExperimentConfig, pl.DataFrame, SimulationResult], tmp_path: Path
) -> None:
    cfg, _, result = small
    # A NaN in the summary must come out as JSON null, not the invalid token NaN.
    summary = {**result.summary, "probe": math.nan}
    write_run(SimulationResult(rows=result.rows, summary=summary), cfg, tmp_path / "run")

    out = tmp_path / "run"
    rows = pl.read_csv(out / "results.csv")
    assert rows.columns == list(ROW_SCHEMA)
    assert rows.height == result.rows.height

    def reject(token: str) -> None:
        raise ValueError(f"invalid JSON constant {token}")

    loaded = json.loads((out / "summary.json").read_text(), parse_constant=reject)
    assert loaded["probe"] is None
    assert loaded["policies"].keys() == result.summary["policies"].keys()

    assert load_config(out / "config.yaml") == cfg
