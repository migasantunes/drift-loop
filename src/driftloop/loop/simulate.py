"""The offline replay of the retrain loop: observe, measure, decide, retrain, evaluate, record.

This module is the executable form of ``docs/EVALUATION.md``, section "The replay".
Each policy runs its own independent copy of the loop, all starting from the same
warm-up model, so the only difference between their curves is when they retrained.

Time discipline: every slice of the interaction table is produced by
``driftloop.replay.clock`` (``history_before`` / ``window_slice``) at the start of
the window being processed. The model is fitted on data strictly before that start,
the drift report compares data strictly before it, and the policy decides before the
window is scored. The window's own rows are used only to score it.

Windows are the outer loop and policies the inner one. That lets each window's slices
be computed once and shared by every policy without caching the whole replay, and it
changes nothing about independence: each policy keeps its own model and state.
"""

from __future__ import annotations

import json
import math
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import yaml

from driftloop.config import ExperimentConfig, ModelConfig
from driftloop.data.schema import TS
from driftloop.drift.detector import SIGNAL_NAMES, DriftDetector
from driftloop.drift.report import DriftReport
from driftloop.eval.protocol import evaluate_window
from driftloop.loop.policies import PolicyContext, RetrainPolicy, make_policy
from driftloop.models.base import Recommender
from driftloop.models.popularity import PopularityRecommender
from driftloop.replay.clock import Window, history_before, make_windows, window_slice

ModelFactory = Callable[[], Recommender]

#: Metric whose means define ``quality_kept`` in the summary.
HEADLINE_METRIC = "recall"
#: Per-window metrics averaged in the summary.
SUMMARY_METRICS: tuple[str, ...] = ("recall", "ndcg", "hit_rate", "coverage")

RESULTS_FILE = "results.csv"
SUMMARY_FILE = "summary.json"
CONFIG_FILE = "config.yaml"

#: Column order and dtypes of ``SimulationResult.rows``. Fixed explicitly so that an
#: all-null column (a disabled signal on a tiny run) still has a numeric dtype.
ROW_SCHEMA: dict[str, pl.DataType] = {
    "policy": pl.String(),
    "window": pl.Int64(),
    "start": pl.Int64(),
    "end": pl.Int64(),
    "scored": pl.Boolean(),
    "k": pl.Int64(),
    "n_users": pl.Int64(),
    "recall": pl.Float64(),
    "ndcg": pl.Float64(),
    "hit_rate": pl.Float64(),
    "coverage": pl.Float64(),
    "retrained": pl.Boolean(),
    "fit_seconds": pl.Float64(),
    "model_age_windows": pl.Int64(),
    "drift_triggered": pl.Boolean(),
    "drift_reasons": pl.String(),
    **{name: pl.Float64() for name in SIGNAL_NAMES},
}


@dataclass(frozen=True)
class SimulationResult:
    """Outcome of one replay.

    ``rows`` has one row per (policy, window) with the columns of ``ROW_SCHEMA``;
    ``summary`` holds per-policy aggregates over the scored windows and run metadata.
    """

    rows: pl.DataFrame
    summary: dict[str, Any]


def model_factory(cfg: ModelConfig) -> ModelFactory:
    """Map the configured model name to a constructor of fresh, unfitted models."""
    if cfg.name == "popularity":
        return PopularityRecommender
    raise ValueError(f"unknown model {cfg.name!r}")


@dataclass
class _PolicyState:
    """What one policy's copy of the loop carries from window to window."""

    policy: RetrainPolicy
    model: Recommender
    #: Exclusive end of the training data of the model in service.
    trained_until: int
    rows: list[dict[str, Any]] = field(default_factory=list)


def _fit(
    factory: ModelFactory, train: pl.DataFrame, until: int, lookback_days: int | None
) -> tuple[Recommender, float]:
    """Fit a brand-new model and time it.

    A new instance every time, never a refit of the one in service, so a retrain
    cannot inherit state from the previous model and the old model stays usable
    until the new one exists (which is how promotion will work in M4).

    An empty training set is an error rather than a model that recommends nothing: it
    means the lookback fell into a gap in the data, and silently serving an empty
    model would show up only as a mysterious zero-recall stretch in the results.
    """
    if train.height == 0:
        span = "all history" if lookback_days is None else f"lookback_days={lookback_days}"
        raise ValueError(
            f"no training data before simulated time t={until} with {span}; "
            "the lookback contains no interactions (a gap in the data?), so lengthen "
            "lookback_days or check the input"
        )
    model = factory()
    began = time.perf_counter()
    model.fit(train, until=until)
    return model, time.perf_counter() - began


def _age_in_windows(t: int, trained_until: int, step: int) -> int:
    """Whole windows between the model's training cut-off and ``t``.

    Training cut-offs are always window starts, so this is exact; 0 means the model
    was trained at the start of the current window.
    """
    return (t - trained_until) // step


def run_simulation(
    interactions: pl.DataFrame,
    cfg: ExperimentConfig,
    *,
    model_factory: ModelFactory | None = None,
) -> SimulationResult:
    """Replay ``interactions`` window by window under every configured policy.

    ``model_factory`` overrides the configured model; tests pass spy models through
    it. ``interactions`` must be a canonical table (sorted by ``ts``).
    """
    factory = model_factory if model_factory is not None else _default_factory(cfg)
    replay = cfg.replay
    windows = _make_windows(interactions, cfg)
    policies = [make_policy(spec) for spec in cfg.policies]
    detector = DriftDetector(cfg.drift.thresholds.to_thresholds(), n_buckets=cfg.drift.n_buckets)
    k = cfg.eval.k
    lookback = replay.lookback

    # Every policy starts from the same model, fitted on the warm-up period at the
    # first window's start. It is fitted once and shared: policies never mutate a
    # model, they replace it, so sharing the object is safe and keeps them identical.
    t0 = windows[0].start
    initial_model, initial_fit_seconds = _fit(
        factory, history_before(interactions, t0, lookback=lookback), t0, replay.lookback_days
    )
    states = [_PolicyState(policy=p, model=initial_model, trained_until=t0) for p in policies]

    previous: Window | None = None
    for window in windows:
        t = window.start
        visible = history_before(interactions, t)
        future = window_slice(interactions, window)
        recent = window_slice(interactions, previous) if previous is not None else None
        train = history_before(interactions, t, lookback=lookback)
        # Reports depend only on (reference cut-off, current window); policies whose
        # models share a cut-off share the report instead of recomputing it.
        reports: dict[int, DriftReport] = {}

        for state in states:
            report: DriftReport | None = None
            if recent is not None:
                if state.trained_until not in reports:
                    reference = history_before(interactions, state.trained_until, lookback=lookback)
                    reports[state.trained_until] = detector.compare(reference, recent)
                report = reports[state.trained_until]

            ctx = PolicyContext(
                window_index=window.index,
                windows_since_retrain=_age_in_windows(t, state.trained_until, replay.step),
                drift=report,
            )
            # A retrain at the cut-off the model already has would refit identical data:
            # skip it so it is neither paid for nor counted.
            retrained = state.policy.should_retrain(ctx) and state.trained_until != t
            fit_seconds = 0.0
            if retrained:
                state.model, fit_seconds = _fit(factory, train, t, replay.lookback_days)
                state.trained_until = t

            result = evaluate_window(state.model, history=visible, future=future, k=k)
            state.rows.append(
                {
                    "policy": state.policy.name,
                    "window": window.index,
                    "start": window.start,
                    "end": window.end,
                    "scored": window.index >= replay.tuning_windows,
                    "k": result.k,
                    "n_users": result.n_users,
                    "recall": result.recall,
                    "ndcg": result.ndcg,
                    "hit_rate": result.hit_rate,
                    "coverage": result.coverage,
                    "retrained": retrained,
                    "fit_seconds": fit_seconds,
                    "model_age_windows": _age_in_windows(t, state.trained_until, replay.step),
                    "drift_triggered": None if report is None else report.triggered,
                    "drift_reasons": None if report is None else ",".join(report.reasons),
                    **{
                        name: None if report is None else report.signals[name]
                        for name in SIGNAL_NAMES
                    },
                }
            )
        previous = window

    rows = pl.DataFrame([r for s in states for r in s.rows], schema=ROW_SCHEMA, orient="row")
    summary = summarize(rows, tuning_windows=replay.tuning_windows)
    summary = {
        "name": cfg.name,
        "model": cfg.model.name,
        "headline_metric": HEADLINE_METRIC,
        "n_interactions": interactions.height,
        "n_windows": len(windows),
        "tuning_windows": replay.tuning_windows,
        "n_scored_windows": len(windows) - replay.tuning_windows,
        "n_rows": rows.height,
        "first_window_start": t0,
        "last_window_end": windows[-1].end,
        "initial_fit_seconds": initial_fit_seconds,
        **summary,
    }
    return SimulationResult(rows=rows, summary=summary)


def _default_factory(cfg: ExperimentConfig) -> ModelFactory:
    return model_factory(cfg.model)


def _make_windows(interactions: pl.DataFrame, cfg: ExperimentConfig) -> list[Window]:
    """Cut the replay into windows and check there is something to compare and score."""
    if interactions.height == 0:
        raise ValueError("no interactions to replay")
    ts = interactions.get_column(TS)
    t_min, t_max = ts.min(), ts.max()
    assert isinstance(t_min, int)
    assert isinstance(t_max, int)
    replay = cfg.replay
    windows = make_windows(t_min, t_max, warmup=replay.warmup, step=replay.step)
    if len(windows) < 2:
        span_days = (t_max - t_min + 1) / 86_400
        raise ValueError(
            f"the replay needs at least two windows but the data spans {span_days:.1f} days, "
            f"which fits {len(windows)} after a {replay.warmup_days}-day warm-up with "
            f"{replay.step_days}-day windows; lengthen the data or shorten warmup_days/step_days"
        )
    if replay.tuning_windows >= len(windows):
        raise ValueError(
            f"tuning_windows={replay.tuning_windows} leaves no scored windows out of "
            f"{len(windows)}; lower it or lengthen the data"
        )
    return windows


def _nan_mean(values: Sequence[float | None]) -> float:
    """Mean over the finite values; NaN when there are none.

    A window with no evaluable users has NaN metrics; skipping it rather than
    counting it as zero keeps an empty window from looking like a quality collapse.
    """
    arr = np.array([math.nan if v is None else v for v in values], dtype=np.float64)
    finite = arr[~np.isnan(arr)]
    return float(finite.mean()) if finite.size else math.nan


def _share(numerator: float, denominator: float) -> float | None:
    """``numerator / denominator``, or None where the ratio is undefined."""
    if math.isnan(numerator) or math.isnan(denominator) or denominator == 0:
        return None
    return numerator / denominator


def summarize(rows: pl.DataFrame, *, tuning_windows: int) -> dict[str, Any]:
    """Per-policy aggregates over the scored windows (index >= ``tuning_windows``).

    ``quality_kept`` is the share of ``always``'s gain in mean recall over ``never``
    that a policy keeps, and ``cost_share`` its retrains as a share of ``always``'s.
    Both need ``never`` and ``always`` among the policies and are None otherwise, or
    when their denominator is zero (``always`` no better than ``never``, or ``always``
    never retrained in the scored windows). Retrains during the tuning windows are not
    counted, for the same reason their quality is not: the headline is about the
    windows the thresholds were not chosen on.
    """
    scored = rows.filter(pl.col("window") >= tuning_windows)
    per_policy: dict[str, dict[str, Any]] = {}
    for policy in rows.get_column("policy").unique(maintain_order=True).to_list():
        part = scored.filter(pl.col("policy") == policy)
        stats: dict[str, Any] = {
            f"mean_{m}": _nan_mean(part.get_column(m).to_list()) for m in SUMMARY_METRICS
        }
        stats["retrains"] = int(part.get_column("retrained").sum())
        stats["retrain_seconds"] = float(part.get_column("fit_seconds").sum())
        stats["windows"] = part.height
        per_policy[policy] = stats

    key = f"mean_{HEADLINE_METRIC}"
    never, always = per_policy.get("never"), per_policy.get("always")
    for stats in per_policy.values():
        if never is None or always is None:
            stats["quality_kept"] = None
            stats["cost_share"] = None
            continue
        stats["quality_kept"] = _share(stats[key] - never[key], always[key] - never[key])
        stats["cost_share"] = _share(float(stats["retrains"]), float(always["retrains"]))
    return {"policies": per_policy}


def _json_safe(value: Any) -> Any:
    """Replace NaN with None recursively: ``json`` would otherwise write invalid JSON."""
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    return value


def write_run(result: SimulationResult, cfg: ExperimentConfig, out_dir: Path) -> None:
    """Write ``results.csv``, ``summary.json`` and the resolved ``config.yaml``.

    The config is written as resolved (defaults filled in), so the run directory alone
    says exactly which settings produced the numbers in it.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    result.rows.write_csv(out_dir / RESULTS_FILE)
    (out_dir / SUMMARY_FILE).write_text(
        json.dumps(_json_safe(result.summary), indent=2) + "\n", encoding="utf-8"
    )
    (out_dir / CONFIG_FILE).write_text(
        yaml.safe_dump(cfg.model_dump(mode="json"), sort_keys=False), encoding="utf-8"
    )
