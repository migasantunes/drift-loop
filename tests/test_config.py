"""Experiment configuration: valid files load, mistakes fail loudly at load time."""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

import polars as pl
import pytest
import yaml
from pydantic import ValidationError

from driftloop.config import (
    SyntheticSettings,
    ThresholdsConfig,
    load_config,
    load_interactions,
)
from driftloop.data.schema import TS, validate_interactions
from driftloop.data.synthetic import SyntheticConfig
from driftloop.drift.detector import DriftThresholds
from driftloop.replay.clock import DAY

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


def _minimal() -> dict[str, Any]:
    return {"name": "t", "data": {"source": "synthetic"}, "policies": ["never"]}


def _write(tmp_path: Path, raw: Any) -> Path:
    path = tmp_path / "c.yaml"
    path.write_text(yaml.safe_dump(raw))
    return path


@pytest.mark.parametrize("name", ["synthetic-demo.yaml", "synthetic-small.yaml"])
def test_shipped_configs_load(name: str) -> None:
    cfg = load_config(CONFIGS / name)
    assert cfg.data.source == "synthetic"
    assert "never" in cfg.policies
    assert "always" in cfg.policies


def test_minimal_config_gets_documented_defaults(tmp_path: Path) -> None:
    cfg = load_config(_write(tmp_path, _minimal()))
    assert cfg.replay.warmup == 28 * DAY
    assert cfg.replay.step == 7 * DAY
    assert cfg.replay.lookback == 28 * DAY
    assert cfg.replay.tuning_windows == 0
    assert cfg.eval.k == 20
    assert cfg.model.name == "popularity"
    assert cfg.drift.thresholds.to_thresholds() == DriftThresholds()


def test_lookback_null_means_all_history(tmp_path: Path) -> None:
    raw = _minimal() | {"replay": {"lookback_days": None}}
    assert load_config(_write(tmp_path, raw)).replay.lookback is None


@pytest.mark.parametrize(
    "patch",
    [
        {"nmae": "typo"},
        {"replay": {"lookback_day": 7}},
        {"data": {"source": "synthetic", "synthetic": {"n_user": 5}}},
        {"drift": {"thresholds": {"psi": 0.1}}},
        {"eval": {"k": 10, "metric": "recall"}},
    ],
)
def test_unknown_keys_are_rejected(tmp_path: Path, patch: dict[str, Any]) -> None:
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        load_config(_write(tmp_path, _minimal() | patch))


@pytest.mark.parametrize(
    "policies", [["sometimes"], ["fixed:0"], ["fixed:x"], ["never", "never"], []]
)
def test_bad_policy_lists_are_rejected(tmp_path: Path, policies: list[str]) -> None:
    with pytest.raises(ValidationError):
        load_config(_write(tmp_path, _minimal() | {"policies": policies}))


def test_policy_specs_are_canonicalised(tmp_path: Path) -> None:
    cfg = load_config(_write(tmp_path, _minimal() | {"policies": ["fixed:04", "on_drift"]}))
    assert cfg.policies == ["fixed:4", "on_drift"]
    with pytest.raises(ValidationError, match="more than once"):
        load_config(_write(tmp_path, _minimal() | {"policies": ["fixed:4", "fixed:04"]}))


@pytest.mark.parametrize(
    "replay",
    [
        {"warmup_days": 0},
        {"step_days": 0},
        {"step_days": -7},
        {"lookback_days": 0},
        {"tuning_windows": -1},
    ],
)
def test_non_positive_durations_are_rejected(tmp_path: Path, replay: dict[str, int]) -> None:
    with pytest.raises(ValidationError):
        load_config(_write(tmp_path, _minimal() | {"replay": replay}))


@pytest.mark.parametrize(
    "patch",
    [
        {"eval": {"k": 0}},
        {"data": {"source": "csv"}},
        {"data": {"source": "synthetic", "synthetic": {"n_users": 0}}},
        {"data": {"source": "synthetic", "synthetic": {"drift_strength": -1}}},
        {"data": {"source": "yambda", "yambda": {"size": "1m"}}},
        {"data": {"source": "yambda", "yambda": {"timestamp_unit_seconds": 0}}},
        {"drift": {"n_buckets": 0}},
        {"model": {"name": "two_tower"}},
        {"name": ""},
    ],
)
def test_invalid_values_are_rejected(tmp_path: Path, patch: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        load_config(_write(tmp_path, _minimal() | patch))


def test_nan_threshold_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "c.yaml"
    path.write_text(yaml.safe_dump(_minimal()) + "drift: {thresholds: {new_item_share: .nan}}\n")
    with pytest.raises(ValidationError, match="NaN"):
        load_config(path)


def test_top_level_must_be_a_mapping(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="mapping"):
        load_config(_write(tmp_path, ["not", "a", "mapping"]))


def test_synthetic_settings_mirror_the_generator() -> None:
    """Field names and defaults must match SyntheticConfig, or configs silently diverge."""
    generator_fields = {f.name: f.default for f in dataclasses.fields(SyntheticConfig)}
    settings = SyntheticSettings()
    assert list(SyntheticSettings.model_fields) == list(generator_fields)
    assert settings.model_dump() == generator_fields
    assert settings.to_synthetic_config() == SyntheticConfig()


def test_threshold_settings_mirror_the_detector() -> None:
    detector_fields = {f.name: f.default for f in dataclasses.fields(DriftThresholds)}
    assert ThresholdsConfig().model_dump() == detector_fields


def test_load_interactions_synthetic(tmp_path: Path) -> None:
    raw = _minimal()
    raw["data"]["synthetic"] = {"n_users": 20, "n_items_initial": 40, "n_days": 3, "seed": 1}
    df = load_interactions(load_config(_write(tmp_path, raw)))
    assert df.height > 0
    validate_interactions(df)


def test_load_interactions_yambda_missing_file(tmp_path: Path) -> None:
    raw = _minimal() | {"data": {"source": "yambda", "yambda": {"data_dir": str(tmp_path)}}}
    with pytest.raises(FileNotFoundError):
        load_interactions(load_config(_write(tmp_path, raw)))


@pytest.mark.parametrize("unit", [1, 5])
def test_load_interactions_yambda_threads_timestamp_unit(tmp_path: Path, unit: int) -> None:
    target = tmp_path / "flat" / "50m"
    target.mkdir(parents=True)
    pl.DataFrame(
        {"uid": [1, 2], "item_id": [10, 20], "timestamp": [0, 3], "played_ratio_pct": [100, 100]},
        schema={
            "uid": pl.UInt32,
            "item_id": pl.UInt32,
            "timestamp": pl.UInt32,
            "played_ratio_pct": pl.UInt16,
        },
    ).write_parquet(target / "listens.parquet")
    yambda_section = {"data_dir": str(tmp_path), "timestamp_unit_seconds": unit}
    raw = _minimal() | {"data": {"source": "yambda", "yambda": yambda_section}}
    df = load_interactions(load_config(_write(tmp_path, raw)))
    assert df.get_column(TS).to_list() == [0, 3 * unit]
