"""The driftloop command, end to end on the small synthetic config."""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl
import yaml
from typer.testing import CliRunner

from driftloop.cli import app
from driftloop.data.schema import validate_interactions

CONFIGS = Path(__file__).resolve().parents[1] / "configs"
SMALL = CONFIGS / "synthetic-small.yaml"
PNG_MAGIC = b"\x89PNG\r\n\x1a\n"

runner = CliRunner()


def test_simulate_then_plot(tmp_path: Path) -> None:
    out = tmp_path / "run"
    result = runner.invoke(app, ["simulate", "--config", str(SMALL), "--out", str(out)])
    assert result.exit_code == 0, result.output
    for name in ("results.csv", "summary.json", "config.yaml"):
        assert (out / name).is_file()
    summary = json.loads((out / "summary.json").read_text())
    for policy in summary["policies"]:
        assert policy in result.output
    assert "quality_kept" in result.output

    result = runner.invoke(app, ["plot", str(out)])
    assert result.exit_code == 0, result.output
    png = (out / "quality.png").read_bytes()
    assert png.startswith(PNG_MAGIC)
    assert len(png) > 10_000

    result = runner.invoke(app, ["plot", str(out), "--metric", "ndcg"])
    assert result.exit_code == 0, result.output


def test_plot_without_results_fails(tmp_path: Path) -> None:
    result = runner.invoke(app, ["plot", str(tmp_path)])
    assert result.exit_code != 0
    assert not (tmp_path / "quality.png").exists()


def test_synth_writes_canonical_parquet(tmp_path: Path) -> None:
    out = tmp_path / "data" / "interactions.parquet"
    result = runner.invoke(app, ["synth", "--config", str(SMALL), "--out", str(out)])
    assert result.exit_code == 0, result.output
    df = validate_interactions(pl.read_parquet(out))
    assert df.height > 0


def test_synth_rejects_a_non_synthetic_config(tmp_path: Path) -> None:
    cfg = tmp_path / "y.yaml"
    cfg.write_text(
        yaml.safe_dump({"name": "y", "data": {"source": "yambda"}, "policies": ["never"]})
    )
    result = runner.invoke(app, ["synth", "--config", str(cfg), "--out", str(tmp_path / "x")])
    assert result.exit_code != 0


def test_bad_config_fails_before_running(tmp_path: Path) -> None:
    cfg = tmp_path / "bad.yaml"
    cfg.write_text(SMALL.read_text().replace("tuning_windows:", "tuning_window:"))
    result = runner.invoke(app, ["simulate", "--config", str(cfg), "--out", str(tmp_path / "r")])
    assert result.exit_code != 0
    assert not (tmp_path / "r").exists()


def test_yambda_download_rejects_unknown_size(tmp_path: Path) -> None:
    # The size is checked before any import or network access.
    result = runner.invoke(app, ["yambda-download", "--size", "1m", "--out", str(tmp_path)])
    assert result.exit_code == 1
    assert "unknown Yambda size" in result.output


def test_help_lists_every_command() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for command in ("simulate", "plot", "synth", "yambda-download"):
        assert command in result.output
