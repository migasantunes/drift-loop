"""The quality chart renders headlessly from a results table."""

from __future__ import annotations

import itertools
import math
from pathlib import Path

import matplotlib
import polars as pl
import pytest

from driftloop.replay.clock import WEEK
from driftloop.report.plot import _spread, plot_quality

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def _rows() -> pl.DataFrame:
    recalls = {
        "never": [0.30, 0.25, 0.20, 0.15],
        "always": [0.30, 0.29, 0.30, 0.28],
        "on_drift": [0.30, 0.25, 0.28, math.nan],  # an unscorable window must not break it
    }
    return pl.DataFrame(
        [
            {
                "policy": policy,
                "window": w,
                "start": w * WEEK,
                "end": (w + 1) * WEEK,
                "scored": w >= 1,
                "k": 10,
                "recall": recall,
                "ndcg": recall / 2,
                "retrained": (policy == "always" and w > 0) or (policy == "on_drift" and w == 2),
            }
            for policy, values in recalls.items()
            for w, recall in enumerate(values)
        ]
    )


def test_backend_is_non_interactive() -> None:
    assert matplotlib.get_backend().lower() == "agg"


@pytest.mark.parametrize("metric", ["recall", "ndcg"])
def test_writes_a_png(tmp_path: Path, metric: str) -> None:
    out = plot_quality(_rows(), tmp_path / "sub" / "q.png", metric=metric)
    assert out == tmp_path / "sub" / "q.png"
    data = out.read_bytes()
    assert data.startswith(PNG_MAGIC)
    assert len(data) > 10_000


def test_minimal_columns_suffice(tmp_path: Path) -> None:
    rows = _rows().select("policy", "window", "recall", "retrained")
    assert plot_quality(rows, tmp_path / "q.png").is_file()


def test_unknown_metric(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unknown metric"):
        plot_quality(_rows(), tmp_path / "q.png", metric="accuracy")


def test_missing_columns(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="missing columns"):
        plot_quality(_rows().drop("retrained"), tmp_path / "q.png")


def test_empty_rows(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="no result rows"):
        plot_quality(_rows().clear(), tmp_path / "q.png")


def test_spread_keeps_order_and_gap() -> None:
    values = [0.5, 0.5, 0.1, 0.52]
    placed = _spread(values, 0.05)
    by_value = sorted(range(len(values)), key=lambda i: values[i])
    ordered = [placed[i] for i in by_value]
    assert all(b - a >= 0.05 - 1e-12 for a, b in itertools.pairwise(ordered))
    assert _spread([0.1, 0.5], 0.05) == pytest.approx([0.1, 0.5])
