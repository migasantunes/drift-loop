"""Tests for the Yambda adapter, built from hand-made frames (no real data, no network)."""

from __future__ import annotations

import sys
import types
from pathlib import Path
from typing import Any

import polars as pl
import pytest

from driftloop.data import yambda
from driftloop.data.schema import INTERACTION_SCHEMA, ITEM, TS, USER, WEIGHT, SchemaError


def _raw() -> pl.DataFrame:
    """Five listens with raw Yambda dtypes, deliberately out of time order."""
    return pl.DataFrame(
        {
            "uid": [1, 2, 3, 4, 5],
            "item_id": [10, 20, 30, 40, 50],
            "timestamp": [7, 0, 3, 2, 1],
            "is_organic": [1, 0, 1, 0, 1],
            "played_ratio_pct": [100, 50, 49, 80, 0],
            "track_length_seconds": [200, 180, 240, 300, 120],
        },
        schema={
            "uid": pl.UInt32,
            "item_id": pl.UInt32,
            "timestamp": pl.UInt32,
            "is_organic": pl.UInt8,
            "played_ratio_pct": pl.UInt16,
            "track_length_seconds": pl.UInt32,
        },
    )


def test_returns_canonical_sorted_table() -> None:
    out = yambda.listens_to_interactions(_raw())
    assert out.columns == list(INTERACTION_SCHEMA)
    assert dict(out.schema) == INTERACTION_SCHEMA
    assert out.get_column(TS).is_sorted()
    assert out.get_column(WEIGHT).to_list() == [1.0] * out.height


def _ts_by_item(out: pl.DataFrame) -> dict[int, int]:
    return dict(zip(out.get_column(ITEM).to_list(), out.get_column(TS).to_list(), strict=True))


def test_timestamp_default_unit_is_seconds() -> None:
    out = yambda.listens_to_interactions(_raw())
    assert _ts_by_item(out) == {20: 0, 40: 2, 10: 7}


def test_timestamp_scaled_by_unit() -> None:
    out = yambda.listens_to_interactions(_raw(), timestamp_unit_seconds=5)
    assert _ts_by_item(out) == {20: 0, 40: 10, 10: 35}


@pytest.mark.parametrize("unit", [1, 5])
def test_large_timestamp_does_not_overflow(unit: int) -> None:
    raw = _raw().head(1).with_columns(pl.lit(2**32 - 1, dtype=pl.UInt32).alias("timestamp"))
    out = yambda.listens_to_interactions(raw, timestamp_unit_seconds=unit)
    assert out.get_column(TS).to_list() == [(2**32 - 1) * unit]


def test_played_ratio_threshold_includes_boundary() -> None:
    out = yambda.listens_to_interactions(_raw())
    # 50 is kept (boundary), 49 and 0 are dropped.
    assert sorted(out.get_column(ITEM).to_list()) == [10, 20, 40]
    strict = yambda.listens_to_interactions(_raw(), min_played_ratio_pct=51)
    assert sorted(strict.get_column(ITEM).to_list()) == [10, 40]
    everything = yambda.listens_to_interactions(_raw(), min_played_ratio_pct=0)
    assert everything.height == 5


def test_organic_filter_follows_organic_value() -> None:
    ones = yambda.listens_to_interactions(_raw(), organic_only=True, organic_value=1)
    assert ones.get_column(USER).to_list() == [1]
    zeros = yambda.listens_to_interactions(_raw(), organic_only=True, organic_value=0)
    assert sorted(zeros.get_column(USER).to_list()) == [2, 4]


def test_organic_value_ignored_without_organic_only() -> None:
    a = yambda.listens_to_interactions(_raw(), organic_value=0)
    b = yambda.listens_to_interactions(_raw(), organic_value=1)
    assert a.equals(b)


def test_lazy_and_eager_input_agree() -> None:
    eager = yambda.listens_to_interactions(_raw())
    lazy = yambda.listens_to_interactions(_raw().lazy())
    assert eager.equals(lazy)


def test_missing_raw_column_rejected() -> None:
    with pytest.raises(SchemaError, match="played_ratio_pct"):
        yambda.listens_to_interactions(_raw().drop("played_ratio_pct"))
    with pytest.raises(SchemaError, match="is_organic"):
        yambda.listens_to_interactions(_raw().drop("is_organic"), organic_only=True)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"min_played_ratio_pct": -1},
        {"organic_value": 2},
        {"timestamp_unit_seconds": 0},
        {"timestamp_unit_seconds": -5},
    ],
)
def test_invalid_arguments_rejected(kwargs: dict[str, int]) -> None:
    with pytest.raises(ValueError, match="must be"):
        yambda.listens_to_interactions(_raw(), **kwargs)  # type: ignore[arg-type]


def test_load_interactions_reads_expected_path(tmp_path: Path) -> None:
    target = tmp_path / "flat" / "50m"
    target.mkdir(parents=True)
    _raw().write_parquet(target / "listens.parquet")
    kwargs: dict[str, Any] = {"organic_only": True, "organic_value": 0}
    out = yambda.load_interactions(tmp_path, **kwargs)
    assert out.equals(yambda.listens_to_interactions(_raw(), **kwargs))
    scaled = yambda.load_interactions(tmp_path, timestamp_unit_seconds=5)
    assert _ts_by_item(scaled) == {20: 0, 40: 10, 10: 35}


def test_load_interactions_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match=r"listens\.parquet"):
        yambda.load_interactions(tmp_path)


@pytest.mark.parametrize("size", ["50M", "1b", ""])
def test_bad_size_rejected(size: str, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unknown Yambda size"):
        yambda.download(size, tmp_path)
    with pytest.raises(ValueError, match="unknown Yambda size"):
        yambda.load_interactions(tmp_path, size)


def test_download_uses_hub_layout(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls: list[dict[str, Any]] = []

    def fake_download(**kwargs: Any) -> str:
        calls.append(kwargs)
        return str(Path(kwargs["local_dir"]) / kwargs["filename"])

    fake = types.ModuleType("huggingface_hub")
    fake.hf_hub_download = fake_download  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake)

    paths = yambda.download("500m", tmp_path, files=("listens.parquet", "likes.parquet"))
    assert paths == [
        tmp_path / "flat" / "500m" / "listens.parquet",
        tmp_path / "flat" / "500m" / "likes.parquet",
    ]
    assert [c["filename"] for c in calls] == [
        "flat/500m/listens.parquet",
        "flat/500m/likes.parquet",
    ]
    assert all(c["repo_id"] == "yandex/yambda" and c["repo_type"] == "dataset" for c in calls)


def test_download_without_huggingface_hub(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    # A None entry in sys.modules makes the import raise ImportError.
    monkeypatch.setitem(sys.modules, "huggingface_hub", None)
    with pytest.raises(ImportError, match="uv sync --extra data"):
        yambda.download("50m", tmp_path)


def test_download_rejects_empty_file_list(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="at least one"):
        yambda.download("50m", tmp_path, files=())
