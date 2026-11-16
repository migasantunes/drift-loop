"""Tests for the canonical interaction schema contract."""

from __future__ import annotations

import polars as pl
import pytest

from driftloop.data.schema import (
    INTERACTION_SCHEMA,
    ITEM,
    TS,
    USER,
    WEIGHT,
    SchemaError,
    empty_interactions,
    to_canonical,
    validate_interactions,
)


def _valid() -> pl.DataFrame:
    return pl.DataFrame(
        {USER: [1, 2, 1], ITEM: [10, 11, 12], TS: [0, 5, 5], WEIGHT: [1.0, 1.0, 2.0]},
        schema=INTERACTION_SCHEMA,
    )


def test_accepts_valid_frame() -> None:
    df = _valid()
    assert validate_interactions(df) is df


def test_empty_frame_is_valid() -> None:
    df = empty_interactions()
    assert df.height == 0
    assert dict(df.schema) == INTERACTION_SCHEMA
    validate_interactions(df)


def test_rejects_missing_column() -> None:
    with pytest.raises(SchemaError, match="expected columns"):
        validate_interactions(_valid().drop(WEIGHT))


def test_rejects_extra_column() -> None:
    with pytest.raises(SchemaError, match="expected columns"):
        validate_interactions(_valid().with_columns(pl.lit(0).alias("extra")))


def test_rejects_wrong_dtype() -> None:
    with pytest.raises(SchemaError, match="'ts' must be"):
        validate_interactions(_valid().with_columns(pl.col(TS).cast(pl.Int32)))


def test_rejects_nulls() -> None:
    df = _valid().with_columns(
        pl.when(pl.col(ITEM) == 11).then(None).otherwise(pl.col(ITEM)).alias(ITEM)
    )
    with pytest.raises(SchemaError, match="null"):
        validate_interactions(df)


def test_rejects_negative_ts() -> None:
    df = _valid().with_columns((pl.col(TS) - 1).alias(TS))
    with pytest.raises(SchemaError, match="non-negative"):
        validate_interactions(df)


def test_rejects_unsorted_ts() -> None:
    df = _valid().reverse()
    with pytest.raises(SchemaError, match="sorted"):
        validate_interactions(df)


def test_to_canonical_fills_weight_casts_orders_and_sorts() -> None:
    raw = pl.DataFrame({TS: [30, 10, 20, 10], ITEM: [3, 1, 2, 4], USER: [7, 8, 9, 6]})
    out = to_canonical(raw)
    assert out.columns == list(INTERACTION_SCHEMA)
    assert dict(out.schema) == INTERACTION_SCHEMA
    assert out.get_column(TS).to_list() == [10, 10, 20, 30]
    # Stable sort: equal timestamps keep their input order.
    assert out.get_column(ITEM).to_list() == [1, 4, 2, 3]
    assert out.get_column(WEIGHT).to_list() == [1.0] * 4


def test_to_canonical_keeps_existing_weight() -> None:
    raw = pl.DataFrame({USER: [1], ITEM: [2], TS: [3], WEIGHT: [0.5]})
    assert to_canonical(raw).get_column(WEIGHT).to_list() == [0.5]
