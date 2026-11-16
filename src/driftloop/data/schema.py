"""The canonical interaction table.

Every dataset (synthetic or real) is converted to this one shape before anything
else touches it, so the replay harness, the models and the drift statistics never
need to know where the data came from.

One row is one positive implicit interaction: user ``user_id`` consumed item
``item_id`` at time ``ts``. Negative or neutral events are dropped by the dataset
adapter, not carried through the system.
"""

from __future__ import annotations

import polars as pl

USER = "user_id"
ITEM = "item_id"
TS = "ts"
WEIGHT = "weight"

#: Column name -> dtype. ``ts`` is integer seconds since the start of the dataset.
INTERACTION_SCHEMA: dict[str, pl.DataType] = {
    USER: pl.UInt32(),
    ITEM: pl.UInt32(),
    TS: pl.Int64(),
    WEIGHT: pl.Float32(),
}


class SchemaError(ValueError):
    """Raised when a frame does not follow the canonical interaction schema."""


def empty_interactions() -> pl.DataFrame:
    """Return an empty frame with the canonical schema."""
    return pl.DataFrame(schema=INTERACTION_SCHEMA)


def validate_interactions(df: pl.DataFrame) -> pl.DataFrame:
    """Check that ``df`` is a canonical interaction table and return it unchanged.

    Checks: exact column set and dtypes, no nulls, non-negative timestamps, and
    rows sorted by ``ts`` (non-decreasing). Sorting is part of the contract because
    the replay harness slices by time and relies on it.
    """
    if set(df.columns) != set(INTERACTION_SCHEMA):
        raise SchemaError(
            f"expected columns {sorted(INTERACTION_SCHEMA)}, got {sorted(df.columns)}"
        )
    for name, dtype in INTERACTION_SCHEMA.items():
        if df.schema[name] != dtype:
            raise SchemaError(f"column {name!r} must be {dtype}, got {df.schema[name]}")
    if df.height == 0:
        return df
    null_count = sum(df.null_count().row(0))
    if null_count:
        raise SchemaError(f"interactions contain {null_count} null values")
    ts = df.get_column(TS)
    ts_min = ts.min()
    assert isinstance(ts_min, int)
    if ts_min < 0:
        raise SchemaError("timestamps must be non-negative seconds since dataset start")
    if not ts.is_sorted():
        raise SchemaError("interactions must be sorted by ts (non-decreasing)")
    return df


def to_canonical(df: pl.DataFrame) -> pl.DataFrame:
    """Cast, order columns and sort by time, then validate.

    Accepts a frame that has the four canonical columns in any order and with any
    compatible numeric dtypes. A missing ``weight`` column is filled with 1.0.
    """
    if WEIGHT not in df.columns:
        df = df.with_columns(pl.lit(1.0).alias(WEIGHT))
    try:
        out = df.select(
            [pl.col(name).cast(dtype) for name, dtype in INTERACTION_SCHEMA.items()]
        ).sort(TS, maintain_order=True)
    except pl.exceptions.PolarsError as exc:
        # For example a negative id that cannot become UInt32, or a missing column.
        raise SchemaError(f"cannot convert to the canonical schema: {exc}") from exc
    return validate_interactions(out)
