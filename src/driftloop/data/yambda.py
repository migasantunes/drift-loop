"""Yambda raw listens -> canonical interaction table, and a download helper.

This is the only module that knows Yambda's column names and units; everything
downstream sees :data:`driftloop.data.schema.INTERACTION_SCHEMA`. See
``docs/DATASET.md`` for the column semantics and the mapping decisions.

The transform stays lazy until the final collect so that, on the 46 M-row table,
Polars can push the filters into the Parquet scan and only materialise the four
canonical columns.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import polars as pl

from driftloop.data.schema import (
    INTERACTION_SCHEMA,
    ITEM,
    TS,
    USER,
    WEIGHT,
    SchemaError,
    validate_interactions,
)

REPO_ID = "yandex/yambda"
SIZES = frozenset({"50m", "500m", "5b"})

RAW_USER = "uid"
RAW_ITEM = "item_id"
RAW_TS = "timestamp"
RAW_ORGANIC = "is_organic"
RAW_PLAYED = "played_ratio_pct"


def _check_size(size: str) -> None:
    if size not in SIZES:
        raise ValueError(f"unknown Yambda size {size!r}; expected one of {sorted(SIZES)}")


def listens_to_interactions(
    listens: pl.DataFrame | pl.LazyFrame,
    *,
    min_played_ratio_pct: int = 50,
    organic_only: bool = False,
    organic_value: int = 1,
    timestamp_unit_seconds: int = 1,
) -> pl.DataFrame:
    """Turn raw Yambda listens into canonical positive interactions.

    A listen is positive when ``played_ratio_pct >= min_played_ratio_pct``: a track
    skipped after a few seconds says little about liking it. ``organic_value`` is a
    parameter because the dataset card contradicts itself on which value of
    ``is_organic`` means organic (open question 1 in ``docs/DATASET.md``); the
    default follows the card's field table until M1 settles it.

    ``timestamp_unit_seconds`` is a parameter because the sources disagree on the raw
    unit. The dataset card says delta times are "binned into 5s units", which reads as
    a count of 5-second units (factor 5); the Yambda paper's formula floors
    ``T_event - T_start`` to a multiple of 5, which means the stored value is already in
    seconds (factor 1, the default). The data settles it: the replayed span is roughly
    eleven months, so the maximum raw timestamp is about 2.9e7 if it is in seconds and
    about 5.7e6 if it counts 5-second units. Check that maximum (and that raw values
    are multiples of 5 under the paper's reading) before trusting any window length.

    ``timestamp`` is widened to Int64 before scaling: raw values are uint32, and
    multiplying in uint32 would silently wrap for any value above
    ``2**32 / timestamp_unit_seconds``.
    """
    if min_played_ratio_pct < 0:
        raise ValueError("min_played_ratio_pct must be >= 0")
    if organic_value not in (0, 1):
        raise ValueError("organic_value must be 0 or 1")
    if timestamp_unit_seconds < 1:
        raise ValueError("timestamp_unit_seconds must be >= 1")

    lf = listens.lazy()
    required = {RAW_USER, RAW_ITEM, RAW_TS, RAW_PLAYED}
    if organic_only:
        required.add(RAW_ORGANIC)
    missing = required - set(lf.collect_schema().names())
    if missing:
        raise SchemaError(f"Yambda listens are missing columns {sorted(missing)}")

    lf = lf.filter(pl.col(RAW_PLAYED) >= min_played_ratio_pct)
    if organic_only:
        lf = lf.filter(pl.col(RAW_ORGANIC) == organic_value)
    out = (
        lf.select(
            pl.col(RAW_USER).cast(INTERACTION_SCHEMA[USER]).alias(USER),
            pl.col(RAW_ITEM).cast(INTERACTION_SCHEMA[ITEM]).alias(ITEM),
            (pl.col(RAW_TS).cast(pl.Int64) * timestamp_unit_seconds)
            .cast(INTERACTION_SCHEMA[TS])
            .alias(TS),
            pl.lit(1.0, dtype=INTERACTION_SCHEMA[WEIGHT]).alias(WEIGHT),
        )
        .sort(TS, maintain_order=True)
        .collect()
    )
    return validate_interactions(out)


def download(size: str, out_dir: Path, files: Sequence[str] = ("listens.parquet",)) -> list[Path]:
    """Fetch ``flat/<size>/<file>`` for each file from the Hub into ``out_dir``.

    Files land at ``out_dir/flat/<size>/<file>``, the layout :func:`load_interactions`
    expects. ``huggingface_hub`` is imported here, not at module level, so the rest
    of the package works without the optional ``data`` extra.
    """
    _check_size(size)
    if not files:
        raise ValueError("files must name at least one file")
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise ImportError(
            "downloading Yambda needs huggingface_hub; install it with `uv sync --extra data`"
        ) from exc

    paths: list[Path] = []
    for name in files:
        local = hf_hub_download(
            repo_id=REPO_ID,
            repo_type="dataset",
            filename=f"flat/{size}/{name}",
            local_dir=out_dir,
        )
        paths.append(Path(local))
    return paths


def load_interactions(
    data_dir: Path,
    size: str = "50m",
    *,
    min_played_ratio_pct: int = 50,
    organic_only: bool = False,
    organic_value: int = 1,
    timestamp_unit_seconds: int = 1,
) -> pl.DataFrame:
    """Scan ``<data_dir>/flat/<size>/listens.parquet`` and convert it to canonical form.

    The keyword arguments are those of :func:`listens_to_interactions`.
    """
    _check_size(size)
    path = data_dir / "flat" / size / "listens.parquet"
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} not found; fetch it with "
            f"driftloop.data.yambda.download({size!r}, {str(data_dir)!r})"
        )
    return listens_to_interactions(
        pl.scan_parquet(path),
        min_played_ratio_pct=min_played_ratio_pct,
        organic_only=organic_only,
        organic_value=organic_value,
        timestamp_unit_seconds=timestamp_unit_seconds,
    )
