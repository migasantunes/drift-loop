"""Drift statistics comparing a reference period with a current period.

Every statistic is computed from interactions alone (no labels, no model), so it can
run the moment a window closes. Inputs are canonical interaction tables (see
``driftloop.data.schema``); the caller is responsible for slicing them by time
through ``replay/clock.py``, which keeps the no-future rule in one place.

Events are counted, not weighted: the question is whether *what people interact
with* has shifted, and a single heavily weighted event should not move a
distribution statistic more than any other event.

The two popularity statistics (PSI, JSD) look only at items the reference knew: they
measure how demand has moved *among known items*. Novelty is measured separately by
``new_item_share``. Mixing the two would let the always-empty unseen bucket of the
reference dominate PSI through its ``eps`` clipping, so PSI would mostly repeat
``new_item_share`` with a value that depends on an arbitrary constant.

Empty inputs give NaN rather than an exception. Before enough data exists a
statistic is undefined, not zero, and NaN makes that visible in the recorded
results while never crossing a threshold.
"""

from __future__ import annotations

import math

import numpy as np
import polars as pl

from driftloop.data.schema import ITEM, USER

_BUCKET = "bucket"
_COUNT = "count"


def _check_n_buckets(n_buckets: int) -> None:
    if n_buckets < 1:
        raise ValueError(f"n_buckets must be >= 1, got {n_buckets}")


def _item_buckets(reference: pl.DataFrame, n_buckets: int) -> pl.DataFrame:
    """Map each reference item to its popularity bucket.

    Items are ranked by reference event count (descending, ties broken by item id so
    the result does not depend on row order). An item goes to bucket
    ``floor(n_buckets * events_before_it / total_events)``: buckets are cut at equal
    shares of reference *events*, not of items, so the head bucket holds a few very
    popular items and the tail buckets many rare ones. Using the cumulative count
    *before* the item puts the most popular item in bucket 0 even when it alone holds
    more than one bucket's worth of events; buckets it spans are then left empty,
    which contributes nothing to PSI or JSD. With fewer items than buckets the same
    rule leaves some buckets empty instead of failing. Integer arithmetic keeps the
    cut points exact.
    """
    counts = (
        reference.group_by(ITEM)
        .agg(pl.len().cast(pl.Int64).alias(_COUNT))
        .sort([_COUNT, ITEM], descending=[True, False])
    )
    total = pl.col(_COUNT).sum()
    before = pl.col(_COUNT).cum_sum() - pl.col(_COUNT)
    return counts.select(
        ITEM,
        ((before * n_buckets) // total).cast(pl.Int64).alias(_BUCKET),
    )


def _shares(bucket_counts: pl.DataFrame, n_buckets: int) -> np.ndarray:
    """Turn a (bucket, count) frame into a dense share vector of length n_buckets + 1."""
    counts = np.zeros(n_buckets + 1, dtype=np.float64)
    idx = bucket_counts.get_column(_BUCKET).to_numpy()
    counts[idx] = bucket_counts.get_column(_COUNT).to_numpy()
    shares: np.ndarray = counts / counts.sum()
    return shares


def popularity_bucket_shares(
    reference: pl.DataFrame, current: pl.DataFrame, *, n_buckets: int = 10
) -> tuple[np.ndarray, np.ndarray]:
    """Share of events per reference-popularity bucket, for reference and current.

    Both vectors have length ``n_buckets + 1``; the last entry is the bucket for items
    absent from ``reference`` (always 0 for the reference itself). Both sum to 1.
    Bucketing by *reference* popularity is what makes the comparison meaningful: a
    shift of attention from head to tail items, or onto brand-new items, shows up as
    mass moving between fixed buckets.

    If either table is empty, both vectors are all NaN.
    """
    _check_n_buckets(n_buckets)
    if reference.height == 0 or current.height == 0:
        nan = np.full(n_buckets + 1, np.nan)
        return nan, nan.copy()

    buckets = _item_buckets(reference, n_buckets)
    ref_counts = (
        reference.join(buckets, on=ITEM, how="inner")
        .group_by(_BUCKET)
        .agg(pl.len().cast(pl.Int64).alias(_COUNT))
    )
    cur_counts = (
        current.join(buckets, on=ITEM, how="left")
        .with_columns(pl.col(_BUCKET).fill_null(n_buckets))
        .group_by(_BUCKET)
        .agg(pl.len().cast(pl.Int64).alias(_COUNT))
    )
    return _shares(ref_counts, n_buckets), _shares(cur_counts, n_buckets)


def known_item_shares(
    reference_shares: np.ndarray, current_shares: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Drop the unseen-item bucket and renormalise current over reference-known items.

    Takes the ``n_buckets + 1`` vectors from ``popularity_bucket_shares`` and returns
    vectors of length ``n_buckets``. The reference needs no renormalising (its unseen
    bucket is 0 by construction). The current shares become "of the events on items
    the reference knew, which share fell in each bucket", which is what separates a
    change in the popularity mix from the arrival of new items. If no current event
    is on a known item (or the input is NaN), the view is undefined and both vectors
    are all NaN.
    """
    ref = np.asarray(reference_shares, dtype=np.float64)[:-1]
    cur = np.asarray(current_shares, dtype=np.float64)[:-1]
    known = cur.sum()
    if np.isnan(ref).any() or not known > 0:
        nan = np.full(ref.shape, np.nan)
        return nan, nan.copy()
    renormalised: np.ndarray = cur / known
    return ref.copy(), renormalised


def psi(reference_shares: np.ndarray, current_shares: np.ndarray, *, eps: float = 1e-4) -> float:
    """Population stability index between two share vectors.

    ``sum((cur - ref) * ln(cur / ref))`` with both shares clipped below at ``eps``.
    Clipping is needed because a popularity bucket can be empty on one side (for
    example a head bucket nobody plays any more, or a bucket left empty by a dominant
    item), and the log would otherwise be infinite. Shares are not renormalised after
    clipping, matching the usual industry definition. NaN input gives NaN.
    """
    ref = np.clip(np.asarray(reference_shares, dtype=np.float64), eps, None)
    cur = np.clip(np.asarray(current_shares, dtype=np.float64), eps, None)
    return float(np.sum((cur - ref) * np.log(cur / ref)))


def jensen_shannon(p: np.ndarray, q: np.ndarray) -> float:
    """Jensen-Shannon divergence in bits, in [0, 1].

    Unlike PSI it needs no ``eps``: each distribution is compared with the mixture
    ``m = (p + q) / 2``, which is positive wherever either input is, so empty buckets
    are handled exactly (``0 * log 0 = 0``). It is also symmetric and bounded, which
    makes a threshold easier to reason about than an unbounded PSI value.
    """
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    if np.isnan(p).any() or np.isnan(q).any():
        return math.nan
    m = 0.5 * (p + q)

    def kl_to_m(x: np.ndarray) -> float:
        mask = x > 0
        return float(np.sum(x[mask] * np.log2(x[mask] / m[mask])))

    value = 0.5 * kl_to_m(p) + 0.5 * kl_to_m(q)
    # Floating-point rounding can push an exact 0 or 1 a hair outside the range.
    return min(max(value, 0.0), 1.0)


def popularity_psi(
    reference: pl.DataFrame, current: pl.DataFrame, *, n_buckets: int = 10, eps: float = 1e-4
) -> float:
    """PSI over the reference-popularity buckets, known items only (see module docstring).

    NaN if either table is empty or no current event is on a reference-known item.
    """
    ref, cur = known_item_shares(*popularity_bucket_shares(reference, current, n_buckets=n_buckets))
    if np.isnan(ref).any():
        return math.nan
    return psi(ref, cur, eps=eps)


def popularity_jsd(reference: pl.DataFrame, current: pl.DataFrame, *, n_buckets: int = 10) -> float:
    """Jensen-Shannon divergence (bits) over the popularity buckets, known items only.

    NaN if either table is empty or no current event is on a reference-known item.
    """
    ref, cur = known_item_shares(*popularity_bucket_shares(reference, current, n_buckets=n_buckets))
    return jensen_shannon(ref, cur)


def _unseen_event_share(reference: pl.DataFrame, current: pl.DataFrame, column: str) -> float:
    if reference.height == 0 or current.height == 0:
        return math.nan
    known = reference.get_column(column).unique().implode()
    share = current.select((~pl.col(column).is_in(known)).mean()).item()
    return float(share)


def new_item_share(reference: pl.DataFrame, current: pl.DataFrame) -> float:
    """Fraction of current events on items the reference never saw; NaN if empty.

    Counted per event, not per distinct item: one new item that takes a large share of
    traffic matters more to a stale model than many new items nobody plays.
    """
    return _unseen_event_share(reference, current, ITEM)


def new_user_share(reference: pl.DataFrame, current: pl.DataFrame) -> float:
    """Fraction of current events from users the reference never saw; NaN if empty.

    Per event, for the same reason as ``new_item_share``: it estimates how much of the
    traffic the current model can only answer with a non-personalised fallback.
    """
    return _unseen_event_share(reference, current, USER)
