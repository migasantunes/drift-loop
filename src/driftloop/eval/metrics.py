"""Top-k ranking metrics on padded recommendation matrices.

Everything here is pure NumPy so the same functions score the popularity baseline
now and the two-tower model later, and so they can be checked against hand-computed
values without any data frame machinery.

Conventions shared by every function:

- ``recs`` is an integer array of shape ``(n_users, k)``; row ``i`` is user ``i``'s
  ranked list, best first, right-padded with ``PAD_ITEM`` when a model ran out of
  candidates. Padding is never a hit and never counts towards coverage.
- ``truth`` holds one 1-D array of relevant item ids per user. Duplicate ids inside a
  user's truth are collapsed: relevance is a set, so listing an item twice must not
  inflate the recall denominator.
- Relevance is binary. Graded relevance (play counts) would reward recommending the
  items a user repeats most, which the protocol deliberately does not target.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from driftloop.models.base import PAD_ITEM


def recall_at_k(recs: np.ndarray, truth: Sequence[np.ndarray]) -> np.ndarray:
    """Per-user share of relevant items that appear in the top k.

    The denominator is the full number of relevant items, not ``min(k, |truth|)``:
    a user with 20 new items and k=10 can reach at most 0.5. That keeps recall
    comparable across k, at the price of a ceiling below 1; NDCG is the metric that
    normalises by what is achievable.
    """
    hits, n_relevant = _hit_matrix(recs, truth)
    recall: np.ndarray = hits.sum(axis=1) / n_relevant
    return recall


def ndcg_at_k(recs: np.ndarray, truth: Sequence[np.ndarray]) -> np.ndarray:
    """Per-user normalised discounted cumulative gain with binary relevance.

    A hit at rank ``r`` (0-based) is worth ``1 / log2(r + 2)``. The ideal ranking puts
    ``min(k, |truth|)`` relevant items first, so a user with more relevant items than
    k can still score 1.0 when the whole list is relevant.
    """
    hits, n_relevant = _hit_matrix(recs, truth)
    k = hits.shape[1]
    discounts = 1.0 / np.log2(np.arange(k, dtype=np.float64) + 2.0)
    dcg = hits @ discounts
    # ideal_cum[m] is the DCG of m hits in the first m slots.
    ideal_cum = np.concatenate(([0.0], np.cumsum(discounts)))
    n_ideal = np.minimum(n_relevant, k).astype(np.int64)
    ndcg: np.ndarray = dcg / ideal_cum[n_ideal]
    return ndcg


def hit_rate_at_k(recs: np.ndarray, truth: Sequence[np.ndarray]) -> np.ndarray:
    """Per-user 1.0 if at least one relevant item is in the top k, else 0.0."""
    hits, _ = _hit_matrix(recs, truth)
    return hits.any(axis=1).astype(np.float64)


def coverage(recs: np.ndarray, n_items_available: int) -> float:
    """Share of the available catalogue that appears in at least one list.

    A popularity model scores well on accuracy while showing everyone the same few
    items; coverage is what exposes that. Padding is excluded so that a model that
    runs out of candidates does not get credit for the pad value.
    """
    recs = _check_recs(recs)
    if n_items_available <= 0:
        raise ValueError(f"n_items_available must be positive, got {n_items_available}")
    distinct = np.unique(recs[recs != PAD_ITEM])
    return float(distinct.size / n_items_available)


def _check_recs(recs: np.ndarray) -> np.ndarray:
    recs = np.asarray(recs)
    if recs.ndim != 2:
        raise ValueError(f"recs must be 2-D (n_users, k), got shape {recs.shape}")
    if recs.size and not np.issubdtype(recs.dtype, np.integer):
        raise ValueError(f"recs must hold integer item ids, got dtype {recs.dtype}")
    return recs.astype(np.int64, copy=False)


def _hit_matrix(recs: np.ndarray, truth: Sequence[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Return a float64 ``(n_users, k)`` hit indicator and the per-user truth size.

    Membership is computed for all users at once: every (row, item) pair is encoded
    as one int64 key and looked up with ``searchsorted`` in the sorted key set of the
    truth. A Python loop with ``np.isin`` per user would be simpler but costs one
    NumPy call per user, which dominates when scoring hundreds of thousands of users.
    """
    recs = _check_recs(recs)
    n_users, k = recs.shape
    if k < 1:
        raise ValueError("recs must have at least one column (k >= 1)")
    if len(truth) != n_users:
        raise ValueError(f"truth has {len(truth)} users but recs has {n_users} rows")
    if n_users == 0:
        return np.zeros((0, k), dtype=np.float64), np.zeros(0, dtype=np.float64)

    truth_arrays = [np.asarray(t) for t in truth]
    for i, t in enumerate(truth_arrays):
        if t.ndim != 1:
            raise ValueError(f"truth[{i}] must be 1-D, got shape {t.shape}")
        if t.size == 0:
            raise ValueError(f"truth[{i}] is empty; users without relevant items must be dropped")
    lengths = np.fromiter((t.size for t in truth_arrays), dtype=np.int64, count=n_users)
    truth_items = np.concatenate(truth_arrays).astype(np.int64)
    if (truth_items < 0).any():
        raise ValueError("truth item ids must be non-negative")

    real = recs != PAD_ITEM
    rec_items = recs[real]
    if rec_items.size and (rec_items < 0).any():
        raise ValueError(f"recs contain negative ids other than PAD_ITEM ({PAD_ITEM})")
    _check_no_duplicates(recs)

    # Key = row * span + item. span exceeds every item id, so keys of different rows
    # never collide; the bound check keeps the encoding inside int64.
    max_item = int(np.max(truth_items))
    if rec_items.size:
        max_item = max(max_item, int(np.max(rec_items)))
    span = max_item + 1
    if n_users * span >= np.iinfo(np.int64).max:
        raise ValueError("too many users times item id range to encode as int64 keys")
    truth_rows = np.repeat(np.arange(n_users, dtype=np.int64), lengths)
    truth_keys = np.unique(truth_rows * span + truth_items)
    n_relevant = np.bincount(truth_keys // span, minlength=n_users).astype(np.float64)

    rows = np.broadcast_to(np.arange(n_users, dtype=np.int64)[:, None], recs.shape)
    rec_keys = rows * span + np.where(real, recs, 0)
    pos = np.searchsorted(truth_keys, rec_keys)
    pos = np.minimum(pos, truth_keys.size - 1)
    hits = (truth_keys[pos] == rec_keys) & real
    return hits.astype(np.float64), n_relevant


def _check_no_duplicates(recs: np.ndarray) -> None:
    """Reject rows that list the same item twice.

    The ``Recommender`` contract forbids duplicates; if a model broke it, a repeated
    relevant item would be counted twice and recall could exceed 1. Failing loudly is
    better than reporting a quietly inflated number.
    """
    if recs.shape[1] < 2:
        return
    ordered = np.sort(recs, axis=1)
    repeated = (ordered[:, 1:] == ordered[:, :-1]) & (ordered[:, 1:] != PAD_ITEM)
    if repeated.any():
        row = int(np.flatnonzero(repeated.any(axis=1))[0])
        raise ValueError(f"recs row {row} contains duplicate item ids")
