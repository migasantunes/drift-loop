"""The contract every recommender in this repo follows.

Keeping this small is deliberate: the replay harness, the retrain loop and (later)
the serving layer only ever talk to a model through this protocol, so a popularity
baseline and a two-tower network are interchangeable.
"""

from __future__ import annotations

from typing import Protocol, Self, runtime_checkable

import numpy as np
import polars as pl

from driftloop.data.schema import ITEM, USER

#: Value used to pad a recommendation row when a model has fewer than k candidates.
PAD_ITEM = -1


class SeenItems:
    """Which items each user has already interacted with.

    Filtering out already-seen items is done by the caller (the harness now, the
    serving layer later), never baked into a model. A model that was trained two
    weeks ago must still not recommend what the user consumed yesterday, and only
    the caller has that fresh history.
    """

    def __init__(self, by_user: dict[int, np.ndarray]) -> None:
        self._by_user = by_user

    @classmethod
    def from_interactions(cls, interactions: pl.DataFrame) -> SeenItems:
        # Stay in Polars/NumPy: the real table has tens of millions of rows, so the
        # pairs are never turned into Python objects. Each user's items are a view
        # into one shared, sorted array.
        pairs = interactions.select(USER, ITEM).unique().sort(USER, ITEM)
        if pairs.height == 0:
            return cls({})
        users = pairs.get_column(USER).to_numpy()
        items = pairs.get_column(ITEM).cast(pl.Int64).to_numpy()
        starts = np.flatnonzero(np.r_[True, users[1:] != users[:-1]])
        ends = np.r_[starts[1:], users.size]
        return cls({int(users[s]): items[s:e] for s, e in zip(starts, ends, strict=True)})

    def get(self, user_id: int) -> np.ndarray:
        """Sorted array of item ids the user has seen (empty for an unknown user)."""
        return self._by_user.get(int(user_id), _EMPTY)

    def __len__(self) -> int:
        return len(self._by_user)


_EMPTY = np.empty(0, dtype=np.int64)


@runtime_checkable
class Recommender(Protocol):
    """A model that can be fitted on history and asked for top-k items per user."""

    #: Short identifier used in result tables, e.g. ``"popularity"``.
    name: str

    #: Simulated time the model was trained up to (exclusive); ``None`` before ``fit``.
    trained_until: int | None

    def fit(self, interactions: pl.DataFrame, *, until: int) -> Self:
        """Train on ``interactions``, all of which must have ``ts < until``.

        Implementations must raise ``ValueError`` if any row has ``ts >= until``:
        that would be training on the future.
        """
        ...

    def recommend(
        self, user_ids: np.ndarray, k: int, *, seen: SeenItems | None = None
    ) -> np.ndarray:
        """Return an int64 array of shape ``(len(user_ids), k)`` with item ids.

        Row ``i`` is the ranked list for ``user_ids[i]``, best first, with no
        duplicates and none of the user's ``seen`` items. Users the model has never
        encountered must still get recommendations (a non-personalised fallback).
        Rows with fewer than ``k`` candidates are right-padded with ``PAD_ITEM``.
        """
        ...
