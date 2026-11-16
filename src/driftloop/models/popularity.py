"""Global popularity baseline.

Every later model has to beat this. In music it is a strong baseline: popular
tracks are popular because many people want them. In a drift study it is also the
clearest case of staleness, since a frozen popularity list stops tracking what is
trending, and that decay is exactly what the retrain policies are meant to fix.
"""

from __future__ import annotations

from typing import Self

import numpy as np
import polars as pl

from driftloop.data.schema import ITEM, TS, WEIGHT
from driftloop.models.base import PAD_ITEM, SeenItems

_SCORE = "_score"


class PopularityRecommender:
    """Recommends the items with the largest total interaction weight.

    Weight rather than raw count is summed so that the dataset adapter's notion of
    interaction strength (for instance a full listen versus a skip-adjacent play)
    carries through; with unit weights the two are the same.
    """

    name = "popularity"

    def __init__(self) -> None:
        self.trained_until: int | None = None
        self._ranking: np.ndarray | None = None

    def fit(self, interactions: pl.DataFrame, *, until: int) -> Self:
        """Rank items by summed weight, ties broken by smaller item id.

        The explicit tie-break makes the ranking a function of the data alone, so two
        runs of the replay produce identical lists and identical metrics.
        """
        if interactions.height:
            ts_max = interactions.get_column(TS).max()
            assert isinstance(ts_max, int)
            if ts_max >= until:
                raise ValueError(
                    f"training data reaches ts={ts_max}, which is not before until={until}"
                )
        ranked = (
            interactions.group_by(ITEM)
            .agg(pl.col(WEIGHT).cast(pl.Float64).sum().alias(_SCORE))
            .sort([_SCORE, ITEM], descending=[True, False])
        )
        self._ranking = ranked.get_column(ITEM).to_numpy().astype(np.int64)
        self.trained_until = until
        return self

    def recommend(
        self, user_ids: np.ndarray, k: int, *, seen: SeenItems | None = None
    ) -> np.ndarray:
        """Top-k of the global ranking per user, minus that user's seen items.

        The ranking is sorted once in ``fit``. For a user who has seen ``s`` items, the
        answer lies in the first ``k + s`` ranked items, because at most ``s`` of them
        can be removed; only that prefix is filtered, so the cost per user is
        ``O((k + s) log s)`` instead of a re-sort of the catalogue.
        """
        if self._ranking is None:
            raise RuntimeError("recommend() called before fit()")
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")
        ranking = self._ranking
        user_ids = np.asarray(user_ids)
        out = np.full((user_ids.size, k), PAD_ITEM, dtype=np.int64)
        default = ranking[:k]
        out[:, : default.size] = default
        if seen is None:
            return out

        for row, user in enumerate(user_ids):
            user_seen = seen.get(int(user))
            if user_seen.size == 0:
                continue
            prefix = ranking[: k + user_seen.size]
            kept = prefix[~np.isin(prefix, user_seen)][:k]
            out[row, : kept.size] = kept
            out[row, kept.size :] = PAD_ITEM
        return out
