"""Scoring one replay window, as specified in ``docs/EVALUATION.md``.

The harness, not the model, decides who is scored, what counts as relevant and what
the user has already seen. Keeping those choices here means every model is graded
by exactly the same rules, and a model cannot make itself look better by, say,
recommending the user's own history back to them.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import polars as pl

from driftloop.data.schema import ITEM, USER
from driftloop.eval.metrics import coverage, hit_rate_at_k, ndcg_at_k, recall_at_k
from driftloop.models.base import Recommender, SeenItems


@dataclass(frozen=True, slots=True)
class WindowResult:
    """Mean metrics over the users scored in one window.

    ``n_users`` is reported next to the means because a window with a handful of
    evaluable users gives a noisy number; the replay curve should be read with it.
    """

    k: int
    n_users: int
    recall: float
    ndcg: float
    hit_rate: float
    coverage: float


def build_ground_truth(
    history: pl.DataFrame, future: pl.DataFrame
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Per user in ``future``, the distinct items that are new to that user.

    "New" means the (user, item) pair never occurs in ``history``. Repeats are
    excluded because most music plays are repeats, and a model that echoes a user's
    history would otherwise look strong while recommending nothing new. Users whose
    window activity is all repeats have no ground truth and are dropped.

    Returns ``(user_ids, truth)`` aligned, users ascending, each truth array sorted.
    """
    pairs = future.select(USER, ITEM).unique()
    seen_pairs = history.select(USER, ITEM).unique()
    new_pairs = pairs.join(seen_pairs, on=[USER, ITEM], how="anti").sort(USER, ITEM)
    if new_pairs.height == 0:
        return np.empty(0, dtype=np.int64), []

    user_col = new_pairs.get_column(USER).to_numpy().astype(np.int64)
    items = new_pairs.get_column(ITEM).to_numpy().astype(np.int64)
    # Rows are sorted by user, so each user's items form one contiguous run.
    starts = np.flatnonzero(np.r_[True, user_col[1:] != user_col[:-1]])
    user_ids = user_col[starts]
    truth = np.split(items, starts[1:])
    return user_ids, truth


def evaluate_window(
    model: Recommender, *, history: pl.DataFrame, future: pl.DataFrame, k: int
) -> WindowResult:
    """Score ``model`` on the window ``future`` given everything visible before it.

    The model only receives the user ids to score and a ``SeenItems`` built from
    ``history``; ``future`` is used solely to decide who is scored and what they
    went on to consume. Coverage is measured against the items that exist in
    ``history``, since nothing else could have been recommended at that time.
    """
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    user_ids, truth = build_ground_truth(history, future)
    if user_ids.size == 0:
        nan = math.nan
        return WindowResult(k=k, n_users=0, recall=nan, ndcg=nan, hit_rate=nan, coverage=nan)

    seen = SeenItems.from_interactions(history)
    recs = np.asarray(model.recommend(user_ids, k, seen=seen))
    if recs.shape != (user_ids.size, k):
        raise ValueError(
            f"model {model.name!r} returned shape {recs.shape}, expected {(user_ids.size, k)}"
        )

    n_items = history.get_column(ITEM).n_unique()
    return WindowResult(
        k=k,
        n_users=int(user_ids.size),
        recall=float(recall_at_k(recs, truth).mean()),
        ndcg=float(ndcg_at_k(recs, truth).mean()),
        hit_rate=float(hit_rate_at_k(recs, truth).mean()),
        coverage=coverage(recs, n_items) if n_items > 0 else math.nan,
    )
