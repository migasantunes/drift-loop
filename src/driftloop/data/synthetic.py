"""Synthetic interactions with controllable drift.

The real dataset tells us how fast *its* recommender goes stale, but not why. A
generator whose non-stationarity is a single dial (``drift_strength``) lets the loop
be tested against known ground truth: at ``drift_strength = 0`` a frozen model must
not fall behind a retrained one, and as the dial goes up the gap between retraining
policies must open. (Absolute recall still declines over time at 0, for every policy:
ground truth is new-to-user items, which get harder to hit as histories grow.)

Generative story, one simulated day at a time:

1. Every item belongs to one latent cluster (think genre). Within a cluster, items
   have a long-tailed (Zipf-like) base popularity, so a few hits dominate.
2. Every user has a preference distribution over clusters (Dirichlet).
3. For each active user, the number of events that day is Poisson. Each event picks
   a cluster from ``user preference x global cluster trend`` and then an item from
   that cluster in proportion to the item's current weight.

Non-stationarity, every part scaled by ``drift_strength``:

- (a) new items are released daily, with a freshness boost that decays over a couple
  of weeks, so the catalogue churns and recent releases are over-represented;
- (b) global cluster popularity follows a log-space random walk (trends);
- (c) each user's cluster preferences follow a slow log-space random walk (taste);
- (d) new users join over time.

With ``drift_strength = 0`` all four switch off and the process is stationary: the
same users and items, with the same probabilities, every day.

The work is vectorised per day: one simulated day is a handful of NumPy operations
over all of that day's events, never a Python loop over events or users.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl

from driftloop.data.schema import ITEM, TS, USER, to_canonical
from driftloop.replay.clock import DAY


@dataclass(frozen=True)
class SyntheticConfig:
    """Parameters of the synthetic interaction process.

    Rates per day are expectations of Poisson counts. Every non-stationary effect is
    multiplied by ``drift_strength``; the remaining fields shape the stationary part.
    """

    #: Users present from day 0.
    n_users: int = 2000
    #: Expected number of users joining per day (times ``drift_strength``).
    n_new_users_per_day: float = 5.0
    #: Items present from day 0. Must be at least ``n_clusters`` so no cluster is empty.
    n_items_initial: int = 3000
    #: Expected number of items released per day (times ``drift_strength``).
    n_new_items_per_day: float = 40.0
    #: Number of latent item clusters.
    n_clusters: int = 20
    #: Length of the simulated period in days.
    n_days: int = 120
    #: Expected events per active user per day.
    events_per_user_per_day: float = 3.0
    #: Master dial for all non-stationarity; 0 gives a stationary process.
    drift_strength: float = 1.0
    #: Seed of the NumPy generator; the output is a pure function of the config.
    seed: int = 0
    #: Dirichlet concentration of user cluster preferences. Small values make users
    #: focus on a few clusters, which is what makes personalisation worth anything.
    preference_concentration: float = 0.3
    #: Zipf exponent of base item popularity within a cluster (0 = uniform).
    item_popularity_exponent: float = 1.0
    #: Peak multiplicative boost of a new release over its base weight, before
    #: scaling by ``drift_strength``: weight x (1 + boost) on release day.
    freshness_boost: float = 20.0
    #: e-folding time of the freshness boost in days ("a couple of weeks" to fade).
    freshness_decay_days: float = 5.0
    #: Daily standard deviation of the log-space cluster trend walk.
    trend_volatility: float = 0.08
    #: Daily standard deviation of the log-space per-user taste walk.
    taste_volatility: float = 0.05

    def __post_init__(self) -> None:
        positive_ints = {
            "n_users": self.n_users,
            "n_items_initial": self.n_items_initial,
            "n_clusters": self.n_clusters,
            "n_days": self.n_days,
        }
        for name, value in positive_ints.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}")
        if self.n_items_initial < self.n_clusters:
            raise ValueError("n_items_initial must be >= n_clusters so every cluster has an item")
        positive = {
            "events_per_user_per_day": self.events_per_user_per_day,
            "preference_concentration": self.preference_concentration,
            "freshness_decay_days": self.freshness_decay_days,
        }
        for name, fvalue in positive.items():
            if not fvalue > 0:
                raise ValueError(f"{name} must be positive, got {fvalue}")
        non_negative = {
            "n_new_users_per_day": self.n_new_users_per_day,
            "n_new_items_per_day": self.n_new_items_per_day,
            "drift_strength": self.drift_strength,
            "item_popularity_exponent": self.item_popularity_exponent,
            "freshness_boost": self.freshness_boost,
            "trend_volatility": self.trend_volatility,
            "taste_volatility": self.taste_volatility,
        }
        for name, fvalue in non_negative.items():
            if not fvalue >= 0:
                raise ValueError(f"{name} must be >= 0, got {fvalue}")


def _arrival_days(rng: np.random.Generator, rate: float, n_days: int) -> np.ndarray:
    """Day of arrival for each newcomer: Poisson(rate) arrivals on each day after day 0.

    Day 0 is excluded because the initial population already covers it. Returned in
    non-decreasing order so ids assigned in this order are in order of arrival.
    """
    if rate == 0 or n_days < 2:
        return np.empty(0, dtype=np.int64)
    counts = rng.poisson(rate, size=n_days - 1)
    return np.repeat(np.arange(1, n_days, dtype=np.int64), counts)


def _zipf_weights(ranks: np.ndarray, exponent: float) -> np.ndarray:
    return np.asarray(ranks, dtype=np.float64) ** -exponent


def generate_interactions(cfg: SyntheticConfig) -> pl.DataFrame:
    """Generate a canonical interaction table from ``cfg``.

    User and item ids are assigned in order of arrival, so ids at or above
    ``n_users`` / ``n_items_initial`` are exactly the newcomers. ``ts`` is seconds
    since the start, uniform within each day.
    """
    rng = np.random.default_rng(cfg.seed)
    s = cfg.drift_strength
    n_c = cfg.n_clusters

    # Users: initial population plus arrivals (d).
    user_join = np.concatenate(
        [
            np.zeros(cfg.n_users, dtype=np.int64),
            _arrival_days(rng, cfg.n_new_users_per_day * s, cfg.n_days),
        ]
    )
    n_users_total = user_join.size
    log_pref = np.log(
        rng.dirichlet(np.full(n_c, cfg.preference_concentration), size=n_users_total)
        + 1e-12  # Dirichlet with small concentration can underflow to exact zeros.
    )

    # Items: initial catalogue plus releases (a). Every cluster gets at least one
    # initial item, otherwise a user could pick a cluster with nothing in it.
    item_release = np.concatenate(
        [
            np.zeros(cfg.n_items_initial, dtype=np.int64),
            _arrival_days(rng, cfg.n_new_items_per_day * s, cfg.n_days),
        ]
    )
    n_items_total = item_release.size
    initial_cluster = rng.permutation(np.arange(cfg.n_items_initial) % n_c)
    new_cluster = rng.integers(0, n_c, size=n_items_total - cfg.n_items_initial)
    item_cluster = np.concatenate([initial_cluster, new_cluster])
    # Base popularity: initial items get a random Zipf rank within their cluster; new
    # releases draw a rank from the same range, so they are not systematically hits.
    cluster_size = np.bincount(initial_cluster, minlength=n_c)
    initial_rank = np.empty(cfg.n_items_initial, dtype=np.int64)
    for c in range(n_c):
        members = np.flatnonzero(initial_cluster == c)
        initial_rank[members] = rng.permutation(members.size) + 1
    new_rank = rng.integers(1, cluster_size[new_cluster] + 1)
    base_weight = _zipf_weights(
        np.concatenate([initial_rank, new_rank]), cfg.item_popularity_exponent
    )

    # Group items by cluster once, so per-day sampling is one searchsorted call.
    order = np.argsort(item_cluster, kind="stable")
    sorted_cluster = item_cluster[order]
    sorted_release = item_release[order]
    sorted_base = base_weight[order]
    cluster_start = np.searchsorted(sorted_cluster, np.arange(n_c))

    boost = cfg.freshness_boost * s
    trend_sd = cfg.trend_volatility * s
    taste_sd = cfg.taste_volatility * s
    log_trend = np.zeros(n_c)

    users_out: list[np.ndarray] = []
    items_out: list[np.ndarray] = []
    ts_out: list[np.ndarray] = []
    for day in range(cfg.n_days):
        if day > 0:
            if trend_sd > 0:
                log_trend += rng.normal(0.0, trend_sd, size=n_c)  # (b)
            if taste_sd > 0:
                log_pref += rng.normal(0.0, taste_sd, size=log_pref.shape)  # (c)

        active = np.flatnonzero(user_join <= day)
        n_events = rng.poisson(cfg.events_per_user_per_day, size=active.size)
        ev_user = np.repeat(active, n_events)
        if ev_user.size == 0:
            continue

        # Cluster choice per event: inverse-CDF over each user's current mixture.
        logits = log_pref[active] + log_trend
        probs = np.exp(logits - logits.max(axis=1, keepdims=True))
        cdf = np.cumsum(probs, axis=1)
        cdf /= cdf[:, -1:]
        row_of_event = np.repeat(np.arange(active.size), n_events)
        u = rng.random(ev_user.size)
        ev_cluster = (cdf[row_of_event] < u[:, None]).sum(axis=1)
        ev_cluster = np.minimum(ev_cluster, n_c - 1)  # guards u ~ 1.0 rounding

        # Item choice within cluster. Cluster c occupies [c, c + 1) of a global CDF,
        # so one searchsorted samples every event's item at once.
        age = day - sorted_release
        released = age >= 0
        weight = np.where(
            released,
            sorted_base * (1.0 + boost * np.exp(-np.maximum(age, 0) / cfg.freshness_decay_days)),
            0.0,
        )
        cum = np.cumsum(weight)
        before = np.concatenate([[0.0], cum])[cluster_start]
        totals = np.add.reduceat(weight, cluster_start)
        within = (cum - before[sorted_cluster]) / totals[sorted_cluster]
        global_cdf = sorted_cluster + within
        target = ev_cluster + rng.random(ev_user.size)
        pos = np.searchsorted(global_cdf, target, side="right")
        # Floating-point rounding can push a target just past its cluster's end. Ids
        # are in arrival order, so released items are a prefix of each cluster: clamp
        # to the last released one, never to an item that does not exist yet.
        last_released = cluster_start + np.add.reduceat(released, cluster_start) - 1
        pos = np.minimum(pos, last_released[ev_cluster])
        ev_item = order[pos]

        users_out.append(ev_user)
        items_out.append(ev_item)
        ts_out.append(day * DAY + rng.integers(0, DAY, size=ev_user.size))

    df = pl.DataFrame(
        {
            USER: np.concatenate(users_out) if users_out else np.empty(0, dtype=np.int64),
            ITEM: np.concatenate(items_out) if items_out else np.empty(0, dtype=np.int64),
            TS: np.concatenate(ts_out) if ts_out else np.empty(0, dtype=np.int64),
        }
    )
    return to_canonical(df)
