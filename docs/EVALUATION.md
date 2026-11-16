# Evaluation protocol

This is the document the rest of the project is built around. The claim drift-loop
makes ("retraining on drift keeps quality close to always-retraining at a fraction
of the cost") is only as good as this protocol.

## The rule

> At simulated time **t**, nothing with a timestamp **>= t** exists.

No model, statistic or decision may read an event from the future. The rule is
enforced in `replay/clock.py` (`history_before`, `window_slice`); nothing else in
the code base filters by time. `Recommender.fit` additionally refuses rows at or
after its `until` argument.

## Why replay instead of a train/test split

A random split scatters each user's future into the training set, so the model is
graded on things it has effectively already seen. Scores come out too high, and,
more importantly for this project, a random split has no notion of a model getting
old. Replay keeps time intact: the model is always asked about what happens next.

## The replay

Time is cut into a warm-up period followed by consecutive windows of equal length:

```
|<------ warm-up ------>|<- w0 ->|<- w1 ->|<- w2 ->| ...
t_min                   t0       t1       t2
```

For each retrain policy, independently, and for each window `w = [t, t + step)`:

1. **Observe.** The visible data is `history_before(t)`.
2. **Measure drift.** Compare the *reference* (the data the current model was
   trained on) with the *current* period (the last completed window,
   `[t - step, t)`). Before the first window there is no completed window, so there
   is no drift report.
3. **Decide.** The policy sees the drift report and how many windows have passed
   since the last retrain, and answers retrain or not.
4. **Retrain (maybe).** Fit a new model on `history_before(t, lookback=...)`. The
   fit is timed.
5. **Evaluate.** Score the model now in service on window `w` (next section).
6. **Record** one row: metrics, drift signals, whether a retrain happened, fit
   seconds, model age in windows.

Every policy starts from the same model, trained on the warm-up period.

## Scoring a window

- **Who:** every user with at least one qualifying event in the window.
- **Ground truth:** the items the user interacted with in the window that they had
  **not** interacted with before `t` (new-to-user items). Users whose ground truth
  is empty are skipped.
- **Recommendations:** `model.recommend(users, k, seen=...)`, where `seen` is built
  from `history_before(t)`. The harness supplies it, so a stale model still benefits
  from fresh history, exactly as it would behind a real serving layer.
- **Metrics**, each computed per user and then averaged over users:
  - **recall@k** = hits in the top k / number of relevant items. A user with more
    than k relevant items cannot reach 1 (20 relevant items at k = 10 caps at 0.5).
  - **NDCG@k** with binary relevance: DCG = sum of 1 / log2(rank + 1) over hits,
    divided by the ideal DCG over min(k, number of relevant items) positions.
  - **hit rate@k** = 1 if the top k contains at least one relevant item.
  - **coverage** (one number per window) = distinct items recommended to anyone /
    distinct items in the history visible at `t`.

Why new-to-user only: in music, most plays are repeats, and a model that replays a
user's own history scores well while recommending nothing. Discovery is the harder
and more useful target. A repeats-included variant is run as a sensitivity check
(M1).

Users who appear for the first time inside the window are kept. A stale model has
never seen them and must fall back to something non-personalised; that penalty is
part of what staleness costs.

## Policies compared

| Policy | Retrains | Role |
|---|---|---|
| `never` | Only the warm-up fit | Lower bound: how fast does a frozen model decay? |
| `always` | Before every window (the first is a no-op: the warm-up model is already current) | Reference point: the most frequent retraining, maximum cost |
| `fixed(n)` | Every n windows | The schedule most teams actually run |
| `on_drift` | When the detector fires | Input-based trigger |
| `on_decay` (M1) | When measured quality drops | Label-based trigger |

## The headline

Two numbers per policy over the scored windows (all windows after the tuning
windows, see "Drift signals"): **mean quality** and **cost**
(number of retrains and total fit seconds). The result to report is the trade-off:
the share of `always`'s quality gain over `never` that a policy keeps, against the
share of `always`'s retrains it pays for. `always` is a reference point, not a
guaranteed optimum: a fresher model is not better in every window, so the kept
share can exceed 1 or fall below 0, and it is noisy when `always` and `never` are
close. One chart shows quality per window for every
policy, with retrain events marked.

## Drift signals

Computed from interactions only, reference versus current period:

- **Popularity PSI.** Items are bucketed by their popularity rank in the reference
  period, each bucket holding a roughly equal share of reference events. The
  population stability index compares the share of events per bucket, **among
  items the reference knew**: has demand moved between head and tail?
- **Popularity Jensen-Shannon divergence** over the same buckets.
- **New-item share.** Fraction of current events on items the reference never saw.
  Novelty is measured here and only here. Putting unseen items into the PSI as an
  extra bucket was tried and dropped: that bucket's reference share is zero, so it
  swamps the statistic and makes it depend on the smoothing constant.
- **New-user share.** Fraction of current events from users the reference never saw.

A detector applies thresholds and reports which signals fired. Thresholds are
configuration, tuned on the first `tuning_windows` windows of the replay and then
frozen. Those windows are **excluded from the headline numbers**: every policy's
mean quality and cost are reported over the remaining windows only. Tuning on
windows that are then scored would be its own kind of leakage.

## The promotion gate (M4)

When the loop wants to replace the model in service, the candidate is judged before
it is promoted:

1. At time `t`, train a **candidate** on `history_before(t - step)`.
2. Score candidate and incumbent on the last completed window `[t - step, t)`,
   exactly as any window is scored: ground truth and seen items are built from
   `history_before(t - step)`.
3. Promote only if the candidate wins by a margin.
4. If promoted, refit on `history_before(t)` and put that in service.

Step 1 is the point: a candidate trained on the window it is judged on would always
look better than the incumbent. Step 4 is a compromise (the model that goes live is
not exactly the one that was judged), and it is recorded as such.

## Known limits

- **No feedback loop.** Replay cannot show how users would have reacted to *our*
  recommendations; the logged behaviour was shaped by the platform's own
  recommender. The dataset's organic flag lets us score on organic events only,
  which reduces this bias but does not remove it.
- **One dataset, one domain.** Decay rates are a property of the data. The method
  transfers; the numbers do not.
- **Cost is training time on one machine**, not a full bill.
- **Absolute recall drifts down for every policy.** With new-to-user ground truth,
  the popular items a user has not yet seen run out as their history grows, so even
  `always` declines over the replay (seen on synthetic data with drift switched
  off). Policies are compared with each other at the same window, never a curve
  with its own starting point.
