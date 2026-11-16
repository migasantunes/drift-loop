# Concepts

The background needed to explain drift-loop from first principles. Where the project
makes a choice, this document explains why; the choices themselves are specified in
`EVALUATION.md`, `ARCHITECTURE.md`, `PLAN.md` and `DATASET.md`.

## 1. Recommenders from implicit feedback

**Implicit feedback** is behaviour recorded as a side effect of using a product: plays,
clicks, purchases. **Explicit feedback** is a rating the user chose to give. drift-loop
uses implicit feedback: in Yambda a listen counts as a positive interaction when at least
half the track was played. Every row of the canonical table (`user_id, item_id, ts,
weight`) is one positive event.

There are no negatives. A track a user never played is not a track they dislike; they
most likely never came across it. So the task is not "predict a label for each
(user, item) pair" but "rank the catalogue so the items the user will actually engage
with come first". Training has to invent negatives, usually by treating other items as
not chosen in that context (section 3).

Real systems use two stages:

- **Retrieval** (candidate generation) cuts millions of items down to a few hundred
  per request. It must be cheap per item, so the model is simple: often a dot product
  between a user vector and precomputed item vectors.
- **Ranking** re-scores those candidates with a richer model and more features,
  which is affordable because there are few of them.

drift-loop builds the retrieval stage. A ranking stage is a stretch goal in M2.

## 2. Baselines worth beating

**Popularity.** Recommend the most-played items to everyone, minus what the user has
already seen. It sounds trivial and is often hard to beat, because popular items are
popular for a reason. It is also the natural fallback for a user the model has never
seen. A time-decayed variant (M1) weights recent plays more, so it follows trends.

**Item-kNN.** Two items are similar if the same users interact with both. Build the
user-item matrix, compute a similarity between item columns (co-occurrence counts,
often normalised as cosine similarity), and score a candidate for a user by summing its
similarity to the items in the user's history. It is personalised, needs no gradient
training, and at this scale must use sparse matrices.

## 3. Two-tower retrieval

A **two-tower model** has two networks. The **user tower** maps a user to a vector
`u`; in M2 it pools the embeddings of the user's recent items. The **item tower** maps
an item to a vector `v` of the same size; in M2 it combines a learned ID embedding with
the dataset's audio embedding. The score is the dot product `s(u, i) = u · v_i`.

Because the towers never see each other's inputs, item vectors can be computed once
and stored; serving a user is one user-tower pass plus a search for the largest dot
products.

**In-batch negatives.** Training treats retrieval as classification over items: given
the user, which item did they choose? A full softmax over 900 thousand items per
example is too expensive. Instead, take a batch of B (user, positive item) pairs and
use the other B - 1 positives in the batch as that user's negatives. Each user gets a
softmax over B items:

```
loss_j = -log( exp(s(u_j, i_j)) / sum_{b=1..B} exp(s(u_j, i_b)) )
```

**Why the logQ correction.** Items appear in a batch in proportion to how often they
occur in the training data. A popular track is therefore used as a negative far more
often than a rare one, and the model learns to push popular items down more than a
full softmax would. The fix, described by Yi et al. (RecSys 2019, "Sampling-Bias-
Corrected Neural Modeling for Large Corpus Item Recommendations"), is to subtract the
log of each item's sampling probability `q_i` from its logit during training:

```
s_corrected(u, i) = u · v_i - log(q_i)
```

applied to every logit in the batch, the positive included. `q_i` is estimated as the
item's share of training events. The correction is used only in the loss; at serving
time the plain dot product is used.

**Cold start.** A brand-new track has no interactions, so its ID embedding is
untrained. Because the item tower also reads the audio embedding, a new track still
gets a vector that places it near tracks that sound similar. Inputs like this, which
describe the item rather than its history, are called **content features**. A new user
is handled by the history-pooling user tower once they have a few plays, and by the
popularity fallback before that.

**Approximate nearest-neighbour (ANN) search.** Finding the top k dot products exactly
means scoring every item. An ANN index (FAISS here; HNSW graphs and IVF clusters are common
types) lets a query visit only a small part of the catalogue, trading a little accuracy
for much lower latency. Index accuracy is measured as recall: the share of the exact
top k that the index returns. M2 compares exact search with an approximate index on both
recall and latency.

## 4. Evaluating over time

A **random split** puts each user's later behaviour into the training set and grades
the model on earlier behaviour it has, in effect, already been told about. Scores come
out too high. Worse for this project, a random split has no notion of time, so it cannot
show a model getting old.

**Chronological replay** keeps time intact. After a warm-up period, time is cut into
equal windows. Before each window the system sees only events before the window start
`t` ("at simulated time t nothing with ts >= t exists"), may retrain, and is then scored
on the window. Every policy starts from the same warm-up model.

**What counts as a hit.** For each user active in the window, the relevant set `R_u`
is the items they interacted with in the window that they had **not** interacted with
before `t`. Music is dominated by repeat listening; a model that re-plays a user's
history would score well while recommending nothing new. Users with an empty `R_u` are
skipped. Users who are new in the window are kept: a stale model must fall back to
popularity for them, and that is part of the cost of staleness.

**Seen-item filtering is done by the caller.** The harness builds each user's seen set
from all history before `t` and passes it to `recommend`. A model trained two weeks ago
cannot know what the user played yesterday; a serving layer can, so the harness does
the same.

**Metrics** (binary relevance; `rec_u` is the top-k list, `rank` starts at 1):

```
hits_u      = |rec_u ∩ R_u|
recall@k    = hits_u / |R_u|
hit_rate@k  = 1 if hits_u > 0 else 0
DCG@k       = sum over ranks r = 1..k of  rel_r / log2(r + 1)      (rel_r = 1 if item at r is in R_u)
IDCG@k      = sum over ranks r = 1..min(k, |R_u|) of  1 / log2(r + 1)
NDCG@k      = DCG@k / IDCG@k
```

Each is computed per user and averaged over users. Worked example with k = 5,
`R_u = {A, B, C}` and `rec_u = [X, A, Y, B, Z]`:

```
recall@5 = 2 / 3                         = 0.667
DCG@5    = 1/log2(3) + 1/log2(5)         = 0.631 + 0.431 = 1.062
IDCG@5   = 1/log2(2) + 1/log2(3) + 1/log2(4) = 1 + 0.631 + 0.5 = 2.131
NDCG@5   = 1.062 / 2.131                 = 0.498
```

Recall ignores order; NDCG rewards putting hits near the top. A user with 20 relevant
items can reach at most recall 0.5 at k = 10; that does not bias comparisons between
models on the same users, but caps absolute values.

**Coverage** is the number of distinct items recommended to anyone divided by the
number of distinct items in the history visible at `t`. Popularity gives nearly everyone the same list and has tiny coverage; two models
with similar recall can differ widely here, which is why it is reported alongside.

## 5. Drift

A deployed recommender goes stale because the world it learned changes:

- **Catalogue.** New tracks arrive; an ID-based model has no vector for them.
- **Popularity.** This month's most-played tracks differ from those at training time.
- **Users.** New users join, and existing users' tastes move.

**Covariate drift** is a change in the input distribution P(X), here which users are
active and which items get played. **Concept drift** is a change in P(Y | X), here what
a user with a given history plays next. In a recommender the two blur, since items are
both inputs and outputs, but the distinction tells you what each detector can see. Statistics on interactions alone
mostly see the first kind; only measured quality sees the second.

drift-loop computes four signals from interactions, comparing the **reference** period
(the data the current model was trained on) with the **current** period (the last
completed window): popularity PSI, popularity Jensen-Shannon divergence, new-item
share and new-user share.

**The buckets.** Items the reference knew are ranked by event count in the reference
period and split into groups holding roughly equal shares of reference events, head
items first. Current events on those items are counted per bucket and renormalised to
sum to 1; current events on unknown items are left out. Each period becomes a
distribution over buckets, `p` for reference and `q` for current, and PSI and JSD
answer one question: has demand moved between head and tail among items the model
knows? Novelty is measured separately, by new-item share.

**Population stability index (PSI)**, with shares clipped below at a small `eps` so an
empty bucket does not divide by zero:

```
PSI = sum over buckets b of  (q_b - p_b) * ln(q_b / p_b)
```

PSI is zero when the distributions match and grows with any shift; before clipping
it equals KL(q || p) + KL(p || q), a symmetric form of KL divergence. A widely used rule
of thumb from credit scoring reads below 0.1 as stable, 0.1 to 0.25 as a moderate shift
and above 0.25 as a large one. That is a convention, not a statistical test, which is
why this project tunes its thresholds.

**Jensen-Shannon divergence (JSD)**, with `m = (p + q) / 2`:

```
KL(p || m) = sum over b of  p_b * log2(p_b / m_b)      (terms with p_b = 0 are 0)
JSD        = 0.5 * KL(p || m) + 0.5 * KL(q || m)
```

With base-2 logarithms it lies between 0 and 1. It needs no clipping, since `m_b > 0`
wherever either share is positive. Beware that some libraries (for example SciPy's
`jensenshannon`) return its square root, the Jensen-Shannon distance.

Worked example, three buckets, reference `p = [1/3, 1/3, 1/3]`:

```
current q            PSI     JSD
[0.40, 0.35, 0.25]   0.037   0.007     slight move towards the head
[0.50, 0.30, 0.20]   0.139   0.025     clear move towards the head
[0.20, 0.30, 0.50]   0.139   0.025     the same move towards the tail
```

Neither tells you the direction of the shift; the bucket shares do.

**Why not an extra bucket for new items?** It was tried and dropped. Unknown items have
a reference share of zero by definition, which is clipped to `eps`, so that bucket
contributes about `q_new * ln(q_new / eps)`. With `eps = 1e-4`, 10% of plays on new
items adds about 0.69, swamping the popularity terms and depending on an arbitrary
constant. With the
bucket dropped, a period where 10% of plays go to new items and the rest keep the
reference mix gives PSI = 0 and new-item share = 0.10: each signal reports one thing.

**Input-based versus performance-based triggers.** An input-based trigger (`on_drift`)
fires on these statistics. It needs no labels and can react before quality drops, but
a shift in inputs does not always hurt the model, and a change in taste may not show
up in inputs at all. A performance-based trigger (`on_decay`, M1) fires when measured
quality on the last completed window falls a set fraction below its level after the
last retrain. It measures what you care about, but needs labels. In music the labels (the next
plays) arrive almost immediately, so it is a strong competitor here; where outcomes
arrive weeks later it would not be.

Thresholds are configuration, tuned on the first few windows of the replay (the
tuning windows) and then frozen. Those windows are left out of the headline numbers;
scoring a policy on the windows its thresholds were chosen on would be leakage.

## 6. Retraining

**Schedule versus trigger.** A schedule (`fixed(n)`: every n windows) is simple,
predictable and what most teams run. A trigger retrains only when something happened.
The two bounds frame both: `never` shows how fast a frozen model decays, `always` shows
the best quality retraining can buy and its maximum cost. The headline result, over the windows
after the tuning windows, is how much of the gap between them a policy closes, against
what it pays.

**Full retrain versus warm start.** A full retrain starts from random weights on the
training lookback. A warm start initialises from the previous checkpoint and trains
briefly on the newer data. Warm starts are cheaper, but new item IDs need fresh
embedding rows, and training only on recent data can make the model forget older
patterns. M2 measures the cost saving rather than assuming it.

**The cost side.** Here cost is the number of retrains and total fit seconds on one
machine, not a cloud bill. In production a retrain also costs pipeline time,
validation and a deployment that can go wrong.

## 7. Shipping a new model safely

**Shadow evaluation** runs a candidate model on the same inputs as the model in service
(the **incumbent**) and records its outputs without showing them to anyone, so the two
can be compared on identical traffic. A **canary** release serves the candidate to a
small share of real users and widens the share only if its metrics hold. drift-loop uses
shadow evaluation; a canary needs real users reacting, which replay cannot provide.

**The promotion gate (M4).** At time `t`:

1. Train a candidate on `history_before(t - step)`.
2. Score candidate and incumbent on the last completed window `[t - step, t)`, like
   any window: ground truth and seen items come from `history_before(t - step)`.
3. Promote only if the candidate wins by a margin.
4. If promoted, refit on `history_before(t)` and serve that.

Step 1 is the point. A candidate trained on the window it is judged on has seen the
answers and would beat the incumbent almost every time: test-set leakage. Holding the
window out means neither model has seen what it is scored on. Step 4 is a compromise,
because the model that goes live is not exactly the one that was judged, and the
project records it as such. The margin in step 3 keeps random noise from swapping models back and forth.

## 8. The production pieces

**Model registry.** Stores model versions with their metrics and a stage such as
candidate or production. Serving loads the production version; rollback means pointing
back to an earlier one. drift-loop uses MLflow.

**History store.** Holds each user's interactions so the serving layer can build the
user's input and filter seen items quickly at request time. M4 chooses between options
such as Redis or SQLite and records why.

**Metrics and dashboards.** Prometheus collects numbers from running services (latency,
model version, model age, drift signals) and Grafana plots them, so a drift event, a
retrain and a recovery are visible in a live system.

**Containers.** A Docker image packages the code with its exact dependencies, so it runs
the same on a laptop, in CI and in the cloud.

**Kubernetes.** Runs containers across machines and keeps them in the declared state.
The API is a Deployment (restarted if its health probes fail, scaled by an autoscaler),
the drift check a CronJob (runs on a schedule) and a retrain a Job (runs to completion).
M5 uses a local cluster (kind); M6 uses Amazon EKS.

**Infrastructure as code.** Terraform describes the cloud resources (network, cluster,
image registry, storage) in files, so the setup can be reviewed, recreated and removed
with `terraform destroy`, which matters when every running hour is billed.

## 9. Interview questions

**1. Why not a random train/test split?** It leaks each user's future into training,
inflating scores, and cannot show a model ageing, which is what drift-loop measures.

**2. What does "at time t nothing with ts >= t exists" protect against, and where is it
enforced?** Anything reading the future. Only `history_before` and `window_slice` in
`replay/clock.py` filter by time, and `fit` rejects rows at or after `until`.

**3. Why only new-to-user items as ground truth?** Most plays are repeats; a model that
re-plays history would score well and help no one discover anything.

**4. Why does the caller filter seen items?** Only the caller has fresh history; a model
trained two weeks ago does not know what the user played yesterday.

**5. Recall@10 versus NDCG@10?** Recall: share of relevant items in the top 10, position
ignored. NDCG: also rewards placing them early; a perfect ordering scores 1.

**6. Why do in-batch negatives need a logQ correction?** Popular items are sampled as
negatives more often and get over-penalised; subtracting `log(q_i)` from each training
logit removes that bias.

**7. What does the item tower do for a track with no plays?** Its audio embedding gives
it a vector near similar-sounding tracks, so it can still be retrieved.

**8. How do you read PSI, and why no bucket for new items?** Zero means demand is spread
across head and tail as in the reference; 0.1 / 0.25 are conventions, so thresholds
are tuned. A new-item bucket has zero reference share, is clipped to `eps` and swamps
the statistic, so novelty is left to new-item share.

**9. Covariate versus concept drift here?** Covariate: which users and items appear.
Concept: what a user with a given history plays next. Interaction statistics mostly
see the first; measured quality sees both.

**10. Input-based or performance-based trigger?** Input-based needs no labels but may
fire on harmless shifts; performance-based measures the target but needs labels, which
arrive quickly in music.

**11. Why must the gate's candidate not train on the window it is judged on?** It would
have seen the answers. Train on `history_before(t - step)`, score on `[t - step, t)`.

**12. What does `always` tell you?** The best quality retraining can buy, at maximum
cost. With `never` it brackets every other policy.
