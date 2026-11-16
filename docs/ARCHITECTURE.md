# Architecture

## The idea in one picture

```
                 historical interactions, replayed in time order
                                   |
                                   v
   +---------+   events   +----------------+   requests   +-----------------+
   | replay  |----------->|  event log /   |<-------------|  recommender    |
   | driver  |            |  history store |------------->|  API (serving)  |
   +---------+            +----------------+   seen items +-----------------+
                              |        |                          ^
                reference vs  |        | training data            | load model
                recent window v        v                          |
                       +----------+  +-----------+   candidate  +-----------+
                       |  drift   |->|  retrain  |------------->|  model    |
                       | detector |  |   job     |              | registry  |
                       +----------+  +-----------+   promotion  +-----------+
                              \           |            gate          ^
                               \          v                          |
                                +--> metrics & dashboards -----------+
```

M0 implements the left-to-right logic of this picture as an **offline simulation**
in one process. M3 to M6 turn the boxes into real services without changing the
logic, which is why the simulation comes first: it is the specification the live
system has to agree with.

## Package layout

```
src/driftloop/
  data/
    schema.py       canonical interaction table: user_id, item_id, ts, weight
    synthetic.py    generator of interactions with controllable drift
    yambda.py       Yambda raw events -> canonical table; download helper
  replay/
    clock.py        windows and the two time-slicing functions (the leakage rule lives here)
  models/
    base.py         Recommender protocol, SeenItems
    popularity.py   popularity baseline
  eval/
    metrics.py      recall@k, ndcg@k, hit rate, coverage
    protocol.py     scoring one window (who, ground truth, seen filtering)
  drift/
    report.py       DriftReport
    stats.py        PSI, Jensen-Shannon divergence, new-item and new-user share
    detector.py     thresholds -> DriftReport
  loop/
    policies.py     never / always / fixed interval / on drift
    simulate.py     the replay loop: observe, measure, decide, retrain, evaluate, record
  report/
    plot.py         quality-over-time chart
  config.py         typed configuration loaded from YAML
  cli.py            `driftloop` command
configs/            one YAML file per experiment
tests/              mirrors src/
docs/               plan, protocol, decisions, dataset, costs, concepts, briefs
```

Planned, created by the milestone that implements them (no empty placeholders):

| Path | Milestone | Contents |
|---|---|---|
| `models/item_knn.py`, `models/decayed_popularity.py` | M1 | Classical baselines |
| `models/two_tower/` | M2 | Towers, training loop, FAISS export |
| `serving/` | M3 | FastAPI app, model loader, metrics |
| `loop/jobs/` , `registry/` | M4 | Drift job, retrain job, promotion gate, MLflow registry client |
| `deploy/compose/`, `deploy/k8s/`, `deploy/terraform/` | M4, M5, M6 | Runtime environments |
| `loadtest/` | M3 | Load-test scenarios |

## Contracts

Three small contracts hold the system together. Changing one is an architectural
decision and gets an entry in `docs/DECISIONS.md`.

1. **Interaction table** (`data/schema.py`). Four columns, fixed dtypes, sorted by
   time, positive interactions only. Dataset-specific logic stops at the adapter.
2. **Time slicing** (`replay/clock.py`). `history_before(df, t)` and
   `window_slice(df, w)` are the only functions that filter by time.
3. **Recommender** (`models/base.py`). `fit(interactions, until=...)` and
   `recommend(user_ids, k, seen=...)`. Seen-item filtering is supplied by the caller.

## Data flow of one simulated window

See `docs/EVALUATION.md` for the reasoning; in code terms:

```
visible   = history_before(interactions, w.start)
recent    = window_slice(interactions, previous_window)
report    = detector.compare(reference=training_data_of_current_model, current=recent)
if policy.should_retrain(context): model = fresh_model().fit(history_before(..., lookback), until=w.start)
result    = evaluate_window(model, history=visible, future=window_slice(interactions, w), k=k)
```

## Technology choices

| Concern | Choice | Reason |
|---|---|---|
| Tables | Polars | Lazy, multi-threaded, handles 46 M rows on a laptop |
| Numerics | NumPy | Metrics and baselines need nothing heavier |
| Config | Pydantic + YAML | Typed, validated, one file per experiment |
| CLI | Typer | Small and typed |
| Packaging | uv, `src/` layout | Reproducible lock file, fast CI |
| Quality | ruff, mypy, pytest | Enforced in CI |
| Model (M2) | PyTorch, FAISS | Industry-standard retrieval stack |
| Tracking and registry (M2, M4) | MLflow | Experiments and model versions in one tool |
| Serving (M3) | FastAPI, Prometheus client | Async HTTP with first-class metrics |
| Orchestration (M5, M6) | Kubernetes (kind, then EKS), Terraform | The stack employers name |
