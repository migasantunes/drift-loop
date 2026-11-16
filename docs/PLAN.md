# Work plan

drift-loop is built in eight milestones. Each one ends in something that runs and
can be shown, so the project is worth having on a CV from M3 onwards even if the
later milestones slip.

| | Milestone | Ends with | Status |
|---|---|---|---|
| M0 | Walking skeleton | Replay harness, baseline, retrain policies and a chart, on synthetic data | Done (scaffold) |
| M1 | Real data | The first frozen-versus-retrained chart on Yambda-50M with classical baselines | Next |
| M2 | Two-tower model | A PyTorch retrieval model with an ANN index that beats the baselines, tracked in MLflow | |
| M3 | Serving | A containerised API with measured p50/p99 latency under load | |
| M4 | The live loop | Registry, drift job, retrain job and promotion gate running against replayed traffic, with dashboards | |
| M5 | Kubernetes | The whole stack on a local cluster, built and shipped by CI | |
| M6 | AWS | The stack on EKS via Terraform, load-tested, recorded, torn down | |
| M7 | Write-up | Results README, final chart, post, CV bullets | |

## How the work is run

- **Lead (Fable)** turns a milestone into briefs under `docs/briefs/`, one per
  independent piece of work, each with files to touch, contracts to respect and
  acceptance criteria.
- **Implementers (Opus sub-agents, `.claude/agents/implementer.md`)** each take one
  brief. Briefs within a milestone touch disjoint files so they can run in parallel.
- **Reviewer (`.claude/agents/reviewer.md`)** checks the result against the brief
  and against `docs/EVALUATION.md` before anything is merged. It has not seen the
  implementation being written.
- **Miguel** owns the decisions. Every milestone has an *interview checkpoint*: a
  short list of questions he should be able to answer without notes before the
  milestone is merged. If he can't, the milestone is not done.

Each milestone is one branch and one pull request into `main`.

---

## M0 - Walking skeleton (done in the scaffold)

**Goal.** Prove the evaluation idea end to end with the simplest possible parts,
so every later milestone plugs a better component into a harness that already works.

**Delivered.**
- Canonical interaction schema and time-slicing helpers (`data/schema.py`, `replay/clock.py`).
- Synthetic interaction generator with controllable drift (`data/synthetic.py`).
- Yambda adapter: raw events to canonical schema (`data/yambda.py`), tested on fake frames.
- Ranking metrics and the per-window evaluation protocol (`eval/`).
- Popularity baseline (`models/popularity.py`).
- Drift statistics, detector and retrain policies (`drift/`, `loop/policies.py`).
- Offline simulation of the loop and a CLI (`loop/simulate.py`, `cli.py`).
- Chart of quality over time per policy (`report/plot.py`).
- Tests, lint, type checks, CI, Dockerfile.

**Interview checkpoint.**
1. What does "at simulated time t nothing with ts >= t exists" protect against, and where is it enforced?
2. Why is a random train/test split wrong for a recommender?
3. Why does the caller, not the model, filter out items the user has already seen?
4. What does the "always retrain" policy tell you that the others can't?

## M1 - Real data

**Goal.** Replace synthetic data with Yambda-50M and get the first honest result.

**Tasks.**
1. Download Yambda-50M (`flat/50m`) and convert it with the existing adapter. Record row counts at every filtering step in `docs/DATASET.md`.
2. Exploratory notebook: events per day, the real time span, catalogue growth, share of events on items younger than a week, repeat-listen rate, organic share. Resolve the two open questions in `docs/DATASET.md` (meaning of `is_organic`, licence wording).
3. Choose the replay settings from the data (warm-up, window length, training lookback) and write them to `configs/yambda-50m.yaml` with the reasoning in `docs/DECISIONS.md`.
4. Add two stronger baselines behind the `Recommender` protocol: time-decayed popularity and item-kNN (co-occurrence, sparse).
5. Add a performance-based policy, `OnDecay`: retrain when the metric on the last completed window falls a set fraction below the level measured right after the last retrain. Labels arrive immediately in this domain, so this is a fair and strong competitor to input-drift detection.
6. Run all policies and all baselines; commit the chart and the summary table to `docs/results/`.
7. Sensitivity runs: organic-only evaluation, and repeats-included ground truth.
8. Add a bursty-drift scenario to the synthetic generator (a sudden catalogue or trend shock) and check that `on_drift` beats a schedule of equal cost there. On the smooth M0 demo it did not (decision D12).
9. Report each policy relative to `always` at the same window as well as in absolute terms, since absolute recall falls for every policy as user histories grow.

**Acceptance.** `driftloop simulate --config configs/yambda-50m.yaml` reproduces the committed table from a clean clone plus the downloaded data. Memory stays under 12 GB.

**Risks.** 46 M rows need care: stay in Polars lazy frames, never convert the full table to Python objects. `SeenItems` may need a compact (CSR) representation at this size.

**Interview checkpoint.** How fast does a frozen model decay on this data, and what drives it: new items, shifting popularity or changing users? How did you pick the window length?

## M2 - Two-tower retrieval model

**Goal.** A learned retrieval model that beats the M1 baselines under the same protocol.

**Tasks.**
1. Add the `train` extra (PyTorch, FAISS, MLflow).
2. User tower: recent-history pooling over item embeddings. Item tower: ID embedding plus the dataset's audio embedding, so brand-new tracks get a usable vector (cold start).
3. Training: in-batch sampled softmax with the logQ popularity correction; time-ordered batches; validation on the last slice of the training period only.
4. Retrieval: export item vectors to a FAISS index; compare exact search with an approximate index (recall versus latency).
5. Wrap it as a `Recommender` so the replay harness runs it unchanged. Support warm-start retraining from the previous checkpoint and measure the cost saving.
6. Track every run in MLflow (params, metrics per window, artefacts).
7. Stretch: a ranking stage (gradient-boosted trees or a small MLP) over the top few hundred candidates.

**Acceptance.** Two-tower recall@K above the best M1 baseline on the first post-training window, with the comparison table committed. If it does not win, the write-up says so and analyses why.

**Interview checkpoint.** Why in-batch negatives need a popularity correction; what the item tower does for a track with zero interactions; warm start versus retraining from scratch.

## M3 - Serving

**Goal.** Recommendations over HTTP with numbers attached.

**Tasks.** FastAPI service (`/recommend`, `/healthz`, `/metrics`); loads the model and index at start-up and reloads on a new version without dropping requests; seen-item filtering from a history store; Prometheus metrics (latency histogram, model version, model age); multi-stage Dockerfile; load test with Locust or k6; latency report at several request rates.

**Acceptance.** `docker run` serves recommendations; committed load-test report with p50/p99 and the request rate at which p99 passes a stated budget.

**Interview checkpoint.** Where the time goes in a request; what happens to in-flight requests during a model swap; how you chose the latency budget.

## M4 - The live loop

**Goal.** The M0 simulation, but as real services.

**Tasks.**
1. Replay driver: feeds historical events to the stack on compressed time (for example one simulated week per minute) and calls the API as users become active.
2. Event log and history store (start with Parquet partitions plus Redis or SQLite; choose and record why).
3. Drift job: runs the detector on a schedule, writes signals to Prometheus.
4. Retrain job: trains a candidate, registers it in the MLflow model registry.
5. Promotion gate, as specified in `docs/EVALUATION.md` (shadow evaluation on the last completed window, candidate trained without that window).
6. Grafana dashboards: online quality, drift signals, model age, latency, retrain events.
7. `docker compose up` brings the whole thing up.

**Acceptance.** A recorded run in which drift fires, a candidate is trained, the gate promotes it, the service swaps models and online quality recovers, all visible on one dashboard. Online numbers agree with the offline simulation within a stated tolerance, and the difference is explained.

**Interview checkpoint.** Why the gate's candidate must not train on the window it is judged on; what you would add before trusting automatic promotion with real users.

## M5 - Kubernetes

**Tasks.** kind cluster; manifests or a Helm chart; API as a Deployment with readiness and liveness probes and a HorizontalPodAutoscaler; drift check as a CronJob; retrain as a Job; model artefacts in object storage (MinIO locally); rolling model updates; GitHub Actions builds and pushes images on tags.

**Acceptance.** `make cluster-up && make deploy` on a clean machine gives a working stack; the M4 scenario runs on it.

## M6 - AWS

**Tasks.** Terraform for VPC, EKS, ECR and S3; public-subnet node group to avoid a NAT gateway unless it is needed; deploy the M5 manifests; run the scenario and the load test; record the demo; `terraform destroy`; log actual spend in `docs/COSTS.md`. Billing alarm and teardown command exist before the first `apply`.

**Acceptance.** Recording and cost log committed; `terraform destroy` leaves nothing billable (checked in the billing console the next day).

## M7 - Write-up

Final README with the headline chart and numbers, architecture diagram, what did not work, CV bullets, and a short post.

---

## Order and scope notes

- M1 to M3 are the core. After M3 there is a real model, a real result and a real service.
- M4 is the part that makes the project unusual. M5 and M6 are where Docker, Kubernetes and AWS become claimable.
- If time is short, cut the M2 ranking stage first, then M6's load test.
