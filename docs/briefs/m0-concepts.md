# Brief: m0-concepts

**Goal.** `docs/CONCEPTS.md`: the background Miguel needs to explain this project in an interview.

**Files you own.** `docs/CONCEPTS.md` only.

**Depends on.** `docs/EVALUATION.md`, `docs/ARCHITECTURE.md`, `docs/PLAN.md`, `docs/DATASET.md`. Your text must agree with them; where they make a choice, explain the reasoning behind that choice rather than proposing a different one.

**Reader.** An MSc AI student who is solid on classical ML (scikit-learn, cross-validation, leakage, feature selection), has trained a small CNN in PyTorch, and has **not** built a recommender, used a vector index, or run anything in production (no Docker, Kubernetes or cloud yet).

**Content, in this order.**
1. Recommenders from implicit feedback: what the data is, why there are no negatives, retrieval versus ranking as two stages.
2. Baselines worth beating: popularity, item-kNN.
3. Two-tower retrieval: the two towers, dot-product scoring, in-batch negatives and why they need a popularity (logQ) correction, cold start and content features, approximate nearest-neighbour search.
4. Evaluating over time: why random splits leak, chronological replay, recall@k and NDCG@k with their formulas (binary relevance), what coverage tells you.
5. Drift: what changes in a recommender (catalogue, popularity, users), covariate versus concept drift in this setting, PSI and Jensen-Shannon divergence with formulas and how to read them, input-based versus performance-based triggers.
6. Retraining: schedule versus trigger, full retrain versus warm start, the cost side.
7. Shipping a new model safely: shadow evaluation, canary, the promotion gate and why the candidate must not train on the window it is judged on.
8. The production pieces and what each is for: model registry, history store, metrics and dashboards, containers, Kubernetes, infrastructure as code. One short paragraph each.
9. Twelve interview questions with concise model answers, tied to this project.

**Style.** Plain, precise prose; define each term when it first appears; short worked examples with small numbers where they help; formulas in fenced code or simple inline notation that renders on GitHub. About 2,000 to 2,800 words. No marketing tone, no claims about results the project has not produced yet. Every formula and factual statement must be correct - if unsure, check it.

**Acceptance.** The document reads top to bottom without needing another source, and nothing in it contradicts `docs/EVALUATION.md`.

**Out of scope.** Code, results, tool installation instructions.
