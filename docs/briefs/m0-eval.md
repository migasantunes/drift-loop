# Brief: m0-eval

**Goal.** Ranking metrics, the per-window evaluation protocol, and the popularity baseline.

**Files you own.**
- `src/driftloop/eval/metrics.py`, `src/driftloop/eval/protocol.py`
- `src/driftloop/models/popularity.py`
- `tests/eval/test_metrics.py`, `tests/eval/test_protocol.py`
- `tests/models/test_popularity.py`, `tests/models/test_seen_items.py`

**Depends on.** `src/driftloop/models/base.py` (`Recommender`, `SeenItems`, `PAD_ITEM`), `src/driftloop/data/schema.py`, `docs/EVALUATION.md` section "Scoring a window". Read them; do not edit them - report bugs.

**Specification.**

`metrics.py` - pure NumPy. `recs` is int64 of shape `(n_users, k)`, padded with `PAD_ITEM`; `truth` is a sequence of 1-D arrays of relevant item ids, one per user, each non-empty.
```python
def recall_at_k(recs: np.ndarray, truth: Sequence[np.ndarray]) -> np.ndarray   # hits / len(truth_u)
def ndcg_at_k(recs: np.ndarray, truth: Sequence[np.ndarray]) -> np.ndarray     # binary relevance, IDCG over min(k, len(truth_u))
def hit_rate_at_k(recs: np.ndarray, truth: Sequence[np.ndarray]) -> np.ndarray # 1.0 if any hit
def coverage(recs: np.ndarray, n_items_available: int) -> float               # distinct non-pad items / n_items_available
```
Each per-user function returns a float64 array of length `n_users`. Padding never counts as a hit. Shape mismatches raise `ValueError`.

`protocol.py`
```python
@dataclass(frozen=True, slots=True)
class WindowResult:
    k: int
    n_users: int
    recall: float
    ndcg: float
    hit_rate: float
    coverage: float

def build_ground_truth(history: pl.DataFrame, future: pl.DataFrame) -> tuple[np.ndarray, list[np.ndarray]]: ...
def evaluate_window(model: Recommender, *, history: pl.DataFrame, future: pl.DataFrame, k: int) -> WindowResult: ...
```
- `build_ground_truth`: for each user in `future`, the distinct items they interact with in `future` that they have **not** interacted with in `history`. Users left with nothing are dropped. Returns `(user_ids, truth)` aligned, users sorted ascending. Do the set logic in Polars (anti-join), not in Python loops over rows.
- `evaluate_window`: builds `SeenItems` from `history`, calls `model.recommend(user_ids, k, seen=seen)`, averages the metrics over users. Coverage denominator is the number of distinct items in `history`. With no evaluable users, returns `n_users=0` and NaN metrics (no exception).
- The function must not pass `future` to the model in any form.

`popularity.py`
```python
class PopularityRecommender:
    name = "popularity"
    def __init__(self) -> None: ...
    trained_until: int | None
    def fit(self, interactions: pl.DataFrame, *, until: int) -> Self: ...
    def recommend(self, user_ids: np.ndarray, k: int, *, seen: SeenItems | None = None) -> np.ndarray: ...
```
- Ranks items by summed `weight`; ties broken by smaller item id so results are deterministic.
- `fit` raises `ValueError` if any row has `ts >= until`; `recommend` before `fit` raises `RuntimeError`.
- Per user: the top-k of the global ranking after removing that user's seen items; pad with `PAD_ITEM` if the catalogue runs out. Must be efficient for thousands of users (do not re-sort per user).
- Satisfies `isinstance(model, Recommender)`.

**Tests.**
- Metrics against hand-computed values (include a case with padding, a case with more relevant items than k, perfect and zero rankings).
- Ground truth: repeats excluded, user with only repeats dropped, user absent from history kept, duplicates in `future` collapsed.
- Protocol: a fake model that records its arguments proves `seen` comes from history and that no future row reaches the model; empty future handled.
- Popularity: ordering, tie-break, seen filtering, unknown user gets the global list, padding, `ts >= until` rejected.
- `SeenItems`: built from a frame, unknown user returns an empty array, duplicates collapsed.

**Acceptance.** `uv run ruff check`, `uv run ruff format --check`, `uv run mypy`, `uv run pytest tests/eval tests/models` all pass.

**Out of scope.** The replay loop, drift, CLI, any other model.
