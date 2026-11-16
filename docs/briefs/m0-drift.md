# Brief: m0-drift

**Goal.** Drift statistics, a threshold detector, and the retrain policies.

**Files you own.**
- `src/driftloop/drift/stats.py`, `src/driftloop/drift/detector.py`
- `src/driftloop/loop/policies.py`
- `tests/drift/test_stats.py`, `tests/drift/test_detector.py`, `tests/loop/test_policies.py`

**Depends on.** `src/driftloop/drift/report.py` (`DriftReport`), `src/driftloop/data/schema.py`, `docs/EVALUATION.md` sections "Drift signals" and "Policies compared". Read them; do not edit them - report bugs.

**Specification.**

`stats.py` - every function takes two canonical interaction tables, `reference` and `current`.
```python
def popularity_bucket_shares(reference, current, *, n_buckets: int = 10) -> tuple[np.ndarray, np.ndarray]
def popularity_psi(reference, current, *, n_buckets: int = 10, eps: float = 1e-4) -> float
def popularity_jsd(reference, current, *, n_buckets: int = 10) -> float
def new_item_share(reference, current) -> float
def new_user_share(reference, current) -> float
```
- Buckets: rank items by event count in `reference` (ties by item id) and split the ranked list into `n_buckets` groups holding roughly equal shares of *reference events* (head items in the first bucket). One extra, last bucket collects items absent from `reference`. `popularity_bucket_shares` returns the share of events per bucket for reference and for current, each of length `n_buckets + 1` and summing to 1.
- PSI = sum((cur - ref) * ln(cur / ref)) with shares clipped below at `eps`.
- JSD: Jensen-Shannon divergence with base-2 logarithm, so it lies in [0, 1].
- `new_item_share` / `new_user_share`: fraction of `current` *events* whose item / user does not occur in `reference`.
- Empty `current` or empty `reference`: return NaN (no exception).
- Polars for the heavy lifting; no Python loops over rows.

`detector.py`
```python
@dataclass(frozen=True)
class DriftThresholds:
    popularity_psi: float | None = 0.2
    popularity_jsd: float | None = None
    new_item_share: float | None = 0.15
    new_user_share: float | None = None

class DriftDetector:
    def __init__(self, thresholds: DriftThresholds | None = None, *, n_buckets: int = 10) -> None: ...
    def compare(self, reference: pl.DataFrame, current: pl.DataFrame) -> DriftReport: ...
```
- `compare` always computes all four signals. A signal fires when its threshold is not None and the value is strictly greater. `triggered` is true if any fires; `reasons` names the fired signals in a stable order. NaN signals never fire.

`policies.py`
```python
@dataclass(frozen=True, slots=True)
class PolicyContext:
    window_index: int
    windows_since_retrain: int
    drift: DriftReport | None

class RetrainPolicy(Protocol):
    name: str
    def should_retrain(self, ctx: PolicyContext) -> bool: ...

class Never; class Always; class FixedInterval(every: int); class OnDrift
def make_policy(spec: str) -> RetrainPolicy   # "never" | "always" | "fixed:<n>" | "on_drift"
```
- `Never` -> always False. `Always` -> always True. `FixedInterval(n)` -> True when `windows_since_retrain >= n` (n >= 1). `OnDrift` -> True when `ctx.drift` is not None and triggered.
- `name` values: `"never"`, `"always"`, `"fixed:<n>"`, `"on_drift"`. `make_policy(p.name)` round-trips. Unknown specs raise `ValueError` with the valid forms listed.
- Policies are stateless: all state arrives in the context.

**Tests.**
- Identical reference and current: PSI and JSD are 0 (to tolerance), new shares are 0.
- Hand-computed PSI and JSD on a tiny example.
- All current events on unseen items: `new_item_share` is 1 and the last bucket holds everything.
- Bucket shares sum to 1; fewer distinct items than buckets still works.
- Empty inputs give NaN and the detector does not trigger.
- Detector: threshold boundary (equal does not fire), `None` disables a signal, reasons listed.
- Policies: truth tables, `make_policy` round-trip and error message.

**Acceptance.** `uv run ruff check`, `uv run ruff format --check`, `uv run mypy`, `uv run pytest tests/drift tests/loop/test_policies.py` all pass.

**Out of scope.** The replay loop itself, CLI, performance-based triggers (M1).
