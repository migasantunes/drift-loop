# Brief: m0-data

**Goal.** A synthetic interaction generator with controllable drift, the Yambda
adapter, and tests for the two foundation modules that already exist
(`data/schema.py`, `replay/clock.py`).

**Files you own.**
- `src/driftloop/data/synthetic.py`
- `src/driftloop/data/yambda.py`
- `tests/data/test_schema.py`, `tests/data/test_synthetic.py`, `tests/data/test_yambda.py`
- `tests/replay/test_clock.py`

**Depends on.** `src/driftloop/data/schema.py`, `src/driftloop/replay/clock.py` (read them; do not edit them - if you find a bug, report it). `docs/DATASET.md`.

**Specification.**

`synthetic.py`
```python
@dataclass(frozen=True)
class SyntheticConfig:
    n_users: int = 2000
    n_new_users_per_day: float = 5.0
    n_items_initial: int = 3000
    n_new_items_per_day: float = 40.0
    n_clusters: int = 20
    n_days: int = 120
    events_per_user_per_day: float = 3.0
    drift_strength: float = 1.0
    seed: int = 0

def generate_interactions(cfg: SyntheticConfig) -> pl.DataFrame: ...
```
- Returns a canonical interaction table (use `to_canonical`). `ts` in seconds, spread within each day.
- Generative story: items belong to latent clusters; each user has a preference distribution over clusters; within a cluster, item choice follows a long-tailed popularity.
- Non-stationarity, all scaled by `drift_strength`: (a) new items are released every day and get a freshness boost that decays over a couple of weeks; (b) cluster-level popularity follows a slow random walk (global trends); (c) individual users' cluster preferences drift slowly; (d) new users join over time.
- `drift_strength = 0` must give a **stationary** process: no new items, no new users, no trend or taste movement. This is used downstream as a control experiment (a frozen model should not decay).
- Deterministic given `seed`. Vectorised NumPy; the default config must generate in a few seconds. Validate config values (positive sizes, `drift_strength >= 0`).
- Add any further config fields you need, with defaults, and document each.

`yambda.py`
```python
def listens_to_interactions(
    listens: pl.DataFrame | pl.LazyFrame, *,
    min_played_ratio_pct: int = 50,
    organic_only: bool = False,
    organic_value: int = 1,
) -> pl.DataFrame: ...

def download(size: str, out_dir: Path, files: Sequence[str] = ("listens.parquet",)) -> list[Path]: ...

def load_interactions(data_dir: Path, size: str = "50m", **kwargs) -> pl.DataFrame: ...
```
- Raw columns and semantics are in `docs/DATASET.md` (`timestamp` is in 5-second units; `ts = timestamp * 5`).
- `listens_to_interactions` stays lazy until the final collect and returns a canonical table.
- `organic_value` is a parameter because the dataset card is ambiguous about which value means organic (open question in `DATASET.md`).
- `download` fetches `flat/<size>/<file>` from the Hub dataset `yandex/yambda` with `huggingface_hub` (imported lazily; raise a clear error naming `uv sync --extra data` if it is missing). Validate `size` against `{"50m", "500m", "5b"}`. Do not call the network in tests.
- `load_interactions` scans `<data_dir>/flat/<size>/listens.parquet` and applies the transform.

**Tests.**
- Schema: accepts a valid frame; rejects wrong columns, wrong dtype, nulls, negative ts, unsorted ts; `to_canonical` fills weight and sorts; empty frame is valid.
- Clock: window boundaries are half-open and consecutive; the trailing partial window is dropped; the inclusive `t_max` edge case; `history_before` never returns `ts >= t` (include a row exactly at `t`); lookback bounds; `window_slice` bounds; invalid arguments raise.
- Synthetic: canonical schema; determinism; different seeds differ; with `drift_strength=0` no item first appears after the first few days and item-popularity distributions of the first and last thirds are close; with drift the share of late events on items unseen in the first third is clearly larger than without.
- Yambda: built from small hand-made frames - unit conversion, played-ratio threshold (boundary value included), organic filter with both `organic_value` settings, lazy and eager input, bad `size` rejected.

**Acceptance.** `uv run ruff check`, `uv run ruff format --check`, `uv run mypy`, `uv run pytest tests/data tests/replay` all pass.

**Out of scope.** CLI commands, configuration loading, any model.
