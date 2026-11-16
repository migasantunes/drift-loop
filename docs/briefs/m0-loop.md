# Brief: m0-loop

**Goal.** The offline simulation of the retrain loop, typed configuration, the CLI and the chart: `make demo` runs a synthetic replay end to end and produces the frozen-versus-retrained picture.

**Files you own.**
- `src/driftloop/config.py`
- `src/driftloop/loop/simulate.py`
- `src/driftloop/report/plot.py`
- `src/driftloop/cli.py`
- `configs/synthetic-demo.yaml`, `configs/synthetic-small.yaml`
- `tests/loop/test_simulate.py`, `tests/loop/test_leakage.py`, `tests/test_config.py`, `tests/test_cli.py`, `tests/report/test_plot.py`

**Depends on.** Everything already in `src/driftloop/` - read the actual modules (`data/`, `replay/clock.py`, `models/`, `eval/`, `drift/`, `loop/policies.py`) and use their real APIs. `docs/EVALUATION.md` section "The replay" is the specification of the loop; implement it exactly. Do not edit existing modules - report anything that blocks you.

**Specification.**

`config.py` - Pydantic models, `extra="forbid"` everywhere so typos fail loudly.
```yaml
name: synthetic-demo
data:
  source: synthetic            # synthetic | yambda
  synthetic: { ...SyntheticConfig fields... }
  yambda: { data_dir: data/yambda, size: 50m, min_played_ratio_pct: 50, organic_only: false, organic_value: 1 }
replay: { warmup_days: 28, step_days: 7, lookback_days: 28, tuning_windows: 0 }   # lookback_days: null = all history
eval: { k: 20 }
model: { name: popularity }
policies: [never, always, "fixed:4", on_drift]
drift: { n_buckets: 10, thresholds: { popularity_psi: 0.2, new_item_share: 0.15 } }
```
`load_config(path) -> ExperimentConfig`; policy specs validated at load time with `make_policy`; `load_interactions(cfg) -> pl.DataFrame` dispatches on `data.source`.

`loop/simulate.py`
```python
ModelFactory = Callable[[], Recommender]

@dataclass(frozen=True)
class SimulationResult:
    rows: pl.DataFrame            # one row per (policy, window)
    summary: dict[str, Any]       # per-policy aggregates + run metadata

def run_simulation(interactions: pl.DataFrame, cfg: ExperimentConfig, *, model_factory: ModelFactory | None = None) -> SimulationResult
def write_run(result: SimulationResult, cfg: ExperimentConfig, out_dir: Path) -> None   # results.csv, summary.json, config.yaml
```
Loop semantics, per policy, independently:
- Initial model: fitted at the first window's start on `history_before(t0, lookback)`. This fit is not counted as a retrain.
- At each window start `t`: drift report = detector.compare(reference = the data the model in service was trained on, i.e. `history_before(trained_until, lookback)`; current = the previous window's slice). No report for the first window. Then the policy decides; a retrain fits a **new** model from the factory on `history_before(t, lookback)` with `until=t`, timed with `time.perf_counter`. Skip a retrain that would be a no-op (`trained_until == t`). Then evaluate the model in service on the window with `evaluate_window(history=history_before(t), future=window_slice(...))`.
- All time filtering goes through `replay.clock`. Note `history` for evaluation is the full visible history (for seen-item filtering), while training uses the lookback.
- Row columns: `policy, window, start, end, k, n_users, recall, ndcg, hit_rate, coverage, retrained, fit_seconds, model_age_windows, drift_triggered` plus one column per drift signal (null where there is no report).
- Summary per policy: mean recall / ndcg / hit rate over windows (NaN-aware), number of retrains, total retrain seconds; and, when both `never` and `always` are among the policies, `quality_kept = (mean_p - mean_never) / (mean_always - mean_never)` and `cost_share = retrains_p / retrains_always` (null when the denominator is zero). All summary aggregates are computed over the **scored windows** only, i.e. windows with index >= `replay.tuning_windows` (see `docs/EVALUATION.md`, "Drift signals"); the per-window rows still contain every window, with a boolean `scored` column. Also run metadata: config name, number of windows, number of scored windows, row counts.
- Compute per-window slices once and share them across policies where that is simple; clarity beats speed here.
- Fewer than two windows -> a clear `ValueError` telling the user to lengthen the data or shorten warm-up/step.

`report/plot.py` - `plot_quality(rows: pl.DataFrame, out_path: Path, *, metric: str = "recall") -> Path`: one line per policy over window index, retrain events marked on each line, readable axis labels and legend, colour-blind-safe palette, non-interactive backend (`Agg`). The function must not need a display.

`cli.py` - Typer app `app` with commands:
```
driftloop simulate --config PATH --out DIR        # runs, writes the run directory, prints the per-policy summary table
driftloop plot RUN_DIR [--metric recall]          # reads results.csv, writes quality.png into RUN_DIR
driftloop synth --config PATH --out FILE          # writes the synthetic interactions as Parquet
driftloop yambda-download --size 50m --out DIR    # thin wrapper over data.yambda.download
```

`configs/synthetic-demo.yaml` - the `make demo` experiment: sized to finish in well under a minute and to show the effect clearly (frozen model decays, retraining recovers, drift-triggered retraining sits between). Choose the synthetic settings and thresholds by running it (the generator's default `drift_strength=1.0` is very aggressive - roughly 0.2 to 0.5 is more realistic); set `tuning_windows` to the windows you looked at while choosing thresholds; comment the file. `configs/synthetic-small.yaml` - a tiny variant for tests (a few seconds).

**Tests.**
- **Leakage canary (`test_leakage.py`).** (1) A spy model asserts inside `fit` that every row has `ts < until` and records `until`; assert each recorded `until` equals a window start. (2) Future-independence: run the simulation on data `D`; build `D'` by scrambling the item ids of every row with `ts >= T` for some window boundary `T`; the result rows for all windows ending at or before `T` must be identical between the two runs (excluding `fit_seconds`). This test must fail if any step reads past the simulated time - prove it to yourself by temporarily breaking the loop.
- Loop semantics: `never` never retrains; `always` retrains before every window except the no-op first; `fixed:n` cadence; `on_drift` retrains exactly when the report triggers; `model_age_windows` resets on retrain; the first window has null drift columns.
- Control experiments on synthetic data: with `drift_strength=0`, `never` and `always` have close mean recall; with drift, `always` beats `never` over the later windows by a clear margin and `on_drift` retrains at least once and fewer times than `always`. Keep the margins robust (fixed seeds, not knife-edge).
- Summary: `quality_kept` and `cost_share` arithmetic on a hand-checkable case, including zero denominators.
- Config: valid file loads; unknown key, bad policy spec and non-positive durations are rejected.
- CLI (Typer `CliRunner`): `simulate` on the small config writes the three files; `plot` writes a non-empty PNG; `synth` writes a Parquet file that validates as canonical.

**Acceptance.** `make check` passes for the whole repository. `make demo` produces `runs/demo/{results.csv,summary.json,config.yaml,quality.png}`.

**Out of scope.** New models, performance-based triggers, MLflow, anything in later milestones.
