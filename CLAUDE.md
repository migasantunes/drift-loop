# drift-loop

A recommender system that notices when it is going stale and retrains itself.
Historical interactions are replayed in time order as if they were live traffic;
the system detects drift, retrains, gates the new model and promotes it.
The headline result is quality over time for a frozen model versus retraining
policies, against what each policy costs.

Owner: Miguel (MSc AI student, aiming at ML engineering). This is a CV project, so
two things matter beyond working code: results must be honest, and Miguel must be
able to explain every design decision in an interview.

## Read first

- `docs/PLAN.md` - milestones, tasks, acceptance criteria. Check the status table before starting work.
- `docs/EVALUATION.md` - the protocol. Any change to how things are measured goes here first.
- `docs/ARCHITECTURE.md` - layout and the three contracts.
- `docs/DECISIONS.md` - why things are the way they are. Append when you decide something.

## Commands

```bash
uv sync                      # install (add --extra data for the dataset download)
make check                   # lint + format check + types + tests: run before every commit
make test                    # tests only
make fmt                     # auto-format and fix lint
uv run driftloop --help      # CLI
make demo                    # synthetic replay end to end, writes runs/demo/
```

## Rules

1. **No reading the future.** At simulated time t nothing with ts >= t exists. Filter
   by time only through `replay/clock.py`. A change that touches time handling needs
   a test that would fail if leakage were introduced.
2. **Contracts are stable.** The interaction schema, the time-slicing functions and
   the `Recommender` protocol change only with an entry in `docs/DECISIONS.md`.
3. **No data in git.** Nothing from any dataset is committed, including samples.
   Tests use the synthetic generator or hand-built frames.
4. **No placeholders.** Don't add empty modules, TODO stubs or unused dependencies.
   A dependency is added by the milestone that first imports it.
5. **Tests come with the code.** New behaviour has tests; `make check` passes.
6. **Honest results.** Report what was measured, with the settings that produced it.
   If a model does not beat a baseline, say so and analyse it. Never tune thresholds
   on the data they are evaluated on.
7. **Scale-aware code.** The real table has 46 M rows. Use Polars expressions and
   lazy frames; never iterate rows in Python or materialise the table as Python objects.
8. **Costs.** Nothing that bills money (AWS) runs without Miguel's explicit go-ahead
   in that session, and `docs/COSTS.md` is updated afterwards.

## Workflow

- One branch and one pull request per milestone (`m1/real-data`, ...). Never commit to `main` directly.
- The lead session plans a milestone into briefs in `docs/briefs/` (template there),
  with disjoint file sets so they can run in parallel.
- Use the `implementer` sub-agent for each brief and the `reviewer` sub-agent before
  opening the pull request. The reviewer starts fresh and does not edit code.
- Sub-agents do not run git commands and do not edit `pyproject.toml` or `uv.lock`;
  they report needed dependency changes to the lead, who applies them.
- Before a milestone is merged, Miguel answers its interview checkpoint in
  `docs/PLAN.md`. Offer to quiz him. Explain rather than only implement: when a
  decision is made, say what the alternatives were.
- Commit messages: imperative, scoped (`eval: add ndcg@k`).

## Style

Python 3.11+, type hints everywhere, ruff-formatted, 100 columns. Docstrings explain
why, not what. Prefer small pure functions over classes; keep I/O at the edges
(CLI, adapters).
