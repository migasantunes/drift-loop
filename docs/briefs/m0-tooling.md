# Brief: m0-tooling

**Goal.** The developer tooling around the package: one-command checks, CI, container image, licence.

**Files you own.**
- `Makefile`
- `.github/workflows/ci.yml`
- `.pre-commit-config.yaml`
- `Dockerfile`, `.dockerignore`
- `LICENSE` (MIT, copyright 2026 José Miguel Luís Antunes)
- `.python-version`

**Depends on.** `pyproject.toml` (read-only), `CLAUDE.md` section "Commands".

**Specification.**

The CLI is being built in parallel; its interface is fixed:
```
driftloop simulate --config PATH --out DIR
driftloop plot RUN_DIR
driftloop synth --config PATH --out FILE
driftloop yambda-download --size 50m --out DIR
```

`Makefile` (with a `help` target that lists targets, `.PHONY` declared):
- `install` -> `uv sync`
- `lint` -> `uv run ruff check .` and `uv run ruff format --check .`
- `fmt` -> `uv run ruff check --fix .` and `uv run ruff format .`
- `types` -> `uv run mypy`
- `test` -> `uv run pytest`
- `check` -> lint, types, test
- `demo` -> `uv run driftloop simulate --config configs/synthetic-demo.yaml --out runs/demo` then `uv run driftloop plot runs/demo`
- `docker-build` -> builds image `drift-loop:dev`
- `clean` -> removes caches and `runs/`

`.github/workflows/ci.yml`: on push to `main` and on pull requests; a `check` job on ubuntu with Python 3.11 and 3.13 that installs uv (official setup action, with caching), runs `uv sync --locked` and `make check`; a `docker` job that builds the image without pushing. Minimal `permissions: contents: read`. Concurrency group that cancels superseded runs. Verify the current major versions of every action you reference (web search) rather than guessing.

`Dockerfile`: multi-stage, uv-based, slim Python base, dependencies installed from `uv.lock` in a cached layer before the source is copied, non-root user, `ENTRYPOINT ["driftloop"]`. No dev dependencies in the final image.

`.pre-commit-config.yaml`: ruff (lint with fix, format) and a few standard hygiene hooks (trailing whitespace, end-of-file, large files, merge conflicts, YAML check). Pin revisions that exist.

**Tests / verification.** Run `make lint`, `make types`, `make test` (they must work on the code present; `make demo` cannot work until the CLI lands - say so in your report, do not fake it). Try `docker build`; if the sandbox cannot pull base images, say so and review the Dockerfile by reading instead. Validate the workflow YAML parses.

**Acceptance.** `make help`, `make lint`, `make types`, `make test` run. Report anything you could not verify.

**Out of scope.** Editing `pyproject.toml`, `uv.lock`, source or tests. Kubernetes, Terraform, compose files (later milestones).
