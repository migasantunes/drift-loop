.DEFAULT_GOAL := help

IMAGE ?= drift-loop:dev

.PHONY: help install lint fmt types test check demo docker-build clean

help: ## List the available targets
	@grep -E '^[a-zA-Z_-]+:.*## ' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*## "} {printf "  %-14s %s\n", $$1, $$2}'

install: ## Install the package and dev dependencies into .venv
	uv sync

lint: ## Lint and check formatting (read-only)
	uv run ruff check .
	uv run ruff format --check .

fmt: ## Auto-format and apply lint fixes
	uv run ruff check --fix .
	uv run ruff format .

types: ## Type-check src/ with mypy
	uv run mypy

test: ## Run the test suite
	uv run pytest

check: lint types test ## Lint, format check, types and tests: run before every commit

demo: ## Synthetic replay end to end, writes runs/demo/
	uv run driftloop simulate --config configs/synthetic-demo.yaml --out runs/demo
	uv run driftloop plot runs/demo

docker-build: ## Build the container image drift-loop:dev (override with IMAGE=...)
	docker build -t $(IMAGE) .

clean: ## Remove caches, build artefacts and runs/
	rm -rf .pytest_cache .ruff_cache .mypy_cache .coverage htmlcov dist build runs
	find . -path ./.venv -prune -o -type d -name __pycache__ -exec rm -rf {} +
