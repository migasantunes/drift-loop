# syntax=docker/dockerfile:1

ARG PYTHON_VERSION=3.13

# --- builder: resolve the locked runtime dependencies into /app/.venv ---
FROM python:${PYTHON_VERSION}-slim AS builder

COPY --from=ghcr.io/astral-sh/uv:0.13.8 /uv /bin/uv

# Bytecode is compiled once at build time; copy mode because the cache is a separate mount;
# never download a Python, use the one from the base image (same in both stages).
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=0

WORKDIR /app

# Dependencies only, from uv.lock, in their own layer: it is rebuilt when the lock changes,
# not when the source does.
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --locked --no-dev --no-install-project

# The project itself, installed non-editable so the final stage needs only the venv.
COPY pyproject.toml uv.lock README.md ./
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --no-editable

# --- runtime: the venv on a clean slim base, no uv, no source, no dev tools ---
FROM python:${PYTHON_VERSION}-slim

RUN groupadd --system app \
    && useradd --system --gid app --home-dir /app --shell /usr/sbin/nologin app \
    && mkdir -p /app \
    && chown app:app /app

COPY --from=builder --chown=app:app /app/.venv /app/.venv

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1

WORKDIR /app
USER app

ENTRYPOINT ["driftloop"]
CMD ["--help"]
