"""Retrain policies: given what the loop knows now, retrain before this window or not?

Policies are stateless value objects. Everything they may use (window index, model
age, the latest drift report) arrives in a ``PolicyContext``, so a policy cannot
accumulate hidden state that would make two replays with the same inputs diverge,
and the same policy object can be shared across runs.

Each policy has a ``name`` that is also its spec string: ``make_policy(p.name)``
rebuilds an equal policy, so configs and recorded results use one vocabulary.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Protocol

from driftloop.drift.report import DriftReport


@dataclass(frozen=True, slots=True)
class PolicyContext:
    """What the loop knows at the start of a window, before deciding.

    ``drift`` is None before the first completed window, when there is nothing to
    compare against the reference yet.
    """

    window_index: int
    windows_since_retrain: int
    drift: DriftReport | None


class RetrainPolicy(Protocol):
    """Decides whether to retrain before the current window."""

    # Read-only so immutable (frozen) implementations satisfy the protocol under mypy.
    @property
    def name(self) -> str: ...

    def should_retrain(self, ctx: PolicyContext) -> bool: ...


@dataclass(frozen=True, slots=True)
class Never:
    """Keep the warm-up model forever: the lower bound on quality and on cost."""

    @property
    def name(self) -> str:
        return "never"

    def should_retrain(self, ctx: PolicyContext) -> bool:
        return False


@dataclass(frozen=True, slots=True)
class Always:
    """Retrain before every window: the reference point with the most frequent retraining.

    It is the upper bound on cost, but not guaranteed to be the best on quality in every
    window: a model fitted on one more window of noisy data can score below an older one.
    """

    @property
    def name(self) -> str:
        return "always"

    def should_retrain(self, ctx: PolicyContext) -> bool:
        return True


@dataclass(frozen=True, slots=True)
class FixedInterval:
    """Retrain once the model is ``every`` windows old: the schedule most teams run.

    ``FixedInterval(1)`` behaves like ``Always`` once the loop is past the warm-up fit.
    """

    every: int

    def __post_init__(self) -> None:
        if self.every < 1:
            raise ValueError(f"FixedInterval.every must be >= 1, got {self.every}")

    @property
    def name(self) -> str:
        return f"fixed:{self.every}"

    def should_retrain(self, ctx: PolicyContext) -> bool:
        return ctx.windows_since_retrain >= self.every


@dataclass(frozen=True, slots=True)
class OnDrift:
    """Retrain when the drift detector fired on the last completed window."""

    @property
    def name(self) -> str:
        return "on_drift"

    def should_retrain(self, ctx: PolicyContext) -> bool:
        return ctx.drift is not None and ctx.drift.triggered


_VALID_SPECS = '"never", "always", "fixed:<n>" (n >= 1), "on_drift"'
_FIXED = re.compile(r"fixed:(\d+)")


def make_policy(spec: str) -> RetrainPolicy:
    """Build a policy from its spec string (the inverse of ``policy.name``)."""
    if spec == "never":
        return Never()
    if spec == "always":
        return Always()
    if spec == "on_drift":
        return OnDrift()
    match = _FIXED.fullmatch(spec)
    if match is not None and int(match.group(1)) >= 1:
        return FixedInterval(int(match.group(1)))
    raise ValueError(f"unknown retrain policy {spec!r}; valid forms: {_VALID_SPECS}")
