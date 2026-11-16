"""Retrain policies: truth tables and the spec round-trip."""

from __future__ import annotations

import pytest

from driftloop.drift.report import DriftReport
from driftloop.loop.policies import (
    Always,
    FixedInterval,
    Never,
    OnDrift,
    PolicyContext,
    RetrainPolicy,
    make_policy,
)

FIRED = DriftReport(signals={"new_item_share": 0.5}, triggered=True, reasons=("new_item_share",))
QUIET = DriftReport(signals={"new_item_share": 0.0}, triggered=False)


def ctx(since: int = 0, drift: DriftReport | None = None, index: int = 0) -> PolicyContext:
    return PolicyContext(window_index=index, windows_since_retrain=since, drift=drift)


CONTEXTS = [ctx(0), ctx(1, QUIET), ctx(5, FIRED), ctx(100, None, index=100)]


@pytest.mark.parametrize("c", CONTEXTS)
def test_never_and_always(c: PolicyContext) -> None:
    assert Never().should_retrain(c) is False
    assert Always().should_retrain(c) is True


@pytest.mark.parametrize(
    ("every", "since", "expected"),
    [(1, 0, False), (1, 1, True), (3, 2, False), (3, 3, True), (3, 4, True)],
)
def test_fixed_interval(every: int, since: int, expected: bool) -> None:
    # Drift is irrelevant to a fixed schedule.
    for drift in (None, QUIET, FIRED):
        assert FixedInterval(every).should_retrain(ctx(since, drift)) is expected


@pytest.mark.parametrize(("drift", "expected"), [(None, False), (QUIET, False), (FIRED, True)])
def test_on_drift(drift: DriftReport | None, expected: bool) -> None:
    # Model age is irrelevant to the drift trigger.
    for since in (0, 1, 50):
        assert OnDrift().should_retrain(ctx(since, drift)) is expected


@pytest.mark.parametrize("every", [0, -2])
def test_fixed_interval_rejects_non_positive(every: int) -> None:
    with pytest.raises(ValueError, match="every"):
        FixedInterval(every)


@pytest.mark.parametrize(
    ("policy", "name"),
    [
        (Never(), "never"),
        (Always(), "always"),
        (FixedInterval(1), "fixed:1"),
        (FixedInterval(12), "fixed:12"),
        (OnDrift(), "on_drift"),
    ],
)
def test_names_round_trip(policy: RetrainPolicy, name: str) -> None:
    assert policy.name == name
    rebuilt = make_policy(policy.name)
    assert rebuilt == policy
    assert type(rebuilt) is type(policy)


@pytest.mark.parametrize(
    "spec", ["", "Never", "fixed", "fixed:", "fixed:0", "fixed:-1", "fixed:x", "fixed: 3", "drift"]
)
def test_unknown_spec_lists_valid_forms(spec: str) -> None:
    with pytest.raises(ValueError, match="valid forms") as err:
        make_policy(spec)
    message = str(err.value)
    for form in ("never", "always", "fixed:<n>", "on_drift"):
        assert form in message


def test_policies_are_immutable() -> None:
    policy = FixedInterval(3)
    with pytest.raises(AttributeError):
        policy.every = 4  # type: ignore[misc]
