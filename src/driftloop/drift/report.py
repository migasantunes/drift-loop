"""The result type shared by drift statistics, retrain policies and the loop."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class DriftReport:
    """Outcome of comparing a reference period with a more recent one.

    ``signals`` maps a statistic name (for example ``"popularity_psi"``) to its
    value. ``triggered`` is the detector's verdict, and ``reasons`` lists the
    signals that crossed their threshold, so a retrain is always explainable.
    """

    signals: dict[str, float]
    triggered: bool
    reasons: tuple[str, ...] = field(default_factory=tuple)
