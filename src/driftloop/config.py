"""Typed experiment configuration, loaded from one YAML file per experiment.

Every model forbids unknown keys: a misspelt key in a config file (``lookback_day``)
would otherwise be silently ignored and the run would use a default the author did
not intend, which is the kind of error that produces a wrong result nobody notices.

Validation happens at load time, including policy specs and the synthetic generator's
own checks, so a bad config fails before any data is generated or any model trained.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import polars as pl
import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from driftloop.data import yambda
from driftloop.data.synthetic import SyntheticConfig, generate_interactions
from driftloop.drift.detector import DriftThresholds
from driftloop.loop.policies import make_policy
from driftloop.replay.clock import DAY


class _Strict(BaseModel):
    """Base for every config section: unknown keys are errors, instances are immutable."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class SyntheticSettings(_Strict):
    """YAML mirror of :class:`driftloop.data.synthetic.SyntheticConfig`.

    The fields are spelt out rather than derived from the dataclass so that the
    config schema is readable and type-checked; a test asserts that the field names
    and defaults stay identical to the generator's, so the two cannot drift apart.
    The generator's own range checks run at load time (see ``_check``).
    """

    n_users: int = 2000
    n_new_users_per_day: float = 5.0
    n_items_initial: int = 3000
    n_new_items_per_day: float = 40.0
    n_clusters: int = 20
    n_days: int = 120
    events_per_user_per_day: float = 3.0
    drift_strength: float = 1.0
    seed: int = 0
    preference_concentration: float = 0.3
    item_popularity_exponent: float = 1.0
    freshness_boost: float = 20.0
    freshness_decay_days: float = 5.0
    trend_volatility: float = 0.08
    taste_volatility: float = 0.05

    def to_synthetic_config(self) -> SyntheticConfig:
        return SyntheticConfig(**self.model_dump())

    @model_validator(mode="after")
    def _check(self) -> SyntheticSettings:
        # SyntheticConfig.__post_init__ raises ValueError, which pydantic reports as a
        # validation error pointing at this section.
        self.to_synthetic_config()
        return self


class YambdaSettings(_Strict):
    """Arguments of :func:`driftloop.data.yambda.load_interactions`."""

    data_dir: Path = Path("data/yambda")
    size: Literal["50m", "500m", "5b"] = "50m"
    min_played_ratio_pct: int = Field(default=50, ge=0)
    organic_only: bool = False
    organic_value: Literal[0, 1] = 1


class DataConfig(_Strict):
    source: Literal["synthetic", "yambda"]
    synthetic: SyntheticSettings = SyntheticSettings()
    yambda: YambdaSettings = YambdaSettings()


class ReplayConfig(_Strict):
    """Replay geometry in days; converted to seconds where the clock needs them.

    Days, not seconds, in the file because that is the unit people reason in, and
    integers because a fractional-day window has no use here and would make window
    boundaries fall at odd times of day.
    """

    warmup_days: int = Field(default=28, gt=0)
    step_days: int = Field(default=7, gt=0)
    #: Rolling training window; ``None`` trains on all visible history.
    lookback_days: int | None = Field(default=28, gt=0)
    #: Leading windows used to choose the drift thresholds; excluded from the summary.
    tuning_windows: int = Field(default=0, ge=0)

    @property
    def warmup(self) -> int:
        return self.warmup_days * DAY

    @property
    def step(self) -> int:
        return self.step_days * DAY

    @property
    def lookback(self) -> int | None:
        return None if self.lookback_days is None else self.lookback_days * DAY


class EvalConfig(_Strict):
    k: int = Field(default=20, ge=1)


class ModelConfig(_Strict):
    #: Only the popularity baseline exists in M0; M1 widens this literal.
    name: Literal["popularity"] = "popularity"


class ThresholdsConfig(_Strict):
    """YAML mirror of :class:`DriftThresholds`; ``null`` disables a signal.

    Defaults match the detector's so that omitting the section and omitting a key
    mean the same thing. A disabled signal is still computed and recorded.
    """

    popularity_psi: float | None = 0.2
    popularity_jsd: float | None = None
    new_item_share: float | None = 0.15
    new_user_share: float | None = None

    def to_thresholds(self) -> DriftThresholds:
        return DriftThresholds(**self.model_dump())

    @model_validator(mode="after")
    def _check(self) -> ThresholdsConfig:
        self.to_thresholds()  # rejects NaN thresholds
        return self


class DriftConfig(_Strict):
    n_buckets: int = Field(default=10, ge=1)
    thresholds: ThresholdsConfig = ThresholdsConfig()


class ExperimentConfig(_Strict):
    name: str = Field(min_length=1)
    data: DataConfig
    replay: ReplayConfig = ReplayConfig()
    eval: EvalConfig = EvalConfig()
    model: ModelConfig = ModelConfig()
    policies: list[str] = Field(min_length=1)
    drift: DriftConfig = DriftConfig()

    @field_validator("policies")
    @classmethod
    def _check_policies(cls, specs: list[str]) -> list[str]:
        """Parse every spec now and store its canonical name.

        Canonical names (``fixed:04`` becomes ``fixed:4``) are what appear in the
        results, so two spellings of the same policy cannot both run and collide.
        """
        names = [make_policy(spec).name for spec in specs]
        duplicates = sorted({n for n in names if names.count(n) > 1})
        if duplicates:
            raise ValueError(f"policies listed more than once: {duplicates}")
        return names


def load_config(path: Path) -> ExperimentConfig:
    """Read and validate an experiment YAML file.

    Raises ``pydantic.ValidationError`` (a ``ValueError``) for a bad config and
    ``ValueError`` when the file is not a YAML mapping.
    """
    raw: Any = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: expected a YAML mapping at the top level")
    return ExperimentConfig.model_validate(raw)


def load_interactions(cfg: ExperimentConfig) -> pl.DataFrame:
    """Produce the canonical interaction table the experiment runs on."""
    data = cfg.data
    if data.source == "synthetic":
        return generate_interactions(data.synthetic.to_synthetic_config())
    y = data.yambda
    return yambda.load_interactions(
        y.data_dir,
        y.size,
        min_played_ratio_pct=y.min_played_ratio_pct,
        organic_only=y.organic_only,
        organic_value=y.organic_value,
    )
