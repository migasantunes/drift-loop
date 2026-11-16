"""Threshold detector: boundaries, disabled signals, reason order and empty input."""

from __future__ import annotations

import math

import polars as pl
import pytest

from driftloop.data.schema import empty_interactions, to_canonical
from driftloop.drift.detector import SIGNAL_NAMES, DriftDetector, DriftThresholds
from driftloop.drift.stats import popularity_jsd, popularity_psi

OFF = DriftThresholds(
    popularity_psi=None, popularity_jsd=None, new_item_share=None, new_user_share=None
)


def frame(items: list[int], users: list[int]) -> pl.DataFrame:
    return to_canonical(
        pl.DataFrame({"user_id": users, "item_id": items, "ts": list(range(len(items)))})
    )


# Half the current events are on a new item (3) and from a new user (9); among known
# items all current demand moved to item 1, so the popularity signals are positive too.
REF = frame([1, 1, 2, 2], users=[1, 2, 1, 2])
CUR = frame([1, 3, 1, 3], users=[1, 9, 2, 9])


def test_signal_names_and_order() -> None:
    assert SIGNAL_NAMES == ("popularity_psi", "popularity_jsd", "new_item_share", "new_user_share")
    report = DriftDetector(OFF).compare(REF, CUR)
    assert tuple(report.signals) == SIGNAL_NAMES


def test_signals_match_stats_functions() -> None:
    report = DriftDetector(OFF, n_buckets=2).compare(REF, CUR)
    assert report.signals["popularity_psi"] == pytest.approx(popularity_psi(REF, CUR, n_buckets=2))
    assert report.signals["popularity_jsd"] == pytest.approx(popularity_jsd(REF, CUR, n_buckets=2))
    assert report.signals["new_item_share"] == pytest.approx(0.5)
    assert report.signals["new_user_share"] == pytest.approx(0.5)


def test_all_disabled_never_fires() -> None:
    report = DriftDetector(OFF).compare(REF, CUR)
    assert not report.triggered
    assert report.reasons == ()
    # Disabled signals are still computed and recorded.
    assert all(not math.isnan(v) for v in report.signals.values())


def test_threshold_equal_does_not_fire_and_strictly_greater_does() -> None:
    at = DriftDetector(DriftThresholds(popularity_psi=None, new_item_share=0.5)).compare(REF, CUR)
    assert not at.triggered
    below = DriftDetector(DriftThresholds(popularity_psi=None, new_item_share=0.49))
    report = below.compare(REF, CUR)
    assert report.triggered
    assert report.reasons == ("new_item_share",)


def test_psi_boundary_uses_computed_value() -> None:
    value = popularity_psi(REF, CUR)
    at = DriftDetector(DriftThresholds(popularity_psi=value, new_item_share=None))
    assert not at.compare(REF, CUR).triggered
    below = DriftDetector(DriftThresholds(popularity_psi=value * 0.999, new_item_share=None))
    assert below.compare(REF, CUR).reasons == ("popularity_psi",)


def test_none_disables_only_that_signal() -> None:
    thresholds = DriftThresholds(
        popularity_psi=None, popularity_jsd=0.0, new_item_share=None, new_user_share=0.0
    )
    report = DriftDetector(thresholds).compare(REF, CUR)
    assert report.reasons == ("popularity_jsd", "new_user_share")


def test_reasons_listed_in_stable_order() -> None:
    zero = DriftThresholds(
        popularity_psi=0.0, popularity_jsd=0.0, new_item_share=0.0, new_user_share=0.0
    )
    report = DriftDetector(zero).compare(REF, CUR)
    assert report.triggered
    assert report.reasons == SIGNAL_NAMES


def test_default_thresholds() -> None:
    detector = DriftDetector()
    assert detector.thresholds == DriftThresholds(0.2, None, 0.15, None)
    assert detector.compare(REF, REF).triggered is False
    assert detector.compare(REF, CUR).triggered is True


@pytest.mark.parametrize(
    ("ref", "cur"),
    [(empty_interactions(), CUR), (REF, empty_interactions()), (empty_interactions(),) * 2],
)
def test_empty_input_gives_nan_and_never_fires(ref: pl.DataFrame, cur: pl.DataFrame) -> None:
    # Negative thresholds would fire on any real number; NaN still must not.
    negative = DriftThresholds(
        popularity_psi=-1.0, popularity_jsd=-1.0, new_item_share=-1.0, new_user_share=-1.0
    )
    report = DriftDetector(negative).compare(ref, cur)
    assert tuple(report.signals) == SIGNAL_NAMES
    assert all(math.isnan(v) for v in report.signals.values())
    assert not report.triggered
    assert report.reasons == ()


def test_nan_threshold_rejected() -> None:
    with pytest.raises(ValueError, match="new_item_share"):
        DriftThresholds(new_item_share=math.nan)


def test_invalid_bucket_count_rejected() -> None:
    with pytest.raises(ValueError, match="n_buckets"):
        DriftDetector(n_buckets=0)


def test_new_items_alone_do_not_move_popularity_signals() -> None:
    # Known-item mix unchanged (items 1 and 2 at 50/50); half the events on new item 3.
    cur = frame([1, 3, 2, 3], users=[1, 2, 1, 2])
    report = DriftDetector().compare(REF, cur)
    assert report.signals["popularity_psi"] == pytest.approx(0.0, abs=1e-12)
    assert report.signals["popularity_jsd"] == pytest.approx(0.0, abs=1e-12)
    assert report.reasons == ("new_item_share",)


def test_only_new_items_leaves_popularity_signals_undefined() -> None:
    zero = DriftThresholds(
        popularity_psi=0.0, popularity_jsd=0.0, new_item_share=0.0, new_user_share=0.0
    )
    report = DriftDetector(zero).compare(REF, frame([7, 8], users=[1, 2]))
    assert math.isnan(report.signals["popularity_psi"])
    assert math.isnan(report.signals["popularity_jsd"])
    assert report.signals["new_item_share"] == 1.0
    assert report.reasons == ("new_item_share",)
