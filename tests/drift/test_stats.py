"""Drift statistics checked against values worked out by hand."""

from __future__ import annotations

import math

import numpy as np
import polars as pl
import pytest

from driftloop.data.schema import empty_interactions, to_canonical
from driftloop.drift.stats import (
    jensen_shannon,
    known_item_shares,
    new_item_share,
    new_user_share,
    popularity_bucket_shares,
    popularity_jsd,
    popularity_psi,
    psi,
)


def frame(
    items: list[int], users: list[int] | None = None, weights: list[float] | None = None
) -> pl.DataFrame:
    """Canonical table with one event per entry of ``items``, timestamps 0, 1, 2, ..."""
    n = len(items)
    data: dict[str, list[int] | list[float]] = {
        "user_id": users if users is not None else [0] * n,
        "item_id": items,
        "ts": list(range(n)),
    }
    if weights is not None:
        data["weight"] = weights
    return to_canonical(pl.DataFrame(data))


# Reference: item 1 x4, item 2 x3, item 3 x2, item 4 x1 (10 events).
# With 2 buckets an item goes to floor(2 * events_before / 10):
#   item 1: before 0 -> 0;  item 2: before 4 -> 0;  item 3: before 7 -> 1;  item 4: before 9 -> 1.
# Reference shares: [7/10, 3/10, 0].
REFERENCE = [1, 1, 1, 1, 2, 2, 2, 3, 3, 4]
# Current: item 1 x1 (bucket 0), item 3 x2 (bucket 1), unseen item 9 x1 -> [1/4, 2/4, 1/4].
# Known items only (PSI, JSD): drop the unseen bucket, renormalise over 3 events -> [1/3, 2/3].
CURRENT = [1, 3, 3, 9]


def test_bucket_shares_hand_example() -> None:
    ref, cur = popularity_bucket_shares(frame(REFERENCE), frame(CURRENT), n_buckets=2)
    np.testing.assert_allclose(ref, [0.7, 0.3, 0.0])
    np.testing.assert_allclose(cur, [0.25, 0.5, 0.25])


def test_known_item_shares_hand_example() -> None:
    ref, cur = known_item_shares(
        *popularity_bucket_shares(frame(REFERENCE), frame(CURRENT), n_buckets=2)
    )
    np.testing.assert_allclose(ref, [0.7, 0.3])
    np.testing.assert_allclose(cur, [1 / 3, 2 / 3])


def test_known_item_shares_undefined_without_known_events() -> None:
    ref, cur = known_item_shares(np.array([0.6, 0.4, 0.0]), np.array([0.0, 0.0, 1.0]))
    assert ref.shape == cur.shape == (2,)
    assert np.isnan(ref).all()
    assert np.isnan(cur).all()


def test_psi_hand_example() -> None:
    # Over known items only: p = [0.7, 0.3], q = [1/3, 2/3]; no bucket is empty, so eps
    # plays no part.
    # (1/3 - 0.7) ln((1/3) / 0.7) + (2/3 - 0.3) ln((2/3) / 0.3) = 0.272044 + 0.292786
    expected = 0.5648298483472879
    value = popularity_psi(frame(REFERENCE), frame(CURRENT), n_buckets=2, eps=1e-4)
    assert value == pytest.approx(expected, rel=1e-12)
    assert popularity_psi(frame(REFERENCE), frame(CURRENT), n_buckets=2, eps=1e-8) == pytest.approx(
        expected, rel=1e-12
    )


def test_jsd_hand_example() -> None:
    # p = [0.7, 0.3], q = [1/3, 2/3], m = [31/60, 29/60]
    # KL(p||m) = 0.7 log2(0.7/(31/60)) + 0.3 log2(0.3/(29/60))
    # KL(q||m) = (1/3) log2((1/3)/(31/60)) + (2/3) log2((2/3)/(29/60))
    # JSD = (KL(p||m) + KL(q||m)) / 2
    expected = 0.09940498762104887
    value = popularity_jsd(frame(REFERENCE), frame(CURRENT), n_buckets=2)
    assert value == pytest.approx(expected, rel=1e-12)


def test_new_item_events_do_not_move_psi_or_jsd() -> None:
    # Same mix among known items (1/3, 2/3), with 0, 1 and 7 events on new items added.
    ref = frame(REFERENCE)
    base = frame([1, 3, 3])
    for extra in ([9], [9, 10, 11, 11, 12, 13, 14]):
        cur = frame([1, 3, 3, *extra])
        assert popularity_psi(ref, cur, n_buckets=2) == pytest.approx(
            popularity_psi(ref, base, n_buckets=2), rel=1e-12
        )
        assert popularity_jsd(ref, cur, n_buckets=2) == pytest.approx(
            popularity_jsd(ref, base, n_buckets=2), rel=1e-12
        )
        assert new_item_share(ref, cur) > 0.0


def test_empty_popularity_bucket_is_clipped_to_eps() -> None:
    # Nobody plays the head bucket any more: q = [0, 1] against p = [0.7, 0.3].
    # (eps - 0.7) ln(eps / 0.7) + (1 - 0.3) ln(1 / 0.3), eps = 1e-4
    expected = (1e-4 - 0.7) * math.log(1e-4 / 0.7) + 0.7 * math.log(1 / 0.3)
    value = popularity_psi(frame(REFERENCE), frame([3, 4, 9]), n_buckets=2, eps=1e-4)
    assert value == pytest.approx(expected, rel=1e-12)


def test_psi_function_on_shares() -> None:
    # Two buckets, no clipping in play: (0.6-0.5) ln(0.6/0.5) + (0.4-0.5) ln(0.4/0.5).
    expected = 0.1 * math.log(1.2) - 0.1 * math.log(0.8)
    assert psi(np.array([0.5, 0.5]), np.array([0.6, 0.4])) == pytest.approx(expected)


def test_jsd_bounds() -> None:
    assert jensen_shannon(np.array([1.0, 0.0]), np.array([0.0, 1.0])) == pytest.approx(1.0)
    assert jensen_shannon(np.array([0.3, 0.7]), np.array([0.3, 0.7])) == 0.0
    # Symmetric, unlike KL.
    p, q = np.array([0.7, 0.3, 0.0]), np.array([0.25, 0.5, 0.25])
    assert jensen_shannon(p, q) == pytest.approx(jensen_shannon(q, p))


def test_identical_periods_have_no_drift() -> None:
    ref = frame(REFERENCE, users=list(range(10)))
    cur = frame(REFERENCE, users=list(range(10)))
    assert popularity_psi(ref, cur) == pytest.approx(0.0, abs=1e-12)
    assert popularity_jsd(ref, cur) == pytest.approx(0.0, abs=1e-12)
    assert new_item_share(ref, cur) == 0.0
    assert new_user_share(ref, cur) == 0.0


def test_all_current_on_unseen_items() -> None:
    ref, cur = frame([1, 2, 3]), frame([7, 8, 8, 9])
    assert new_item_share(ref, cur) == 1.0
    ref_shares, cur_shares = popularity_bucket_shares(ref, cur, n_buckets=3)
    np.testing.assert_array_equal(cur_shares, [0.0, 0.0, 0.0, 1.0])
    assert ref_shares[-1] == 0.0
    # Nothing left among known items to compare: the popularity signals are undefined.
    assert math.isnan(popularity_psi(ref, cur, n_buckets=3))
    assert math.isnan(popularity_jsd(ref, cur, n_buckets=3))


@pytest.mark.parametrize("n_buckets", [1, 2, 3, 7, 10, 50])
def test_shares_sum_to_one_with_right_length(n_buckets: int) -> None:
    ref, cur = popularity_bucket_shares(frame(REFERENCE), frame(CURRENT), n_buckets=n_buckets)
    assert ref.shape == cur.shape == (n_buckets + 1,)
    assert ref.sum() == pytest.approx(1.0)
    assert cur.sum() == pytest.approx(1.0)
    assert ref[-1] == 0.0


def test_fewer_items_than_buckets() -> None:
    # Three equally popular items, ten buckets: floor(10 * {0, 1, 2} / 3) = buckets 0, 3, 6.
    ref, cur = popularity_bucket_shares(frame([1, 2, 3]), frame([3, 3, 4]), n_buckets=10)
    expected_ref = np.zeros(11)
    expected_ref[[0, 3, 6]] = 1 / 3
    np.testing.assert_allclose(ref, expected_ref)
    expected_cur = np.zeros(11)
    expected_cur[6] = 2 / 3
    expected_cur[10] = 1 / 3
    np.testing.assert_allclose(cur, expected_cur)


def test_dominant_head_item_stays_in_first_bucket() -> None:
    # Item 1 holds 6 of 10 events, more than one bucket's worth (4 buckets -> 2.5 each).
    # Buckets: item 1 floor(4*0/10)=0, item 2 floor(4*6/10)=2, item 3 floor(4*8/10)=3.
    ref, _ = popularity_bucket_shares(frame([1] * 6 + [2, 2, 3, 3]), frame([1]), n_buckets=4)
    np.testing.assert_allclose(ref, [0.6, 0.0, 0.2, 0.2, 0.0])


def test_ties_broken_by_item_id_not_row_order() -> None:
    # Items 2 and 5 are equally popular; the lower id ranks first whatever the row order.
    for ref_items in ([5, 2], [2, 5]):
        _, cur = popularity_bucket_shares(frame(ref_items), frame([2]), n_buckets=2)
        np.testing.assert_array_equal(cur, [1.0, 0.0, 0.0])


def test_events_are_counted_not_weighted() -> None:
    plain = popularity_bucket_shares(frame(REFERENCE), frame(CURRENT), n_buckets=2)
    weighted = popularity_bucket_shares(
        frame(REFERENCE, weights=[9.0] * 4 + [0.1] * 6),
        frame(CURRENT, weights=[5.0, 1.0, 1.0, 0.5]),
        n_buckets=2,
    )
    np.testing.assert_array_equal(plain[0], weighted[0])
    np.testing.assert_array_equal(plain[1], weighted[1])


def test_new_item_and_user_share_count_events() -> None:
    ref = frame([1, 2], users=[1, 2])
    cur = frame([1, 2, 2, 3], users=[1, 3, 3, 4])
    # Items: only item 3 is new -> 1 of 4 events. Users: 3, 3, 4 are new -> 3 of 4 events.
    assert new_item_share(ref, cur) == pytest.approx(0.25)
    assert new_user_share(ref, cur) == pytest.approx(0.75)


@pytest.mark.parametrize("which", ["reference", "current", "both"])
def test_empty_inputs_give_nan(which: str) -> None:
    full = frame(REFERENCE)
    ref = empty_interactions() if which in ("reference", "both") else full
    cur = empty_interactions() if which in ("current", "both") else full
    assert math.isnan(popularity_psi(ref, cur))
    assert math.isnan(popularity_jsd(ref, cur))
    assert math.isnan(new_item_share(ref, cur))
    assert math.isnan(new_user_share(ref, cur))
    ref_shares, cur_shares = popularity_bucket_shares(ref, cur, n_buckets=3)
    assert ref_shares.shape == cur_shares.shape == (4,)
    assert np.isnan(ref_shares).all()
    assert np.isnan(cur_shares).all()


def test_invalid_bucket_count_rejected() -> None:
    with pytest.raises(ValueError, match="n_buckets"):
        popularity_bucket_shares(frame([1]), frame([1]), n_buckets=0)
