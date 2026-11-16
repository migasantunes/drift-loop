"""SeenItems, the caller-side record of what each user has already consumed."""

from __future__ import annotations

import numpy as np
import polars as pl

from driftloop.data.schema import empty_interactions, to_canonical
from driftloop.models.base import SeenItems


def test_built_from_frame_with_duplicates_collapsed() -> None:
    df = to_canonical(
        pl.DataFrame({"user_id": [1, 1, 1, 2], "item_id": [10, 11, 10, 10], "ts": [0, 1, 2, 3]})
    )
    seen = SeenItems.from_interactions(df)
    assert len(seen) == 2
    assert sorted(seen.get(1).tolist()) == [10, 11]
    assert seen.get(2).tolist() == [10]
    assert seen.get(1).dtype == np.int64


def test_unknown_user_returns_empty_array() -> None:
    seen = SeenItems.from_interactions(empty_interactions())
    assert len(seen) == 0
    out = seen.get(42)
    assert out.size == 0
    assert out.dtype == np.int64


def test_accepts_numpy_integer_user_ids() -> None:
    seen = SeenItems({7: np.array([1, 2], dtype=np.int64)})
    assert seen.get(np.uint32(7)).tolist() == [1, 2]  # type: ignore[arg-type]
