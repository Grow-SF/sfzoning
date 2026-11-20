"""Outer and inner folds by sheet."""

from __future__ import annotations

import random

from sfzmaps.frame.data import SheetId
from sfzmaps.frame.folds import INNER_K, OUTER_K, split

SERIES = ("1921", "1960", "2000-digital-zoning")


def _twelve() -> list[SheetId]:
    return [(s, f"sheet-{i}") for s in SERIES for i in range(4)]


def test_constants():
    assert (OUTER_K, INNER_K) == (4, 3)


def test_outer_folds_hold_each_sheet_once():
    sheets = _twelve()
    folds = split(sheets, 4)
    assert len(folds) == 4
    held_all = [h for f in folds for h in f.held]
    assert sorted(held_all) == sorted(sheets)
    for f in folds:
        assert sorted(s for s, _ in f.held) == sorted(SERIES)
        assert not set(f.train) & set(f.held)
        assert set(f.train) | set(f.held) == set(sheets)


def test_inner_folds_of_nine():
    nine = [(s, f"sheet-{i}") for s in SERIES for i in range(3)]
    folds = split(nine, 3)
    held_all = [h for f in folds for h in f.held]
    assert sorted(held_all) == sorted(nine)
    for f in folds:
        assert len(f.held) == 3
        assert len(f.train) == 6
        assert not set(f.train) & set(f.held)


def test_split_is_deterministic():
    sheets = _twelve()
    shuffled = sheets[:]
    random.Random(5).shuffle(shuffled)
    assert split(sheets, 4) == split(shuffled, 4)


def test_small_series_never_trains_on_its_held_sheet():
    sheets = [("a", "1"), ("a", "2"), ("b", "1"), ("b", "2"), ("b", "3")]
    for f in split(sheets, 4):
        assert not set(f.train) & set(f.held)
        assert {s for s, _ in f.held} == {"a", "b"}


def test_series_with_one_sheet_is_refused():
    import pytest

    with pytest.raises(ValueError):
        split([("a", "1"), ("b", "1"), ("b", "2")], 3)
