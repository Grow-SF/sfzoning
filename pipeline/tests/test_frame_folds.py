"""Folds by sheet: a stratified round-robin."""

from __future__ import annotations

import random

import pytest

from sfzmaps.frame.data import SheetId
from sfzmaps.frame.folds import OUTER_K, split

SERIES = ("1921", "1960", "2000-digital-zoning")


def _uneven() -> list[SheetId]:
    """18 sheets as the labels will be: series of 1 to 6 sheets."""
    sizes = {"1921": 5, "1935": 1, "1948": 2, "1960": 3, "1980": 6, "2000": 1}
    return [(s, f"sheet-{i}") for s, n in sizes.items() for i in range(n)]


def test_constants():
    assert OUTER_K == 4


def test_round_robin_in_series_then_sheet_order():
    """Sorted by (series, sheet) and dealt into k folds, counting across series: fold i holds out i, i + k, ..."""
    sheets = [("b", "2"), ("a", "1"), ("c", "1"), ("a", "3"), ("b", "1"), ("a", "2")]
    folds = split(sheets, 4)
    assert [f.held for f in folds] == [(("a", "1"), ("b", "2")), (("a", "2"), ("c", "1")), (("a", "3"),),
                                       (("b", "1"),)]
    assert folds[2].train == (("a", "1"), ("a", "2"), ("b", "1"), ("b", "2"), ("c", "1"))


@pytest.mark.parametrize("k", [2, 3, 4, 7, 18])
def test_every_sheet_held_out_once_and_never_trained_on(k):
    sheets = _uneven()
    folds = split(sheets, k)
    assert len(folds) == k
    held_all = [h for f in folds for h in f.held]
    assert sorted(held_all) == sorted(sheets)
    for f in folds:
        assert f.held
        assert not set(f.train) & set(f.held)
        assert set(f.train) | set(f.held) == set(sheets)


def test_a_small_series_is_spread_across_folds():
    """A series with no more sheets than folds has each sheet held out by a different fold."""
    sheets = _uneven()
    for k in (4, 6):
        folds = split(sheets, k)
        for series in {s for s, _ in sheets}:
            n = sum(s == series for s, _ in sheets)
            if n <= k:
                assert all(sum(s == series for s, _ in f.held) <= 1 for f in folds), series
                assert sum(any(s == series for s, _ in f.held) for f in folds) == n


def test_a_series_of_one_sheet_is_allowed():
    folds = split([("a", "1"), ("b", "1"), ("b", "2")], 3)
    assert [f.held for f in folds] == [(("a", "1"),), (("b", "1"),), (("b", "2"),)]


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
    sheets = _uneven()
    shuffled = sheets[:]
    random.Random(5).shuffle(shuffled)
    assert split(sheets, 4) == split(shuffled, 4) == split(sheets + sheets[:3], 4)


def test_split_refuses_too_few_or_too_many_folds():
    sheets = [("a", "1"), ("b", "1"), ("b", "2")]
    for k in (0, 1, 4):
        with pytest.raises(ValueError):
            split(sheets, k)
