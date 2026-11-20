"""Folds by sheet: every labelled sheet is scored by a model that never trained on it."""

from __future__ import annotations

from dataclasses import dataclass

from sfzmaps.frame.data import SheetId

OUTER_K = 4


@dataclass(frozen=True)
class Fold:
    train: tuple[SheetId, ...]
    held: tuple[SheetId, ...]


def split(sheets: list[SheetId], k: int) -> list[Fold]:
    """A stratified round-robin: the sheets sorted by (series, sheet) and dealt into `k` folds in that order, counting
    across series, so fold i holds out sheets i, i + k, i + 2k, ... and trains on the rest.  Each sheet is held out
    exactly once; a series of any size is allowed, and one with no more sheets than `k` has each held out by a
    different fold.  ValueError unless 2 <= k <= the number of sheets."""
    ordered = sorted(set(sheets))
    if not 2 <= k <= len(ordered):
        raise ValueError(f"{k} folds of {len(ordered)} sheet(s): need 2 <= k <= the number of sheets")
    folds = []
    for i in range(k):
        held = ordered[i::k]
        folds.append(Fold(train=tuple(s for s in ordered if s not in held), held=tuple(held)))
    return folds
