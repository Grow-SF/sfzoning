"""Folds by sheet: every labelled sheet is scored by a model that never trained on it."""

from __future__ import annotations

from dataclasses import dataclass

from sfzmaps.frame.data import SheetId

OUTER_K = 4
INNER_K = 3


@dataclass(frozen=True)
class Fold:
    train: tuple[SheetId, ...]
    held: tuple[SheetId, ...]


def split(sheets: list[SheetId], k: int) -> list[Fold]:
    """Fold i holds out each series' i-th sheet (modulo the series size); the rest train."""
    by_series: dict[str, list[SheetId]] = {}
    for sheet in sorted(set(sheets)):
        by_series.setdefault(sheet[0], []).append(sheet)
    for series, members in by_series.items():
        if len(members) < 2:
            raise ValueError(f"series {series!r} has {len(members)} sheet(s); need at least 2")
    folds = []
    for i in range(k):
        held = {members[i % len(members)] for members in by_series.values()}
        train = tuple(s for s in sorted(set(sheets)) if s not in held)
        folds.append(Fold(train=train, held=tuple(sorted(held))))
    return folds
