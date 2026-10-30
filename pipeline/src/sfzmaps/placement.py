"""Answer-key-free placement check: do the sheet's own labels land where they say?

A block number printed inside a block should map, through the final sheet transform, into
that modern block; a street name printed on a street should map onto that street. On a
correctly placed sheet most readable labels agree (misreads and renumbered blocks are the
exceptions); on a sheet placed one neighbourhood over, or at the wrong scale, almost none
do, however self-consistent the corner fit looks. `label_agreement` measures the share.

The tolerances are generous on purpose (a label's position within its block or along its
street is not exact): BLOCK_BUFFER_FT around the modern block, STREET_TOL_FT from the
street centerline (half a street plus the label's offset from the centre line).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from shapely.geometry import Point

from sfzmaps.reading import Label
from sfzmaps.sources.datasf import Reference, _norm_block, normalize_street
from sfzmaps.transform import SheetTransform

BLOCK_BUFFER_FT = 25.0
STREET_TOL_FT = 60.0


@dataclass
class Agreement:
    fraction: float   # agreeing / checked; NaN when nothing could be checked
    n: int            # labels checked (block numbers with a modern block + known street names)
    n_block: int
    n_street: int


def _blocks(ref: Reference) -> dict:
    out: dict = {}
    for num, g in zip(ref.blocks["block_num"], ref.blocks.geometry):
        if g is not None and not g.is_empty and (str(num) not in out or g.area > out[str(num)].area):
            out[str(num)] = g
    return out


def label_agreement(t: SheetTransform, labels: list[Label], ref: Reference) -> Agreement:
    from sfzmaps.anchors import _street_lines

    blocks = _blocks(ref)
    lines = _street_lines(ref)
    checks = []
    for lb in labels:
        if lb.kind == "block_number" and any(c.isdigit() for c in lb.text):
            g = blocks.get(_norm_block(lb.text))
            if g is not None:
                checks.append((lb, "block", g))
        elif lb.kind == "street_name":
            g = lines.get(normalize_street(lb.text))
            if g is not None:
                checks.append((lb, "street", g))
    if not checks:
        return Agreement(float("nan"), 0, 0, 0)
    w = t.to_world(np.array([[lb.x, lb.y] for lb, _, _ in checks], float))
    ok = []
    for (lb, kind, g), (x, y) in zip(checks, w):
        p = Point(x, y)
        ok.append(g.distance(p) <= (BLOCK_BUFFER_FT if kind == "block" else STREET_TOL_FT))
    nb = sum(k == "block" for _, k, _ in checks)
    return Agreement(float(np.mean(ok)), len(checks), nb, len(checks) - nb)
