"""The frame finder's scores, counted in ink (1 - luminance), on a labelled sheet's decision.

The goal is ink: frame ink left in the map and map ink cut away, not areas.  The frame and the label boxes (the
label's `remove`: title and legend blocks, compass roses, section badges) are scored apart, because the boxes are a
few percent of the ink to delete and one pooled figure hides them (spec amendment 4, after the straight-line test):
- frame ink (class 1 of `data.class_mask`) and how much of it is left in;
- label-box ink (class 2) and how much is left in, and per labelled box the share of its ink deleted;
- map ink (class 0) cut, apart by what cut it: the frame (`outside`) or a fitted box (`removed`).

Pooled shares are summed ink over summed ink (`pooled_split`): a sheet with nothing left in still counts.
"""

from __future__ import annotations

import numpy as np

from sfzmaps.frame import blocks, data


def percent(share: float) -> str:
    """A share as a percentage for the owner (three significant figures), "n/a" for NaN."""
    return "n/a" if share != share else f"{100.0 * share:.3g}%"


def _share(part: float, total: float) -> float:
    return float(part / total) if total > 0 else float("nan")


def split_scores(outside: np.ndarray, removed: np.ndarray, polygons, remove, lum: np.ndarray) -> dict:
    """A sheet's decision (`outside` the frame, `removed` boxes; bool, page-sized) scored against its label (the map
    `polygons` and the `remove` boxes, page px): the ink amounts `frame_ink`, `frame_left`, `box_ink`, `box_left`,
    `map_ink`, `map_cut_frame` (map ink in `outside` and not in `removed`) and `map_cut_boxes` (map ink in `removed`);
    and `boxes`, per labelled box its `kind` and `size` (`blocks.label_kind`: [long, short] px) and the share of its
    ink deleted (NaN for a box with no ink)."""
    outside = np.asarray(outside, bool)
    removed = np.asarray(removed, bool)
    shape = np.shape(lum)
    cls = data.class_mask(polygons, remove, shape)
    ink = 1.0 - np.asarray(lum, np.float64)
    kept = ~(outside | removed)
    frame, box, map_ = cls == 1, cls == 2, cls == 0
    boxes = []
    for poly in remove:
        m = data.map_mask([poly], shape) == 1
        total = float(ink[m].sum())
        kind, a, b = blocks.label_kind(poly)
        boxes.append({"kind": kind, "size": [round(float(a), 1), round(float(b), 1)], "ink": total,
                      "deleted_share": _share(float(ink[m & ~kept].sum()), total)})
    return {
        "frame_ink": float(ink[frame].sum()),
        "frame_left": float(ink[frame & kept].sum()),
        "box_ink": float(ink[box].sum()),
        "box_left": float(ink[box & kept].sum()),
        "boxes": boxes,
        "map_ink": float(ink[map_].sum()),
        "map_cut_frame": float(ink[map_ & outside & ~removed].sum()),
        "map_cut_boxes": float(ink[map_ & removed].sum()),
    }


def pooled_split(rows: list[dict]) -> dict:
    """The four shares of `split_scores` rows, summed ink over summed ink: frame and label-box ink left in, map ink
    cut by the frame and by boxes.  NaN where the rows have no ink of that class (`percent` shows "n/a")."""
    def tot(k):
        return float(sum(r[k] for r in rows))
    return {"frame_left_share": _share(tot("frame_left"), tot("frame_ink")),
            "box_left_share": _share(tot("box_left"), tot("box_ink")),
            "map_cut_frame_share": _share(tot("map_cut_frame"), tot("map_ink")),
            "map_cut_boxes_share": _share(tot("map_cut_boxes"), tot("map_ink"))}
