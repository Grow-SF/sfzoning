"""The printed areas of a sheet: the ground enclosed by what the scan draws as a boundary (design: "read the print,
then the lots", stage 1a).

An area is a 4-connected piece of the frame cut where the boundary probability is at most 1/2 (`areas`: the rule of
`train.blocks`).  A boundary finder is judged against today's blocks placed on the page (`against_blocks`):

  merged     an area that holds more than half of the px of two or more of today's blocks: a boundary between them was
             missed
  shattered  a block none of whose areas holds more than half of its px: a boundary was invented inside it

Both count only px inside the frame cut and off the boundaries.  Today's blocks are a stand-in: where 1921 prints a
street today lacks (or the reverse) the count is wrong in the print's favour, so the pictures (`picture`) come first.

`python -m sfzmaps.solver.hatchnet.areas --series 1921 --sheet <sheet>` compares the street outputs on disk (v1, v2,
v2c and v1 or v2c) on one sheet: table.json and pictures under data/solver/experiments/hatch-cnn/areas/.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from scipy import ndimage

from sfzmaps.solver.hatchnet import data, train

FINDERS = {"v1": "streets", "v2": "streets-v2/streets", "v2c": "streets-v2c/streets"}     # under hatch-cnn/
BOTH = "v1_or_v2c"                      # a boundary where either says street
MERGED_BGR = (60, 60, 230)
TINT = 0.45                             # the share of an area's colour in its px (drawing only)


def areas(boundary_prob: np.ndarray, crop: np.ndarray) -> np.ndarray:
    """The areas (int32: 0 boundary or off the frame cut, 1.. each area)."""
    return train.blocks(boundary_prob, crop)


def _overlap(area: np.ndarray, block: np.ndarray):
    """(pairs (n, 2) of (area, block) that share px, their px counts)."""
    on = (area > 0) & (block > 0)
    key = area[on].astype(np.int64) * (int(block.max()) + 1) + block[on]
    k, n = np.unique(key, return_counts=True)
    return np.stack([k // (int(block.max()) + 1), k % (int(block.max()) + 1)], 1), n


def _tables(area: np.ndarray, block: np.ndarray):
    """Per block seen in an area: its px in areas, its largest single-area px and that area; the pairs and counts."""
    pairs, n = _overlap(area, block)
    nb = int(block.max()) + 1
    in_areas = np.bincount(pairs[:, 1], n, minlength=nb)
    order = np.argsort(n, kind="stable")
    top = np.zeros(nb, np.int64)
    top_area = np.zeros(nb, np.int64)
    top[pairs[order, 1]] = n[order]                  # the largest count is written last
    top_area[pairs[order, 1]] = pairs[order, 0]
    return in_areas, top, top_area


def merged(area: np.ndarray, block: np.ndarray) -> np.ndarray:
    """bool per px: in an area that holds more than half of the px of two or more of today's blocks."""
    in_areas, top, top_area = _tables(area, block)
    held = (in_areas > 0) & (top * 2 > in_areas)                         # blocks mostly in one area
    count = np.bincount(top_area[held], minlength=int(area.max()) + 1)
    return (count >= 2)[area] & (area > 0)


def against_blocks(area: np.ndarray, block: np.ndarray) -> dict:
    """The finder's areas against today's blocks (see the module)."""
    in_areas, top, top_area = _tables(area, block)
    seen = in_areas > 0
    held = seen & (top * 2 > in_areas)
    count = np.bincount(top_area[held], minlength=int(area.max()) + 1)
    is_merged = count >= 2
    sizes = np.bincount(area[area > 0], minlength=int(area.max()) + 1)
    total = int(sizes.sum())
    pairs, _ = _overlap(area, block)
    return {"areas": int(len(np.unique(pairs[:, 0]))), "blocks": int(seen.sum()),
            "merged_areas": int(is_merged.sum()),
            "merged_px_share": float(sizes[is_merged].sum() / total) if total else 0.0,
            "blocks_in_merged": int((held & is_merged[top_area]).sum()),
            "shattered_blocks": int((seen & ~held).sum()),
            "largest_area_px_share": float(sizes.max() / total) if total else 0.0}


def picture(lum: np.ndarray, area: np.ndarray, is_merged: np.ndarray) -> np.ndarray:
    """BGR uint8: the scan in grey; each area tinted its own colour, merged areas red; boundaries untinted."""
    grey = (np.clip(lum, 0, 1) * 255).astype(np.float32)
    img = np.repeat(grey[..., None], 3, axis=2)
    rng = np.random.default_rng(0)
    colours = rng.integers(70, 236, (int(area.max()) + 1, 3)).astype(np.float32)
    on = area > 0
    tint = colours[area[on]]
    tint[is_merged[on]] = MERGED_BGR
    img[on] = (1 - TINT) * img[on] + TINT * tint
    return img.astype(np.uint8)


WINDOWS = {"6220002": (2800, 4750, 2560, 3650)}       # rows r0-r1, cols c0-c1 on Section 12 (20180102100525362)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--series", default="1921")
    ap.add_argument("--sheet", required=True)
    a = ap.parse_args(argv)
    root = data.EXPERIMENT
    out = root / "areas"
    out.mkdir(parents=True, exist_ok=True)
    r = data.read_rasters(data.data_dir(a.series) / f"{a.sheet}.npz")
    block, _ = data.block_ids(r.block == 1)
    probs = {}
    for name, sub in FINDERS.items():
        with np.load(Path(root) / sub / f"{a.sheet}.npz") as z:
            probs[name] = z["street_prob"]
    probs[BOTH] = np.maximum(probs["v1"], probs["v2c"])
    table = {}
    for name, p in probs.items():
        ar = areas(p, r.crop)
        table[name] = against_blocks(ar, block)
        img = picture(r.ink, ar, merged(ar, block))
        cv2.imwrite(str(out / f"{name}_{a.sheet}.png"), img)
        for w, (r0, r1, c0, c1) in WINDOWS.items():
            cv2.imwrite(str(out / f"{name}_{a.sheet}_{w}.png"), img[r0:r1, c0:c1])
    (out / "table.json").write_text(json.dumps(table, indent=1) + "\n")
    print(json.dumps(table, indent=1))


if __name__ == "__main__":
    main()
