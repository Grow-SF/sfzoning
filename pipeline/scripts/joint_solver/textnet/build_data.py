"""Training data for our own text detector, on the merged city page (k8/MALLP_c1/raw.png: the raw scan stitched onto the
4-ft city grid, nothing erased).

  ink      page < 128
  text     1 inside a terra text box (every kind), as a rotated rectangle mapped from its sheet onto the merged page
  kind     0 none, 1 zoning code, 2 block number, 3 street name, 4 other
  angle    reading direction in degrees on the merged page (counter-clockwise from left-to-right), inside text only
  unknown  inside another pass's box or near a label-read point, but in no terra box: text terra may have missed, so it is
           taught neither as text nor as not-text
  quarter  0 nw, 1 ne, 2 sw, 3 se (the joint solver's split), per pixel
Writes textnet/data.npz.   usage: build_data.py"""
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import shapely

S = Path("/private/tmp/claude-501/-Users-sbuss-workspace-sfzoning/eb18d2fc-e22a-48a6-afbc-57c31cd3e134/scratchpad")
os.environ.setdefault("SFZ_RND", str(S))
os.environ.setdefault("SFZMAPS_DATA_DIR", str(S / "data_ay2"))
sys.path.insert(0, str(S / "pkg"))
from joint_solver.data import FT_PER_PX, city  # noqa: E402
from sfzmaps import config  # noqa: E402
from sfzmaps.transform import SheetTransform  # noqa: E402

SERIES = "2000-digital-zoning"
KINDS = {"zoning code": 1, "block number": 2, "street name": 3, "other": 4}
OTHER_PASSES = ["textboxes2_cw", "textboxes2_cwpa", "textboxes2_cwpl", "textboxes2_cwx", "textboxes2_pa", "textboxes2_pb",
                "textboxes2_pd", "textboxes2_pl"]
READ_POINT_RADIUS = 30            # px around a label-read point treated as unknown
TYPICAL_HEIGHT = 14.0             # px, text height used when a slanted box's own shape cannot give it


def word_rect(box, angle_deg):
    """the rotated rectangle (page px corners, reading direction) of a word from its axis box and reading angle"""
    x0, y0, x1, y1 = box
    W, H = x1 - x0, y1 - y0
    a = np.deg2rad(angle_deg)
    c, s = abs(np.cos(a)), abs(np.sin(a))
    det = c * c - s * s
    if abs(det) > 0.25:
        L, T = (c * W - s * H) / det, (c * H - s * W) / det
    else:
        T = TYPICAL_HEIGHT
        L = (W - T * s) / c if c >= s else (H - T * c) / s
    L, T = max(L, 2.0), max(T, 2.0)
    v = np.array([np.cos(a), -np.sin(a)])                         # reading direction, image coords (y down)
    n = np.array([-v[1], v[0]])
    ctr = np.array([(x0 + x1) / 2, (y0 + y1) / 2])
    return np.array([ctr - v * L / 2 - n * T / 2, ctr + v * L / 2 - n * T / 2, ctr + v * L / 2 + n * T / 2,
                     ctr - v * L / 2 + n * T / 2])


def main():
    C = city()
    page = cv2.imread(str(S / "k8" / "MALLP_c1" / "raw.png"), 0)
    PH, PW = page.shape
    _, row0, col0 = C.scan()

    def to_merged(t, pts):
        col, row = C.to_grid(t.to_world(np.asarray(pts, float))).T
        return np.c_[col - col0, row0 - row].astype(np.float64)

    text = np.zeros((PH, PW), np.uint8)
    kind = np.zeros((PH, PW), np.uint8)
    angle = np.full((PH, PW), -999, np.int16)
    other = np.zeros((PH, PW), np.uint8)
    n_words = n_off = 0
    for nn in range(1, 14):
        sheet = f"{SERIES}-p{nn:02d}"
        t = SheetTransform.from_json(json.loads((config.WORK_DIR / SERIES / sheet / "transform.json").read_text()))
        drawn = shapely.from_geojson((config.WORK_DIR / SERIES / sheet / "warped.footprint.geojson").read_text())

        def on_map(box):
            x0, y0, x1, y1 = box
            return bool(shapely.contains(drawn, shapely.Point(t.to_world(np.array([[(x0 + x1) / 2, (y0 + y1) / 2]]))[0])))

        for b in json.load(open(S / "k8" / f"p{nn:02d}" / "textboxes_gpt-6.1-terra.json")):
            if not on_map(b["box"]):
                n_off += 1
                continue
            a = float(b.get("angle_deg") or 0.0)
            q = to_merged(t, word_rect(b["box"], a))
            dv = q[1] - q[0]                                      # reading direction on the merged page
            am = int(round(np.rad2deg(np.arctan2(-dv[1], dv[0]))))
            poly = np.round(q).astype(np.int32).reshape(-1, 1, 2)
            cv2.fillPoly(text, [poly], 1)
            cv2.fillPoly(kind, [poly], KINDS.get(b["kind"], 4))
            cv2.fillPoly(angle, [poly], am)
            n_words += 1
        for p in OTHER_PASSES:
            f = S / "k8" / f"p{nn:02d}" / f"{p}.json"
            if f.exists():
                for b in json.load(open(f)):
                    if not on_map(b["box"]):
                        continue
                    q = to_merged(t, word_rect(b["box"], float(b.get("angle_deg") or 0.0)))
                    cv2.fillPoly(other, [np.round(q).astype(np.int32).reshape(-1, 1, 2)], 1)
        f = config.WORK_DIR / SERIES / sheet / "district_labels.json"
        if f.exists():
            pts = [(lb["x"], lb["y"]) for lb in json.load(open(f))["labels"] if on_map((lb["x"], lb["y"], lb["x"], lb["y"]))]
            if pts:
                for x, y in to_merged(t, pts):
                    cv2.circle(other, (int(round(x)), int(round(y))), READ_POINT_RADIUS, 1, -1)
    unknown = (other > 0) & (text == 0)
    # quarter per pixel: east/north split at the scored lots' median centroid (separable in columns and rows)
    xmed, ymed = np.median(C.centroid[C.scored], 0)
    xs = C.origin[0] + (np.arange(PW) + col0) * FT_PER_PX
    ys = C.origin[1] + (row0 - np.arange(PH)) * FT_PER_PX
    east = xs >= xmed
    north = ys >= ymed
    quarter = np.where(north[:, None], np.where(east[None, :], 1, 0), np.where(east[None, :], 3, 2)).astype(np.uint8)
    out = S / "textnet" / "data.npz"
    np.savez_compressed(out, ink=page < 128, text=text, kind=kind, angle=angle, unknown=unknown, quarter=quarter)
    print(f"words {n_words:,} (off the map area, left out: {n_off:,}); text px {int(text.sum()):,} ({text.mean():.2%}); unknown px {int(unknown.sum()):,}; "
          f"ink px {int((page < 128).sum()):,}; ink inside text {(page < 128)[text > 0].mean():.1%}; saved {out}")


if __name__ == "__main__":
    main()
