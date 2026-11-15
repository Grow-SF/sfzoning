"""Re-derive the text detector's validation area (`sfzmaps.solver.textnet.data.VALIDATION_AREA`) from the data.

    uv run python scripts/choose_text_validation_area.py

Reads, for every sheet Terra read of the training maps (`paths.TEXT_TRAIN_SERIES`, `paths.TEXT_SHEETS`): its warped
footprint and corrected placement (data/warp/<series>/<sheet>/warped.footprint.geojson, transform.json, as the
warp stages make them) and Terra's boxes (`textnet.boxes.terra_boxes`, from the saved answers in git).  CPU only, no
model, no paid read; under a minute.

The rule:
  * the mapped area is the union of the training maps' sheets' warped footprints; the candidates are rectangles of
    AREA_SHARE of it, lying inside every training map's sheets (at least INSIDE of each rectangle inside the
    intersection of the maps' unions), with corners every GRID_FT from the intersection's south-west corner and
    width : height in ASPECTS;
  * each candidate's score: for each training map, its share of the map's Terra box area (each box's axis area in page
    px, placed by its centre's map point), in all and of each kind the map prints (KIND_MIN of its boxes or more);
    score = max(the largest deviation of a map's share from AREA_SHARE, half the largest deviation of a kind's);
  * candidates ranked by score, a candidate overlapping a better one by more than DISTINCT of its area dropped.

The code uses the rule's second candidate, not its first.  The first is the residential south-west (Parkside,
Lakeshore, Ingleside): sparse, simple lettering, an area too easy to tell training recipes apart, and the scores are
to be taken on dense areas, never only the simple west.  The second is the central strip through the Mission, Noe
Valley and Glen Park.  The script prints the distinct candidates, each map's shares for the first few, and which one
the code's VALIDATION_AREA is.
"""
import json

import numpy as np
import shapely

from sfzmaps import config
from sfzmaps.solver import paths
from sfzmaps.solver.textnet import boxes, data
from sfzmaps.transform import SheetTransform

AREA_SHARE = 0.10        # the rectangle's share of the mapped area, and the share of each map's text it should hold
INSIDE = 0.995           # at least this share of a rectangle inside every training map's sheets
GRID_FT = 250.0          # the corners' grid
ASPECTS = (0.5, 0.75, 1.0, 1.33, 2.0)     # width / height
KIND_MIN = 0.05          # a kind is scored on a map that has at least this share of its boxes of that kind
DISTINCT = 0.3           # a candidate overlapping a better one by more than this share of its area is the same place
SHOW = 8                 # distinct candidates printed
CHOSEN_RANK = 2          # the candidate the code uses (see the module)
SQ_MI = 5280.0 ** 2
KINDS = list(data.KINDS)


def read_maps() -> tuple[dict, dict]:
    """Each training map's sheets' union (shapely, EPSG:2227 ft) and its Terra boxes as rows (x, y, area, kind)."""
    records = boxes.terra_records()
    by_sheet: dict[tuple[str, str], list[dict]] = {}
    for r in records:
        by_sheet.setdefault((r["series"], r["sheet"]), []).append(r)
    cover, found = {}, {}
    for series in paths.TEXT_TRAIN_SERIES:
        polys, rows = [], []
        for sheet in paths.TEXT_SHEETS[series]:
            folder = config.DATA_DIR / "warp" / series / sheet
            polys.append(shapely.from_geojson((folder / "warped.footprint.geojson").read_text()))
            transform = SheetTransform.from_json(json.loads((folder / "transform.json").read_text()), folder=folder)
            got = boxes.terra_boxes(series, sheet, by_sheet.get((series, sheet), []))
            if not got:
                continue
            b = np.array([x["box"] for x in got], float)
            centre = np.c_[(b[:, 0] + b[:, 2]) / 2, (b[:, 1] + b[:, 3]) / 2]
            rows.append(np.c_[transform.to_world(centre), (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1]),
                              [KINDS.index(x["kind"]) for x in got]])
        cover[series] = shapely.union_all(polys)
        found[series] = np.concatenate(rows)
    return cover, found


def shares(rect, found: dict) -> dict:
    """Each map's share of its Terra box area inside `rect` (x0, y0, x1, y1): in all, and of each kind."""
    x0, y0, x1, y1 = rect
    out = {}
    for series, c in found.items():
        m = (c[:, 0] >= x0) & (c[:, 0] < x1) & (c[:, 1] >= y0) & (c[:, 1] < y1)
        out[series] = (c[m, 2].sum() / c[:, 2].sum(),
                       [c[m & (c[:, 3] == k), 2].sum() / max(c[c[:, 3] == k, 2].sum(), 1) for k in range(4)])
    return out


def main() -> None:
    cover, found = read_maps()
    union = shapely.union_all(list(cover.values()))
    inter = shapely.intersection_all(list(cover.values()))
    target = AREA_SHARE * union.area
    print(f"training maps' sheets: union {union.area / SQ_MI:.1f} sq mi, intersection {inter.area / SQ_MI:.1f} sq mi; "
          f"rectangles of {target / SQ_MI:.2f} sq mi")
    counts = {s: np.bincount(c[:, 3].astype(int), minlength=4) for s, c in found.items()}
    bx0, by0, bx1, by1 = inter.bounds
    results = []
    for aspect in ASPECTS:
        w = np.sqrt(target * aspect)
        h = target / w
        for x0 in np.arange(bx0, bx1 - w, GRID_FT):
            for y0 in np.arange(by0, by1 - h, GRID_FT):
                rect = (x0, y0, x0 + w, y0 + h)
                box = shapely.box(*rect)
                if shapely.intersection(box, inter).area < INSIDE * box.area:
                    continue
                sh = shares(rect, found)
                dev = max(abs(a - AREA_SHARE) for a, _ in sh.values())
                kdev = max(abs(k - AREA_SHARE) for s, (_, ks) in sh.items() for i, k in enumerate(ks)
                           if counts[s][i] >= KIND_MIN * counts[s].sum())
                results.append((max(dev, kdev / 2), dev, kdev, rect))
    results.sort(key=lambda t: t[0])
    print(f"{len(results)} candidates inside every training map; the distinct ones, best first:")
    picked = []
    for score, dev, kdev, rect in results:
        box = shapely.box(*rect)
        if any(shapely.intersection(box, p).area > DISTINCT * box.area for _, p in picked):
            continue
        picked.append((rect, box))
        n = len(picked)
        print(f"#{n} score {score:.3f} (all {dev:.3f}, kinds {kdev:.3f}): "
              f"({', '.join(f'{v:.0f}' for v in rect)})")
        if n <= 3:
            print(f"    {'map':22s} all    {'  '.join(k[:6] for k in KINDS)}")
            for s, (a, ks) in shares(rect, found).items():
                print(f"    {s:22s} {a:.3f}  {'   '.join(f'{k:.3f}' for k in ks)}")
        if n == SHOW:
            break
    chosen = np.round(picked[CHOSEN_RANK - 1][0])
    code = np.array(data.VALIDATION_AREA)
    print(f"chosen: #{CHOSEN_RANK} ({', '.join(f'{v:.0f}' for v in chosen)}); the code's VALIDATION_AREA "
          f"({', '.join(f'{v:.0f}' for v in code)}) {'is it' if np.abs(chosen - code).max() <= 1 else 'DIFFERS'}")


if __name__ == "__main__":
    main()
