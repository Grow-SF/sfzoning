"""Trial data roots for placement (`sfzmaps find trial`): the finder's blocks are tried in place of the traced blocks in
a separate data folder, so that today's placements can never be touched.

A trial `T` has two data roots, `arm_root(T, arm)` = data/find/trial/<T>/<arm>/, one per arm: `traced` (today's
blocks, run again, to show the trial reproduces today) and `finder` (the finder's blocks).  `prepare` makes a root:
`raw`, `ref` and `cache` are links to today's, and each trial sheet's work files up to and including its blocks
(`KEPT`, with their freshness records and saved answers) are copied, never linked, into the root's work/<series>/
<sheet>/.  Nothing made after the blocks is copied, so the placement steps run in the root (SFZMAPS_DATA_DIR=<root>)
make it again from the root's blocks.  A sheet whose placement does not start from its blocks (one with a review or a
seed) does not test the blocks and is refused, as is a duplicate scan.

`numbers` reads a sheet's numbers from a root's own work/ and warp/ files (blocks, good corners, control points, the
placement check and the warp's choice); today's numbers are `numbers(config.DATA_DIR, sid)`, the same function for
every root.  `reproduced` says, file by file, whether a sheet came out the same in two roots, with no tolerance.
`trained_ground` measures how much of a sheet's ground the run was trained on: the share under the run's lot-labelled
sheets and the run's hand crops on it.

The comparison (`sfzmaps find trial compare`): `check_blocks` refuses a finder blocks file traced again or changed
since `find blocks` wrote it; `reach_ft` says how far the sheet's block-side points (`side_points`, the points the
warp scores) lie from the nearest control point; `share_on` the share of control points on the run's lot-labelled
ground (`trained_union`); `places` picks three boxes from today's placement (the most points off the print, the most
points, the middle) and `placed_picture` shows each arm's warped scan there side by side, today's block lines on top.

These read placement outputs only to report them; `labels.json` is copied as a file and never opened.  Writes only
inside the trial root it is given (and `TRIAL/<T>/numbers.json`, `TRIAL/<T>/compare.json` and `TRIAL/<T>/look/`,
written by the commands)."""
from __future__ import annotations

import functools
import json
import shutil
import tempfile
from pathlib import Path

import numpy as np
import shapely

from sfzmaps import config, inputs
from sfzmaps.find import pages
from sfzmaps.find.pages import SheetId
from sfzmaps.solver import hatching

TRIAL = pages.OUT / "trial"
ARMS = ("traced", "finder")
# The work files a sheet's placement starts from: everything up to and including its blocks, and the paper warp
# (`sfzmaps warp` reads it and copies it when a sheet is left uncorrected).
KEPT = ("page.png", "page.json", "labels.json", "blocks_px.geojson", "warped_paper.tif",
        "warped_paper.footprint.geojson")
SIDECARS = (".inputs.json", ".answers")   # a file's freshness record and its saved answers, copied beside it
LINKED = ("raw", "ref", "cache")
TODAY = ("work", "warp", "out")           # today's placements: a trial root never lies in them


def arm_root(trial: str, arm: str) -> Path:
    """The data root of one arm of a trial: TRIAL/<trial>/<arm>/."""
    if arm not in ARMS:
        raise ValueError(f"{arm!r}: an arm is one of {', '.join(ARMS)}")
    if not trial or trial in (".", "..") or "/" in trial or "\\" in trial:
        raise ValueError(f"{trial!r}: a trial's name is one folder name (no path separator, no '..')")
    return TRIAL / trial / arm


def _inside(p: Path, folder: Path) -> bool:
    return p == folder or folder in p.parents


def _refusals(root: Path, sids: list[SheetId], source: Path) -> list[str]:
    """Why `prepare` would refuse, one line each; empty when it would not."""
    why = []
    r, s = root.resolve(), source.resolve()
    if r == s:
        why.append(f"{root}: is today's data root; a trial root is a separate folder")
    for x in TODAY:
        if _inside(r, (s / x).resolve()):
            why.append(f"{root}: lies inside today's data ({s / x}); a trial root is a separate folder")
    for x in LINKED:
        if not (source / x).exists():
            why.append(f"{source / x}: missing; a trial root links today's {x}/")
    why += _root_entries(root, source)
    dups: dict[str, dict] = {}
    for series, sheet in sids:
        name = f"{series}/{sheet}"
        dups.setdefault(series, hatching.duplicate_scans(series))
        if sheet in dups[series]:
            why.append(f"{name}: a duplicate scan of {dups[series][sheet]}, which is never read")
        if inputs.review_path(series, sheet).is_file():
            why.append(f"{name}: has a review ({inputs.review_path(series, sheet)}), so its placement does not start "
                       f"from its blocks")
        if inputs.seed_path(series, sheet).is_file():
            why.append(f"{name}: has a seed ({inputs.seed_path(series, sheet)}), so its placement does not start "
                       f"from its blocks")
        folder = source / "work" / series / sheet
        missing = [f for f in KEPT if not (folder / f).is_file()]
        if missing:
            why.append(f"{name}: no {', '.join(missing)} in {folder}")
    return why


def _root_entries(root: Path, source: Path) -> list[str]:
    """Why what an existing `root` already holds stops `prepare`: the root itself a link; an entry that is a link or
    leads into today's work/, warp/ or out/, other than `raw`, `ref` and `cache` linked to today's; or any other entry
    but `work` (a root is made whole by `prepare`; `work` is the prepared-once case)."""
    if root.is_symlink():
        return [f"{root}: is a link; a trial root is a folder of its own"]
    if not root.is_dir():
        return []
    s = source.resolve()
    why = []
    for e in sorted(root.iterdir()):
        if e.name in LINKED and e.is_symlink() and e.resolve() == (source / e.name).resolve():
            continue
        if e.is_symlink() or any(_inside(e.resolve(), (s / x).resolve()) for x in TODAY):
            why.append(f"{e}: a link or a path into today's data; a trial root holds only raw, ref and cache linked "
                       f"to today's")
        elif e.name != "work":
            why.append(f"{e}: not made by `prepare`; delete {root} to make the root again")
    return why


def _unshared(made: list[Path], root: Path, source: Path) -> list[str]:
    """The files under root/work/ that are not regular files of their own (a link, or the same file as today's)."""
    bad = []
    for f in made:
        twin = source / f.relative_to(root)
        if f.is_symlink() or not f.is_file() or (f.stat().st_ino, f.stat().st_dev) == \
                (twin.stat().st_ino, twin.stat().st_dev):
            bad.append(str(f))
    return bad


def prepare(root: Path, sids: list[SheetId], source: Path) -> list[Path]:
    """Make the trial data root `root` from today's data root `source`: `raw`, `ref` and `cache` linked to today's,
    and each sheet's `KEPT` files with their sidecars (`SIDECARS`, where there are any) copied into
    root/work/<series>/<sheet>/.  Returns the files copied.  ValueError, naming why and writing nothing, when `root` is
    `source` or lies in its work/, warp/ or out/, already holds a link or a path into today's data (other than `raw`,
    `ref` and `cache` linked to today's) or anything else `prepare` did not make, or a sheet is a duplicate scan, has
    a review or a seed, or lacks a `KEPT` file.  FileExistsError, writing nothing, when root/work exists: a root is
    prepared once (delete the arm's folder to redo it).

    The root is made whole in a folder beside it and renamed into place only when every file is copied; a failed
    prepare removes that folder and leaves `root` as it was.  After the rename every copied file is checked to be a
    regular file of its own (not a link, not today's file); if one is not, the root is removed and ValueError says
    which."""
    root, source = Path(root), Path(source)
    why = _refusals(root, sids, source)
    if why:
        raise ValueError("no trial root made:\n" + "\n".join(why))
    if (root / "work").exists():
        raise FileExistsError(f"{root / 'work'} exists: a trial root is prepared once; delete {root} to redo it")
    root.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix=f".{root.name}.making-", dir=root.parent))
    try:
        for x in LINKED:
            (tmp / x).symlink_to((source / x).resolve(), target_is_directory=True)
        for series, sheet in sids:
            src, dst = source / "work" / series / sheet, tmp / "work" / series / sheet
            dst.mkdir(parents=True, exist_ok=True)
            for f in KEPT:
                for name in (f, *(f + s for s in SIDECARS)):
                    if (src / name).is_file():
                        shutil.copy2(src / name, dst / name)
        if root.is_dir():                       # holds nothing but raw, ref and cache linked to today's (`_refusals`)
            for x in LINKED:
                (root / x).unlink(missing_ok=True)
            root.rmdir()
        tmp.rename(root)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    made = sorted(f for f in (root / "work").rglob("*") if f.is_file() or f.is_symlink())
    bad = _unshared(made, root, source)
    if bad:
        shutil.rmtree(root)
        raise ValueError("trial root removed: not copies of their own:\n" + "\n".join(bad))
    return made


def _json(f: Path):
    """The file's JSON, or None when there is no file."""
    return json.loads(f.read_text()) if f.is_file() else None


def numbers(root: Path, sid: SheetId) -> dict:
    """A sheet's numbers, read only from root/work/<series>/<sheet>/ and root/warp/<series>/<sheet>/: `blocks` (the
    blocks file's features), `good_corners` (its corners), `control_points` (the enabled control points of
    transform.json), `passed`, `reason`, `holdout_p50_ft` and `holdout_p95_ft` (score.json), `warp` (the warp's
    chosen correction, or "uncorrected: <why>") and `within1` (the chosen correction's share of block-side points
    within 1 px of a printed line; None when uncorrected or not measured).  A number whose file is not there is
    None."""
    work, warp = Path(root) / "work" / sid[0] / sid[1], Path(root) / "warp" / sid[0] / sid[1]
    blocks, transform = _json(work / "blocks_px.geojson"), _json(work / "transform.json")
    score, scores = _json(work / "score.json") or {}, _json(warp / "scores.json")
    if scores is None:
        chosen = within1 = None
    elif "uncorrected" in scores:
        chosen, within1 = f"uncorrected: {scores['uncorrected']}", None
    else:
        chosen = scores.get("chosen")
        within1 = scores.get(f"within1_{chosen}")
    return {"blocks": None if blocks is None else len(blocks["features"]),
            "good_corners": None if blocks is None else len(blocks.get("corners") or []),
            "control_points": None if transform is None else sum(bool(g.get("enabled")) for g in transform["gcps"]),
            "passed": score.get("passed"), "reason": score.get("reason"),
            "holdout_p50_ft": score.get("holdout_p50_ft"), "holdout_p95_ft": score.get("holdout_p95_ft"),
            "warp": chosen, "within1": within1}


def row(sid: SheetId, label: str, n: dict) -> str:
    """One printed line of a sheet's numbers in one root."""
    def ft(v):
        return "-" if v is None else f"{v:.2f}"
    w1 = "-" if n["within1"] is None else f"{n['within1']:.3f}"
    check = "-" if n["passed"] is None else ("passed" if n["passed"] else f"failed: {n['reason']}")
    return (f"{sid[0]}/{sid[1]} {label}: {n['blocks']} blocks, {n['good_corners']} good corners, "
            f"{n['control_points']} control points; {check}, held-out p50 {ft(n['holdout_p50_ft'])} / p95 "
            f"{ft(n['holdout_p95_ft'])} ft; warp {n['warp']}, within 1 px {w1}")


def _same(a, b) -> bool:
    """Exactly equal JSON values (NaN equal to NaN; float text round-trips exactly)."""
    return a is not None and b is not None and json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)


def _arrays(f: Path) -> dict | None:
    if not f.is_file():
        return None
    with np.load(f) as z:
        return {k: z[k] for k in z.files}


def _same_arrays(a: dict | None, b: dict | None) -> bool:
    return (a is not None and b is not None and a.keys() == b.keys()
            and all(a[k].dtype == b[k].dtype and a[k].shape == b[k].shape and a[k].tobytes() == b[k].tobytes()
                    for k in a))


def reproduced(a: Path, b: Path, sid: SheetId) -> dict[str, bool]:
    """Whether the sheet came out exactly the same in roots `a` and `b` (no tolerance), file by file: `transform` (the
    control points of work/transform.json), `refine` (the matches of refine.json), `score` (score.json),
    `warp_scores` (warp/scores.json, its `device` aside) and `correction` (every array of correction.npz; when
    both warps are uncorrected there is none, and that is the same).  A file missing from either root is not the
    same."""
    def work(r, f):
        return _json(Path(r) / "work" / sid[0] / sid[1] / f)

    def warp(r):
        return Path(r) / "warp" / sid[0] / sid[1]
    ta, tb = work(a, "transform.json"), work(b, "transform.json")
    ra, rb = work(a, "refine.json"), work(b, "refine.json")
    sa, sb = _json(warp(a) / "scores.json"), _json(warp(b) / "scores.json")
    ca, cb = _arrays(warp(a) / "correction.npz"), _arrays(warp(b) / "correction.npz")
    both_uncorrected = sa is not None and sb is not None and "uncorrected" in sa and "uncorrected" in sb
    return {"transform": _same(ta and ta["gcps"], tb and tb["gcps"]),
            "refine": _same(ra and ra["matches"], rb and rb["matches"]),
            "score": _same(work(a, "score.json"), work(b, "score.json")),
            "warp_scores": _same(sa and {k: v for k, v in sa.items() if k != "device"},
                                 sb and {k: v for k, v in sb.items() if k != "device"}),
            "correction": (ca is None and cb is None) if both_uncorrected else _same_arrays(ca, cb)}


def _footprint(source: Path, name: str):
    f = Path(source) / "warp" / name / "warped.footprint.geojson"
    if not f.is_file():
        raise FileNotFoundError(f"{f}: no warped footprint")
    return shapely.from_geojson(f.read_text())


def _lot_labelled(run_folder: Path, source: Path) -> tuple[list[str], dict]:
    """The run's training items (its log.json), and the warped footprint of each of its lot-labelled sheets (the
    items with no `__` in their name) under `source`.  FileNotFoundError when a footprint is missing."""
    items = [it["name"] for it in json.loads((Path(run_folder) / "log.json").read_text())["items"]]
    return items, {name: _footprint(source, name) for name in items if "__" not in name}


def trained_union(run_folder: Path, source: Path):
    """The ground the run was trained on with lot labels: the union of its lot-labelled sheets' warped footprints
    (map ft); None when it has none.  FileNotFoundError when a footprint is missing."""
    lot_labelled = _lot_labelled(run_folder, source)[1]
    return shapely.union_all(list(lot_labelled.values())) if lot_labelled else None


def share_on(points_world: np.ndarray, ground) -> float | None:
    """The share of the points (map ft) that lie on `ground` (a shapely geometry, or None for no ground: 0); None
    when there are no points."""
    pts = np.asarray(points_world, float).reshape(-1, 2)
    if not len(pts):
        return None
    if ground is None:
        return 0.0
    return float(shapely.intersects_xy(ground, pts[:, 0], pts[:, 1]).mean())


def trained_ground(sid: SheetId, run_folder: Path, source: Path) -> dict:
    """How much of the sheet's ground the run was trained on.  `lot_labelled_share`: the share of the sheet's warped
    footprint (source/warp/<series>/<sheet>/warped.footprint.geojson) that lies under the union of the warped
    footprints of the run's lot-labelled sheets (the items of its log.json with no `__` in their name); `by_sheet`:
    each such sheet's own share, where it is above 0; `hand_crops`: the run's training items on this sheet (its hand
    crops).  FileNotFoundError when a footprint is missing."""
    sheet = _footprint(source, f"{sid[0]}/{sid[1]}")
    items, lot_labelled = _lot_labelled(run_folder, source)
    area = sheet.area
    under = shapely.union_all(list(lot_labelled.values())) if lot_labelled else None
    share = 0.0 if under is None or area == 0 else sheet.intersection(under).area / area
    by_sheet = {}
    for name, fp in lot_labelled.items():
        s = sheet.intersection(fp).area / area if area else 0.0
        if s > 0:
            by_sheet[name] = s
    crops = [name for name in items if "__" in name and name.split("__")[0].split("/")[-2:] == [sid[0], sid[1]]]
    return {"lot_labelled_share": share, "by_sheet": by_sheet, "hand_crops": crops}


def trial_sheets(trial: str) -> list[SheetId]:
    """The sheets in any arm root of the trial (the folders under its work/<series>/)."""
    found = set()
    for arm in ARMS:
        work = arm_root(trial, arm) / "work"
        if work.is_dir():
            found |= {(s.name, h.name) for s in work.iterdir() if s.is_dir() for h in s.iterdir() if h.is_dir()}
    return sorted(found)


# The comparison (`sfzmaps find trial compare`): how far the block-side points lie from the nearest control point,
# three places on each sheet, and the scan warped onto today's map there by each arm's placement.  These report;
# nothing here decides or gates anything.
PLACES = ("worst", "busiest", "centre")
WINDOW_PX = 1200            # picture setting: a place's side, in px of today's warped scan (two panels and the gap
                            # stay under the owner's viewer's 8,000 px limit)
LINE = (255, 0, 255)        # picture setting: today's block lines, magenta (BGR): bright on white paper and on black ink
LINE_PX = 3                 # picture setting: the width today's block lines are drawn, px (a picture of the line,
                            # centred on it, not its exact width)
MAX_SIDE = 8000             # the largest picture side the owner's viewer opens, px: a larger picture is refused
RESULT = "RESULT:"


def reach_ft(points_world: np.ndarray, gcps_world: np.ndarray) -> dict:
    """How far each point lies from its nearest control point (map ft): `p50` and `p90` of those distances; both None
    when there is no control point or no point."""
    from scipy.spatial import cKDTree
    pts = np.asarray(points_world, float).reshape(-1, 2)
    gcps = np.asarray(gcps_world, float).reshape(-1, 2)
    if not len(pts) or not len(gcps):
        return {"p50": None, "p90": None}
    d = cKDTree(gcps).query(pts)[0]
    return {"p50": float(np.percentile(d, 50)), "p90": float(np.percentile(d, 90))}


def control_points(root: Path, sid: SheetId) -> np.ndarray:
    """The enabled control points of root/work/<series>/<sheet>/transform.json, as map ft (`wx`, `wy`); none when
    there is no file."""
    t = _json(Path(root) / "work" / sid[0] / sid[1] / "transform.json")
    g = [(c["wx"], c["wy"]) for c in (t or {}).get("gcps", []) if c.get("enabled")]
    return np.array(g, float).reshape(-1, 2)


def _sheet(sid: SheetId, source: Path):
    """The warp's view of the sheet (`sheetwarp.sheet.Sheet`), reading its work files from `source`/work."""
    from dataclasses import dataclass

    from sfzmaps.sheetwarp.sheet import Sheet

    @dataclass
    class SheetIn(Sheet):
        root: Path = Path()

        @property
        def folder(self):
            return Path(self.root) / "work" / self.series / self.name
    return SheetIn(sid[0], sid[1], Path(source))


@functools.lru_cache(maxsize=8)
def _block_sides(sid: SheetId, source: Path):
    """`sheetwarp.targets.block_side_points` of the sheet in data root `source` (kept, it is slow to make)."""
    from sfzmaps.sheetwarp import targets
    return targets.block_side_points(_sheet(sid, source))


def side_points(sid: SheetId) -> np.ndarray:
    """The block-side points of the sheet under today's data root (map ft): a point every 2 ft along every side of
    every block of the year's lots inside the sheet's drawn map, the points the warp's within-1-px score uses."""
    return _block_sides(tuple(sid), Path(config.DATA_DIR)).pts


@functools.lru_cache(maxsize=8)
def off_print(sid: SheetId, source: Path) -> np.ndarray:
    """Per block-side point of the sheet (`_block_sides`), whether it lies more than 1 px from a printed block line
    running its way under the placement in data root `source`, corrected by its warp's correction where it has one
    (source/warp/<series>/<sheet>/correction.npz, named in that folder's transform.json): the warp's own measure
    (`targets.measure`).  A point off the measured crop counts as off the print, as in the warp's score."""
    from sfzmaps.sheetwarp import targets
    from sfzmaps.transform import Correction
    sheet = _sheet(sid, source)
    warp = Path(source) / "warp" / sid[0] / sid[1]
    t = _json(warp / "transform.json") or {}
    correction = Correction.load(warp / t["correction"]["file"]) if "correction" in t else None
    d = targets.measure(targets.fit_targets(sheet), _block_sides(sid, source), correction)
    return ~(d <= 1)


def _resolution(tif: Path) -> float:
    import rasterio
    with rasterio.open(tif) as s:
        return float(s.res[0])


def _cells(pts: np.ndarray, x0: float, y1: float, side: float, ncol: int) -> np.ndarray:
    """Each point's grid box (row-major index; rows down from y1, columns right from x0), -1 left or above."""
    c = np.floor((pts[:, 0] - x0) / side)
    r = np.floor((y1 - pts[:, 1]) / side)
    ok = (c >= 0) & (c < ncol) & (r >= 0)
    return np.where(ok, r * ncol + c, -1).astype(np.int64)


def places(sid: SheetId, source: Path) -> dict[str, tuple[float, float, float, float]]:
    """Three boxes on the sheet (map ft: x0, y0, x1, y1), each WINDOW_PX times the resolution of the warped scan in
    data root `source` (warp/<series>/<sheet>/warped.tif) on a side, on a grid laid from the top-left corner of the
    sheet's warped footprint: `worst`, the box holding the most block-side points off the print under that
    placement (`off_print`); `busiest`, the box holding the most block-side points; `centre`, the box holding the
    footprint's centroid.  Ties go to the first box, row by row from the top.  `worst` is left out when no point is
    off the print.  Pass today's data root, so that both arms are shown on the same ground."""
    folder = Path(source) / "warp" / sid[0] / sid[1]
    side = WINDOW_PX * _resolution(folder / "warped.tif")
    fp = _footprint(source, f"{sid[0]}/{sid[1]}")
    x0, _, x1, y1 = fp.bounds
    ncol = max(1, int(np.ceil((x1 - x0) / side)))
    pts = np.asarray(_block_sides(tuple(sid), Path(source)).pts, float).reshape(-1, 2)
    off = np.asarray(off_print(tuple(sid), Path(source)), bool)
    cell = _cells(pts, x0, y1, side, ncol)
    c = fp.centroid
    centre = int(_cells(np.array([[c.x, c.y]]), x0, y1, side, ncol)[0])
    n = max(int(cell.max(initial=-1)), centre, 0) + 1

    def most(sel):
        return int(np.argmax(np.bincount(cell[sel & (cell >= 0)], minlength=n)))

    def box_of(k):
        r, col = divmod(k, ncol)
        return (x0 + col * side, y1 - (r + 1) * side, x0 + (col + 1) * side, y1 - r * side)
    got = {"worst": box_of(most(off))} if off.any() else {}
    return {**got, "busiest": box_of(most(np.ones(len(pts), bool))), "centre": box_of(centre)}


def shared(boxes: dict) -> dict[str, list[str]]:
    """The places grouped by box, in PLACES order: each group's first place names it, with every place in it."""
    groups: dict[str, list[str]] = {}
    for place in (p for p in PLACES if p in boxes):
        first = next((k for k, v in groups.items() if boxes[k] == boxes[place]), place)
        groups.setdefault(first, []).append(place)
    return groups


def in_box(points_world: np.ndarray, box_ft) -> np.ndarray:
    """Whether each point (map ft) lies in the box (x0, y0, x1, y1; the low edges in, the high ones out)."""
    p = np.asarray(points_world, float).reshape(-1, 2)
    return (p[:, 0] >= box_ft[0]) & (p[:, 0] < box_ft[2]) & (p[:, 1] >= box_ft[1]) & (p[:, 1] < box_ft[3])


def block_lines(boxes) -> list:
    """Today's block lines round the boxes (map ft): the outline of each of today's blocks (`datasf.build_blocks` of
    today's active lots, ref/parcels.parquet) that has a lot touching one of the boxes."""
    import geopandas as gpd

    from sfzmaps.sources import datasf
    lots = gpd.read_parquet(config.REF_DIR / "parcels.parquet")
    near = lots[lots.intersects(shapely.union_all([shapely.box(*b) for b in boxes]))]
    blocks = datasf.build_blocks(lots[lots["block_num"].isin(set(near["block_num"]))])
    return [g.boundary for g in blocks.geometry]


def _window(s, box_ft) -> tuple[int, int, int, int]:
    """The px window (column, row, width, height) of the open raster `s` nearest the box (map ft)."""
    rx, ry = s.res
    c0, r0 = int(round((box_ft[0] - s.bounds.left) / rx)), int(round((s.bounds.top - box_ft[3]) / ry))
    return c0, r0, int(round((box_ft[2] - box_ft[0]) / rx)), int(round((box_ft[3] - box_ft[1]) / ry))


def _panel(tif: Path, box_ft, lines) -> np.ndarray:
    """The warped scan inside the box at its own resolution (BGR uint8; white where there is no scan), with the lines
    drawn LINE_PX wide in LINE over it, each placed by the raster's own georeferencing (the window starts on a whole
    px, not exactly on the box's corner)."""
    import cv2
    import rasterio
    from rasterio.windows import Window
    with rasterio.open(tif) as s:
        rx, ry = s.res
        c0, r0, w, h = _window(s, box_ft)
        wx0, wy1 = s.bounds.left + c0 * rx, s.bounds.top - r0 * ry       # the window's top-left corner, map ft
        a = s.read(window=Window(c0, r0, w, h), boundless=True, fill_value=0).astype(np.float32)
        rows, cols = np.arange(r0, r0 + h), np.arange(c0, c0 + w)
        inside = (((rows >= 0) & (rows < s.height))[:, None] & ((cols >= 0) & (cols < s.width))[None, :])
    rgb = a[:3] if len(a) >= 3 else np.repeat(a[:1], 3, axis=0)
    alpha = (a[3] / 255.0 if len(a) >= 4 else np.ones((h, w), np.float32)) * inside
    img = rgb * alpha + 255.0 * (1 - alpha)
    img = np.ascontiguousarray(np.clip(np.rint(img), 0, 255).astype(np.uint8).transpose(1, 2, 0)[:, :, ::-1])
    clip = shapely.box(*box_ft).buffer(LINE_PX * max(rx, ry))
    for g in lines:
        g = g.boundary if g.geom_type in ("Polygon", "MultiPolygon") else g
        for ln in shapely.get_parts(shapely.intersection(g, clip)):
            if ln.geom_type not in ("LineString", "LinearRing") or ln.is_empty:
                continue
            xy = np.asarray(ln.coords)[:, :2]
            px = np.stack([(xy[:, 0] - wx0) / rx - 0.5, (wy1 - xy[:, 1]) / ry - 0.5], axis=1)
            cv2.polylines(img, [np.rint(px * 16).astype(np.int32)], False, LINE, LINE_PX, cv2.LINE_8, shift=4)
    return img


def placed_key() -> tuple:
    return (("the scan warped onto today's map by each placement, at the warped scan's own resolution; white = no scan "
             "there", (255, 255, 255)),
            (f"magenta = today's block lines (DataSF blocks), drawn on top {LINE_PX} px wide (a picture setting)", LINE))


def placed_picture(before_tif: Path, after_tif: Path, box_ft, lines, title: str) -> np.ndarray:
    """The picture (BGR uint8) of one place: each warped scan (`before_tif` on the left, `after_tif` on the right)
    read inside the box (map ft: x0, y0, x1, y1) at its own resolution, today's block `lines` (map ft) drawn over each
    LINE_PX wide in LINE, the two side by side `look.GAP` apart (a panel shorter than the other padded with white
    below).  A key and `title` above; ValueError when the title does not start "RESULT:", or, reading nothing, when
    the picture would be more than MAX_SIDE px on a side."""
    import rasterio

    from sfzmaps.find import look
    if not title.startswith(RESULT):
        raise ValueError(f"a result picture's title starts {RESULT!r}: {title!r}")
    sizes = []
    for tif in (before_tif, after_tif):
        with rasterio.open(tif) as s:
            sizes.append(_window(s, box_ft)[2:])
    wide, high = sum(w for w, _ in sizes) + look.GAP, max(h for _, h in sizes)
    if max(wide, high) > MAX_SIDE:
        raise ValueError(f"the picture would be {wide} x {high} px, over the {MAX_SIDE} px a side the owner's viewer "
                         f"opens; take a smaller box")
    left, right = _panel(Path(before_tif), box_ft, lines), _panel(Path(after_tif), box_ft, lines)
    h = max(left.shape[0], right.shape[0])

    def tall(p):
        return np.vstack([p, np.full((h - p.shape[0], p.shape[1], 3), 255, np.uint8)])
    gap = np.full((h, look.GAP, 3), 255, np.uint8)
    return look._keyed(np.hstack([tall(left), gap, tall(right)]), title, placed_key())


def write_picture(pic: np.ndarray, f: Path) -> Path | None:
    """Write the picture to `f` and, when its longer side is over `look.SMALL_SIDE`, a reduced overview copy beside it
    (<name>.small.png), returned; None when there is none."""
    import cv2

    from sfzmaps.find import look
    cv2.imwrite(str(f), pic)
    if max(pic.shape[:2]) <= look.SMALL_SIDE:
        return None
    small = f.with_name(f"{f.stem}.small.png")
    cv2.imwrite(str(small), look.reduced(pic, look.SMALL_SIDE))
    return small


def check_blocks(trial: str, sid: SheetId) -> None:
    """ValueError, naming the sheet, when the finder arm's blocks file is not the one `sfzmaps find blocks` wrote: its
    sha256 differs from the one recorded in TRIAL/<trial>/blocks/<series>/<sheet>.json (re-traced or changed since),
    or the file or the record is missing."""
    import hashlib
    label = f"{sid[0]}/{sid[1]}"
    rec = TRIAL / trial / "blocks" / sid[0] / f"{sid[1]}.json"
    f = arm_root(trial, "finder") / "work" / sid[0] / sid[1] / "blocks_px.geojson"
    if not rec.is_file():
        raise ValueError(f"{label}: no record of the finder's blocks ({rec}); make them with `sfzmaps find blocks "
                         f"--trial {trial} --sheet {label}`")
    if not f.is_file():
        raise ValueError(f"{label}: the finder arm has no blocks file ({f})")
    got, want = hashlib.sha256(f.read_bytes()).hexdigest(), json.loads(rec.read_text()).get("sha256")
    if got != want:
        raise ValueError(f"{label}: the finder arm's blocks file was re-traced or changed since `sfzmaps find blocks` "
                         f"wrote it ({f}: sha256 {got}, recorded {want})")


def unplaced(root: Path, sid: SheetId, label: str) -> list[str]:
    """Why the sheet has no placement to compare in data root `root` (`label` names the root: today, traced or
    finder), one line each naming the step to run; empty when it has control points and a warped scan."""
    series, sheet = sid
    env = "" if label == "today" else f"SFZMAPS_DATA_DIR={Path(root).resolve()} "
    why = []
    t = Path(root) / "work" / series / sheet / "transform.json"
    if not len(control_points(root, sid)):
        why.append(f"{series}/{sheet} {label}: no placement yet (no control points in {t}); run "
                   f"`{env}SFZMAPS_MAX_USD=0 pixi run uv run sfzmaps run --year {series} --sheet {sheet}`")
    w = Path(root) / "warp" / series / sheet / "warped.tif"
    if not w.is_file():
        why.append(f"{series}/{sheet} {label}: no warped scan ({w}); run "
                   f"`{env}SFZMAPS_MAX_USD=0 pixi run uv run sfzmaps warp {series}`")
    return why
