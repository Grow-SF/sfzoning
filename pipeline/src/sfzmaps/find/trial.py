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

These read placement outputs only to report them; `labels.json` is copied as a file and never opened.  Writes only
inside the trial root it is given (and `TRIAL/<T>/numbers.json`, written by the command)."""
from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

import numpy as np
import shapely

from sfzmaps import inputs
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


def trained_ground(sid: SheetId, run_folder: Path, source: Path) -> dict:
    """How much of the sheet's ground the run was trained on.  `lot_labelled_share`: the share of the sheet's warped
    footprint (source/warp/<series>/<sheet>/warped.footprint.geojson) that lies under the union of the warped
    footprints of the run's lot-labelled sheets (the items of its log.json with no `__` in their name); `by_sheet`:
    each such sheet's own share, where it is above 0; `hand_crops`: the run's training items on this sheet (its hand
    crops).  FileNotFoundError when a footprint is missing."""
    items = [it["name"] for it in json.loads((Path(run_folder) / "log.json").read_text())["items"]]
    sheet = _footprint(source, f"{sid[0]}/{sid[1]}")
    lot_labelled = {name: _footprint(source, name) for name in items if "__" not in name}
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
