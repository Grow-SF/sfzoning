"""Correct and warp the placed sheets of a series, into data/warp/<series>/<sheet>/.

A sheet's folder holds: transform.json (today's placement plus "correction": {"file": "correction.npz"}, written
last; it is what everything downstream loads), correction.npz (the chosen correction and which kind it is),
warped.tif and warped_paper.tif with their footprints, and scores.json (what the choice used, and the old placement
check's result).  A sheet with nothing to fit, or whose chosen correction has a value that isn't finite, is written
with today's placement uncorrected, and scores.json says why.  Any other failure fails the sheet."""
import functools
import json
import logging
import multiprocessing
import platform
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from importlib.metadata import version
from pathlib import Path

import numpy as np

from sfzmaps import config, fresh
from sfzmaps.inputs import page_patch_path
from sfzmaps.sheetwarp import apply, scores, targets
from sfzmaps.sheetwarp.device import fit_device
from sfzmaps.sheetwarp import first as first_fit
from sfzmaps.sheetwarp import probabilistic as probabilistic_fit
from sfzmaps.sheetwarp.sheet import PLACED_FILES, NothingToFit, Sheet, placed_sheets
from sfzmaps.sheetwarp.street_edges_and_later_blocks import street_edges_and_later_blocks
from sfzmaps.transform import Correction

log = logging.getLogger(__name__)
OUT = config.DATA_DIR / "warp"
STREET_EDGES_AND_LATER_BLOCKS = {"2000-digital-zoning": {"2000-digital-zoning-p09", "2000-digital-zoning-p10"}}
STREET_EDGES_KIND = "street edges and later blocks"


def fit_sheet(sheet: Sheet) -> tuple[Correction, str, dict]:
    """Both corrections of the sheet, their scores and the choice: (the chosen correction, its kind, the scores).
    Raises NothingToFit where there is nothing to work on."""
    sides = targets.block_side_points(sheet)
    aim = targets.fit_targets(sheet)
    held = first_fit.held_sides(sheet, sides, targets.measure(aim, sides))
    first = first_fit.first_correction(sheet, sides, aim, held)
    inputs = probabilistic_fit.probabilistic_inputs(sheet, sides, aim)
    weights = probabilistic_fit.block_line_weights(sheet, aim)
    probabilistic = probabilistic_fit.probabilistic_correction(inputs, aim, weights)
    s = scores.scores(aim, sides, inputs, first, probabilistic)
    if sheet.name in STREET_EDGES_AND_LATER_BLOCKS.get(sheet.series, ()):
        points = street_edges_and_later_blocks(sheet, sides)
        street_inputs = probabilistic_fit.probabilistic_inputs(sheet, points, aim)
        return probabilistic_fit.probabilistic_correction(street_inputs, aim, weights, stop_when_settled=False), STREET_EDGES_KIND, s
    kind = scores.choose(s)
    return {"first": first, "probabilistic": probabilistic}[kind], kind, s


# Modules the warp imports but whose code it never runs: the freshness bookkeeping, the vision-model calls (reading
# imports them; the warp only loads page.json), the sheet statuses and the district readers (districts.read imports
# them; the warp only takes the year and its lots) and the corner finding (linework imports it; the warp only
# measures the street width).
NOT_RUN_BY_THE_WARP = frozenset({
    "sfzmaps.fresh", "sfzmaps.llm", "sfzmaps.manifest", "sfzmaps.stitch", "sfzmaps.districts.labels",
    "sfzmaps.districts.sampling", "sfzmaps.districts.pattern", "sfzmaps.sources.datasf"})
# The Python packages the fits and the warp compute with: a new version of any can change the numbers.
NUMERIC_PACKAGES = ("torch", "numpy", "scipy", "opencv-python-headless", "shapely", "scikit-image")


def warp_code_files() -> list[Path]:
    """The source files of the code the warp runs: a change to any of them redoes every sheet."""
    return fresh.code_files(["sfzmaps.sheetwarp.run"], cut=NOT_RUN_BY_THE_WARP)


@functools.cache
def _library_versions() -> dict[str, str]:
    """The versions of NUMERIC_PACKAGES, and the GDAL of the gdalwarp on PATH (the one the warp runs)."""
    try:
        gdal = subprocess.run(["gdalwarp", "--version"], capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        gdal = "no gdalwarp"
    return {**{p: version(p) for p in NUMERIC_PACKAGES}, "gdal": gdal}


def numeric_versions() -> dict[str, str]:
    """What changes the numbers: the library versions, the device the fits run on (the GPU and the CPU round
    differently, each the same every run) and the macOS version (Metal changes with it; empty elsewhere)."""
    return {**_library_versions(), "fit device": fit_device().type, "macos": platform.mac_ver()[0]}


def _freshness_inputs(sheet: Sheet) -> list[Path]:
    """The files the warp reads for the sheet: its placed files and its paper warp (copied as they are when it is
    left uncorrected), the old placement check and the page's hand-made patch where there are any, the lots and
    streets, and the code it runs."""
    optional = [sheet.folder / "score.json", page_patch_path(sheet.series, sheet.name)]
    return ([sheet.folder / f for f in (*PLACED_FILES, "warped_paper.tif", "warped_paper.footprint.geojson")]
            + [p for p in optional if p.exists()]
            + [config.REF_DIR / "parcels.parquet", config.REF_DIR / "streets.parquet"]
            + warp_code_files())


def _outputs_complete(out: Path) -> bool:
    """Every file a finished sheet has: transform.json, scores.json and the four warp files, and correction.npz when
    transform.json names it."""
    try:
        corrected = "correction" in json.loads((out / "transform.json").read_text())
    except (OSError, ValueError):
        return False
    expected = ["scores.json", *apply.WARP_FILES] + (["correction.npz"] if corrected else [])
    return all((out / f).exists() for f in expected)


def _require_finite(correction: Correction, kind: str) -> None:
    """NothingToFit where the chosen correction has a value that isn't finite: there is no correction to warp with."""
    if not (np.isfinite(correction.grid).all() and np.isfinite(correction.origin).all()
            and np.isfinite(correction.spacing)):
        raise NothingToFit(f"the {kind} correction has values that aren't finite")


def _placement_check(sheet: Sheet) -> dict:
    """The old pipeline's check of the sheet's placement, if it made one."""
    try:
        d = json.loads((sheet.folder / "score.json").read_text())
    except (OSError, ValueError):
        return {}
    return {"placement_check": {"passed": d.get("passed"), "reason": d.get("reason")}}


def correct_sheet(series: str, name: str, force: bool = False) -> str:
    """Fit, choose and warp one sheet into OUT/series/name.  "skipped" when every output is there and its
    inputs (`_freshness_inputs`) and `numeric_versions` are as they were when it was last done."""
    sheet = Sheet(series, name)
    out = OUT / series / name
    placement = out / "transform.json"
    read, versions = _freshness_inputs(sheet), numeric_versions()
    if _outputs_complete(out) and fresh.is_fresh(placement, read, force, versions):
        return "skipped"
    placement.unlink(missing_ok=True)               # written last: a sheet cut short is redone
    fresh.sidecar(placement).unlink(missing_ok=True)
    out.mkdir(parents=True, exist_ok=True)
    today = json.loads((sheet.folder / "transform.json").read_text())
    try:
        correction, kind, s = fit_sheet(sheet)
        _require_finite(correction, kind)
    except NothingToFit as e:
        (out / "correction.npz").unlink(missing_ok=True)
        (out / "scores.json").write_text(json.dumps({"uncorrected": str(e), **_placement_check(sheet)}, indent=1))
        apply.warp_with_correction(sheet, None, out)
        placement.write_text(json.dumps(today))
        fresh.record(placement, read, versions)
        return f"uncorrected: {e}"
    correction.save(out / "correction.npz", kind=kind)
    (out / "scores.json").write_text(json.dumps({**s, "chosen": kind, "device": fit_device().type, **_placement_check(sheet)}, indent=1))
    apply.warp_with_correction(sheet, correction, out)
    placement.write_text(json.dumps({**today, "correction": {"file": "correction.npz"}}))
    fresh.record(placement, read, versions)
    return f"corrected ({kind})"


def _log_to_stderr(level: int) -> None:
    """A spawned worker's logging, set up as the command's (it starts without any), so the fits' progress lines
    reach the command's stderr and the build's log."""
    logging.basicConfig(level=level, format="%(message)s")


def correct_series(series: str, jobs: int, force: bool = False) -> dict[str, str]:
    """Every placed sheet of the series, `jobs` at a time in worker processes (each fit sets its own thread count);
    name -> result.  Raises after every sheet has finished if any raised, naming them."""
    names = placed_sheets(series)
    results, failed = {}, {}

    def finished(name, get_result):
        try:
            results[name] = get_result()
            print(f"{series}/{name}: {results[name]}", flush=True)
        except Exception as e:
            failed[name] = f"{type(e).__name__}: {e}"
            print(f"{series}/{name}: FAILED {failed[name]}", flush=True)
            log.error("%s/%s failed", series, name, exc_info=e)      # a worker's traceback comes along as the cause

    if jobs <= 1:
        for name in names:
            finished(name, lambda name=name: correct_sheet(series, name, force))
    else:
        with ProcessPoolExecutor(jobs, mp_context=multiprocessing.get_context("spawn"), initializer=_log_to_stderr,
                                 initargs=(logging.getLogger().getEffectiveLevel(),)) as pool:
            futures = {pool.submit(correct_sheet, series, name, force): name for name in names}
            for f in as_completed(futures):
                finished(futures[f], f.result)
    if failed:
        raise RuntimeError(f"{len(failed)} of {len(names)} sheets of {series} failed: "
                           + "; ".join(f"{series}/{n}: {m}" for n, m in sorted(failed.items())))
    return {n: results[n] for n in names}
