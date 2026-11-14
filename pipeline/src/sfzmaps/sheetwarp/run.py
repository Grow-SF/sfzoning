"""Correct and warp the placed sheets of a series, into data/warp/<series>/<sheet>/.

A sheet's folder holds: transform.json (today's placement plus "correction": {"file": "correction.npz"}, written
last; it is what everything downstream loads), correction.npz (the chosen correction and which kind it is),
warped.tif and warped_paper.tif with their footprints, and scores.json (what the choice used, and the old placement
check's result).  A sheet with nothing to fit is written with today's placement uncorrected, and scores.json says
why.  Any other failure fails the sheet."""
import json
import multiprocessing
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from sfzmaps import config, fresh
from sfzmaps.sheetwarp import apply, scores, targets
from sfzmaps.sheetwarp import first as first_fit
from sfzmaps.sheetwarp import probabilistic as probabilistic_fit
from sfzmaps.sheetwarp.sheet import PLACED_FILES, NothingToFit, Sheet, placed_sheets
from sfzmaps.sheetwarp.street_edges_and_later_blocks import street_edges_and_later_blocks
from sfzmaps.transform import Correction

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
        return probabilistic_fit.probabilistic_correction(street_inputs, aim, weights), STREET_EDGES_KIND, s
    kind = scores.choose(s)
    return {"first": first, "probabilistic": probabilistic}[kind], kind, s


def _freshness_inputs(sheet: Sheet) -> list[Path]:
    return ([sheet.folder / f for f in PLACED_FILES]
            + [config.REF_DIR / "parcels.parquet", config.REF_DIR / "streets.parquet"]
            + fresh.code_files(["sfzmaps.sheetwarp.run"]))


def _placement_check(sheet: Sheet) -> dict:
    """The old pipeline's check of the sheet's placement, if it made one."""
    try:
        d = json.loads((sheet.folder / "score.json").read_text())
    except (OSError, ValueError):
        return {}
    return {"placement_check": {"passed": d.get("passed"), "reason": d.get("reason")}}


def correct_sheet(series: str, name: str, force: bool = False) -> str:
    """Fit, choose and warp one sheet into OUT/series/name.  "skipped" when its inputs (the sheet's placed files,
    the lots and streets, and this code) are as they were when it was last done."""
    sheet = Sheet(series, name)
    out = OUT / series / name
    placement, inputs = out / "transform.json", _freshness_inputs(sheet)
    if fresh.is_fresh(placement, inputs, force):
        return "skipped"
    placement.unlink(missing_ok=True)               # written last: a sheet cut short is redone
    fresh.sidecar(placement).unlink(missing_ok=True)
    out.mkdir(parents=True, exist_ok=True)
    today = json.loads((sheet.folder / "transform.json").read_text())
    try:
        correction, kind, s = fit_sheet(sheet)
    except NothingToFit as e:
        (out / "correction.npz").unlink(missing_ok=True)
        (out / "scores.json").write_text(json.dumps({"uncorrected": str(e), **_placement_check(sheet)}, indent=1))
        apply.warp_with_correction(sheet, None, out)
        placement.write_text(json.dumps(today))
        fresh.record(placement, inputs)
        return f"uncorrected: {e}"
    correction.save(out / "correction.npz", kind=kind)
    (out / "scores.json").write_text(json.dumps({**s, "chosen": kind, **_placement_check(sheet)}, indent=1))
    apply.warp_with_correction(sheet, correction, out)
    placement.write_text(json.dumps({**today, "correction": {"file": "correction.npz"}}))
    fresh.record(placement, inputs)
    return f"corrected ({kind})"


def correct_series(series: str, jobs: int = 3, force: bool = False) -> dict[str, str]:
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

    if jobs <= 1:
        for name in names:
            finished(name, lambda name=name: correct_sheet(series, name, force))
    else:
        with ProcessPoolExecutor(jobs, mp_context=multiprocessing.get_context("spawn")) as pool:
            futures = {pool.submit(correct_sheet, series, name, force): name for name in names}
            for f in as_completed(futures):
                finished(futures[f], f.result)
    if failed:
        raise RuntimeError(f"{len(failed)} of {len(names)} sheets of {series} failed: "
                           + "; ".join(f"{series}/{n}: {m}" for n, m in sorted(failed.items())))
    return {n: results[n] for n in names}
