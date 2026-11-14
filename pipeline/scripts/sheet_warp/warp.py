"""The sheet warp: correct each sheet's placement so its printed block lines land on the block lines of its year, and
warp the sheets with the corrections (README.md).  Run from pipeline/:

  uv run python scripts/sheet_warp/warp.py all [YEAR ...]       fit, choose and warp (default: every year it corrects)
  uv run python scripts/sheet_warp/warp.py fit YEAR [SHEET ...]  fit one year's sheets (default: all it corrects)
  uv run python scripts/sheet_warp/warp.py again YEAR [SHEET ...] fit the probabilistic correction again and compare:
                                                                 the same every run?

It reads the pipeline's data (data/work, data/ref) and writes only under data/sheet-warp/:
  fits/<year>/<sheet>/     each sheet's corrections and what they were fitted from; fits/scores.jsonl, fits/choice.json
  stand-in-data/           the pipeline's data folder with each sheet's chosen correction, and the warped sheets
  fits-environment/        the PyTorch environment (fits_environment/pyproject.toml, made by uv on first use)
  logs/<year>/<sheet>.log
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
STEPS = HERE / "steps"
PIPELINE = HERE.parents[1]
DATA = PIPELINE / "data"
WORK = DATA / "work"
OUT = DATA / "sheet-warp"
FITS, STAND_IN, LOGS = OUT / "fits", OUT / "stand-in-data", OUT / "logs"
YEARS = ["1921", "1929", "1935", "1948", "1960", "1964", "1968", "1972", "1975", "1980", "1980-1990-zoning",
         "1990-1996-zoning", "2000-digital-zoning"]
EXTRA_EVIDENCE = {"2000-digital-zoning": ["2000-digital-zoning-p09", "2000-digital-zoning-p10"]}
JOBS = 3                       # sheets fitted at once


def sheets(year: str) -> list[str]:
    """the year's sheets the warp corrects: every fitted sheet of the pattern maps (1921-1948, which print no codes),
    and the fitted sheets the pipeline's label reader classes as zoning-use sheets in the other years (not
    height-and-bulk, special-use or blank).  From the pipeline's data only; no model is asked."""
    from sfzmaps import llm
    from sfzmaps.districts import labels
    from sfzmaps.districts.series import SERIES

    llm.CACHE_ONLY = True
    need = ("transform.json", "warped.tif", "blocks_px.geojson", "refine.json")
    fitted = sorted(d.name for d in (WORK / year).iterdir() if d.is_dir() and all((d / f).exists() for f in need))
    if SERIES[year].reader == "A":
        return fitted
    roles = labels.sheet_roles(year, fitted)
    return [sh for sh in fitted if roles[sh] == "use"]


def run(cmd: list, log: Path, env: dict | None = None) -> None:
    with open(log, "a") as f:
        f.write(f"$ {' '.join(map(str, cmd))}\n")
        f.flush()
        r = subprocess.run([str(c) for c in cmd], cwd=PIPELINE, env={**os.environ, **(env or {})}, stdout=f, stderr=f)
    if r.returncode:
        raise RuntimeError(f"{next(Path(str(c)).name for c in cmd if str(c).endswith('.py'))} failed: see {log}")


def py(step: str, *args) -> list:
    """a step in the pipeline's environment"""
    return [sys.executable, "-W", "ignore", STEPS / step, *args]


def torch_py(step: str, *args) -> list:
    """a step in the PyTorch environment (uv, locked: fits_environment/)"""
    return ["uv", "run", "--quiet", "--frozen", "--project", HERE / "fits_environment", "python", STEPS / step, *args]


TORCH_ENV = {"UV_PROJECT_ENVIRONMENT": str(OUT / "fits-environment")}


def fit_sheet(year: str, sheet: str, fits: Path = FITS) -> str:
    """both corrections of one sheet, each skipped when already made"""
    d, log = fits / year / sheet, LOGS / year / f"{sheet}.log"
    d.mkdir(parents=True, exist_ok=True)
    log.parent.mkdir(parents=True, exist_ok=True)
    env = {"SERIES": year, "SHEET": sheet, "WARP_FITS": str(fits), **TORCH_ENV}
    if not (d / "first_correction.npz").exists():
        for step in (py("block_side_points.py"), py("fit_targets.py"), py("measure_fit.py", "today"),
                     py("held_sides.py"), py("first_correction_inputs.py"), py("propagate_shifts.py"),
                     torch_py("first_correction_fit.py"), py("measure_fit.py", "first")):
            run(step, log, env)
    if not (d / "probabilistic_correction.npz").exists():
        for step in (py("probabilistic_inputs.py"), py("block_line_weights.py"), torch_py("probabilistic_fit.py"),
                     py("measure_fit.py", "probabilistic"), py("score_fits.py")):
            run(step, log, env)
        (d / "block_line_weights.npz").unlink()                          # large, and quick to make again
    return sheet


def fit_year(year: str, only: list[str] | None = None, fits: Path = FITS) -> None:
    todo = only or sheets(year)
    print(f"{year}: fitting {len(todo)} sheets", flush=True)
    with ThreadPoolExecutor(JOBS) as ex:
        for f in [ex.submit(fit_sheet, year, sh, fits) for sh in todo]:
            print(f"  {f.result()} done", flush=True)


def extra_evidence(year: str, sheet: str) -> Path:
    """the probabilistic correction fitted with later years' blocks and street edges as well, where the year's own
    blocks don't explain the print (2000 p09 and p10: README)"""
    d = FITS / year / sheet
    out = d / "extra_evidence_correction.npz"
    if not out.exists():
        env = {"SERIES": year, "SHEET": sheet, "WARP_FITS": str(FITS), **TORCH_ENV}
        for step in (py("extra_evidence_points.py"), py("probabilistic_inputs.py", "extra_evidence"),
                     py("block_line_weights.py"), torch_py("probabilistic_fit.py", "extra_evidence")):
            run(step, LOGS / year / f"{sheet}.log", env)
        (d / "block_line_weights.npz").unlink()
    return out


def choose() -> dict:
    """each sheet's correction: the probabilistic one where it puts at least as many points within 1 px as the first
    and never folds, else the first; extra evidence where it applies.  Written into the stand-in data folder."""
    rows = {}
    for line in open(FITS / "scores.jsonl"):
        r = json.loads(line)
        rows[(r["series"], r["sheet"])] = r
    choice = {}
    for (year, sheet), r in sorted(rows.items()):
        fit = ("probabilistic" if r["within1_probabilistic"] >= r["within1_first"] and r["smallest_cell_probabilistic"] > 0
               else "first")
        f = FITS / year / sheet / f"{fit}_correction.npz"
        if sheet in EXTRA_EVIDENCE.get(year, []):
            fit, f = "extra evidence", extra_evidence(year, sheet)
        choice[f"{year}/{sheet}"] = {"fit": fit, "file": str(f), **{k: r[k] for k in (
            "within1_today", "within1_first", "within1_probabilistic", "smallest_cell_probabilistic")}}
    stand_in({tuple(k.split("/")): Path(v["file"]) for k, v in choice.items()})
    json.dump(choice, open(FITS / "choice.json", "w"), indent=1)
    print(f"chosen: {sum(v['fit'] == 'probabilistic' for v in choice.values())} probabilistic, "
          f"{sum(v['fit'] == 'first' for v in choice.values())} first, "
          f"{sum(v['fit'] == 'extra evidence' for v in choice.values())} extra evidence", flush=True)
    return choice


def stand_in(chosen: dict) -> None:
    """the stand-in data folder: every file a link to the pipeline's own, except each corrected sheet's folder, whose
    transform.json carries "correction": its chosen correction.  The pipeline's data is never written."""
    STAND_IN.mkdir(parents=True, exist_ok=True)
    for name in os.listdir(DATA):
        if name not in ("work", "sheet-warp") and not os.path.lexists(STAND_IN / name):
            os.symlink(DATA / name, STAND_IN / name)
    (STAND_IN / "work").mkdir(exist_ok=True)
    years = {y for y, _ in chosen}
    for name in os.listdir(WORK):
        dst = STAND_IN / "work" / name
        if name in years:
            if dst.is_symlink():
                dst.unlink()
            dst.mkdir(exist_ok=True)
            for sh in os.listdir(WORK / name):
                if (name, sh) not in chosen and not os.path.lexists(dst / sh):
                    os.symlink(WORK / name / sh, dst / sh)
        elif not os.path.lexists(dst):
            os.symlink(WORK / name, dst)
    for (year, sheet), corr in chosen.items():
        src, dst = WORK / year / sheet, STAND_IN / "work" / year / sheet
        if dst.is_symlink():
            dst.unlink()
        dst.mkdir(exist_ok=True)
        for f in os.listdir(src):
            if f != "transform.json" and not os.path.lexists(dst / f):
                os.symlink(src / f, dst / f)
        tj = json.loads((src / "transform.json").read_text())
        tj["correction"] = {"file": str(corr)}
        (dst / "transform.json").write_text(json.dumps(tj))


def warp_sheet(year: str, sheet: str) -> str:
    """one sheet warped with its chosen correction, into the stand-in data folder"""
    env = {"SERIES": year, "SFZ_CORR": "1", "PYTHONPATH": str(HERE / "correction_patch"),
           "SFZMAPS_DATA_DIR": str(STAND_IN),
           "PATH": f"{PIPELINE.parent / '.pixi' / 'envs' / 'default' / 'bin'}{os.pathsep}{os.environ['PATH']}"}
    run(py("warp_with_correction.py", sheet, "both"), LOGS / year / f"{sheet}.log", env)
    return f"{year}/{sheet}"


def warp_all(choice: dict) -> None:
    print(f"warping {len(choice)} sheets", flush=True)
    with ThreadPoolExecutor(2) as ex:
        for f in [ex.submit(warp_sheet, *k.split("/")) for k in choice]:
            print(f"  {f.result()} warped", flush=True)


def again(year: str, only: list[str] | None = None) -> None:
    """the probabilistic correction fitted a second time from the same inputs (fits-again/), and how far the two
    fields differ per sheet: largest, 99th percentile and median, page px"""
    todo = only or sorted(p.name for p in (FITS / year).iterdir() if p.is_dir())
    second = OUT / "fits-again"
    for sh in todo:
        d = second / year / sh
        shutil.rmtree(d, ignore_errors=True)
        shutil.copytree(FITS / year / sh, d)
        (d / "probabilistic_correction.npz").unlink()
    with ThreadPoolExecutor(JOBS) as ex:
        list(ex.map(lambda sh: fit_sheet(year, sh, second), todo))
    print(f"{'sheet':26s} {'max':>6s} {'p99':>6s} {'median':>7s}  (px)")
    for sh in todo:
        a = np.load(FITS / year / sh / "probabilistic_correction.npz")["Gc"]
        b = np.load(second / year / sh / "probabilistic_correction.npz")["Gc"]
        m = np.hypot(*(a - b))
        print(f"{sh:26s} {m.max():6.2f} {np.percentile(m, 99):6.2f} {np.median(m):7.3f}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)
    sub.add_parser("all").add_argument("years", nargs="*")
    for name in ("fit", "again"):
        p = sub.add_parser(name)
        p.add_argument("year")
        p.add_argument("sheets", nargs="*")
    a = ap.parse_args()
    if a.command == "all":
        for year in a.years or YEARS:
            fit_year(year)
        warp_all(choose())
    elif a.command == "fit":
        fit_year(a.year, a.sheets)
    else:
        again(a.year, a.sheets)


if __name__ == "__main__":
    main()
