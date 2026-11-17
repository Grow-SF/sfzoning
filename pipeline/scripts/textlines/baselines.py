"""Score today's text detectors on the text-vs-line gate (`sfzmaps.solver.textnet.linegate`): the baselines every later
training run is compared against.  CPU, one thread, $0.  One process, one `CityGeo` (about 7 GB at the peak): never run
two at once.

    OMP_NUM_THREADS=1 uv run python scripts/textlines/baselines.py

Sources (all four quarters unless said):
  r328         the held-out detectors, nested `data/solver/text-reader/<sheet>/chance.png` (copied with held.json)
  all-sheets   the all-sheets detector, flat `data/solver/text/chance/` (trained on every sheet: in-sample, a reference)
  trial        the detector-overcall recipe trial, flat `data/solver/experiments/detector-overcall/chance/trial/`
  old-<q>      the old r30 quarter models, flat `data/solver/text-<q>/chance/`, each scored on its OWN quarter only

Writes under `data/solver/experiments/textlines/baselines/`: `<source>/{scores.json, scores.md, blocks.npz}` (a source
already written is skipped: remove its folder to redo it), `summary.md`, `panel.png` and `audit/`.
"""
import os

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS",
           "OPENCV_FOR_THREADS_NUM"):
    os.environ[_k] = "1"
import json  # noqa: E402
import shutil  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import cv2  # noqa: E402
import numpy as np  # noqa: E402

cv2.setNumThreads(1)
sys.path.insert(0, str(Path(__file__).resolve().parent))
import audit  # noqa: E402
import panel  # noqa: E402

from sfzmaps.solver import paths, runs  # noqa: E402
from sfzmaps.solver.textnet import data, linegate  # noqa: E402

OUT = runs.EXPERIMENTS_DIR / "textlines" / "baselines"
TRIAL = runs.EXPERIMENTS_DIR / "detector-overcall" / "chance" / "trial"
QUARTERS = tuple(paths.QUARTERS)
SUMMARY_MEASURES = ("m1a_in_box", "m1a", "m1b_in_box", "m1c", "m2", "overlap_band")
TITLE = "text detectors on the 2000 sheets: red = ink called text (chance >= 0.5), grey = other ink"


def copy_today() -> None:
    """Copy the held-out detectors' chance (nested) and held.json to r328/, and the all-sheets chance to all-sheets/."""
    r328, every = OUT / "r328", OUT / "all-sheets"
    for sheet in paths.SHEETS:
        src, dst = paths.READER_TEXT_DIR / sheet / "chance.png", r328 / sheet / "chance.png"
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.exists():
            shutil.copy2(src, dst)
        src, dst = paths.TEXT_CHANCE_DIR / f"{sheet}.png", every / f"{sheet}.png"
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.exists():
            shutil.copy2(src, dst)
    shutil.copy2(paths.READER_TEXT_DIR / "held.json", r328 / "held.json")


def sources() -> dict[str, tuple[linegate.ChanceSource, tuple[str, ...]]]:
    """Each baseline's chance source and the quarters it is scored on."""
    out = {"r328": (linegate.folder_source(OUT / "r328", "nested"), QUARTERS),
           "all-sheets": (linegate.folder_source(OUT / "all-sheets", "flat"), QUARTERS),
           "trial": (linegate.folder_source(TRIAL, "flat"), QUARTERS)}
    for q in QUARTERS:
        out[f"old-{q}"] = (linegate.folder_source(paths.SOLVER_DIR / f"text-{q}" / "chance", "flat"), (q,))
    return out


def score(name: str, source, quarters, sheets, geo) -> dict:
    """The gate on `sheets` for one source, written to OUT/<name>; its scores.json when it exists already."""
    folder = OUT / name
    if (folder / linegate.BLOCKS_FILE).is_file() and (folder / linegate.SCORES_FILE).is_file():
        print(name, "already scored", flush=True)
        return json.loads((folder / linegate.SCORES_FILE).read_text())
    result, tallies = linegate.gate_sheets(sheets, source, quarters, geo)
    if linegate.LEARNED_RULES.is_file():
        found = json.loads(linegate.LEARNED_RULES.read_text()).get("run / font.npz")
        if found:
            result["rules"]["font.npz"] = linegate.summarise_rules(found)
    linegate.write(result, tallies, folder)
    print(name, "scored", flush=True)
    return result


def summary(results: dict[str, dict]) -> str:
    """summary.md: for each source the headline measures, called >= 0.5 with the 95% interval."""
    lines = ["# Baselines on the text-vs-line gate", "",
             ("Each cell: the share of the measure's ink called text (chance >= 0.5), with the 95% bootstrap interval over "
              "blocks. Lower is better for m1a_in_box (district-border ink in a word box), m1a (all district-border ink), "
              "m1b_in_box (run-rule line ink in a word box) and m1c (ink outside every box); higher is better for m2 "
              "(glyph ink in a word box, off the line rule). overlap_band is reported on its own. all-sheets is in-sample "
              "(trained on every sheet): a reference only. old-q rows are scored on their own quarter only."), "",
             "| source | " + " | ".join(SUMMARY_MEASURES) + " |", "|---|" + "---|" * len(SUMMARY_MEASURES)]
    for name, r in results.items():
        cells = []
        for m in SUMMARY_MEASURES:
            v = r[m]
            cells.append("-" if v is None else f"{v['called_05']:.3f} [{v['ci_05'][0]:.3f}, {v['ci_05'][1]:.3f}]")
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    lines += ["", "Ink px (m1a in weight) behind each cell:", "", "| source | " + " | ".join(SUMMARY_MEASURES) + " |",
              "|---|" + "---|" * len(SUMMARY_MEASURES)]
    for name, r in results.items():
        lines.append(f"| {name} | " + " | ".join(linegate._px(r[m]["px"]) for m in SUMMARY_MEASURES) + " |")
    return "\n".join(lines) + "\n"


def old_quarters_source(sheets: list, geo) -> linegate.ChanceSource:
    """The four old quarter models as one chance: each px from the model of its own quarter."""
    by_sheet = {sd.sheet: sd for sd in sheets}
    each = {q: linegate.folder_source(paths.SOLVER_DIR / f"text-{q}" / "chance", "flat") for q in QUARTERS}
    cache: dict[str, np.ndarray] = {}

    def source(sheet: str) -> np.ndarray:
        if sheet not in cache:
            quarter = geo.quarter(by_sheet[sheet])
            out = np.zeros(quarter.shape, np.uint8)
            for i, q in enumerate(QUARTERS):
                m = quarter == i
                if m.any():
                    out[m] = each[q](sheet)[m]
            cache[sheet] = out
        return cache[sheet]
    return source


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    copy_today()
    paths.check_sheets()
    sheets = data.load_sheets([paths.SERIES], hold_out_validation=False)
    geo = linegate.CityGeo()
    srcs = sources()
    results = {name: score(name, s, q, sheets, geo) for name, (s, q) in srcs.items()}
    (OUT / "summary.md").write_text(summary(results))
    print("summary written", flush=True)

    ink = {sd.sheet: data.page_ink(sd) for sd in sheets if sd.sheet in {w[0] for w in panel.WINDOWS.values()}}
    panel_sources = {"r328": srcs["r328"][0], "all-sheets": srcs["all-sheets"][0], "trial": srcs["trial"][0],
                     "old quarters": old_quarters_source(sheets, geo)}
    panel.page_ink = lambda sheet: ink[sheet]
    panel.render(panel_sources, OUT / "panel.png", TITLE)
    print("panel written", flush=True)

    ink_of = lambda sheet: data.page_ink(next(sd for sd in sheets if sd.sheet == sheet))  # noqa: E731
    (OUT / "audit").mkdir(exist_ok=True)
    audits = {}
    src, quarters = srcs["r328"]
    cands, total = audit.candidates(sheets, src, quarters, geo)
    audits["r328"] = (cands, total)
    cands, total = [], 0
    for q in QUARTERS:
        c, t = audit.candidates(sheets, srcs[f"old-{q}"][0], (q,), geo)
        cands += c
        total += t
    audits["old-quarters"] = (cands, total)
    for name, (cands, total) in audits.items():
        items = audit.draw(cands, 64, 0)
        audit.sheet(items, OUT / "audit" / f"{name}.png", ink_of)
        (OUT / "audit" / f"{name}.json").write_text(json.dumps(
            {"letter_sized_pieces": total, "missed_below_0.5": len(cands), "items": items}, indent=1) + "\n")
        print(name, "audit: letter-sized", total, "missed", len(cands), flush=True)


if __name__ == "__main__":
    main()
