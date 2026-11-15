"""Where the 2000 solver's earlier stages and its text detector read and write (the page, the lots, the prints, the
detectors, the labels).  The walls' and the solve's folders are in `runs`, so a change to those reruns nothing here."""
from pathlib import Path

from sfzmaps import config

SERIES = "2000-digital-zoning"
QUARTERS = ("nw", "ne", "sw", "se")
# The sheets of the series, by name: what the text stages read, file by file.  Fixed here rather than listed from
# data/warp, so that dvc.yaml (whose deps are made from it) is the same in a fresh clone before anything is pulled;
# `check_sheets` stops a stage that reads them when the warp has placed other sheets.
SHEETS = tuple(f"{SERIES}-p{i:02d}" for i in range(1, 14))

SOLVER_DIR = config.DATA_DIR / "solver"
PAGE_DIR = SOLVER_DIR / "page"
LOTS_DIR = SOLVER_DIR / "lots"
PRINTS_DIR = SOLVER_DIR / "prints"
TEXT_SCORES_DIR = SOLVER_DIR / "text-scores"    # the text detectors' scores on their held-out quarters
LABELS_DIR = SOLVER_DIR / "labels"

TABLE_DIR = config.INPUTS_DIR / "zoning-2000"
CUTS_FILE = config.INPUTS_DIR / "merged-page" / "2000-digital-zoning.json"
SAVED_ANSWERS = config.PIPELINE_DIR / "saved-reads" / "answers"


def text_dir(quarter: str) -> Path:
    """The text detector's folder for one quarter of the city: its model.pt and chance/."""
    return SOLVER_DIR / f"text-{quarter}"


def chance_dir(quarter: str) -> Path:
    """Where the chance of text of each sheet's page goes, for the detector that held out `quarter`."""
    return text_dir(quarter) / "chance"


def sheet_names() -> list[str]:
    """The sheets of the 2000 series the warp placed: the folders of data/warp/<series> that hold a transform.json,
    sorted by name."""
    root = config.DATA_DIR / "warp" / SERIES
    if not root.is_dir():
        return []
    return sorted(d.name for d in root.iterdir() if (d / "transform.json").is_file())


def check_sheets() -> list[str]:
    """SHEETS, once the sheets the warp placed (`sheet_names`) are seen to be exactly those: a stage that reads its
    sheets file by file from SHEETS (the text detector's) must neither miss a sheet placed since nor read one no
    longer placed.  ValueError otherwise."""
    placed = sheet_names()
    if placed != list(SHEETS):
        extra, missing = sorted(set(placed) - set(SHEETS)), sorted(set(SHEETS) - set(placed))
        raise ValueError(f"the warp placed {len(placed)} sheets of {SERIES}, not the {len(SHEETS)} of "
                         f"sfzmaps.solver.paths.SHEETS (placed but not listed: {', '.join(extra) or 'none'}; listed "
                         f"but not placed: {', '.join(missing) or 'none'}): run the warp-{SERIES} stage, or change "
                         "SHEETS and `sfzmaps build --write-dvc-yaml`")
    return list(SHEETS)
