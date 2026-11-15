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


def _pages(prefix: str, numbers) -> tuple[str, ...]:
    return tuple(f"{prefix}{i:02d}" for i in numbers)


# The text detector's maps: it learns from the training maps (both lettering styles: the 2000 drawing and its two
# earlier scans, modern italic; the 1960 and the 1964 / 1968 drawings, older serif) and is scored on the test maps,
# whole drawings it never saw (1980, modern; 1972 / 1975, one drawing, older).
TEXT_TRAIN_SERIES = ("2000-digital-zoning", "1980-1990-zoning", "1990-1996-zoning", "1960", "1964", "1968")
TEXT_TEST_SERIES = ("1980", "1972", "1975")
# The sheets Terra read (the saved answers of `text-on-every-sheet`), by series, in `textnet.boxes.sheet_order`: the
# detector reads only these.  Fixed here, like SHEETS, so dvc.yaml is the same before anything is pulled;
# `textnet.boxes.check_text_sheets` stops a stage when the saved answers name other sheets.
TEXT_SHEETS: dict[str, tuple[str, ...]] = {
    "2000-digital-zoning": SHEETS,
    "1980-1990-zoning": _pages("1980-1990-zoning-p", range(1, 14)),
    "1990-1996-zoning": _pages("1990-1996-zoning-p", range(22, 34)),     # 12 of its 33 pages
    "1960": tuple(str(i) for i in range(1, 14)),
    "1964": _pages("20180102105639444-p", range(2, 18)),
    "1968": _pages("20180102110513967-p", range(2, 22)),
    "1980": _pages("20180102115319417-p", [*range(2, 32), 33, 34]),
    "1972": _pages("20180102113004574-p", [*range(1, 30), 31]),
    "1975": _pages("20180102114142444-p", [*range(2, 31), 32]),
}

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
