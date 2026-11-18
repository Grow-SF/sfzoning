"""Where the solver's earlier stages and its text detector read and write (the page, the lots, the prints, the
detector, the labels).  The walls' and the solve's folders are in `runs`, so a change to those reruns nothing here.

The solver solves one series at a time: SERIES, the 2000 map (SERIES_2000) unless the environment names another in
SERIES_ENV (`sfzmaps solver --series S` sets it and starts the command again in a process that reads it, so every
module, and every process a command starts, sees the same series).  The 2000 map's folders are data/solver/; another
series' are data/solver/experiments/all-years/<series>/ (`solver_dir`), in the same layout.  Another series is solved
with the models fitted on the 2000 map (its walls, trust, placement, street sides and lot sizes: TRANSFER_DIR, made by
`sfzmaps solver transfer-models`), never with a fit on a table of its own."""
import os

from sfzmaps import config

SERIES_ENV = "SFZMAPS_SOLVER_SERIES"
SERIES_2000 = "2000-digital-zoning"
SERIES = os.environ.get(SERIES_ENV) or SERIES_2000
QUARTERS = ("nw", "ne", "sw", "se")
# The sheets of the 2000 series, by name: the pages the text detector writes its chance of text for, file by file.
# Fixed here rather than listed from data/warp, so that dvc.yaml (whose deps are made from it) is the same in a fresh
# clone before anything is pulled; `check_sheets` stops a stage that reads them when the warp has placed other sheets.
SHEETS_2000 = tuple(f"{SERIES_2000}-p{i:02d}" for i in range(1, 14))
# Every series is solved on the April 2000 table's lots; only the 2000 map's solve reads the table's codes (its fits
# learn from them).  For every other series `table.known_and_codes` gives no lot a code and `lots` writes the table
# without them, so the codes are no evidence or prior there: the series is solved with the 2000 map's models as they
# were fitted (`transfer`).  The 1996-2000 scan of the same drawing is scored on the table afterwards, outside the
# solve (TABLE_SCORED_SERIES; `scripts/all_years/score_series.py`, on the 2000 map's paths).
TABLE_SERIES = (SERIES_2000,)
TABLE_SCORED_SERIES = (SERIES_2000, "1996-2000-zoning")
# The series printed on the 2000 drawing's sheets (the same frame and layout): their crop is the printed frame
# (`frame.find_frame`); every other series' is its neatline interior as the warp finds it (`page.sheet_crop`).
FRAME_SERIES = (SERIES_2000, "2000-digital-height", "2000-digital-su", "1996-2000-zoning", "1996-2000-height",
                "1996-2000-su", "1990-1996-zoning", "1990-1996-height", "1990-1996-su", "1980-1990-zoning",
                "1980-1990-height", "1980-1990-su")


def is_2000() -> bool:
    """The series being solved is the 2000 map, whose own table every fit learns from."""
    return SERIES == SERIES_2000


def has_table() -> bool:
    """The series being solved reads the April 2000 table's codes (TABLE_SERIES: the 2000 map alone)."""
    return SERIES in TABLE_SERIES


def solver_dir(series: str):
    """The folder of a series' solver outputs: data/solver for the 2000 map, data/solver/experiments/all-years/<series>
    for any other."""
    root = config.DATA_DIR / "solver"
    return root if series == SERIES_2000 else root / "experiments" / "all-years" / series


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
    "2000-digital-zoning": SHEETS_2000,
    "1980-1990-zoning": _pages("1980-1990-zoning-p", range(1, 14)),
    "1990-1996-zoning": _pages("1990-1996-zoning-p", range(22, 34)),     # 12 of its 33 pages
    "1960": tuple(str(i) for i in range(1, 14)),
    "1964": _pages("20180102105639444-p", range(2, 18)),
    "1968": _pages("20180102110513967-p", range(2, 22)),
    "1980": _pages("20180102115319417-p", [*range(2, 32), 33, 34]),
    "1972": _pages("20180102113004574-p", [*range(1, 30), 31]),
    "1975": _pages("20180102114142444-p", [*range(2, 31), 32]),
}

SOLVER_DIR = solver_dir(SERIES)
PAGE_DIR = SOLVER_DIR / "page"
LOTS_DIR = SOLVER_DIR / "lots"
PRINTS_DIR = SOLVER_DIR / "prints"
TEXT_DIR = SOLVER_DIR / "text"                  # the text detector's stage: its model and its chance of text
TEXT_MODEL = solver_dir(SERIES_2000) / "text" / "model.pt"     # the one detector, whatever the series
TEXT_CHANCE_DIR = TEXT_DIR / "chance"            # its chance of text on each sheet's page of the series, <sheet>.png
TEXT_SCORES_DIR = SOLVER_DIR / "text-scores"    # its scores on the test drawings
READER_TEXT_DIR = SOLVER_DIR / "text-reader"    # the code reader's detectors, one per held-out quarter, and their
                                                # outputs on each 2000 sheet
LABELS_DIR = SOLVER_DIR / "labels"

TABLE_DIR = config.INPUTS_DIR / "zoning-2000"
CUTS_FILE = config.INPUTS_DIR / "merged-page" / f"{SERIES}.json"   # none for most series: `page.load_cuts`
SAVED_ANSWERS = config.PIPELINE_DIR / "saved-reads" / "answers"
# The models fitted on the 2000 map that solve every other series (`transfer`), made from the 2000 map's stages.
TRANSFER_DIR = config.DATA_DIR / "solver" / "experiments" / "all-years" / "models-2000"


def sheet_names() -> list[str]:
    """The sheets of the series the warp placed: the folders of data/warp/<series> that hold a transform.json,
    sorted by name."""
    root = config.DATA_DIR / "warp" / SERIES
    if not root.is_dir():
        return []
    return sorted(d.name for d in root.iterdir() if (d / "transform.json").is_file())


def check_sheets() -> list[str]:
    """SHEETS, once the sheets the warp placed (`sheet_names`) are seen to be exactly those: a stage that reads its
    sheets file by file from SHEETS (the text detector's chance, as the walls read it) must neither miss a sheet placed
    since nor read one no longer placed.  ValueError otherwise."""
    placed = sheet_names()
    if placed != list(SHEETS):
        extra, missing = sorted(set(placed) - set(SHEETS)), sorted(set(SHEETS) - set(placed))
        raise ValueError(f"the warp placed {len(placed)} sheets of {SERIES}, not the {len(SHEETS)} of "
                         f"sfzmaps.solver.paths.SHEETS (placed but not listed: {', '.join(extra) or 'none'}; listed "
                         f"but not placed: {', '.join(missing) or 'none'}): run the warp-{SERIES} stage, or change "
                         "SHEETS and `sfzmaps build --write-dvc-yaml`")
    return list(SHEETS)


# The sheets of the series being solved: the 2000 map's fixed list; another series' as the warp placed them.
SHEETS = SHEETS_2000 if SERIES == SERIES_2000 else tuple(sheet_names())
