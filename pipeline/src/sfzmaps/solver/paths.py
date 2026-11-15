"""Where the 2000 solver's hand-made inputs and built outputs live."""
import importlib
from pathlib import Path

from sfzmaps import config

SERIES = "2000-digital-zoning"
QUARTERS = ("nw", "ne", "sw", "se")

SOLVER_DIR = config.DATA_DIR / "solver"
PAGE_DIR = SOLVER_DIR / "page"
LOTS_DIR = SOLVER_DIR / "lots"
PRINTS_DIR = SOLVER_DIR / "prints"
LABELS_DIR = SOLVER_DIR / "labels"

TABLE_DIR = config.INPUTS_DIR / "zoning-2000"
CUTS_FILE = config.INPUTS_DIR / "merged-page" / "2000-digital-zoning.json"
LABEL_FIXES = config.INPUTS_DIR / "2000-digital-zoning" / "label_fixes.csv"
SAVED_ANSWERS = config.PIPELINE_DIR / "saved-reads" / "answers"


def text_dir(quarter: str) -> Path:
    """The text detector's folder for one quarter of the city."""
    return SOLVER_DIR / f"text-{quarter}"


def sheet_names() -> list[str]:
    """The placed sheets of the 2000 series."""
    # Imported by name, not with an import statement: the build's import scan (fresh.code_files) would
    # follow a statement and make every solver stage depend on the whole warp and old pipeline.
    sheet = importlib.import_module("sfzmaps.sheetwarp.sheet")
    return sheet.placed_sheets(SERIES)
