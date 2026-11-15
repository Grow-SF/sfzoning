"""Where the 2000 solver's hand-made inputs and built outputs live."""
from pathlib import Path

from sfzmaps import config

SERIES = "2000-digital-zoning"
QUARTERS = ("nw", "ne", "sw", "se")

SOLVER_DIR = config.DATA_DIR / "solver"
PAGE_DIR = SOLVER_DIR / "page"
LOTS_DIR = SOLVER_DIR / "lots"
PRINTS_DIR = SOLVER_DIR / "prints"
LABELS_DIR = SOLVER_DIR / "labels"
WALLS_DIR = SOLVER_DIR / "walls"
SOLVE_DIR = SOLVER_DIR / "solve"      # placement.npz, runs/, the data-error list

TABLE_DIR = config.INPUTS_DIR / "zoning-2000"
CUTS_FILE = config.INPUTS_DIR / "merged-page" / "2000-digital-zoning.json"
LABEL_FIXES = config.INPUTS_DIR / "2000-digital-zoning" / "label_fixes.csv"
SAVED_ANSWERS = config.PIPELINE_DIR / "saved-reads" / "answers"


def text_dir(quarter: str) -> Path:
    """The text detector's folder for one quarter of the city."""
    return SOLVER_DIR / f"text-{quarter}"


def sheet_names() -> list[str]:
    """The sheets of the 2000 series the solver reads: the folders of data/warp/<series> that hold a
    transform.json, sorted by name."""
    root = config.DATA_DIR / "warp" / SERIES
    if not root.is_dir():
        return []
    return sorted(d.name for d in root.iterdir() if (d / "transform.json").is_file())
