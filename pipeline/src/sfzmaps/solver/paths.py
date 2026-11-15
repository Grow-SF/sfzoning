"""Where the 2000 solver's hand-made inputs and built outputs live."""
import os
import shutil
from pathlib import Path

from sfzmaps import config

SERIES = "2000-digital-zoning"
QUARTERS = ("nw", "ne", "sw", "se")

SOLVER_DIR = config.DATA_DIR / "solver"
PAGE_DIR = SOLVER_DIR / "page"
LOTS_DIR = SOLVER_DIR / "lots"
PRINTS_DIR = SOLVER_DIR / "prints"
TEXT_SCORES_DIR = SOLVER_DIR / "text-scores"    # the text detectors' scores on their held-out quarters
LABELS_DIR = SOLVER_DIR / "labels"
WALLS_DIR = SOLVER_DIR / "walls"
SOLVE_DIR = SOLVER_DIR / "solve"      # the stage's run: placement.npz, runs/, the data-error list, the report
EXPERIMENTS_DIR = SOLVER_DIR / "experiments"    # every other run, one folder each (no stage's output, never shipped)
FINAL_RUN = "final"                   # the `solver-solve` stage's run

TABLE_DIR = config.INPUTS_DIR / "zoning-2000"
CUTS_FILE = config.INPUTS_DIR / "merged-page" / "2000-digital-zoning.json"
LABEL_FIXES = config.INPUTS_DIR / "2000-digital-zoning" / "label_fixes.csv"
SAVED_ANSWERS = config.PIPELINE_DIR / "saved-reads" / "answers"


def text_dir(quarter: str) -> Path:
    """The text detector's folder for one quarter of the city: its model.pt and chance/."""
    return SOLVER_DIR / f"text-{quarter}"


def chance_dir(quarter: str) -> Path:
    """Where the chance of text of each sheet's page goes, for the detector that held out `quarter`."""
    return text_dir(quarter) / "chance"


def solve_dir(run: str) -> Path:
    """The folder of a solve's run: SOLVE_DIR for the stage's run (FINAL_RUN), EXPERIMENTS_DIR/<run> for any
    other."""
    if not run or "/" in run or run.startswith("."):
        raise ValueError(f"{run!r} is not a run name")
    return SOLVE_DIR if run == FINAL_RUN else EXPERIMENTS_DIR / run


def check_final(run: str, walls_dir=None, true_borders: bool = False) -> None:
    """The stage's run (FINAL_RUN) is shipped, so it is solved only with the stage's own walls (WALLS_DIR) and never
    as the ceiling check: refuse it with other walls or with the true borders."""
    if run == FINAL_RUN and (walls_dir is not None or true_borders):
        raise ValueError(f"the run {FINAL_RUN!r} is the solver-solve stage's: solve it without --walls and "
                         "--true-borders, or give the experiment another name")


def fresh_folder(folder: Path) -> Path:
    """An empty folder beside `folder` (.<name>.part) to build its next contents in, for `replace_folder`."""
    part = Path(folder).with_name(f".{Path(folder).name}.part")
    shutil.rmtree(part, ignore_errors=True)
    part.mkdir(parents=True)
    return part


def replace_folder(part: Path, folder: Path) -> None:
    """Put the folder `part` (from `fresh_folder`) in the place of `folder`: renamed over it, so nothing of the old
    contents is left and a reader never sees a part-written folder."""
    folder = Path(folder)
    old = folder.with_name(f".{folder.name}.old.{os.getpid()}")
    if folder.exists():
        os.replace(folder, old)
    os.replace(part, folder)
    shutil.rmtree(old, ignore_errors=True)


def sheet_names() -> list[str]:
    """The sheets of the 2000 series the solver reads: the folders of data/warp/<series> that hold a
    transform.json, sorted by name."""
    root = config.DATA_DIR / "warp" / SERIES
    if not root.is_dir():
        return []
    return sorted(d.name for d in root.iterdir() if (d / "transform.json").is_file())
