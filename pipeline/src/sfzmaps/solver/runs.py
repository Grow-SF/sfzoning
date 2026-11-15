"""Where the walls and the solve read and write: the stages' folders, the experiments' (each in its own folder, never
a stage's output), the hand leader moves the placement learns from, and building a folder afresh.  Kept apart from
`paths` (what the earlier stages and the text detector read) so that a change here reruns only the walls and the
solve."""
import os
import shutil
from pathlib import Path

from sfzmaps import config
from sfzmaps.solver import paths

WALLS_DIR = paths.SOLVER_DIR / "walls"
SOLVE_DIR = paths.SOLVER_DIR / "solve"     # the stage's run: placement.npz, runs/, the data-error list, the report
EXPERIMENTS_DIR = paths.SOLVER_DIR / "experiments"     # every other run and fit, one folder each (never shipped)
FINAL_RUN = "final"                        # the `solver-solve` stage's run
WALLS_PREFIX = "walls-"                    # an experiment folder of walls (`walls_experiment_dir`), never a solve's

LABEL_FIXES = config.INPUTS_DIR / "2000-digital-zoning" / "label_fixes.csv"


def solve_dir(run: str) -> Path:
    """The folder of a solve's run: SOLVE_DIR for the stage's run (FINAL_RUN), EXPERIMENTS_DIR/<run> for any
    other.  A name starting with WALLS_PREFIX is refused: those folders hold walls fitted as experiments, which a
    solve built afresh in their place would delete."""
    if not run or "/" in run or run.startswith("."):
        raise ValueError(f"{run!r} is not a run name")
    if run.startswith(WALLS_PREFIX):
        raise ValueError(f"{run!r} is not a run name: {WALLS_PREFIX}* folders hold experiments' walls")
    return SOLVE_DIR if run == FINAL_RUN else EXPERIMENTS_DIR / run


def walls_experiment_dir(candidates) -> Path:
    """The folder of walls fitted with the candidate features `candidates` as an experiment (`sfzmaps solver walls
    --candidates`): EXPERIMENTS_DIR/walls-<the names sorted, joined by "+"; "none" for none>.  A solve reads them
    with `--walls`."""
    names = sorted(candidates)
    return EXPERIMENTS_DIR / f"{WALLS_PREFIX}{'+'.join(names) or 'none'}"


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
