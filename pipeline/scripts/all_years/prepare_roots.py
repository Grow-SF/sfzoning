"""Seed each series' solver folder (data/solver/<series>) from the all-years experiment's outputs that don't depend on
the lots: its merged page and the detector's chance of text (neither reprojects anything); and the 2000 models
(data/solver/models-2000), fitted from the 2000 stages' shipped outputs.  Prints, per series, what was seeded and
what is still to run.

    uv run python scripts/all_years/prepare_roots.py"""
import json
import shutil

from sfzmaps import config
from sfzmaps.solver import paths

EXP = config.DATA_DIR / "solver" / "experiments" / "all-years"


def main():
    if not paths.TRANSFER_DIR.is_dir() and (EXP / "models-2000").is_dir():
        shutil.copytree(EXP / "models-2000", paths.TRANSFER_DIR)
    out = {}
    for s in paths.ALL_YEARS:
        sheets = sorted(d.name for d in (config.DATA_DIR / "warp" / s).iterdir() if (d / "transform.json").is_file())
        src, dst = EXP / s, paths.solver_dir(s)
        done = []
        page = src / "page"
        frames = json.loads((page / "frames.json").read_text()) if (page / "frames.json").is_file() else {}
        if sorted(frames) == sheets and (page / "ink.png").is_file() and not (dst / "page").exists():
            shutil.copytree(page, dst / "page")
        if (dst / "page" / "ink.png").is_file():
            done.append("page")
        chance = src / "text" / "chance"
        if chance.is_dir() and sorted(f.stem for f in chance.glob("*.png")) == sheets and not (dst / "text").exists():
            shutil.copytree(src / "text", dst / "text")
        if (dst / "text" / "chance").is_dir():
            done.append("text-chance")
        out[s] = done
        print(s, len(sheets), "sheets;", ", ".join(done) or "nothing seeded")
    return out


if __name__ == "__main__":
    main()
