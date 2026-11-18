"""Tracked builds: the stage table, pipeline/dvc.yaml, `sfzmaps stage` and `sfzmaps build`.

DVC (pipeline/dvc.yaml + dvc.lock, committed) records the content hash of every input and output
of every stage, so a checkout can tell whether its data matches its code and inputs. DVC runs
stages one at a time, so `sfzmaps build` runs them itself, up to --jobs at once in dependency
order, and asks DVC only what is stale (`dvc status`) and to record each success (`dvc commit`).
A stage that is planned only because an upstream stage is stale is asked about again (`dvc status
STAGE`) once those have run and been recorded, and doesn't run when what they wrote is unchanged.

A stage is the smallest unit whose writes don't cross another's (DVC refuses two stages writing
into one tracked folder, and a height series' districts write into its zoning series' out/):

  edition-<E>   one series (1921 ... 1980): run, paper, districts
  period-<P>    a period's zoning, height and special-use series (1980-1990 ...): run and paper
                for each, then districts for height before zoning (zoning reads the heights)
  today         districts for today's DataSF zoning
  shared        the merged archive the site reads (districts --shared)
  warp-<S>      the sheet warp of one series (sfzmaps warp S), after the stage above that places it
  solver-<part> one of the 2000 solver's inputs (sfzmaps solver <part>), after the 2000 warp
  solver-publish  the 2000 map's zoning from the solve (sfzmaps solver publish): the published files of
                out/2000-digital-zoning/districts/, which its period's stage leaves to it
  solver-publish-<S>  the same for each other series published from its solve (districts.solved.SOLVED;
                sfzmaps solver publish --series S), after its edition or period and the stages writing its solve

The old pipeline's stages (edition-*, period-*, today, shared) are frozen: the sheet warp replaces
them, so a build runs only the warp-* stages (`build --force NAME` still runs an old one).

A stage never spends money: it runs with SFZMAPS_MAX_USD=0, so a cached answer is free and any
live vision call fails the stage. Paid reads are run deliberately, then `sfzmaps build` again.
"""
from __future__ import annotations

import contextlib
import fcntl
import json
import logging
import os
import subprocess
import sys
import time
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from sfzmaps import config

log = logging.getLogger(__name__)

# DVC runs pinned, outside the project environment: upgrading it must not change uv.lock (a dep
# of every stage) and so restale everything.
DVC = ["uv", "tool", "run", "--from", "dvc==3.67.1", "dvc"]
# Code that can't change a build output: the editor, the build driver, bundling, evaluation,
# shipping (ship, cachepack, datapack, release), the toolchain check, the GIS downloads and the GPU's turns.
CODE_EXCLUDE = {"editor", "build.py", "share.py", "evaluate.py", "ship.py", "cachepack.py",
                "datapack.py", "release.py", "toolchain.py", "downloads", "savedreads.py", "solver", "gpu.py",
                "__pycache__"}
# (solver: the 2000 solver's stages name their own code, as an import closure.)
# Modules a command stage imports that can't change its output, left out of its code deps (`Stage.cut`): the GPU's
# turns (`sfzmaps.gpu`) only decide when a stage computes on the GPU, never what.
OUTPUT_NEUTRAL = ("sfzmaps.gpu",)
# The exit code of a pipeline command stopped by llm.FATAL (the $0 cap, or no API key): "needs a
# paid read", not a bug. Same value as cli.PAID_READ_EXIT (a test keeps them equal).
PAID_READ_EXIT = 75
# How `build --system-tools` reaches its stages' commands (`sfzmaps warp` checks its tools too). Same value as
# cli.SYSTEM_TOOLS_ENV (a test keeps them equal).
SYSTEM_TOOLS_ENV = "SFZMAPS_SYSTEM_TOOLS"
# Sheets the sheet warp fits at once, in all: `sfzmaps warp` alone, or shared by the warp stages a build runs at
# once.  Not in config.py (the warp's code fingerprint): changing it must not refit every sheet.
WARP_WORKERS = 6
# How `build --jobs` reaches a stage's process: the warp stages a build runs at once share WARP_WORKERS.
BUILD_JOBS_ENV = "SFZMAPS_BUILD_JOBS"


@dataclass(frozen=True)
class Stage:
    name: str
    series: tuple[str, ...]      # run and paper, in this order
    districts: tuple[str, ...]   # districts --series, in this order
    upstream: tuple[str, ...] = ()
    frozen: bool = False         # never run by a build or dvc repro; only `build --force NAME`
    warp: str | None = None      # the series this stage corrects (`sfzmaps warp`): a sheet-warp stage
    command: tuple[str, ...] = ()   # the `sfzmaps` argv tail of a command stage (no old-pipeline machinery)
    reads: tuple[str, ...] = ()     # a command stage's data and input deps, relative to pipeline/
    writes: tuple[str, ...] = ()    # a command stage's outs
    code: tuple[str, ...] = ()      # a command stage's entry modules; its code deps are their import closure
    cut: tuple[str, ...] = OUTPUT_NEUTRAL   # modules left out of that closure (`fresh.code_files`), with what only
                                            # they import

    @property
    def shared(self) -> bool:
        return self.name == "shared"

    @property
    def refdata(self) -> bool:
        return self.name == "refdata"

    @property
    def saved_reads(self) -> bool:
        return self.name == "saved-reads"


def stages() -> dict[str, Stage]:
    """Every stage, in dependency order (shared last)."""
    from sfzmaps.districts.series import SERIES

    periods: dict[str, list[str]] = {}
    for s, info in SERIES.items():
        periods.setdefault(info.period, []).append(s)
    # The DataSF reference data, recorded and shipped like any output (a fresh checkout gets
    # main's, not today's): frozen, because fetching it again would change every build.
    out: dict[str, Stage] = {"refdata": Stage("refdata", (), (), frozen=True)}
    # The images the saved vision reads were given (pipeline/saved-reads): made by a replay outside the build, only
    # checked here and recorded, so they ship and pull with everything else.  Nothing depends on them.
    out["saved-reads"] = Stage("saved-reads", (), (), frozen=True)
    for period, members in periods.items():
        role = {s: SERIES[s].role for s in members}
        reads = sorted((s for s in members if role[s] != "skip"), key=lambda s: role[s] != "height")
        name = f"edition-{period}" if len(members) == 1 else f"period-{period}"
        out[name] = Stage(name, tuple(members), tuple(reads), ("refdata",), frozen=True)
    out["today"] = Stage("today", (), ("today",), ("refdata",), frozen=True)
    # The old pipeline is replaced by the sheet warp: its stages never run again, only `build --force`.
    old = tuple(n for n in out if n != "saved-reads")
    owner = {s: n for n, st in out.items() for s in st.series}
    for s in SERIES:
        out[f"warp-{s}"] = Stage(f"warp-{s}", (), (), (owner[s], "refdata"), warp=s)
    # The 2000 solver's inputs: command stages, each naming its code, its reads and its writes.
    # Of a checked-in inputs folder, only the file read: its README or PROVENANCE.md never rerun the chain, and the
    # table's by-eye corrections (read only in scoring) rerun only the solve.
    out["solver-page"] = Stage("solver-page", (), (), ("warp-2000-digital-zoning",), command=("solver", "page"),
                               reads=("data/warp/2000-digital-zoning", "data/work/2000-digital-zoning",
                                      "inputs/merged-page/2000-digital-zoning.json"),
                               writes=("data/solver/page",), code=("sfzmaps.solver.page",))
    out["solver-lots"] = Stage("solver-lots", (), (), ("solver-page",), command=("solver", "lots"),
                               reads=("inputs/zoning-2000/april2000.geojson.gz", "data/solver/page"),
                               writes=("data/solver/lots",), code=("sfzmaps.solver.lots",))
    out["solver-prints"] = Stage("solver-prints", (), (), ("solver-lots",), command=("solver", "prints"),
                                 reads=("saved-reads/answers", "data/work/2000-digital-zoning",
                                        "data/warp/2000-digital-zoning", "data/solver/lots", "data/solver/page"),
                                 writes=("data/solver/prints",), code=("sfzmaps.solver.prints",))
    # The text detector: one model (`sfzmaps solver text`), trained on every sheet Terra read of the training maps
    # (`paths.TEXT_SHEETS`, fixed lists, so dvc.yaml doesn't depend on what data is there), file by file: of each
    # sheet only its page and Sol's label points, and of the saved answers Terra's boxes and the 2000 sheets' other
    # text passes.  Trained on the whole of each map, it reads no placement, so a warp retrains nothing; the 2000
    # pages it scores are among those it reads.  Its scores (`sfzmaps solver text-scores`) read its model and the test
    # drawings' sheets, with their placements (the long lines are measured in each sheet's feet).  Both cut `runs`
    # too: they take only its folder swap, which decides how their outputs are put in place, never what they are.  The
    # detector's stage also cuts the placements' code (`sfzmaps.transform`, which changes often): it reads no
    # placement, so that code never runs in it, and a change there must not retrain the detector.
    from sfzmaps.solver import paths
    text_cut = (*OUTPUT_NEUTRAL, "sfzmaps.solver.runs")
    terra = "saved-reads/answers/text-on-every-sheet.jsonl.gz"
    out["solver-text"] = Stage(
        "solver-text", (), (), tuple(dict.fromkeys(owner[s] for s in paths.TEXT_TRAIN_SERIES)),
        command=("solver", "text"),
        reads=(*_text_sheet_files("data/work", paths.TEXT_TRAIN_SERIES, ("page.png", "district_labels.json")), terra,
               "saved-reads/answers/experiments-2000/text-positions"),
        writes=("data/solver/text",), code=("sfzmaps.solver.textnet.stage",), cut=(*text_cut, "sfzmaps.transform"))
    out["solver-text-scores"] = Stage(
        "solver-text-scores", (), (), ("solver-text", *(f"warp-{s}" for s in paths.TEXT_TEST_SERIES)),
        command=("solver", "text-scores"),
        reads=("data/solver/text/model.pt",
               *_text_sheet_files("data/work", paths.TEXT_TEST_SERIES, ("page.png", "district_labels.json")),
               *_text_sheet_files("data/warp", paths.TEXT_TEST_SERIES, ("transform.json", "correction.npz")), terra),
        writes=("data/solver/text-scores",), code=("sfzmaps.solver.textnet.score_stage",), cut=text_cut)
    # The code reader's detectors (`sfzmaps solver text-reader`): four, each with a quarter of the city and the
    # answered blocks held out on every training map, by map area.  So they read, beside what the detector's stage
    # reads, each training sheet's placement (the warp's transform.json and correction.npz, file by file), and for the
    # answered blocks what `codereader.labels.load_labels` reads: the prints (prints.parquet and the Terra boxes), the
    # lots (each grid px's nearest lot and each lot's block) and the owner's answers (the code-check cases).
    code_check = "inputs/code-check-2000"
    out["solver-text-reader"] = Stage(
        "solver-text-reader", (), (), ("solver-lots", "solver-prints", *(f"warp-{s}" for s in paths.TEXT_TRAIN_SERIES)),
        command=("solver", "text-reader"),
        reads=(*_text_sheet_files("data/work", paths.TEXT_TRAIN_SERIES, ("page.png", "district_labels.json")),
               *_text_sheet_files("data/warp", paths.TEXT_TRAIN_SERIES, ("transform.json", "correction.npz")), terra,
               "saved-reads/answers/experiments-2000/text-positions", "data/solver/prints/prints.parquet",
               "data/solver/prints/terra_boxes.json", "data/solver/lots/lots.npz", "data/solver/lots/table.parquet",
               f"{code_check}/codecheck_cases_published.json", f"{code_check}/codecheck_answers.json"),
        writes=("data/solver/text-reader",), code=("sfzmaps.solver.textnet.reader_stage",), cut=text_cut)
    # the word boxes (`solver/wordboxes.py`): grouped from the held-out detectors' chance and kind (each sheet's
    # chance.png and kind.npz) on the sheets' pages, placed through the warp's placements, the lots and the mapped
    # area.  The level is chosen and scored on Terra's boxes (so they are read, never as input to the boxes) and the
    # owner's answers are counted (through `codereader.labels.load_labels`, as the reader's stage reads them).
    out["solver-boxes"] = Stage(
        "solver-boxes", (), (), ("solver-lots", "solver-prints", "solver-page", "solver-text-reader"),
        command=("solver", "boxes"),
        reads=(*(f"data/solver/text-reader/{s}/{n}" for s in paths.SHEETS for n in ("chance.png", "kind.npz")),
               *_text_sheet_files("data/work", (paths.SERIES,), ("page.png",)),
               *_text_sheet_files("data/warp", (paths.SERIES,), ("transform.json", "correction.npz",
                                                                  "warped.footprint.geojson")),
               terra, "data/solver/page/sheets.png", "data/solver/prints/prints.parquet",
               "data/solver/prints/terra_boxes.json",
               "data/solver/lots/lots.npz", "data/solver/lots/table.parquet",
               f"{code_check}/codecheck_cases_published.json", f"{code_check}/codecheck_answers.json"),
        writes=("data/solver/boxes",), code=("sfzmaps.solver.wordboxes",), cut=text_cut)
    # The text reader (`solver/textread/`).  Its inputs (`sfzmaps solver text-crops`): crops at Terra's words and
    # non-text windows on the 2000 drawing's three scans (their pages, Sol's points and placements, file by file;
    # Terra's boxes and the other passes' boxes), each with the quarters and the answered blocks it touches (the
    # answered blocks through `codereader.labels.load_labels`, as the code reader's detectors' stage reads them, checked
    # against its held.json, whose unknown radius it takes); the template reader's font and prior learned from the
    # training labels; each quarter's synthetic code crops.  Its training (`text-train`) reads those alone, so a change
    # to the boxes never retrains it.  Its reading (`text-read`) reads them, the networks, the word boxes and the pages.
    reader_maps = ("2000-digital-zoning", "1980-1990-zoning", "1990-1996-zoning")
    out["solver-text-crops"] = Stage(
        "solver-text-crops", (), (), ("solver-lots", "solver-prints", "solver-text-reader",
                                      *(f"warp-{s}" for s in reader_maps)),
        command=("solver", "text-crops"),
        reads=(*_text_sheet_files("data/work", reader_maps, ("page.png", "district_labels.json")),
               *_text_sheet_files("data/warp", reader_maps, ("transform.json", "correction.npz")), terra,
               "saved-reads/answers/experiments-2000/text-positions", "data/solver/prints/prints.parquet",
               "data/solver/prints/terra_boxes.json", "data/solver/lots/lots.npz", "data/solver/lots/table.parquet",
               "data/solver/text-reader/held.json",
               f"{code_check}/codecheck_cases_published.json", f"{code_check}/codecheck_answers.json"),
        writes=("data/solver/text-crops",), code=("sfzmaps.solver.textread.inputs",), cut=text_cut)
    out["solver-text-train"] = Stage(
        "solver-text-train", (), (), ("solver-text-crops",), command=("solver", "text-train"),
        reads=("data/solver/text-crops",), writes=("data/solver/text-train",),
        code=("sfzmaps.solver.textread.train",), cut=text_cut)
    out["solver-text-read"] = Stage(
        "solver-text-read", (), (), ("solver-text-crops", "solver-text-train", "solver-boxes"),
        command=("solver", "text-read"),
        reads=(*(f"data/solver/text-crops/{n}" for n in ("crops.npz", "crops.parquet", "answered.npz",
                                                         "answered.parquet", "prior.json")),
               "data/solver/text-train", "data/solver/boxes/boxes.parquet",
               *_text_sheet_files("data/work", (paths.SERIES,), ("page.png",))),
        writes=("data/solver/text-read",), code=("sfzmaps.solver.textread.read",), cut=text_cut)
    out["solver-labels"] = Stage("solver-labels", (), (), ("solver-prints",), command=("solver", "labels"),
                                 # of the prints, only prints.parquet: the Terra boxes and the summary aren't read
                                 reads=("data/solver/prints/prints.parquet", "data/solver/lots", "data/solver/page",
                                        "data/ref"),
                                 writes=("data/solver/labels",), code=("sfzmaps.solver.labels_stage",))
    # the walls read what `walls.build_walls` reads: the merged page (and its frames), the cut boxes, the lots, the
    # strokes, the Terra boxes, the warp's placements, and for the text discount the detector's chance (stage
    # solver-text's, the default `textgrid` source) and the sheets' pages it is read on.  (The held-out detectors'
    # chance is an experiment's, `walls --chance held-out`, never the stage's.)
    out["solver-walls"] = Stage("solver-walls", (), (), ("solver-labels", "solver-text"),
                                command=("solver", "walls"),
                                reads=("data/solver/page", "inputs/merged-page/2000-digital-zoning.json",
                                       "data/solver/lots", "data/solver/labels/strokes.npz",
                                       "data/solver/prints/terra_boxes.json", "data/solver/text/chance",
                                       "data/work/2000-digital-zoning", "data/warp/2000-digital-zoning"),
                                writes=("data/solver/walls",), code=("sfzmaps.solver.walls",))
    # the solve: the placement, the four held-out solves, the join, the data errors and the report.  Of the inputs
    # folders the hand leader moves and the by-eye corrections the score counts against; the table itself reaches it
    # as the lots' table.parquet
    out["solver-solve"] = Stage("solver-solve", (), (), ("solver-walls",), command=("solver", "solve", "final"),
                                reads=("data/solver/lots", "data/solver/page/ink.png",
                                       "data/solver/prints/prints.parquet", "data/solver/labels", "data/solver/walls",
                                       "data/warp/2000-digital-zoning", "inputs/hand-labels-2000/label_fixes.csv",
                                       "inputs/zoning-2000/table-corrections.csv"),
                                writes=("data/solver/solve",), code=("sfzmaps.solver.solve",))
    # Each map published from its solve (`districts/solved.py`, the series of `solved.SOLVED`): the read's lots with
    # the solve's codes, then unzoned blocks, overrides, heights and parks as for every series.  Not in the edition's or
    # period's stage, which makes the sheets the solver reads (a cycle): that stage reads the zoning (and the heights),
    # and the publish stage writes the published files of its districts folder (`solved.PUBLISHED`,
    # `_period_out_files`).  Last but `shared`, after whatever stages write the series' solve.
    from sfzmaps.districts import solved
    publishers = []
    for s in SERIES:
        if s in solved.SOLVED:
            st = publish_stage(s, out, owner)
            out[st.name] = st
            publishers.append(st.name)
    out["shared"] = Stage("shared", (), (), (*old, *publishers), frozen=True)
    return out


def _height_series(series: str) -> str | None:
    """The period's height series of a zoning series (assemble.height_series_of, without importing the publish)."""
    from sfzmaps.districts.series import SERIES

    period = SERIES[series].period
    return next((h for h, i in SERIES.items() if i.period == period and i.role == "height"), None)


def publish_stage(series: str, table: dict[str, Stage], owner: dict[str, str]) -> Stage:
    """The stage publishing `series` from its solve (`sfzmaps solver publish --series S`; 2000's, `solver-publish`,
    without --series).  It reads what its edition's or period's stage read (districts_raw) and, in a period, the
    heights; the series' overrides; the parks, the parcels and the research table's categories; and the solve's files
    under the series' root (`solved.solve_of`).  Its upstream: refdata, that stage, and every command stage writing one
    of the solve's files.  ValueError when no stage writes one of them (a series put in SOLVED before its solver's
    stages exist)."""
    from sfzmaps.districts import solved
    from sfzmaps.districts.series import SERIES

    s = series
    if SERIES[s].role != "use":
        raise ValueError(f"{s} is not a zoning series: only a zoning map is published from a solve")
    height = _height_series(s)
    # 2000's overrides.csv as its stage was first recorded; another series' inputs folder (tracked for every series),
    # since DVC needs every dep to exist and 1921-1948 have no overrides.csv yet
    overrides = f"inputs/{s}/overrides.csv" if s == solved.SERIES else f"inputs/{s}"
    solve = [p.relative_to(config.PIPELINE_DIR).as_posix() for p in solved.solve_of(s).inputs()]
    reads = (f"data/work/{s}/districts_raw.parquet", f"data/work/{s}/districts_raw.json",
             *((f"data/out/{s}/districts/height.parquet", f"data/out/{s}/districts/height.json") if height else ()),
             overrides, *((f"inputs/{height}",) if height else ()), "inputs/parks.csv", "inputs/park_maps.csv",
             "data/ref/parcels.parquet", "../docs/research/zoning-districts.csv", *solve)
    writers = [n for n, st in table.items() if st.command and any(_overlaps(r, o) for r in solve for o in st.writes)]
    missing = [r for r in solve if not any(_overlaps(r, o) for n in writers for o in table[n].writes)]
    if missing:
        raise ValueError(f"{s} is in districts.solved.SOLVED but no stage writes {', '.join(missing)}: add its "
                         "solver's stages first (they write under solved.default_root)")
    return Stage(solved.stage_name(s), (), (), ("refdata", owner[s], *writers),
                 command=("solver", "publish") if s == solved.SERIES else ("solver", "publish", "--series", s),
                 reads=reads, writes=tuple(f"data/out/{s}/districts/{n}" for n in solved.PUBLISHED),
                 code=("sfzmaps.districts.solved",))


def _overlaps(a: str, b: str) -> bool:
    """Two pipeline paths: the same, or one inside the other."""
    return a == b or a.startswith(b + "/") or b.startswith(a + "/")


def _period_out_files(series: str) -> tuple[str, ...]:
    """What an edition's or period's stage writes in out/<series> of a series published from its solve
    (districts.solved.SOLVED): its tiles, manifest and review page, and in a period the read of its heights in
    districts/; the rest of districts/ is the series' publish stage's (DVC lets no two stages write one folder)."""
    heights = ("districts/height.json", "districts/height.parquet", "districts/height.parquet.answers",
               "districts/height.parquet.inputs.json") if _height_series(series) else ()
    return (f"{series}.pmtiles", f"{series}.inputs.json", f"{series}-paper.pmtiles", f"{series}-paper.inputs.json",
            "changes.wgs84.geojson", "manifest.json", "review.html", *heights)


def _text_sheet_files(root: str, series_list, names: tuple[str, ...]) -> tuple[str, ...]:
    """`root`/<series>/<sheet>/<name> for each series of `series_list`, each sheet Terra read of it
    (`paths.TEXT_SHEETS`) and each of `names`: by series, then by sheet."""
    from sfzmaps.solver import paths
    return tuple(f"{root}/{s}/{sheet}/{name}" for s in series_list for sheet in paths.TEXT_SHEETS[s] for name in names)


def downstream(names: set[str], table: dict[str, Stage]) -> set[str]:
    """names plus every stage that (transitively) depends on one of them."""
    found = set(names)
    changed = True
    while changed:
        changed = False
        for st in table.values():
            if st.name not in found and found.intersection(st.upstream):
                found.add(st.name)
                changed = True
    return found


def upstream(names: set[str], table: dict[str, Stage]) -> set[str]:
    """names plus everything they (transitively) depend on."""
    found, todo = set(), list(names)
    while todo:
        n = todo.pop()
        if n not in found:
            found.add(n)
            todo.extend(table[n].upstream)
    return found


def plan_run(stale: set[str], targets: list[str] | None, table: dict[str, Stage],
             force: bool = False) -> list[str]:
    """Stages to run, in table order: the stale ones and everything downstream of them; with
    targets, only those targets and their upstream (force: the targets even if fresh)."""
    unknown = [t for t in targets or [] if t not in table]
    if unknown:
        raise ValueError(f"unknown stage(s): {', '.join(unknown)}; stages: {', '.join(table)}")
    frozen = {n for n, st in table.items() if st.frozen}
    run = downstream(set(stale) & set(table) - frozen, table) - frozen
    if targets:
        run &= upstream(set(targets), table)
        if force:
            run |= set(targets)
    return [n for n in table if n in run]


# --- dvc.yaml --------------------------------------------------------------------------------

def _code_deps() -> list[str]:
    src = config.PIPELINE_DIR / "src" / "sfzmaps"
    return [f"src/sfzmaps/{p.name}" for p in sorted(src.iterdir()) if p.name not in CODE_EXCLUDE]


def _command_stage_deps(st: Stage) -> list[str]:
    """A command stage's deps: the source files of its entry modules' import closure (without its `cut` modules and
    what only they import), the dependency pins, and its named reads."""
    from sfzmaps import fresh

    files = [p.relative_to(config.PIPELINE_DIR).as_posix()
             for p in fresh.code_files(list(st.code), cut=frozenset(st.cut))]
    return sorted(files) + ["pyproject.toml", "uv.lock", *st.reads]


def stage_deps(st: Stage) -> list[str]:
    if st.command:
        return _command_stage_deps(st)
    if st.refdata:
        return []
    if st.saved_reads:
        return ["saved-reads/images.manifest.tsv"]
    deps = _code_deps() + ["pyproject.toml", "uv.lock", "../pixi.toml", "../pixi.lock"]
    if st.warp:
        return deps + ["data/ref", f"data/work/{st.warp}"]
    if st.shared:
        from sfzmaps.districts.series import SERIES
        # data/ref and park_maps.csv: the check that park_maps.csv still matches the maps
        return deps + [f"data/out/{s}/districts" for s, i in SERIES.items() if i.role == "use"] \
            + ["data/out/today/districts", "data/ref", "inputs/park_maps.csv"]
    deps += ["data/ref"]
    if st.districts:
        deps += ["../docs/research/zoning-districts.csv"]   # the rules: each code's category
    if st.series:
        deps += ["data/raw", "inputs/label_scale.json"] + [f"inputs/{s}" for s in st.series]
    if st.districts:
        deps += ["inputs/parks.csv", "inputs/park_maps.csv"]
    return deps


def stage_outs(st: Stage) -> list[str]:
    if st.command:
        return list(st.writes)
    if st.shared:
        return ["data/out/shared"]
    if st.refdata:
        return ["data/ref"]
    if st.saved_reads:
        return ["data/saved-reads/images"]
    if st.warp:
        return [f"data/warp/{st.warp}"]
    from sfzmaps.districts import solved
    names = st.series or st.districts
    return [o for s in names for o in (
        [f"data/work/{s}", *(f"data/out/{s}/{f}" for f in _period_out_files(s))] if s in solved.SOLVED
        else [f"data/work/{s}", f"data/out/{s}"])]


def dvc_yaml() -> str:
    """pipeline/dvc.yaml, generated (`sfzmaps build --write-dvc-yaml`); a test keeps the
    committed file equal to this."""
    lines = ["# Generated by `sfzmaps build --write-dvc-yaml` from sfzmaps.build; don't edit.",
             "# Run builds with `uv run sfzmaps build` (parallel); `dvc repro` works too (serial).",
             "stages:"]
    for st in stages().values():
        lines += [f"  {st.name}:", f"    cmd: uv run --frozen sfzmaps stage {st.name}"]
        if st.frozen:
            lines += ["    frozen: true"]
        if stage_deps(st):
            lines += ["    deps:"] + [f"      - {d}" for d in stage_deps(st)]
        lines += ["    outs:"]
        for o in stage_outs(st):
            # persist: DVC must never delete a stage's outputs before running it (they are
            # the pipeline's own incremental state)
            lines += [f"      - {o}:", "          persist: true"]
    return "\n".join(lines) + "\n"


def dvc_yaml_path() -> Path:
    return config.PIPELINE_DIR / "dvc.yaml"


# --- sfzmaps stage ---------------------------------------------------------------------------

def stage_commands(name: str) -> list[list[str]]:
    """The pipeline commands a stage runs, in order (each `sfzmaps ...` argv tail)."""
    st = stages()[name]
    if st.command:
        return [list(st.command)]
    if st.warp:
        return [["warp", st.warp]]
    if st.shared:
        return [["districts", "--shared"]]
    if st.refdata:
        return [["refdata"]]
    if st.saved_reads:
        return []                      # checked in run_stage itself (savedreads.check)
    cmds = [["run", "--year", s] for s in st.series] + [["paper", "--year", s] for s in st.series]
    return cmds + [["districts", "--series", s] for s in st.districts]


def _errored_sheets(series: tuple[str, ...]) -> set[str]:
    found = set()
    for s in series:
        for sj in (config.WORK_DIR / s).glob("*/score.json"):
            try:
                reason = str(json.loads(sj.read_text()).get("reason") or "")
            except (OSError, ValueError):
                continue
            if reason.startswith("error"):
                found.add(f"{s}/{sj.parent.name}")
    return found


class StageFailed(Exception):
    def __init__(self, msg: str, paid_read: bool = False):
        super().__init__(msg)
        self.paid_read = paid_read


def _sfzmaps_exe() -> list[str]:
    exe = Path(sys.executable).with_name("sfzmaps")
    return [str(exe)] if exe.exists() else [sys.executable, "-m", "sfzmaps.cli"]


def warp_workers(build_jobs: int) -> int:
    """The sheets one warp stage fits at once when a build runs `build_jobs` stages at once: WARP_WORKERS
    shared between them, at least one each."""
    return max(1, WARP_WORKERS // max(1, build_jobs))


def run_stage(name: str, run: Callable[[list[str], dict], tuple[int, str]] | None = None) -> None:
    """Run one stage's commands in order, at $0 (SFZMAPS_MAX_USD=0). StageFailed when a command
    fails, any sheet of the stage is in error afterwards (run keeps going past a failing sheet
    and exits 0, and a retry must not accept it) or paper skips a series."""
    st = stages()[name]
    if st.saved_reads:
        from sfzmaps import savedreads
        problems = savedreads.check()
        if problems:
            raise StageFailed(f"{len(problems)} saved-read image(s) don't match pipeline/saved-reads/"
                              f"images.manifest.tsv: {'; '.join(problems[:5])}")
        return
    env = {**os.environ, "SFZMAPS_MAX_USD": "0"}

    def default_run(argv: list[str], env: dict) -> tuple[int, str]:
        # stderr (the commands' logging) is passed through line by line as it comes, and kept
        # so the paper "skipped" check can read it
        p = subprocess.Popen(_sfzmaps_exe() + argv, env=env, stderr=subprocess.PIPE, text=True)
        kept = []
        for line in p.stderr:
            sys.stderr.write(line)
            sys.stderr.flush()
            kept.append(line)
        return p.wait(), "".join(kept)

    run = run or default_run
    for argv in stage_commands(name):
        if argv[0] == "warp":
            argv = [*argv, "--jobs", str(warp_workers(int(os.environ.get(BUILD_JOBS_ENV) or 1)))]
        code, err = run(argv, env)
        what = "sfzmaps " + " ".join(argv)
        if code == PAID_READ_EXIT:
            raise StageFailed(f"{what} needs a paid vision read (a build runs at $0). Run it "
                              f"yourself with SFZMAPS_MAX_USD set, then build again.", paid_read=True)
        if code != 0:
            raise StageFailed(f"{what} failed (exit {code})")
        if argv[0] == "paper" and f"{argv[2]}: skipped (" in err:
            raise StageFailed(f"{what} skipped the series: {err.strip().splitlines()[-1]}")
    bad = sorted(_errored_sheets(st.series))
    if bad:
        raise StageFailed(f"{len(bad)} sheet(s) in error: {', '.join(bad[:8])}"
                          f"{' ...' if len(bad) > 8 else ''} (see out/run-*.log)")


# --- sfzmaps build ---------------------------------------------------------------------------

def dvc(*args: str, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(DVC + list(args), cwd=config.PIPELINE_DIR, check=check,
                          capture_output=True, text=True)


def stale_stages(status_json: str) -> set[str]:
    """Stage names from `dvc status --json` (keys may carry a dvc.yaml path prefix)."""
    data = json.loads(status_json or "{}")
    return {k.rsplit(":", 1)[-1] for k in data if not k.endswith(".dvc")}


def still_stale(name: str, run: Callable[..., subprocess.CompletedProcess] | None = None) -> tuple[bool, list[str]]:
    """Whether stage `name` is stale by `dvc status --json NAME` (that stage alone), and its changed deps. A build
    asks this of a stage planned only because an upstream stage was stale, once those have run and been recorded:
    if what the upstream wrote is unchanged where the stage reads it, it needn't run. A status that fails counts as
    stale."""
    p = (run or dvc)("status", "--json", name, check=False)
    if p.returncode != 0:
        return True, []
    return name in stale_stages(p.stdout), changed_deps(p.stdout).get(name, [])


def changed_deps(status_json: str) -> dict[str, list[str]]:
    """Stage name -> its dependencies `dvc status --json` reports changed. A stage's commands get
    this as SFZMAPS_DVC_CHANGED_DEPS, so sfzmaps.fresh can adopt outputs made before fingerprint
    sidecars existed when DVC proves their inputs unchanged."""
    out: dict[str, list[str]] = {}
    for k, items in json.loads(status_json or "{}").items():
        deps = [p for it in items if isinstance(it, dict) for p in (it.get("changed deps") or {})]
        out[k.rsplit(":", 1)[-1]] = sorted(deps)
    return out


def git_dirty(paths: tuple[str, ...] = ("src", "inputs", "dvc.yaml", "pyproject.toml", "uv.lock",
                                        "../pixi.toml", "../pixi.lock")) -> list[str]:
    p = subprocess.run(["git", "status", "--porcelain", "--", *paths], cwd=config.PIPELINE_DIR,
                       check=True, capture_output=True, text=True)
    return [line for line in p.stdout.splitlines() if line.strip()]


def git_head() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=config.PIPELINE_DIR, check=True,
                          capture_output=True, text=True).stdout.strip()


@contextlib.contextmanager
def build_lock(path: Path | None = None):
    """One build per data folder at a time."""
    path = path or config.DATA_DIR / ".build.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit(f"another build holds {path}; wait for it to finish") from None
        yield


def needs_prepare(order: list[str], table: dict[str, Stage]) -> bool:
    """Whether prepare() is needed: only the old pipeline's stages use today's blocks and zoning."""
    return any(not (table[n].warp or table[n].command) for n in order)


def prepare() -> None:
    """Shared prerequisites, made once before any stage runs (stages run in parallel, and each
    would otherwise make them itself): today's blocks for the review pages, today's zoning."""
    from sfzmaps import reviewpage
    from sfzmaps.sources import datasf

    reviewpage.ensure_blocks()
    if not (config.REF_DIR / "zoning_current.parquet").exists():
        datasf.fetch_zoning_current(config.REF_DIR)


@dataclass
class BuildResult:
    ran: list[str]
    failed: dict[str, str]
    skipped: list[str]
    fresh: list[str] = field(default_factory=list)   # up to date after its upstreams: not run

    @property
    def ok(self) -> bool:
        return not self.failed and not self.skipped


def run_build(order: list[str], table: dict[str, Stage], jobs: int, *,
              run_stage: Callable[[str], None], commit: Callable[[str], None],
              guard: Callable[[], str | None] = lambda: None,
              recheck: Callable[[str], bool] | None = None, must_run: set[str] | None = None) -> BuildResult:
    """Run `order` (from plan_run), up to `jobs` at once, each once its upstream stages in
    `order` succeeded; commit each success (one at a time). A failure skips its downstream
    stages. guard() is checked before each commit: a message stops all further commits and
    starts (the inputs changed under the build).

    must_run: the stages that run whatever (stale at the start, or forced); None, every stage in
    `order`. Any other stage is there only because an upstream stage was stale: once its upstream
    stages in `order` are recorded, recheck(name) says whether it is stale now, and if not it
    doesn't run (`fresh`) and counts as done for its own downstream. recheck is called from this
    thread only, between the commits, so it may ask DVC."""
    todo = list(order)
    done: set[str] = set()
    failed: dict[str, str] = {}
    skipped: list[str] = []
    ran: list[str] = []
    fresh: list[str] = []
    stop: str | None = None
    check = set() if must_run is None or recheck is None else set(order) - set(must_run)

    def ready(n: str) -> bool:
        return all(u in done or u not in order for u in table[n].upstream)

    def blocked(n: str) -> bool:
        return any(u in failed or u in skipped for u in table[n].upstream)

    with ThreadPoolExecutor(max_workers=max(1, jobs)) as pool:
        running: dict[Future, str] = {}
        while todo or running:
            for n in [n for n in todo if blocked(n) or stop]:
                todo.remove(n)
                skipped.append(n)
            freed = False
            for n in [n for n in todo if ready(n)]:
                if len(running) >= max(1, jobs):
                    break
                todo.remove(n)
                if n in check and not recheck(n):
                    fresh.append(n)
                    done.add(n)
                    freed = True
                    log.info("build: %s is up to date after its upstreams", n)
                    continue
                running[pool.submit(run_stage, n)] = n
                log.info("build: started %s", n)
            if freed:         # its downstream may be ready now
                continue
            if not running:   # nothing could start: what is left can't run
                skipped.extend(todo)
                break
            finished, _ = wait(list(running), return_when=FIRST_COMPLETED)
            for fut in finished:
                n = running.pop(fut)
                err = fut.exception()
                if err is not None:
                    failed[n] = str(err)
                    log.error("build: %s failed: %s", n, err)
                    continue
                ran.append(n)
                stop = stop or guard()
                if stop:
                    failed[n] = f"not recorded: {stop}"
                    continue
                try:
                    commit(n)
                except Exception as e:  # noqa: BLE001 - reported with the rest of the build
                    failed[n] = f"ran, but recording it failed: {e}"
                    continue
                done.add(n)
                log.info("build: %s done", n)
    return BuildResult(ran, failed, skipped, fresh)


def require_toolchain(system_tools: bool) -> None:
    """Every system tool from the repository's pixi environment (a build records one known
    toolchain: pixi.toml and pixi.lock are deps of every stage), unless system_tools."""
    from sfzmaps import toolchain

    bad = toolchain.not_from_env()
    if not bad:
        return
    names = ", ".join(f"{t.name} ({t.path or 'missing'})" for t in bad)
    if system_tools:
        print(f"WARNING: --system-tools: building with tools outside pixi: {names}")
        return
    raise SystemExit(f"these tools aren't from the pixi environment: {names}\nRun the build "
                     "with `pixi run build`, or inside `pixi shell` (pipeline/README.md, Toolchain), "
                     "or pass --system-tools.")


def build(targets: list[str] | None, jobs: int, dry_run: bool, force: bool,
          system_tools: bool = False) -> int:
    """`sfzmaps build`: exit status."""
    table = stages()
    if not dvc_yaml_path().exists() or dvc_yaml_path().read_text() != dvc_yaml():
        raise SystemExit("pipeline/dvc.yaml is out of date: run `uv run sfzmaps build --write-dvc-yaml`")
    if not dry_run:
        require_toolchain(system_tools)
    dirty = git_dirty()
    if dirty and not dry_run:
        raise SystemExit("uncommitted changes in code or inputs; commit them first, so the build "
                         "records what it was built from:\n  " + "\n  ".join(dirty))
    with build_lock():
        st = dvc("status", "--json", check=False)
        if st.returncode != 0:
            raise SystemExit(f"dvc status failed:\n{st.stderr.strip()}")
        stale = stale_stages(st.stdout)
        order = plan_run(stale, targets, table, force)
        if not order:
            print("nothing to build: every stage is up to date")
            return 0
        # the stages that run whatever; the others are planned only because an upstream stage is stale, and are
        # asked again once it has run
        must_run = (stale & set(order)) | (set(targets or []) if force else set())
        later = [n for n in order if n not in must_run]
        print(f"{'would build' if dry_run else 'building'} {len(order)} stage(s), "
              f"{jobs} at a time: {', '.join(order)}"
              + (f"\n  (of them, {', '.join(later)} only if still stale once their upstream stages have run)"
                 if later else ""))
        if dry_run:
            return 0
        if needs_prepare(order, table):
            prepare()
        head = git_head()
        logs = config.DATA_DIR / "build-logs"
        logs.mkdir(parents=True, exist_ok=True)

        from sfzmaps import fresh
        changed = changed_deps(st.stdout)

        def one(name: str) -> None:
            env = {**os.environ, fresh.CHANGED_DEPS_ENV: json.dumps(changed.get(name, [])),
                   SYSTEM_TOOLS_ENV: "1" if system_tools else "0",
                   BUILD_JOBS_ENV: str(jobs)}
            with open(logs / f"{name}.log", "w") as f:
                p = subprocess.run(["uv", "run", "--frozen", "sfzmaps", "stage", name],
                                   cwd=config.PIPELINE_DIR, stdout=f, stderr=subprocess.STDOUT, env=env)
            if p.returncode != 0:
                tail = (logs / f"{name}.log").read_text().strip().splitlines()[-1:] or [""]
                raise StageFailed(f"exit {p.returncode}: {tail[0][:300]} (log: {logs / (name + '.log')})")

        def guard() -> str | None:
            if git_head() != head:
                return "HEAD moved during the build"
            if git_dirty():
                return "code or inputs changed during the build"
            return None

        def commit(name: str) -> None:
            p = dvc("commit", "--force", name, check=False)
            if p.returncode != 0:
                raise RuntimeError(f"dvc commit: {p.stderr.strip()[-300:]}")

        def recheck(name: str) -> bool:
            again, deps = still_stale(name)
            changed[name] = deps
            return again

        t0 = time.monotonic()
        res = run_build(order, table, jobs, run_stage=one, commit=commit, guard=guard, recheck=recheck,
                        must_run=must_run)
        print(f"built {len(res.ran)} stage(s) in {(time.monotonic() - t0) / 60:.1f} min")
        for n in res.fresh:
            print(f"up to date after its upstreams: {n}")
        for n, msg in res.failed.items():
            print(f"FAILED  {n}: {msg}")
        for n in res.skipped:
            print(f"skipped {n} (an upstream stage failed or the build stopped)")
        return 0 if res.ok else 1
