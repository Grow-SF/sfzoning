"""Tracked builds: the stage table, pipeline/dvc.yaml, `sfzmaps stage` and `sfzmaps build`.

DVC (pipeline/dvc.yaml + dvc.lock, committed) records the content hash of every input and output
of every stage, so a checkout can tell whether its data matches its code and inputs. DVC runs
stages one at a time, so `sfzmaps build` runs them itself, up to --jobs at once in dependency
order, and asks DVC only what is stale (`dvc status`) and to record each success (`dvc commit`).

A stage is the smallest unit whose writes don't cross another's (DVC refuses two stages writing
into one tracked folder, and a height series' districts write into its zoning series' out/):

  edition-<E>   one series (1921 ... 1980): run, paper, districts
  period-<P>    a period's zoning, height and special-use series (1980-1990 ...): run and paper
                for each, then districts for height before zoning (zoning reads the heights)
  today         districts for today's DataSF zoning
  shared        the merged archive the site reads (districts --shared)

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
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from sfzmaps import config

log = logging.getLogger(__name__)

# DVC runs pinned, outside the project environment: upgrading it must not change uv.lock (a dep
# of every stage) and so restale everything.
DVC = ["uv", "tool", "run", "--from", "dvc==3.67.1", "dvc"]
# Code that can't change a build output: the editor, the build driver, bundling, evaluation,
# shipping (ship, cachepack, datapack, release), the toolchain check and the GIS downloads.
CODE_EXCLUDE = {"editor", "build.py", "share.py", "evaluate.py", "ship.py", "cachepack.py",
                "datapack.py", "release.py", "toolchain.py", "downloads", "savedreads.py", "__pycache__"}
# The exit code of a pipeline command stopped by llm.FATAL (the $0 cap, or no API key): "needs a
# paid read", not a bug. Same value as cli.PAID_READ_EXIT (a test keeps them equal).
PAID_READ_EXIT = 75


@dataclass(frozen=True)
class Stage:
    name: str
    series: tuple[str, ...]      # run and paper, in this order
    districts: tuple[str, ...]   # districts --series, in this order
    upstream: tuple[str, ...] = ()
    frozen: bool = False         # never run by a build or dvc repro; only `build --force NAME`

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
        out[name] = Stage(name, tuple(members), tuple(reads), ("refdata",))
    out["today"] = Stage("today", (), ("today",), ("refdata",))
    out["shared"] = Stage("shared", (), (), tuple(n for n in out if n != "saved-reads"))
    return out


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


def stage_deps(st: Stage) -> list[str]:
    if st.refdata:
        return []
    if st.saved_reads:
        return ["saved-reads/images.manifest.tsv"]
    deps = _code_deps() + ["pyproject.toml", "uv.lock", "../pixi.toml", "../pixi.lock"]
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
    if st.shared:
        return ["data/out/shared"]
    if st.refdata:
        return ["data/ref"]
    if st.saved_reads:
        return ["data/saved-reads/images"]
    names = st.series or st.districts
    return [f"data/{kind}/{s}" for s in names for kind in ("work", "out")]


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

    @property
    def ok(self) -> bool:
        return not self.failed and not self.skipped


def run_build(order: list[str], table: dict[str, Stage], jobs: int, *,
              run_stage: Callable[[str], None], commit: Callable[[str], None],
              guard: Callable[[], str | None] = lambda: None) -> BuildResult:
    """Run `order` (from plan_run), up to `jobs` at once, each once its upstream stages in
    `order` succeeded; commit each success (one at a time). A failure skips its downstream
    stages. guard() is checked before each commit: a message stops all further commits and
    starts (the inputs changed under the build)."""
    todo = list(order)
    done: set[str] = set()
    failed: dict[str, str] = {}
    skipped: list[str] = []
    ran: list[str] = []
    stop: str | None = None

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
            for n in [n for n in todo if ready(n)][: max(0, jobs - len(running))]:
                todo.remove(n)
                running[pool.submit(run_stage, n)] = n
                log.info("build: started %s", n)
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
    return BuildResult(ran, failed, skipped)


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
        order = plan_run(stale_stages(st.stdout), targets, table, force)
        if not order:
            print("nothing to build: every stage is up to date")
            return 0
        print(f"{'would build' if dry_run else 'building'} {len(order)} stage(s), "
              f"{jobs} at a time: {', '.join(order)}")
        if dry_run:
            return 0
        prepare()
        head = git_head()
        logs = config.DATA_DIR / "build-logs"
        logs.mkdir(parents=True, exist_ok=True)

        from sfzmaps import fresh
        changed = changed_deps(st.stdout)

        def one(name: str) -> None:
            env = {**os.environ, fresh.CHANGED_DEPS_ENV: json.dumps(changed.get(name, []))}
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

        t0 = time.monotonic()
        res = run_build(order, table, jobs, run_stage=one, commit=commit, guard=guard)
        print(f"built {len(res.ran)} stage(s) in {(time.monotonic() - t0) / 60:.1f} min")
        for n, msg in res.failed.items():
            print(f"FAILED  {n}: {msg}")
        for n in res.skipped:
            print(f"skipped {n} (an upstream stage failed or the build stopped)")
        return 0 if res.ok else 1
