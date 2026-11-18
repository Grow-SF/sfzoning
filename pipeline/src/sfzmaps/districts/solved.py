"""A zoning map's published codes from its series' solver solve (the `solver-publish*` stages, `sfzmaps solver publish
--series S`).

A series' solve sits under its root, in the 2000 solver's layout (`Solve`): its code for each of its lots
(<root>/solve/runs/final.npz, in the row order of <root>/lots/table.parquet), its report (<root>/solve/report.json) and
its lots' facts (<root>/lots/lots.npz).  2000's root is data/solver, its lots the April 2000 table's; another series'
is data/solver/<series> (`default_root`), or any folder given with `--root`.  The map is published on DataSF's lots of
the series' year (`read.load_parcels`), the lots its district read covers (districts_raw.parquet).  Each of those lots
takes:

  - the code of the solve's lots covering the largest share of its area, their overlaps summed per code.  The solve's
    lots are taken row by row, never looked up by blklot: the 45 rows of the 2000 table whose blklot another row also
    has are each their own polygon with their own solved code;
  - where no solve lot overlaps it (vacated streets, Mission Bay's new lots), the code of the nearest one;
  - in the one spelling (`solver.codes.canon`), with " NCD" put back on a named Neighborhood Commercial District (the
    2000 table writes 'UPR MARKET', the research table knows 'UPPER MARKET NCD'; a solve that writes 'NOE VALLEY NCD'
    keeps it), so every code has its category in the series' era (`categories.category_for` at the series' year, as
    the old reader's codes get theirs) and unknown_codes.json stays empty.  A code with no category stops the publish.

Solve lots the solve left blank take part in none of this.  A lot's confidence (`confidence_source`):
  - "score": the solve's measured share of scored lots right (report.json's every-lot wrong lots over lots.npz's
    scored lots), the same for every lot, wherever the solve was scored (2000, on the April 2000 table);
  - "posterior": otherwise the solve's own posterior, final.npz's `posterior` (the chance of the code it gave each of
    its lots), each published lot taking the area-weighted mean over the solve lots that give it its code (the
    nearest one's for a lot no solve lot overlaps);
  - neither: the publish stops (no confidence is made up).
Its height code and sheet stay the read's, and the series' other steps (unzoned blocks, overrides, heights, parks)
apply as for every series (`assemble.publish`).

A series' edition or period stage makes the sheets its solver reads, so it can't also read the solve (a cycle): for a
series of SOLVED it reads the zoning and the heights and leaves the published files (`PUBLISHED`) to the series'
publish stage (`stage_name`)."""
from __future__ import annotations

import json
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from sfzmaps.solver import paths, runs
from sfzmaps.solver.codes import canon

SERIES = "2000-digital-zoning"     # the first series published from a solve, and `solver publish`'s default
# The zoning series published from a solve.  Each has a publish stage (`stage_name`) and its edition or period stage
# hands its zoning publish over to it (build.stages).  Add a series once its solver's stages write its root
# (`default_root`), then `sfzmaps build --write-dvc-yaml`; the build's tests check the wiring.
SOLVED = frozenset({SERIES})
# what a publish stage writes in out/<series>/districts/: assemble.PUBLISHED and the publish's fingerprint
PUBLISHED = ("parcels.fgb", "districts.fgb", "stats.json", "stats.json.inputs.json", "unknown_codes.json")
# The named Neighborhood Commercial Districts of the April 2000 table, in the one spelling: each is published as
# '<name> NCD', the spelling the research table (and the read) gives them.
NAMED_NCDS = frozenset({
    "24TH MISSION", "24TH-NOE VALLEY", "BROADWAY", "CASTRO", "HAIGHT", "HAYES-GOUGH", "INNER CLEMENT", "NORTH BEACH",
    "OUTER CLEMENT", "POLK", "SACRAMENTO", "UNION", "UPPER FILLMORE", "UPPER MARKET", "VALENCIA", "WEST PORTAL AVENUE",
})
# how a stats.json provenance names a series' solver ("the 2000 solver's run final"); otherwise the series' name
LABELS = {SERIES: "2000"}

_ROOTS: dict[str, Path | None] = {}       # `rooted`: series published from a solve for the length of a block


def stage_name(series: str) -> str:
    """The publish stage of a series of SOLVED: 2000's is `solver-publish`, any other's `solver-publish-<series>`."""
    return "solver-publish" if series == SERIES else f"solver-publish-{series}"


def default_root(series: str) -> Path:
    """Where a series' solver writes, in 2000's layout: data/solver for 2000, data/solver/<series> for another."""
    return paths.SOLVER_DIR if series == SERIES else paths.SOLVER_DIR / series


@dataclass(frozen=True)
class Solve:
    """A series' solve: its folder (runs/final.npz, report.json) and its lots' (table.parquet, lots.npz)."""
    series: str
    solve_dir: Path
    lots_dir: Path

    def run_file(self) -> Path:
        return self.solve_dir / "runs" / f"{runs.FINAL_RUN}.npz"

    def report_file(self) -> Path:
        return self.solve_dir / "report.json"

    def table_file(self) -> Path:
        return self.lots_dir / "table.parquet"

    def lots_file(self) -> Path:
        return self.lots_dir / "lots.npz"

    def inputs(self) -> list[Path]:
        """The solver's files a publish reads."""
        return [self.run_file(), self.report_file(), self.table_file(), self.lots_file()]


def solve_of(series: str = SERIES, root: Path | str | None = None) -> Solve:
    """A series' solve: under `root` if given (or set by `rooted`), else 2000's stage folders (runs.SOLVE_DIR,
    paths.LOTS_DIR) for 2000 and `default_root` for any other series."""
    root = root if root is not None else _ROOTS.get(series)
    if root is not None:
        return Solve(series, Path(root) / "solve", Path(root) / "lots")
    if series == SERIES:
        return Solve(series, runs.SOLVE_DIR, paths.LOTS_DIR)
    r = default_root(series)
    return Solve(series, r / "solve", r / "lots")


def is_solved(series: str) -> bool:
    """A series published from its solve: one of SOLVED, or one `rooted` for the moment."""
    return series in SOLVED or series in _ROOTS


@contextmanager
def rooted(series: str, root: Path | str | None = None):
    """For the length of the block, publish `series` from the solve under `root` (its default when None), whether or
    not it is one of SOLVED: `sfzmaps solver publish --series S [--root DIR]` by hand, and the page of differences."""
    had, old = series in _ROOTS, _ROOTS.get(series)
    _ROOTS[series] = Path(root) if root is not None else None
    try:
        yield solve_of(series)
    finally:
        if had:
            _ROOTS[series] = old
        else:
            _ROOTS.pop(series, None)


def run_file(series: str = SERIES) -> Path:
    """The solve's code per solve lot: the series' solve stage's run."""
    return solve_of(series).run_file()


def report_file(series: str = SERIES) -> Path:
    return solve_of(series).report_file()


def table_file(series: str = SERIES) -> Path:
    return solve_of(series).table_file()


def lots_file(series: str = SERIES) -> Path:
    return solve_of(series).lots_file()


def inputs(series: str = SERIES) -> list[Path]:
    """The solver's files a publish of `series` reads."""
    return solve_of(series).inputs()


def solved_codes(series: str = SERIES) -> np.ndarray:
    """The solve's code per solve lot, in the one spelling ("" where it gave none)."""
    R = np.load(run_file(series))
    return np.array([canon(x) for x in R["names"].astype(str)], dtype=object)[R["pred"]].astype(str)


def score(series: str = SERIES) -> float | None:
    """The solve's share of scored lots right, every lot counted: 1 - report.json's every-lot wrong lots / lots.npz's
    scored lots.  None where the solve has no score: no report, no every-lot wrong lots in it, or no scored lot."""
    s = solve_of(series)
    if not s.report_file().is_file():
        return None
    wrong = json.loads(s.report_file().read_text()).get("measures", {}).get("every_lot", {}).get("wrong_lots")
    with np.load(s.lots_file()) as z:
        scored = int(z["scored"].sum()) if "scored" in z.files else 0
    if wrong is None or scored == 0:
        return None
    return round(1.0 - wrong / scored, 3)


def accuracy() -> float:
    """2000's solve's share of scored lots right (`score`)."""
    return score(SERIES)


def posterior(series: str = SERIES) -> np.ndarray | None:
    """The solve's own posterior per solve lot (final.npz `posterior`: the chance of the code it gave the lot), or None
    if the run has none.  ValueError for one of another length, or out of [0, 1] where the solve gave a code."""
    with np.load(run_file(series)) as R:
        if "posterior" not in R.files:
            return None
        p, pred, names = R["posterior"].astype(float), R["pred"], R["names"].astype(str)
    if p.shape != pred.shape:
        raise ValueError(f"{run_file(series)}: posterior has {p.shape} values and pred {pred.shape}")
    coded = np.array([canon(x) for x in names], dtype=object)[pred] != ""
    if not (np.isfinite(p[coded]).all() and ((p[coded] >= 0) & (p[coded] <= 1)).all()):
        raise ValueError(f"{run_file(series)}: posterior outside [0, 1] where the solve gave a code")
    return p


def confidence_source(series: str = SERIES) -> str:
    """'score' where the solve was scored, else 'posterior' where the run gives one; ValueError otherwise."""
    if score(series) is not None:
        return "score"
    if posterior(series) is not None:
        return "posterior"
    s = solve_of(series)
    raise ValueError(f"{series}: the solve has no score ({s.report_file()} every_lot.wrong_lots over {s.lots_file()} "
                     f"scored lots) and no posterior ({s.run_file()} `posterior`), so no confidence to publish")


def provenance(series: str = SERIES) -> dict:
    """Where the published codes come from, for stats.json: the solve's run (its sha256) and the confidence given (the
    score's share right; or that it is the posterior per lot, and its mean over the solve's coded lots)."""
    from sfzmaps import fresh

    out = {"source": f"the {LABELS.get(series, series)} solver's run {runs.FINAL_RUN}",
           "run_sha256": fresh.sha256(run_file(series))}
    if confidence_source(series) == "score":
        out["confidence"] = score(series)
    else:
        p = posterior(series)
        out["confidence_from"] = "posterior"
        out["mean_confidence"] = round(float(np.nanmean(p[solved_codes(series) != ""])), 3)
    return out


def published_code(name: str, year: int) -> str | None:
    """The published spelling of a solved code: the one spelling, a named NCD with ' NCD' after it (one of the 2000
    table's NAMED_NCDS, or a code written with ' NCD').  None for no code; ValueError for a code the research table
    can't place in the series' era (it would land in unknown_codes.json)."""
    from sfzmaps.districts.categories import category_for
    from sfzmaps.solver.codes import tidy

    c = canon(name)
    if not c:
        return None
    if c in NAMED_NCDS or tidy(name).split(" & ")[0].strip().endswith(" NCD"):
        c = f"{c} NCD"
    if category_for(c, year) == "unknown":
        raise ValueError(f"the solved code {c!r} has no published spelling: the research table has no category for it "
                         f"in {year}")
    return c


def majority_codes(lots, table, codes) -> tuple[np.ndarray, np.ndarray]:
    """Per lot (shapely geometries, None where a lot has none): the code of the table lots (`table` geometries,
    `codes` per row, "" for none) covering the largest share of it, their overlaps summed per code (a tie goes to the
    code first in sort order), or with no overlap the code of the nearest table lot; and how ('area', 'nearest', or
    '' for a lot with no geometry).  Only the table lots with a code take part."""
    code, how, _ = _majority(lots, table, codes, None)
    return code, how


def _majority(lots, table, codes, values) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """`majority_codes`, and with `values` (one per table row) each lot's value for its code: the mean over the table
    lots giving it that code, weighted by their overlap (the nearest lot's own; NaN for a lot with no geometry)."""
    import shapely
    from shapely import STRtree

    lots = np.asarray(lots, dtype=object)
    codes = np.asarray(codes).astype(str)
    keep = codes != ""
    tg, tc = _valid(np.asarray(table, dtype=object)[keep]), codes[keep]
    tv = None if values is None else np.asarray(values, float)[keep]
    has = np.array([isinstance(g, shapely.Geometry) and not shapely.is_empty(g) for g in lots], bool)
    idx = np.flatnonzero(has)
    lg = _valid(lots[idx])
    tree = STRtree(tg)
    li, ti = tree.query(lg, predicate="intersects")
    area = shapely.area(shapely.intersection(lg[li], tg[ti]))
    ov = pd.DataFrame({"lot": idx[li], "code": tc[ti], "area": area})
    if tv is None:
        ov = ov[ov["area"] > 0].groupby(["lot", "code"], as_index=False)["area"].sum()
    else:
        ov["weighted"] = area * tv[ti]
        ov = ov[ov["area"] > 0].groupby(["lot", "code"], as_index=False)[["area", "weighted"]].sum()
    best = ov.sort_values(["lot", "area", "code"], ascending=[True, False, True]).drop_duplicates("lot")
    out = np.full(len(lots), "", dtype=object)
    how = np.full(len(lots), "", dtype=object)
    val = None if tv is None else np.full(len(lots), np.nan)
    out[best["lot"].to_numpy()] = best["code"].to_numpy()
    how[best["lot"].to_numpy()] = "area"
    if val is not None:
        val[best["lot"].to_numpy()] = (best["weighted"] / best["area"]).to_numpy()
    rest = np.flatnonzero(has & (how == ""))
    if len(rest):
        qi, qt = tree.query_nearest(_valid(lots[rest]), all_matches=False)
        out[rest[qi]] = tc[qt]
        how[rest[qi]] = "nearest"
        if val is not None:
            val[rest[qi]] = tv[qt]
    return out.astype(str), how.astype(str), val


def _valid(geoms: np.ndarray) -> np.ndarray:
    import shapely

    geoms = np.asarray(geoms, dtype=object)
    bad = ~shapely.is_valid(geoms)
    if bad.any():
        geoms = geoms.copy()
        geoms[bad] = shapely.make_valid(geoms[bad])
    return geoms


def solved_read(raw: pd.DataFrame, parcels, year: int, series: str = SERIES) -> pd.DataFrame:
    """The read `raw` (read.COLUMNS) with each lot's code from the series' solve (`majority_codes` on the lots of
    `parcels`, published spelling at `year`) and its confidence (`confidence_source`); its height codes and sheets
    kept."""
    import geopandas as gpd

    s = solve_of(series)
    table = gpd.read_parquet(s.table_file())
    codes = solved_codes(series)
    if len(codes) != len(table):
        how = ("`sfzmaps solver solve final` after `sfzmaps solver lots`" if series == SERIES else
               f"{series}'s solve after its lots")
        raise ValueError(f"{s.run_file()} codes {len(codes):,} lots and {s.table_file()} has {len(table):,}: rerun "
                         f"{how}")
    source = confidence_source(series)
    geom = parcels.drop_duplicates("blklot").set_index("blklot").geometry
    lots = geom.reindex(raw["blklot"].astype(str)).to_numpy()
    code, _, post = _majority(lots, table.geometry.to_numpy(), codes,
                              posterior(series) if source == "posterior" else None)
    spelled = {c: published_code(c, year) for c in set(code) if c}
    out = raw.copy()
    out["code"] = pd.Series([spelled.get(c) for c in code], index=out.index, dtype=object)
    given = score(series) if source == "score" else post
    out["confidence"] = np.where(out["code"].notna(), given, 0.0)
    return out


def publish(series: str = SERIES, *, force: bool = False, root: Path | str | None = None) -> Path:
    """Publish `series` from its solve (under `root`, else its own; `assemble.republish`): out/<series>/districts/,
    unless current."""
    from sfzmaps.districts import assemble

    with rooted(series, root):
        return assemble.republish(series, force=force)
