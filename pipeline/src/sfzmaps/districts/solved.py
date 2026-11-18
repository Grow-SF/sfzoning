"""The 2000 map's published codes from the 2000 solver's solve (the `solver-publish` stage, `sfzmaps solver publish`).

The solve gives a code to each lot of the April 2000 table (`runs.SOLVE_DIR`/runs/final.npz, in the row order of
`paths.LOTS_DIR`/table.parquet).  The map is published on DataSF's lots of 2000 (`read.load_parcels`), the lots its
district read covers (districts_raw.parquet).  Each of those lots takes:

  - the code of the table lots covering the largest share of its area, their overlaps summed per code.  The table's
    rows are taken one by one, never looked up by blklot: the 45 rows whose blklot another row also has are each
    their own polygon with their own solved code;
  - where no table lot overlaps it (vacated streets, Mission Bay's new lots), the code of the nearest table lot;
  - in the one spelling (`solver.codes.canon`), with " NCD" put back on a named Neighborhood Commercial District (the
    table writes 'UPR MARKET', the research table knows 'UPPER MARKET NCD'), so every code has a category and
    unknown_codes.json stays empty.  A code with no published spelling stops the publish.

Table lots the solve left blank take part in none of this.  A lot's confidence is the solve's measured share of
scored lots it gets right (report.json's every-lot wrong lots over lots.npz's scored lots), its height code and sheet
stay the read's, and the period's other steps (unzoned blocks, overrides, heights, parks) apply as for every series
(`assemble.publish`).

The period's own stage makes the sheets the solver reads, so it can't also read the solve (a cycle): it reads the
zoning and the heights and leaves this series' published files (`PUBLISHED`) to `solver-publish`."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from sfzmaps.solver import paths, runs
from sfzmaps.solver.codes import canon

SERIES = "2000-digital-zoning"
SOLVED = frozenset({SERIES})       # the zoning series published from the solve
# what `solver-publish` writes in out/<series>/districts/: assemble.PUBLISHED and the publish's fingerprint
PUBLISHED = ("parcels.fgb", "districts.fgb", "stats.json", "stats.json.inputs.json", "unknown_codes.json")
# The named Neighborhood Commercial Districts of the April 2000 table, in the one spelling: each is published as
# '<name> NCD', the spelling the research table (and the read) gives them.
NAMED_NCDS = frozenset({
    "24TH MISSION", "24TH-NOE VALLEY", "BROADWAY", "CASTRO", "HAIGHT", "HAYES-GOUGH", "INNER CLEMENT", "NORTH BEACH",
    "OUTER CLEMENT", "POLK", "SACRAMENTO", "UNION", "UPPER FILLMORE", "UPPER MARKET", "VALENCIA", "WEST PORTAL AVENUE",
})


def run_file() -> Path:
    """The solve's code per table lot: the `solver-solve` stage's run."""
    return runs.SOLVE_DIR / "runs" / f"{runs.FINAL_RUN}.npz"


def report_file() -> Path:
    return runs.SOLVE_DIR / "report.json"


def table_file() -> Path:
    return paths.LOTS_DIR / "table.parquet"


def lots_file() -> Path:
    return paths.LOTS_DIR / "lots.npz"


def inputs() -> list[Path]:
    """The solver's files a publish reads."""
    return [run_file(), report_file(), table_file(), lots_file()]


def solved_codes() -> np.ndarray:
    """The solve's code per table lot, in the one spelling ("" where it gave none)."""
    R = np.load(run_file())
    return np.array([canon(x) for x in R["names"].astype(str)], dtype=object)[R["pred"]].astype(str)


def accuracy() -> float:
    """The solve's share of scored lots right, every lot counted: 1 - wrong lots / scored lots."""
    wrong = json.loads(report_file().read_text())["measures"]["every_lot"]["wrong_lots"]
    scored = int(np.load(lots_file())["scored"].sum())
    return round(1.0 - wrong / scored, 3)


def provenance() -> dict:
    """Where the published codes come from, for stats.json: the solve's run (its sha256) and the confidence given."""
    from sfzmaps import fresh

    return {"source": f"the 2000 solver's run {runs.FINAL_RUN}", "run_sha256": fresh.sha256(run_file()),
            "confidence": accuracy()}


def published_code(name: str, year: int) -> str | None:
    """The published spelling of a solved code: the one spelling, a named NCD with ' NCD' after it.  None for no code;
    ValueError for a code the research table can't place (it would land in unknown_codes.json)."""
    from sfzmaps.districts.categories import category_for

    c = canon(name)
    if not c:
        return None
    if c in NAMED_NCDS:
        return f"{c} NCD"
    if category_for(c, year) == "unknown":
        raise ValueError(f"the solved code {c!r} has no published spelling: the research table has no category for it")
    return c


def majority_codes(lots, table, codes) -> tuple[np.ndarray, np.ndarray]:
    """Per lot (shapely geometries, None where a lot has none): the code of the table lots (`table` geometries,
    `codes` per row, "" for none) covering the largest share of it, their overlaps summed per code (a tie goes to the
    code first in sort order), or with no overlap the code of the nearest table lot; and how ('area', 'nearest', or
    '' for a lot with no geometry).  Only the table lots with a code take part."""
    import shapely
    from shapely import STRtree

    lots = np.asarray(lots, dtype=object)
    codes = np.asarray(codes).astype(str)
    keep = codes != ""
    tg, tc = _valid(np.asarray(table, dtype=object)[keep]), codes[keep]
    has = np.array([isinstance(g, shapely.Geometry) and not shapely.is_empty(g) for g in lots], bool)
    idx = np.flatnonzero(has)
    lg = _valid(lots[idx])
    tree = STRtree(tg)
    li, ti = tree.query(lg, predicate="intersects")
    area = shapely.area(shapely.intersection(lg[li], tg[ti]))
    ov = pd.DataFrame({"lot": idx[li], "code": tc[ti], "area": area})
    ov = ov[ov["area"] > 0].groupby(["lot", "code"], as_index=False)["area"].sum()
    best = ov.sort_values(["lot", "area", "code"], ascending=[True, False, True]).drop_duplicates("lot")
    out = np.full(len(lots), "", dtype=object)
    how = np.full(len(lots), "", dtype=object)
    out[best["lot"].to_numpy()] = best["code"].to_numpy()
    how[best["lot"].to_numpy()] = "area"
    rest = np.flatnonzero(has & (how == ""))
    if len(rest):
        qi, qt = tree.query_nearest(_valid(lots[rest]), all_matches=False)
        out[rest[qi]] = tc[qt]
        how[rest[qi]] = "nearest"
    return out.astype(str), how.astype(str)


def _valid(geoms: np.ndarray) -> np.ndarray:
    import shapely

    geoms = np.asarray(geoms, dtype=object)
    bad = ~shapely.is_valid(geoms)
    if bad.any():
        geoms = geoms.copy()
        geoms[bad] = shapely.make_valid(geoms[bad])
    return geoms


def solved_read(raw: pd.DataFrame, parcels, year: int) -> pd.DataFrame:
    """The read `raw` (read.COLUMNS) with each lot's code from the solve (`majority_codes` on the lots of `parcels`,
    published spelling) and the solve's measured confidence; its height codes and sheets kept."""
    import geopandas as gpd

    table = gpd.read_parquet(table_file())
    codes = solved_codes()
    if len(codes) != len(table):
        raise ValueError(f"{run_file()} codes {len(codes):,} lots and {table_file()} has {len(table):,}: rerun "
                         "`sfzmaps solver solve final` after `sfzmaps solver lots`")
    geom = parcels.drop_duplicates("blklot").set_index("blklot").geometry
    lots = geom.reindex(raw["blklot"].astype(str)).to_numpy()
    code, _ = majority_codes(lots, table.geometry.to_numpy(), codes)
    spelled = {c: published_code(c, year) for c in set(code) if c}
    out = raw.copy()
    out["code"] = pd.Series([spelled.get(c) for c in code], index=out.index, dtype=object)
    out["confidence"] = np.where(out["code"].notna(), accuracy(), 0.0)
    return out


def publish(force: bool = False) -> Path:
    """Publish SERIES from the solve (`assemble.republish`): out/<series>/districts/, unless current."""
    from sfzmaps.districts import assemble

    return assemble.republish(SERIES, force=force)
