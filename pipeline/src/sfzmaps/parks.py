"""Which lots are parks: the candidates behind inputs/parks.csv, found from the reference data.

  ref/rec_park_properties.parquet   DataSF Recreation and Parks Properties (gtr9-ntp6): city parks
  ref/nps_tracts.parquet            NPS land tracts in San Francisco, each with its conveyance date

A lot is a park's when at least half of it lies inside that park's outline. City parks are grouped
by name (Golden Gate Park's seven sections are one park); libraries, offices, the rowing club
concession and the properties outside the city are left out. A federal tract counts only where
the NPS owns it outright (fee) and today's zoning is P, which leaves out the VA hospital, the
Coast Guard, and houses and a hotel inside the recreation area's boundary.

`sfzmaps parks` writes these candidates as CSV. inputs/parks.csv is chosen from them by hand, with
the year each became a park (`since`) and where that year comes from; a candidate with no
documented year stays out of it.

The second rule needs no date (owner's decision, 2026-11-02): a park lot that a zoning map shows
as public is a park on every later map. `sfzmaps parks --maps` writes inputs/park_maps.csv, the
first map showing each candidate lot as public (`first_maps`), from the published maps. The
district step applies both rules (assemble.park_starts); the shared step fails if the file no
longer matches the maps (`check_first_maps`).
"""
from __future__ import annotations

import logging
import re
from datetime import date
from pathlib import Path

import geopandas as gpd
import pandas as pd

from sfzmaps import config

log = logging.getLogger(__name__)

MIN_SHARE = 0.5
# Rec and Park property types that aren't parks, and properties outside San Francisco
NOT_PARK_TYPES = {"Library", "Other Non-Park Property", "Concession", "Family Camp"}
OUTSIDE_CITY = {"Sharp Park", "Camp Mather"}


def park_name(property_name: str) -> str:
    """'Golden Gate Park - Section 3' -> 'Golden Gate Park'."""
    return re.sub(r"\s*-\s*Section \d+$", "", property_name.strip())


def _lots_in(parcels: gpd.GeoDataFrame, outlines: gpd.GeoDataFrame) -> pd.DataFrame:
    """(blklot, name, share) for each lot at least MIN_SHARE inside one name's outlines; a lot
    inside two goes to the one covering more of it."""
    shapes = outlines.dissolve(by="name").reset_index()[["name", "geometry"]]
    x = gpd.overlay(parcels[["blklot", "geometry"]], shapes, how="intersection", keep_geom_type=True)
    x["covered"] = x.geometry.area
    x = x.groupby(["blklot", "name"], as_index=False)["covered"].sum()
    area = parcels.set_index("blklot").geometry.area
    x["share"] = x["covered"] / x["blklot"].map(area)
    x = x[x["share"] >= MIN_SHARE].sort_values("share", ascending=False).drop_duplicates("blklot")
    return x[["blklot", "name", "share"]]


def candidates(parcels: gpd.GeoDataFrame | None = None) -> pd.DataFrame:
    """Every park lot the reference data shows: name, blklot, source, year (the NPS conveyance
    year; none for city parks, whose dataset has no dates), acres."""
    if parcels is None:
        parcels = gpd.read_parquet(config.REF_DIR / "parcels.parquet")
        parcels = parcels[parcels["active"]]
    parcels = parcels[parcels.geometry.notna() & ~parcels.geometry.is_empty].drop_duplicates("blklot")

    rp = gpd.read_parquet(config.REF_DIR / "rec_park_properties.parquet")
    rp = rp[~rp["propertytype"].isin(NOT_PARK_TYPES) & ~rp["property_name"].isin(OUTSIDE_CITY)
            & ~rp["property_name"].str.contains("Library")]
    city = _lots_in(parcels, rp.assign(name=rp["property_name"].map(park_name)))
    city["source"] = "Rec and Park properties"
    city["year"] = pd.NA

    nps = gpd.read_parquet(config.REF_DIR / "nps_tracts.parquet")
    nps = nps[nps["Interest"] == "Federal Land (Fee)"].copy()
    nps["name"] = nps["ALPHA_TRACT"]
    fed = _lots_in(parcels, nps)
    fed = fed[~fed["blklot"].isin(city["blklot"])]
    # today's zoning over most of the lot: P, or none at all (Aquatic Park's lot is mostly water)
    zoning = gpd.read_parquet(config.REF_DIR / "zoning_current.parquet")
    z = gpd.overlay(parcels[parcels["blklot"].isin(fed["blklot"])][["blklot", "geometry"]],
                    zoning[["zoning_sim", "geometry"]], how="intersection", keep_geom_type=True)
    z["a"] = z.geometry.area
    main = z.sort_values("a", ascending=False).drop_duplicates("blklot").set_index("blklot")["zoning_sim"]
    fed = fed[fed["blklot"].map(main).fillna("P") == "P"]
    convey = pd.to_datetime(nps.set_index("ALPHA_TRACT")["ConveyDate"], unit="ms").dt.year
    fed["year"] = fed["name"].map(convey).astype("Int64")
    fed["source"] = "NPS tract " + fed["name"]

    out = pd.concat([city, fed], ignore_index=True)
    out["acres"] = out["blklot"].map(parcels.set_index("blklot").geometry.area / 43560).round(2)
    return out[["name", "blklot", "source", "year", "acres"]].sort_values(["name", "blklot"])


# ---------------------------------------------------------------- the first map showing each park

def map_series() -> list[str]:
    """The zoning maps in year order (ties in table order), today's last."""
    from sfzmaps.districts.series import SERIES

    return [s for s, i in SERIES.items() if i.role == "use"] + ["today"]


def map_year(series: str) -> int:
    from sfzmaps.districts.series import SERIES

    return date.today().year if series == "today" else SERIES[series].year


def map_label(series: str) -> str:
    """How the lot card names a map: 'the 1948 map', 'the 1980-1990 map', "today's map"."""
    if series == "today":
        return "today's map"
    from sfzmaps.districts.series import SERIES

    return f"the {SERIES[series].period.removesuffix('-digital')} map"


def first_maps(cands: pd.DataFrame | None = None) -> pd.DataFrame:
    """For each park lot of the reference data (`candidates`), the first zoning map whose code for
    it is in category `public`: name, blklot, series. The code is the map's own (the read, with its
    overrides; apply_parks never changes it), so this is what the map shows before any park rule.
    Lots no map shows as public are left out. Reads each series' published parcels.fgb."""
    from sfzmaps.districts.categories import category_for

    if cands is None:
        cands = candidates()
    from sfzmaps.districts import lots as era_lots
    from sfzmaps.districts.read import series_year

    lots = set(cands["blklot"])
    first: dict[str, str] = {}
    for s in map_series():
        path = config.OUT_DIR / s / "districts" / "parcels.fgb"
        if not path.exists():
            continue
        g = gpd.read_file(path, columns=["blklot", "code"], ignore_geometry=True)
        code = dict(zip(g["blklot"], g["code"]))
        # a map drawn on the lots of its year names a newer park lot by the lot it sits in
        yr = series_year(s)
        remap = era_lots.era_lot_of(yr) if yr is not None and not lots <= set(code) else {}
        year = map_year(s)
        for b in sorted(lots):
            c = code.get(b) if b in code else code.get(remap.get(b))
            if isinstance(c, str) and c.strip() and category_for(c, year) == "public":
                first.setdefault(b, s)
    out = cands[cands["blklot"].isin(first)][["name", "blklot"]].copy()
    out["series"] = out["blklot"].map(first)
    return out.sort_values(["name", "blklot"]).reset_index(drop=True)


def write_first_maps(cands: pd.DataFrame | None = None) -> Path:
    """inputs/park_maps.csv from the published maps (`sfzmaps parks --maps`)."""
    from sfzmaps import inputs

    path = inputs.park_maps_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    first_maps(cands).to_csv(path, index=False)
    return path


def check_first_maps() -> None:
    """Fail when inputs/park_maps.csv no longer matches the published maps (a read or an override
    changed what a map shows): run `sfzmaps parks --maps`, then build again. Skipped without the
    park reference data."""
    from sfzmaps import inputs

    ref = ("parcels.parquet", "rec_park_properties.parquet", "nps_tracts.parquet", "zoning_current.parquet")
    if not all((config.REF_DIR / f).exists() for f in ref):
        log.warning("no park reference data in %s: inputs/park_maps.csv not checked", config.REF_DIR)
        return
    path = inputs.park_maps_path()
    have = pd.read_csv(path, dtype=str) if path.exists() else pd.DataFrame(columns=["name", "blklot", "series"])
    rows = lambda d: sorted(map(tuple, d[["name", "blklot", "series"]].astype(str).to_numpy()))  # noqa: E731
    if rows(have) != rows(first_maps()):
        raise RuntimeError(f"{path} is out of date with the published maps: run "
                           "`sfzmaps parks --maps`, then build again")
