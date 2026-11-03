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
"""
from __future__ import annotations

import re

import geopandas as gpd
import pandas as pd

from sfzmaps import config

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
