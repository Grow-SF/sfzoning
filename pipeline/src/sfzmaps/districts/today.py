"""Today's zoning per parcel from DataSF zoning districts (3i4a-hu95)."""
from __future__ import annotations

from datetime import date

import geopandas as gpd
import pandas as pd

from sfzmaps.districts.categories import category_for


def assign_today(parcels: gpd.GeoDataFrame, zoning: gpd.GeoDataFrame,
                 year: int | None = None) -> gpd.GeoDataFrame:
    """Code from the zoning polygon covering the largest share of each parcel."""
    year = year or date.today().year
    p = parcels[["blklot", "geometry"]].reset_index(drop=True)
    p["pid"] = p.index
    z = zoning.to_crs(p.crs)[["zoning", "zoning_sim", "geometry"]].copy()
    code = z["zoning"].where(z["zoning"].notna() & (z["zoning"].astype(str).str.strip() != ""),
                             z["zoning_sim"])
    z = gpd.GeoDataFrame({"code": code, "geometry": z.geometry}, crs=z.crs)
    z = z[z["code"].notna()]
    ov = gpd.overlay(p[["pid", "geometry"]], z, how="intersection", keep_geom_type=False)
    ov["area"] = ov.geometry.area
    best = ov.sort_values("area", ascending=False).drop_duplicates("pid").set_index("pid")["code"]
    out = pd.DataFrame({"blklot": p["blklot"], "code": p["pid"].map(best)})
    out["category"] = [category_for(c, year) if isinstance(c, str) else "unknown"
                       for c in out["code"]]
    out["confidence"] = 1.0
    return out
