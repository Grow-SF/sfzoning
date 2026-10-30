"""Assembly: read a series (reader A / B / DataSF today), overrides, height join, rules join ->
story category, then the published outputs under out/<series>/districts/:

  parcels.fgb      one row per mapped parcel: blklot, code, height_code, category, confidence,
                   source_sheet (EPSG:2227)
  districts.fgb    parcels dissolved by code (code, category)
  <series>-districts.pmtiles   vector tiles, layers `parcels` and `districts` (EPSG:3857, z11-16)
  stats.json       {category: {parcels, area_sqft}, unknown_share, mapped_area_sqft}
  unknown_codes.json   codes read but absent from the research table -> parcel count

A height series (role "height") instead writes out/<zoning series>/districts/height.parquet,
joined by blklot into its period's zoning output (republished if that already exists).
"""
from __future__ import annotations

import json
import logging
import shutil
import subprocess
import warnings
from datetime import date
from pathlib import Path

import geopandas as gpd
import pandas as pd

from sfzmaps import config
from sfzmaps.districts.categories import CATEGORIES, category_for
from sfzmaps.districts.series import SERIES

log = logging.getLogger(__name__)

MEDIUM = 0.6            # confidence cap for parcels read from needs_review sheets
LOW_CONFIDENCE = 0.5    # the review page outlines parcels below this
MIN_ZOOM, MAX_ZOOM = 11, 16
COLUMNS = ["blklot", "code", "height_code", "confidence", "source_sheet"]
TILE_PARCEL_PROPS = ["blklot", "code", "height_code", "category", "confidence"]
# Story-category colours shared by the review page (and a starting point for the site).
COLORS = {"single_family": "#f6d55c", "two_unit": "#f4a259", "three_unit": "#e07a3f",
          "multi_family": "#c0392b", "mixed": "#9b59b6", "commercial": "#3b7dd8",
          "industrial": "#7f8c8d", "public": "#2e9e5b", "unrestricted": "#4a2f27",
          "unknown": "#d9d9d9"}


def out_dir(series: str) -> Path:
    return config.OUT_DIR / series / "districts"


def _raw_path(series: str) -> Path:
    return config.WORK_DIR / series / "districts_raw.parquet"


def zoning_series_of(series: str) -> str:
    """The period's zoning series for a height series ("1980-1990-height" -> "...-zoning")."""
    period = SERIES[series].period
    return next(s for s, i in SERIES.items() if i.period == period and i.role == "use")


def load_parcels() -> gpd.GeoDataFrame:
    p = gpd.read_parquet(config.REF_DIR / "parcels.parquet")
    return p[p["active"]].reset_index(drop=True)


# ---------------------------------------------------------------- reading

def sheet_statuses(series: str) -> dict[str, str]:
    from sfzmaps import manifest, stitch

    return {sid: manifest.status_of(series, sid, sc) for sid, sc in stitch.load_scores(series).items()}


def cap_confidence(df: pd.DataFrame, statuses: dict[str, str]) -> pd.DataFrame:
    """Parcels from needs_review sheets keep their code at confidence <= MEDIUM."""
    df = df.copy()
    bad = df["source_sheet"].map(lambda s: statuses.get(s) == "needs_review")
    df.loc[bad, "confidence"] = df.loc[bad, "confidence"].clip(upper=MEDIUM)
    return df


def merge_use_height(use: pd.DataFrame, height: pd.DataFrame, role: str) -> pd.DataFrame:
    """One row per parcel read on a use sheet (code) and/or a height sheet (height_code);
    confidence and source_sheet come from the series' role."""
    u = use[["blklot", "code", "confidence", "source_sheet"]].drop_duplicates("blklot")
    h = height[["blklot", "height_code", "confidence", "source_sheet"]].drop_duplicates("blklot")
    m = u.merge(h, on="blklot", how="outer", suffixes=("_u", "_h"))
    mine, other = ("_u", "_h") if role == "use" else ("_h", "_u")
    out = pd.DataFrame({
        "blklot": m["blklot"],
        "code": m["code"].astype(object).where(m["code"].notna(), None),
        "height_code": m["height_code"].astype(object).where(m["height_code"].notna(), None),
        "confidence": m["confidence" + mine].fillna(0.0).astype(float),
        "source_sheet": m["source_sheet" + mine].where(m["source_sheet" + mine].notna(),
                                                       m["source_sheet" + other]),
    })
    return out.sort_values("blklot").reset_index(drop=True)[COLUMNS]


def _read_b(series: str, parcels, order: list[str]) -> pd.DataFrame:
    """Reader B per sheet kind: use codes from use sheets, height codes from height sheets
    (labels.sheet_role), each parcel on its highest-priority sheet of that kind."""
    from sfzmaps.districts import labels, sampling

    roles = labels.sheet_roles(series, order)
    log.info("%s sheet roles: %s", series, roles)
    parts = {}
    for kind, col, conf in (("use", "code", "use_confidence"), ("height", "height_code", "height_confidence")):
        kind_order = [s for s in order if roles[s] == kind]
        sheet = sampling.assign_sheets(parcels, series, sheets=kind_order).to_numpy()
        out = []
        for sid in kind_order:
            sub = parcels[sheet == sid]
            if not len(sub):
                continue
            log.info("%s sheet %s (%s): %d parcels", series, sid, kind, len(sub))
            r = labels.assign_parcels(sub.reset_index(drop=True), config.WORK_DIR / series / sid)
            out.append(pd.DataFrame({"blklot": r["blklot"], col: r[col], "confidence": r[conf],
                                     "source_sheet": sid}))
        parts[kind] = pd.concat(out, ignore_index=True) if out else \
            pd.DataFrame(columns=["blklot", col, "confidence", "source_sheet"])
    return merge_use_height(parts["use"], parts["height"], SERIES[series].role)


def read_raw(series: str, *, sheets: list[str] | None = None, client=None) -> pd.DataFrame:
    """Reader output for every mapped parcel (COLUMNS). Each parcel is read on its
    highest-priority covering sheet (of the right kind, for reader B); `sheets` limits the
    read to those sheets."""
    from sfzmaps.districts import pattern, sampling

    info = SERIES[series]
    parcels = load_parcels()
    if info.reader == "B":
        order = [s for s in sampling.sheet_order(series) if sheets is None or s in sheets]
        return cap_confidence(_read_b(series, parcels, order), sheet_statuses(series))
    parcels["sheet"] = sampling.assign_sheets(parcels, series).to_numpy()
    order = [s for s in sampling.sheet_order(series) if sheets is None or s in sheets]
    swatches = pattern.legend_swatches(series, client=client)
    out = []
    for sid in order:
        sub = parcels[parcels["sheet"] == sid]
        if not len(sub):
            continue
        log.info("%s sheet %s: %d parcels", series, sid, len(sub))
        r = pattern.read_sheet(series, sid, sub, swatches=swatches, client=client)
        r["height_code"] = None
        r["source_sheet"] = sid
        out.append(r[COLUMNS])
    if not out:
        return pd.DataFrame(columns=COLUMNS)
    return cap_confidence(pd.concat(out, ignore_index=True), sheet_statuses(series))


def read_today() -> pd.DataFrame:
    from sfzmaps.districts import today
    from sfzmaps.sources import datasf

    path = config.REF_DIR / "zoning_current.parquet"
    if not path.exists():
        datasf.fetch_zoning_current(config.REF_DIR)
    df = today.assign_today(load_parcels(), gpd.read_parquet(path))
    df["height_code"] = None
    df["source_sheet"] = "datasf"
    return df[COLUMNS]


# ---------------------------------------------------------------- overrides / joins

def apply_overrides(df: pd.DataFrame, series: str) -> pd.DataFrame:
    """overrides/<series>.csv rows (blklot or block, code) set code at confidence 1.
    A block row applies to every parcel of the block; a blklot row wins over a block row.
    Override keys that match no parcel warn."""
    path = config.DATA_DIR / "overrides" / f"{series}.csv"
    if not path.exists():
        return df
    ov = pd.read_csv(path, dtype=str, keep_default_na=False)
    df = df.copy()
    df["code"] = df["code"].astype(object)
    blk = df["blklot"].astype(str)
    block = blk.str[:-3]
    rows = ov.to_dict("records")
    rows.sort(key=lambda r: bool((r.get("blklot") or "").strip()))   # block rows first
    for r in rows:
        code = (r.get("code") or "").strip()
        key = (r.get("blklot") or "").strip()
        mask = blk == key if key else block == (r.get("block") or "").strip()
        if not code or not (key or (r.get("block") or "").strip()):
            continue
        if not mask.any():
            msg = f"override {path.name}: {'blklot' if key else 'block'} {key or r.get('block')} matches no parcel"
            log.warning(msg)
            warnings.warn(msg, UserWarning, stacklevel=2)
            continue
        df.loc[mask, "code"] = code
        df.loc[mask, "confidence"] = 1.0
    return df


def join_height(df: pd.DataFrame, height: pd.DataFrame) -> pd.DataFrame:
    """height_code from a height series, by blklot (where it read one)."""
    h = height.dropna(subset=["height_code"]).drop_duplicates("blklot").set_index("blklot")["height_code"]
    df = df.copy()
    joined = df["blklot"].map(h)
    df["height_code"] = joined.where(joined.notna(), df["height_code"]).astype(object)
    return df


# Blocks outside city zoning until a year: the Port's piers and wharves (block 9900) were State
# Harbor Commission land until the Burton Act transfer (1968, effective 1969), so the 1921-1968
# maps draw them as bare outlines that read as the white (2R) pattern.  They stay uncoded.
UNZONED_UNTIL = {"9900": 1969}


def mask_unzoned(df: pd.DataFrame, year: int) -> pd.DataFrame:
    """Code None, confidence 0 for parcels in UNZONED_UNTIL blocks before that block's year."""
    df = df.copy()
    block = df["blklot"].astype(str).str[:4]
    m = block.map(lambda b: b in UNZONED_UNTIL and year < UNZONED_UNTIL[b])
    df["code"] = df["code"].astype(object)
    df.loc[m, "code"] = None
    df.loc[m, "confidence"] = 0.0
    return df


def categorise(df: pd.DataFrame, year: int) -> pd.DataFrame:
    df = df.copy()
    df["category"] = [category_for(c, year) if isinstance(c, str) and c.strip() else "unknown"
                      for c in df["code"]]
    return df


# ---------------------------------------------------------------- outputs

def stats(g: gpd.GeoDataFrame) -> dict:
    """Parcels and land area by category. Stacked parcels (condos sharing one footprint)
    count once toward area."""
    area = g.geometry.area.where(~g.geometry.to_wkb().duplicated(), 0.0)
    out: dict = {}
    for c in CATEGORIES:
        m = g["category"] == c
        out[c] = {"parcels": int(m.sum()), "area_sqft": round(float(area[m].sum()), 1)}
    n = len(g)
    out["unknown_share"] = round(out["unknown"]["parcels"] / n, 4) if n else 0.0
    out["mapped_area_sqft"] = round(sum(out[c]["area_sqft"] for c in CATEGORIES), 1)
    return out


def dissolve(g: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    coded = g[g["code"].notna()]
    if coded.empty:
        return gpd.GeoDataFrame({"code": [], "category": []}, geometry=[], crs=g.crs)
    d = coded[["code", "category", "geometry"]].dissolve(by="code", aggfunc="first").reset_index()
    d = d[d.geometry.notna() & ~d.geometry.is_empty]
    return d[["code", "category", "geometry"]]


def _has_pmtiles() -> bool:
    if not shutil.which("ogr2ogr"):
        return False
    r = subprocess.run(["ogr2ogr", "--formats"], capture_output=True, text=True)
    return "PMTiles" in r.stdout


def write_pmtiles(parcels: gpd.GeoDataFrame, districts: gpd.GeoDataFrame, out: Path,
                  name: str) -> Path | None:
    if not _has_pmtiles():
        log.warning("ogr2ogr has no PMTiles driver; skipping %s", out)
        return None
    gpkg = out.with_suffix(".tmp.gpkg")
    tmp = out.with_name(out.stem + ".tmp.pmtiles")
    for p in (gpkg, tmp):
        p.unlink(missing_ok=True)
    try:
        parcels[TILE_PARCEL_PROPS + ["geometry"]].to_crs("EPSG:3857").to_file(gpkg, layer="parcels", driver="GPKG")
        districts.to_crs("EPSG:3857").to_file(gpkg, layer="districts", driver="GPKG")
        subprocess.run(["ogr2ogr", "-f", "PMTiles", str(tmp), str(gpkg), "-t_srs", "EPSG:3857",
                        "-dsco", f"MINZOOM={MIN_ZOOM}", "-dsco", f"MAXZOOM={MAX_ZOOM}",
                        "-dsco", f"NAME={name}", "-dsco", "MAX_SIZE=2000000"],
                       check=True, capture_output=True)
        tmp.replace(out)
    finally:
        gpkg.unlink(missing_ok=True)
        tmp.unlink(missing_ok=True)
    return out


def publish(series: str, df: pd.DataFrame, year: int) -> Path:
    """Unzoned blocks -> overrides -> height join -> rules join -> category, then every
    output file."""
    df = apply_overrides(mask_unzoned(df, year), series)
    hp = out_dir(series) / "height.parquet"
    if hp.exists():
        df = join_height(df, pd.read_parquet(hp))
    df = categorise(df, year)
    geo = load_parcels()[["blklot", "geometry"]].drop_duplicates("blklot")
    g = gpd.GeoDataFrame(df.merge(geo, on="blklot", how="inner"), geometry="geometry", crs=geo.crs)
    g = g[g.geometry.notna() & ~g.geometry.is_empty]       # a few DataSF parcels have no shape
    g = g[["blklot", "code", "height_code", "category", "confidence", "source_sheet", "geometry"]]
    g["confidence"] = g["confidence"].astype(float).round(3)
    d = out_dir(series)
    d.mkdir(parents=True, exist_ok=True)
    g.to_file(d / "parcels.fgb", driver="FlatGeobuf")
    districts = dissolve(g)
    districts.to_file(d / "districts.fgb", driver="FlatGeobuf")
    write_pmtiles(g, districts, d / f"{series}-districts.pmtiles", f"SF zoning districts {series}")
    (d / "stats.json").write_text(json.dumps(stats(g), indent=1))
    unk = g[(g["category"] == "unknown") & g["code"].notna()]["code"].value_counts()
    (d / "unknown_codes.json").write_text(json.dumps({str(k): int(v) for k, v in unk.items()}, indent=1))
    return d


def run_series(series: str, *, sheets: list[str] | None = None, client=None) -> Path:
    """Read, assign, join and publish one series (or `today`); returns its districts dir."""
    if series == "today":
        return publish("today", read_today(), date.today().year)
    info = SERIES[series]
    if info.role == "skip":
        raise ValueError(f"{series}: special-use series are out of scope")
    raw = read_raw(series, sheets=sheets, client=client)
    raw_path = _raw_path(series)
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    if info.role == "height":
        zs = zoning_series_of(series)
        d = out_dir(zs)
        d.mkdir(parents=True, exist_ok=True)
        raw[["blklot", "height_code", "confidence", "source_sheet"]].to_parquet(d / "height.parquet")
        zraw = _raw_path(zs)
        if zraw.exists():
            publish(zs, pd.read_parquet(zraw), SERIES[zs].year)
        return d
    raw.to_parquet(raw_path)
    return publish(series, raw, info.year)


def republish(series: str) -> Path:
    """Re-apply overrides / height / rules to the last read without reading again."""
    if series == "today":
        return run_series("today")
    return publish(series, pd.read_parquet(_raw_path(series)), SERIES[series].year)
