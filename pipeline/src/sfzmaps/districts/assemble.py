"""Assembly: read a series (reader A / B / DataSF today), overrides, height join, rules join ->
story category, then the published outputs under out/<series>/districts/:

  parcels.fgb      one row per mapped parcel: blklot, code, height_code, category, confidence,
                   source_sheet, note (why a park lot shows as public; apply_parks) (EPSG:2227)
  districts.fgb    parcels dissolved by code (code, category)
  stats.json       {category: {parcels, area_sqft}, unknown_share, mapped_area_sqft,
                   unmapped_parcels, parks: {count, acres} (park_tally),
                   provenance: {use, height}} (today: also notes)
  unknown_codes.json   codes read but absent from the research table -> parcel count

and, only for the review page (`sfzmaps review`, review_tiles), <series>-districts.pmtiles:
vector tiles, layers `parcels` and `districts` (EPSG:3857, z11-16). The site reads the shared
archive instead (districts/shared.py).

A height series (role "height") instead writes out/<zoning series>/districts/height.parquet,
joined by blklot into its period's zoning output (republished if that already exists).

Provenance: every read writes {reader, region_read, git, read_at} next to its output
(work/<series>/districts_raw.json, out/<zoning series>/districts/height.json); publish copies
both into stats.json.  A --sheet read is partial: it goes to districts_raw.partial.* and
out/<series>/districts-partial/ and never touches the full-series outputs.

Freshness (sfzmaps.fresh): a full read is skipped when its output's fingerprint still matches
read_inputs() (the series' sheet files, its hand-made inputs, the parcels and the code the read
runs), and publishing is skipped when the outputs' fingerprint still matches publish_inputs()
(the read, heights, overrides, rules and the code that joins them). So a build with nothing
changed for a series rewrites nothing, and an overrides edit republishes without reading again.
"""
from __future__ import annotations

import json
import logging
import shutil
import subprocess
import re
import warnings
from datetime import date, datetime
from pathlib import Path

import geopandas as gpd
import pandas as pd

from sfzmaps import config, fresh, inputs
from sfzmaps.districts.categories import CATEGORIES, category_for
# the reading half lives in districts.read, so publishing code can change without a new read
from sfzmaps.districts.read import (COLUMNS, MEDIUM, cap_confidence, load_parcels,  # noqa: F401
                                    merge_use_height, read_raw, sheet_statuses)
from sfzmaps.districts.series import SERIES

log = logging.getLogger(__name__)

LOW_CONFIDENCE = 0.5    # the review page outlines parcels below this
MIN_ZOOM, MAX_ZOOM = 11, 16
TILE_PARCEL_PROPS = ["blklot", "code", "height_code", "category", "confidence"]
# Story-category colours shared by the review page (and a starting point for the site).
COLORS = {"single_family": "#fe4400", "two_unit": "#f89600", "three_unit": "#ffee66",
          "homes_low": "#a0f5be", "homes_mid": "#5cda80", "homes_high": "#2cb25a",
          "homes_top": "#009600", "commercial_light": "#89d7ff", "commercial_dense": "#1149af",
          "industrial": "#85420b", "public": "#97a1b1", "unrestricted": "#8562af",
          "unknown": "#ffffff"}


def out_dir(series: str, *, partial: bool = False) -> Path:
    return config.OUT_DIR / series / ("districts-partial" if partial else "districts")


def _raw_path(series: str, *, partial: bool = False) -> Path:
    return config.WORK_DIR / series / ("districts_raw.partial.parquet" if partial else "districts_raw.parquet")


def _prov_path(series: str, *, partial: bool = False) -> Path:
    return _raw_path(series, partial=partial).with_suffix(".json")


def _height_prov_path(zoning_series: str, *, partial: bool = False) -> Path:
    return out_dir(zoning_series, partial=partial) / "height.json"


def height_series_of(series: str) -> str | None:
    """The period's height series for a zoning series, or None."""
    info = SERIES.get(series)
    if info is None:
        return None
    return next((s for s, i in SERIES.items() if i.period == info.period and i.role == "height"), None)


# ---------------------------------------------------------------- provenance

def git_rev() -> str:
    """Short HEAD sha, + "-dirty" when the work tree has changes; "unknown" outside git."""
    try:
        sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=config.PIPELINE_DIR,
                             capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"],
                               cwd=config.PIPELINE_DIR, capture_output=True, text=True,
                               check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return sha + ("-dirty" if dirty else "")


def provenance(series: str, *, cache_only: bool = False, region_misses: int = 0) -> dict:
    """How a read was made: reader, whether per-region reads were on (regionread.ENABLED),
    the code version and when.  A cache-only read also records how many region mosaics were
    not cached (those regions fell back to inheritance)."""
    from sfzmaps.districts import regionread

    info = SERIES.get(series)
    out = {"reader": info.reader if info else "datasf", "region_read": bool(regionread.ENABLED),
           "git": git_rev(), "read_at": datetime.now().astimezone().isoformat(timespec="seconds")}
    if cache_only:
        out["cache_only"] = True
        out["region_misses"] = int(region_misses)
    return out


def _write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1))
    tmp.replace(path)


def _read_json(path: Path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None


def read_provenance(series: str, *, partial: bool = False) -> dict:
    """{"use": ..., "height": ...} provenance of the reads behind a series' output (None where
    there is no such read or record)."""
    if series == "today":
        return {"use": _read_json(_prov_path("today")), "height": None}
    height = _read_json(_height_prov_path(series, partial=True)) if partial else None
    return {"use": _read_json(_prov_path(series, partial=partial)),
            "height": height or _read_json(_height_prov_path(series))}


def zoning_series_of(series: str) -> str:
    """The period's zoning series for a height series ("1980-1990-height" -> "...-zoning")."""
    period = SERIES[series].period
    return next(s for s, i in SERIES.items() if i.period == period and i.role == "use")


# ---------------------------------------------------------------- freshness

# Modules that make the per-sheet files a read fingerprints by content (warps, fits, scores, the
# stitched tiles): their code reaches a read only through those files, so a change there reruns
# the read only when it changes the files.
SHEET_PRODUCERS = frozenset({"sfzmaps.runner", "sfzmaps.warp", "sfzmaps.refine", "sfzmaps.placement",
                             "sfzmaps.anchors", "sfzmaps.tiles", "sfzmaps.pointsfile"})
# The code publish() runs: the joins, the rules and the categories (not the readers).
PUBLISH_MODULES = ("sfzmaps.districts.assemble", "sfzmaps.districts.categories",
                   "sfzmaps.districts.rules", "sfzmaps.districts.series", "sfzmaps.inputs",
                   "sfzmaps.config", "sfzmaps.fresh", "sfzmaps.parks")
PUBLISHED = ("parcels.fgb", "districts.fgb", "stats.json", "unknown_codes.json")


def _rules_csv() -> Path:
    from sfzmaps.districts import rules

    return rules.RULES_CSV


def _read_by_districts(p: Path) -> bool:
    """A sheet file the reads may use: not another step's fingerprint, nor the paper layer."""
    n = p.name
    return not (n.endswith(".inputs.json") or n.startswith("warped_paper") or ".tmp" in n)


def read_inputs(series: str) -> list[Path]:
    """Everything a full read of a series is made from: its sheets' files (work/<series>/<sheet>/),
    its hand-made inputs except the overrides (applied when publishing), the label scale, the
    parcels, the legend boxes and the code the read runs (districts.read and what it imports);
    for the legend-pattern reader (A), also the research table, whose codes it matches."""
    from sfzmaps.districts import pattern

    root = config.WORK_DIR / series
    sheets = sorted(p for p in root.rglob("*") if p.parent != root and p.is_file()
                    and _read_by_districts(p)) if root.is_dir() else []
    hand = inputs.series_dir(series)
    made = sorted(p for p in hand.rglob("*") if p.is_file() and p != inputs.overrides_path(series)) \
        if hand.is_dir() else []
    info = SERIES.get(series)
    from sfzmaps.districts import labels
    if series in labels.TWIN_SERIES and labels.TWIN_FUSE:     # each sheet joins its twin's lines
        twin = labels.TWIN_SERIES[1 - labels.TWIN_SERIES.index(series)]
        sheets += sorted(p for n in ("page.png", "district_labels.json")
                         for p in (config.WORK_DIR / twin).glob(f"*/{n}"))
    files = [*sheets, *made, inputs.label_scale_path(), config.REF_DIR / "parcels.parquet",
             *([_rules_csv()] if info is None or info.reader == "A" else []), pattern.LEGENDS_JSON,
             *fresh.code_files(["sfzmaps.districts.read"], cut=SHEET_PRODUCERS | _OTHER_READER[
                 "A" if info is None else info.reader])]
    return [p for p in files if p.exists()]


# read.py runs one reader per series; the other reader's code (and what only it imports) doesn't
# reach the read, so a change there must not re-read the series (a reader B change used to re-read
# 1921-1948 too, ~10 minutes of every build).
_OTHER_READER = {"A": frozenset({"sfzmaps.districts.labels"}),
                 "B": frozenset({"sfzmaps.districts.pattern"})}


def publish_inputs(series: str) -> list[Path]:
    """Everything publish() makes a series' outputs from (today: the DataSF zoning it reads)."""
    from sfzmaps.districts import today

    if series == "today":
        files = [config.REF_DIR / "zoning_current.parquet", Path(today.__file__)]
    else:
        files = [_raw_path(series), _prov_path(series), out_dir(series) / "height.parquet",
                 _height_prov_path(series), inputs.overrides_path(series)]
        hs = height_series_of(series)
        if hs:
            files.append(inputs.overrides_path(hs))
    files += [config.REF_DIR / "parcels.parquet", _rules_csv(),
              inputs.parks_path(), inputs.park_maps_path(),     # apply_parks, park_tally
              *(fresh.module_file(m) for m in PUBLISH_MODULES)]
    return [p for p in files if p is not None and p.exists()]


def _read_current(series: str, out: Path, prov_path: Path) -> bool:
    """A full read's output still matches what it was read from. A cache-only read that fell back
    on uncached regions never counts: a later paid read may have filled them."""
    prov = _read_json(prov_path)
    if not prov or (prov.get("cache_only") and int(prov.get("region_misses") or 0) > 0):
        return False
    return fresh.is_fresh(out, read_inputs(series))


def publish_current(series: str) -> bool:
    d = out_dir(series)
    return all((d / n).exists() for n in PUBLISHED) and \
        fresh.is_fresh(d / "stats.json", publish_inputs(series))


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

_BLOCK_RE = re.compile(r"\d{4}[A-Z]?")


def blocks_of(blklots: pd.Series, parcels: pd.DataFrame | None = None) -> pd.Series:
    """Assessor block of each blklot: the parcel table's block_num (blocks may carry a letter,
    0792T; lots may too, 2330006A, so no fixed slice of the blklot works).  Blklots not in the
    table fall back to the leading 4 digits plus a letter right after them."""
    blk = blklots.astype(str)
    try:
        p = parcels if parcels is not None else load_parcels(every=True)
        m = p.drop_duplicates("blklot").set_index("blklot")["block_num"].astype(str)
        got = blk.map(m)
    except Exception:  # noqa: BLE001 - no parcel table (unit tests): the pattern decides
        got = pd.Series([None] * len(blk), index=blk.index, dtype=object)
    fb = blk.map(lambda b: (_BLOCK_RE.match(b).group(0) if _BLOCK_RE.match(b) else b[:4]))
    return got.where(got.notna(), fb).astype(str)


def apply_overrides(df: pd.DataFrame, series: str, *, column: str = "code",
                    blocks: pd.Series | None = None, remap: dict[str, str] | None = None) -> pd.DataFrame:
    """inputs/<series>/overrides.csv rows (blklot or block, code) set `column` (code at confidence 1;
    for a height series' file, height_code: its value column may be height_code or code).
    A block row applies to every parcel of the block (block_num); a blklot row wins over a
    block row.  Override keys that match no parcel warn.  A row's `note` (optional column, e.g.
    from a reader's correction) goes to `override_note`, shown on the lot card (publish).
    remap: {lot: the series' lot it sits in} (lots.era_lot_of) for a key that isn't one of the
    series' lots (an override written for today's lots, on a map drawn on the lots of its year)."""
    path = inputs.overrides_path(series)
    if not path.exists():
        return df
    ov = pd.read_csv(path, dtype=str, keep_default_na=False)
    df = df.copy()
    df[column] = df[column].astype(object)
    blk = df["blklot"].astype(str)
    block = blocks.reindex(df.index) if blocks is not None else blocks_of(blk)
    rows = ov.to_dict("records")
    rows.sort(key=lambda r: bool((r.get("blklot") or "").strip()))   # block rows first
    for r in rows:
        code = (r.get(column) or r.get("code") or "").strip()
        key = (r.get("blklot") or "").strip()
        if key and remap and not (blk == key).any() and key in remap:
            key = remap[key]
        mask = blk == key if key else block == (r.get("block") or "").strip()
        if not code or not (key or (r.get("block") or "").strip()):
            continue
        if not mask.any():
            msg = f"override {path.name}: {'blklot' if key else 'block'} {key or r.get('block')} matches no parcel"
            log.warning(msg)
            warnings.warn(msg, UserWarning, stacklevel=2)
            continue
        df.loc[mask, column] = code
        if column == "code":
            df.loc[mask, "confidence"] = 1.0
            note = (r.get("note") or "").strip()
            if note:
                if "override_note" not in df:
                    df["override_note"] = None
                df.loc[mask, "override_note"] = note
    return df


def add_override_notes(df: pd.DataFrame) -> pd.DataFrame:
    """Each overridden lot's override note joins its card note (after apply_parks)."""
    if "override_note" not in df:
        return df
    df = df.copy()
    m = df["override_note"].notna()
    df.loc[m, "note"] = [f"{n}. {o}" if isinstance(n, str) and n else o
                         for n, o in zip(df.loc[m, "note"], df.loc[m, "override_note"])]
    return df.drop(columns="override_note")


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


def mask_unzoned(df: pd.DataFrame, year: int, *, blocks: pd.Series | None = None) -> pd.DataFrame:
    """Code None, confidence 0 for parcels in UNZONED_UNTIL blocks (block_num) before that
    block's year."""
    df = df.copy()
    block = blocks.reindex(df.index) if blocks is not None else blocks_of(df["blklot"])
    m = block.map(lambda b: b in UNZONED_UNTIL and year < UNZONED_UNTIL[b])
    df["code"] = df["code"].astype(object)
    df.loc[m, "code"] = None
    df.loc[m, "confidence"] = 0.0
    return df


def categorise(df: pd.DataFrame, year: int) -> pd.DataFrame:
    df = df.copy()
    # once per distinct code: a lookup scans the research table, and a series has ~200k parcels
    # but only dozens of codes
    known: dict[str, str] = {}

    def category(c) -> str:
        if not (isinstance(c, str) and c.strip()):
            return "unknown"
        if c not in known:
            known[c] = category_for(c, year)
        return known[c]

    df["category"] = [category(c) for c in df["code"]]
    return df


def _parks_csv() -> pd.DataFrame:
    path = inputs.parks_path()
    if not path.exists():
        return pd.DataFrame({"name": [], "blklot": [], "since": [], "public_since": []})
    p = pd.read_csv(path, dtype={"name": str, "blklot": str})
    p["blklot"] = p["blklot"].str.strip()
    # public_since (optional): the year the land became public, when that is before it became a park
    if "public_since" not in p:
        p["public_since"] = p["since"]
    p["public_since"] = pd.to_numeric(p["public_since"], errors="coerce").fillna(p["since"])
    p["public_since"] = p[["since", "public_since"]].min(axis=1)
    return p


def _park_maps_csv() -> pd.DataFrame:
    path = inputs.park_maps_path()
    if not path.exists():
        return pd.DataFrame({"name": [], "blklot": [], "series": []}, dtype=object)
    p = pd.read_csv(path, dtype=str)
    p["blklot"] = p["blklot"].str.strip()
    return p


def park_starts() -> pd.DataFrame:
    """Every park lot with when it became one, one row per lot: name, blklot, since (the year it
    is a park from), reason (how the lot card says it: '1870' for a dated source in
    inputs/parks.csv, 'the 1964 map' for the first map showing it as public, inputs/park_maps.csv),
    public_since (the year it is `public` from: since, or earlier for land public before it was a
    park). With both, the earlier wins (owner's decision, 2026-11-02), except where parks.csv says
    the land was public before it was a park (the Presidio, an Army post until 1994): a map showing
    it public says so, not that it was a park."""
    from sfzmaps import parks

    dated = _parks_csv().drop_duplicates("blklot").set_index("blklot")
    maps = _park_maps_csv().drop_duplicates("blklot").set_index("blklot")
    rows = []
    for b in dated.index.union(maps.index):
        name, since, reason, public = None, None, None, None
        if b in dated.index:
            r = dated.loc[b]
            name, since, reason, public = r["name"], int(r["since"]), str(int(r["since"])), int(r["public_since"])
        if b in maps.index and (public is None or public >= since):
            s = maps.loc[b, "series"]
            year = parks.map_year(s)
            if since is None or year < since:
                name, since, reason = name or maps.loc[b, "name"], year, parks.map_label(s)
                public = min(public, year) if public is not None else year
        rows.append({"name": name, "blklot": b, "since": since, "reason": reason, "public_since": public})
    out = pd.DataFrame(rows, columns=["name", "blklot", "since", "reason", "public_since"])
    return out.astype({"since": "int64", "public_since": "int64"})


def parks_in_effect(year: int) -> pd.DataFrame:
    """park_starts rows (name, blklot) of the parks that existed in `year`: since <= year."""
    p = park_starts()
    return p.loc[p["since"] <= year, ["name", "blklot"]].reset_index(drop=True)


# Lot card notes for a lot from a year on, whatever its map shows (owner's decision, 2026-11-02)
LOT_NOTES = {"1300001": (1996, "Federal land, outside city zoning. "
                               "The Presidio Trust leases about 1,400 homes here.")}


PARK_ERA_SHARE = 0.5     # an era's lot is park when park lots of later times cover this share of it


def _era_park_lots(lots: set[str], parcels: pd.DataFrame, remap: dict[str, str]) -> dict[str, str]:
    """{lot of the series' year: a park lot standing for it} for park lots (keyed by today's lots)
    that aren't lots of the year, when park lots cover at least PARK_ERA_SHARE of that lot: a
    park re-surveyed in the 2000s keeps its park colour on the older maps, while a small park
    carved from a former rail yard doesn't make the whole yard public."""
    later = [b for b in lots if b not in set(parcels["blklot"]) and b in remap]
    if not later:
        return {}
    every = load_parcels(every=True).drop_duplicates("blklot").set_index("blklot").geometry
    era_area = parcels.drop_duplicates("blklot").set_index("blklot").geometry.area
    cover: dict[str, float] = {}
    first: dict[str, str] = {}
    for b in later:
        e = remap[b]
        if b in every.index and e in era_area.index:
            cover[e] = cover.get(e, 0.0) + float(every[b].area)
            first.setdefault(e, b)
    return {e: first[e] for e, a in cover.items() if era_area[e] > 0 and a >= PARK_ERA_SHARE * era_area[e]}


def apply_parks(df: pd.DataFrame, parcels: pd.DataFrame, year: int,
                remap: dict[str, str] | None = None) -> pd.DataFrame:
    """Park lots (park_starts) are `public` in every series from the year they became parks on,
    whatever the map zones them (owner's decision: old maps zoned park land residential as a quirk
    of the law, not because homes could go there); earlier series keep the map's zoning. Land
    public before it was a park (`public_since`, e.g. the Presidio, an Army post from 1776 and a
    park from 1994) is `public` from that year, and only counts as a park from `since`. The read
    code stays, and `note` says since when and what the map shows ("Park since the 1964 map; this
    map zones it R-1"). A lot of a park in effect that the map doesn't cover is added, so the park
    is whole. LOT_NOTES add to a lot's note from their year on."""
    df = df.copy()
    df["note"] = None
    p = park_starts()
    reason = dict(zip(p["blklot"], p["reason"]))
    parks = set(p.loc[p["since"] <= year, "blklot"])
    lots = parks | set(p.loc[p["public_since"] <= year, "blklot"])
    if remap:       # park lots recorded after the map's year stand for the year's lot they cover
        stand = _era_park_lots(lots, parcels, remap)
        for e, b in stand.items():
            reason.setdefault(e, reason.get(b))
            if b in parks:
                parks.add(e)
            lots.add(e)
    missing = sorted(lots & set(parcels["blklot"]) - set(df["blklot"]))
    if missing:
        df = pd.concat([df, pd.DataFrame({"blklot": missing, "code": None, "height_code": None,
                                          "confidence": 0.0, "source_sheet": None,
                                          "category": "unknown", "note": None})], ignore_index=True)
    m = df["blklot"].isin(lots) & (df["category"] != "public")
    what = [f"Park since {reason[b]}" if b in parks else "Public land" for b in df.loc[m, "blklot"]]
    df.loc[m, "note"] = [f"{w}; this map zones it {c}" if isinstance(c, str) and c.strip()
                         else f"{w}; this map gives it no district"
                         for w, c in zip(what, df.loc[m, "code"])]
    df.loc[m, "category"] = "public"
    for b, (start, text) in LOT_NOTES.items():
        at = df["blklot"] == b
        if year >= start and at.any():
            df.loc[at, "note"] = [f"{n}. {text}" if isinstance(n, str) else text for n in df.loc[at, "note"]]
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


def park_tally(year: int, geo: gpd.GeoDataFrame) -> dict:
    """The parks in effect in `year` (park_starts: a dated source or an earlier map showing them):
    how many (by name) and their lots' land area in acres."""
    p = parks_in_effect(year)
    shapes = geo[geo["blklot"].isin(set(p["blklot"]))]
    return {"count": int(p["name"].nunique()), "acres": round(float(shapes.geometry.area.sum()) / 43560, 1)}


def dissolve(g: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    coded = g[g["code"].notna()]
    if coded.empty:
        return gpd.GeoDataFrame({"code": [], "category": []}, geometry=[], crs=g.crs)
    # by category too: a park (apply_parks) keeps the code its map reads but is its own district
    d = coded[["code", "category", "geometry"]].dissolve(by=["code", "category"]).reset_index()
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


TODAY_NOTES = (
    "Categories are read from each parcel's by-right base district in the research table. "
    "RH-1 is single_family here although San Francisco's 2022 fourplex density exception "
    "allows more units per lot in RH districts; the exception is not a base-district rule, "
    "so it is not reflected (owner decision pending).")


def publish(series: str, df: pd.DataFrame, year: int, *, partial: bool = False) -> Path:
    """Unzoned blocks -> overrides -> height join -> height overrides -> rules join ->
    category, then every output file (partial: out/<series>/districts-partial/).  Published on
    the lots of the series' year (read.series_year)."""
    from sfzmaps.districts.read import series_year

    yr = series_year(series)
    parcels = load_parcels(yr)
    df = df.reset_index(drop=True)
    blocks = blocks_of(df["blklot"], parcels)
    from sfzmaps.districts import lots
    remap = lots.era_lot_of(yr) if yr is not None else None
    df = apply_overrides(mask_unzoned(df, year, blocks=blocks), series, blocks=blocks, remap=remap)
    hp = out_dir(series) / "height.parquet"
    if partial and (out_dir(series, partial=True) / "height.parquet").exists():
        hp = out_dir(series, partial=True) / "height.parquet"
    if hp.exists():
        df = join_height(df, pd.read_parquet(hp))
    hs = height_series_of(series)
    if hs:
        df = apply_overrides(df, hs, column="height_code", blocks=blocks, remap=remap)
    df = add_override_notes(apply_parks(categorise(df, year), parcels, year, remap=remap))
    geo = parcels[["blklot", "geometry"]].drop_duplicates("blklot")
    g = gpd.GeoDataFrame(df.merge(geo, on="blklot", how="inner"), geometry="geometry", crs=geo.crs)
    g = g[g.geometry.notna() & ~g.geometry.is_empty]       # a few DataSF parcels have no shape
    g = g[["blklot", "code", "height_code", "category", "confidence", "source_sheet", "note", "geometry"]]
    g["confidence"] = g["confidence"].astype(float).round(3)
    d = out_dir(series, partial=partial)
    d.mkdir(parents=True, exist_ok=True)
    g.to_file(d / "parcels.fgb", driver="FlatGeobuf")
    districts = dissolve(g)
    districts.to_file(d / "districts.fgb", driver="FlatGeobuf")
    tiles = review_tiles_path(series, partial=partial)    # made again on demand (review_tiles)
    for p in (tiles, fresh.sidecar(tiles)):
        p.unlink(missing_ok=True)
    st = stats(g)
    st["unmapped_parcels"] = int((~geo["blklot"].isin(set(g["blklot"]))).sum())
    st["parks"] = park_tally(year, geo)
    st["provenance"] = read_provenance(series, partial=partial)
    if partial:
        st["partial"] = True
    if series == "today":
        st["notes"] = TODAY_NOTES
    (d / "stats.json").write_text(json.dumps(st, indent=1))
    unk = g[(g["category"] == "unknown") & g["code"].notna()]["code"].value_counts()
    (d / "unknown_codes.json").write_text(json.dumps({str(k): int(v) for k, v in unk.items()}, indent=1))
    return d


def review_tiles_path(series: str, *, partial: bool = False) -> Path:
    return out_dir(series, partial=partial) / f"{series}-districts.pmtiles"


def review_tiles(series: str) -> Path | None:
    """The review page's vector tiles of a published series (layers `parcels` and `districts`),
    made from its parcels.fgb and districts.fgb, again only when those changed. Nothing else reads
    them, so builds don't make them. None if the series isn't published or GDAL can't write
    PMTiles."""
    d = out_dir(series)
    src = [d / "parcels.fgb", d / "districts.fgb"]
    out = review_tiles_path(series)
    if not all(p.exists() for p in src):
        return None
    if fresh.is_fresh(out, src):
        return out
    import pyogrio

    g = pyogrio.read_dataframe(src[0], use_arrow=True)
    districts = pyogrio.read_dataframe(src[1], use_arrow=True)
    if write_pmtiles(g, districts, out, f"SF zoning districts {series}") is None:
        return None
    fresh.record(out, src)
    return out


def _publish_full(series: str, *, force: bool = False, today_provenance: bool = False) -> Path:
    """Publish a series' full outputs from its saved read, unless they are current."""
    d = out_dir(series)
    if not force and publish_current(series):
        log.info("%s: published outputs are current; not republished", series)
        return d
    if series == "today":
        if today_provenance:
            _write_json(_prov_path("today"), provenance("today"))
        publish("today", read_today(), date.today().year)
    else:
        publish(series, pd.read_parquet(_raw_path(series)), SERIES[series].year)
    fresh.record(d / "stats.json", publish_inputs(series))
    return d


def run_series(series: str, *, sheets: list[str] | None = None, client=None,
               cache_only: bool = False, force: bool = False) -> Path:
    """Read, assign, join and publish one series (or `today`); returns its districts dir.
    sheets: a partial read (districts_raw.partial.*, out/<series>/districts-partial/), which
    leaves the full-series outputs alone.  cache_only: vision answers from the disk cache only
    (region reads skip uncached mosaics; any other miss raises llm.LLMCacheMiss).  A full read
    whose inputs haven't changed is not read again, nor republished when its outputs are current
    (read_inputs, publish_inputs); force: read and publish anyway."""
    from sfzmaps import llm
    from sfzmaps.districts import regionread

    if series == "today":
        return _publish_full("today", force=force, today_provenance=True)
    info = SERIES[series]
    if info.role == "skip":
        raise ValueError(f"{series}: special-use series are out of scope")
    partial = bool(sheets)
    if info.role == "height":
        zs = zoning_series_of(series)
        out, prov_path = out_dir(zs, partial=partial) / "height.parquet", _height_prov_path(zs, partial=partial)
    else:
        out, prov_path = _raw_path(series, partial=partial), _prov_path(series, partial=partial)
    if not partial and not force and _read_current(series, out, prov_path):
        log.info("%s: read is current; not read again", series)
    else:
        _read_and_save(series, out, prov_path, sheets=sheets, client=client, cache_only=cache_only)
        if not partial:
            fresh.record(out, read_inputs(series))
    if info.role == "height":
        if partial:
            return out.parent
        if _raw_path(zs).exists():
            _publish_full(zs, force=force)
        return out.parent
    if partial:
        return publish(series, pd.read_parquet(out), info.year, partial=True)
    return _publish_full(series, force=force)


def _read_and_save(series: str, out: Path, prov_path: Path, *, sheets, client, cache_only: bool) -> None:
    """Read a series and save the read (`out`, its answers sidecar and its provenance)."""
    from sfzmaps import answers, llm
    from sfzmaps.districts import regionread

    skipped0 = regionread.Stats.skipped
    # everything the read was made from: pattern / region reads, plus the answers of the legend
    # and district-label artifacts it consumed (answers.consume), fresh or not
    with llm.cache_only(cache_only or llm.CACHE_ONLY), llm.recording() as keys:
        raw = read_raw(series, sheets=sheets, client=client)
    prov = provenance(series, cache_only=cache_only or llm.CACHE_ONLY,
                      region_misses=regionread.Stats.skipped - skipped0)
    if sheets:
        prov["sheets"] = list(sheets)
    if SERIES[series].role == "height":
        raw = raw[["blklot", "height_code", "confidence", "source_sheet"]]
    out.parent.mkdir(parents=True, exist_ok=True)
    answers.write(out, keys)                               # sidecar first
    raw.to_parquet(out)
    _write_json(prov_path, prov)


def republish(series: str, *, force: bool = False) -> Path:
    """Re-apply overrides / height / rules to the last read without reading again (no vision
    calls), unless the outputs are current.  A height series republishes its period's zoning
    series."""
    if series != "today" and SERIES[series].role == "height":
        series = zoning_series_of(series)
    return _publish_full(series, force=force)


# ---------------------------------------------------------------- backfill

def _git_at(ts: float) -> str:
    """Short sha of the last commit made before a timestamp ("unknown" outside git)."""
    when = datetime.fromtimestamp(ts).astimezone().isoformat(timespec="seconds")
    try:
        sha = subprocess.run(["git", "log", "-1", f"--before={when}", "--format=%h"],
                             cwd=config.PIPELINE_DIR, capture_output=True, text=True,
                             check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return sha or "unknown"


def backfill_provenance(region_read: dict[str, bool], notes: dict[str, str] | None = None) -> list[Path]:
    """Write provenance for reads made before it was recorded, from the read files' mtimes:
    read_at = mtime, git = last commit before it (work-tree state unknown), region_read from
    `region_read` (series -> bool; height series name their height read).  Existing
    provenance files are never overwritten.  Returns the files written."""
    notes = notes or {}
    written = []
    for s, info in [*SERIES.items(), ("today", None)]:
        if info is not None and info.role == "skip":
            continue
        if info is not None and info.role == "height":
            zs = zoning_series_of(s)
            src, dst = out_dir(zs) / "height.parquet", _height_prov_path(zs)
        elif s == "today":
            src, dst = out_dir("today") / "stats.json", _prov_path("today")
        else:
            src, dst = _raw_path(s), _prov_path(s)
        if dst.exists() or not src.exists():
            continue
        ts = src.stat().st_mtime
        prov = {"reader": info.reader if info else "datasf",
                "region_read": bool(region_read.get(s, False)), "git": _git_at(ts),
                "read_at": datetime.fromtimestamp(ts).astimezone().isoformat(timespec="seconds"),
                "backfilled": True}
        if s in notes:
            prov["note"] = notes[s]
        _write_json(dst, prov)
        written.append(dst)
    return written


# ---------------------------------------------------------------- status

def needs_rerun(prov: dict | None, reader: str) -> bool:
    """A reader-B read made without per-region reads (or cache-only with uncached regions),
    or no recorded read at all, is not the current method's answer."""
    if prov is None:
        return True
    if reader != "B":
        return False
    return not prov.get("region_read") or int(prov.get("region_misses", 0) or 0) > 0


def status_rows() -> list[dict]:
    """One row per published (non-height, non-skip) series and today: use and height
    provenance and whether either read needs rerunning with the current method."""
    rows = []
    for s, info in [*[(s, i) for s, i in SERIES.items() if i.role == "use"], ("today", None)]:
        prov = read_provenance(s)
        hs = height_series_of(s) if info else None
        use_rr = needs_rerun(prov["use"], info.reader if info else "datasf")
        h_rr = needs_rerun(prov["height"], "B") if hs else False
        rows.append({"series": s, "use": prov["use"], "height": prov["height"] if hs else None,
                     "has_height": bool(hs), "needs_rerun": use_rr or h_rr,
                     "rerun": [k for k, v in (("use", use_rr), ("height", h_rr)) if v]})
    return rows


def _fmt_prov(p: dict | None) -> str:
    if p is None:
        return "none"
    reader = p.get("reader", "?")
    rr = {"A": "pattern", "datasf": "datasf"}.get(reader) or \
        ("region-read" if p.get("region_read") else "inherit")
    extra = " cache-only" if p.get("cache_only") else ""
    extra += " backfilled" if p.get("backfilled") else ""
    return f"{rr} {p.get('git', '?')} {str(p.get('read_at', '?'))[:16]}{extra}"


def format_status(rows: list[dict]) -> str:
    table = [("series", "use", "height", "needs_rerun")]
    for r in rows:
        table.append((r["series"], _fmt_prov(r["use"]), _fmt_prov(r["height"]) if r["has_height"] else "-",
                      ("yes (" + ", ".join(r["rerun"]) + ")") if r["needs_rerun"] else "no"))
    w = [max(len(t[i]) for t in table) for i in range(4)]
    return "\n".join("  ".join(c.ljust(w[i]) for i, c in enumerate(t)).rstrip() for t in table)
