"""Assembly: read a series (reader A / B / DataSF today), overrides, height join, rules join ->
story category, then the published outputs under out/<series>/districts/:

  parcels.fgb      one row per mapped parcel: blklot, code, height_code, category, confidence,
                   source_sheet (EPSG:2227)
  districts.fgb    parcels dissolved by code (code, category)
  <series>-districts.pmtiles   vector tiles, layers `parcels` and `districts` (EPSG:3857, z11-16)
  stats.json       {category: {parcels, area_sqft}, unknown_share, mapped_area_sqft,
                   unmapped_parcels, provenance: {use, height}} (today: also notes)
  unknown_codes.json   codes read but absent from the research table -> parcel count

A height series (role "height") instead writes out/<zoning series>/districts/height.parquet,
joined by blklot into its period's zoning output (republished if that already exists).

Provenance: every read writes {reader, region_read, git, read_at} next to its output
(work/<series>/districts_raw.json, out/<zoning series>/districts/height.json); publish copies
both into stats.json.  A --sheet read is partial: it goes to districts_raw.partial.* and
out/<series>/districts-partial/ and never touches the full-series outputs.
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

_BLOCK_RE = re.compile(r"\d{4}[A-Z]?")


def blocks_of(blklots: pd.Series, parcels: pd.DataFrame | None = None) -> pd.Series:
    """Assessor block of each blklot: the parcel table's block_num (blocks may carry a letter,
    0792T; lots may too, 2330006A, so no fixed slice of the blklot works).  Blklots not in the
    table fall back to the leading 4 digits plus a letter right after them."""
    blk = blklots.astype(str)
    try:
        p = parcels if parcels is not None else load_parcels()
        m = p.drop_duplicates("blklot").set_index("blklot")["block_num"].astype(str)
        got = blk.map(m)
    except Exception:  # noqa: BLE001 - no parcel table (unit tests): the pattern decides
        got = pd.Series([None] * len(blk), index=blk.index, dtype=object)
    fb = blk.map(lambda b: (_BLOCK_RE.match(b).group(0) if _BLOCK_RE.match(b) else b[:4]))
    return got.where(got.notna(), fb).astype(str)


def apply_overrides(df: pd.DataFrame, series: str, *, column: str = "code",
                    blocks: pd.Series | None = None) -> pd.DataFrame:
    """overrides/<series>.csv rows (blklot or block, code) set `column` (code at confidence 1;
    for a height series' file, height_code: its value column may be height_code or code).
    A block row applies to every parcel of the block (block_num); a blklot row wins over a
    block row.  Override keys that match no parcel warn."""
    path = config.DATA_DIR / "overrides" / f"{series}.csv"
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


TODAY_NOTES = (
    "Categories are read from each parcel's by-right base district in the research table. "
    "RH-1 is single_family here although San Francisco's 2022 fourplex density exception "
    "allows more units per lot in RH districts; the exception is not a base-district rule, "
    "so it is not reflected (owner decision pending).")


def publish(series: str, df: pd.DataFrame, year: int, *, partial: bool = False) -> Path:
    """Unzoned blocks -> overrides -> height join -> height overrides -> rules join ->
    category, then every output file (partial: out/<series>/districts-partial/)."""
    parcels = load_parcels()
    df = df.reset_index(drop=True)
    blocks = blocks_of(df["blklot"], parcels)
    df = apply_overrides(mask_unzoned(df, year, blocks=blocks), series, blocks=blocks)
    hp = out_dir(series) / "height.parquet"
    if partial and (out_dir(series, partial=True) / "height.parquet").exists():
        hp = out_dir(series, partial=True) / "height.parquet"
    if hp.exists():
        df = join_height(df, pd.read_parquet(hp))
    hs = height_series_of(series)
    if hs:
        df = apply_overrides(df, hs, column="height_code", blocks=blocks)
    df = categorise(df, year)
    geo = parcels[["blklot", "geometry"]].drop_duplicates("blklot")
    g = gpd.GeoDataFrame(df.merge(geo, on="blklot", how="inner"), geometry="geometry", crs=geo.crs)
    g = g[g.geometry.notna() & ~g.geometry.is_empty]       # a few DataSF parcels have no shape
    g = g[["blklot", "code", "height_code", "category", "confidence", "source_sheet", "geometry"]]
    g["confidence"] = g["confidence"].astype(float).round(3)
    d = out_dir(series, partial=partial)
    d.mkdir(parents=True, exist_ok=True)
    g.to_file(d / "parcels.fgb", driver="FlatGeobuf")
    districts = dissolve(g)
    districts.to_file(d / "districts.fgb", driver="FlatGeobuf")
    write_pmtiles(g, districts, d / f"{series}-districts.pmtiles", f"SF zoning districts {series}")
    st = stats(g)
    st["unmapped_parcels"] = int((~geo["blklot"].isin(set(g["blklot"]))).sum())
    st["provenance"] = read_provenance(series, partial=partial)
    if partial:
        st["partial"] = True
    if series == "today":
        st["notes"] = TODAY_NOTES
    (d / "stats.json").write_text(json.dumps(st, indent=1))
    unk = g[(g["category"] == "unknown") & g["code"].notna()]["code"].value_counts()
    (d / "unknown_codes.json").write_text(json.dumps({str(k): int(v) for k, v in unk.items()}, indent=1))
    return d


def run_series(series: str, *, sheets: list[str] | None = None, client=None,
               cache_only: bool = False) -> Path:
    """Read, assign, join and publish one series (or `today`); returns its districts dir.
    sheets: a partial read (districts_raw.partial.*, out/<series>/districts-partial/), which
    leaves the full-series outputs alone.  cache_only: vision answers from the disk cache only
    (region reads skip uncached mosaics; any other miss raises llm.LLMCacheMiss)."""
    from sfzmaps import llm
    from sfzmaps.districts import regionread

    if series == "today":
        _write_json(_prov_path("today"), provenance("today"))
        return publish("today", read_today(), date.today().year)
    info = SERIES[series]
    if info.role == "skip":
        raise ValueError(f"{series}: special-use series are out of scope")
    partial = bool(sheets)
    skipped0 = regionread.Stats.skipped
    with llm.cache_only(cache_only or llm.CACHE_ONLY):
        raw = read_raw(series, sheets=sheets, client=client)
    prov = provenance(series, cache_only=cache_only or llm.CACHE_ONLY,
                      region_misses=regionread.Stats.skipped - skipped0)
    if partial:
        prov["sheets"] = list(sheets)
    if info.role == "height":
        zs = zoning_series_of(series)
        d = out_dir(zs, partial=partial)
        d.mkdir(parents=True, exist_ok=True)
        raw[["blklot", "height_code", "confidence", "source_sheet"]].to_parquet(d / "height.parquet")
        _write_json(_height_prov_path(zs, partial=partial), prov)
        if partial:
            return d
        zraw = _raw_path(zs)
        if zraw.exists():
            publish(zs, pd.read_parquet(zraw), SERIES[zs].year)
        return d
    raw_path = _raw_path(series, partial=partial)
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    raw.to_parquet(raw_path)
    _write_json(_prov_path(series, partial=partial), prov)
    return publish(series, raw, info.year, partial=partial)


def republish(series: str) -> Path:
    """Re-apply overrides / height / rules to the last read without reading again (no vision
    calls).  A height series republishes its period's zoning series."""
    if series == "today":
        return publish("today", read_today(), date.today().year)
    if SERIES[series].role == "height":
        series = zoning_series_of(series)
    return publish(series, pd.read_parquet(_raw_path(series)), SERIES[series].year)


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
