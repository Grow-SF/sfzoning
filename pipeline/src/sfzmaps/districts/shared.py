"""One set of shapes holding every era's districts, so the explorer can switch years instantly.

Every series classifies the same DataSF lots, so its parcels have the same geometry, and most of
its district lines run along the same streets. The tiles store each shape once with a number, and
a small table says what each number looks like in each year:

  out/shared/districts-<hash>.pmtiles       lots (`i`) and district lines (`s`), geometry only
  out/shared/districts-<hash>.table.gz      each era's category per lot, and which line sets it draws
  out/shared/districts-<hash>.cards.json.gz each era's code, height and confidence, for the lot card
  out/shared/districts.json                 the pointer the site reads

All are named by their inputs, so none is ever overwritten in place.

The per-series outputs stay the source of truth (and the story, stats and review pages use them).
Rebuild after publishing a series; `stale_series()` says when that is due. Makes no API calls.
"""
import gzip
import hashlib
import json
import logging
import subprocess
from pathlib import Path

import geopandas as gpd
import mapbox_vector_tile
import numpy as np
import pandas as pd
import pyogrio
import shapely
from pmtiles.reader import MmapSource, all_tiles

from sfzmaps import config
from sfzmaps.districts.assemble import MAX_ZOOM, MIN_ZOOM, _has_pmtiles

log = logging.getLogger(__name__)

# Part of the archive's name: raise it when the same inputs would now build a different archive.
ARCHIVE_FORMAT = 4
# The site refuses a pointer whose version it doesn't know, and uses the per-year files instead.
MANIFEST_VERSION = 2
# Same order as the table's numbers; the pointer carries it, so the site never assumes it.
CATEGORIES = ("single_family", "two_unit", "three_unit", "multi_family", "mixed", "commercial",
              "industrial", "public", "unrestricted", "unknown")
NO_VALUE = 255
# District-line vertices match across years to a tenth of a foot (EPSG:2227 is in feet).
EDGE_SNAP = 10

# Well above the largest tile (22 MB raw at z11-12), so the writer never thins a tile to fit: at
# the per-series limit, 1948's z11 tile stops at 128,000 parcels and loses its district lines.
MAX_TILE_BYTES = 64_000_000
MAX_TILE_FEATURES = 1_000_000
# Lots too small for a zoom's tile resolution drop out: at z16, two slivers of 2-4 m2 do.
MIN_LOT_COVERAGE = {z: 0.9 for z in range(MIN_ZOOM, MAX_ZOOM)} | {MAX_ZOOM: 0.9999}


def shared_dir() -> Path:
    return config.OUT_DIR / "shared"


def _series_dir(series: str) -> Path:
    return config.OUT_DIR / series / "districts"


def era_key(prefix: str, series: str) -> str:
    return f"{prefix}_{series.replace('-', '_')}"


def published_series(candidates: list[str]) -> list[str]:
    """The candidates with full published outputs (a partial run writes districts-partial/)."""
    return [s for s in candidates if (_series_dir(s) / "parcels.fgb").exists()]


def merge(series: list[str]) -> gpd.GeoDataFrame:
    """One row per footprint with every era's columns. A row's position is its lot number; where
    lots overlap, the map draws the lowest number, which is the smallest lot."""
    wkb: dict[str, bytes] = {}
    frames: list[pd.DataFrame] = []
    geoms: list[gpd.GeoDataFrame] = []
    for s in series:
        g = pyogrio.read_dataframe(_series_dir(s) / "parcels.fgb")
        for lot, shape in zip(g["blklot"], g.geometry.to_wkb()):
            if wkb.setdefault(lot, shape) != shape:
                raise ValueError(f"lot {lot} has different geometry in {s}; "
                                 "republish every series from the same parcels")
        geoms.append(g[["blklot", "geometry"]])
        frames.append(pd.DataFrame({
            "blklot": g["blklot"],
            era_key("c", s): g["category"],
            era_key("k", s): g["code"],
            era_key("h", s): g["height_code"],
            era_key("f", s): g["confidence"],
        }).set_index("blklot"))
    lots = pd.concat(geoms, ignore_index=True).drop_duplicates("blklot").set_index("blklot")
    wide = pd.concat(frames, axis=1).dropna(axis=1, how="all")
    parcels = gpd.GeoDataFrame(lots.join(wide).reset_index(), geometry="geometry", crs=geoms[0].crs)
    return _numbered(_one_per_footprint(parcels))


def _numbered(lots: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Lots that overlap another come first, smallest first; the rest keep their order, which runs
    block by block, because neighbours' similar values make the table compress several times better."""
    area = lots.geometry.area.to_numpy()
    a, b = lots.sindex.query(lots.geometry, predicate="intersects")
    touching = a != b
    overlap = shapely.area(shapely.intersection(lots.geometry.values[a[touching]], lots.geometry.values[b[touching]])) > 0
    overlapping = np.unique(a[touching][overlap])
    first = overlapping[np.argsort(area[overlapping], kind="stable")]
    rest = np.setdiff1d(np.arange(len(lots)), overlapping)
    order = np.concatenate([first, rest])
    return gpd.GeoDataFrame(lots.iloc[order].reset_index(drop=True), geometry="geometry", crs=lots.crs)


def _ring_segments(geom) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    """A polygon's edges, snapped to EDGE_SNAP and direction-free, so neighbours' edges compare equal."""
    out = []
    for ring in shapely.get_rings(shapely.get_parts(geom)):
        xy = np.round(shapely.get_coordinates(ring) * EDGE_SNAP).astype(np.int64).tolist()
        for a, b in zip(map(tuple, xy), map(tuple, xy[1:])):
            if a != b:
                out.append((a, b) if a < b else (b, a))
    return out


def district_edges(series: list[str]) -> tuple[gpd.GeoDataFrame, list[list[int]]]:
    """Every era's district lines with each stretch stored once: most edges run along the same
    streets year after year. Returns (lines tagged `s`, each set `s`'s era positions)."""
    masks: dict[tuple, int] = {}
    crs = None
    for bit, s in enumerate(series):
        districts = pyogrio.read_dataframe(_series_dir(s) / "districts.fgb")
        crs = crs or districts.crs
        for geom in districts.geometry:
            for seg in _ring_segments(geom):
                masks[seg] = masks.get(seg, 0) | (1 << bit)
    by_mask: dict[int, list] = {}
    for seg, mask in masks.items():
        by_mask.setdefault(mask, []).append(seg)
    rows, geometry, sets = [], [], []
    for mask in sorted(by_mask):
        coords = np.array(by_mask[mask], dtype=float) / EDGE_SNAP
        chains = shapely.get_parts(shapely.line_merge(shapely.multilinestrings(shapely.linestrings(coords))))
        rows += [len(sets)] * len(chains)
        geometry += list(chains)
        sets.append([e for e in range(len(series)) if mask >> e & 1])
    log.info("%d edges in %d lines across %d year sets", len(masks), len(geometry), len(sets))
    return gpd.GeoDataFrame({"s": rows}, geometry=geometry, crs=crs), sets


def _column(parcels: gpd.GeoDataFrame, name: str) -> pd.Series:
    """A value column; merge drops one that is empty for every lot (a map with no heights)."""
    return parcels[name] if name in parcels else pd.Series(index=parcels.index, dtype=object)


def colour_table(parcels: gpd.GeoDataFrame, series: list[str], sets: list[list[int]]) -> bytes:
    """What the map looks up: each era's category number per lot, then whether each era draws each
    line set. A byte each, era by era. NO_VALUE marks a lot the era doesn't cover."""
    numbers = {c: i for i, c in enumerate(CATEGORIES)}
    table = np.full((len(series), len(parcels)), NO_VALUE, dtype=np.uint8)
    for e, s in enumerate(series):
        col = _column(parcels, era_key("c", s))
        odd = set(col.dropna()) - set(CATEGORIES)
        if odd:
            raise ValueError(f"{s} has categories the map has no colour for: {sorted(odd)}")
        table[e] = col.map(numbers).fillna(NO_VALUE).astype(np.uint8).to_numpy()
    drawn = np.zeros((len(series), len(sets)), dtype=np.uint8)
    for i, eras in enumerate(sets):
        drawn[eras, i] = 1
    return table.tobytes() + drawn.tobytes()


def lot_cards(parcels: gpd.GeoDataFrame, series: list[str]) -> dict:
    """What a clicked lot's card reads, by lot number. Loaded only on the first click."""
    strings: dict[str, int] = {}

    def numbered(col: pd.Series) -> list:
        return [None if pd.isna(v) else strings.setdefault(str(v), len(strings)) for v in col]

    eras = {}
    for s in series:
        eras[s] = {
            "code": numbered(_column(parcels, era_key("k", s))),
            "height": numbered(_column(parcels, era_key("h", s))),
            "confidence": [None if pd.isna(v) else round(float(v), 2) for v in _column(parcels, era_key("f", s))],
        }
    return {"blklot": parcels["blklot"].astype(str).tolist(), "strings": list(strings), "eras": eras}


def _one_per_footprint(parcels: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Condo lots stack on one footprint. Identical in every era, they would only add memory (a
    third of it) and darken the colour by blending over themselves, so one lot stands for them."""
    values = [c for c in parcels.columns if c not in ("blklot", "geometry")]
    keys = pd.DataFrame({"_shape": parcels.geometry.to_wkb()}).join(parcels[values])
    kept = parcels[~keys.duplicated()].reset_index(drop=True)
    log.info("%d lots drawn as %d footprints", len(parcels), len(kept))
    return gpd.GeoDataFrame(kept, geometry="geometry", crs=parcels.crs)


def source_hashes(series: list[str]) -> dict[str, str]:
    out = {}
    for s in series:
        h = hashlib.sha256()
        for name in ("parcels.fgb", "districts.fgb"):
            h.update((_series_dir(s) / name).read_bytes())
        out[s] = h.hexdigest()
    return out


def archive_name(hashes: dict[str, str]) -> str:
    digest = hashlib.sha256(json.dumps([ARCHIVE_FORMAT, hashes], sort_keys=True).encode()).hexdigest()[:12]
    return f"districts-{digest}.pmtiles"


def write_manifest(series: list[str], stem: str, lots: int, line_sets: int) -> Path:
    """The pointer the site reads. Written last, so it only ever names finished files."""
    d = shared_dir()
    d.mkdir(parents=True, exist_ok=True)
    path = d / "districts.json"
    path.write_text(json.dumps({
        "version": MANIFEST_VERSION,
        "archive": f"shared/{stem}.pmtiles",
        "table": f"shared/{stem}.table.gz",
        "cards": f"shared/{stem}.cards.json.gz",
        "categories": list(CATEGORIES),
        "lots": lots,
        "lineSets": line_sets,
        "eras": series,
        "sources": source_hashes(series),
    }, indent=1))
    return path


def stale_series() -> list[str]:
    """Series whose outputs changed, or were first published, since the archive was built."""
    path = shared_dir() / "districts.json"
    if not path.exists():
        return []
    built = json.loads(path.read_text())["sources"]
    now = source_hashes(published_series(list(built)))
    return [s for s, h in built.items() if now.get(s) != h]


def lots_by_zoom(path: Path) -> dict[int, set[int]]:
    """Every lot in each zoom's tiles, decoded as the browser decodes them. (GDAL's own reader
    returns far too few lots from this archive's largest tiles, and reports no error.)"""
    found: dict[int, set[int]] = {}
    with path.open("rb") as f:
        get = MmapSource(f)
        for (z, _, _), data in all_tiles(get):
            raw = gzip.decompress(data) if data[:2] == b"\x1f\x8b" else data
            features = mapbox_vector_tile.decode(raw).get("parcels", {}).get("features", [])
            found.setdefault(z, set()).update(int(x["properties"]["i"]) for x in features)
    return found


def _check_coverage(path: Path, lots: set[int]) -> None:
    """Fail rather than publish a map that has silently lost lots at some zoom."""
    found = lots_by_zoom(path)
    for z, need in MIN_LOT_COVERAGE.items():
        share = len(found.get(z, set()) & lots) / len(lots)
        log.info("z%d holds %.3f%% of lots", z, 100 * share)
        if share < need:
            raise RuntimeError(f"z{z} holds only {share:.2%} of lots (need {need:.2%})")


def build(series: list[str]) -> Path:
    """Write the archive if these inputs have none yet, then the pointer. Returns the archive."""
    if not _has_pmtiles():
        raise RuntimeError("ogr2ogr has no PMTiles driver")
    series = published_series(series)
    if not series:
        raise ValueError("no published series to merge")
    name = archive_name(source_hashes(series))
    d = shared_dir()
    d.mkdir(parents=True, exist_ok=True)
    out = d / name
    table, cards = d / f"{out.stem}.table.gz", d / f"{out.stem}.cards.json.gz"
    meta = d / f"{out.stem}.counts.json"
    if not (out.exists() and table.exists() and cards.exists() and meta.exists()):
        parcels = merge(series)
        lines, sets = district_edges(series)
        gpkg = d / f"{out.stem}.tmp.gpkg"
        tmp = d / f"{out.stem}.tmp.pmtiles"
        for p in (gpkg, tmp):
            p.unlink(missing_ok=True)
        try:
            shapes = gpd.GeoDataFrame({"i": np.arange(len(parcels))}, geometry=parcels.geometry.values, crs=parcels.crs)
            shapes.to_crs("EPSG:3857").to_file(gpkg, layer="parcels", driver="GPKG")
            lines.to_crs("EPSG:3857").to_file(gpkg, layer="districts", driver="GPKG")
            subprocess.run(["ogr2ogr", "-f", "PMTiles", str(tmp), str(gpkg), "-t_srs", "EPSG:3857",
                            "-dsco", f"MINZOOM={MIN_ZOOM}", "-dsco", f"MAXZOOM={MAX_ZOOM}",
                            "-dsco", "NAME=SF zoning districts, all eras",
                            "-dsco", f"MAX_SIZE={MAX_TILE_BYTES}",
                            "-dsco", f"MAX_FEATURES={MAX_TILE_FEATURES}"],
                           check=True, capture_output=True)
            _check_coverage(tmp, set(range(len(parcels))))
            tmp.replace(out)
        finally:
            gpkg.unlink(missing_ok=True)
            tmp.unlink(missing_ok=True)
        table.write_bytes(gzip.compress(colour_table(parcels, series, sets), 9))
        cards.write_bytes(gzip.compress(json.dumps(lot_cards(parcels, series), separators=(",", ":")).encode(), 9))
        meta.write_text(json.dumps({"lots": len(parcels), "lineSets": len(sets)}))
    counts = json.loads(meta.read_text())
    write_manifest(series, out.stem, counts["lots"], counts["lineSets"])
    return out
