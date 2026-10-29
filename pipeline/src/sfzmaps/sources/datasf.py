"""DataSF parcels/streets -> assessor blocks, block corners, street intersections."""
from __future__ import annotations

import io
import math
import re
from dataclasses import dataclass
from pathlib import Path

import geopandas as gpd
import httpx
import numpy as np
import pandas as pd
from shapely.geometry import MultiPolygon, Point, Polygon
from shapely.ops import unary_union

from sfzmaps import config
from sfzmaps.config import REF_DIR, WORK_CRS

PARCELS_URL = "https://data.sf.gov/resource/acdm-wktn.geojson"
STREETS_URL = "https://data.sf.gov/resource/3psu-pn9h.geojson"
PAGE_SIZE = 50000
MIN_AREA_SQFT = 50.0
SIMPLIFY_FT = 2.0
CORNER_ANGLE_DEG = 30.0
NODE_CLUSTER_FT = 5.0
MAX_PAGES = 100

PARCEL_COLS = ["block_num", "lot_num", "blklot", "active", "date_rec_drop", "date_map_drop"]
STREET_COLS = ["cnn", "street", "st_type", "streetname", "f_node_cnn", "t_node_cnn", "active"]

_SUFFIXES = {
    "ST", "STREET", "AVE", "AVENUE", "BLVD", "BOULEVARD", "RD", "ROAD", "DR", "DRIVE",
    "CT", "COURT", "PL", "PLACE", "LN", "LANE", "TER", "TERRACE", "WAY", "ALY", "ALLEY",
    "CIR", "CIRCLE", "HWY",
}
_UNITS = {"FIRST": 1, "SECOND": 2, "THIRD": 3, "FOURTH": 4, "FIFTH": 5, "SIXTH": 6,
          "SEVENTH": 7, "EIGHTH": 8, "NINTH": 9}
_TEENS = {"TENTH": 10, "ELEVENTH": 11, "TWELFTH": 12, "THIRTEENTH": 13, "FOURTEENTH": 14,
          "FIFTEENTH": 15, "SIXTEENTH": 16, "SEVENTEENTH": 17, "EIGHTEENTH": 18,
          "NINETEENTH": 19}
_TENS_ORD = {"TWENTIETH": 20, "THIRTIETH": 30, "FORTIETH": 40}
_TENS = {"TWENTY": 20, "THIRTY": 30, "FORTY": 40}


def _ordinal(n: int) -> str:
    suf = "TH" if 10 <= n % 100 <= 20 else {1: "ST", 2: "ND", 3: "RD"}.get(n % 10, "TH")
    return f"{n}{suf}"


def normalize_street(name: str, strip_type: bool = True) -> str:
    s = re.sub(r"[^A-Z0-9 ]", " ", name.upper().replace("-", " "))
    toks = s.split()
    # Spelled-out ordinals ("TWENTY FIFTH" -> 25TH).
    out: list[str] = []
    i = 0
    while i < len(toks):
        t = toks[i]
        if t in _TENS and i + 1 < len(toks) and toks[i + 1] in _UNITS:
            out.append(_ordinal(_TENS[t] + _UNITS[toks[i + 1]]))
            i += 2
            continue
        n = _UNITS.get(t) or _TEENS.get(t) or _TENS_ORD.get(t)
        if n:
            t = _ordinal(n)
        elif re.fullmatch(r"0+\d+(ST|ND|RD|TH)", t):
            t = t.lstrip("0")
        out.append(t)
        i += 1
    if strip_type and len(out) > 1 and out[-1] in _SUFFIXES:
        out.pop()
    return " ".join(out)


def _norm_block(b: str) -> str:
    digits = re.sub(r"\D", "", str(b))
    return digits.zfill(4)[:4]


def _clean_polygon(poly: Polygon) -> Polygon:
    holes = [r for r in poly.interiors if Polygon(r).area >= MIN_AREA_SQFT]
    return Polygon(poly.exterior, holes)


def build_blocks(parcels: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Dissolve active parcels by block; drop holes and slivers under 50 sq ft."""
    p = parcels[parcels["active"].astype(str).str.lower() == "true"].copy()
    p["block_num"] = p["block_num"].map(_norm_block)
    p = p[p.geometry.notna() & ~p.geometry.is_empty]
    rows = []
    for block, grp in p.groupby("block_num"):
        u = unary_union(list(grp.geometry))
        parts = [u] if isinstance(u, Polygon) else list(getattr(u, "geoms", []))
        parts = [_clean_polygon(g) for g in parts if isinstance(g, Polygon)]
        parts = [g for g in parts if g.area >= MIN_AREA_SQFT]
        if not parts:
            continue
        geom = parts[0] if len(parts) == 1 else MultiPolygon(parts)
        rows.append({"block_num": block, "geometry": geom})
    return gpd.GeoDataFrame(rows, columns=["block_num", "geometry"], geometry="geometry",
                            crs=parcels.crs)


def _ring_corners(ring_coords: list[tuple[float, float]]) -> list[tuple[float, float]]:
    pts = np.asarray(ring_coords[:-1], dtype=float)
    n = len(pts)
    out = []
    for i in range(n):
        a, b, c = pts[i - 1], pts[i], pts[(i + 1) % n]
        v1, v2 = b - a, c - b
        if not (np.linalg.norm(v1) and np.linalg.norm(v2)):
            continue
        turn = math.degrees(abs(math.atan2(v1[0] * v2[1] - v1[1] * v2[0], v1 @ v2)))
        if turn > CORNER_ANGLE_DEG:
            out.append((b[0], b[1]))
    return out


def block_corners(blocks: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    rows = []
    for block, geom in zip(blocks["block_num"], blocks.geometry):
        simp = geom.simplify(SIMPLIFY_FT)
        parts = [simp] if isinstance(simp, Polygon) else list(getattr(simp, "geoms", []))
        idx = 0
        for part in parts:
            if not isinstance(part, Polygon) or part.is_empty:
                continue
            for x, y in _ring_corners(list(part.exterior.coords)):
                rows.append({"block_num": block, "corner_idx": idx, "geometry": Point(x, y)})
                idx += 1
    return gpd.GeoDataFrame(rows, columns=["block_num", "corner_idx", "geometry"],
                            geometry="geometry", crs=blocks.crs)


def _node(v) -> str:
    return re.sub(r"\.0+$", "", str(v))


def _street_name(street, streetname) -> str:
    # DataSF's `street` field is already type-free: normalize but don't strip again.
    if street is not None and not pd.isna(street) and str(street).strip():
        return normalize_street(str(street), strip_type=False)
    return normalize_street(str(streetname or ""))


def _line(geom):
    return geom if geom.geom_type == "LineString" else max(geom.geoms, key=lambda g: g.length)


def intersections(streets: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Nodes where >=2 distinct streets meet.

    A node is located by geometry, not by assuming f_node = first vertex: among the
    endpoints (both ends) of all segments carrying the node id, pick the point shared
    by the most segments (endpoints within NODE_CLUSTER_FT) and average that cluster.
    """
    s = streets[streets["active"].astype(str).str.lower() == "true"]
    if "streetname" not in s.columns:
        s = s.assign(streetname=None)
    ends: dict[str, list[tuple[int, tuple[float, float]]]] = {}  # declared end first
    names: dict[str, set[str]] = {}
    for i, (f, t, street, sname, geom) in enumerate(
        zip(s["f_node_cnn"], s["t_node_cnn"], s["street"], s["streetname"], s.geometry)
    ):
        if geom is None or geom.is_empty:
            continue
        c = _line(geom).coords
        first, last = (c[0][0], c[0][1]), (c[-1][0], c[-1][1])
        nm = _street_name(street, sname)
        for node, declared, other in ((f, first, last), (t, last, first)):
            if node is None or pd.isna(node):
                continue
            k = _node(node)
            ends.setdefault(k, []).extend([(i, declared), (i, other)])
            names.setdefault(k, set()).add(nm)
    rows = []
    for k, cand in ends.items():
        if len(names[k]) < 2:
            continue
        xy = np.array([p for _, p in cand])
        seg = np.array([i for i, _ in cand])
        best, best_n = None, -1
        for p in xy:
            near = np.hypot(*(xy - p).T) <= NODE_CLUSTER_FT
            n = len(set(seg[near]))
            if n > best_n:
                best, best_n = near, n
        a = xy[best].mean(axis=0)
        rows.append({"node_cnn": k, "streets": sorted(names[k]), "geometry": Point(a[0], a[1])})
    return gpd.GeoDataFrame(rows, columns=["node_cnn", "streets", "geometry"],
                            geometry="geometry", crs=streets.crs)


def _page_all(fetch_page, page_size: int = PAGE_SIZE, max_pages: int = MAX_PAGES,
              label: str = "dataset") -> list[gpd.GeoDataFrame]:
    """Page via fetch_page(offset, limit); stop on an empty or short page."""
    frames: list[gpd.GeoDataFrame] = []
    for n in range(max_pages):
        gdf = fetch_page(n * page_size, page_size)
        if len(gdf) == 0:
            if n == 0:
                raise RuntimeError(f"{label}: first page came back empty")
            return frames
        frames.append(gdf)
        if len(gdf) < page_size:
            return frames
    raise RuntimeError(f"{label}: exceeded {max_pages} pages")


def _fetch_all(url: str, columns: list[str], geom_col: str) -> gpd.GeoDataFrame:
    with httpx.Client(timeout=300, follow_redirects=True) as client:
        def fetch_page(offset: int, limit: int) -> gpd.GeoDataFrame:
            params = {"$limit": limit, "$offset": offset, "$order": ":id",
                      "$select": ",".join(columns + [geom_col])}
            r = client.get(url, params=params)
            r.raise_for_status()
            return gpd.read_file(io.BytesIO(r.content))

        frames = _page_all(fetch_page, label=url)
    gdf = pd.concat(frames, ignore_index=True)
    gdf = gpd.GeoDataFrame(gdf, geometry="geometry", crs=frames[0].crs or "EPSG:4326")
    return gdf.to_crs(WORK_CRS)


def fetch_reference(dest: Path = REF_DIR) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    parcels = _fetch_all(PARCELS_URL, PARCEL_COLS, "shape")
    parcels.to_parquet(dest / "parcels.parquet")
    streets = _fetch_all(STREETS_URL, STREET_COLS, "line")
    streets.to_parquet(dest / "streets.parquet")


@dataclass
class Reference:
    blocks: gpd.GeoDataFrame
    corners: gpd.GeoDataFrame
    intersections: gpd.GeoDataFrame
    streets: gpd.GeoDataFrame


def load_reference(src: Path | None = None) -> Reference:
    src = src or config.REF_DIR
    parcels = gpd.read_parquet(src / "parcels.parquet")
    streets = gpd.read_parquet(src / "streets.parquet")
    blocks = build_blocks(parcels)
    return Reference(blocks, block_corners(blocks), intersections(streets), streets)
