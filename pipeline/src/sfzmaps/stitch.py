"""Stitch a series' warped sheets into one VRT (overlaps resolved by sheet quality).

Paper is transparent, so overlapping sheets would show both sheets' ink.  Every sheet gets a
cutline: its footprint (warped.footprint.geojson written by warp.warp_sheet: the neatline
interior minus mask boxes, EPSG:2227; the raster bounds when that file is missing) minus the
union of the footprints of all higher-priority sheets (draw_order: passed over needs_review,
then lower holdout_p50_ft on top).  A sheet that overlaps a higher-priority one is re-warped
with `gdalwarp -cutline` onto its own pixel grid (nearest neighbour, so no resampling) into
OUT_DIR/<series>/cut/<sheet>.tif; the others are used as they are.  The VRT lists the
resulting rasters in draw order, so it also stays correct if a cutline is imperfect.
"""
from __future__ import annotations

import json
import math
import re
import shutil
import subprocess

from pathlib import Path

from sfzmaps import config, manifest
from sfzmaps.score import SheetScore


def load_scores(series: str) -> dict[str, SheetScore]:
    """Scores from each sheet dir's score.json (SheetScore.empty('no_score') when missing)."""
    from sfzmaps import runner

    scores: dict[str, SheetScore] = {}
    base = config.WORK_DIR / series
    for d in sorted(p for p in base.iterdir() if p.is_dir()) if base.exists() else []:
        sj = d / "score.json"
        try:
            scores[d.name] = runner._score_load(json.loads(sj.read_text()))
        except (OSError, ValueError, KeyError):
            scores[d.name] = SheetScore.empty("no_score")
    return scores


# Paper mode: a lower sheet runs this far under the sheet above it, so footprint tracing
# error (traced at most FOOTPRINT_MAX_PX wide, ~25 ft per step) never opens a gap between them.
PAPER_OVERLAP_FT = 25.0
# Paper mode: plain paper drawn under every sheet, filling the gaps between sheets narrower than
# this many source px (each sheet's border band, warp.NEATLINE_INSET_PX, is cut on both sides
# of a shared neatline).  It is underneath, so it never covers a sheet's ink.
PAPER_BACKING_GAP_PX = 70


def raster_name(paper: bool) -> str:
    """Per-sheet warp to stitch: the transparent-paper warp, or the paper one (warp.make_rgba)."""
    return "warped_paper.tif" if paper else "warped.tif"


def draw_order(series: str, scores: dict[str, SheetScore], paper: bool = False) -> list[str]:
    """Sheet ids with a warped.tif, bottom first: needs_review under passing/reviewed sheets,
    and within a class the larger holdout_p50_ft first (so the best ends up on top)."""
    def key(sid: str):
        sc = scores[sid]
        top = manifest.status_of(series, sid, sc) != "needs_review"
        p50 = sc.holdout_p50_ft
        return (top, -(p50 if math.isfinite(p50) else math.inf), sid)

    have = [s for s in scores if (config.WORK_DIR / series / s / raster_name(paper)).exists()]
    return sorted(have, key=key)


def footprint(series: str, sheet_id: str, paper: bool = False):
    """Shapely footprint (EPSG:2227) of a sheet's warped.tif (or warped_paper.tif)."""
    from shapely.geometry import box, shape

    d = config.WORK_DIR / series / sheet_id
    fp = d / raster_name(paper).replace(".tif", ".footprint.geojson")
    try:
        feats = json.loads(fp.read_text())["features"]
        from shapely.ops import unary_union
        g = unary_union([shape(f["geometry"]) for f in feats])
        if not g.is_empty:
            return g
    except (OSError, ValueError, KeyError):
        pass
    import rasterio
    with rasterio.open(d / raster_name(paper)) as ds:
        return box(*ds.bounds)


def polygonal(geom):
    """Only the polygon parts of geom (a difference can return a GeometryCollection with
    slivers as lines/points, which gdalwarp rejects as a cutline); empty if none."""
    from shapely.geometry import MultiPolygon, Polygon

    if isinstance(geom, (Polygon, MultiPolygon)):
        return geom
    polys = []
    for g in getattr(geom, "geoms", []):
        g = polygonal(g)
        if not g.is_empty:
            polys.extend(g.geoms if isinstance(g, MultiPolygon) else [g])
    return MultiPolygon(polys) if polys else Polygon()


def _cut(series: str, sheet_id: str, geom, out: Path, paper: bool = False) -> None:
    import geopandas as gpd
    import rasterio

    src = config.WORK_DIR / series / sheet_id / raster_name(paper)
    cl = out.with_suffix(".cutline.geojson")
    gpd.GeoDataFrame(geometry=[geom], crs=config.WORK_CRS).to_file(cl, driver="GeoJSON")
    with rasterio.open(src) as ds:
        b, res = ds.bounds, ds.res
    subprocess.run(["gdalwarp", "-overwrite", "-q", "-r", "near", "-te", str(b.left), str(b.bottom), str(b.right), str(b.top),
                    "-tr", repr(res[0]), repr(res[1]), "-cutline", str(cl), "-multi", "-wo", "NUM_THREADS=ALL_CPUS",
                    "-co", "COMPRESS=DEFLATE", "-co", "TILED=YES", "-co", "BIGTIFF=IF_SAFER",
                    str(src), str(out)], check=True, capture_output=True)
    cl.unlink(missing_ok=True)


def cut_sources(series: str, order: list[str], paper: bool = False) -> list[Path]:
    """Raster per sheet of `order` (bottom first) after the cutlines; sheets whose cut
    footprint is empty are dropped.  With opaque paper the cutlines are what keep a lower
    sheet's margin from covering a better neighbour."""
    from shapely.ops import unary_union

    cutdir = config.OUT_DIR / series / ("cut-paper" if paper else "cut")
    shutil.rmtree(cutdir, ignore_errors=True)
    fps = {s: footprint(series, s, paper) for s in order}
    out = []
    for i, sid in enumerate(order):
        above = unary_union([fps[o] for o in order[i + 1:]]) if i + 1 < len(order) else None
        src = config.WORK_DIR / series / sid / raster_name(paper)
        if not paper and (above is None or not fps[sid].intersects(above)):
            out.append(src)
            continue
        if paper:   # always clipped to its content: this is the ink layer over the blank paper
            geom = polygonal(fps[sid] if above is None else fps[sid].difference(above.buffer(-PAPER_OVERLAP_FT)))
        else:
            geom = polygonal(fps[sid].difference(above))
        if geom.is_empty or geom.area < 1.0:
            continue
        cutdir.mkdir(parents=True, exist_ok=True)
        _cut(series, sid, geom, cutdir / f"{sid}.tif", paper)
        out.append(cutdir / f"{sid}.tif")
    return out


def paper_backing(series: str, order: list[str]) -> Path | None:
    """RGBA GeoTIFF of plain paper (the sheets' median paper colour) over the union of the
    sheets' paper footprints closed by PAPER_BACKING_GAP_PX: the narrow gaps between abutting
    sheets, and nothing beyond the sheets' outline.  None if there is nothing to fill."""
    import statistics

    import geopandas as gpd
    import rasterio
    from rasterio.enums import ColorInterp
    from shapely.ops import unary_union

    fps, colours, res = [], [], []
    for sid in order:
        fps.append(footprint(series, sid, True))
        try:
            props = json.loads((config.WORK_DIR / series / sid / "warped_paper.footprint.geojson").read_text())["features"][0]["properties"]
            colours.append(props["paper_rgb"])
            res.append(float(props["res_ft"]))
        except (OSError, ValueError, KeyError, IndexError):
            pass
    if not fps or not res:
        return None
    r = statistics.median(res)
    gap = PAPER_BACKING_GAP_PX * r
    geom = polygonal(unary_union(fps).buffer(gap, join_style="mitre").buffer(-gap, join_style="mitre"))
    if geom.is_empty:
        return None
    rgb = [int(statistics.median(c[i] for c in colours)) for i in range(3)] if colours else [255, 255, 255]
    out = config.OUT_DIR / series / "cut-paper" / "_backing.tif"
    out.parent.mkdir(parents=True, exist_ok=True)
    cl = out.with_suffix(".geojson")
    gpd.GeoDataFrame(geometry=[geom], crs=config.WORK_CRS).to_file(cl, driver="GeoJSON")
    b = geom.bounds
    subprocess.run(["gdal_rasterize", "-q", *sum((["-burn", str(v)] for v in [*rgb, 255]), []), "-ot", "Byte",
                    "-init", "0", "-a_srs", config.WORK_CRS, "-tr", repr(r), repr(r), "-te", *map(str, b),
                    "-co", "COMPRESS=DEFLATE", "-co", "TILED=YES", "-co", "BIGTIFF=IF_SAFER", str(cl), str(out)],
                   check=True, capture_output=True)
    cl.unlink(missing_ok=True)
    with rasterio.open(out, "r+") as ds:
        ds.colorinterp = [ColorInterp.red, ColorInterp.green, ColorInterp.blue, ColorInterp.alpha]
        ds.nodata = None   # the alpha band is the mask; a nodata of 0 would also blank black ink
    return out


# Sheets the 1964-1980 books bind beside the zoning sheets: height (1H, 13SH), special-use
# (2SU, 1SUa) and sign-district (SSD) maps.  They are placed like any sheet, but a zoning
# edition's map layers show only its zoning sheets.
_NON_ZONING_SHEET = re.compile(r"\d*(H|SH|SU[a-z]?|SSD)")


def layer_sheets(series: str, order: list[str]) -> list[str]:
    """`order` without the non-zoning sheets when `series` is a zoning (use) edition; height
    and other series keep every sheet.  An unread sheet number keeps the sheet."""
    from sfzmaps.districts.series import SERIES

    info = SERIES.get(series)
    if info is None or info.role != "use":
        return order

    def number(sid: str) -> str:
        try:
            n = json.loads((config.WORK_DIR / series / sid / "page.json").read_text()).get("sheet_number")
        except (OSError, ValueError):
            return ""
        return str(n or "").strip()

    return [s for s in order if not _NON_ZONING_SHEET.fullmatch(number(s))]


def build_vrt(series: str, scores: dict[str, SheetScore] | None = None, paper: bool = False) -> Path:
    scores = load_scores(series) if scores is None else scores
    order = layer_sheets(series, draw_order(series, scores, paper))
    if not order:
        raise RuntimeError(f"no warped sheets for series {series}; run `sfzmaps run --year {series}`")
    out = config.OUT_DIR / series / (f"{series}-paper.vrt" if paper else f"{series}.vrt")
    out.parent.mkdir(parents=True, exist_ok=True)
    lst = out.with_suffix(".txt")
    srcs = cut_sources(series, order, paper)
    if paper:
        # bottom to top: plain-paper backing (gaps between abutting sheets), every sheet's
        # whole paper (blank map is real map), then every sheet's inked content in priority
        # order, so blank paper never hides a neighbour's ink
        backing = paper_backing(series, order)
        whole = [config.WORK_DIR / series / s / raster_name(True) for s in order]
        srcs = ([backing] if backing else []) + whole + srcs
    lst.write_text("\n".join(str(p) for p in srcs) + "\n")
    # The RGBA alpha band is honored as the mask; the last listed source wins in overlaps.
    # -srcnodata/-vrtnodata None: the alpha bands are the masks.  Without them a source with a
    # nodata value makes every (0, 0, 0) pixel, i.e. pure black ink, transparent.
    subprocess.run(["gdalbuildvrt", "-overwrite", "-resolution", "highest", "-srcnodata", "None", "-vrtnodata", "None",
                    "-input_file_list", str(lst), str(out)], check=True, capture_output=True)
    return out
