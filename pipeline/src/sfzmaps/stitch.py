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


def draw_order(series: str, scores: dict[str, SheetScore]) -> list[str]:
    """Sheet ids with a warped.tif, bottom first: needs_review under passing/reviewed sheets,
    and within a class the larger holdout_p50_ft first (so the best ends up on top)."""
    def key(sid: str):
        sc = scores[sid]
        top = manifest.status_of(series, sid, sc) != "needs_review"
        p50 = sc.holdout_p50_ft
        return (top, -(p50 if math.isfinite(p50) else math.inf), sid)

    have = [s for s in scores if (config.WORK_DIR / series / s / "warped.tif").exists()]
    return sorted(have, key=key)


def footprint(series: str, sheet_id: str):
    """Shapely footprint (EPSG:2227) of a sheet's warped.tif."""
    from shapely.geometry import box, shape

    d = config.WORK_DIR / series / sheet_id
    fp = d / "warped.footprint.geojson"
    try:
        feats = json.loads(fp.read_text())["features"]
        from shapely.ops import unary_union
        g = unary_union([shape(f["geometry"]) for f in feats])
        if not g.is_empty:
            return g
    except (OSError, ValueError, KeyError):
        pass
    import rasterio
    with rasterio.open(d / "warped.tif") as ds:
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


def _cut(series: str, sheet_id: str, geom, out: Path) -> None:
    import geopandas as gpd
    import rasterio

    src = config.WORK_DIR / series / sheet_id / "warped.tif"
    cl = out.with_suffix(".cutline.geojson")
    gpd.GeoDataFrame(geometry=[geom], crs=config.WORK_CRS).to_file(cl, driver="GeoJSON")
    with rasterio.open(src) as ds:
        b, res = ds.bounds, ds.res
    subprocess.run(["gdalwarp", "-overwrite", "-q", "-r", "near", "-te", str(b.left), str(b.bottom), str(b.right), str(b.top),
                    "-tr", repr(res[0]), repr(res[1]), "-cutline", str(cl), "-multi", "-wo", "NUM_THREADS=ALL_CPUS",
                    "-co", "COMPRESS=DEFLATE", "-co", "TILED=YES", "-co", "BIGTIFF=IF_SAFER",
                    str(src), str(out)], check=True, capture_output=True)
    cl.unlink(missing_ok=True)


def cut_sources(series: str, order: list[str]) -> list[Path]:
    """Raster per sheet of `order` (bottom first) after the cutlines; sheets whose cut
    footprint is empty are dropped."""
    from shapely.ops import unary_union

    cutdir = config.OUT_DIR / series / "cut"
    shutil.rmtree(cutdir, ignore_errors=True)
    fps = {s: footprint(series, s) for s in order}
    out = []
    for i, sid in enumerate(order):
        above = unary_union([fps[o] for o in order[i + 1:]]) if i + 1 < len(order) else None
        src = config.WORK_DIR / series / sid / "warped.tif"
        if above is None or not fps[sid].intersects(above):
            out.append(src)
            continue
        geom = polygonal(fps[sid].difference(above))
        if geom.is_empty or geom.area < 1.0:
            continue
        cutdir.mkdir(parents=True, exist_ok=True)
        _cut(series, sid, geom, cutdir / f"{sid}.tif")
        out.append(cutdir / f"{sid}.tif")
    return out


def build_vrt(series: str, scores: dict[str, SheetScore] | None = None) -> Path:
    scores = load_scores(series) if scores is None else scores
    order = draw_order(series, scores)
    if not order:
        raise RuntimeError(f"no warped sheets for series {series}; run `sfzmaps run --year {series}`")
    out = config.OUT_DIR / series / f"{series}.vrt"
    out.parent.mkdir(parents=True, exist_ok=True)
    lst = out.with_suffix(".txt")
    lst.write_text("\n".join(str(p) for p in cut_sources(series, order)) + "\n")
    # The RGBA alpha band is honored as the mask; the last listed source wins in overlaps.
    subprocess.run(["gdalbuildvrt", "-overwrite", "-resolution", "highest",
                    "-input_file_list", str(lst), str(out)], check=True, capture_output=True)
    return out
