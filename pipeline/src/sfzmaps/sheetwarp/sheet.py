"""One placed sheet of a series, and what the warp reads from it.  Everything is read lazily and kept in memory;
nothing is written."""
import json
from dataclasses import dataclass
from functools import cached_property

import cv2
import numpy as np
import shapely
from scipy import ndimage
from skimage.morphology import skeletonize

from sfzmaps import config
from sfzmaps.districts import read
from sfzmaps.transform import SheetTransform

PLACED_FILES = ("page.png", "page.json", "transform.json", "refine.json", "blocks_px.geojson", "warped.tif",
                "warped.footprint.geojson")
EXACT_DISTANCE_BELOW_PX = 512.0      # round(d*d) of OpenCV's float32 distance d is the exact squared distance below this:
                                     # OpenCV's float32 error differs by platform (x86 vs ARM), and the room to round
                                     # right shrinks with d*d; at 512 px even an error of many float32 ulps rounds right


class NothingToFit(Exception):
    """Raised by any step that has nothing to work on; the message is the reason."""


def distance_transform(mask: np.ndarray, cap: float | None = None) -> np.ndarray:
    """`scipy.ndimage.distance_transform_edt(mask)`, the same float64 values: for each True pixel the Euclidean
    distance to the nearest False one, 0 on False pixels; with `cap`, no value above it.  Computed with OpenCV's
    exact transform, each float32 distance d brought back to scipy's value sqrt(round(d*d)) (its squared distance
    is a whole number of px).  scipy computes it where that isn't exact: a distance not capped that reaches
    EXACT_DISTANCE_BELOW_PX, and a mask with no False pixel (OpenCV has nothing to measure from; scipy measures from
    just outside the corner)."""
    exact = not mask.all()
    if exact:
        d = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE).astype(np.float64)
        d = np.sqrt(np.rint(d * d))
        exact = (cap is not None and cap < EXACT_DISTANCE_BELOW_PX) or d.max() < EXACT_DISTANCE_BELOW_PX
    if not exact:
        d = ndimage.distance_transform_edt(mask)
    return d if cap is None else np.minimum(d, cap)


def placed_sheets(series: str) -> list[str]:
    """Sorted names of the sheets of the series that have every file of PLACED_FILES."""
    folder = config.WORK_DIR / series
    if not folder.is_dir():
        return []
    return sorted(d.name for d in folder.iterdir() if d.is_dir() and all((d / f).exists() for f in PLACED_FILES))


@dataclass
class Sheet:
    series: str
    name: str

    @property
    def folder(self):
        return config.WORK_DIR / self.series / self.name

    @cached_property
    def transform(self) -> SheetTransform:
        """today's placement"""
        return SheetTransform.from_json(json.loads((self.folder / "transform.json").read_text()))

    @cached_property
    def page(self) -> np.ndarray:
        return cv2.imread(str(self.folder / "page.png"), cv2.IMREAD_GRAYSCALE)

    @cached_property
    def ink(self) -> np.ndarray:
        return self.page < 128

    @cached_property
    def distance_to_paper(self) -> np.ndarray:
        """per page pixel, the distance (px) to the nearest paper pixel: 0 on paper; on the ink's centre line, the
        ink's half width"""
        return distance_transform(self.ink)

    @cached_property
    def year(self):
        """the year of the lots the sheet is fitted to"""
        return read.series_year(self.series)

    @cached_property
    def lots(self):
        return read.load_parcels(year=self.year)

    @cached_property
    def ft_per_px(self) -> float:
        """the scan's scale: map feet per page pixel (today's placement, median over the page)"""
        return float(self.transform.median_scale_ft_per_px(self.page.shape))

    @cached_property
    def block_line_half_width(self) -> float:
        """the block lines' half width (px): the 75th percentile of the ink's half width along its centre line,
        where today's placement puts the block outlines (blocks_px.geojson, within 1 px).  Wider ink is heavier
        than a block line."""
        ink = self.ink
        m = np.zeros(ink.shape, np.uint8)
        for f in json.loads((self.folder / "blocks_px.geojson").read_text())["features"]:
            for q in shapely.get_parts(shapely.geometry.shape(f["geometry"])):
                if q.geom_type == "Polygon":
                    cv2.polylines(m, [np.round(np.asarray(q.exterior.coords)[:, :2]).astype(np.int32)], True, 1, 3)
        hw = self.distance_to_paper[skeletonize(ink) & (m > 0)]
        return float(np.percentile(hw, 75)) if len(hw) else 2.0

    def area(self, cuts: int = 2):
        """the area the warp measures: the sheet's drawn map (its warped footprint) less the Presidio (block 1300,
        and 8 ft round it).  The block-side steps cut the Presidio out twice and the street edges and later blocks
        once: each cut rounds the outline's corners a little differently, and every fit was made this way."""
        presidio = self.lots[self.lots["block_num"] == "1300"].union_all().buffer(8.0)
        area = shapely.from_geojson((self.folder / "warped.footprint.geojson").read_text())
        for _ in range(cuts):
            area = area.difference(presidio)
        return area
