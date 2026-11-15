"""What a lot's size says about its district: for each code, how much more (or less) often lots of a given size have it
than lots in general.

Lot sizes are binned by log area (sq ft) at LOG_AREA_EDGES, quarter-steps from e^5 (148 sq ft) to e^14 (1.2 M sq
ft); smaller and larger lots join the end bins.  Per code, the count of lots in each bin is blended with half of each
neighbouring bin's count (the end bins blend only inward), then smoothed toward the overall size distribution with
SMOOTHING pseudo-lots spread as that distribution, so a code with few lots carries little evidence and one with
none carries none.  The evidence is the pointwise mutual information log p(bin | code) - log p(bin).

There is one fit per held-out quarter q, on the scored lots outside q (`lots.quarter`, from the lot's centroid) with
a known code in the April 2000 table as downloaded (`table.known_and_codes`; the by-eye corrections are never read).

Writes, under data/solver/labels/:
  lotsize.npz  codes  the table's known codes of scored lots, sorted (str)
               edges  LOG_AREA_EDGES (37)
               PMI    float64 (4, codes, 36): [quarter held out (nw, ne, sw, se), code, size bin], the evidence
               Placement (`solver/placement.py`) bins each lot's log area with
               clip(digitize(log(max(area, 1)), edges) - 1, 0, 35) and reads PMI[quarter][code] by bin.
"""
import geopandas as gpd
import numpy as np
import shapely

from sfzmaps.solver import paths
from sfzmaps.solver.lots import load_lots

LOG_AREA_EDGES = np.arange(5.0, 14.01, 0.25)     # bin edges of log(area / sq ft): 36 bins
SMOOTHING = float(len(LOG_AREA_EDGES) - 1)       # pseudo-lots toward the overall sizes: one per bin (36)


def size_bins(area: np.ndarray) -> np.ndarray:
    """Each lot's size bin (int): its log area (sq ft, at least 1) binned at LOG_AREA_EDGES, outside lots in the
    end bins."""
    log_area = np.log(np.maximum(np.asarray(area, float), 1.0))
    return np.clip(np.digitize(log_area, LOG_AREA_EDGES) - 1, 0, len(LOG_AREA_EDGES) - 2)


def pmi(bins: np.ndarray, code: np.ndarray, n_codes: int) -> np.ndarray:
    """The size evidence (n_codes, bins) learned from lots with size bin `bins` and code index `code`: the
    half-neighbour blend of each code's counts, smoothed toward the overall distribution, as log p(bin | code) -
    log p(bin)."""
    nb = len(LOG_AREA_EDGES) - 1
    counts = np.zeros((n_codes, nb))
    np.add.at(counts, (code, bins), 1.0)
    padded = np.pad(counts, ((0, 0), (1, 1)))
    blended = counts + 0.5 * (padded[:, :-2] + padded[:, 2:])
    overall = blended.sum(0) / blended.sum()
    given_code = (blended + SMOOTHING * overall[None, :]) / (blended.sum(1, keepdims=True) + SMOOTHING)
    return np.log(given_code) - np.log(overall)[None, :]


def pmi_by_quarter(bins: np.ndarray, code: np.ndarray, quarter: np.ndarray, n_codes: int) -> np.ndarray:
    """`pmi` once per held-out quarter (4, n_codes, bins), each from the lots outside it.  Per training lot: its
    size bin, code index and quarter (nw 0, ne 1, sw 2, se 3)."""
    return np.stack([pmi(bins[quarter != qi], code[quarter != qi], n_codes) for qi in range(len(paths.QUARTERS))])


def build_lotsize() -> dict:
    """Fit the size evidence once per held-out quarter and write lotsize.npz into LABELS_DIR.  Returns the codes, and
    per quarter how many codes have no training lots."""
    city = load_lots()
    area = shapely.area(gpd.read_parquet(paths.LOTS_DIR / "table.parquet").geometry.values)
    train = city.scored & city.known
    codes = np.array(sorted(set(city.codes[train])))
    index = {c: i for i, c in enumerate(codes)}
    code = np.array([index[c] for c in city.codes[train]], np.int64)
    evidence = pmi_by_quarter(size_bins(area[train]), code, city.quarter[train], len(codes))
    if not np.isfinite(evidence).all():
        raise ValueError("a size bin has no training lots near it: the evidence there is undefined")
    paths.LABELS_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(paths.LABELS_DIR / "lotsize.npz", codes=codes, edges=LOG_AREA_EDGES, PMI=evidence)
    held = city.quarter[train]
    return {"codes": len(codes), "codes_without_lots": {
        q: int((np.bincount(code[held != qi], minlength=len(codes)) == 0).sum()) for qi, q in enumerate(paths.QUARTERS)}}
