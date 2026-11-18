"""Context packs: what a place knows, supplied at read time to shift the reader's probabilities (item 4 of the map-text
plan).  Optional: without a pack the reader reads the ink alone.  Nothing here trains a model; the reader's weights
hold no place.  No number here decides by hand: every spread, share and weight is measured or fitted by likelihood.

A pack (`Pack`, one parquet file per map, under PACKS_DIR) holds, in map coordinates (EPSG:2227 ft):
  streets  `name` (DataSF's `streetname`, "MISSION ST"), `street` (its base name), `type` (`st_type`, "" for none),
           `active`, and the centreline: one LineString a row, as DataSF cuts them
  blocks   `number` (the Assessor's block, "0767") and its outline (the union of the block's lots)
  codes    the era's code list: printed form -> code (each code also a form of itself)
  spreads  what is measured per fold (`Spreads`): "all" (every quarter) and, per quarter name, with that quarter left
           out: where printed words sit relative to what they name, the share of each kind's words off the pack's
           list (its escape), and each code's printed forms' shares P(form | code)
  fits     per quarter name, what was fitted with that quarter left out (`ReadFit`: mode, tau, w, temperature)
It never holds a code at a lot: on the 2000 map that is the answer key.  `build_sf_pack` refuses to add one, and a
pack's frames take no column but their own.  The file carries its layout (LAYOUT); `Pack.load` refuses another
(`PackLayoutError`).

The prior (`prior`, a `Prior`): for each kind (KIND_ORDER, the detector's `textnet.data.KINDS` order), its items, each
with its odds against the kind's escape ("a string the pack doesn't list here") and its printed forms' shares, and the
escape's mass E_k:
  zoning code   each code of the era list, flat: odds (1 - e_code) / (N e_code), e_code the measured share of code
                words off the list; its forms by the measured P(form | code)
  street name   each street j near the box, by Bayes on where the box sits (its nearest centreline: d ft from the box's
                centre, t the angle between the box's axis and the centreline there, 0 to 90 degrees):
                    odds_j = (1 - m) / m x f(d) g(t) / (2 L)  /  (1 / (90 A))
                f, g the measured densities of street-name words' distance and angle to the street they name, m the
                measured share of street-name words that name no street where they sit, L the pack's centreline length
                (a street named in proportion to its length, its label either side), A the map's area (the blocks'
                convex hull): a word naming no street here sits anywhere, at any angle.  Its forms flat.
  block number  each block b near the box:
                    odds_b = (1 - m_b) / m_b x (A / N) x [p_in / area_b inside b;  (1 - p_in) f_out(d) / (perimeter_b +
                             2 pi d) at d ft outside it]
                N the pack's blocks, p_in the measured share of block-number words inside the block they name, f_out the
                density of the others' distance outside it, spread over the ring at that distance.
  other         no list (E = 1)
The kind's escape mass is E_k = 1 / (1 + sum odds).  An item whose odds don't change 1 + odds in float64 is the escape's
(exactly: it would lift nothing), which bounds how far streets and blocks are looked for (`Pack.reach`).

The posterior (`posterior`), with a weight w_k per kind (fitted):
    log p(s | ink) + log lift(s),      lift(s) = sum_k c_k Lift_k(s) / sum_k c_k,   c_k = kind_p[k] E_k
    Lift_k(s) = 1 + sum_items P(s | item) ((1 + odds_item)^w_k - 1)
normalised over the strings named in `ink_logp` and OTHER (every string not named: its ink is 1 - the named strings',
its lift 1).  At w = 1 this is the plain mixture of the kinds' priors (each string's prior E_k + its items' masses),
over the escape's level; w tempers each item's odds as a whole before its forms share it, so a code printed several
ways is not marked down for it.  A string off the lists has lift 1 and a listed one more: the pack only ever lifts, so
a string off the list with clearly stronger ink wins.  Without a pack, or with every w 0, the posterior is the ink.

The ink (`ink_logp`): log p(s | ink) exactly by CTC (`textread.ctc.string_log_probs`) at the CTC temperature tau
(each frame's log-probabilities x tau, renormalised), mixed over the two reading directions as the reader mixes them.

What is measured (`measure_spreads`, on pairs of OUR boxes and Terra's words, whose text is what they name, each pair
weighted by the chance the box holds the word, with no IoU or other cut; the codes on Terra's code words), by weighted
EM, every word refused that is in or touching an answered block or the held-out quarter:
  - street-name words: whether each names the street it sits by (near: weighted kernel densities f(d) g(t), each
    reflected at its ends, the bandwidth by leave-one-out likelihood, leaving out every pair of the word's box, which
    sit at its one place) or not (far: the distance to the street its text
    names exponential, the angle uniform: misreads and renamed streets point miles away).  m = 1 - the near share, the
    words whose text no street prints counted far.
  - EM starts from every (weighted) decile split of the distances and the likeliest fixed point is kept.
  - block-number words: near when inside the block they name, else away (p_in = 1, f_out = 0).  The same EM with a
    near-outside density was tried on our boxes: only 2 or 3 of some 2,300 found words sit just outside their block,
    the rest of the outside words are misreads 75 ft to 6 miles off, and the likeliest fit moved the "near" density
    onto them (bandwidth 1,340 ft, reach 8 miles).  Near-outside is not identifiable from these words, so a box outside
    every block it might name is read by its ink alone (lift 1), never filtered.
  - codes: e_code the share of code words whose text no code prints; P(form | code) the forms' shares of each code's
    words (flat for a code not seen).

What a box holds (`box_targets`), the fit's target: the text of the Terra words the box overlaps, as an expectation.
The box is read as one line: the line's place across the box is drawn from the box's ink inside the words, and holds
the words whose own ink spans that place, less any two at one place along it (no one line holds both).  Each word is
held by its share across the reading (`across_share`: its ink between the box's two long sides), each of its
characters by its share along the reading (`char_shares`: its ink between the box's two ends, the characters spread
evenly along the word); the line's words are read in order, a space between two; the box holds a contiguous run of that
text.  So a box over "MISSION" and "ST." is asked for "MISSION ST.", a box over half of MISSION for its half, a box
over a block number with a code printed under it for one or the other, and a word of the next line that grazes the box
only by its share.  A run no read can be (too long for the box's CTC frames) has no chance from the ink at all: it is
counted (`dropped`), not listed.

The fit (`fit_prior`): by maximum likelihood of each box's target on the boxes of the other three quarters (log sum_t
P(t) post(t)), each box's log-likelihood weighted by the chance it holds any of its words' text times its ink's share
inside their rectangles (never the owner's answers: a box with a word in or touching an answered block, or touching the
held-out quarter, is refused), the ink calibrated apart from the context: tau on the ink alone, then w per kind on top;
unbounded searches (log tau, log w), a fit running away reported.  `joint` fits them together too.

The read-time temperature (`fit_temperature`, `tempered`): the posterior a box is read with names the strings its
prior lists, the reader's best path and OTHER, and a temperature T on that very vector is fitted by maximum likelihood
of each target's slot there (`slot_targets`: its own when named, else OTHER's).  T is cross-fitted: quarter q's T is
fitted on the other three quarters' held-out vectors (each made by the fold that left that quarter out, so q's targets
never enter T's likelihood, though they did help fit those folds' tau and w); per mode, and q keeps the mode whose NLL
there is lower after its own T.  The highest-posterior box among several compares these calibrated probabilities.

The read recipe (`read_posterior`): with the box's own quarter's `ReadFit` and spreads, prior -> ink at tau ->
posterior at w -> tempered at T.
"""
import math
import re
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from scipy import optimize

from sfzmaps.solver import paths

CRS = "EPSG:2227"
PACKS_DIR = paths.SOLVER_DIR / "experiments" / "wordread" / "packs"
KIND_ORDER = ("zoning code", "block number", "street name", "other")   # kind_p's order: textnet.data.KINDS'
CODE, BLOCK, STREET, OTHER_KIND = KIND_ORDER
OTHER = "other"            # the posterior's key for every string not named in `ink_logp` (never a reader's string)
FOLDS = ("all", *paths.QUARTERS)   # a pack's spreads: every quarter, then each quarter left out
MODES = ("sequential", "joint")    # how a fold's tau and w were fitted (`fit_prior`)
LAYOUT = 2                 # the pack file's layout: 2 adds each fold's `ReadFit`; a file of another is refused
STREET_COLS = ("name", "street", "type", "active")
BLOCK_COLS = ("number",)
SPREAD_ARRAYS = ("street_ft", "street_deg", "street_r", "block_ft", "block_r")
SPREAD_SCALARS = ("street_h", "street_deg_h", "street_away", "block_h", "block_in", "block_away", "code_away")
KERNEL_ZERO = math.sqrt(-2.0 * math.log(np.finfo(float).tiny * np.finfo(float).eps))   # a Gaussian kernel is exactly
#                            0.0 in float64 past this many bandwidths
EM_TOL = 1e-10             # EM stops when its log-likelihood rises by less (relative) or no responsibility moves more
EM_MAX_ITER = 2000         # ... or after this many iterations (reported when reached)
REACH_STEPS = 4096         # the geometric grid `Pack.reach` scans (numerical; the reach is rounded outward)
VOCAB = " &'(),-./0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"   # the reader's alphabet (`textread.model.VOCAB`)
SQRT2PI = math.sqrt(2.0 * math.pi)

ONES = ("", "FIRST", "SECOND", "THIRD", "FOURTH", "FIFTH", "SIXTH", "SEVENTH", "EIGHTH", "NINTH", "TENTH", "ELEVENTH",
        "TWELFTH", "THIRTEENTH", "FOURTEENTH", "FIFTEENTH", "SIXTEENTH", "SEVENTEENTH", "EIGHTEENTH", "NINETEENTH")
TENS = {20: ("TWENTY", "TWENTIETH"), 30: ("THIRTY", "THIRTIETH"), 40: ("FORTY", "FORTIETH"), 50: ("FIFTY", "FIFTIETH"),
        60: ("SIXTY", "SIXTIETH"), 70: ("SEVENTY", "SEVENTIETH"), 80: ("EIGHTY", "EIGHTIETH"),
        90: ("NINETY", "NINETIETH")}
ORDINAL = re.compile(r"^0*(\d+)(ST|ND|RD|TH)$")


# --- printed forms ----------------------------------------------------------------------------------------------------

def spelled(n: int) -> str | None:
    """An ordinal number as the maps spell it (23 -> "TWENTY-THIRD"), 1 to 99; None otherwise."""
    if 1 <= n < 20:
        return ONES[n]
    if 20 <= n < 100:
        tens, ones = n // 10 * 10, n % 10
        return TENS[tens][1] if ones == 0 else f"{TENS[tens][0]}-{ONES[ones]}"
    return None


def _readable(s: str) -> bool:
    return bool(s) and all(c in VOCAB for c in s)


def street_forms(street: str, kind: str) -> list[str]:
    """The strings a street's label may print (sorted): each word of its base name and the whole name, a numbered
    name without its leading zeros and spelled out ("03RD" -> "3RD", "THIRD"), the type with and without a full
    stop, and the name with the type.  Only strings in the reader's alphabet; DataSF's "UNNAMED" streets print
    nothing."""
    street = " ".join(str(street or "").upper().split())
    kind = str(kind or "").upper().strip()
    if not street or street.startswith("UNNAMED"):
        return []
    variants = [[]]
    for word in street.split(" "):
        m = ORDINAL.match(word)
        options = [word]
        if m:
            n = int(m.group(1))
            options = [f"{n}{m.group(2)}"] + ([spelled(n)] if spelled(n) else [])
        variants = [v + [o] for v in variants for o in options]
    names = [" ".join(v) for v in variants]
    out = set(names) | {w for v in variants for w in v}
    if kind:
        out |= {kind, f"{kind}."} | {f"{n} {kind}" for n in names} | {f"{n} {kind}." for n in names}
    return sorted(s for s in out if _readable(s))


def block_forms(number: str) -> list[str]:
    """How a block's number prints: without its leading zeros ("0767" -> "767"; the maps never print them)."""
    s = str(number or "").upper().strip().lstrip("0") or "0"
    return [s] if _readable(s) else []


# --- weighted kernel densities ----------------------------------------------------------------------------------------

def _kernels(at: np.ndarray, x: np.ndarray, h: float, lo, hi) -> np.ndarray:
    """Gaussian kernels of bandwidth h at `at` (rows) from `x` (columns), reflected at lo and hi (None: open)."""
    def k(c):
        z = (at[:, None] - c[None, :]) / h
        return np.exp(-0.5 * z * z) / (h * SQRT2PI)
    out = k(x)
    if lo is not None:
        out = out + k(2.0 * lo - x)
    if hi is not None:
        out = out + k(2.0 * hi - x)
    return out


def _same_group(groups, n: int) -> np.ndarray:
    """(n, n): whether rows i and j are of one group (one of our boxes); each row its own when `groups` is None."""
    if groups is None:
        return np.eye(n, dtype=bool)
    g = pd.factorize(pd.Series(list(groups)))[0]
    return g[:, None] == g[None, :]


def loo_density(x: np.ndarray, w: np.ndarray, h: float, lo=None, hi=None, groups=None) -> np.ndarray:
    """The weighted kernel density at each x_i from every point of another group (leave-one-group-out): `groups`
    (one label a row; each row its own when None) are our boxes, so a box's other rows, at its own place, never
    vouch for it."""
    x, w = np.asarray(x, float), np.asarray(w, float)
    same = _same_group(groups, len(x))
    k = np.where(same, 0.0, _kernels(x, x, h, lo, hi) * w[None, :])
    rest = w.sum() - (same * w[None, :]).sum(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(rest > 0, k.sum(1) / np.where(rest > 0, rest, 1.0), 0.0)


def loo_bandwidth(x: np.ndarray, w: np.ndarray, lo=None, hi=None, start: float | None = None, groups=None) -> float:
    """The bandwidth that maximises the weighted leave-one-group-out log-likelihood, sum_i w_i log f_-g(i)(x_i)
    (`loo_density`; `groups` our boxes), searched without bounds in log h (Brent, from `start` or the weighted
    spread); nan when fewer than two points carry weight or they don't spread."""
    x, w = np.asarray(x, float), np.asarray(w, float)
    on = w > 0
    if on.sum() < 2:
        return math.nan
    x, w = x[on], w[on]
    groups = None if groups is None else np.asarray(list(groups), object)[on]
    sd = math.sqrt(float(np.average((x - np.average(x, weights=w)) ** 2, weights=w)))
    if not sd > 0:
        return math.nan

    def nll(u):
        f = loo_density(x, w, math.exp(u), lo, hi, groups)
        with np.errstate(divide="ignore"):
            v = -float((w * np.log(f)).sum())
        return v if math.isfinite(v) else 1e100      # a bandwidth leaving a point at density 0: worse than any other

    at = math.log(start) if start is not None and math.isfinite(start) and start > 0 else None
    bracket = (at - 0.25, at) if at is not None else (math.log(sd) - 3.0, math.log(sd))
    res = optimize.minimize_scalar(nll, bracket=bracket, method="brent")
    return float(math.exp(res.x))


@dataclass(frozen=True)
class Density:
    """A weighted Gaussian kernel density on [lo, hi] (None: open), reflected at its ends; the kernels of weight 0
    are dropped (they add exactly nothing).  Zero everywhere when the bandwidth is nan."""
    x: np.ndarray
    w: np.ndarray
    h: float
    lo: float | None = 0.0
    hi: float | None = None

    @cached_property
    def _on(self) -> tuple[np.ndarray, np.ndarray]:
        on = np.asarray(self.w) > 0
        return np.asarray(self.x, float)[on], np.asarray(self.w, float)[on]

    def __call__(self, at) -> np.ndarray:
        at = np.asarray(at, float)
        x, w = self._on
        if not math.isfinite(self.h) or not len(x) or not at.size:
            return np.zeros(at.shape)
        flat = at.reshape(-1)
        out = np.concatenate([(_kernels(flat[i:i + 4096], x, self.h, self.lo, self.hi) * w).sum(1)
                              for i in range(0, len(flat), 4096)]) / w.sum()
        return out.reshape(at.shape)

    @property
    def top(self) -> float:
        """Past this, every kernel is exactly 0.0 in float64 (KERNEL_ZERO bandwidths beyond the farthest point)."""
        x, _ = self._on
        return float(x.max() + KERNEL_ZERO * self.h) if len(x) and math.isfinite(self.h) else 0.0


# --- what is measured per fold ----------------------------------------------------------------------------------------

@dataclass
class Spreads:
    """What a fold measures (see the module).  Streets: each found street-name word's distance (ft) and angle (deg) to
    the street its text names and its responsibility of naming the street it sits by (`street_r`), the near
    densities' bandwidths, and m (`street_away`).  Blocks: each found block-number word's distance outside the block
    it names (0 inside) and its responsibility, the near-outside bandwidth, p_in (`block_in`) and m_b
    (`block_away`).  Codes: e_code (`code_away`) and P(form | code) (`form_weights`).  `em`: the fits' diagnostics."""
    street_ft: np.ndarray
    street_deg: np.ndarray
    street_r: np.ndarray
    street_h: float
    street_deg_h: float
    street_away: float
    block_ft: np.ndarray
    block_r: np.ndarray
    block_h: float
    block_in: float
    block_away: float
    code_away: float
    form_weights: dict[str, float] = field(default_factory=dict)
    em: dict = field(default_factory=dict)

    def __post_init__(self):
        for k in SPREAD_ARRAYS:
            setattr(self, k, np.asarray(getattr(self, k), float))
        for k in SPREAD_SCALARS:
            setattr(self, k, float(getattr(self, k)))
        for k in ("street_away", "block_away", "code_away"):
            v = getattr(self, k)
            if not 0 < v < 1:
                raise ValueError(f"{k} must be a share strictly between 0 and 1, not {v}: the data can't support it")
        if not 0 <= self.block_in <= 1:
            raise ValueError(f"block_in must be a share, not {self.block_in}")

    @cached_property
    def street_density(self) -> Density:
        return Density(self.street_ft, self.street_r, self.street_h, 0.0, None)

    @cached_property
    def angle_density(self) -> Density:
        return Density(self.street_deg, self.street_r, self.street_deg_h, 0.0, 90.0)

    @cached_property
    def block_out_density(self) -> Density:
        out = self.block_ft > 0
        return Density(self.block_ft[out], self.block_r[out], self.block_h, 0.0, None)

    def summary(self) -> dict:
        def q(a, w):
            """Display only: the near weight and the near-weighted quantiles."""
            on = w > 0
            a, w = a[on], w[on]
            if not len(a):
                return {"near_weight": 0.0}
            o = np.argsort(a)
            cum = np.cumsum(w[o]) / w.sum()
            qs = {f"q{p}": round(float(a[o][min(np.searchsorted(cum, p / 100), len(a) - 1)]), 2) for p in (50, 90, 99)}
            return {"near_weight": round(float(w.sum()), 2), **qs}
        out = self.block_ft > 0
        return {"street_ft": q(self.street_ft, self.street_r), "street_deg": q(self.street_deg, self.street_r),
                "block_outside_ft": q(self.block_ft[out], self.block_r[out]),
                "street_bandwidth_ft": self.street_h, "street_bandwidth_deg": self.street_deg_h,
                "block_bandwidth_ft": self.block_h, "street_away": round(self.street_away, 4),
                "block_away": round(self.block_away, 4), "block_in": round(self.block_in, 4),
                "code_away": round(self.code_away, 4), "forms_weighted": len(self.form_weights), "em": self.em}


@dataclass(frozen=True)
class ReadFit:
    """A fold's fitted read (see the module): the mode it was fitted in ("sequential" or "joint"), the ink's
    temperature tau, each listing kind's weight w, and the temperature on the read-time posterior."""
    mode: str
    tau: float
    w: dict[str, float]
    temperature: float

    def __post_init__(self):
        if self.mode not in MODES:
            raise ValueError(f"a fit's mode is one of {MODES}, not {self.mode!r}")
        if not (self.tau > 0 and self.temperature > 0) or any(not v >= 0 for v in self.w.values()):
            raise ValueError(f"tau and the temperature must be > 0 and every w >= 0: {self}")
        if set(self.w) - {CODE, BLOCK, STREET}:
            raise ValueError(f"weights only for the listing kinds ({CODE}, {BLOCK}, {STREET}): {sorted(self.w)}")


class PackLayoutError(ValueError):
    """A pack file of another layout (`LAYOUT`) than this code reads: rebuild it (`build_pack.py build`, `fit`)."""


# --- the pack ---------------------------------------------------------------------------------------------------------

def _frame(gdf: gpd.GeoDataFrame, allowed: tuple[str, ...], what: str) -> gpd.GeoDataFrame:
    gdf = gdf.copy()
    extra = sorted(set(gdf.columns) - set(allowed) - {gdf.geometry.name})
    if extra:
        raise ValueError(f"a pack holds no lot codes or anything but its own fields: {what} have columns {extra}")
    if gdf.crs is None:
        raise ValueError(f"a pack's {what} need a CRS ({CRS})")
    gdf = gdf if gdf.crs == CRS else gdf.to_crs(CRS)
    return gdf.rename_geometry("geometry") if gdf.geometry.name != "geometry" else gdf


@dataclass(eq=False)
class Pack:
    """One map's context pack (see the module).  Never a code at a lot.  `fits`: each quarter's `ReadFit` (fitted
    with that quarter left out, like its spreads; `read_posterior` reads a box with its own quarter's)."""
    year: int
    streets: gpd.GeoDataFrame
    blocks: gpd.GeoDataFrame
    codes: dict[str, str]
    spreads: dict[str, Spreads] = field(default_factory=dict)
    fits: dict[str, ReadFit] = field(default_factory=dict)

    def __post_init__(self):
        bad = sorted(set(self.fits) - set(paths.QUARTERS))
        if bad:
            raise ValueError(f"a fit is a quarter's (fitted with it left out), not {bad}")
        self.streets = _frame(self.streets, STREET_COLS, "streets").reset_index(drop=True)
        self.blocks = _frame(self.blocks, BLOCK_COLS, "blocks").reset_index(drop=True)
        if "active" not in self.streets:
            self.streets["active"] = True
        self.codes = dict(sorted(self.codes.items()))

    @cached_property
    def street_tree(self) -> shapely.STRtree:
        return shapely.STRtree(self.streets.geometry.values)

    @cached_property
    def block_tree(self) -> shapely.STRtree:
        return shapely.STRtree(self.blocks.geometry.values)

    @cached_property
    def street_names(self) -> np.ndarray:
        return self.streets["name"].astype(str).to_numpy()

    @cached_property
    def forms_of_street(self) -> dict[str, list[str]]:
        """Each street name's printed forms (`street_forms` of its base name and type)."""
        out = {}
        for name, street, kind in zip(self.streets["name"], self.streets["street"], self.streets["type"]):
            if str(name) not in out:
                out[str(name)] = street_forms(street, kind)
        return out

    @cached_property
    def named_street(self) -> np.ndarray:
        """Whether each centreline's street prints anything (`forms_of_street`)."""
        return np.array([bool(self.forms_of_street[n]) for n in self.street_names], bool)

    @cached_property
    def street_length(self) -> float:
        """L: the length of every centreline whose street prints anything (ft)."""
        return float(shapely.length(self.streets.geometry.values[self.named_street]).sum())

    @cached_property
    def forms_of_block(self) -> list[list[str]]:
        return [block_forms(n) for n in self.blocks["number"]]

    @cached_property
    def named_blocks(self) -> int:
        return sum(bool(f) for f in self.forms_of_block)

    @cached_property
    def block_area(self) -> np.ndarray:
        return shapely.area(self.blocks.geometry.values)

    @cached_property
    def block_perimeter(self) -> np.ndarray:
        return shapely.length(self.blocks.geometry.values)

    @cached_property
    def area(self) -> float:
        """A: the map's area, the convex hull of the blocks (ft^2)."""
        pts = shapely.get_coordinates(self.blocks.geometry.values)
        return float(shapely.convex_hull(shapely.multipoints(pts)).area) if len(pts) else 0.0

    @cached_property
    def code_forms(self) -> dict[str, list[str]]:
        """Each code's printed forms."""
        out: dict[str, list[str]] = {}
        for form, code in self.codes.items():
            if _readable(form):
                out.setdefault(code, []).append(form)
        return out

    def form_shares(self, sp: "Spreads") -> dict[str, dict[str, float]]:
        """Each code's forms with their share P(form | code) under spreads `sp` (flat for a code it didn't see)."""
        out = {}
        for code, forms in self.code_forms.items():
            w = np.array([sp.form_weights.get(f, 0.0) for f in forms], float)
            share = w / w.sum() if w.sum() > 0 else np.full(len(forms), 1.0 / len(forms))
            out[code] = {f: float(s) for f, s in zip(forms, share) if s > 0}
        return out

    def reach(self, sp: Spreads) -> tuple[float, float]:
        """How far (ft) streets and blocks are looked for under `sp`: past it even an aligned street, or the smallest
        block, has odds that leave 1 + odds unchanged in float64 (`street_odds`, `block_odds`): it would lift nothing.
        Scanned on a geometric grid out to where every kernel is exactly 0, rounded outward."""
        key = id(sp)
        if key not in self._reaches:
            def last(grid, odds):
                keep = np.flatnonzero(1.0 + odds != 1.0)
                return float(grid[min(keep[-1] + 1, len(grid) - 1)]) if len(keep) else 0.0
            deg = np.linspace(0.0, 90.0, 721)
            best_deg = float(deg[int(np.argmax(sp.angle_density(deg)))])
            g = np.geomspace(1e-3, max(sp.street_density.top, 1e-3) * 1.001, REACH_STEPS)
            street = last(g, street_odds(self, sp, g, np.full(len(g), best_deg)))
            small = float(self.block_perimeter.min()) if len(self.blocks) else 1.0
            g = np.geomspace(1e-3, max(sp.block_out_density.top, 1e-3) * 1.001, REACH_STEPS)
            block = last(g, block_odds(self, sp, g, np.full(len(g), small), np.ones(len(g))))
            self._reaches[key] = (sp, (street, block))     # sp kept: its id stays its own
        return self._reaches[key][1]

    @cached_property
    def _reaches(self) -> dict:
        return {}

    def counts(self) -> dict:
        return {"year": self.year, "street_centrelines": int(len(self.streets)),
                "streets": int(self.streets["name"].nunique()),
                "streets_with_forms": int(sum(bool(f) for f in self.forms_of_street.values())),
                "active_centrelines": int(self.streets["active"].astype(bool).sum()),
                "street_length_ft": round(self.street_length, 1), "blocks": int(len(self.blocks)),
                "map_area_ft2": round(self.area, 1), "codes": len(self.code_forms), "code_forms": len(self.codes),
                "spreads": {k: {**s.summary(), "reach_ft": dict(zip(("street", "block"), self.reach(s)))}
                            for k, s in self.spreads.items()}}

    # the file: one parquet, a row per street centreline, block, code form, measured value and fitted value
    def save(self, f: Path) -> None:
        rows = [{"layer": "meta", "name": "year", "value": float(self.year)},
                {"layer": "meta", "name": "layout", "value": float(LAYOUT)}]
        rows += [{"layer": "street", "name": r.name, "street": r.street, "type": r.type, "active": bool(r.active),
                  "geometry": r.geometry} for r in self.streets.itertuples(index=False)]
        rows += [{"layer": "block", "name": r.number, "geometry": r.geometry}
                 for r in self.blocks.itertuples(index=False)]
        rows += [{"layer": "code", "name": code, "form": form} for form, code in self.codes.items()]
        for fold, s in self.spreads.items():
            for k in SPREAD_ARRAYS:
                rows += [{"layer": "spread", "name": k, "fold": fold, "value": float(v)} for v in getattr(s, k)]
            rows += [{"layer": "spread", "name": k, "fold": fold, "value": getattr(s, k)} for k in SPREAD_SCALARS]
            rows += [{"layer": "form_weight", "fold": fold, "form": f, "value": float(v)}
                     for f, v in sorted(s.form_weights.items())]
        for fold, fit in self.fits.items():
            rows += [{"layer": "fit", "fold": fold, "name": "mode", "form": fit.mode},
                     {"layer": "fit", "fold": fold, "name": "tau", "value": float(fit.tau)},
                     {"layer": "fit", "fold": fold, "name": "temperature", "value": float(fit.temperature)}]
            rows += [{"layer": "fit", "fold": fold, "name": "w", "form": k, "value": float(v)}
                     for k, v in sorted(fit.w.items())]
        cols = ["layer", "name", "street", "type", "active", "form", "fold", "value", "geometry"]
        df = pd.DataFrame(rows, columns=cols)
        df["geometry"] = [g if isinstance(g, shapely.Geometry) else None for g in df["geometry"]]
        for c in ("layer", "name", "street", "type", "form", "fold"):
            df[c] = df[c].astype(object)
        df["active"] = df["active"].astype("boolean")
        out = gpd.GeoDataFrame(df, geometry="geometry", crs=CRS)
        Path(f).parent.mkdir(parents=True, exist_ok=True)
        out.to_parquet(f, index=False)

    @classmethod
    def load(cls, f: Path) -> "Pack":
        """A pack file (`save`); PackLayoutError when it is of another layout than LAYOUT."""
        df = gpd.read_parquet(f)
        meta = df[df["layer"] == "meta"].set_index("name")["value"] if "layer" in df else pd.Series(dtype=float)
        layout = int(meta["layout"]) if "layout" in meta else 1           # layout 1 had no version row
        if layout != LAYOUT:
            raise PackLayoutError(f"{f} is a pack of layout {layout}; this code reads layout {LAYOUT}")
        year = int(meta["year"])
        st = df[df["layer"] == "street"]
        streets = gpd.GeoDataFrame({"name": st["name"].to_numpy(object), "street": st["street"].to_numpy(object),
                                    "type": st["type"].fillna("").to_numpy(object),
                                    "active": st["active"].astype(bool).to_numpy()},
                                   geometry=st.geometry.values, crs=df.crs)
        bl = df[df["layer"] == "block"]
        blocks = gpd.GeoDataFrame({"number": bl["name"].to_numpy(object)}, geometry=bl.geometry.values, crs=df.crs)
        co = df[df["layer"] == "code"]
        codes = dict(zip(co["form"], co["name"]))
        sp, fw = df[df["layer"] == "spread"], df[df["layer"] == "form_weight"]
        spreads = {}
        names = set(sp["fold"])
        for fold in [f for f in FOLDS if f in names] + sorted(names - set(FOLDS)):
            mine, w = sp[sp["fold"] == fold], fw[fw["fold"] == fold]
            spreads[fold] = Spreads(
                **{k: mine.loc[mine["name"] == k, "value"].to_numpy(float) for k in SPREAD_ARRAYS},
                **{k: float(mine.loc[mine["name"] == k, "value"].iloc[0]) for k in SPREAD_SCALARS},
                form_weights=dict(zip(w["form"], w["value"].astype(float))))
        ft = df[df["layer"] == "fit"]
        fits = {}
        for fold in [q for q in paths.QUARTERS if q in set(ft["fold"])]:
            mine = ft[ft["fold"] == fold]

            def one(name, col="value"):
                return mine.loc[mine["name"] == name, col].iloc[0]
            w = mine[mine["name"] == "w"]
            fits[fold] = ReadFit(mode=str(one("mode", "form")), tau=float(one("tau")),
                                 w=dict(zip(w["form"].astype(str), w["value"].astype(float))),
                                 temperature=float(one("temperature")))
        return cls(year=year, streets=streets, blocks=blocks, codes=codes, spreads=spreads, fits=fits)


# --- building San Francisco's pack ------------------------------------------------------------------------------------

def era_codes(year: int) -> dict[str, str]:
    """The era's code list as printed form -> code.  2000: the April 2000 table's code names
    (`codereader.labels.table_codes`: the list alone, never a code at a lot), each as itself, with the reader's printed
    forms of them (`text-crops/prior.json` `forms`).  No other era's list is held: an empty list (the codes' kind goes
    to the escape)."""
    if year != 2000:
        return {}
    import json

    from sfzmaps.solver.codereader import labels
    from sfzmaps.solver.textread import store
    names = set(labels.table_codes())
    out = {n: n for n in names}
    out.update({f: c for f, c in json.loads((store.CROPS_DIR / "prior.json").read_text())["forms"].items()
                if c in names})
    return dict(sorted(out.items()))


def sf_blocks(lots_table: Path | None = None) -> gpd.GeoDataFrame:
    """The Assessor's blocks: the union of each block's lots (`lots/table.parquet`'s `block_num` and geometry only;
    its codes are never read)."""
    f = lots_table or paths.LOTS_DIR / "table.parquet"
    lots = gpd.read_parquet(f, columns=["block_num", "geometry"])
    lots = lots[lots["block_num"].notna() & lots.geometry.notna() & ~lots.geometry.is_empty]
    lots = lots.assign(block_num=lots["block_num"].astype(str), geometry=shapely.make_valid(lots.geometry.values))
    out = lots.dissolve(by="block_num", sort=True).reset_index()
    out = out.rename(columns={"block_num": "number"})[["number", "geometry"]]
    return gpd.GeoDataFrame(out, geometry="geometry", crs=lots.crs).to_crs(CRS)


def sf_streets(path: Path | None = None) -> gpd.GeoDataFrame:
    """DataSF's street centrelines (`solver.streetside.load_streets`, the solver's loader), every named one."""
    from sfzmaps.solver import streetside
    s = streetside.load_streets(path)
    s = s[s["streetname"].notna() & s.geometry.notna() & ~s.geometry.is_empty]
    out = gpd.GeoDataFrame({"name": s["streetname"].astype(str).to_numpy(object),
                            "street": s["street"].fillna(s["streetname"]).astype(str).to_numpy(object),
                            "type": s["st_type"].fillna("").astype(str).to_numpy(object),
                            "active": s["active"].fillna(False).astype(bool).to_numpy()},
                           geometry=s.geometry.values, crs=s.crs)
    return out.to_crs(CRS)


def build_sf_pack(year: int, lot_codes: bool = False, codes: dict[str, str] | None = None) -> Pack:
    """San Francisco's pack for `year`'s map: today's DataSF centrelines (`sf_streets`) and the Assessor's blocks
    (`sf_blocks`), and the era's code list (`era_codes`, or `codes`).  No spreads (`measure_spreads` adds them, per
    fold).  ValueError when asked for lot codes: on the 2000 map they are the answer key, and no other era's are
    held."""
    if lot_codes:
        if year == 2000:
            raise ValueError("a 2000 pack never holds the 2000 table's codes at lots: that is the answer key")
        raise ValueError(f"no lot codes are held for {year}; a pack holds streets, blocks and the code list only")
    return Pack(year=year, streets=sf_streets(), blocks=sf_blocks(), codes=era_codes(year) if codes is None else codes)


# --- a box on the map -------------------------------------------------------------------------------------------------

def boxes_on_map(corners, transform) -> pd.DataFrame:
    """Word boxes ((n, 4, 2) page px corners, each box's first edge along its axis) on the map through a sheet's
    placement (`transform.to_world`): each centre `x`, `y` (ft) and `axis` (degrees counter-clockwise from east,
    modulo 180, the mean of the two long edges' directions on the map)."""
    c = np.asarray(corners, float).reshape(-1, 4, 2)
    w = np.asarray(transform.to_world(c.reshape(-1, 2)), float).reshape(-1, 4, 2)
    v = (w[:, 1] - w[:, 0] + w[:, 2] - w[:, 3]) / 2
    ctr = w.mean(1)
    return pd.DataFrame({"x": ctr[:, 0], "y": ctr[:, 1], "axis": np.rad2deg(np.arctan2(v[:, 1], v[:, 0])) % 180.0})


def box_on_map(corners, transform) -> dict:
    """One word box (4 page px corners) on the map (`boxes_on_map`): {x, y, axis}."""
    return {k: float(v) for k, v in boxes_on_map(np.asarray(corners, float)[None], transform).iloc[0].items()}


def _angle_between(a, b) -> np.ndarray:
    """The angle between two axes (degrees, modulo 180): 0 to 90."""
    d = np.mod(np.asarray(a, float) - np.asarray(b, float), 180.0)
    return np.minimum(d, 180.0 - d)


def _line_axes(lines, at: np.ndarray) -> np.ndarray:
    """Each centreline's axis (degrees modulo 180) at distance `at` along it: the direction of the straight piece
    holding that point (the first, at a vertex)."""
    out = np.zeros(len(lines))
    for i, (line, s) in enumerate(zip(lines, np.asarray(at, float))):
        xy = shapely.get_coordinates(line)
        step = np.diff(xy, axis=0)
        ends = np.cumsum(np.hypot(step[:, 0], step[:, 1]))
        k = min(int(np.searchsorted(ends, s, side="left")), len(step) - 1)
        out[i] = np.rad2deg(np.arctan2(step[k, 1], step[k, 0])) % 180.0
    return out


def street_offsets(pack: Pack, x: float, y: float, axis: float, rows=None, reach: float = 0.0):
    """The centrelines near (x, y) (within `reach` ft; or only `rows`): their rows, the distance (ft) and the angle
    between `axis` and each centreline at its nearest point (degrees, 0 to 90)."""
    pt = shapely.Point(x, y)
    if rows is None:
        rows = pack.street_tree.query(pt, predicate="dwithin", distance=float(reach))
    rows = np.sort(np.asarray(rows, np.int64))
    if not len(rows):
        return rows, np.zeros(0), np.zeros(0)
    lines = pack.streets.geometry.values[rows]
    d = shapely.distance(lines, pt)
    deg = _angle_between(axis, _line_axes(lines, shapely.line_locate_point(lines, pt)))
    return rows, d, deg


def block_offsets(pack: Pack, x: float, y: float, rows=None, reach: float = 0.0):
    """The blocks near (x, y) (within `reach` ft; or only `rows`): their rows and how far (x, y) lies outside each
    (ft, 0 inside)."""
    pt = shapely.Point(x, y)
    if rows is None:
        rows = pack.block_tree.query(pt, predicate="dwithin", distance=float(reach))
    rows = np.sort(np.asarray(rows, np.int64))
    return rows, shapely.distance(pack.blocks.geometry.values[rows], pt) if len(rows) else np.zeros(0)


# --- the prior --------------------------------------------------------------------------------------------------------

@dataclass
class Prior:
    """A box's prior (see the module): `kind_p` (KIND_ORDER, normalised), `escape` E_k per kind, and per kind its
    items: (odds against the escape, {form: P(form | item)})."""
    kind_p: np.ndarray
    escape: np.ndarray
    items: dict[str, list[tuple[float, dict[str, float]]]]

    @cached_property
    def strings(self) -> list[str]:
        """Every string an item prints (sorted): the strings the reader scores."""
        return sorted({f for its in self.items.values() for _, forms in its for f in forms})

    def mass(self, kind: str) -> dict[str, float]:
        """The kind's prior mass of each listed string at w = 1 (item mass E_k odds, shared by its forms)."""
        e = float(self.escape[KIND_ORDER.index(kind)])
        out: dict[str, float] = {}
        for odds, forms in self.items.get(kind, []):
            for f, p in forms.items():
                out[f] = out.get(f, 0.0) + e * odds * p
        return out


def street_odds(pack: Pack, sp: Spreads, d, deg) -> np.ndarray:
    """A street's odds against "a street the pack doesn't hold" at distance d (ft) and angle deg (see the module)."""
    if pack.street_length <= 0 or pack.area <= 0:
        return np.zeros(np.shape(d))
    return ((1 - sp.street_away) / sp.street_away * sp.street_density(d) * sp.angle_density(deg)
            * (90.0 * pack.area) / (2.0 * pack.street_length))


def block_odds(pack: Pack, sp: Spreads, d, perimeter, area) -> np.ndarray:
    """A block's odds against "a block the pack doesn't hold" at distance d (ft, 0 inside) from a block of this
    perimeter and area (see the module)."""
    d = np.asarray(d, float)
    if not pack.named_blocks or pack.area <= 0:
        return np.zeros(d.shape)
    with np.errstate(divide="ignore", invalid="ignore"):
        inside = np.where(d <= 0, sp.block_in / np.asarray(area, float), 0.0)
        ring = np.asarray(perimeter, float) + 2.0 * math.pi * d
        outside = np.where(d > 0, (1 - sp.block_in) * sp.block_out_density(d) / ring, 0.0)
    return (1 - sp.block_away) / sp.block_away * (pack.area / pack.named_blocks) * (inside + outside)


def _kind(odds: np.ndarray, forms: list[dict[str, float]]) -> tuple[float, list]:
    """A kind's escape mass and its items: those whose odds lift anything in float64 (1 + odds != 1); the rest are
    the escape's."""
    odds = np.asarray(odds, float)
    e = 1.0 / (1.0 + float(odds.sum())) if len(odds) else 1.0
    return e, [(float(o), f) for o, f in zip(odds, forms) if 1.0 + o != 1.0 and f]


def prior(pack: Pack | None, box_map: dict, kind_p, spreads: Spreads | None) -> Prior | None:
    """The box's prior (see the module).  `box_map`: the box's centre `x`, `y` and `axis` on the map (`box_on_map`);
    `kind_p`: its kind probabilities (KIND_ORDER; normalised here); `spreads`: the fold's measured spreads (required:
    a prior is scored on the quarter its spreads left out).  No pack: None (the ink alone)."""
    kp = np.asarray(kind_p, float).reshape(-1)
    if kp.shape != (len(KIND_ORDER),) or not np.isfinite(kp).all() or (kp < 0).any() or kp.sum() <= 0:
        raise ValueError(f"kind_p must be {len(KIND_ORDER)} non-negative numbers ({KIND_ORDER}), not {kind_p}")
    if pack is None:
        return None
    if spreads is None:
        raise ValueError("a prior needs the fold's measured spreads (`measure_spreads`)")
    sp = spreads
    kp = kp / kp.sum()
    x, y, axis = float(box_map["x"]), float(box_map["y"]), float(box_map["axis"])
    escape = np.ones(len(KIND_ORDER))
    items: dict[str, list] = {}
    if kp[0] > 0 and pack.code_forms:            # a kind the box can't be (kind_p 0) has no say: not built
        shares = pack.form_shares(sp)
        odds = (1 - sp.code_away) / (len(shares) * sp.code_away)
        escape[0], items[CODE] = _kind(np.full(len(shares), odds), list(shares.values()))
    rows, d = block_offsets(pack, x, y, reach=pack.reach(sp)[1]) if kp[1] > 0 else (np.zeros(0, int), np.zeros(0))
    if len(rows):
        named = np.array([bool(pack.forms_of_block[r]) for r in rows], bool)
        rows, d = rows[named], d[named]
        odds = block_odds(pack, sp, d, pack.block_perimeter[rows], pack.block_area[rows])
        forms = [{f: 1.0 / len(pack.forms_of_block[r]) for f in pack.forms_of_block[r]} for r in rows]
        escape[1], items[BLOCK] = _kind(odds, forms)
    rows, d, deg = street_offsets(pack, x, y, axis, reach=pack.reach(sp)[0]) if kp[2] > 0 else \
        (np.zeros(0, int), np.zeros(0), np.zeros(0))
    if len(rows):
        keep = pack.named_street[rows]
        rows, d, deg = rows[keep], d[keep], deg[keep]
        nearest: dict[str, int] = {}                 # each street by its nearest centreline
        for k, r in enumerate(rows):
            name = pack.street_names[r]
            if name not in nearest or d[k] < d[nearest[name]]:
                nearest[name] = k
        names = sorted(nearest)
        idx = np.array([nearest[n] for n in names], np.int64)
        if len(idx):
            odds = street_odds(pack, sp, d[idx], deg[idx])
            forms = [{f: 1.0 / len(pack.forms_of_street[n]) for f in pack.forms_of_street[n]} for n in names]
            escape[2], items[STREET] = _kind(odds, forms)
    return Prior(kind_p=kp, escape=escape, items={k: v for k, v in items.items() if v})


# --- the ink ----------------------------------------------------------------------------------------------------------

def ink_logp(reads, q: float, strings, tau: float = 1.0) -> dict[str, float]:
    """log p(s | ink) for each of `strings`, exactly by CTC (`textread.ctc.string_log_probs`) at temperature `tau`
    (each frame's log-probabilities x tau, renormalised; 1: as read), mixed over the two reading directions as the
    reader mixes them (`textread.ctc.mixture`): `reads` the crop's log-probabilities read as cut and turned a half
    turn (frames x classes each), `q` the classifier's p(the crop as cut is upside down)."""
    import torch

    from sfzmaps.solver.textread import ctc
    strings = list(strings)
    out = []
    for r in reads:
        lp = torch.as_tensor(np.asarray(r, np.float64))
        if tau != 1.0:
            lp = torch.log_softmax(lp * tau, -1)
        out.append(ctc.string_log_probs(lp, strings))
    with np.errstate(divide="ignore"):
        ink = np.logaddexp(np.log(1.0 - q) + out[0], np.log(q) + out[1])
    return dict(zip(strings, ink.tolist()))


# --- the posterior ----------------------------------------------------------------------------------------------------

def _log_expm1(a: np.ndarray) -> np.ndarray:
    """log(exp(a) - 1) for a >= 0 (-inf at 0), without overflow."""
    a = np.asarray(a, float)
    with np.errstate(divide="ignore"):
        return np.where(a > 0, a + np.log(-np.expm1(-np.maximum(a, 1e-300))), -np.inf)


def log_lift(prior: Prior | None, strings, w: dict[str, float]) -> np.ndarray:
    """log lift(s) of each of `strings` (see the module) at the kinds' weights `w` (a weight for every kind that
    lists anything; KeyError otherwise)."""
    strings = list(strings)
    if prior is None:
        return np.zeros(len(strings))
    col = {s: i for i, s in enumerate(strings)}
    c = prior.kind_p * prior.escape
    total = float(c.sum())
    if not total > 0:
        raise ValueError("every kind's escape mass is 0: the prior lists everything with certainty")
    c = c / total
    terms = [[] for _ in strings]              # per string: (kind index, log Lift_k - 0)
    for kind, its in prior.items.items():
        k = KIND_ORDER.index(kind)
        if c[k] == 0:
            continue
        wk = float(w[kind])
        logs = [[] for _ in strings]
        for odds, forms in its:
            a = _log_expm1(np.array([wk * math.log1p(odds)]))[0]
            for f, p in forms.items():
                if f in col and p > 0:
                    logs[col[f]].append(math.log(p) + a)
        for i, ls in enumerate(logs):
            if ls:
                terms[i].append((k, float(np.logaddexp(0.0, np.logaddexp.reduce(ls)))))
    out = np.zeros(len(strings))
    for i, t in enumerate(terms):
        if t:
            base = 1.0 - sum(c[k] for k, _ in t)
            parts = [math.log(c[k]) + lk for k, lk in t] + ([math.log(base)] if base > 0 else [])
            out[i] = float(np.logaddexp.reduce(parts))
    return out


def posterior(ink_logp: dict[str, float], prior: Prior | None, w: dict[str, float]) -> dict[str, float]:
    """The log posterior (see the module) of every string of `ink_logp` and of OTHER (every string not named).
    `ink_logp`: log p(s | ink) for every listed string of `prior` (ValueError when one is missing) and any others;
    `prior`: `prior`'s output, None for none (the ink alone); `w`: each listing kind's weight (>= 0)."""
    if OTHER in ink_logp:
        raise ValueError(f"{OTHER!r} is the posterior's own key, not a string")
    if any(v < 0 for v in w.values()):
        raise ValueError(f"the weights must be >= 0, not {w}")
    if prior is not None:
        missing = [s for s in prior.strings if s not in ink_logp]
        if missing:
            raise ValueError(f"no ink for {len(missing)} listed strings (e.g. {missing[:3]}): the reader scores them")
    names = list(ink_logp)
    ink = np.array([ink_logp[s] for s in names], float)
    score = ink + log_lift(prior, names, w)
    with np.errstate(over="ignore"):
        rest = 1.0 - float(np.exp(ink).sum())
    allv = np.r_[score, math.log(rest) if rest > 0 else -math.inf]
    z = float(np.logaddexp.reduce(allv[np.isfinite(allv)])) if np.isfinite(allv).any() else 0.0
    return {**{s: float(v - z) for s, v in zip(names, score)}, OTHER: float(allv[-1] - z)}


# --- holding the answers out ------------------------------------------------------------------------------------------

def refuse_answered(words: pd.DataFrame, held_blocks=None, quarter: int | None = None) -> None:
    """ValueError when any word is in or touches an answered block (`block` in `held_blocks`, default
    `textnet.reader_stage.held_blocks()`; or, for a row of several words, any of its `blocks`; or `touch_blocks`), or
    lies in or touches the held-out `quarter` (`quarter`, `touch_<q>`): nothing is fitted on the owner's answers, nor
    on the quarter it is scored on."""
    if "block" not in words:
        raise ValueError("each word needs its block (`block`) so that answered blocks can be refused")
    if held_blocks is None:
        from sfzmaps.solver.textnet import reader_stage
        held_blocks = reader_stage.held_blocks()
    held = {str(b) for b in held_blocks}
    bad = words["block"].astype(str).isin(held)
    if "blocks" in words:
        bad |= np.array([any(str(b) in held for b in bs) for bs in words["blocks"]], bool)
    if bad.any():
        raise ValueError(f"{int(bad.sum())} words lie in an answered block (e.g. {words.loc[bad, 'block'].iloc[0]}): "
                         "a prior is never fitted on the owner's answers")
    if "touch_blocks" in words and words["touch_blocks"].astype(bool).any():
        raise ValueError(f"{int(words['touch_blocks'].astype(bool).sum())} words touch an answered block")
    if quarter is not None:
        touch = f"touch_{paths.QUARTERS[int(quarter)]}"
        if "quarter" not in words or touch not in words:
            raise ValueError(f"each word needs its quarter and `{touch}` to keep the held-out quarter out")
        if (words["quarter"].astype(int) == int(quarter)).any():
            raise ValueError(f"words of the held-out quarter {quarter}: fit on the other three quarters")
        if words[touch].astype(bool).any():
            raise ValueError(f"{int(words[touch].astype(bool).sum())} words touch the held-out quarter {quarter}")


# --- measuring: the offsets, EM and the code list ---------------------------------------------------------------------

def _street_rows_by_form(pack: Pack) -> dict[str, np.ndarray]:
    names: dict[str, list[str]] = {}
    for name, forms in pack.forms_of_street.items():
        for f in forms:
            names.setdefault(f, []).append(name)
    rows_of = pd.Series(np.arange(len(pack.streets))).groupby(pack.street_names).agg(list).to_dict()
    return {f: np.array(sorted(r for n in ns for r in rows_of[n]), np.int64) for f, ns in names.items()}


def word_offsets(pack: Pack, words: pd.DataFrame) -> pd.DataFrame:
    """Each word against what its text names: `found` (a street, or block, prints it), and from the nearest that does,
    `d_ft` (from its centreline, or outside the block: 0 inside) and `d_deg` (the word's axis to the centreline there).
    `words`: text, kind, x, y, axis (`box_on_map`)."""
    by_form = _street_rows_by_form(pack)
    block_rows: dict[str, list[int]] = {}
    for r, forms in enumerate(pack.forms_of_block):
        for f in forms:
            block_rows.setdefault(f, []).append(r)
    out = []
    for w in words.itertuples(index=False):
        rec = {"found": False, "d_ft": np.nan, "d_deg": np.nan}
        pt = shapely.Point(w.x, w.y)
        if w.kind == STREET and w.text in by_form:
            rows = by_form[w.text]
            d = shapely.distance(pack.streets.geometry.values[rows], pt)
            k = int(np.argmin(d))
            _, dk, deg = street_offsets(pack, w.x, w.y, w.axis, rows=rows[k:k + 1])
            rec = {"found": True, "d_ft": float(dk[0]), "d_deg": float(deg[0])}
        elif w.kind == BLOCK and w.text in block_rows:
            rows = np.array(block_rows[w.text], np.int64)
            rec = {"found": True, "d_ft": float(shapely.distance(pack.blocks.geometry.values[rows], pt).min()),
                   "d_deg": np.nan}
        out.append(rec)
    return pd.DataFrame(out, index=words.index, columns=["found", "d_ft", "d_deg"])


def _em(features: list[tuple[np.ndarray, float | None, float | None]], d: np.ndarray, far_angle: bool,
        r0: np.ndarray, v: np.ndarray | None = None, groups=None) -> dict:
    """One EM run from responsibilities r0, each point counting with weight `v` (the chance its box holds its word;
    1 when None): near = the product of weighted kernel densities over `features` (each (values, lo, hi), weights
    r v, leaving out every row of the point's own box (`groups`; `loo_density`), bandwidth by that leave-one-box-out
    likelihood), far = exponential in `d` (x uniform on [0, 90] when
    `far_angle`).  Each iteration's bandwidth search starts from the last one.  It stops when the log-likelihood stops
    rising (by more than EM_TOL relative: numerical convergence) or no responsibility moves by more than EM_TOL; the
    likeliest state seen is returned (the bandwidths are re-chosen each step, so the log-likelihood need not rise
    monotonically): its r (the responsibilities its bandwidths were chosen for), bandwidths, far scale,
    log-likelihood, iterations and whether it converged before EM_MAX_ITER."""
    r = np.asarray(r0, float).copy()
    v = np.ones(len(d)) if v is None else np.asarray(v, float)
    hs = [math.nan] * len(features)
    best = None
    it, converged = 0, False
    for it in range(1, EM_MAX_ITER + 1):
        pi = float((v * r).sum() / v.sum())
        hs = [loo_bandwidth(x, r * v, lo, hi, start=h, groups=groups) for (x, lo, hi), h in zip(features, hs)]
        near = np.ones(len(d))
        for (x, lo, hi), h in zip(features, hs):
            near = near * (loo_density(x, r * v, h, lo, hi, groups) if math.isfinite(h) else np.zeros(len(d)))
        far_w = (1.0 - r) * v
        lam = float((far_w * d).sum() / far_w.sum()) if far_w.sum() > 0 else math.nan
        far = (np.exp(-d / lam) / lam if math.isfinite(lam) and lam > 0 else np.zeros(len(d))) / \
            (90.0 if far_angle else 1.0)
        mix = pi * near + (1 - pi) * far
        with np.errstate(invalid="ignore", divide="ignore"):
            new = np.where(mix > 0, pi * near / np.where(mix > 0, mix, 1.0), 0.0)
            ll = float(np.where(v > 0, v * np.log(mix), 0.0).sum())
        rose = best is None or ll > best["ll"] + EM_TOL * abs(best["ll"])
        if best is None or ll > best["ll"]:
            best = {"r": r.copy(), "h": list(hs), "lam": lam, "ll": ll}
        moved = float(np.max(np.abs(new - r))) if len(r) else 0.0
        r = new
        if not rose or moved <= EM_TOL:
            converged = True
            break
    return {**best, "iterations": it, "converged": converged}


def em_split(features, d: np.ndarray, far_angle: bool, v: np.ndarray | None = None, groups=None) -> dict:
    """`_em` (weights `v`, boxes `groups`) from every decile split of `d` (near below it) and from an even start;
    the likeliest fixed point."""
    d = np.asarray(d, float)
    if not len(d):
        return {"r": np.zeros(0), "h": [math.nan] * len(features), "lam": math.nan, "ll": 0.0, "iterations": 0,
                "converged": True, "start": None}
    wv = np.ones(len(d)) if v is None else np.asarray(v, float)
    o = np.argsort(d, kind="stable")
    cum = np.cumsum(wv[o]) / wv.sum()

    def decile(p):                                 # the v-weighted decile of d
        return d[o][min(int(np.searchsorted(cum, p / 10)), len(d) - 1)]

    starts, seen = [], set()
    for name, r0 in [("even", np.full(len(d), 0.5))] + \
            [(f"d<=q{p}", (d <= decile(p)).astype(float)) for p in range(1, 10)]:
        if r0.tobytes() not in seen:               # a start like one already run would end where it did
            seen.add(r0.tobytes())
            starts.append((name, r0))
    best = None
    for name, r0 in starts:
        got = _em(features, d, far_angle, r0, v, groups)
        if best is None or got["ll"] > best["ll"]:
            best = {**got, "start": name}
    return best


def measure_spreads(pack: Pack, words: pd.DataFrame, code_words: pd.DataFrame | None = None, held_blocks=None,
                    quarter: int | None = None) -> tuple[Spreads, pd.DataFrame]:
    """The fold's `Spreads` (see the module), measured on `words` (our boxes paired with Terra's words: text, kind,
    x, y, axis, block, quarter, touches, and optionally `weight`, the chance the box holds the word: every share and
    density is weighted by it; and `box`: the leave-one-out bandwidths leave out every row of a word's box) and, for
    the codes, on `code_words` (Terra's code words: text, block, quarter, touches; `words`' code words when None),
    both refused answered blocks and the held-out quarter (`refuse_answered`); and the words' offsets with their
    responsibilities (`r`)."""
    code_words = words[words["kind"] == CODE] if code_words is None else code_words
    refuse_answered(words, held_blocks, quarter)
    refuse_answered(code_words, held_blocks, quarter)
    v_all = words["weight"].to_numpy(float) if "weight" in words else np.ones(len(words))
    off = word_offsets(pack, words).assign(r=0.0)
    found = off["found"].to_numpy(bool)
    st = (words["kind"] == STREET).to_numpy()
    bl = (words["kind"] == BLOCK).to_numpy()
    s = off[st & found]
    vs = v_all[st & found]
    groups = words.loc[s.index, "box"].to_numpy() if "box" in words else None     # a box's rows: one group
    em_s = em_split([(s["d_ft"].to_numpy(), 0.0, None), (s["d_deg"].to_numpy(), 0.0, 90.0)], s["d_ft"].to_numpy(),
                    True, vs, groups)
    off.loc[s.index, "r"] = em_s["r"]
    b = off[bl & found]
    bd = b["d_ft"].to_numpy()
    inside = bd <= 0
    off.loc[b.index[inside], "r"] = 1.0             # inside the block it names: near; outside it: away (see module)
    vb = v_all[bl & found]
    near_b = float(vb[inside].sum())
    texts = code_words["text"].astype(str)
    listed = texts.isin(set(pack.codes))
    weights = texts[listed].map(pack.codes).to_frame("code").assign(form=texts[listed]) \
        .groupby(["code", "form"]).size()
    weights = (weights / weights.groupby(level="code").transform("sum")).droplevel("code").to_dict()
    n_s, n_b = float(v_all[st].sum()), float(v_all[bl].sum())
    if not n_s or not n_b or not len(code_words) or not near_b:
        raise ValueError(f"too few words to measure: {n_s} street, {n_b} block, {len(code_words)} code words")
    em_info = {"street": {"start": em_s["start"], "iterations": em_s["iterations"], "converged": em_s["converged"],
                          "ll": em_s["ll"], "far_scale_ft": em_s["lam"], "weight": n_s},
               "block": {"inside": round(near_b, 3), "outside": round(float(vb[~inside].sum()), 3),
                         "not_found": round(n_b - float(vb.sum()), 3)}}
    sp = Spreads(street_ft=s["d_ft"].to_numpy(), street_deg=s["d_deg"].to_numpy(), street_r=em_s["r"] * vs,
                 street_h=em_s["h"][0], street_deg_h=em_s["h"][1],
                 street_away=1.0 - float((em_s["r"] * vs).sum()) / n_s,
                 block_ft=bd, block_r=off.loc[b.index, "r"].to_numpy() * vb, block_h=math.nan,
                 block_in=1.0, block_away=1.0 - near_b / n_b,
                 code_away=1.0 - float(listed.mean()), form_weights={str(k): float(v) for k, v in weights.items()},
                 em=em_info)
    return sp, off


# --- fitting tau and w ------------------------------------------------------------------------------------------------

def _group_lse(v: np.ndarray, starts: np.ndarray) -> np.ndarray:
    """logsumexp of each run of `v` beginning at `starts` (v sorted by run; every run non-empty)."""
    if not len(starts):
        return np.zeros(0)
    m = np.maximum.reduceat(v, starts)
    safe = np.where(np.isfinite(m), m, 0.0)
    with np.errstate(divide="ignore"):
        return safe + np.log(np.add.reduceat(np.exp(v - np.repeat(safe, np.diff(np.r_[starts, len(v)]))), starts))


class _Words:
    """The fit's words flattened (`posterior`'s arithmetic, vectorised): every word's named strings (its prior's and
    its truth), each string's lift entries (kind, log1p(odds), log P(string | item)), the kinds' shares c_k, each
    truth's position and each word's weight (`weight`, 1 when absent); the ink at a temperature computed on demand
    (cached), once per box (`box`: the rows of one box share its reads; each row its own when absent)."""

    def __init__(self, words: pd.DataFrame):
        self.weight = words["weight"].to_numpy(float) if "weight" in words else np.ones(len(words))
        boxes = words["box"].tolist() if "box" in words else list(range(len(words)))
        self.box_reads: dict = {}
        for b, r, q in zip(boxes, words["reads"], words["q"]):
            self.box_reads.setdefault(b, (r, float(q)))
        self.boxes = boxes
        priors = list(words["prior"])
        targets = list(words["targets"]) if "targets" in words else [[(t, 1.0)] for t in words["truth"]]
        self.named, word_of = [], []
        tg_g, tg_lp, tg_row = [], [], []
        ent_g, ent_k, ent_l, ent_p = [], [], [], []
        c_rows = []
        g0 = 0
        for i, (tg, pr) in enumerate(zip(targets, priors)):
            tg = [(s, float(p)) for s, p in tg if p > 0]
            if not tg:
                raise ValueError(f"row {i} has no target with a chance above 0")
            names = sorted(set(pr.strings if pr is not None else []) | {s for s, _ in tg})
            col = {s: g0 + j for j, s in enumerate(names)}
            self.named.append(names)
            for s, p in tg:
                tg_g.append(col[s])
                tg_lp.append(math.log(p))
                tg_row.append(i)
            word_of += [i] * len(names)
            c = np.zeros(len(KIND_ORDER))
            if pr is not None:
                c = pr.kind_p * pr.escape
                c = c / c.sum()
                for kind, its in pr.items.items():
                    k = KIND_ORDER.index(kind)
                    if c[k] == 0:
                        continue
                    for odds, forms in its:
                        for f, p in forms.items():
                            if p > 0:
                                ent_g.append(col[f])
                                ent_k.append(k)
                                ent_l.append(math.log1p(odds))
                                ent_p.append(math.log(p))
            c_rows.append(c)
            g0 += len(names)
        self.tg_g, self.tg_lp = np.array(tg_g, np.int64), np.array(tg_lp, float)
        tg_row = np.array(tg_row, np.int64)
        self.tg_starts = np.flatnonzero(np.r_[True, tg_row[1:] != tg_row[:-1]]) if len(tg_row) else np.zeros(0, int)
        self.word_of = np.array(word_of, np.int64)
        self.starts = np.flatnonzero(np.r_[True, self.word_of[1:] != self.word_of[:-1]]) if g0 else np.zeros(0, int)
        c = np.array(c_rows, float).reshape(-1, len(KIND_ORDER))
        order = np.lexsort((np.array(ent_k, np.int64), np.array(ent_g, np.int64)))
        g, k = np.array(ent_g, np.int64)[order], np.array(ent_k, np.int64)[order]
        self.ent_k, self.ent_l, self.ent_p = k, np.array(ent_l, float)[order], np.array(ent_p, float)[order]
        key = g * len(KIND_ORDER) + k
        self.grp_starts = np.flatnonzero(np.r_[True, key[1:] != key[:-1]]) if len(key) else np.zeros(0, int)
        self.grp_g, grp_k = g[self.grp_starts], k[self.grp_starts]
        self.grp_logc = np.log(c[self.word_of[self.grp_g], grp_k]) if len(self.grp_g) else np.zeros(0)
        base = np.ones(g0)                    # 1 - the shares of the kinds that list each string
        np.subtract.at(base, self.grp_g, c[self.word_of[self.grp_g], grp_k])
        with np.errstate(divide="ignore"):
            self.log_base = np.log(np.maximum(base, 0.0))
        self.gg_starts = np.flatnonzero(np.r_[True, self.grp_g[1:] != self.grp_g[:-1]]) if len(self.grp_g) \
            else np.zeros(0, int)
        self.gg = self.grp_g[self.gg_starts] if len(self.grp_g) else np.zeros(0, int)
        self.kinds = [KIND_ORDER[j] for j in sorted(set(self.ent_k.tolist()))]
        self.box_strings: dict = {}
        for b, names in zip(self.boxes, self.named):
            self.box_strings.setdefault(b, set()).update(names)
        self.box_strings = {b: sorted(s) for b, s in self.box_strings.items()}
        self._ink: dict[float, tuple[np.ndarray, np.ndarray]] = {}

    def ink(self, tau: float) -> tuple[np.ndarray, np.ndarray]:
        """Every named string's ink at tau (flat) and each word's ink left for the strings it doesn't name."""
        if tau not in self._ink:
            per_box = {b: ink_logp(self.box_reads[b][0], self.box_reads[b][1], s, tau)
                       for b, s in self.box_strings.items()}
            flat = np.concatenate([np.array([per_box[b][s] for s in names], float)
                                   for b, names in zip(self.boxes, self.named)]) if self.named else np.zeros(0)
            with np.errstate(over="ignore"):
                rest = 1.0 - np.add.reduceat(np.exp(flat), self.starts) if len(flat) else np.zeros(0)
            self._ink[tau] = (flat, rest)
        return self._ink[tau]

    def lift(self, w: dict[str, float]) -> np.ndarray:
        """log lift of every named string (flat) at the kinds' weights."""
        out = np.zeros(len(self.word_of))
        if not len(self.ent_k):
            return out
        wk = np.array([float(w.get(k, 0.0)) for k in KIND_ORDER])[self.ent_k]
        log_s = _group_lse(self.ent_p + _log_expm1(wk * self.ent_l), self.grp_starts)
        terms = self.grp_logc + np.logaddexp(0.0, log_s)
        per_g = _group_lse(terms, self.gg_starts)
        out[self.gg] = np.logaddexp(per_g, self.log_base[self.gg])
        return out

    def scores(self, tau: float, w: dict[str, float]) -> tuple[np.ndarray, np.ndarray]:
        """Every named string's unnormalised log posterior (flat) and each word's OTHER (log of the ink it leaves)."""
        ink, rest = self.ink(tau)
        with np.errstate(divide="ignore"):
            return ink + self.lift(w), np.log(np.maximum(rest, 0.0))

    def loglik(self, tau: float, w: dict[str, float]) -> np.ndarray:
        """Each word's log posterior of its target: log sum_t P(t) post(t) over its targets."""
        score, log_rest = self.scores(tau, w)
        z = np.logaddexp(_group_lse(score, self.starts), log_rest)
        return _group_lse(self.tg_lp + score[self.tg_g], self.tg_starts) - z

    def rows(self, tau: float, w: dict[str, float]) -> list[tuple[np.ndarray, list]]:
        """Each word's log posterior over its named strings and then OTHER (normalised), and its targets: (place in
        that vector, log P(target))."""
        score, log_rest = self.scores(tau, w)
        ends = np.r_[self.starts[1:], len(score)]
        tg_ends = np.r_[self.tg_starts[1:], len(self.tg_g)]
        out = []
        for i, (a, b) in enumerate(zip(self.starts, ends)):
            v = np.r_[score[a:b], log_rest[i]]
            sel = slice(self.tg_starts[i], tg_ends[i])
            out.append((v - float(np.logaddexp.reduce(v[np.isfinite(v)])),
                        [(int(g - a), float(lp)) for g, lp in zip(self.tg_g[sel], self.tg_lp[sel])]))
        return out


def ctc_fits(s: str, frames: int) -> bool:
    """Whether CTC can emit `s` in `frames` frames at all (a frame a character, and a blank between two repeated
    ones): a string that can't is impossible from that read at any temperature."""
    return len(s) + sum(a == b for a, b in zip(s, s[1:])) <= frames


def readable(words: pd.DataFrame) -> np.ndarray:
    """Which rows have a target their box's reads can emit at all (`ctc_fits`; `targets`, or `truth` when absent):
    a row with none is a physically impossible observation of that box."""
    targets = list(words["targets"]) if "targets" in words else [[(t, 1.0)] for t in words["truth"]]
    return np.array([any(p > 0 and ctc_fits(s, int(np.shape(r[0])[0])) for s, p in tg)
                     for tg, r in zip(targets, words["reads"])], bool)


ALONG = (1, 3)         # a quadrilateral's two ends (corner 1 to 2, corner 3 to 0): between them is "along the reading"
ACROSS = (0, 2)        # its two long sides (corner 0 to 1, corner 2 to 3): between them is "in its line"


def _inner_sides(quad, edges, x, y) -> np.ndarray:
    """Which points lie on the inner side of every one of a convex quadrilateral's `edges` (k: the edge from corner k
    to corner k + 1; edges included): all four, inside it; ALONG, between its two ends; ACROSS, between its two long
    sides."""
    c = np.asarray(quad, float).reshape(4, 2)
    x, y = np.asarray(x, float), np.asarray(y, float)
    twice_area = sum(c[k, 0] * c[(k + 1) % 4, 1] - c[(k + 1) % 4, 0] * c[k, 1] for k in range(4))
    sign = 1.0 if twice_area >= 0 else -1.0
    ok = np.ones(np.shape(x), bool)
    for k in edges:
        a, b = c[k], c[(k + 1) % 4]
        ok &= sign * ((b[0] - a[0]) * (y - a[1]) - (b[1] - a[1]) * (x - a[0])) >= 0
    return ok


def _in_quad(quad, x, y) -> np.ndarray:
    """Which points lie inside a convex quadrilateral (4, 2) (its edges included)."""
    return _inner_sides(quad, (0, 1, 2, 3), x, y)


def _between(quad, edges, reach: float) -> shapely.Polygon:
    """The region between a quadrilateral's two opposite `edges` (ALONG or ACROSS), as a polygon running `reach` past
    the quadrilateral each way (so far that whatever it is cut with lies within it)."""
    c = np.asarray(quad, float).reshape(4, 2)
    (a0, a1), (b0, b1) = [(c[k], c[(k + 1) % 4]) for k in edges]
    u = (a1 - a0) / max(float(np.hypot(*(a1 - a0))), np.finfo(float).tiny)
    v = (b1 - b0) / max(float(np.hypot(*(b1 - b0))), np.finfo(float).tiny)
    return shapely.Polygon([a0 - u * reach, a1 + u * reach, b0 - v * reach, b1 + v * reach])


def _reach(*quads) -> float:
    """A reach (`_between`) past every point of `quads`."""
    pts = np.concatenate([np.asarray(q, float).reshape(-1, 2) for q in quads])
    return 2.0 * float(np.ptp(pts, axis=0).max()) + 1.0


def _area_share(shape: shapely.Geometry, region: shapely.Geometry) -> float:
    """The share of a shape's area inside a region; of its length when it has no area (a gap of no width); whether it
    lies inside when it is a point."""
    if shape.area > 0:
        return float(shape.intersection(region).area / shape.area)
    if shape.length > 0:
        return float(shape.intersection(region).length / shape.length)
    return float(region.covers(shape))


def char_shares(text: str, word_rect, box_rect, ink_x, ink_y) -> np.ndarray:
    """Each character of a word's `text`, its extent the word's rectangle (4, 2; the first edge along the reading)
    divided evenly along the reading (no per-character widths are measured): the share of its ink px (the word's ink
    px centres `ink_x`, `ink_y`, each a character's by where it projects on the reading) between our box's two ends,
    or, for a character with no ink px (a space), the share of its extent's area between them.  Along the reading
    only: how much of the word's line the box covers across the reading is the word's own (`across_share`, and the
    pair's weight)."""
    n = len(text)
    c = np.asarray(word_rect, float).reshape(4, 2)
    a = c[1] - c[0]
    x, y = np.asarray(ink_x, float), np.asarray(ink_y, float)
    u = ((x - c[0, 0]) * a[0] + (y - c[0, 1]) * a[1]) / max(float(a @ a), 1e-12)
    idx = np.clip(np.floor(u * n).astype(int), 0, n - 1) if n else np.zeros(0, int)
    inside = _inner_sides(box_rect, ALONG, x, y) if len(x) else np.zeros(0, bool)
    strip = _between(box_rect, ALONG, _reach(word_rect, box_rect))
    out = np.zeros(n)
    for i in range(n):
        mine = idx == i
        if mine.any():
            out[i] = float(inside[mine].mean())
        else:
            ch = shapely.Polygon([c[0] + a * i / n, c[0] + a * (i + 1) / n, c[3] + a * (i + 1) / n, c[3] + a * i / n])
            out[i] = _area_share(ch, strip)
    return out


def across_share(word_rect, box_rect, ink_x, ink_y) -> float:
    """The chance a word lies in our box's line: the share of its ink px (`ink_x`, `ink_y`) between the box's two long
    sides (across the reading only), or, for a word with no ink px, of its rectangle's area."""
    x, y = np.asarray(ink_x, float), np.asarray(ink_y, float)
    if len(x):
        return float(_inner_sides(box_rect, ACROSS, x, y).mean())
    band = _between(box_rect, ACROSS, _reach(word_rect, box_rect))
    return _area_share(shapely.Polygon(np.asarray(word_rect, float).reshape(4, 2)), band)


def _log(p: float) -> float:
    return math.log(p) if p > 0 else -math.inf


def _add(d: dict, key, lp: float) -> None:
    if lp > -math.inf:
        d[key] = float(np.logaddexp(d[key], lp)) if key in d else lp


def _runs(chars, fits) -> tuple[dict, list, list]:
    """The contiguous runs of a character sequence [(char, share)], each character inside independently by its share:
    the log mass of each non-empty string (spaces at a run's ends print nothing), the log masses of the empty runs,
    and those of the runs no read can be (`fits` false: a run only grows, so it is followed whole, unlisted).  Runs
    that are not contiguous carry no mass."""
    sb = 0.0                                         # nothing inside yet
    si: dict = {}                                    # the run so far -> log mass, still open
    sa: dict = {}                                    # the run -> log mass, closed: every later character outside
    di = da = -math.inf                              # runs no read can be, open and closed
    for ch, p in chars:
        lin, lout = _log(p), _log(1.0 - p)
        tb, ti, ta = sb + lout, {}, {}
        _add(ti, "" if ch == " " else ch, sb + lin)
        for s, v in si.items():
            _add(ti, s + ch if s or ch != " " else "", v + lin)
            _add(ta, s, v + lout)
        for s, v in sa.items():
            _add(ta, s, v + lout)
        di, da = di + lin, float(np.logaddexp(da + lout, di + lout))
        for s in [s for s in ti if not fits(s)]:
            di = float(np.logaddexp(di, ti.pop(s)))
        sb, si, sa = tb, ti, ta
    got: dict = {}
    empty = [sb]
    for s, v in [*si.items(), *sa.items()]:
        s = s.rstrip()
        if s:
            _add(got, s, v)
        else:
            empty.append(v)
    return got, empty, [float(np.logaddexp(di, da))]


def _apart(line, alongs) -> list[tuple]:
    """The largest sets of a line's words no two of which overlap along the box (`alongs`: each word's [start, end]):
    the maximal cliques of "apart" (Bron-Kerbosch with a pivot); the line itself when none overlap."""
    line = list(line)
    apart = {i: {k for k in line if k != i and min(alongs[i][1], alongs[k][1]) - max(alongs[i][0], alongs[k][0]) <= 0}
             for i in line}
    out: list[tuple] = []

    def grow(r: set, p: set, x: set) -> None:
        if not p and not x:
            out.append(tuple(sorted(r)))
            return
        pivot = max(p | x, key=lambda v: (len(p & apart[v]), -v))
        for v in sorted(p - apart[pivot]):
            grow(r | {v}, p & apart[v], x & apart[v])
            p = p - {v}
            x = x | {v}

    grow(set(), set(line), set())
    return sorted(out)


def box_targets(words, box_rect, frames: int | None = None, box_ink=None) -> dict:
    """What our box holds of the Terra words it overlaps, as the expectation over the text it may hold (see the
    module).  `words`: each (text, rectangle (4, 2; the first edge along its reading), its ink px centres x, y);
    `box_ink`: the box's own ink px centres (x, y).
      - the box is read as one line: where it lies across the box is drawn from the box's ink px inside the words'
        rectangles that some word's span holds (from the box's width across, where they span it, without ink), and
        the line holds every word whose span across the box holds that place: words of one line together, words
        stacked one over another (which no one line can hold at the same place) apart.  A word's span across is its
        own ink's (its ink px in no other word's rectangle: Terra's rectangles are loose and overlap), its
        rectangle's when it has none;
      - nor can one line hold two words at one place along it (a small label printed inside a big one): the line
        reads each largest set of its words that lie apart along the box (`_apart`), by their own ink in the box;
      - each word of the line is held by its `across_share`; a word not held empties the box's text (the box holds
        none of its line's text);
      - each character lies inside along the reading by its `char_shares`, independently;
      - the line's words are read in reading order (their centres along the box's axis, the axis turned to Terra's
        reading direction), a space between two (its share that of the gap between the one's end and the next one's
        start: `_area_share` of their hull between the box's ends);
      - the box holds one contiguous run of that text (spaces at its ends print nothing): every chance is conditioned
        on the run being contiguous.
    Returns `targets`, [(string, P(string | a non-empty run))] for the strings CTC can emit in `frames` frames
    (`ctc_fits`; all when None), likeliest first; `dropped`, the chance of a run it can't emit (no read is it);
    `p_nonempty`, the chance the run is not empty; `order`, the words' positions in reading order; `across`; `lines`,
    each line's words (positions in reading order) and chance."""
    b = np.asarray(box_rect, float).reshape(4, 2)
    rects = [np.asarray(w[1], float).reshape(4, 2) for w in words]
    across = [across_share(r, b, w[2], w[3]) for w, r in zip(words, rects)]
    shares = [char_shares(w[0], r, b, w[2], w[3]) for w, r in zip(words, rects)]
    axis = b[1] - b[0]
    terra = sum((a * (r[1] - r[0]) / max(float(np.hypot(*(r[1] - r[0]))), np.finfo(float).tiny)
                 for a, r in zip(across, rects)), np.zeros(2))
    if float(terra @ axis) < 0:
        axis = -axis
    order = sorted(range(len(words)), key=lambda j: (float(rects[j].mean(0) @ axis), j))
    rank = {j: k for k, j in enumerate(order)}
    out = {"targets": [], "dropped": 0.0, "p_nonempty": 0.0, "order": order, "across": across, "lines": []}
    if not words:
        return out
    unit = axis / max(float(np.hypot(*axis)), np.finfo(float).tiny)
    normal = np.array([-unit[1], unit[0]])
    spans, alongs, mass = [], [], []                 # each word's own ink (in no other's rectangle): its spans across
    for j, (w, r) in enumerate(zip(words, rects)):   # and along the box, and its px inside the box
        x, y = np.asarray(w[2], float), np.asarray(w[3], float)
        own = np.ones(len(x), bool)
        for k, other in enumerate(rects):
            if k != j and len(x):
                own &= ~_in_quad(other, x, y)
        pts = np.c_[x[own], y[own]] if own.any() else (np.c_[x, y] if len(x) else r)
        spans.append([float((pts @ normal).min()), float((pts @ normal).max())])
        alongs.append([float((pts @ unit).min()), float((pts @ unit).max())])
        mass.append(float(_in_quad(b, pts[:, 0], pts[:, 1]).sum()) if pts is not r else 0.0)
    spans = np.array(spans)
    lines: dict[tuple, float] = {}
    bx, by = (np.zeros(0), np.zeros(0)) if box_ink is None else (np.asarray(box_ink[0], float),
                                                                    np.asarray(box_ink[1], float))
    inside = np.zeros(len(bx), bool)
    for r in rects:
        inside |= _in_quad(r, bx, by) if len(bx) else np.zeros(0, bool)
    if inside.any():                                 # the line's place: one of the box's ink px in the words
        t = np.c_[bx[inside], by[inside]] @ normal
        member = (spans[:, :1] <= t[None, :]) & (t[None, :] <= spans[:, 1:])
        keys, counts = np.unique(member.T, axis=0, return_counts=True)
        for key, c in zip(keys, counts):
            if key.any():                            # ink between the words' own spans holds no line
                lines[tuple(sorted(np.flatnonzero(key).tolist(), key=rank.get))] = float(c)
    else:                                            # no ink: where the words span the box's width across
        lo, hi = float((b @ normal).min()), float((b @ normal).max())
        cuts = np.unique(np.clip(np.r_[spans.ravel(), lo, hi], lo, hi))
        for u0, u1 in zip(cuts[:-1], cuts[1:]):
            mid = (u0 + u1) / 2
            key = tuple(sorted(np.flatnonzero((spans[:, 0] <= mid) & (mid <= spans[:, 1])).tolist(), key=rank.get))
            if key and u1 > u0:
                lines[key] = lines.get(key, 0.0) + float(u1 - u0)
    lines = {k: v / sum(lines.values()) for k, v in lines.items()} if lines else {}
    reads: dict[tuple, float] = {}                   # one line can't hold two words at one place along it: the
    for line, p_line in lines.items():               # line reads each largest set of its words apart along the
        sets = _apart(line, alongs)                  # box, by their own ink in the box
        w = np.array([sum(mass[j] for j in s) for s in sets], float)
        w = w / w.sum() if w.sum() > 0 else np.full(len(sets), 1.0 / len(sets))
        for s, ws in zip(sets, w):
            key = tuple(sorted(s, key=rank.get))
            reads[key] = reads.get(key, 0.0) + p_line * float(ws)
    lines = reads
    out["lines"] = [([rank[j] for j in k], p) for k, p in sorted(lines.items(), key=lambda kv: -kv[1])]
    strip = _between(b, ALONG, _reach(b, *rects))

    def ends(j):                                     # (start, end) edges along the axis
        r = rects[j]
        return ((r[0], r[3]), (r[1], r[2])) if float((r[1] - r[0]) @ axis) >= 0 else ((r[1], r[2]), (r[0], r[3]))

    def gap(i, k) -> float:
        hull = shapely.convex_hull(shapely.multipoints(np.array([*ends(i)[1], *ends(k)[0]])))
        return _area_share(hull, strip)

    def fits(s: str) -> bool:
        return frames is None or ctc_fits(s.rstrip(), frames)

    got: dict = {}
    empty: list[float] = []
    dropped: list[float] = []
    for line, p_line in lines.items():
        held = p_line * float(np.prod([across[j] for j in line]))
        if p_line > held:
            empty.append(math.log(p_line - held))    # a word of the line not held: the box holds none of its text
        if not held > 0:
            continue
        chars = []
        for k, j in enumerate(line):
            chars += ([(" ", gap(line[k - 1], j))] if k else []) + list(zip(words[j][0], shares[j]))
        g, e, dr = _runs(chars, fits)
        for s, v in g.items():
            _add(got, s, v + math.log(held))
        empty += [v + math.log(held) for v in e]
        dropped += [v + math.log(held) for v in dr if v > -math.inf]
    ne = list(got.values()) + dropped
    if not ne:
        return out
    z_ne = float(np.logaddexp.reduce(ne))
    z_all = float(np.logaddexp.reduce(ne + empty))
    out["targets"] = sorted(((s, math.exp(v - z_ne)) for s, v in got.items()), key=lambda kv: (-kv[1], kv[0]))
    out["dropped"] = math.exp(float(np.logaddexp.reduce(dropped)) - z_ne) if dropped else 0.0
    out["p_nonempty"] = math.exp(z_ne - z_all)
    return out


def substring_targets(text: str, shares) -> list[tuple[str, float]]:
    """The parts of a word our box may hold, with their chances: every contiguous run of its characters, each
    character inside by its share (independently), conditioned on the box holding a contiguous, non-empty run:
    P(run) ~ prod_inside share x prod_outside (1 - share).  Spaces at a run's ends are dropped (they print nothing);
    equal strings add.  A run with chance exactly 0 is no target, so a word wholly inside (every share 1) has just
    itself."""
    s = np.clip(np.asarray(shares, float), 0.0, 1.0)
    n = len(text)
    with np.errstate(divide="ignore"):
        lin, lout = np.log(s), np.log1p(-s)
    got: dict[str, float] = {}
    for a in range(n):
        for b in range(a + 1, n + 1):
            lp = float(lin[a:b].sum() + lout[:a].sum() + lout[b:].sum())
            sub = text[a:b].strip()
            if not np.isfinite(lp) or not sub:
                continue
            got[sub] = float(np.logaddexp(got.get(sub, -np.inf), lp))
    if not got:
        return []
    z = float(np.logaddexp.reduce(list(got.values())))
    return sorted(((k, math.exp(v - z)) for k, v in got.items()), key=lambda kv: (-kv[1], kv[0]))


def truth_loglik(words: pd.DataFrame, tau: float, w: dict[str, float]) -> np.ndarray:
    """Each word's log posterior of its target at (tau, w) (`words` as `fit_prior` takes them)."""
    return _Words(words).loglik(tau, w)


def truth_rows(words: pd.DataFrame, tau: float, w: dict[str, float]) -> list[tuple[np.ndarray, list]]:
    """Each word's normalised log posterior over its named strings then OTHER, and its targets (`_Words.rows`)."""
    return _Words(words).rows(tau, w)


def target_logp(logpost: np.ndarray, targets) -> float:
    """log sum_t P(t) post(t) of a row's targets (an int: one target at that place)."""
    if isinstance(targets, (int, np.integer)):
        return float(logpost[targets])
    return float(np.logaddexp.reduce([lp + logpost[k] for k, lp in targets]))


def tempered(logpost: np.ndarray, temperature: float) -> np.ndarray:
    """A log posterior (over hypotheses) divided by a temperature and renormalised (1: unchanged)."""
    v = np.asarray(logpost, float) / temperature
    fin = np.isfinite(v)
    return v - (float(np.logaddexp.reduce(v[fin])) if fin.any() else 0.0)


def slot_targets(labels, targets, dropped: float = 0.0) -> list[tuple[int, float]]:
    """A row's targets on the hypotheses named at read time (`labels`: the named strings then OTHER, as `posterior`
    returns them): each target string at its own slot when named, else at OTHER's, and so is the chance `dropped` of a
    string no read can be; as (slot, log P) (`target_logp`)."""
    at = {s: i for i, s in enumerate(labels)}
    other = at[OTHER]
    out = [(at.get(s, other), math.log(p)) for s, p in targets if p > 0]
    return out + ([(other, math.log(dropped))] if dropped > 0 else [])


def read_posterior(pack: Pack | None, fold: str, ink_logp_fn, box: dict, strings=()) -> dict[str, float]:
    """The read recipe (Task 11 reads every box with it): a box's calibrated log posterior over the strings its prior
    lists, the `strings` named besides (the reader's best path: then OTHER is not one lump), and OTHER (every string
    not named):
      1. `prior` at the box's place, with its fold's spreads (`pack.spreads[fold]`);
      2. the ink at the fold's tau: `ink_logp_fn(names, tau)` -> {name: log p(name | ink)} for every name (e.g.
         `lambda names, tau: ink_logp(reads, q, names, tau)`);
      3. `posterior` at the fold's w;
      4. `tempered` at the fold's temperature (fitted on this very vector: see the module).
    `fold`: the box's own quarter (paths.QUARTERS), whose fit and spreads left it out; never "all", which saw every
    quarter.  `box`: its centre `x`, `y` and `axis` on the map (`box_on_map`) and its `kind_p` (KIND_ORDER).  No pack
    (None): the ink alone as read (tau 1, no temperature: nothing is fitted)."""
    if fold not in paths.QUARTERS:
        raise ValueError(f"a box is read with its own quarter's fit ({paths.QUARTERS}), not {fold!r}: the fit of "
                         "'all' saw every quarter")
    if pack is None:
        return posterior(dict(ink_logp_fn(sorted(set(strings)), 1.0)), None, {})
    if fold not in pack.fits or fold not in pack.spreads:
        raise ValueError(f"the pack has no fit or spreads for {fold} (`scripts/wordread/build_pack.py fit`)")
    fit = pack.fits[fold]
    pr = prior(pack, box, box["kind_p"], pack.spreads[fold])
    names = sorted(set(pr.strings) | set(strings))
    post = posterior(dict(ink_logp_fn(names, fit.tau)), pr, fit.w)
    return dict(zip(post, tempered(np.array(list(post.values())), fit.temperature).tolist()))


def fit_temperature(rows: list[tuple[np.ndarray, object]], words: pd.DataFrame, weights=None, held_blocks=None,
                    quarter: int | None = None) -> dict:
    """One temperature on the read-time log posterior, by maximum likelihood of each row's target slot (`rows`: (the
    log posterior over the hypotheses named at read time then OTHER, its targets as (slot, log P) or a slot:
    `slot_targets`); `weights` per row, 1 when None): unbounded search in log T (Brent from 1).  `words`: each row's
    block, quarter and touches, refused like `fit_prior`'s (`refuse_answered`): never the owner's answers, nor the
    held-out `quarter`."""
    if len(words) != len(rows):
        raise ValueError(f"{len(rows)} rows but {len(words)} rows of words to check them against")
    refuse_answered(words, held_blocks, quarter)
    v = np.ones(len(rows)) if weights is None else np.asarray(weights, float)

    def nll(u):
        t = math.exp(u)
        return -float(sum(vi * target_logp(tempered(lp, t), k) for (lp, k), vi in zip(rows, v)))

    res = optimize.minimize_scalar(nll, bracket=(-0.5, 0.0), method="brent")
    return {"temperature": float(math.exp(res.x)), "nll": float(res.fun), "nll_at_1": nll(0.0),
            "converged": bool(res.success)}


def fit_prior(words: pd.DataFrame, held_blocks=None, quarter: int | None = None, joint: bool = False) -> dict:
    """The ink's temperature and the kinds' weights by maximum likelihood of each word's `truth` (Terra's string),
    the ink calibrated apart from the context: tau first, on the ink alone (the reader's p(truth | ink) at tau), then w
    per kind on top of it (`posterior`).  `words`: per pair of one of our boxes and a Terra word, `truth` (the word's
    string), `reads` (the box's two reads' log-probabilities), `q`, `prior` (a `Prior`), `block`, `touch_blocks`,
    optionally `box` (rows of one box share its ink) and `weight` (the chance the box holds the word: each pair's
    log-likelihood counts that much) and, with `quarter`, `quarter` and its touch column; refused when any is the
    owner's or the held-out quarter's (`refuse_answered`).  A pair whose truth its box can't emit at all (`readable`)
    is left out and counted (`impossible`).  Unbounded searches: log tau
    by Brent from tau 1, log w by BFGS from w 1.  With `joint`, also tau and w fitted together (each tau's best w),
    reported under "joint".  Returns tau, w, the log-likelihood with them, with the ink as read (tau 1, no prior) and
    with the ink at tau alone, counts, and whether each search converged."""
    refuse_answered(words, held_blocks, quarter)
    can = readable(words)
    impossible = {"pairs": int((~can).sum()),
                  "weight": float(words["weight"].to_numpy(float)[~can].sum()) if "weight" in words else
                  float((~can).sum())}
    words = words[can]
    if not len(words):
        raise ValueError("no words to fit on")
    data = _Words(words)
    kinds = data.kinds
    zero = dict.fromkeys(kinds, 0.0)
    inner: dict[float, tuple[np.ndarray, float, bool]] = {}

    def best_w(tau: float):
        if tau not in inner:
            def nll(u):
                return -float((data.weight * data.loglik(tau, dict(zip(kinds, np.exp(u))))).sum())
            if kinds:
                res = optimize.minimize(nll, np.zeros(len(kinds)), method="BFGS")
                inner[tau] = (res.x, float(res.fun), (bool(res.success), str(res.message)))
            else:
                inner[tau] = (np.zeros(0), nll(np.zeros(0)), (True, "no kind lists anything"))
        return inner[tau]

    def result(tau, outer_ok):
        u, fun, ok = best_w(tau)
        return {"tau": tau, "w": {k: float(math.exp(v)) for k, v in zip(kinds, u)},
                "log_w": dict(zip(kinds, map(float, u))), "loglik": -fun,
                "loglik_ink_at_tau": float((data.weight * data.loglik(tau, zero)).sum()), "tau_converged": outer_ok,
                "w_converged": ok[0], "w_message": ok[1]}

    ink_only = optimize.minimize_scalar(lambda v: -float((data.weight * data.loglik(math.exp(v), zero)).sum()),
                                        bracket=(math.log(0.5), 0.0), method="brent")
    out = {**result(float(math.exp(ink_only.x)), bool(ink_only.success)),
           "loglik_ink_as_read": float((data.weight * data.loglik(1.0, zero)).sum()), "n": int(len(words)),
           "weight": float(data.weight.sum()), "impossible": impossible,
           "truth_listed": int(sum(t in (p.strings if p is not None else [])
                                   for t, p in zip(words["truth"], words["prior"])))}
    if joint:
        both = optimize.minimize_scalar(lambda v: best_w(math.exp(v))[1], bracket=(math.log(0.5), 0.0),
                                        method="brent")
        out["joint"] = result(float(math.exp(both.x)), bool(both.success))
    out["tau_evaluations"] = len(data._ink)
    return out
