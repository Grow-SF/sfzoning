"""Context packs: what a place knows, supplied at read time to shift the reader's probabilities (item 4 of the map-text
plan).  Optional: without a pack the reader reads the ink alone.  Nothing here trains a model; the reader's weights
hold no place.

A pack (`Pack`, one parquet file per map, under PACKS_DIR) holds, in map coordinates (EPSG:2227 ft):
  streets  `name` (DataSF's `streetname`, "MISSION ST"), `street` (its base name), `type` (`st_type`, "" for none),
           `active`, and the centreline: one LineString a row, as DataSF cuts them
  blocks   `number` (the Assessor's block, "0767") and its outline (the union of the block's lots)
  codes    the era's code list: printed form -> code (each code also a form of itself)
  spreads  where printed words sit relative to what they name (`Spreads`), measured on Terra's words: "all" (every
           quarter) and, per quarter name, measured with that quarter left out
It never holds a code at a lot: on the 2000 map that is the answer key.  `build_sf_pack` refuses to add one, and a
pack's frames take no column but their own.

The prior (`prior`): p(s | kind, place), mixed over the box's kind probabilities `kind_p` (KIND_ORDER, the detector's
`textnet.data.KINDS` order):
  zoning code   the era list, flat over codes; a code's mass is shared by its printed forms
  street name   which street the box names, by Bayes on where it sits.  For each street j near the box, by its
                nearest centreline (d_j ft from the box's centre; t_j the angle between the box's axis and the
                centreline there, 0 to 90 degrees), and for a street the pack doesn't hold (0):
                    P(j | box) ~ (1 - m) f(d_j) g(t_j) / (2 L)          P(0 | box) ~ m / (90 A)
                f and g the measured densities of street-name words' distance and angle to their own street; m the
                measured share of street-name words that name no street where they sit; L the pack's centreline
                length (a street is named in proportion to its length, its label either side) and A the map's area
                (the blocks' convex hull): a word naming no pack street sits anywhere, at any angle
  block number  the same for blocks:
                    P(b | box) ~ (1 - m_b) / N  x  p_in / area_b inside b,  (1 - p_in) f_out(d) / perimeter_b at d ft
                                                   outside it
                    P(0 | box) ~ m_b / A
                N the pack's blocks, p_in the measured share of block-number words inside their own block, f_out the
                density of the others' distance outside it
  other         no list
An item's P(item | box) of its kind's mass goes to its printed forms, flat; P(0 | box) of it goes to the escape.  So a
box far from every street gives the street-name kind's mass to the escape, never to the nearest street; a kind with no
list gives it all.  `prior` returns log masses: each listed string, and ESCAPE (every string not listed).

The spreads (`measure_spreads`, on Terra's words; no width, reach or cutoff is chosen by hand):
  - A word's own street (block) is the nearest one whose printed forms hold its text.  It names that street where it
    sits when it is also the nearest street (block) of all; the rest, and the words whose text no street (block)
    prints, make m (m_b).  (Misreads and renamed streets name a far one: 8% of the 2000 street words, 6% of the block
    numbers, from 100 ft to 6 miles away.)
  - f, f_out: Gaussian kernel densities of log(1 + d) (Scott's rule), as densities of d; g: of the angle, reflected
    at 0 and 90 degrees.
  - A string whose mass is below MIN_MASS is not listed (its mass goes to the escape): the posterior's floor E reads
    it as the ink alone anyway, to within (1 - e) MIN_MASS / E.  So streets and blocks are looked for only within
    the distance past which even the best-placed one (aligned; the smallest block) would fall below it (`reach`).

The posterior (`posterior`): log p(s | ink) + w log prior'(s), normalised over the strings named in `ink_logp` and
OTHER (every string not named, its ink 1 - the named strings' ink), where
    any string s:        prior'(s) = E,                       E = (1 - e) prior(ESCAPE) + e
    a listed string s:   prior'(s) = E + (1 - e) prior(s)
The escape is flat over strings (the ink alone tells them apart), and every string has its level E: a share e of the
prior always goes to it, whatever the pack says, and a listed string has its pack mass on top.  So the pack only ever
lifts the strings it lists: a listed string it hardly favours (a street a mile away) is read as by the ink alone,
never pushed below it, and a string off the list with clearly stronger ink wins (e > 0).  Without a pack (prior None),
or at e = 1 or w = 0, the posterior is the ink alone.  The reader gives `ink_logp` exactly by CTC
(`textread.ctc.string_log_probs`, mixed over the two reading directions) for every listed string, and for any other
strings it wants named.

The weight w and escape e (`fit_prior`): by maximum likelihood of Terra's string on Terra-matched words of the other
three quarters.  Never on the owner's answers: a word in or touching an answered block is refused (it raises), and so
is a word of the held-out quarter.  The spreads are refused the same words.
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
from scipy import optimize, stats

from sfzmaps.solver import paths

CRS = "EPSG:2227"
PACKS_DIR = paths.SOLVER_DIR / "experiments" / "wordread" / "packs"
KIND_ORDER = ("zoning code", "block number", "street name", "other")   # kind_p's order: textnet.data.KINDS'
ESCAPE = "escape"          # the prior's key for every string off its lists (lower case: never a reader's string)
OTHER = "other"            # the posterior's key for the escape's members not named in `ink_logp`
FOLDS = ("all", *paths.QUARTERS)   # a pack's spreads: every quarter, then each quarter left out
STREET_COLS = ("name", "street", "type", "active")
BLOCK_COLS = ("number",)
SPREAD_ARRAYS = ("street_ft", "street_deg", "block_ft")
SPREAD_SHARES = ("street_away", "block_away")
W_MAX = 64.0               # the weight is searched in [0, W_MAX] (a fit at the bound is reported: the prior rules)
E_LOGIT = (-16.0, 16.0)    # the escape mass is searched in logit space between these
TANGENT_FT = 1.0           # a centreline's direction at a point: from this far before it to this far after
MIN_MASS = 1e-12           # a string with less of the prior's mass than this is not listed
REACH_GRID = np.geomspace(0.1, 1e6, 1201)   # ft: where `reach` looks for the last distance an item keeps MIN_MASS
VOCAB = " &'(),-./0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"   # the reader's alphabet (`textread.model.VOCAB`)

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


# --- the measured spreads ---------------------------------------------------------------------------------------------

class _LogKDE:
    """A density of d >= 0 (per ft) from a Gaussian kernel density of log(1 + d) (Scott's rule); 0 everywhere when
    fewer than two distinct values were measured."""

    def __init__(self, sample: np.ndarray):
        u = np.log1p(np.asarray(sample, float))
        self.kde = stats.gaussian_kde(u) if len(np.unique(u)) >= 2 else None

    @property
    def width(self) -> float:
        return float(np.sqrt(self.kde.covariance[0, 0])) if self.kde is not None else 0.0

    def __call__(self, d) -> np.ndarray:
        d = np.asarray(d, float)
        if self.kde is None or not d.size:
            return np.zeros(d.shape)
        return self.kde(np.log1p(d).reshape(1, -1)).reshape(d.shape) / (1.0 + d)


class _AngleKDE:
    """A density of an angle in [0, 90] degrees (per degree) from a Gaussian kernel density reflected at 0 and 90."""

    def __init__(self, sample: np.ndarray):
        a = np.asarray(sample, float)
        self.kde = stats.gaussian_kde(a) if len(np.unique(a)) >= 2 else None

    def __call__(self, t) -> np.ndarray:
        t = np.asarray(t, float)
        if self.kde is None or not t.size:
            return np.zeros(t.shape)
        flat = t.reshape(-1)
        return (self.kde(flat) + self.kde(-flat) + self.kde(180.0 - flat)).reshape(t.shape)


@dataclass
class Spreads:
    """Where printed words sit relative to what they name (see the module), measured on the words that name their
    street (block) where they sit: street-name words' distance (ft) from their street's centreline and their axis'
    angle to it (degrees, 0 to 90); block-number words' distance outside their block (ft, 0 inside); and the shares
    of words that name no street (block) where they sit."""
    street_ft: np.ndarray
    street_deg: np.ndarray
    block_ft: np.ndarray
    street_away: float
    block_away: float

    def __post_init__(self):
        for k in SPREAD_ARRAYS:
            setattr(self, k, np.sort(np.asarray(getattr(self, k), float)))
        for k in SPREAD_SHARES:
            v = float(getattr(self, k))
            if not 0 < v < 1:
                raise ValueError(f"{k} must be a share strictly between 0 and 1, not {v}")
            setattr(self, k, v)

    @cached_property
    def street_density(self) -> _LogKDE:
        return _LogKDE(self.street_ft)

    @cached_property
    def angle_density(self) -> _AngleKDE:
        return _AngleKDE(self.street_deg)

    @cached_property
    def block_out_density(self) -> _LogKDE:
        return _LogKDE(self.block_ft[self.block_ft > 0])

    @property
    def block_in(self) -> float:
        """p_in: the share of the measured block-number words inside their block."""
        return float((self.block_ft == 0).mean()) if len(self.block_ft) else 0.0

    def summary(self) -> dict:
        def q(a):
            return {"n": int(len(a)), **({f"q{p}": round(float(np.quantile(a, p / 100)), 2) for p in (50, 90, 99)}
                                        if len(a) else {}), "max": round(float(a[-1]), 2) if len(a) else None}
        return {"street_ft": q(self.street_ft), "street_deg": q(self.street_deg), "block_ft": q(self.block_ft),
                "street_away": round(self.street_away, 4), "block_away": round(self.block_away, 4),
                "block_in": round(self.block_in, 4)}


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
    """One map's context pack (see the module).  Never a code at a lot."""
    year: int
    streets: gpd.GeoDataFrame
    blocks: gpd.GeoDataFrame
    codes: dict[str, str]
    spreads: dict[str, Spreads] = field(default_factory=dict)

    def __post_init__(self):
        self.streets = _frame(self.streets, STREET_COLS, "streets").reset_index(drop=True)
        self.blocks = _frame(self.blocks, BLOCK_COLS, "blocks").reset_index(drop=True)
        if "active" not in self.streets:
            self.streets["active"] = True
        self.codes = dict(sorted(self.codes.items()))

    # caches for the prior
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

    def reach(self, sp: Spreads) -> tuple[float, float]:
        """How far (ft) streets and blocks are looked for under spreads `sp`: past it, even an aligned street or the
        smallest block would have odds below MIN_MASS (`street_odds`, `block_odds`), so less than MIN_MASS of its
        kind."""
        key = id(sp)
        if key not in self._reaches:
            def last(odds):
                keep = np.flatnonzero(odds >= MIN_MASS)
                return float(REACH_GRID[min(keep[-1] + 1, len(REACH_GRID) - 1)]) if len(keep) else 0.0
            best_deg = np.linspace(0.0, 90.0, 181)[int(np.argmax(sp.angle_density(np.linspace(0.0, 90.0, 181))))]
            street = last(street_odds(self, sp, REACH_GRID, np.full(len(REACH_GRID), best_deg)))
            small = float(self.block_perimeter.min()) if len(self.blocks) else 1.0
            block = last(block_odds(self, sp, REACH_GRID, np.full(len(REACH_GRID), small), np.ones(len(REACH_GRID))))
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

    # the file: one parquet, a row per street centreline, block, code form and measured spread value
    def save(self, f: Path) -> None:
        rows = [{"layer": "meta", "name": "year", "value": float(self.year)}]
        rows += [{"layer": "street", "name": r.name, "street": r.street, "type": r.type, "active": bool(r.active),
                  "geometry": r.geometry} for r in self.streets.itertuples(index=False)]
        rows += [{"layer": "block", "name": r.number, "geometry": r.geometry}
                 for r in self.blocks.itertuples(index=False)]
        rows += [{"layer": "code", "name": code, "form": form} for form, code in self.codes.items()]
        for fold, s in self.spreads.items():
            for quantity in SPREAD_ARRAYS:
                rows += [{"layer": "spread", "name": quantity, "fold": fold, "value": float(v)}
                         for v in getattr(s, quantity)]
            rows += [{"layer": "spread", "name": k, "fold": fold, "value": getattr(s, k)} for k in SPREAD_SHARES]
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
        df = gpd.read_parquet(f)
        year = int(df.loc[(df["layer"] == "meta") & (df["name"] == "year"), "value"].iloc[0])
        st = df[df["layer"] == "street"]
        streets = gpd.GeoDataFrame({"name": st["name"].to_numpy(object), "street": st["street"].to_numpy(object),
                                    "type": st["type"].fillna("").to_numpy(object),
                                    "active": st["active"].astype(bool).to_numpy()},
                                   geometry=st.geometry.values, crs=df.crs)
        bl = df[df["layer"] == "block"]
        blocks = gpd.GeoDataFrame({"number": bl["name"].to_numpy(object)}, geometry=bl.geometry.values, crs=df.crs)
        co = df[df["layer"] == "code"]
        codes = dict(zip(co["form"], co["name"]))
        sp = df[df["layer"] == "spread"]
        spreads = {}
        for fold in [f for f in FOLDS if f in set(sp["fold"])] + sorted(set(sp["fold"]) - set(FOLDS)):
            mine = sp[sp["fold"] == fold]
            spreads[fold] = Spreads(**{q: mine.loc[mine["name"] == q, "value"].to_numpy(float) for q in SPREAD_ARRAYS},
                                    **{k: float(mine.loc[mine["name"] == k, "value"].iloc[0]) for k in SPREAD_SHARES})
        return cls(year=year, streets=streets, blocks=blocks, codes=codes, spreads=spreads)


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
    forms = json.loads((store.CROPS_DIR / "prior.json").read_text())["forms"]
    out = {n: n for n in names}
    out.update({f: c for f, c in forms.items() if c in names})
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
    (`sf_blocks`), and the era's code list (`era_codes`, or `codes`).  No spreads (`measure_spreads` adds them).
    ValueError when asked for lot codes: on the 2000 map they are the answer key, and no other era's are held."""
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
    """Each centreline's axis (degrees modulo 180) at distance `at` along it."""
    length = shapely.length(lines)
    a = shapely.line_interpolate_point(lines, np.clip(at - TANGENT_FT, 0, length))
    b = shapely.line_interpolate_point(lines, np.clip(at + TANGENT_FT, 0, length))
    d = shapely.get_coordinates(b) - shapely.get_coordinates(a)
    return np.rad2deg(np.arctan2(d[:, 1], d[:, 0])) % 180.0


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

def _shares(odds: np.ndarray) -> tuple[np.ndarray, float]:
    """P(item | box) from each item's odds against "none of them", and P(none | box)."""
    z = 1.0 + float(odds.sum())
    return odds / z, 1.0 / z


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
    n = sum(bool(f) for f in pack.forms_of_block)
    if not n or pack.area <= 0:
        return np.zeros(d.shape)
    p_in = sp.block_in
    with np.errstate(divide="ignore", invalid="ignore"):
        inside = np.where(d <= 0, p_in / np.asarray(area, float), 0.0)
        outside = np.where(d > 0, (1 - p_in) * sp.block_out_density(d) / np.asarray(perimeter, float), 0.0)
    return (1 - sp.block_away) / sp.block_away * (pack.area / n) * (inside + outside)


def street_shares(pack: Pack, sp: Spreads, x: float, y: float, axis: float) -> tuple[dict[str, float], float]:
    """P(j | box) for each street name j near the box, and P(0 | box) (see the module)."""
    rows, d, deg = street_offsets(pack, x, y, axis, reach=pack.reach(sp)[0])
    keep = pack.named_street[rows]
    rows, d, deg = rows[keep], d[keep], deg[keep]
    nearest: dict[str, int] = {}                 # each street by its nearest centreline
    for k, r in enumerate(rows):
        name = pack.street_names[r]
        if name not in nearest or d[k] < d[nearest[name]]:
            nearest[name] = k
    names = sorted(nearest)
    idx = np.array([nearest[n] for n in names], np.int64)
    if not len(idx):
        return {}, 1.0
    share, none = _shares(street_odds(pack, sp, d[idx], deg[idx]))
    return dict(zip(names, share.tolist())), none


def block_shares(pack: Pack, sp: Spreads, x: float, y: float) -> tuple[dict[int, float], float]:
    """P(b | box) for each block row b near the box, and P(0 | box) (see the module)."""
    rows, d = block_offsets(pack, x, y, reach=pack.reach(sp)[1])
    keep = np.array([bool(pack.forms_of_block[r]) for r in rows], bool)
    rows, d = rows[keep], d[keep]
    if not len(rows):
        return {}, 1.0
    share, none = _shares(block_odds(pack, sp, d, pack.block_perimeter[rows], pack.block_area[rows]))
    return dict(zip(rows.tolist(), share.tolist())), none


def prior(pack: Pack | None, box_map: dict, kind_p, spreads: Spreads | None = None) -> dict[str, float]:
    """The log prior over the strings a box may print (see the module): each listed string and ESCAPE.  `box_map`:
    the box's centre `x`, `y` and `axis` on the map (`box_on_map`); `kind_p`: its kind probabilities (KIND_ORDER;
    normalised here); `spreads`: the pack's "all" when None.  No pack: every mass on the escape."""
    kp = np.asarray(kind_p, float).reshape(-1)
    if kp.shape != (len(KIND_ORDER),) or not np.isfinite(kp).all() or (kp < 0).any() or kp.sum() <= 0:
        raise ValueError(f"kind_p must be {len(KIND_ORDER)} non-negative numbers ({KIND_ORDER}), not {kind_p}")
    kp = kp / kp.sum()
    if pack is None:
        return {ESCAPE: 0.0}
    sp = spreads if spreads is not None else pack.spreads.get("all")
    if sp is None:
        raise ValueError("a prior needs the pack's measured spreads (`measure_spreads`)")
    x, y, axis = float(box_map["x"]), float(box_map["y"]), float(box_map["axis"])
    mass: dict[str, float] = {}
    escape = 0.0

    def add(forms, m):
        for f in forms:
            mass[f] = mass.get(f, 0.0) + m / len(forms)

    k = kp[KIND_ORDER.index("zoning code")]
    if k > 0:
        if pack.code_forms:
            for forms in pack.code_forms.values():
                add(forms, k / len(pack.code_forms))
        else:
            escape += k
    k = kp[KIND_ORDER.index("block number")]
    if k > 0:
        share, none = block_shares(pack, sp, x, y)
        for r, s in share.items():
            add(pack.forms_of_block[r], k * s)
        escape += k * none
    k = kp[KIND_ORDER.index("street name")]
    if k > 0:
        share, none = street_shares(pack, sp, x, y, axis)
        for name, s in share.items():
            add(pack.forms_of_street[name], k * s)
        escape += k * none
    escape += kp[KIND_ORDER.index("other")]
    escape += sum(m for m in mass.values() if m < MIN_MASS)
    out = {s: math.log(m) for s, m in sorted(mass.items()) if m >= MIN_MASS}
    out[ESCAPE] = math.log(escape) if escape > 0 else -math.inf
    return out


# --- the posterior ----------------------------------------------------------------------------------------------------

def ink_logp(reads, q: float, strings) -> dict[str, float]:
    """log p(s | ink) for each of `strings`, exactly by CTC (`textread.ctc.string_log_probs`), mixed over the two
    reading directions as the reader mixes them (`textread.ctc.mixture`): `reads` the crop's log-probabilities read
    as cut and turned a half turn (frames x classes each), `q` the classifier's p(the crop as cut is upside down)."""
    from sfzmaps.solver.textread import ctc
    strings = list(strings)
    lp0, lp180 = (ctc.string_log_probs(r, strings) for r in reads)
    with np.errstate(divide="ignore"):
        ink = np.logaddexp(np.log(1.0 - q) + lp0, np.log(q) + lp180)
    return dict(zip(strings, ink.tolist()))


def _wlog(w: float, x: float) -> float:
    """w x, with 0 x (-inf) = 0: a prior with no weight says nothing, even of an impossible string."""
    return 0.0 if w == 0 else w * x


def _lse(v) -> float:
    v = np.asarray(v, float)
    m = v.max() if len(v) else -math.inf
    return float(m + np.log(np.exp(v - m).sum())) if np.isfinite(m) else -math.inf


def posterior(ink_logp: dict[str, float], prior: dict[str, float] | None, w: float, e: float) -> dict[str, float]:
    """The log posterior (see the module) of every string of `ink_logp` and of OTHER (every string neither listed nor
    named).  `ink_logp`: log p(s | ink) by CTC for every listed string of `prior` (ValueError when one is missing)
    and any others; `prior`: `prior`'s output, or None for none (the ink alone); `w` >= 0; 0 < e <= 1."""
    if not 0 < e <= 1:
        raise ValueError(f"the escape mass e must be in (0, 1], not {e}: a prior with no escape filters")
    if w < 0:
        raise ValueError(f"the prior's weight w must be >= 0, not {w}")
    if ESCAPE in ink_logp or OTHER in ink_logp:
        raise ValueError(f"{ESCAPE!r} and {OTHER!r} are the posterior's own keys, not strings")
    pr = {ESCAPE: 0.0} if prior is None else prior
    listed = [s for s in pr if s != ESCAPE]
    missing = [s for s in listed if s not in ink_logp]
    if missing:
        raise ValueError(f"no ink for {len(missing)} listed strings (e.g. {missing[:3]}): the reader scores them all")
    log_e = math.log((1 - e) * math.exp(pr.get(ESCAPE, -math.inf)) + e)
    log_keep = math.log(1 - e) if e < 1 else -math.inf
    score = {}
    for s, lp in ink_logp.items():
        score[s] = lp + _wlog(w, float(np.logaddexp(log_e, log_keep + pr[s])) if s in pr else log_e)
    with np.errstate(over="ignore"):
        rest = 1.0 - float(np.exp(np.array(list(ink_logp.values()), float)).sum())
    score[OTHER] = (math.log(rest) if rest > 0 else -math.inf) + _wlog(w, log_e)
    z = _lse(list(score.values()))
    return {s: v - z for s, v in score.items()}


# --- holding the answers out ------------------------------------------------------------------------------------------

def refuse_answered(words: pd.DataFrame, held_blocks=None, quarter: int | None = None) -> None:
    """ValueError when any word is in or touches an answered block (`block` in `held_blocks`, default
    `textnet.reader_stage.held_blocks()`; or `touch_blocks`), or lies in the held-out `quarter`: a prior is never
    fitted on the owner's answers, nor on the quarter it is scored on."""
    if "block" not in words:
        raise ValueError("each word needs its block (`block`) so that answered blocks can be refused")
    if held_blocks is None:
        from sfzmaps.solver.textnet import reader_stage
        held_blocks = reader_stage.held_blocks()
    held = {str(b) for b in held_blocks}
    bad = words["block"].astype(str).isin(held)
    if bad.any():
        raise ValueError(f"{int(bad.sum())} words lie in an answered block (e.g. {words.loc[bad, 'block'].iloc[0]}): "
                         "a prior is never fitted on the owner's answers")
    if "touch_blocks" in words and words["touch_blocks"].astype(bool).any():
        raise ValueError(f"{int(words['touch_blocks'].astype(bool).sum())} words touch an answered block")
    if quarter is not None:
        if "quarter" not in words:
            raise ValueError("each word needs its quarter to keep the held-out quarter out")
        if (words["quarter"].astype(int) == int(quarter)).any():
            raise ValueError(f"words of the held-out quarter {quarter}: fit on the other three quarters")


# --- the spreads, measured --------------------------------------------------------------------------------------------

def _street_rows_by_form(pack: Pack) -> dict[str, np.ndarray]:
    names: dict[str, list[str]] = {}
    for name, forms in pack.forms_of_street.items():
        for f in forms:
            names.setdefault(f, []).append(name)
    rows_of = pd.Series(np.arange(len(pack.streets))).groupby(pack.street_names).agg(list).to_dict()
    return {f: np.array(sorted(r for n in ns for r in rows_of[n]), np.int64) for f, ns in names.items()}


def word_offsets(pack: Pack, words: pd.DataFrame) -> pd.DataFrame:
    """Each Terra word against what it names (see the module): `found` (a street, or block, prints its text), its own
    one (the nearest that does), `here` (that is also the nearest of all), `d_ft` (from its own street's centreline,
    or outside its own block) and `d_deg` (its axis to its own street's centreline).  `words`: text, kind, x, y, axis
    (`box_on_map`)."""
    by_form = _street_rows_by_form(pack)
    named_rows = np.flatnonzero(pack.named_street)
    named_tree = shapely.STRtree(pack.streets.geometry.values[named_rows])
    block_rows: dict[str, list[int]] = {}
    for r, forms in enumerate(pack.forms_of_block):
        for f in forms:
            block_rows.setdefault(f, []).append(r)
    out = []
    for w in words.itertuples(index=False):
        rec = {"found": False, "here": False, "d_ft": np.nan, "d_deg": np.nan}
        pt = shapely.Point(w.x, w.y)
        if w.kind == "street name" and w.text in by_form:
            rows = by_form[w.text]
            d = shapely.distance(pack.streets.geometry.values[rows], pt)
            k = int(np.argmin(d))
            _, dk, deg = street_offsets(pack, w.x, w.y, w.axis, rows=rows[k:k + 1])
            near = named_rows[int(named_tree.query_nearest(pt, all_matches=False)[0])]
            rec = {"found": True, "here": bool(pack.street_names[near] == pack.street_names[rows[k]]),
                   "d_ft": float(dk[0]), "d_deg": float(deg[0])}
        elif w.kind == "block number" and w.text in block_rows:
            rows = np.array(block_rows[w.text], np.int64)
            d = shapely.distance(pack.blocks.geometry.values[rows], pt)
            k = int(np.argmin(d))
            near = int(pack.block_tree.query_nearest(pt, all_matches=False)[0]) if d[k] > 0 else int(rows[k])
            here = near == rows[k] or pack.blocks.geometry.values[near].distance(pt) >= d[k]   # a tie is its own
            rec = {"found": True, "here": bool(here), "d_ft": float(d[k]), "d_deg": np.nan}
        out.append(rec)
    return pd.DataFrame(out, index=words.index, columns=["found", "here", "d_ft", "d_deg"])


def measure_spreads(pack: Pack, words: pd.DataFrame, held_blocks=None, quarter: int | None = None) -> tuple:
    """The spreads (`Spreads`, see the module) measured on Terra's `words` (`word_offsets`), refusing answered-block
    words and the held-out `quarter` (`refuse_answered`); and the offsets per word."""
    refuse_answered(words, held_blocks, quarter)
    off = word_offsets(pack, words)
    out = {}
    for kind, key in (("street name", "street"), ("block number", "block")):
        mine = (words["kind"] == kind).to_numpy()
        here = mine & off["here"].to_numpy(bool)
        if not here.any() or here.sum() == mine.sum():
            raise ValueError(f"can't measure the {kind} spreads: {int(here.sum())} of {int(mine.sum())} words name "
                             "what they sit by")
        out[f"{key}_away"] = 1.0 - here.sum() / mine.sum()
        out[f"{key}_ft"] = off.loc[here, "d_ft"].to_numpy()
        if key == "street":
            out["street_deg"] = off.loc[here, "d_deg"].to_numpy()
    return Spreads(**out), off


# --- fitting w and e --------------------------------------------------------------------------------------------------

def _arrays(words: pd.DataFrame):
    """Per word, padded: the listed strings' ink (n, K; -inf pads) and log prior (n, K; 0 pads), the escape class's ink
    (log(1 - the listed ink)), prior(ESCAPE), and the truth: its listed column (-1 off the list) and its ink."""
    n = len(words)
    rows = list(zip(words["truth"], words["ink"], words["prior"]))
    K = max([len([s for s in pr if s != ESCAPE]) for _, _, pr in rows] + [1])
    A, B = np.full((n, K), -np.inf), np.zeros((n, K))
    esc_ink, esc_pr, col, t_ink = np.zeros(n), np.zeros(n), np.full(n, -1), np.zeros(n)
    for i, (truth, ink, pr) in enumerate(rows):
        listed = [s for s in pr if s != ESCAPE]
        if truth not in ink:
            raise ValueError(f"word {i}: no ink for its truth {truth!r}")
        missing = [s for s in listed if s not in ink]
        if missing:
            raise ValueError(f"word {i}: no ink for listed strings {missing[:3]}")
        A[i, :len(listed)] = [ink[s] for s in listed]
        B[i, :len(listed)] = [pr[s] for s in listed]
        with np.errstate(over="ignore"):
            rest = 1.0 - float(np.exp(A[i, :len(listed)]).sum())
        esc_ink[i] = math.log(rest) if rest > 0 else -math.inf
        esc_pr[i] = math.exp(pr.get(ESCAPE, -math.inf))
        col[i] = listed.index(truth) if truth in listed else -1
        t_ink[i] = ink[truth]
    return A, B, esc_ink, esc_pr, col, t_ink


def _loglik(params, arrays) -> np.ndarray:
    """log posterior of each word's truth at (w, e) (`posterior`'s, vectorised)."""
    w, e = params
    A, B, esc_ink, esc_pr, col, t_ink = arrays
    log_keep = math.log(1 - e) if e < 1 else -math.inf
    log_e = np.log((1 - e) * esc_pr + e)
    listed = A + (w * np.logaddexp(log_e[:, None], log_keep + B) if w else 0.0)
    escape = esc_ink + w * log_e
    m = np.maximum(listed.max(1), escape)
    m = np.where(np.isfinite(m), m, 0.0)
    with np.errstate(divide="ignore"):
        z = m + np.log(np.exp(listed - m[:, None]).sum(1) + np.exp(escape - m))
    on = col >= 0
    num = np.where(on, listed[np.arange(len(col)), np.maximum(col, 0)], t_ink + w * log_e)
    return num - z


def truth_loglik(words: pd.DataFrame, w: float, e: float) -> np.ndarray:
    """Each word's log posterior of its `truth` at (w, e) (`posterior`; `words` as `fit_prior` takes them).  w = 0 is
    the ink alone."""
    return _loglik((w, e), _arrays(words))


def fit_prior(words: pd.DataFrame, held_blocks=None, quarter: int | None = None, w_max: float = W_MAX) -> dict:
    """w and e by maximum likelihood of each word's `truth` (Terra's string) under `posterior` (see the module).
    `words`: per Terra-matched word, `truth`, `ink` (log p(s | ink) for the truth and every listed string), `prior`
    (`prior`'s output), `block` and, optionally, `touch_blocks` and `quarter`; refused when any is the owner's
    (`refuse_answered`, with `held_blocks` and the held-out `quarter`).  A coarse grid, then L-BFGS-B from its best:
    deterministic.  Returns w, e, the summed log-likelihood with them and with the ink alone, counts and whether
    either sits at its search bound."""
    refuse_answered(words, held_blocks, quarter)
    if not len(words):
        raise ValueError("no words to fit on")
    arrays = _arrays(words)

    def nll(v):
        return -float(_loglik((v[0], 1.0 / (1.0 + math.exp(-v[1]))), arrays).sum())

    grid = [(w, z) for w in np.r_[0.0, np.geomspace(0.01, w_max, 48)] for z in np.linspace(*E_LOGIT, 33)]
    vals = [nll(g) for g in grid]
    start = grid[int(np.argmin(vals))]
    res = optimize.minimize(nll, np.array(start, float), method="L-BFGS-B", bounds=[(0.0, w_max), E_LOGIT])
    best = res.x if res.fun <= min(vals) else np.array(start, float)
    w, e = float(best[0]), 1.0 / (1.0 + math.exp(-float(best[1])))
    ll = _loglik((w, e), arrays)
    ink_only = _loglik((0.0, 0.5), arrays)
    return {"w": round(w, 4), "e": float(f"{e:.4g}"), "loglik": float(ll.sum()), "loglik_ink_only":
            float(ink_only.sum()), "n": int(len(words)), "truth_listed": int((arrays[4] >= 0).sum()),
            "w_at_bound": bool(w <= 1e-6 or w >= w_max - 1e-6),
            "e_at_bound": bool(best[1] <= E_LOGIT[0] + 1e-6 or best[1] >= E_LOGIT[1] - 1e-6)}
