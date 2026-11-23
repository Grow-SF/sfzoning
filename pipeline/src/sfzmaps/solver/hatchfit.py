"""The 1921 hatching per drawn region, step 2: a model of the legend's six patterns fitted on the scans' patches
(`hatching`), and each lot's evidence for each district as ADDITIVE costs, so that the joint solve sums the evidence
over whole regions.

The model: a multinomial logistic regression (the spike's logistic fit: median imputation with an indicator for each
measure that can be missing, standardisation, an L2 penalty), one row per patch.  A patch is labelled by every lot it
overlaps that the pattern reader read (`reads`; its reads are training labels only, never an input), with weight =
the read's confidence x the lot's share of the patch (`targets`): a patch straddling a border weighs on both sides,
each by its share.  Each legend swatch is a row of weight 1 for its district.  The features (`design`) are the
patch's own measures: the morphology, the periodicity, 1R's own hatch, the power per period band and orientation,
the hatch power and the variance, the share of the patch in its block, the periods relative to the sheet's 1R
period (so the scans at other resolutions compare), and the orientation moments in the page's frame and in the
block's (m2 = |m2| e^{2 i theta}: one family of lines; m4: a right-angle cross) each times the hatch power, since |m2|
alone is a power-normalised coherence that a few px of stray ink have in full: the fit learns how far each district
follows the block's axis.  Nothing of a block or a sheet as a whole (its period, its scale, its block's elongation)
is a feature: it would act as a prior, counted again for every patch of the region.

Held out by quarter (the solver's quarters, `grid.quarter_of`; nw ne sw se as `paths.QUARTERS` and lots.npz
`quarter`): quarter q's model is fitted only on patches that touch no lot of q and whose centre is not in q
(`touches`).  Its L2 strength is fitted by the held-out log loss with each of the other three quarters left out in
turn (`fit_quarter`).

The evidence (`lot_evidence`): for quarter q's model, a patch's log-likelihood ratio is log p_k(patch) - log pi_k
(pi: the districts' weighted shares in its training rows), and a lot's is the sum, over the patches it lies in, of
its share of the patch times theirs.  A patch's shares over its lots add to one, so every patch counts once over the
lots it covers: a region pools the evidence of every patch inside it, and a lot between hatch lines takes the hatch
its patches see.  The costs are U = -tau_q x that, less each row's minimum.  tau_q is the slope of a weighted
logistic calibration (weight: the read's confidence) of the lots' pooled log ratios against the reads on the three
training quarters, each lot's ratios pooled from the fit that left its own quarter out (the L2 search's fits, at the
chosen strength).  A lot no patch covers has a zero row: no evidence, never a default district.

Writes, under data/solver/experiments/hatch-regions/evidence/:
  evidence_{q}.npz   U (float32, lots x districts), names (the 1921 legend's six, sorted: ruling R2), tau, and
                     covered (bool per lot: some patch lies on it), alpha (the L2 strength), prior (pi)
  summary.json       per quarter: the L2 search, alpha, tau, the training rows and weight per district, and the
                     confusion of the evidence's argmin against the reads of the held-out quarter
and with --score, data/solver/experiments/hatch-regions/score_patches.json: the argmin of U per lot (each lot from its
own quarter's file; no solve) scored on the owner's answers (`answers1921`; never trained on) beside the published
map, the final solve and the spike's round 2.

    OMP_NUM_THREADS=1 SFZMAPS_MAX_USD=0 uv run python -m sfzmaps.solver.hatchfit --series 1921 --workers 4
    uv run python -m sfzmaps.solver.hatchfit --series 1921 --score

$0: no API calls.  The reads() and the logistic fit are ported from the spike (branch
worktree-agent-ab10cd06a6d7b4538, scripts/spikes/hatch1921/measure.py and model.py).
"""
from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass, field
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import minimize, minimize_scalar
from scipy.special import logsumexp

from sfzmaps.solver import grid, hatching, paths
from sfzmaps.solver.codes import canon

SERIES = hatching.SERIES
EXPERIMENT = hatching.EXPERIMENT
EVIDENCE = EXPERIMENT / "evidence"
NAMES: tuple[str, ...] = tuple(sorted(("1R", "2R", "C", "LI", "HI", "U")))   # the 1921 legend's districts (R2)
SKIP_VIA = hatching.SKIP_VIA       # reads of a lot not on the sheet (the reader's own marks)
# The L2 strengths searched (log10, on the mean loss over the training weight, the features standardised): at the
# optimum alpha W = -(the data term's gradient), whose entries are at most E|x| <= 1 for a standardised x, so alpha
# = 10 holds every coefficient within 0.1 nats per standard deviation (every patch near the prior), and 1e-8 lets
# them reach 1e8 (no penalty in effect).  The search stops when the strength is known to within 10^0.05 (12%).
LOG10_ALPHA = (-8.0, 1.0)
XATOL = 0.05
GRID = tuple(f"g{b}_{o}" for b in range(len(hatching.BANDS) - 1) for o in range(hatching.N_ORI))
PLAIN = (*hatching.MORPH_KEYS, "periodicity", "dir1r_acf", "dir1r_share", "hatch_power", "var", *GRID)
ANGLES = tuple(f"{frame}_{m}_{part}_hp" for frame in ("page", "block") for m in ("m2", "m4") for part in ("re", "im"))
COLUMNS = (*PLAIN, "area_share", "logperiod_rel", "peak_period_rel", *ANGLES)


# --- the reads (training labels only) --------------------------------------------------------------------------------

def reads(work: Path = hatching.WORK, warp: Path = hatching.WARP) -> pd.DataFrame:
    """The pattern reader's reads (sheet, blklot, code in the one spelling, confidence, via) on every sheet with reads
    and a placement, less those it marks off the sheet or with no tile (SKIP_VIA) and those with no code (the
    spike's `reads`)."""
    rows = []
    for d in sorted(p for p in Path(work).iterdir() if p.is_dir()):
        f = d / "district_pattern.json"
        if not (f.is_file() and (Path(warp) / d.name / "transform.json").is_file()):
            continue
        for p in json.loads(f.read_text())["parcels"]:
            code = p.get("code")
            if not isinstance(code, str) or not code.strip() or p.get("via") in SKIP_VIA:
                continue
            rows.append({"sheet": d.name, "blklot": str(p["blklot"]), "code": canon(code),
                         "confidence": float(p["confidence"]), "via": p.get("via", "")})
    return pd.DataFrame(rows, columns=["sheet", "blklot", "code", "confidence", "via"])


def lot_labels(rd: pd.DataFrame, blklots, names=NAMES) -> np.ndarray:
    """(lots x districts): each lot's read as its confidence in its district's column (every lot of the table with
    the read's blklot; a lot read more than once takes the mean of its reads); 0 where unread or read as a code not
    in `names`."""
    blklots = pd.Series(np.asarray(blklots).astype(str))
    out = np.zeros((len(blklots), len(names)))
    count = np.zeros(len(blklots))
    rd = rd[rd["code"].isin(names)]
    lots = pd.Series(np.arange(len(blklots))).groupby(blklots.to_numpy()).apply(list)
    for r in rd.itertuples():
        for lot in lots.get(r.blklot, []):
            out[lot, names.index(r.code)] += r.confidence
            count[lot] += 1
    return out / np.maximum(count, 1)[:, None]


# --- the data -------------------------------------------------------------------------------------------------------

@dataclass
class Data:
    """What the fits read: the patches (sheet, patch, cx, cy, side, period and the measures; one row each, in a fixed
    order), patch_lots (sheet, patch, lot, share), each lot's quarter (lots.npz `quarter`), each lot's read
    (`lot_labels`) and the legend swatches (code, period and the measures)."""
    patches: pd.DataFrame
    patch_lots: pd.DataFrame
    quarter: np.ndarray
    labels: np.ndarray
    swatches: pd.DataFrame


def load_swatches(root: Path, sheets: dict) -> pd.DataFrame:
    """The legend swatches measured as patches (`hatching.measure_swatches`: swatches.parquet, or measured now), with
    the legend sheet's period and each box's px."""
    legend = sheets["legend"]
    period = sheets["sheets"][legend["sheet"]]["period"]
    f = root / "swatches.parquet"
    if f.is_file():
        sw = pd.read_parquet(f)
    else:
        src = hatching.sources()
        legend = {**legend, "boxes": json.loads(src.legends.read_text())[SERIES]["swatches"]}
        sw = hatching.measure_swatches(src, legend, period)
    return sw.assign(period=float(period), box_px=(sw["x1"] - sw["x0"]) * (sw["y1"] - sw["y0"]))


def load_data(root: Path = EXPERIMENT, solver: Path = hatching.SOLVER_1921) -> Data:
    """Task 2's patches and their lots' shares (as the run wrote them: patch_lots/<sheet>.parquet), the lots'
    quarters and the reads on the solver's lot table (its row order is the lot index)."""
    sheets = sorted(p.stem for p in (root / "patches").glob("*.parquet"))
    if not sheets:
        raise FileNotFoundError(f"no patches under {root / 'patches'}: run `python -m sfzmaps.solver.hatching`")
    patches = pd.concat([pd.read_parquet(root / "patches" / f"{s}.parquet") for s in sheets], ignore_index=True)
    links = pd.concat([pd.read_parquet(root / "patch_lots" / f"{s}.parquet").assign(sheet=s) for s in sheets],
                      ignore_index=True)
    with np.load(solver / "lots" / "lots.npz") as z:
        quarter = z["quarter"]
    blklots = pd.read_parquet(solver / "lots" / "table.parquet", columns=["blklot"])["blklot"]
    if len(blklots) != len(quarter):
        raise ValueError(f"the lot table has {len(blklots)} lots, lots.npz {len(quarter)}")
    meta = json.loads((root / "sheets.json").read_text())
    return Data(patches, links[["sheet", "patch", "lot", "share"]], quarter, lot_labels(reads(), blklots),
                load_swatches(root, meta))


def _rows(patches: pd.DataFrame, patch_lots: pd.DataFrame) -> np.ndarray:
    """Each link's row in `patches`, by (sheet, patch) (by patch alone when either has no sheet)."""
    keys = ["sheet", "patch"] if "sheet" in patches and "sheet" in patch_lots else ["patch"]
    index = pd.MultiIndex.from_frame(patches[keys].astype({"patch": np.int64}))
    if not index.is_unique:
        raise ValueError(f"patches repeat a {'/'.join(keys)}")
    rows = index.get_indexer(pd.MultiIndex.from_frame(patch_lots[keys].astype({"patch": np.int64})))
    if (rows < 0).any():
        raise ValueError(f"{int((rows < 0).sum())} patch_lots rows name no patch")
    return rows


def _spread(patches: pd.DataFrame, patch_lots: pd.DataFrame, n_lots: int) -> sparse.csr_matrix:
    """(lots x patches): each lot's share of each patch."""
    rows = _rows(patches, patch_lots)
    lots = patch_lots["lot"].to_numpy(np.int64)
    if len(lots) and (lots.min() < 0 or lots.max() >= n_lots):
        raise ValueError(f"patch_lots names lots outside 0..{n_lots - 1}")
    return sparse.csr_matrix((patch_lots["share"].to_numpy(float), (lots, rows)), shape=(n_lots, len(patches)))


def targets(patches: pd.DataFrame, patch_lots: pd.DataFrame, labels: np.ndarray) -> np.ndarray:
    """(patches x districts): each patch's training weight per district, the sum over the read lots it overlaps of
    the lot's share of the patch x the read's confidence (`labels`)."""
    return np.asarray(_spread(patches, patch_lots, len(labels)).T @ labels)


def touches(data: Data) -> np.ndarray:
    """(patches x 4) bool: the quarters a patch touches, its centre's and every one of its lots'."""
    out = np.zeros((len(data.patches), len(paths.QUARTERS)), bool)
    centre = grid.quarter_of(data.patches[["cx", "cy"]].to_numpy(float))
    out[np.arange(len(out)), centre] = True
    out[_rows(data.patches, data.patch_lots), data.quarter[data.patch_lots["lot"].to_numpy(np.int64)]] = True
    return out


# --- the features ---------------------------------------------------------------------------------------------------

def design(df: pd.DataFrame) -> np.ndarray:
    """The model's features (COLUMNS) of patches or swatches: their measures (see the module), NaN where a measure is
    missing (no hatch-band power: no period; no block: no block frame)."""
    box = df["box_px"] if "box_px" in df else df["side"] ** 2
    hp = df["hatch_power"]
    cols = [df[c] for c in PLAIN]
    cols += [df["area"] / box, df["logperiod"] - np.log(df["period"]), np.log(df["peak_period"] / df["period"])]
    for pre in ("", "blk_"):
        cols += [df[f"{pre}{m}_{part}"] * hp for m in ("m2", "m4") for part in ("re", "im")]
    return np.column_stack([np.asarray(c, float) for c in cols])


@dataclass
class Prep:
    """The spike's SimpleImputer(median, add_indicator) + StandardScaler, fitted on a fit's training rows."""
    median: np.ndarray
    missing: np.ndarray
    mean: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(cls, F: np.ndarray) -> "Prep":
        nan = np.isnan(F)
        with np.errstate(all="ignore"):
            median = np.where(nan.all(0), 0.0, np.nanmedian(np.where(nan.all(0), 0.0, F), 0))
        missing = np.flatnonzero(nan.any(0))
        X = np.c_[np.where(nan, median, F), nan[:, missing]]
        scale = X.std(0)
        return cls(median, missing, X.mean(0), np.where(scale > 0, scale, 1.0))

    def __call__(self, F: np.ndarray) -> np.ndarray:
        nan = np.isnan(F)
        return (np.c_[np.where(nan, self.median, F), nan[:, self.missing]] - self.mean) / self.scale


# --- the fit --------------------------------------------------------------------------------------------------------

@dataclass
class Fit:
    """One fitted logistic: its preparation, coefficients (features x districts), intercepts, the districts' weighted
    shares in its training rows (pi), its parameters as a vector (a warm start for the next fit), and whether L-BFGS
    converged and in how many iterations."""
    prep: Prep
    W: np.ndarray
    b: np.ndarray
    prior: np.ndarray
    theta: np.ndarray
    converged: bool = True
    iterations: int = 0

    def log_proba(self, F: np.ndarray) -> np.ndarray:
        Z = self.prep(F) @ self.W + self.b
        return Z - logsumexp(Z, axis=1, keepdims=True)


def fit_logistic(F: np.ndarray, Y: np.ndarray, alpha: float, start: np.ndarray | None = None) -> Fit:
    """The multinomial logistic minimising the weighted log loss (Y: rows x districts, each row's weight per
    district) per unit of weight + alpha / 2 |W|^2 (the intercepts unpenalised), by L-BFGS from `start` (zeros)."""
    prep = Prep.fit(F)
    X = prep(F)
    n, f = X.shape
    K = Y.shape[1]
    total = Y.sum()
    if total <= 0:
        raise ValueError("no training weight")
    s = Y.sum(1, keepdims=True)

    def objective(theta):
        W, b = theta[:f * K].reshape(f, K), theta[f * K:]
        Z = X @ W + b
        logp = Z - logsumexp(Z, axis=1, keepdims=True)
        G = (np.exp(logp) * s - Y) / total
        loss = -(Y * logp).sum() / total + 0.5 * alpha * (W * W).sum()
        return loss, np.concatenate([(X.T @ G + alpha * W).ravel(), G.sum(0)])

    x0 = np.zeros(f * K + K) if start is None or len(start) != f * K + K else start
    res = minimize(objective, x0, jac=True, method="L-BFGS-B", options={"maxiter": 15000})
    theta = res.x
    prior = Y.sum(0) / total
    return Fit(prep, theta[:f * K].reshape(f, K), theta[f * K:], prior, theta, bool(res.success), int(res.nit))


def heldout_loss(fit: Fit, F: np.ndarray, Y: np.ndarray) -> tuple[float, float]:
    """The weighted log loss of rows (its sum, and their weight)."""
    return float(-(Y * fit.log_proba(F)).sum()), float(Y.sum())


def pool(llr: np.ndarray, spread: sparse.csr_matrix) -> np.ndarray:
    """(lots x districts): each lot's sum over its patches of its share x the patch's log ratios."""
    return np.asarray(spread @ llr)


def calibrate(L: np.ndarray, labels: np.ndarray) -> float:
    """tau: the slope of the weighted logistic calibration softmax(tau L) of the lots' pooled log ratios L against
    their reads (labels: each lot's confidence per district).  Convex in tau: Brent's search from (0, 1) finds it."""
    total = labels.sum()

    def nll(t):
        z = t * L
        return float(-(labels * (z - logsumexp(z, axis=1, keepdims=True))).sum() / total)

    t = float(minimize_scalar(nll, bracket=(0.0, 1.0), method="brent").x)
    if not t > 0:
        raise ValueError(f"the patches' pooled evidence calibrates to tau = {t}: it says nothing of the reads")
    return t


@dataclass
class Model:
    """Quarter q's model: the districts (`names`, the columns of its probabilities), the fit, the L2 strength alpha,
    tau, the patch rows it trained on (`trained`, rows of Data.patches), the L2 search ((log10 alpha, held-out loss)
    as evaluated), the training rows and weight per district, and how many of its fits (the search's, tau's and its
    own) stopped short of L-BFGS's convergence."""
    names: tuple[str, ...]
    fit: Fit
    alpha: float
    tau: float
    trained: np.ndarray
    search: list = field(default_factory=list)
    rows: dict = field(default_factory=dict)
    unconverged: int = 0

    @property
    def prior(self) -> np.ndarray:
        return self.fit.prior

    def log_proba(self, df: pd.DataFrame) -> np.ndarray:
        return self.fit.log_proba(design(df))


def _onehot(codes, names=NAMES) -> np.ndarray:
    out = np.zeros((len(codes), len(names)))
    for i, c in enumerate(codes):
        out[i, names.index(canon(c))] = 1.0
    return out


def fit_quarter(q: str, data: Data | None = None) -> Model:
    """Quarter q's model (see the module): fitted on the patches that touch neither q's lots nor its ground, with
    the legend's swatches; its L2 strength the one whose fits, each leaving one of the other three quarters out,
    predict the patches of the quarter left out best (weighted log loss); tau calibrated on the other quarters' lots
    from those fits."""
    data = data or load_data()
    qi = paths.QUARTERS.index(q)
    F, Fs = design(data.patches), design(data.swatches)
    Y, Ys = targets(data.patches, data.patch_lots, data.labels), _onehot(data.swatches["code"])
    T = touches(data)
    base = (Y.sum(1) > 0) & ~T[:, qi]
    others = [r for r in range(len(paths.QUARTERS)) if r != qi]
    folds = {r: (base & ~T[:, r], base & T[:, r]) for r in others}
    starts: dict[int, np.ndarray] = {}
    search: list[tuple[float, float]] = []
    unconverged = []

    def fold_fit(r, alpha):
        tr = folds[r][0]
        fit = fit_logistic(np.r_[F[tr], Fs], np.r_[Y[tr], Ys], alpha, starts.get(r))
        starts[r] = fit.theta
        unconverged.append(not fit.converged)
        return fit

    def loss(log_alpha):
        tot = w = 0.0
        for r in others:
            ev = folds[r][1]
            s, ws = heldout_loss(fold_fit(r, 10.0 ** log_alpha), F[ev], Y[ev])
            tot, w = tot + s, w + ws
        search.append((float(log_alpha), tot / w))
        return tot / w

    minimize_scalar(loss, bounds=LOG10_ALPHA, method="bounded", options={"xatol": XATOL})
    log_alpha = min(search, key=lambda t: (t[1], t[0]))[0]
    alpha = 10.0 ** log_alpha
    # tau: each training quarter's lots pooled from the fit that left that quarter out, at the chosen strength
    spread = _spread(data.patches, data.patch_lots, len(data.quarter))
    Ls, labs = [], []
    for r in others:
        fit = fold_fit(r, alpha)
        near = T[:, r]
        llr = np.zeros((len(F), len(NAMES)))
        llr[near] = fit.log_proba(F[near]) - np.log(fit.prior)
        L = pool(llr, spread)
        lots = (data.quarter == r) & (data.labels.sum(1) > 0) & (np.diff(spread.indptr) > 0)
        Ls.append(L[lots])
        labs.append(data.labels[lots])
    tau = calibrate(np.concatenate(Ls), np.concatenate(labs))
    fit = fit_logistic(np.r_[F[base], Fs], np.r_[Y[base], Ys], alpha, starts.get(others[0]))
    rows = {n: {"rows": int((Y[base][:, k] > 0).sum()), "weight": float(Y[base][:, k].sum()),
                "swatches": int(Ys[:, k].sum())} for k, n in enumerate(NAMES)}
    return Model(NAMES, fit, alpha, tau, np.flatnonzero(base), sorted(search), rows,
                 int(sum(unconverged)) + int(not fit.converged))


def lot_evidence(model: Model, patches: pd.DataFrame, patch_lots: pd.DataFrame, n_lots: int, names) -> np.ndarray:
    """(n_lots x len(names)) costs: U[lot, k] = -tau sum_p share_lp (log p_k(p) - log pi_k) over the patches the lot
    lies in, its columns in the order of `names` (each one of the model's districts), less each row's minimum.  A lot
    no patch covers has a zero row."""
    names = [str(n) for n in names]
    unknown = [n for n in names if n not in model.names]
    if unknown:
        raise ValueError(f"the patch model has no district {unknown}; it knows {list(model.names)}")
    llr = model.log_proba(patches) - np.log(model.prior)
    L = pool(llr[:, [model.names.index(n) for n in names]], _spread(patches, patch_lots, n_lots))
    U = -model.tau * L
    return U - U.min(1, keepdims=True)


# --- the run --------------------------------------------------------------------------------------------------------

def confusion(truth: np.ndarray, pred: np.ndarray, names=NAMES) -> dict:
    """{read: {argmin: lots}} over the districts."""
    return {a: {b: int(((truth == i) & (pred == j)).sum()) for j, b in enumerate(names)} for i, a in enumerate(names)}


def quarter_job(job) -> dict:
    """One quarter: its model and every lot's evidence under it, and the confusion of the evidence's argmin against
    the reads of the quarter's own lots (held out)."""
    q, root, solver = job
    data = load_data(root, solver)
    model = fit_quarter(q, data)
    n = len(data.quarter)
    U = lot_evidence(model, data.patches, data.patch_lots, n, NAMES)
    covered = np.bincount(data.patch_lots["lot"].to_numpy(np.int64), minlength=n)[:n] > 0
    qi = paths.QUARTERS.index(q)
    held = (data.quarter == qi) & covered & (data.labels.sum(1) > 0)
    truth, pred = data.labels[held].argmax(1), U[held].argmin(1)
    w = data.labels[held].max(1)
    heldout = {"lots": int(held.sum()), "right": int((truth == pred).sum()),
               "weighted_share": float((w * (truth == pred)).sum() / w.sum()) if held.any() else None,
               "confusion_read_by_argmin": confusion(truth, pred)}
    return {"q": q, "U": U.astype(np.float32), "covered": covered, "tau": model.tau, "alpha": model.alpha,
            "prior": model.prior, "summary": {
                "alpha": model.alpha, "log10_alpha": float(np.log10(model.alpha)), "tau": model.tau,
                "search": [[round(a, 4), loss] for a, loss in model.search],
                "alpha_at_bound": bool(min(abs(np.log10(model.alpha) - b) for b in LOG10_ALPHA) < XATOL),
                "training_patches": int(len(model.trained)), "training": model.rows,
                "unconverged_fits": model.unconverged, "final_fit_iterations": model.fit.iterations,
                "prior": dict(zip(NAMES, map(float, model.prior))),
                "covered_lots": int(covered.sum()), "lots": n, "heldout_reads": heldout}}


def save_npz(f: Path, **arrays) -> None:
    from sfzmaps.solver.textread.store import save_npz as save    # dated members: the same inputs, the same bytes
    f.parent.mkdir(parents=True, exist_ok=True)
    save(f, **arrays)


def run(out: Path = EVIDENCE, workers: int = 4, root: Path = EXPERIMENT, solver: Path = hatching.SOLVER_1921) -> dict:
    """Every quarter's evidence (the quarters side by side, `workers` at most); writes evidence_{q}.npz and
    summary.json into `out`."""
    jobs = [(q, root, solver) for q in paths.QUARTERS]
    if workers <= 1:
        done = [quarter_job(j) for j in jobs]
    else:
        with get_context("spawn").Pool(min(workers, len(jobs))) as p:
            done = p.map(quarter_job, jobs)
    summary = {"names": list(NAMES), "quarters": {}}
    for r in sorted(done, key=lambda r: paths.QUARTERS.index(r["q"])):
        save_npz(out / f"evidence_{r['q']}.npz", U=r["U"], names=np.array(NAMES), tau=np.float64(r["tau"]),
                 covered=r["covered"], alpha=np.float64(r["alpha"]), prior=np.asarray(r["prior"], float))
        summary["quarters"][r["q"]] = r["summary"]
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=1, sort_keys=True) + "\n")
    return summary


# --- the score ------------------------------------------------------------------------------------------------------

def evidence_predictions(folder: Path = EVIDENCE, solver: Path = hatching.SOLVER_1921) -> pd.Series:
    """The evidence alone, no solve: each covered lot's cheapest district under its own quarter's file, by blklot
    (the lot table's; stacked lots given their footprint's code: `answers1921.fill_footprints`).  A lot no patch
    covers has no code."""
    import geopandas as gpd

    from sfzmaps.solver import answers1921 as A
    table = gpd.read_parquet(solver / "lots" / "table.parquet", columns=["blklot", "geometry"])
    with np.load(solver / "lots" / "lots.npz") as z:
        quarter = z["quarter"]
    code = np.full(len(table), None, object)
    for k, q in enumerate(paths.QUARTERS):
        with np.load(folder / f"evidence_{q}.npz") as z:
            names, U, covered = z["names"].astype(str), z["U"], z["covered"]
        m = (quarter == k) & covered
        code[m] = names[U[m].argmin(1)]
    blklot = table["blklot"].astype(str).to_numpy()
    s = pd.Series(code, index=blklot, name="code")
    s = s[~s.index.duplicated()].dropna().astype(str)
    g = table.assign(blklot=blklot)
    g = g[~g["blklot"].duplicated()].set_index("blklot")
    return A.fill_footprints(s, g.geometry)


def score(folder: Path = EVIDENCE, dest: Path = EXPERIMENT / "score_patches.json") -> dict:
    """The evidence's argmin (`patches`) on the owner's answers of the three 1921 pages, beside the published map,
    the final solve and round 2: all answers, each page, and the `1921` page's random and disagree strata."""
    from sfzmaps.solver import answers1921 as A
    answers = A.default_answers()
    preds = {"patches": evidence_predictions(folder), "published": A.published(), "solve-final": A.solve_run("final"),
             "round2": A.method("round2")}
    strata = {c["id"]: c.get("stratum", "") for c in
              json.loads(A.page_inputs("1921")[1][0].read_text())["cases"]}
    answers["stratum"] = [strata.get(c, "") if p == "1921" else "" for c, p in zip(answers["case"], answers["page"])]
    pages = answers.attrs["pages"]
    subsets = {"all": (answers, answers.attrs["dropped"])}
    for p in A.PAGES:
        subsets[f"page {p}"] = (answers[answers["page"] == p], pages.get(p, {}).get("dropped", 0))
    by_stratum = {}                                  # the 1921 page's canttell / unsure answers, by stratum
    for f in sorted(A.page_inputs("1921")[0].glob("*.json")):
        ans = json.loads(f.read_text())
        ans = ans.get("data", ans)
        if ans["answer"] in A.DROPPED:
            s = strata.get(ans["case"], "")
            by_stratum[s] = by_stratum.get(s, 0) + 1
    for s in ("random", "disagree"):
        subsets[f"page 1921 {s}"] = (answers[(answers["page"] == "1921") & (answers["stratum"] == s)],
                                     by_stratum.get(s, 0))
    res = {}
    for name, (sub, n_dropped) in subsets.items():
        a = sub.copy()
        a.attrs = {"dropped": n_dropped}
        res[name] = A.compare(preds, a)
        print(f"{name}: {len(a)} answers, {a['block'].nunique()} blocks")
        for m in preds:
            print("  " + A._line(m, res[name][m]))
    out = {"answers": {"kept": len(answers), "dropped": answers.attrs["dropped"],
                       "blocks": int(answers["block"].nunique()), "by_page": answers.attrs["pages"]},
           "methods": list(preds), "subsets": res}
    A.write_json(dest, out)
    return out


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--series", choices=[SERIES], default=SERIES)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", type=Path, default=EVIDENCE)
    ap.add_argument("--score", action="store_true", help="score the evidence's argmin: score_patches.json")
    a = ap.parse_args(argv)
    if a.score:
        score(a.out)
        return
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ.setdefault(v, "1")                 # the workers start with one maths thread each
    summary = run(a.out, a.workers)
    for q, s in summary["quarters"].items():
        h = s["heldout_reads"]
        print(f"{q}: alpha {s['alpha']:.3g}, tau {s['tau']:.4f}, {s['training_patches']} training patches; held-out "
              f"reads {h['right']}/{h['lots']} right at the argmin", flush=True)


if __name__ == "__main__":
    main()
