"""The 1921 measure: how many of the owner's hand answers a method gets right, with a block bootstrap interval.

The owner answered 1921 lot reads on the code-check pages (`label-checks`: `1921-likely`, `1921-lowconf`, `1921`).  Each
answer file holds the case id, the answer (ok, fix, canttell, unsure, unzoned), the fixed `code`, and the code the page
showed.  Truth (the owner's rulings, 2026-11-18):

  ok         the case's `code`, the code the page showed
  fix        the answer's `code`
  unzoned    2R (the ordinance: white is 2R)
  canttell, unsure   dropped, and counted (`answers.attrs["dropped"]`)

The owner's answers are the score, never training data.  Counting: a method is right on an answer when its code for
the answer's lot equals the truth; an answer none of whose lots (a stacked-condo footprint has several) the method
has a code for is `unmatched`, and is not in `n`.  Intervals resample whole BLOCKS (the answers of one block share
their errors), 2000 resamples, `numpy.random.default_rng(0)`: a fixed seed for byte-identical reports.

    uv run python -m sfzmaps.solver.answers1921 --run published --run solve-final --run round2
    uv run python -m sfzmaps.solver.answers1921 --baselines

$0: no API calls.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import shapely

from sfzmaps import config

SERIES = "1921"
SOLVER_1921 = config.DATA_DIR / "solver" / SERIES       # this series' folders, whatever series the solver is set to
EXPERIMENT = config.DATA_DIR / "solver" / "experiments" / "hatch-regions"
ANSWERS_DIR = config.DATA_DIR / "solver" / "experiments" / "label-checks" / "answers"
SPIKE_CSV = config.DATA_DIR / "solver" / "experiments" / "hatch1921" / "owner_answers_cv_r2.csv"
PAGES = ("1921-likely", "1921-lowconf", "1921")
DROPPED = ("canttell", "unsure")
UNZONED = "2R"
N_BOOT = 2000
SEED = 0


# --- the owner's answers -----------------------------------------------------------------------------------------

def _page_of(path: Path) -> str:
    """`answers-1921-likely` (an answers folder) or `1921-likely.json` / `1921-lowconf-retired.json` (cases)."""
    name = path.name.removesuffix(".json").removeprefix("answers-").removesuffix("-retired")
    return name


def owner_answers(answer_dirs: list[Path], case_files: list[Path]) -> pd.DataFrame:
    """One row per kept answer: blklot, block, truth, page, case (and `blklots`, every lot of the case's footprint).
    A page's answers dir pairs with the case files of the same page (`answers-1921-likely` with `1921-likely.json`);
    case ids repeat across pages.  Case files of one page are read in the order given and the first holding an id
    wins (a later file can hold the cases the page retired).  A missing or empty answers dir gives no rows.
    `attrs["dropped"]` counts the canttell / unsure answers left out."""
    cases: dict[str, dict[str, dict]] = {}
    for cf in case_files:
        page = cases.setdefault(_page_of(Path(cf)), {})
        for c in json.loads(Path(cf).read_text())["cases"]:
            page.setdefault(c["id"], c)
    rows, dropped = [], 0
    for ad in answer_dirs:
        ad = Path(ad)
        page = _page_of(ad)
        if not ad.is_dir():
            continue
        for f in sorted(ad.glob("*.json")):
            a = json.loads(f.read_text())
            a = a.get("data", a)
            if a["answer"] in DROPPED:
                dropped += 1
                continue
            case = cases.get(page, {}).get(a["case"])
            if case is None:
                raise KeyError(f"answer {f} is for case {a['case']}, which no case file of page {page} holds")
            if a["answer"] == "ok":
                truth = case["code"]
            elif a["answer"] == "fix":
                truth = a["code"]
            elif a["answer"] == "unzoned":
                truth = UNZONED
            else:
                raise ValueError(f"{f}: unknown answer {a['answer']!r}")
            lots = [str(b) for b in case.get("blklots") or [case["blklot"]]]
            rows.append({"blklot": str(case["blklot"]), "block": str(case["block"]), "truth": truth, "page": page,
                         "case": a["case"], "blklots": tuple(lots)})
    out = pd.DataFrame(rows, columns=["blklot", "block", "truth", "page", "case", "blklots"])
    out.attrs["dropped"] = dropped
    return out


def default_answers() -> pd.DataFrame:
    """Every answer of the three 1921 pages, with the cases of the pages (and the retired ones the owner answered)."""
    dirs = [ANSWERS_DIR / f"answers-{p}" for p in PAGES]
    cases = EXPERIMENT / "cases"
    files = [cases / f"{p}.json" for p in PAGES] + [cases / "1921-lowconf-retired.json"]
    return owner_answers(dirs, [f for f in files if f.is_file()])


# --- the predictions ---------------------------------------------------------------------------------------------

def predictions_from_run(run_npz: Path, lots_table: Path | None = None) -> pd.Series:
    """A solver run's code per blklot: the run's `names[pred]` in the one spelling, each lot index named by the row of
    `data/solver/1921/lots/table.parquet` (the 1921 solver's, not the 2000 map's; the first row of a blklot that repeats)."""
    from sfzmaps.solver.codes import canon
    R = np.load(run_npz, allow_pickle=True)
    codes = np.array([canon(x) for x in np.asarray(R["names"]).astype(str)])[R["pred"]]
    blklot = pd.read_parquet(lots_table or SOLVER_1921 / "lots" / "table.parquet", columns=["blklot"])["blklot"].astype(str)
    if len(blklot) != len(codes):
        raise ValueError(f"{run_npz} has {len(codes)} lots, the lots table {len(blklot)}")
    s = pd.Series(codes, index=blklot.to_numpy(), name="code")
    return s[~s.index.duplicated()]


def footprint_keys(geoms: np.ndarray) -> np.ndarray:
    """A key per outline: stacked condo lots on one outline share one."""
    wkb = shapely.to_wkb(shapely.normalize(np.asarray(geoms)))
    return np.array([hashlib.sha1(b).hexdigest()[:16] for b in wkb])


def fill_footprints(pred: pd.Series, geoms: pd.Series) -> pd.Series:
    """`pred` with a code for every blklot of `geoms` (blklot -> outline) whose footprint holds a coded one: a lot the
    method has no code for takes that of the lot stacked on its outline (the first by blklot)."""
    g = geoms[~geoms.index.duplicated()]
    key = pd.Series(footprint_keys(g.to_numpy()), index=g.index)
    have = pred[pred.index.isin(key.index)]
    by_key = have.groupby(key.loc[have.index].to_numpy()).first()
    missing = key.index.difference(pred.index)
    fill = pd.Series(key.loc[missing].map(by_key).to_numpy(), index=missing).dropna()
    return pd.concat([pred, fill]).sort_index()


# --- the score ---------------------------------------------------------------------------------------------------

def _lookup(pred: pd.Series, answers: pd.DataFrame) -> pd.Series:
    """The prediction per answer (NaN where the method has none): for the answer's own blklot, else for another lot of
    its footprint."""
    out = []
    for r in answers.itertuples():
        lots = [r.blklot] + [b for b in (getattr(r, "blklots", None) or ()) if b != r.blklot]
        out.append(next((pred[b] for b in lots if b in pred.index), np.nan))
    return pd.Series(out, index=answers.index, dtype=object)


def _blocks(values: np.ndarray, count: np.ndarray, blocks: np.ndarray):
    """Per block, the sum of `values` and of `count`."""
    codes, uniq = pd.factorize(blocks, sort=True)
    v = np.bincount(codes, weights=values, minlength=len(uniq))
    c = np.bincount(codes, weights=count, minlength=len(uniq))
    return v, c


def _boot(v: np.ndarray, c: np.ndarray) -> list[float]:
    """The 95% interval of sum(v) / sum(c) over blocks resampled with replacement."""
    if len(v) == 0 or c.sum() == 0:
        return [float("nan"), float("nan")]
    rng = np.random.default_rng(SEED)
    pick = rng.integers(0, len(v), size=(N_BOOT, len(v)))
    num, den = v[pick].sum(1), c[pick].sum(1)
    share = num[den > 0] / den[den > 0]
    lo, hi = np.percentile(share, [2.5, 97.5])
    return [float(lo), float(hi)]


def score(pred: pd.Series, answers: pd.DataFrame, split_lots: list[str] | None = None) -> dict:
    """n (answers the method has a code for), right, share, ci95 (block bootstrap), unmatched (answers it has none
    for, not in n), dropped (canttell / unsure answers left out before this) and split_lots: the answers' lots in the
    PR #56 split list `split_lots` (sorted; [] where no list is given: PR #56 is not on main), which are scored
    like any other."""
    got = _lookup(pred, answers)
    ok = got.notna().to_numpy()
    hit = (got.to_numpy() == answers["truth"].to_numpy()) & ok
    v, c = _blocks(hit[ok].astype(float), np.ones(ok.sum()), answers["block"].to_numpy()[ok])
    n, right = int(ok.sum()), int(hit.sum())
    split = set(map(str, split_lots or ()))
    listed = sorted({b for r in answers.itertuples() for b in [r.blklot, *(getattr(r, "blklots", None) or ())]
                     if b in split})
    return {"n": n, "right": right, "share": right / n if n else float("nan"), "ci95": _boot(v, c),
            "unmatched": int((~ok).sum()), "dropped": int(answers.attrs.get("dropped", 0)), "split_lots": listed}


def compare(preds: dict[str, pd.Series], answers: pd.DataFrame, split_lots: list[str] | None = None) -> dict:
    """`score` per method, and for every method other than "published" its paired difference against "published" on
    the answers both have a code for: `vs_published` = n, diff (right answers more), share_diff and ci95 of the share
    difference (block bootstrap)."""
    out = {m: score(p, answers, split_lots) for m, p in preds.items()}
    if "published" in preds:
        gp = _lookup(preds["published"], answers)
        for m, p in preds.items():
            if m == "published":
                continue
            g = _lookup(p, answers)
            both = (g.notna() & gp.notna()).to_numpy()
            t = answers["truth"].to_numpy()
            d = ((g.to_numpy() == t).astype(float) - (gp.to_numpy() == t).astype(float))[both]
            v, c = _blocks(d, np.ones(both.sum()), answers["block"].to_numpy()[both])
            n = int(both.sum())
            out[m]["vs_published"] = {"n": n, "diff": int(d.sum()), "share_diff": float(d.sum() / n) if n else float("nan"),
                                      "ci95": _boot(v, c)}
    return out


# --- the baselines and the CLI -----------------------------------------------------------------------------------

def published() -> pd.Series:
    """The published 1921 map's code per blklot, one row per blklot, stacked lots given their footprint's code."""
    import geopandas as gpd
    g = gpd.read_file(config.OUT_DIR / SERIES / "districts" / "parcels.fgb", columns=["blklot", "code"])
    g["blklot"] = g["blklot"].astype(str)
    g = g[~g["blklot"].duplicated()].set_index("blklot")
    coded = g[g["code"].notna() & (g["code"].astype(str).str.strip() != "")]
    return fill_footprints(coded["code"].astype(str), g.geometry)


def solve_run(run: str) -> pd.Series:
    """A solver run's code per blklot, stacked lots given their footprint's code (the table's outlines)."""
    import geopandas as gpd
    pred = predictions_from_run(SOLVER_1921 / "solve" / "runs" / f"{run}.npz")
    lots = gpd.read_parquet(SOLVER_1921 / "lots" / "table.parquet", columns=["blklot", "geometry"])
    lots["blklot"] = lots["blklot"].astype(str)
    lots = lots[~lots["blklot"].duplicated()].set_index("blklot")
    return fill_footprints(pred, lots.geometry)


def spike(column: str) -> pd.Series:
    """A hatching-spike round's per-answer call (`round1`, or `cv_model` for round 2: cross-validated, each answer
    predicted by a model that did not see it) next to its answers."""
    d = pd.read_csv(SPIKE_CSV, dtype=str)
    return d.set_index("blklot")[column]


def method(name: str) -> pd.Series:
    if name == "published":
        return published()
    if name == "round1":
        return spike("round1")
    if name == "round2":
        return spike("cv_model")
    if name.startswith("solve-"):
        return solve_run(name.removeprefix("solve-"))
    raise ValueError(f"unknown method {name!r}: published, solve-<run>, round1 or round2")


def _line(name: str, s: dict) -> str:
    lo, hi = s["ci95"]
    extra = f" vs published {s['vs_published']['diff']:+d}" if "vs_published" in s else ""
    return (f"{name:14s} {s['right']:4d}/{s['n']:<4d} {s['share']:6.1%}  [{lo:.1%}, {hi:.1%}]  "
            f"unmatched {s['unmatched']}{extra}")


def write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, sort_keys=True) + "\n")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run", action="append", default=[], metavar="NAME",
                    help="published, solve-<run>, round1, round2 (repeat); writes score_<name>.json each")
    ap.add_argument("--baselines", action="store_true",
                    help="score published, solve-final, round1 and round2 together: score_baselines.json")
    ap.add_argument("--split-list", type=Path, help="the PR #56 split lots, one blklot per line (none: not on main)")
    a = ap.parse_args(argv)
    names = ["published", "solve-final", "round1", "round2"] if a.baselines else a.run
    if not names:
        ap.error("give --run NAME or --baselines")
    splits = a.split_list.read_text().split() if a.split_list else []
    answers = default_answers()
    print(f"{len(answers)} owner answers kept, {answers.attrs['dropped']} canttell/unsure dropped, "
          f"{answers['block'].nunique()} blocks; " + ("" if splits else "PR #56 not on main: no split list"))
    preds = {n: method(n) for n in names}
    res = compare(preds, answers, splits)
    for n in names:
        print(_line(n, res[n]))
    for n in ([] if a.baselines else names):
        write_json(EXPERIMENT / f"score_{n}.json", {n: res[n]})
    if a.baselines:
        write_json(EXPERIMENT / "score_baselines.json",
                   {"answers": {"kept": len(answers), "dropped": answers.attrs["dropped"],
                                "blocks": int(answers["block"].nunique()),
                                "pages": answers["page"].value_counts().sort_index().to_dict()},
                    "methods": res})


if __name__ == "__main__":
    main()
