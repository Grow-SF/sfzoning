"""Decide between the control and a variant of the text detector's training by the rule written before the runs (the
plan 2026-11-17-text-detector-text-vs-line, Task 7), and whether either is adoptable against the held-out detectors
(r328) and the old quarter model of the quarter.  What it is for: one verdict with every condition's numbers, by one
rule for every comparison.  CPU, $0.

    uv run python scripts/textlines/decide.py audit ne        # the baselines' missed-text audits on one quarter
    uv run python scripts/textlines/decide.py panel ne control-ne-s0 composites-ne-s0 composites-ne-s1 ...
    uv run python scripts/textlines/decide.py regate-extra control-ne-s0 ...   # 1990-96 gated again (new measures)
    uv run python scripts/textlines/decide.py verdict ne --control control --variant composites [--eye-ok yes]
    uv run python scripts/textlines/decide.py verdict sw --control control --variant composites --noise ne ...

The rule (written before the runs; the m2_clean_words guard is the controller's ruling for Task 7):
- A measure's value is the share of its ink called text (chance >= 0.5); an arm's value is the mean over its runs.
  A difference is the variant less the control (an arm less a baseline) on the blocks both scored, with the 95%
  paired bootstrap interval over blocks (`linegate.paired`).
- The noise of a measure in a comparison = max(the largest difference between two of the noise runs (the control's
  seeds), the half-width of the comparison's paired interval).
- Variant over control needs all of: m1a_in_box lower with the interval below 0 (M1a carries the line verdict, since
  M1b shares the training's line rule); m2 not lower by more than its noise; m1c not higher by more than its noise;
  m2_clean_words not lower by more than its noise; eye_ok True (the controller looked at both panels).
- The verdict: "pending-eye" when eye_ok is None (`numbers_say` holds what it would be with eye_ok True); "variant"
  when every condition holds; "mixed" when the variant gains on a line measure (m1a_in_box or m1b_in_box lower, the
  interval below 0) and a condition fails (M1b falls and M1a does not; better lines, less text; the eye says no);
  "control" when it gains on neither.
- Adoptable, for each arm against each baseline (r328 and old-<quarter>; the all-sheets detector is in-sample, a
  reference only): m1a_in_box and m1b_in_box lower, the intervals below 0; m2 not lower and m1c not higher by more
  than the noise; m2_clean_words not lower by more than its noise (the ruling: `as_written` leaves it out); the
  audit's text pieces called text not lower beyond the baseline's binomial interval, i.e. the arm's estimate of the
  text pieces it misses (missed letter-sized pieces x the share of the drawn ones judged plainly text by eye) at most
  the baseline's Clopper-Pearson 95% upper bound.  An audit not yet counted by eye leaves it None (pending).

`verdict ...` writes `runs/verdict-<variant>-<quarter>.{json,md}`; `audit q` writes `baselines/audit/<name>-<q>.{json,
png}` for r328 and old-<q> on that quarter only (the same pool and draw as a run's own audit); `panel` writes
`runs/compare-<q>.png`.  The eye counts of the audits are EYE_AUDIT below.
"""
import os

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS",
           "OPENCV_FOR_THREADS_NUM"):
    os.environ.setdefault(_k, "1")
import argparse  # noqa: E402
import itertools  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
from collections.abc import Mapping  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
from scipy.stats import beta  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))

from sfzmaps.solver import paths, runs  # noqa: E402
from sfzmaps.solver.textnet import linegate  # noqa: E402

TEXTLINES = runs.EXPERIMENTS_DIR / "textlines"
RUNS_DIR = TEXTLINES / "runs"
BASELINES = TEXTLINES / "baselines"
EXTRA = "1990-1996-zoning"

VERDICT = ("m1a_in_box", "m1b_in_box", "m2", "m1c", "m2_clean_words")
LOWER_IS_BETTER = {"m1a_in_box": True, "m1b_in_box": True, "m1c": True, "m2": False, "m2_clean_words": False,
                   "m1a": True, "m1a_in_box_context": True, "m1a_context": True, "m1b_near": True, "m1b_far": True,
                   "overlap_band": False}
LINES = ("m1a_in_box", "m1b_in_box")              # a fall here is a gain
GUARDS = ("m2", "m1c", "m2_clean_words")          # these must not move the wrong way beyond the noise
DIAGNOSTICS = ("m1a", "m1a_in_box_context", "m1a_context", "m1b_near", "m1b_far", "overlap_band")   # in no verdict
EXTRA_MEASURES = ("m1b_in_box", "m1c", "m2", "m2_clean_words")
CI = 0.95

# The audits counted by eye: the audit file (relative to TEXTLINES) -> how many of its drawn pieces are plainly a
# letter or digit (text the detector missed).  Counted by the Task 7 implementer on each contact sheet (64 tiles); a
# piece of digits fused into a line is borderline and not counted (the borderline tiles are listed in the results
# note).
EYE_AUDIT: dict[str, int] = {
    "baselines/audit/r328-ne.json": 0,
    "baselines/audit/old-ne-ne.json": 1,           # tile 42, the T of a large "STR..."
    "runs/control-ne-s0/audit.json": 1,            # tile 22, the 0 of "40"
    "runs/control-ne-s1/audit.json": 0,
    "runs/control-ne-s2/audit.json": 0,
    "runs/composites-ne-s0/audit.json": 0,
    "runs/composites-ne-s1/audit.json": 0,
    "runs/composites-ne-s2/audit.json": 0,
    "baselines/audit/r328-sw.json": 0,
    "baselines/audit/old-sw-sw.json": 3,           # tiles 25, 39, 63: street-name letters and the T of "...CTS"
    "runs/control-sw-s0/audit.json": 3,            # tiles 25, 38, 44: street-name letters
}


def _all(values) -> bool | None:
    """False when any is False, else None when any is None, else True."""
    values = list(values)
    if any(v is False for v in values):
        return False
    if any(v is None for v in values):
        return None
    return True


def value(folder: Path, measure: str, quarter: str | None = None) -> float | None:
    """`measure`'s share called text (chance >= 0.5) in the run or baseline `folder` (its blocks.npz), on `quarter`'s
    blocks only when given; None when it has no px there."""
    with np.load(Path(folder) / linegate.BLOCKS_FILE) as z:
        px, called, q = z[f"{measure}.px"], z[f"{measure}.called_05"], z[f"{measure}.quarter"]
    if quarter is not None:
        keep = q == paths.QUARTERS.index(quarter)
        px, called = px[keep], called[keep]
    return float(called.sum() / px.sum()) if px.sum() > 0 else None


def seed_noise(noise_runs: list[Path], measure: str) -> float:
    """The largest difference in `measure` between two of `noise_runs` (0 for one run)."""
    vals = [value(r, measure) for r in noise_runs]
    return max((abs(a - b) for a, b in itertools.combinations(vals, 2)), default=0.0)


def compare(a: list[Path], b: list[Path], measure: str, noise_runs: list[Path], boot: int, seed: int) -> dict:
    """`linegate.paired` (b less a) with the noise and which way it moved: `falls` / `rises` when the interval lies
    below / above 0, `held` when it did not move the wrong way by more than the noise (lower-is-better measures may
    rise by the noise at most, the others fall by it at most)."""
    p = linegate.paired(a, b, measure, boot, seed)
    half = (p["hi"] - p["lo"]) / 2
    seeds = seed_noise(noise_runs, measure)
    noise = max(seeds, half)
    held = p["diff"] <= noise if LOWER_IS_BETTER[measure] else p["diff"] >= -noise
    return p | {"half_width": half, "seed_noise": seeds, "noise": noise, "falls": p["hi"] < 0, "rises": p["lo"] > 0,
                "held": bool(held)}


def clopper_pearson(k: int, n: int) -> tuple[float, float]:
    lo = 0.0 if k == 0 else float(beta.ppf((1 - CI) / 2, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.ppf(1 - (1 - CI) / 2, k + 1, n - k))
    return lo, hi


def baseline_audit_file(folder: Path, quarter: str) -> Path:
    """Where `audit` writes a baseline's audit on one quarter: <baselines>/audit/<name>-<quarter>.json."""
    folder = Path(folder)
    return folder.parent / "audit" / f"{folder.name}-{quarter}.json"


def audit_estimate(files: list[Path], counts: Mapping) -> dict | None:
    """The missed text pieces of an arm (`files`: each run's audit.json) by its eye counts (`counts`: file -> plainly
    text among its drawn pieces): the mean missed letter-sized pieces x the pooled share judged text, with its
    Clopper-Pearson interval.  None when a file is not counted (or missing)."""
    counts = {Path(k).resolve(): v for k, v in counts.items()}
    got = []
    for f in files:
        f = Path(f).resolve()
        if f not in counts or not f.is_file():
            return None
        a = json.loads(f.read_text())
        got.append((a["letter_sized_pieces"], a["missed_below_0.5"], len(a["items"]), int(counts[f])))
    pieces, missed = float(np.mean([g[0] for g in got])), float(np.mean([g[1] for g in got]))
    n, k = sum(g[2] for g in got), sum(g[3] for g in got)
    lo, hi = clopper_pearson(k, n)
    return {"pieces": pieces, "missed": missed, "drawn": n, "text": k, "share": k / n, "share_ci": [lo, hi],
            "missed_text": missed * k / n, "missed_text_ci": [missed * lo, missed * hi]}


def adoptable(arm: list[Path], baselines: Mapping[str, Path], quarter: str | None, noise_runs: list[Path],
              audit_counts: Mapping, boot: int, seed: int) -> dict:
    """The adoptable rule (the module) for one arm against each baseline; `adoptable` (with the clean-word guard) and
    `as_written` (without it) are True, False or None (pending: only the eye counts of the audit are missing; also
    None when no baseline is given)."""
    out: dict = {}
    mine = audit_estimate([r / "audit.json" for r in arm], audit_counts)
    for name, folder in baselines.items():
        d = {m: compare([folder], arm, m, noise_runs, boot, seed) for m in VERDICT}
        theirs = audit_estimate([baseline_audit_file(folder, quarter)], audit_counts) if quarter else None
        audit_ok = None if mine is None or theirs is None else mine["missed_text"] <= theirs["missed_text_ci"][1]
        cond = {"m1a_in_box falls": d["m1a_in_box"]["falls"], "m1b_in_box falls": d["m1b_in_box"]["falls"],
                "m2 held": d["m2"]["held"], "m1c held": d["m1c"]["held"], "audit held": audit_ok,
                "m2_clean_words held": d["m2_clean_words"]["held"]}
        out[name] = {"diffs": d, "audit": {"arm": mine, "baseline": theirs}, "conditions": cond,
                     "as_written": _all(v for k, v in cond.items() if k != "m2_clean_words held"),
                     "adoptable": _all(cond.values())}
    every = list(out.values())
    out["as_written"] = _all(o["as_written"] for o in every) if every else None
    out["adoptable"] = _all(o["adoptable"] for o in every) if every else None
    return out


def _quarter_of(folder: Path) -> str | None:
    f = Path(folder) / "run.json"
    return json.loads(f.read_text()).get("quarter") if f.is_file() else None


def extra_values(folder: Path, series: str = EXTRA) -> dict | None:
    """A run's headline scores on another map's sheets (`extra/<series>/scores.json`), called >= 0.5; None if none."""
    f = Path(folder) / "extra" / series / linegate.SCORES_FILE
    if not f.is_file():
        return None
    s = json.loads(f.read_text())
    return {m: (s[m]["called_05"] if isinstance(s.get(m), dict) else None) for m in EXTRA_MEASURES} | {
        "px": {m: (s[m]["px"] if isinstance(s.get(m), dict) else None) for m in EXTRA_MEASURES}}


def _numbers_say(d: dict, cond: dict) -> str:
    gains = any(d[m]["falls"] for m in LINES)
    if all(v for k, v in cond.items() if k != "eye_ok"):
        return "variant"
    return "mixed" if gains else "control"


def verdict(control: list[Path], variant: list[Path], baselines: Mapping[str, Path], eye_ok: bool | None, *,
            quarter: str | None = None, noise_runs: list[Path] | None = None, audit_counts: Mapping | None = None,
            boot: int = 1000, seed: int = 0) -> dict:
    """The decision (the module's rule) between the runs `control` and `variant` (run folders of one quarter), and
    each arm's adoptability against `baselines` (name -> folder).  `noise_runs` (default `control`) give the seed
    noise; `audit_counts` (audit file -> plainly text among its drawn pieces) the audits' eye counts.  With no variant
    runs only the control's adoptability is decided (`verdict` None)."""
    control, variant = [Path(p) for p in control], [Path(p) for p in variant]
    noise_runs = [Path(p) for p in (noise_runs or control)]
    quarter = quarter or _quarter_of(control[0])
    audit_counts = audit_counts or {}
    base = {n: Path(f) for n, f in baselines.items()}
    out: dict = {"quarter": quarter, "control": [str(p) for p in control], "variant": [str(p) for p in variant],
                 "baselines": {n: str(f) for n, f in base.items()}, "noise_runs": [str(p) for p in noise_runs],
                 "eye_ok": eye_ok, "boot": boot, "seed": seed,
                 "values": {p.name: {m: value(p, m) for m in (*VERDICT, *DIAGNOSTICS)} for p in [*control, *variant]}
                 | {n: {m: value(f, m, quarter) for m in (*VERDICT, *DIAGNOSTICS)} for n, f in base.items()},
                 "seed_noise": {m: seed_noise(noise_runs, m) for m in (*VERDICT, *DIAGNOSTICS)},
                 "extra": {p.name: extra_values(p) for p in [*control, *variant]}}
    arms = {"control": control} | ({"variant": variant} if variant else {})
    out["adoptable"] = {a: adoptable(r, base, quarter, noise_runs, audit_counts, boot, seed) for a, r in arms.items()}
    if not variant:
        return out | {"verdict": None, "numbers_say": None}
    d = {m: compare(control, variant, m, noise_runs, boot, seed) for m in (*VERDICT, *DIAGNOSTICS)}
    cond = {"m1a_in_box falls": d["m1a_in_box"]["falls"], "m2 held": d["m2"]["held"], "m1c held": d["m1c"]["held"],
            "m2_clean_words held": d["m2_clean_words"]["held"], "eye_ok": eye_ok}
    says = _numbers_say(d, cond)
    if eye_ok is None:
        v = "pending-eye"
    elif says == "variant" and eye_ok is False:
        v = "mixed"
    else:
        v = says
    return out | {"variant_vs_control": d, "conditions": cond, "numbers_say": says, "verdict": v}


# --- the write-up ----------------------------------------------------------------------------------------------------

def _n(v, digits: int = 4) -> str:
    return "-" if v is None else f"{v:.{digits}f}"


def _diff(p: dict) -> str:
    return f"{p['diff']:+.4f} [{p['lo']:+.4f}, {p['hi']:+.4f}]"


def markdown(v: dict) -> str:
    """The verdict as tables: every run's and baseline's values, the paired differences with their intervals and
    noise, the conditions, adoptability, the audits and the 1990-96 numbers."""
    ms = (*VERDICT, *DIAGNOSTICS)
    lines = [(f"## Quarter {v['quarter']}: verdict {v['verdict']} (the numbers say {v['numbers_say']}; eye_ok "
              f"{v['eye_ok']})"), "",
             ("Share of each measure's ink called text (chance >= 0.5). Lower is better for m1a*, m1b*, m1c; higher "
              "for m2, m2_clean_words, overlap_band. Baselines on this quarter's blocks."), "",
             "| run | " + " | ".join(ms) + " |", "|---|" + "---|" * len(ms)]
    for name, vals in v["values"].items():
        lines.append(f"| {name} | " + " | ".join(_n(vals[m]) for m in ms) + " |")
    lines += ["", f"Seed noise (largest difference between two of {', '.join(Path(p).name for p in v['noise_runs'])}): "
              + ", ".join(f"{m} {_n(v['seed_noise'][m])}" for m in ms), ""]
    if v.get("variant_vs_control"):
        lines += ["### Variant less control (mean of each arm's runs; 95% paired interval over blocks)", "",
                  "| measure | diff [95%] | half-width | seed noise | noise | falls | rises | held |",
                  "|---|---|---|---|---|---|---|---|"]
        for m, p in v["variant_vs_control"].items():
            lines.append(f"| {m}{'' if m in VERDICT else ' (diagnostic)'} | {_diff(p)} | {_n(p['half_width'])} | "
                         f"{_n(p['seed_noise'])} | {_n(p['noise'])} | {p['falls']} | {p['rises']} | {p['held']} |")
        lines += ["", "Conditions: " + "; ".join(f"{k}: {c}" for k, c in v["conditions"].items()), ""]
    for arm, a in v["adoptable"].items():
        lines += [(f"### Adoptable: {arm} (with the clean-word guard: {a['adoptable']}; as written: "
                   f"{a['as_written']})"), "",
                  "| baseline | " + " | ".join(VERDICT) + " | audit held | adoptable | as written |",
                  "|---|" + "---|" * (len(VERDICT) + 3)]
        for name, b in a.items():
            if not isinstance(b, dict):
                continue
            cells = [f"{_diff(b['diffs'][m])} noise {_n(b['diffs'][m]['noise'])}" for m in VERDICT]
            lines.append(f"| {name} | " + " | ".join(cells) + f" | {b['conditions']['audit held']} | "
                         f"{b['adoptable']} | {b['as_written']} |")
        lines.append("")
        audits = {n: b["audit"]["baseline"] for n, b in a.items() if isinstance(b, dict)}
        mine = next((b["audit"]["arm"] for b in a.values() if isinstance(b, dict)), None)
        for who, e in {arm: mine, **audits}.items():
            if e is None:
                lines.append(f"- audit {who}: not counted by eye yet")
            else:
                lines.append(
                    f"- audit {who}: {e['pieces']:.0f} letter-sized pieces, {e['missed']:.0f} missed; "
                    f"{e['text']}/{e['drawn']} drawn plainly text ({e['share']:.3f} [{e['share_ci'][0]:.3f}, "
                    f"{e['share_ci'][1]:.3f}]); missed text pieces {e['missed_text']:.1f} "
                    f"[{e['missed_text_ci'][0]:.1f}, {e['missed_text_ci'][1]:.1f}]")
        lines.append("")
    if any(v["extra"].values()):
        lines += [f"### {EXTRA} (this quarter's px; no table, so no M1a)", "",
                  "| run | " + " | ".join(EXTRA_MEASURES) + " |", "|---|" + "---|" * len(EXTRA_MEASURES)]
        for name, e in v["extra"].items():
            if e:
                lines.append(f"| {name} | " + " | ".join(_n(e[m]) for m in EXTRA_MEASURES) + " |")
        lines.append("")
    return "\n".join(lines) + "\n"


# --- the command line ------------------------------------------------------------------------------------------------

def _runs_of(name: str, quarter: str) -> list[Path]:
    found = sorted(p for p in RUNS_DIR.glob(f"{name}-{quarter}-s*") if (p / linegate.BLOCKS_FILE).is_file())
    if not found:
        raise SystemExit(f"no scored runs {name}-{quarter}-s* in {RUNS_DIR}")
    return found


def baseline_sources(quarter: str) -> dict[str, Path]:
    return {"r328": BASELINES / "r328", f"old-{quarter}": BASELINES / f"old-{quarter}"}


def audit_baselines(quarter: str) -> None:
    """r328's and old-<quarter>'s missed-text audits on one quarter: the same pool, draw and picture as a run's own
    audit (`variant.score`), written to baselines/audit/<name>-<quarter>.{json,png}.  One CityGeo: about 7 GB."""
    import audit
    import variant

    from sfzmaps.solver.textnet import data
    sheets = data.load_sheets([paths.SERIES], hold_out_validation=False)
    geo = linegate.CityGeo()
    srcs = {"r328": linegate.folder_source(BASELINES / "r328", "nested"),
            f"old-{quarter}": linegate.folder_source(paths.SOLVER_DIR / f"text-{quarter}" / "chance", "flat")}
    ink_of = lambda sheet: data.page_ink(next(sd for sd in sheets if sd.sheet == sheet))  # noqa: E731
    for name, src in srcs.items():
        cands, total = audit.candidates(sheets, src, (quarter,), geo)
        items = audit.draw(cands, variant.AUDIT_N, 0)
        f = baseline_audit_file(BASELINES / name, quarter)
        audit.sheet(items, f.with_suffix(".png"), ink_of)
        f.write_text(json.dumps({"letter_sized_pieces": total, "missed_below_0.5": len(cands), "items": items},
                                indent=1) + "\n")
        print(name, quarter, "letter-sized", total, "missed", len(cands), "->", f, flush=True)


def regate_extra(names: list[str], series: str = EXTRA) -> None:
    """Gate again the runs `names`' predictions on another map's sheets (their saved `chance/<series>/`), as
    `variant.score` does, so that runs gated before a measure was added (m2_clean_words) carry it too.  The measures
    already there must come out the same (px and share called), or it stops before writing."""
    import variant

    from sfzmaps.solver.textnet import data
    sheets = data.load_sheets([series], hold_out_validation=False)
    for name in names:
        run = RUNS_DIR / name
        quarter = _quarter_of(run)
        out = run / "extra" / series
        old = json.loads((out / linegate.SCORES_FILE).read_text())
        r, t = linegate.gate_sheets(sheets, linegate.folder_source(run / "chance" / series, "flat"), (quarter,),
                                    variant.ExtraGeo())
        for m in ("m1b_in_box", "m1c", "m2", "overlap_band"):
            if (old[m]["px"], old[m]["called_05"]) != (r[m]["px"], r[m]["called_05"]):
                raise SystemExit(f"{name}: {m} differs from the gate written before ({old[m]['px']}, "
                                 f"{old[m]['called_05']} now {r[m]['px']}, {r[m]['called_05']}): not rewritten")
        linegate.write(r, t, out)
        print(name, series, {m: r[m]["called_05"] for m in EXTRA_MEASURES}, flush=True)


def compare_panel(quarter: str, names: list[str]) -> Path:
    """The picture panel with the runs `names` side by side, then r328 and the old quarter model."""
    import panel
    srcs = {n: linegate.folder_source(RUNS_DIR / n / "chance", "flat") for n in names}
    srcs |= {"r328": linegate.folder_source(BASELINES / "r328", "nested"),
             f"old-{quarter}": linegate.folder_source(paths.SOLVER_DIR / f"text-{quarter}" / "chance", "flat")}
    out = RUNS_DIR / f"compare-{quarter}.png"
    return panel.render(srcs, out, f"red = ink called text (chance >= 0.5), grey = other ink; runs are 0 outside "
                                   f"{quarter}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("audit")
    a.add_argument("quarter", choices=paths.QUARTERS)
    p = sub.add_parser("panel")
    p.add_argument("quarter", choices=paths.QUARTERS)
    p.add_argument("runs", nargs="+")
    x = sub.add_parser("regate-extra")
    x.add_argument("runs", nargs="+")
    v = sub.add_parser("verdict")
    v.add_argument("quarter", choices=paths.QUARTERS)
    v.add_argument("--control", default="control")
    v.add_argument("--variant", default=None, help="the variant's run name (none: only the control's adoptability)")
    v.add_argument("--noise", default=None, help="the quarter of the control runs that give the seed noise")
    v.add_argument("--eye-ok", choices=("yes", "no"), default=None)
    args = ap.parse_args()
    if args.cmd == "audit":
        audit_baselines(args.quarter)
    elif args.cmd == "panel":
        print(compare_panel(args.quarter, args.runs))
    elif args.cmd == "regate-extra":
        regate_extra(args.runs)
    else:
        control = _runs_of(args.control, args.quarter)
        variant = _runs_of(args.variant, args.quarter) if args.variant else []
        noise = _runs_of(args.control, args.noise) if args.noise else None
        eye = None if args.eye_ok is None else args.eye_ok == "yes"
        counts = {TEXTLINES / k: n for k, n in EYE_AUDIT.items()}
        r = verdict(control, variant, baseline_sources(args.quarter), eye, quarter=args.quarter, noise_runs=noise,
                    audit_counts=counts)
        stem = RUNS_DIR / f"verdict-{args.variant or args.control}-{args.quarter}"
        stem.with_suffix(".json").write_text(json.dumps(r, indent=1) + "\n")
        stem.with_suffix(".md").write_text(markdown(r))
        print(markdown(r))


if __name__ == "__main__":
    main()
