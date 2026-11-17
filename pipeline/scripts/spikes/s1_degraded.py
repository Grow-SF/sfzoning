"""Spike S1: the 2000 solver's six measures as the current readings are degraded at known misread and miss rates.

A measurement, not shipped code (brief: .superpowers/sdd/2026-11-16-search-move-and-reader-spikes/task-S01-brief.md;
the degrading and the solves: reader_spike.py).

    uv run python scripts/spikes/s1_degraded.py solve     # every point of the grid, four quarters each
    uv run python scripts/spikes/s1_degraded.py report    # results.json, results.md and one figure per measure
(Run both with the pinned copy of the package first on PYTHONPATH and SFZMAPS_DATA_DIR / SFZMAPS_INPUTS_DIR set, as
`reader_spike.pinned_environment` gives them: another agent may be editing src/ meanwhile.)

Start: the stage's readings (readings_{q}.npz).  At a point (m, r, seed), of the 1,881 labels the energy charges:
  - miss: a label (drawn with chance m) loses every reading;
  - misread: a label not missed (drawn with chance r) has its top reading name another code, its trust kept.  The
    code comes from the saved readers' disagreements (`reader_spike.confusion`); codes no reader confused fall back
    to the codes printed in the label's quarter (counted per run).
  Per seed every label has fixed uniforms, the same in all four quarters and at every rate (`reader_spike.draws`).
The grid (BASE), each at seeds 0, 1 and 2: m in 5, 10, 20, 40% at r = 0; r in 2, 5, 10, 20% at m = 0; (m 10, r 5)
and (m 20, r 10); and S0's own rates on the answered labels (s0_reader_only.reader_on_answers), as measured and as
their excess over the stage's readings' own rates there.  The point (0, 0) is final itself (the control reproduces
it exactly: spike-s0/results.json; and the pinned control here).

The measures: every lot scored (`report.six_measures`), and the companion "wrong lots outside final's largest
incident" (`reader_spike.final_largest_incident`).  Per run: each against final with the paired block bootstrap
(`reader_spike.paired`).  Per grid point: the mean of its three seeds against final, with a two-level bootstrap:
each draw resamples the blocks (the same draw for every run and final) and, per grid point, three of its seeds with
replacement, so the interval holds both the map's sampling and which labels the seed degraded.  The cost per point
of misread (miss): the least-squares slope through the origin of the seed-mean change on the realised misread (miss)
share in percent, over that series' grid points; its interval from the same two-level draws."""
import os

for _k, _v in {"OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "VECLIB_MAXIMUM_THREADS": "1",
               "MKL_NUM_THREADS": "1", "OPENCV_FOR_THREADS_NUM": "1", "NUMEXPR_NUM_THREADS": "1"}.items():
    os.environ.setdefault(_k, _v)

import json  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import reader_spike as R  # noqa: E402

from sfzmaps.solver import report as report_mod  # noqa: E402
from sfzmaps.solver import runs  # noqa: E402

OUT = R.S1_DIR
KEY = ("lines_drawn", "drawn_are_table_lines", "labels_right", "holes", "incidents", "wrong_lots")
COMPANION = "wrong_lots_outside_1"
MEAS = ("wrong_lots", COMPANION, "incidents", "labels_right")     # the measures with bootstrap intervals
BASE = ([(m, 0) for m in (5, 10, 20, 40)] + [(0, r) for r in (2, 5, 10, 20)] + [(10, 5), (20, 10)])
SEEDS = (0, 1, 2)
LOW = 10.0          # the low-range slope uses the grid points at or below this share (%)
PICK_SEED = R.BOOT_SEED + 1     # the two-level bootstrap's seed picks


def s0_rates() -> dict:
    """S0's measured miss and misread rates on the answered labels the solver charges, and the stage's readings'
    own (s0_reader_only.reader_on_answers)."""
    import s0_reader_only as S0
    return S0.reader_on_answers(R.posteriors()[0])["rates_on_charged_answered"]


def s0_points(rates: dict) -> dict:
    """The two S0 grid points, rates in percent to 0.1; r set so the misread share among all labels, r (1 - m), is
    S0's."""
    def point(miss, misread):
        return round(100 * miss, 1), round(100 * misread / (1 - miss), 1)
    return {"s0-measured": point(rates["reader_miss"], rates["reader_misread"]),
            "s0-excess": point(max(rates["reader_miss"] - rates["stage_miss"], 0.0),
                               max(rates["reader_misread"] - rates["stage_misread"], 0.0))}


def base_name(m, r) -> str:
    return f"m{m:g}-r{r:g}"


def bases(rates: dict | None = None) -> dict:
    """grid point name -> (m, r) in percent: BASE and the two S0 points."""
    out = {base_name(m, r): (m, r) for m, r in BASE}
    for tag, (m, r) in s0_points(s0_rates() if rates is None else rates).items():
        out[tag] = (m, r)
    return out


def points(rates: dict | None = None) -> dict:
    """run name -> (m, r, seed): every grid point at every seed."""
    return {R.s1_name(m, r, s): (m, r, s) for m, r in bases(rates).values() for s in SEEDS}


def realised(run: str) -> dict:
    """The run's realised shares of the charged labels (the same in every quarter: checked)."""
    _, folder = R.kind_of(run)
    infos = [json.loads(R.run_file(run, f"-{q}", folder).with_suffix(".json").read_text())["info"]
             for q in R.QUARTERS]
    if any({k: v for k, v in i.items() if k != "fallback"} != {k: v for k, v in infos[0].items() if k != "fallback"}
           for i in infos):
        raise ValueError(f"{run}: the quarters missed or misread different labels")
    i = infos[0]
    return {"miss": i["missed"] / i["labels"], "misread": i["misread"] / i["labels"], "labels": i["labels"],
            "fallback_by_quarter": [x["fallback"] for x in infos]}


def simulated_confidence() -> dict:
    """How sure a simulated misread is: a misread keeps its top reading's trust, and the labels are drawn alike, so
    its sureness is that of the charged labels' top readings: the chance the energy gives them (TOP_READING_FIT of
    the trust, from the fit that held each label's lot's quarter out)."""
    import math
    import s0_reader_only as S0
    a, b = R.E.TOP_READING_FIT
    qs = []
    for _, trust in S0.stage_tops().values():
        t = min(max(trust, 1e-3), 1 - 1e-3)
        qs.append(1 / (1 + math.exp(-(a + b * math.log(t / (1 - t))))))
    qs = np.array(qs)
    return {"labels": len(qs), "mean": float(qs.mean()), "at_least_0.9": float((qs >= 0.9).mean())}


def gaps(run: str) -> dict:
    """Per quarter: the run's own energy at final's answer minus at its own (`reader_spike.search_gap`; below 0:
    the run's search stopped at an answer worse than final's map by the run's own energy)."""
    out = {}
    for q in R.QUARTERS:
        f = R.run_file(run, f"-{q}", OUT).with_suffix(".gap.json")
        if not f.is_file():
            raise FileNotFoundError(f"{f}: run `reader_spike.py gap {run}` (pinned)")
        out[q] = json.loads(f.read_text())["final_answer_minus_own"]
    return out


def slope(x: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Least squares through the origin of each row of Y on x (Y: draws x points)."""
    return (Y @ x) / float(x @ x)


def build() -> dict:
    rates = s0_rates()
    G = bases(rates)
    B = R.Blocks()
    W = B.weights()
    s_final = B.stats(R.codes_of(runs.FINAL_RUN))
    b_final = R.boot_measures(W, s_final)
    final = R.six(runs.FINAL_RUN) | {COMPANION: R.wrong_outside_1(runs.FINAL_RUN)}
    s0 = R.six("s0") | {COMPANION: R.wrong_outside_1("s0")}
    res = {"rates_on_answered": rates, "grid": {k: list(v) for k, v in G.items()},
           "final": {k: final[k] for k in KEY + (COMPANION,)}, "s0_run": {k: s0[k] for k in KEY + (COMPANION,)},
           "bootstrap": {"draws": R.BOOT_DRAWS, "seed": R.BOOT_SEED, "blocks": B.nb, "seed_picks": PICK_SEED},
           "runs": {}}
    hashes = json.loads(R.run_file("control", "-nw", R.S0_DIR).with_suffix(".json").read_text())["solver_sha256"]
    cp = [json.loads(R.run_file("control-pinned", f"-{q}", R.S0_DIR).with_suffix(".json").read_text())
          for q in R.QUARTERS]
    res["control_pinned"] = {
        "join_equal_final": bool(np.array_equal(R.codes_of("control-pinned"), R.codes_of(runs.FINAL_RUN))),
        "measures_equal_final": R.six("control-pinned") == R.six(runs.FINAL_RUN),
        "loaded_from": sorted({c["solver_loaded_from"] for c in cp}),
        "solver_sha256_equal_control": all(c["solver_sha256"] == hashes for c in cp)}
    if not (res["control_pinned"]["join_equal_final"] and res["control_pinned"]["measures_equal_final"]
            and res["control_pinned"]["solver_sha256_equal_control"]):
        raise SystemExit(f"the pinned control does not reproduce final: {res['control_pinned']}")
    boots = {}
    for name, (m, r, seed) in points(rates).items():
        meas = R.six(name) | {COMPANION: R.wrong_outside_1(name)}
        s = B.stats(R.codes_of(name))
        logs = [json.loads(R.run_file(name, f"-{q}", OUT).with_suffix(".json").read_text()) for q in R.QUARTERS]
        h = [lg["solver_sha256"] for lg in logs] + [lg["hashes_after"] for lg in logs]
        res["runs"][name] = {"m": m, "r": r, "seed": seed, "realised": realised(name),
                             "measures": {k: meas[k] for k in KEY + (COMPANION,)},
                             "vs_final": R.paired(W, s, s_final),
                             "same_solver_files": all(x == hashes for x in h),
                             "loaded_from_pinned": all(lg.get("solver_loaded_from", "").startswith(
                                 str(R.PINNED_SRC.resolve())) for lg in logs),
                             "search_gap": gaps(name)}
        boots[name] = R.boot_measures(W, s)
    # per grid point: the seed mean, and the two-level bootstrap of its change against final
    picks = np.random.default_rng(PICK_SEED).integers(0, len(SEEDS), (R.BOOT_DRAWS, len(G), len(SEEDS)))
    draws = {}
    res["grid_points"] = {}
    for gi, (g, (m, r)) in enumerate(G.items()):
        names = [R.s1_name(m, r, s) for s in SEEDS]
        entry = {"m": m, "r": r, "runs": names,
                 "realised": {k: float(np.mean([res["runs"][n]["realised"][k] for n in names]))
                              for k in ("miss", "misread")},
                 "mean": {k: float(np.mean([res["runs"][n]["measures"][k] for n in names]))
                          for k in KEY + (COMPANION,)},
                 "seeds": {k: [res["runs"][n]["measures"][k] for n in names] for k in KEY + (COMPANION,)},
                 "search_missed": sum(v < 0 for n in names for v in res["runs"][n]["search_gap"].values())}
        draws[g] = {}
        for k in MEAS:
            D = np.stack([boots[n][k] - b_final[k] for n in names], 1)          # draws x seeds
            two = np.take_along_axis(D, picks[:, gi, :], 1).mean(1)
            draws[g][k] = two
            entry.setdefault("change", {})[k] = {"mean": entry["mean"][k] - final[k],
                                                 "interval": R.interval(two),
                                                 "block_only_interval": R.interval(D.mean(1))}
        res["grid_points"][g] = entry
    # the costs per point of reader error
    costs = {}
    for series, on in (("misread", lambda p: p[0] == 0 and p[1] > 0), ("miss", lambda p: p[1] == 0 and p[0] > 0)):
        gs = [g for g, p in G.items() if on(p) and g in {base_name(*b) for b in BASE}]
        x = np.array([100 * res["grid_points"][g]["realised"][series] for g in gs])
        low = x <= LOW + 1e-9
        for k in MEAS:
            y = np.array([res["grid_points"][g]["change"][k]["mean"] for g in gs])
            Y = np.stack([draws[g][k] for g in gs], 1)
            costs.setdefault(series, {})[k] = {
                "grid_points": gs, "x": x.tolist(), "slope": float(slope(x, y[None])[0]),
                "interval": R.interval(slope(x, Y)), "slope_low": float(slope(x[low], y[low][None])[0]),
                "interval_low": R.interval(slope(x[low], Y[:, low])), "low_points": int(low.sum())}
    res["cost_per_point"] = costs
    # both at once over every grid point of BASE: change = N x misread% + M x miss%
    gs = [base_name(*b) for b in BASE]
    X = np.array([[100 * res["grid_points"][g]["realised"]["misread"], 100 * res["grid_points"][g]["realised"]["miss"]]
                  for g in gs])
    res["joint_fit"] = {}
    for k in MEAS:
        y = np.array([res["grid_points"][g]["change"][k]["mean"] for g in gs])
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        cb = np.linalg.lstsq(X, np.stack([draws[g][k] for g in gs], 1).T, rcond=None)[0]
        res["joint_fit"][k] = {"per_misread_point": float(coef[0]), "per_miss_point": float(coef[1]),
                               "interval_misread": R.interval(cb[0]), "interval_miss": R.interval(cb[1]),
                               "residual_rms": float(np.sqrt(np.mean((X @ coef - y) ** 2)))}
    res["simulated_misread_confidence"] = simulated_confidence()
    res["model"] = {"codes_with_confusions": len(R.confusion()),
                    "misreads": sum(round(p["realised"]["misread"] * p["realised"]["labels"])
                                    for p in res["runs"].values()),
                    "fallback_draws_all_quarters": sum(sum(p["realised"]["fallback_by_quarter"])
                                                       for p in res["runs"].values())}
    figures(res)
    return res


# --- figures ---------------------------------------------------------------------------------------------------------

INK, INK2, GRIDC, SURF = (11, 11, 11), (82, 81, 78), (226, 225, 220), (252, 252, 251)
BLUE, ORANGE, AQUA = (42, 120, 214), (235, 104, 52), (27, 175, 122)
TITLES = {"lines_drawn": "Table district lines the map draws", "drawn_are_table_lines": "Drawn lines that are table "
          "lines", "labels_right": "Zoning labels whose district comes out right", "holes": "Holes",
          "incidents": "Wrong-lot incidents", "wrong_lots": "Wrong lots",
          COMPANION: "Wrong lots outside final's largest incident"}
FIGURES = KEY + (COMPANION,)


def font(size):
    try:
        return ImageFont.load_default(size)
    except TypeError:
        return ImageFont.load_default()


def ticks(lo: float, hi: float, n: int = 6) -> np.ndarray:
    """Round tick values (1, 2 or 5 times a power of ten apart) inside [lo, hi], about n of them."""
    raw = (hi - lo) / n
    p = 10 ** np.floor(np.log10(raw))
    step = min((m * p for m in (1, 2, 5, 10) if m * p >= raw), default=10 * p)
    return np.arange(np.ceil(lo / step) * step, hi + 1e-12, step)


def figures(res: dict) -> None:
    """One PNG per measure (`s1_<measure>.png`): the measure against the share of labels degraded (realised, %):
    misread only (blue circles) and miss only (orange squares), every seed small and their mean large, the line
    through the means from final at 0; both at once (aqua diamonds, at the sum of the two shares, labelled); the real
    S0 solve (filled star) and the S1 points at S0's measured and excess rates (hollow stars, seed means), S0's at the
    sum of its measured miss and misread shares.  Bars: final's value + the two-level bootstrap interval of the
    seed mean's change (MEAS only)."""
    final = res["final"]
    G = res["grid_points"]
    runs_ = res["runs"]
    s0x = 100 * (res["rates_on_answered"]["reader_miss"] + res["rates_on_answered"]["reader_misread"])
    for k in FIGURES:
        Wd, Hd = 1300, 730
        im = Image.new("RGB", (Wd, Hd), SURF)
        d = ImageDraw.Draw(im)
        x0, y0, x1, y1 = 120, 90, 1160, 600
        pct = k in ("lines_drawn", "drawn_are_table_lines", "labels_right")
        allv = [final[k], res["s0_run"][k]] + [p["measures"][k] for p in runs_.values()]
        bars = {}
        for g, e in G.items():
            if k in MEAS:
                lo, hi = e["change"][k]["interval"]
                bars[g] = (final[k] + lo, final[k] + hi)
                allv += list(bars[g])
        lo_v, hi_v = min(allv), max(allv)
        pad = (hi_v - lo_v) * 0.08 or 1
        lo_v, hi_v = lo_v - pad, hi_v + pad
        xmax = 45.0

        def X(v):
            return x0 + (x1 - x0) * v / xmax

        def Y(v):
            return y1 - (y1 - y0) * (v - lo_v) / (hi_v - lo_v)

        for t in ticks(lo_v, hi_v):
            d.line([(x0, Y(t)), (x1, Y(t))], fill=GRIDC, width=1)
            d.text((x0 - 8, Y(t)), f"{t:.1%}" if pct else f"{t:,.0f}", fill=INK2, font=font(15), anchor="rm")
        for t in range(0, 46, 5):
            d.line([(X(t), y1), (X(t), y1 + 5)], fill=INK2, width=1)
            d.text((X(t), y1 + 10), f"{t}%", fill=INK2, font=font(15), anchor="mt")
        d.line([(x0, y1), (x1, y1)], fill=INK2, width=1)
        d.text((x0, 20), TITLES[k] + " as the readings are degraded", fill=INK, font=font(22))
        d.text((x0, 50), "every lot scored; x: share of the 1,881 charged labels degraded (realised); small marks: each "
               "seed, large: the mean of 3" + ("; bars: two-level bootstrap 95% interval of the mean" if k in MEAS
                                              else ""), fill=INK2, font=font(15))
        d.text(((x0 + x1) / 2, y1 + 34), "labels missed or misread (%)", fill=INK2, font=font(16), anchor="mt")

        def bar(g, x):
            if g in bars:
                d.line([(x, Y(bars[g][0])), (x, Y(bars[g][1]))], fill=INK2, width=1)

        def mark(shape, cx, cy, col, size):
            if shape == "o":
                d.ellipse([cx - size, cy - size, cx + size, cy + size], fill=col, outline=SURF, width=2)
            elif shape == "s":
                d.rectangle([cx - size, cy - size, cx + size, cy + size], fill=col, outline=SURF, width=2)
            else:
                s = size + 2
                d.polygon([(cx, cy - s), (cx + s, cy), (cx, cy + s), (cx - s, cy)], fill=col, outline=SURF)

        def star(cx, cy, hollow, size=10):
            ang = np.linspace(-np.pi / 2, 1.5 * np.pi, 11)[:-1]
            rad = np.where(np.arange(10) % 2 == 0, size, size * 0.45)
            d.polygon([(cx + r * np.cos(a), cy + r * np.sin(a)) for a, r in zip(ang, rad)],
                      fill=None if hollow else INK, outline=INK, width=2)

        for label, col, shape, on in (("misread", BLUE, "o", lambda e: e["m"] == 0 and e["r"] > 0),
                                      ("miss", ORANGE, "s", lambda e: e["r"] == 0 and e["m"] > 0)):
            gs = sorted((g for g, e in G.items() if on(e) and not g.startswith("s0")),
                        key=lambda g: G[g]["realised"][label])
            line = [(0.0, final[k])] + [(100 * G[g]["realised"][label], G[g]["mean"][k]) for g in gs]
            d.line([(X(x), Y(v)) for x, v in line], fill=col, width=2)
            for g in gs:
                x = 100 * G[g]["realised"][label]
                bar(g, X(x))
                for n in G[g]["runs"]:
                    mark(shape, X(100 * runs_[n]["realised"][label]), Y(runs_[n]["measures"][k]), col, 3)
                mark(shape, X(x), Y(G[g]["mean"][k]), col, 6)
            d.text((X(line[-1][0]) + 10, Y(line[-1][1])), label, fill=INK, font=font(16), anchor="lm")
        for g, e in G.items():
            if e["m"] > 0 and e["r"] > 0 and not g.startswith("s0"):
                x = 100 * (e["realised"]["miss"] + e["realised"]["misread"])
                bar(g, X(x))
                for n in e["runs"]:
                    mark("d", X(100 * (runs_[n]["realised"]["miss"] + runs_[n]["realised"]["misread"])),
                         Y(runs_[n]["measures"][k]), AQUA, 2)
                mark("d", X(x), Y(e["mean"][k]), AQUA, 6)
                d.text((X(x) + 12, Y(e["mean"][k]) + 2), f"miss {e['m']}% + misread {e['r']}%", fill=INK,
                       font=font(13), anchor="lm")
        for g, txt, dy in (("s0-measured", "S1 at S0's rates", -16), ("s0-excess", "S1 at S0's excess rates", 16)):
            e = G[g]
            x = 100 * (e["realised"]["miss"] + e["realised"]["misread"])
            bar(g, X(x))
            star(X(x), Y(e["mean"][k]), True)
            d.text((X(x) - 14, Y(e["mean"][k]) + dy), txt, fill=INK, font=font(13), anchor="rm")
        star(X(s0x), Y(res["s0_run"][k]), False)
        d.text((X(s0x) + 16, Y(res["s0_run"][k])), "S0: the template reader alone", fill=INK, font=font(14),
               anchor="lm")
        d.ellipse([X(0) - 5, Y(final[k]) - 5, X(0) + 5, Y(final[k]) + 5], fill=INK, outline=SURF, width=2)
        d.text((X(0) + 9, Y(final[k]) - 12), "final", fill=INK, font=font(14), anchor="lm")
        ly = 690
        for i, (txt, col, shape) in enumerate((("misread only", BLUE, "o"), ("miss only", ORANGE, "s"),
                                               ("both (x = their sum)", AQUA, "d"))):
            lx = x0 + 200 * i
            mark(shape, lx + 5, ly, col, 6)
            d.text((lx + 18, ly), txt, fill=INK, font=font(15), anchor="lm")
        star(x0 + 650, ly, False, 8)
        d.text((x0 + 664, ly), "S0", fill=INK, font=font(15), anchor="lm")
        star(x0 + 720, ly, True, 8)
        d.text((x0 + 734, ly), "S1 at S0's rates (seed mean)", fill=INK, font=font(15), anchor="lm")
        im.save(OUT / f"s1_{k}.png")


# --- the report ------------------------------------------------------------------------------------------------------

def fmt(k, v):
    return f"{v:.2%}" if k in ("lines_drawn", "drawn_are_table_lines", "labels_right") else f"{v:,.0f}"


def fmt_change(k, c) -> str:
    if k == "labels_right":
        return (f"{c['mean'] * 100:+.2f} pt [{c['interval'][0] * 100:+.2f}, {c['interval'][1] * 100:+.2f}]")
    return f"{c['mean']:+,.0f} [{c['interval'][0]:+,.0f}, {c['interval'][1]:+,.0f}]"


def markdown(res: dict) -> str:
    f = res["final"]
    rates = res["rates_on_answered"]
    G = res["grid_points"]
    cp = res["control_pinned"]
    out = ["# Spike S1: the six measures as the readings are degraded", "",
           "The stage's readings degraded at known rates and solved with the stage's walls, placement and search "
           "(see the scripts' docstrings). Every lot scored; final is the point (0, 0). Misreads draw the wrong code "
           f"from the saved readers' disagreements ({res['model']['codes_with_confusions']} codes have some); codes "
           "no reader confused fall back to the codes printed in the label's quarter "
           f"({res['model']['fallback_draws_all_quarters']} of {4 * res['model']['misreads']:,} misread draws over "
           "every run and quarter). A misread keeps its trust, so it is as sure as the labels' top readings are: "
           f"the energy gives them {res['simulated_misread_confidence']['mean']:.2f} on average, "
           f"{res['simulated_misread_confidence']['at_least_0.9']:.0%} at 0.9 or more (S0's reader misreads at 0.46 "
           "on average, the stage's readings at 0.50: spike-s0/results.md).", "",
           f"Every S1 solve ran on a pinned copy of the package at commit {R.PINNED_COMMIT} (another agent was editing "
           "src/sfzmaps/solver/ during the runs). The control solved the same way (`control-pinned`) gives final lot "
           f"for lot: {cp['join_equal_final']}; its six measures equal final's: {cp['measures_equal_final']}; its "
           f"solver files hash as the control's: {cp['solver_sha256_equal_control']}. Every run's solver files hash "
           f"as the control's: {all(p['same_solver_files'] for p in res['runs'].values())}; every run loaded the "
           f"pinned copy: {all(p['loaded_from_pinned'] for p in res['runs'].values())}.", "",
           f"S0's measured rates on the {rates['labels']} answered labels the solver charges: miss "
           f"{rates['reader_miss']:.1%}, misread {rates['reader_misread']:.1%} (the stage's readings: miss "
           f"{rates['stage_miss']:.1%}, misread {rates['stage_misread']:.1%}). The grid points `s0-measured` "
           f"(m {G['s0-measured']['m']}%, r {G['s0-measured']['r']}%) and `s0-excess` (m {G['s0-excess']['m']}%, "
           f"r {G['s0-excess']['r']}%) put those rates, and their excess over the stage's, on the curve. The S1 "
           "rates come on top of the stage's readings' own errors.", "",
           "The companion \"wrong lots outside final's #1\" leaves out the 1,838 lots of final's largest incident, a "
           "fixed lot set from final (see spike-s0/results.md).", "",
           "## The grid: seed means (3 seeds each)", "",
           "| point | miss | misread | " + " | ".join(report_mod.MEASURES[k] for k in KEY)
           + " | outside final's #1 |", "|---|---|---|" + "---|" * (len(KEY) + 1),
           "| final | 0 | 0 | " + " | ".join(fmt(k, f[k]) for k in KEY + (COMPANION,)) + " |",
           "| S0 (the template reader) | – | – | " + " | ".join(fmt(k, res["s0_run"][k]) for k in KEY + (COMPANION,))
           + " |"]
    for g, e in G.items():
        out.append(f"| `{g}` | {e['realised']['miss']:.1%} | {e['realised']['misread']:.1%} | "
                   + " | ".join(fmt(k, e["mean"][k]) for k in KEY + (COMPANION,)) + " |")
    out += ["", "The change of each seed mean against final, with the two-level bootstrap's 95% interval (blocks and "
            "seeds resampled):", "",
            "| point | wrong lots | outside final's #1 | incidents | labels right |", "|---|---|---|---|---|"]
    for g, e in G.items():
        out.append(f"| `{g}` | " + " | ".join(fmt_change(k, e["change"][k]) for k in MEAS) + " |")
    out += ["", "## The seed spread", "", "| point | wrong lots (seeds 0, 1, 2) | outside final's #1 | incidents | "
            "labels right | quarter solves stopped above final's map (of 12) |", "|---|---|---|---|---|---|"]
    for g, e in G.items():
        sv = e["seeds"]
        out.append(f"| `{g}` | {', '.join(f'{x:,}' for x in sv['wrong_lots'])} | "
                   f"{', '.join(f'{x:,}' for x in sv[COMPANION])} | {', '.join(f'{x:,}' for x in sv['incidents'])} | "
                   f"{', '.join(f'{x:.2%}' for x in sv['labels_right'])} | {e['search_missed']} |")
    neg = [(n, q, v) for n, p in res["runs"].items() for q, v in p["search_gap"].items() if v < 0]
    out += ["", "The last column: quarter solves whose search stopped at an answer that the run's own energy rates "
            "worse than final's map (final's answer minus the run's, by the run's energy, below 0). There the "
            "degraded readings would still have kept final's map, and the search missed it. "
            f"{len(neg)} of {4 * len(res['runs'])} quarter solves" + (
                ", the largest gaps: " + ", ".join(f"`{n}` {q} {v:+.1f}" for n, q, v in sorted(neg, key=lambda t: t[2])[:6])
                if neg else "") + "."]
    c = res["cost_per_point"]
    out += ["", "## The cost per point of reader error", "",
            "Least-squares slope through final (0, 0) of the seed-mean change on the realised share in percent, over "
            "the series' grid points (misread only: r 2, 5, 10, 20%; miss only: m 5, 10, 20, 40%); 95% interval from "
            f"the two-level bootstrap. \"Low\": the points at or below {LOW:g}%.", "",
            "| series | measure | per point (all) | 95% | per point (low) | 95% |", "|---|---|---|---|---|---|"]
    for series in ("misread", "miss"):
        for k in MEAS:
            v = c[series][k]
            sc, unit = (100, " pt") if k == "labels_right" else (1, "")
            out.append(f"| {series} | {k.replace('_', ' ').replace('outside 1', 'outside final’s #1')} | "
                       f"{v['slope'] * sc:+.2f}{unit} | [{v['interval'][0] * sc:+.2f}, {v['interval'][1] * sc:+.2f}] | "
                       f"{v['slope_low'] * sc:+.2f}{unit} | [{v['interval_low'][0] * sc:+.2f}, "
                       f"{v['interval_low'][1] * sc:+.2f}] |")
    out += [""]
    for k in MEAS:
        j = res["joint_fit"][k]
        sc = 100 if k == "labels_right" else 1
        name = k.replace("_", " ").replace("outside 1", "outside final’s #1")
        out.append(f"- Both at once over the ten grid points ({name}: change = N × misread% + M × "
                   f"miss%): N = {j['per_misread_point'] * sc:+.2f} [{j['interval_misread'][0] * sc:+.2f}, "
                   f"{j['interval_misread'][1] * sc:+.2f}], M = {j['per_miss_point'] * sc:+.2f} "
                   f"[{j['interval_miss'][0] * sc:+.2f}, {j['interval_miss'][1] * sc:+.2f}], residual rms "
                   f"{j['residual_rms'] * sc:.2f}.")
    w, wo, inc, lr = (c["misread"]["wrong_lots"], c["misread"][COMPANION], c["misread"]["incidents"],
                      c["misread"]["labels_right"])
    mw, mwo, minc, mlr = (c["miss"]["wrong_lots"], c["miss"][COMPANION], c["miss"]["incidents"],
                          c["miss"]["labels_right"])
    out += ["", f"**Each point of misread costs about {w['slope']:.0f} wrong lots [{w['interval'][0]:.0f}, "
            f"{w['interval'][1]:.0f}]; each point of miss about {mw['slope']:.0f} [{mw['interval'][0]:.0f}, "
            f"{mw['interval'][1]:.0f}].** Outside final's largest incident: {wo['slope']:.0f} "
            f"[{wo['interval'][0]:.0f}, {wo['interval'][1]:.0f}] and {mwo['slope']:.0f} [{mwo['interval'][0]:.0f}, "
            f"{mwo['interval'][1]:.0f}]. In incidents: {inc['slope']:.1f} [{inc['interval'][0]:.1f}, "
            f"{inc['interval'][1]:.1f}] and {minc['slope']:.1f} [{minc['interval'][0]:.1f}, "
            f"{minc['interval'][1]:.1f}]. In labels right: {lr['slope'] * 100:.2f} pt [{lr['interval'][0] * 100:.2f}, "
            f"{lr['interval'][1] * 100:.2f}] and {mlr['slope'] * 100:.2f} pt [{mlr['interval'][0] * 100:.2f}, "
            f"{mlr['interval'][1] * 100:.2f}].", "",
            "## Figures", "",
            "One per measure, to read how fast each falls as the readings get worse and where S0 sits: the measure "
            "against the share of labels degraded; blue circles misread only, orange squares miss only, aqua "
            "diamonds both at once (at the sum of the two shares); small marks each seed, large marks their mean, "
            "the lines through the means from final at 0; the filled star the real S0 solve, the hollow stars the S1 "
            "points at S0's measured and excess rates. Bars: final's value plus the two-level bootstrap interval of "
            "the mean's change (wrong lots, the companion, incidents, labels right).", ""]
    out += [f"- `s1_{k}.png`: {TITLES[k]}." for k in FIGURES]
    out += ["", "## Every run", "", "| run | miss | misread | " + " | ".join(report_mod.MEASURES[k] for k in KEY)
            + " | outside final's #1 | wrong lots − final [block 95%] |", "|---|---|---|" + "---|" * (len(KEY) + 2)]
    for n, p in res["runs"].items():
        v = p["vs_final"]["wrong_lots"]["diff_interval"]
        out.append(f"| `{n}` | {p['realised']['miss']:.1%} | {p['realised']['misread']:.1%} | "
                   + " | ".join(fmt(k, p["measures"][k]) for k in KEY + (COMPANION,))
                   + f" | {p['measures']['wrong_lots'] - f['wrong_lots']:+,} [{v[0]:+,.0f}, {v[1]:+,.0f}] |")
    return "\n".join(out) + "\n"


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "solve":
        R.run_jobs([("control-pinned", q) for q in R.QUARTERS] + [(n, q) for n in points() for q in R.QUARTERS])
    elif cmd == "points":
        print(json.dumps(points(), indent=1))
    elif cmd == "report":
        OUT.mkdir(parents=True, exist_ok=True)
        res = build()
        (OUT / "results.json").write_text(json.dumps(res, indent=1, default=float) + "\n")
        md = markdown(res)
        f = OUT / "results.md"
        findings = ""
        if f.is_file() and "## Findings" in f.read_text():         # the findings are written by hand: kept
            findings = f.read_text()[f.read_text().index("## Findings"):]
        f.write_text(md + ("\n" + findings if findings else ""))
        print(md)
    else:
        raise SystemExit(__doc__)
