"""Spike S1: the 2000 solver's six measures as the current readings are degraded at known misread and miss rates.

A measurement, not shipped code (brief: .superpowers/sdd/2026-11-16-search-move-and-reader-spikes/task-S01-brief.md;
the degrading and the solves: reader_spike.py).

    uv run python scripts/spikes/s1_degraded.py solve     # every point of the grid, four quarters each
    uv run python scripts/spikes/s1_degraded.py report    # results.json, results.md and one figure per measure
(Run both with the pinned copy of the package first on PYTHONPATH and SFZMAPS_DATA_DIR / SFZMAPS_INPUTS_DIR set, as
`reader_spike.pinned_environment` gives them: another agent may be editing src/ meanwhile.)

Start: the stage's readings (readings_{q}.npz).  At a point (m, r, seed), of the 1,881 labels the energy charges:
  - miss: a label (drawn with chance m) loses every reading;
  - misread ("sure"): a label not missed (drawn with chance r) has its top reading name another code, its trust
    kept.  The code comes from the saved readers' disagreements (`reader_spike.confusion`); codes no reader confused
    fall back to the codes printed in the label's quarter (counted per run).
  - misread "unsure" (fix round 1, the runs `...-u`): the same labels and codes as the sure run of that point and
    seed, but each misread label's P is the template reader's own: the wrong code at the mass the reader gave a wrong
    top, the replaced code at the mass it gave the answer, one pair drawn from its 65 misreads on the answered labels
    (`reader_spike.unsure_problem`); every other label keeps the function's P.
  Per seed every label has fixed uniforms, the same in all four quarters and at every rate (`reader_spike.draws`).
The grid (BASE), each at seeds 0, 1 and 2: m in 5, 10, 20, 40% at r = 0; r in 2, 5, 10, 20% at m = 0; (m 10, r 5)
and (m 20, r 10); S0's own rates on the answered labels (s0_reader_only.reader_on_answers), as measured and as their
excess over the stage's readings' own rates there; and the unsure misreads at r 5, 10, 20%.  The point (0, 0) is
final itself (the control reproduces it exactly: spike-s0/results.json; and the pinned control here).

The measures: every lot scored (`report.six_measures`), and the companion "wrong lots outside final's largest
incident" (`reader_spike.final_largest_incident`; chosen after seeing S0, one-sided).  The intervals (fix round 1):
  - wrong lots (and the companion): per grid point, the seed mean's t interval over its 3 seeds (df 2), which holds
    the search's run-to-run noise and which labels were degraded; per run, the changed sets' bootstrap
    (`reader_spike.cluster_interval`: one search outcome, one unit).  The blocks' bootstrap is kept as "lot
    sampling only".
  - incidents and labels right: per grid point, the two-level bootstrap: blocks resampled (one draw for every run)
    and the 3 seeds resampled with replacement, the same seed picks at every grid point (the seeds are nested across
    rates).
  - the cost per point: per rate, change / realised share for each seed, and its t interval over the seeds; for a
    series, each rate weighted alike (the mean of the per-rate ratios, per seed), with its t interval over the
    seeds (paired: seed s is nested across the rates)."""
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
MEAS = ("wrong_lots", COMPANION, "incidents", "labels_right")     # the measures with intervals
LOTS = ("wrong_lots", COMPANION)                                   # t intervals over the seeds
BASE = ([(m, 0) for m in (5, 10, 20, 40)] + [(0, r) for r in (2, 5, 10, 20)] + [(10, 5), (20, 10)])
UNSURE_RATES = (5, 10, 20)
SEEDS = (0, 1, 2)
LOW = 10.0          # the low range: grid points whose nominal rate is at most this (%)
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
    """grid point name -> (m, r, unsure) in percent: BASE, the two S0 points and the unsure misreads."""
    out = {base_name(m, r): (m, r, False) for m, r in BASE}
    for tag, (m, r) in s0_points(s0_rates() if rates is None else rates).items():
        out[tag] = (m, r, False)
    for r in UNSURE_RATES:
        out[base_name(0, r) + R.UNSURE_SUFFIX] = (0, r, True)
    return out


def run_name(m, r, seed, unsure=False) -> str:
    return R.s1_name(m, r, seed) + (R.UNSURE_SUFFIX if unsure else "")


def points(rates: dict | None = None) -> dict:
    """run name -> (m, r, seed, unsure): every grid point at every seed."""
    return {run_name(m, r, s, u): (m, r, s, u) for m, r, u in bases(rates).values() for s in SEEDS}


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
    """How sure a simulated (sure) misread is: a misread keeps its top reading's trust, and the labels are drawn
    alike, so its sureness is that of the charged labels' top readings: the chance the energy gives them
    (TOP_READING_FIT of the trust, from the fit that held each label's lot's quarter out)."""
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
    the run's search stopped above a map its own energy prefers; it splits no wrong lots)."""
    out = {}
    for q in R.QUARTERS:
        f = R.run_file(run, f"-{q}", R.S1_DIR).with_suffix(".gap.json")
        if not f.is_file():
            raise FileNotFoundError(f"{f}: run `reader_spike.py gap {run}` (pinned)")
        out[q] = json.loads(f.read_text())["final_answer_minus_own"]
    return out


def series_of(G: dict) -> dict:
    """The three series: their grid points in rate order, and which share is their x."""
    return {"misread (sure)": ([base_name(0, r) for r in (2, 5, 10, 20)], "misread"),
            "miss": ([base_name(m, 0) for m in (5, 10, 20, 40)], "miss"),
            "misread (unsure)": ([base_name(0, r) + R.UNSURE_SUFFIX for r in UNSURE_RATES], "misread")}


def nominal(G: dict, g: str, share: str) -> float:
    m, r, _ = G[g]
    return float(r if share == "misread" else m)


def build() -> dict:
    rates = s0_rates()
    G = bases(rates)
    B = R.Blocks()
    W = B.weights()
    final_codes = R.codes_of(runs.FINAL_RUN)
    s_final = B.stats(final_codes)
    b_final = R.boot_measures(W, s_final)
    final = R.six(runs.FINAL_RUN) | {COMPANION: R.wrong_outside_1(runs.FINAL_RUN)}
    s0 = R.six("s0") | {COMPANION: R.wrong_outside_1("s0")}
    excess = {"miss": max(rates["reader_miss"] - rates["stage_miss"], 0.0),
              "misread": max(rates["reader_misread"] - rates["stage_misread"], 0.0)}
    res = {"rates_on_answered": rates, "s0_excess": excess, "grid": {k: list(v) for k, v in G.items()},
           "final": {k: final[k] for k in KEY + (COMPANION,)}, "s0_run": {k: s0[k] for k in KEY + (COMPANION,)},
           "bootstrap": {"draws": R.BOOT_DRAWS, "seed": R.BOOT_SEED, "blocks": B.nb, "seed_picks": PICK_SEED,
                         "changed_set_seed": R.CLUSTER_SEED},
           "runs": {}}
    hashes = json.loads(R.run_file("control", "-nw", R.S0_DIR).with_suffix(".json").read_text())["solver_sha256"]
    cp = [json.loads(R.run_file("control-pinned", f"-{q}", R.S0_DIR).with_suffix(".json").read_text())
          for q in R.QUARTERS]
    res["control_pinned"] = {
        "join_equal_final": bool(np.array_equal(R.codes_of("control-pinned"), final_codes)),
        "measures_equal_final": R.six("control-pinned") == R.six(runs.FINAL_RUN),
        "loaded_from": sorted({c["solver_loaded_from"] for c in cp}),
        "solver_sha256_equal_control": all(c["solver_sha256"] == hashes for c in cp)}
    if not (res["control_pinned"]["join_equal_final"] and res["control_pinned"]["measures_equal_final"]
            and res["control_pinned"]["solver_sha256_equal_control"]):
        raise SystemExit(f"the pinned control does not reproduce final: {res['control_pinned']}")
    keep1 = ~R.final_largest_incident()
    boots = {}
    for name, (m, r, seed, unsure) in points(rates).items():
        codes = R.codes_of(name)
        meas = R.six(name) | {COMPANION: R.wrong_outside_1(name)}
        s = B.stats(codes)
        logs = [json.loads(R.run_file(name, f"-{q}", R.S1_DIR).with_suffix(".json").read_text()) for q in R.QUARTERS]
        h = [lg["solver_sha256"] for lg in logs] + [lg["hashes_after"] for lg in logs]
        vs = R.paired(W, s, s_final)
        for k, keep in (("wrong_lots", None), (COMPANION, keep1)):
            d = R.set_deltas(codes, final_codes, keep)
            vs[k]["changed_set_interval"] = R.cluster_interval(d)
            vs[k]["changed_sets"] = int(len(d))
        res["runs"][name] = {"m": m, "r": r, "seed": seed, "unsure": unsure, "realised": realised(name),
                             "measures": {k: meas[k] for k in KEY + (COMPANION,)}, "vs_final": vs,
                             "same_solver_files": all(x == hashes for x in h),
                             "loaded_from_pinned": all(lg.get("solver_loaded_from", "").startswith(
                                 str(R.PINNED_SRC.resolve())) for lg in logs),
                             "search_gap": gaps(name)}
        boots[name] = R.boot_measures(W, s)
    # per grid point: the seed mean and its change; wrong lots by t over seeds, the rest by the two-level bootstrap
    # with the same seed picks at every grid point (the seeds are nested across rates)
    picks = np.random.default_rng(PICK_SEED).integers(0, len(SEEDS), (R.BOOT_DRAWS, len(SEEDS)))
    res["grid_points"] = {}
    for g, (m, r, unsure) in G.items():
        names = [run_name(m, r, s, unsure) for s in SEEDS]
        entry = {"m": m, "r": r, "unsure": unsure, "runs": names,
                 "realised": {k: float(np.mean([res["runs"][n]["realised"][k] for n in names]))
                              for k in ("miss", "misread")},
                 "mean": {k: float(np.mean([res["runs"][n]["measures"][k] for n in names]))
                          for k in KEY + (COMPANION,)},
                 "seeds": {k: [res["runs"][n]["measures"][k] for n in names] for k in KEY + (COMPANION,)},
                 "search_missed": sum(v < 0 for n in names for v in res["runs"][n]["search_gap"].values()),
                 "change": {}}
        for k in MEAS:
            diffs = [res["runs"][n]["measures"][k] - final[k] for n in names]
            D = np.stack([boots[n][k] - b_final[k] for n in names], 1)          # draws x seeds
            two = np.take_along_axis(D, picks, 1).mean(1)
            entry["change"][k] = {"mean": float(np.mean(diffs)), "seeds": diffs,
                                  "interval": R.t_interval(diffs) if k in LOTS else R.interval(two),
                                  "method": "t over seeds" if k in LOTS else "two-level bootstrap",
                                  "t_interval": R.t_interval(diffs), "two_level_interval": R.interval(two),
                                  "block_only_interval": R.interval(D.mean(1))}
        res["grid_points"][g] = entry
    # the cost per point: per rate, change / realised share per seed; per series, the rates weighted alike
    costs = {}
    for series, (gs, share) in series_of(G).items():
        per_rate = {}
        for g in gs:
            e = res["grid_points"][g]
            per_rate[g] = {}
            for k in MEAS:
                ratios = [(res["runs"][n]["measures"][k] - final[k]) / (100 * res["runs"][n]["realised"][share])
                          for n in e["runs"]]
                per_rate[g][k] = {"mean": float(np.mean(ratios)), "interval": R.t_interval(ratios),
                                  "seeds": ratios}
        out = {"per_rate": per_rate, "rates": {g: nominal(G, g, share) for g in gs}}
        for tag, sel in (("all", gs), ("low", [g for g in gs if nominal(G, g, share) <= LOW])):
            out[tag] = {"grid_points": sel}
            for k in MEAS:
                per_seed = [float(np.mean([per_rate[g][k]["seeds"][i] for g in sel])) for i in range(len(SEEDS))]
                out[tag][k] = {"mean": float(np.mean(per_seed)), "interval": R.t_interval(per_seed),
                               "seeds": per_seed}
        costs[series] = out
    res["cost_per_point"] = costs
    # sure against unsure misreads, paired by seed (the same labels and codes misread)
    res["sure_vs_unsure"] = {}
    for r in UNSURE_RATES:
        sure, uns = base_name(0, r), base_name(0, r) + R.UNSURE_SUFFIX
        row = {}
        for k in MEAS:
            d = [res["runs"][run_name(0, r, s, True)]["measures"][k] - res["runs"][run_name(0, r, s)]["measures"][k]
                 for s in SEEDS]
            row[k] = {"sure": res["grid_points"][sure]["change"][k]["mean"],
                      "unsure": res["grid_points"][uns]["change"][k]["mean"],
                      "unsure_minus_sure": float(np.mean(d)), "interval": R.t_interval(d), "seeds": d}
        res["sure_vs_unsure"][r] = row
    res["simulated_misread_confidence"] = simulated_confidence()
    res["reader_misread_masses"] = {"pairs": len(R.reader_misread_masses()),
                                    "wrong_mean": float(np.mean([p[0] for p in R.reader_misread_masses()])),
                                    "right_mean": float(np.mean([p[1] for p in R.reader_misread_masses()]))}
    sure_runs = [p for p in res["runs"].values() if not p["unsure"]]
    res["model"] = {"codes_with_confusions": len(R.confusion()),
                    "misreads": sum(round(p["realised"]["misread"] * p["realised"]["labels"]) for p in sure_runs),
                    "fallback_draws_all_quarters": sum(sum(p["realised"]["fallback_by_quarter"]) for p in sure_runs)}
    figures(res)
    return res


# --- figures ---------------------------------------------------------------------------------------------------------

INK, INK2, GRIDC, SURF = (11, 11, 11), (82, 81, 78), (226, 225, 220), (252, 252, 251)
BLUE, ORANGE, AQUA, YELLOW = (42, 120, 214), (235, 104, 52), (27, 175, 122), (237, 161, 0)
TITLES = {"lines_drawn": "Table district lines the map draws", "drawn_are_table_lines": "Drawn lines that are table "
          "lines", "labels_right": "Zoning labels whose district comes out right", "holes": "Holes",
          "incidents": "Wrong-lot incidents", "wrong_lots": "Wrong lots",
          COMPANION: "Wrong lots outside final's largest incident (post hoc, one-sided)"}
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
    """One PNG per measure (`s1_<measure>.png`): the measure against the share of labels degraded on top of the
    stage's readings (realised, %): sure misreads (blue circles), misses (orange squares), unsure misreads (yellow
    triangles, the reader's own sureness), each seed small and their mean large, the lines through the means from
    final at 0; both at once (aqua diamonds, at the sum of the two shares); the real S0 solve (filled star) at its
    excess error over the stage's readings on the answered labels (miss + misread, 7.6%), and the S1 points at S0's
    excess and measured rates (hollow stars).  Bars: final's value + the 95% interval of the mean's change (t over
    the seeds for wrong lots, the two-level bootstrap for incidents and labels right)."""
    final = res["final"]
    G = res["grid_points"]
    runs_ = res["runs"]
    s0x = 100 * (res["s0_excess"]["miss"] + res["s0_excess"]["misread"])
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
        sub = ("every lot scored; x: share of the 1,881 charged labels degraded on top of the stage's own errors; "
               "small marks: each seed, large: the mean of 3")
        if k in MEAS:
            sub += ("; bars: 95% of the mean (t over seeds)" if k in LOTS else
                    "; bars: 95% of the mean (blocks and seeds)")
        d.text((x0, 50), sub, fill=INK2, font=font(14))
        d.text(((x0 + x1) / 2, y1 + 34), "labels missed or misread on top of the stage's readings (%)", fill=INK2,
               font=font(16), anchor="mt")

        def bar(g, x):
            if g in bars:
                d.line([(x, Y(bars[g][0])), (x, Y(bars[g][1]))], fill=INK2, width=1)

        def mark(shape, cx, cy, col, size):
            if shape == "o":
                d.ellipse([cx - size, cy - size, cx + size, cy + size], fill=col, outline=SURF, width=2)
            elif shape == "s":
                d.rectangle([cx - size, cy - size, cx + size, cy + size], fill=col, outline=SURF, width=2)
            elif shape == "t":
                s = size + 2
                d.polygon([(cx, cy - s), (cx + s, cy + s * 0.8), (cx - s, cy + s * 0.8)], fill=col, outline=SURF)
            else:
                s = size + 2
                d.polygon([(cx, cy - s), (cx + s, cy), (cx, cy + s), (cx - s, cy)], fill=col, outline=SURF)

        def star(cx, cy, hollow, size=10):
            ang = np.linspace(-np.pi / 2, 1.5 * np.pi, 11)[:-1]
            rad = np.where(np.arange(10) % 2 == 0, size, size * 0.45)
            d.polygon([(cx + r * np.cos(a), cy + r * np.sin(a)) for a, r in zip(ang, rad)],
                      fill=None if hollow else INK, outline=INK, width=2)

        lines = (("sure misread", BLUE, "o", "misread", lambda e: e["m"] == 0 and e["r"] > 0 and not e["unsure"]),
                 ("miss", ORANGE, "s", "miss", lambda e: e["r"] == 0 and e["m"] > 0),
                 ("unsure misread", YELLOW, "t", "misread", lambda e: e["unsure"]))
        for label, col, shape, share, on in lines:
            gs = sorted((g for g, e in G.items() if on(e) and not g.startswith("s0")),
                        key=lambda g: G[g]["realised"][share])
            line = [(0.0, final[k])] + [(100 * G[g]["realised"][share], G[g]["mean"][k]) for g in gs]
            d.line([(X(x), Y(v)) for x, v in line], fill=col, width=2)
            for g in gs:
                x = 100 * G[g]["realised"][share]
                bar(g, X(x))
                for n in G[g]["runs"]:
                    mark(shape, X(100 * runs_[n]["realised"][share]), Y(runs_[n]["measures"][k]), col, 3)
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
        for g, txt, dy in (("s0-measured", "S1, measured", 16), ("s0-excess", "S1, excess",
                                                                   -16)):
            e = G[g]
            x = 100 * (e["realised"]["miss"] + e["realised"]["misread"])
            bar(g, X(x))
            star(X(x), Y(e["mean"][k]), True)
            d.text((X(x) + 14, Y(e["mean"][k]) + dy), txt, fill=INK, font=font(13), anchor="lm")
        star(X(s0x), Y(res["s0_run"][k]), False)
        d.text((X(s0x) - 16, Y(res["s0_run"][k])), "S0 (excess)", fill=INK,
               font=font(14), anchor="rm")
        d.ellipse([X(0) - 5, Y(final[k]) - 5, X(0) + 5, Y(final[k]) + 5], fill=INK, outline=SURF, width=2)
        d.text((X(0) + 9, Y(final[k]) - 12), "final", fill=INK, font=font(14), anchor="lm")
        ly = 690
        for i, (txt, col, shape) in enumerate((("sure misread", BLUE, "o"), ("miss", ORANGE, "s"),
                                               ("unsure misread", YELLOW, "t"), ("both (x = sum)", AQUA, "d"))):
            lx = x0 + 165 * i
            mark(shape, lx + 5, ly, col, 6)
            d.text((lx + 18, ly), txt, fill=INK, font=font(15), anchor="lm")
        star(x0 + 680, ly, False, 8)
        d.text((x0 + 694, ly), "S0", fill=INK, font=font(15), anchor="lm")
        star(x0 + 740, ly, True, 8)
        d.text((x0 + 754, ly), "S1 at S0's excess, measured rates", fill=INK, font=font(15), anchor="lm")
        im.save(OUT / f"s1_{k}.png")


# --- the report ------------------------------------------------------------------------------------------------------

def fmt(k, v):
    return f"{v:.2%}" if k in ("lines_drawn", "drawn_are_table_lines", "labels_right") else f"{v:,.0f}"


def fmt_change(k, c) -> str:
    if k == "labels_right":
        return f"{c['mean'] * 100:+.2f} pt [{c['interval'][0] * 100:+.2f}, {c['interval'][1] * 100:+.2f}]"
    return f"{c['mean']:+,.0f} [{c['interval'][0]:+,.0f}, {c['interval'][1]:+,.0f}]"


def fmt_cost(k, c) -> str:
    sc, unit, dec = (100, " pt", 2) if k == "labels_right" else (1, "", 1 if k == "incidents" else 0)
    return (f"{c['mean'] * sc:+,.{dec}f}{unit} [{c['interval'][0] * sc:+,.{dec}f}, "
            f"{c['interval'][1] * sc:+,.{dec}f}]")


NAMES = {"wrong_lots": "wrong lots", COMPANION: "outside final's #1", "incidents": "incidents",
         "labels_right": "labels right"}


def markdown(res: dict) -> str:
    f = res["final"]
    rates = res["rates_on_answered"]
    G = res["grid_points"]
    cp = res["control_pinned"]
    sc = res["simulated_misread_confidence"]
    rm = res["reader_misread_masses"]
    out = ["# Spike S1: the six measures as the readings are degraded", "",
           "The stage's readings degraded at known rates and solved with the stage's walls, placement and search "
           "(see the scripts' docstrings). Every lot scored; final is the point (0, 0): the stage's solve at 44dffb1, "
           "with the search every spike solve runs (the stage was since rebuilt with the compound fusion move on; "
           "spike-s0/results.md). Misreads draw the wrong code "
           f"from the saved readers' disagreements ({res['model']['codes_with_confusions']} codes have some); codes "
           "no reader confused fall back to the codes printed in the label's quarter "
           f"({res['model']['fallback_draws_all_quarters']} of {4 * res['model']['misreads']:,} misread draws over "
           "every sure run and quarter).", "",
           "- **Sure misreads** (the grid's misreads) keep their trust, so they are as sure as the labels' top "
           f"readings: the energy gives them {sc['mean']:.2f} on average, {sc['at_least_0.9']:.0%} at 0.9 or more.",
           "- **Unsure misreads** (fix round 1, r 5, 10, 20%, `-u`): the same labels and codes misread, but each "
           "misread label's P is the template reader's own: the wrong code at the mass the reader gave one of its "
           f"{rm['pairs']} misreads on the answered labels (mean {rm['wrong_mean']:.2f}), the replaced code at the "
           f"mass it gave the answer there (mean {rm['right_mean']:.2f}), the rest spread evenly.", "",
           f"Every S1 solve ran on a pinned copy of the package at commit {R.PINNED_COMMIT} (another agent was editing "
           "src/sfzmaps/solver/ during the runs). The control solved the same way (`control-pinned`) gives final lot "
           f"for lot: {cp['join_equal_final']}; its six measures equal final's: {cp['measures_equal_final']}; its "
           f"solver files hash as the control's: {cp['solver_sha256_equal_control']}. Every run's solver files hash "
           f"as the control's: {all(p['same_solver_files'] for p in res['runs'].values())}; every run loaded the "
           f"pinned copy: {all(p['loaded_from_pinned'] for p in res['runs'].values())}.", "",
           f"S0's measured rates on the {rates['labels']} answered labels the solver charges: miss "
           f"{rates['reader_miss']:.1%}, misread {rates['reader_misread']:.1%} (the stage's readings: miss "
           f"{rates['stage_miss']:.1%}, misread {rates['stage_misread']:.1%}). The x axis is degradation on top of "
           "the stage's own errors, so S0's comparable x is its excess, "
           f"{100 * (res['s0_excess']['miss'] + res['s0_excess']['misread']):.1f}% (the grid point `s0-excess`); "
           "`s0-measured` puts its full rates on the curve too.", "",
           "**The companion \"outside final's #1\" was chosen after seeing S0, and it is one-sided:** it leaves out the "
           "1,838 lots of final's largest incident (a search trap under final's readings) and keeps every other run's "
           "own traps. Read it beside wrong lots, the two bracketing from either side.", "",
           "## Intervals", "",
           "- **Wrong lots and the companion: t over the 3 seeds** (df 2, t = 4.30). It holds the search's run-to-run "
           "noise and which labels the seed degraded. With 3 seeds it is wide, and at the low rates it includes 0: "
           "**3 seeds can't bound the wrong-lot cost there**.",
           "- Per run, the changed sets' bootstrap (connected lots whose code differs from final, one search outcome "
           "each, resampled) and the blocks' (lot sampling only) are in \"Every run\" and results.json. Both are far "
           "narrower than the seed spread: neither holds the search's noise.",
           "- **Incidents and labels right: the two-level bootstrap** (blocks resampled, one draw for every run; the 3 "
           "seeds resampled with replacement, the same picks at every grid point because the seeds are nested across "
           "rates). The seeds agree on these two, so the intervals are tight.", "",
           "## The grid: seed means (3 seeds each)", "",
           "| point | miss | misread | " + " | ".join(report_mod.MEASURES[k] for k in KEY)
           + " | outside final's #1 |", "|---|---|---|" + "---|" * (len(KEY) + 1),
           "| final | 0 | 0 | " + " | ".join(fmt(k, f[k]) for k in KEY + (COMPANION,)) + " |",
           "| S0 (the template reader) | – | – | " + " | ".join(fmt(k, res["s0_run"][k]) for k in KEY + (COMPANION,))
           + " |"]
    for g, e in G.items():
        out.append(f"| `{g}` | {e['realised']['miss']:.1%} | {e['realised']['misread']:.1%} | "
                   + " | ".join(fmt(k, e["mean"][k]) for k in KEY + (COMPANION,)) + " |")
    out += ["", "The change of each seed mean against final, 95% (wrong lots and the companion: t over seeds; "
            "incidents and labels right: two-level bootstrap):", "",
            "| point | wrong lots | outside final's #1 | incidents | labels right |", "|---|---|---|---|---|"]
    for g, e in G.items():
        out.append(f"| `{g}` | " + " | ".join(fmt_change(k, e["change"][k]) for k in MEAS) + " |")
    out += ["", "## The seed spread", "", "| point | wrong lots (seeds 0, 1, 2) | outside final's #1 | incidents | "
            "labels right | quarter solves whose search stopped above final's map (of 12) |",
            "|---|---|---|---|---|---|"]
    for g, e in G.items():
        sv = e["seeds"]
        out.append(f"| `{g}` | {', '.join(f'{x:,}' for x in sv['wrong_lots'])} | "
                   f"{', '.join(f'{x:,}' for x in sv[COMPANION])} | {', '.join(f'{x:,}' for x in sv['incidents'])} | "
                   f"{', '.join(f'{x:.2%}' for x in sv['labels_right'])} | {e['search_missed']} |")
    neg = [(n, q, v) for n, p in res["runs"].items() for q, v in p["search_gap"].items() if v < 0]
    miss_only = [n for n, p in res["runs"].items() if p["r"] == 0 and not p["unsure"]]
    miss_neg = [x for x in neg if x[0] in miss_only]
    out += ["", "The last column (the search check): quarter solves whose run's own energy rates final's map lower "
            "than the answer its search stopped at. It shows only that the search stopped above a map its own energy "
            "prefers; **it splits no wrong lots between the search and the readings**, and the search's share of any "
            "cost is not measured here. It can only fire where final's map stays a low-energy answer under the run's "
            "readings: when labels are dropped, not when they are misread (a misread makes final's map costly), so "
            "its silence on misread runs says nothing about their search. "
            f"{len(neg)} of {4 * len(res['runs'])} quarter solves: {len(miss_neg)} of the {4 * len(miss_only)} "
            f"miss-only ones, {len(neg) - len(miss_neg)} of the rest" + (
                ", the largest gaps: " + ", ".join(f"`{n}` {q} {v:+.1f}" for n, q, v in sorted(neg, key=lambda t: t[2])[:6])
                if neg else "") + "."]
    c = res["cost_per_point"]
    out += ["", "## The cost per point of reader error", "",
            "Per rate: each seed's change over its realised share in percent (1 point ≈ 19 of the 1,881 labels), the "
            "mean of the 3 seeds and its t interval. Through-origin least squares (round 0) weighted the top rate by "
            "about 75%; here each rate weighs alike.", "",
            "| series | rate | " + " | ".join(NAMES[k] for k in MEAS) + " |", "|---|---|" + "---|" * len(MEAS)]
    for series, v in c.items():
        for g, row in v["per_rate"].items():
            out.append(f"| {series} | {v['rates'][g]:g}% | " + " | ".join(fmt_cost(k, row[k]) for k in MEAS) + " |")
    out += ["", "Each series with its rates weighted alike (per seed, the mean of its per-rate ratios; t over the 3 "
            f"seeds, paired across rates). \"Low\": the rates at or below {LOW:g}% (sure misread r 2, 5, 10; miss m 5, "
            "10; unsure misread r 5, 10).", "",
            "| series | rates | " + " | ".join(NAMES[k] for k in MEAS) + " |", "|---|---|" + "---|" * len(MEAS)]
    for series, v in c.items():
        for tag in ("all", "low"):
            rs = ", ".join(f"{v['rates'][g]:g}" for g in v[tag]["grid_points"])
            out.append(f"| {series} | {tag}: {rs} | " + " | ".join(fmt_cost(k, v[tag][k]) for k in MEAS) + " |")
    ms, mi, mu = c["misread (sure)"], c["miss"], c["misread (unsure)"]
    out += ["", f"**Each point of sure misread costs about {ms['all']['wrong_lots']['mean']:,.0f} wrong lots "
            f"[{ms['all']['wrong_lots']['interval'][0]:,.0f}, {ms['all']['wrong_lots']['interval'][1]:,.0f}] "
            f"({ms['low']['wrong_lots']['mean']:,.0f} [{ms['low']['wrong_lots']['interval'][0]:,.0f}, "
            f"{ms['low']['wrong_lots']['interval'][1]:,.0f}] at 10% and below); each point of miss about "
            f"{mi['all']['wrong_lots']['mean']:,.0f} [{mi['all']['wrong_lots']['interval'][0]:,.0f}, "
            f"{mi['all']['wrong_lots']['interval'][1]:,.0f}] ({mi['low']['wrong_lots']['mean']:,.0f} "
            f"[{mi['low']['wrong_lots']['interval'][0]:,.0f}, {mi['low']['wrong_lots']['interval'][1]:,.0f}] at 10% "
            f"and below); each point of unsure misread about {mu['all']['wrong_lots']['mean']:,.0f} "
            f"[{mu['all']['wrong_lots']['interval'][0]:,.0f}, {mu['all']['wrong_lots']['interval'][1]:,.0f}].** "
            "t over 3 seeds."]
    out += ["", "## Sure against unsure misreads", "",
            "The same labels and replacement codes per seed; only the misread labels' P differs. Change against final "
            "(seed means) and unsure − sure, paired by seed (t over 3 seeds):", "",
            "| r | measure | sure | unsure | unsure − sure [95%] |", "|---|---|---|---|---|"]
    for r, row in res["sure_vs_unsure"].items():
        for k in MEAS:
            v = row[k]
            if k == "labels_right":
                out.append(f"| {r}% | {NAMES[k]} | {v['sure'] * 100:+.2f} pt | {v['unsure'] * 100:+.2f} pt | "
                           f"{v['unsure_minus_sure'] * 100:+.2f} pt [{v['interval'][0] * 100:+.2f}, "
                           f"{v['interval'][1] * 100:+.2f}] |")
            else:
                out.append(f"| {r}% | {NAMES[k]} | {v['sure']:+,.0f} | {v['unsure']:+,.0f} | "
                           f"{v['unsure_minus_sure']:+,.0f} [{v['interval'][0]:+,.0f}, {v['interval'][1]:+,.0f}] |")
    out += ["", "## Figures", "",
            "One per measure, to read how fast each falls as the readings get worse and where S0 sits: the measure "
            "against the share of labels degraded on top of the stage's readings; blue circles sure misreads, orange "
            "squares misses, yellow triangles unsure misreads, aqua diamonds both at once (at the sum of the two "
            "shares); small marks each seed, large marks their mean, the lines through the means from final at 0; "
            "the filled star the real S0 solve at its excess error (7.6%), the hollow stars the S1 points at S0's "
            "excess and measured rates. Bars: final's value plus the 95% interval of the mean's change (t over seeds "
            "for wrong lots and the companion; blocks and seeds for incidents and labels right).", ""]
    out += [f"- `s1_{k}.png`: {TITLES[k]}." for k in FIGURES]
    out += ["", "## Every run", "", "| run | miss | misread | " + " | ".join(report_mod.MEASURES[k] for k in KEY)
            + " | outside final's #1 | wrong lots − final | changed sets' 95% | blocks' 95% (lot sampling) |",
            "|---|---|---|" + "---|" * (len(KEY) + 4)]
    for n, p in res["runs"].items():
        v = p["vs_final"]["wrong_lots"]
        out.append(f"| `{n}` | {p['realised']['miss']:.1%} | {p['realised']['misread']:.1%} | "
                   + " | ".join(fmt(k, p["measures"][k]) for k in KEY + (COMPANION,))
                   + f" | {p['measures']['wrong_lots'] - f['wrong_lots']:+,} | [{v['changed_set_interval'][0]:+,.0f}, "
                   f"{v['changed_set_interval'][1]:+,.0f}] | [{v['diff_interval'][0]:+,.0f}, "
                   f"{v['diff_interval'][1]:+,.0f}] |")
    return "\n".join(out) + "\n"


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "solve":
        R.run_jobs([("control-pinned", q) for q in R.QUARTERS] + [(n, q) for n in points() for q in R.QUARTERS],
                   max_procs=4, load_start=20.0, min_procs=0)
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
