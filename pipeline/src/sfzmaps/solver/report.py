"""The solve's report: a run's six measures against J7's, with the data-error parcels left out and with every lot,
and what in the definitions differs from J7's.

The six (`measures.py` defines them): the share of the table's district lines the map draws; the share of the lines
it draws that are table lines; the share of zoning labels whose district comes out right; holes; wrong-lot
incidents; wrong lots.  Wrong lots and incidents count every scored lot in both columns, as J7's did.  J7's numbers
are those of the R&D write-up (`docs/research/joint-solver/joint-solver.md`), where 33 data-error parcels were left
out.  Each run's data errors come from its own holes, so the left-out set differs from run to run: runs of this
build compare on the every-lot column, and the left-out column is there for the comparison with J7.

Beside them: wrong lots by quarter, the largest incidents, the holes by cause, the walls' held-out log loss
(`walls_evaluation`), the placement's held-out hit rates, and a check of the energy's trust calibration
(`top_reading_calibration`).

Writes report.json and report.md into the run's folder (`runs.solve_dir`).  Nothing in them is a time: the same
run gives the same bytes."""
import json
from pathlib import Path

import numpy as np
from scipy.special import expit

from sfzmaps.solver import energy, paths, runs
from sfzmaps.solver.city import city, load_run, wall_odds
from sfzmaps.solver.codes import table_vocabulary
from sfzmaps.solver.labels import readings
from sfzmaps.solver.measures import Table, data_error_lots, incidents, score

J7 = {"lines_drawn": 0.963, "drawn_are_table_lines": 0.969, "labels_right": 0.950, "holes": 15, "incidents": 824,
      "wrong_lots": 8625}
J7_DATA_ERRORS = 33         # parcels J7 left out
# J7's walls' held-out log loss against the scoring table, on J7's own pairs (both lots known and in the quarter)
J7_WALLS_LOG_LOSS = {"nw": 0.0422, "ne": 0.0676, "sw": 0.0313, "se": 0.0412}
PROBABILITY_FLOOR = 1e-6    # the walls' log loss counts a probability below this as this (as the walls' own does)
MEASURES = {"lines_drawn": "Table district lines the map draws",
            "drawn_are_table_lines": "Lines the map draws that are table lines",
            "labels_right": "Zoning labels whose district comes out right",
            "holes": "Holes (short breaks in a drawn district line)",
            "incidents": "Wrong-lot incidents (one connected mistake = 1)",
            "wrong_lots": "Wrong lots"}
DEFINITION_CHANGES = (
    "The new warp: every sheet is placed by its corrected placement from the sheet warp "
    "(data/warp/2000-digital-zoning), so the merged page, the prints and the hand leader moves sit where it puts them.",
    "Scored lots without the old run: a lot is scored when the table gives it one district and more than half of it "
    "is on the mapped area; J7 also required it to lie inside a closed area of an older run.",
    "Fits on the uncorrected table: the trust, the street sides, the lot sizes, the walls and the placement learn "
    "from the table as downloaded; J7 used the by-eye corrections in three fits. The scores still count against the "
    "corrected table.",
    "The always-zero trust features dropped: the trust is one logistic regression on 23 features; J7's had six more "
    "(the place answers), all zero.",
    "Strokes from the raw page: the strokes the placement reads for leader lines are the merged page's ink thinned, "
    "text and lines alike; J7's came from a cleaned page with the letters erased.",
    "The walls' feature set: the walls read the raw page's strokes (`wallevidence`), with the candidate features "
    "their evaluation.json names, in one ridge logistic regression per quarter; J7's walls came from its cleaned page "
    "and its chain of graph steps.",
    "Unchanged, and named here: the placement's training answers from the table (stay when the lot under the text has "
    "the label's code, else the nearest walled area whose lots mostly have it) count an area's lots across the quarter "
    "split, as J7's did, so a label near the split can learn from table codes of the quarter its model is scored on.",
)
ONE_MEASURE = ("Each run's data errors come from its own holes, so the left-out set differs from run to run: compare "
               "runs of this build on the every-lot column; the left-out column is there for the comparison with J7.")
CALIBRATION_BINS = (0.0, 0.5, 0.7, 0.9, 0.97, 0.99, 1.0)     # of the top reading's trust
CALIBRATION_STEPS = 50      # Newton steps of the refit, at most
DIGITS = 6                  # the report's floats are rounded to this many decimals


def _r(x) -> float:
    return round(float(x), DIGITS)


def _rounded(v):
    """v with every float in it (in dicts and lists too) rounded to DIGITS decimals."""
    if isinstance(v, dict):
        return {k: _rounded(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_rounded(x) for x in v]
    return _r(v) if isinstance(v, (float, np.floating)) else v


def six_measures(codes: np.ndarray, exclude: np.ndarray) -> dict:
    """The six measures of `codes` (one per lot) and their companions, leaving the lots of `exclude` out of the
    borders, labels and holes."""
    s = score(codes, Table(exclude))
    i = incidents(codes)
    return {"lines_drawn": _r(s["recall"]), "lines_drawn_by_length": _r(s["recall_ft"]),
            "drawn_are_table_lines": _r(s["precision"]), "drawn_are_table_lines_by_length": _r(s["precision_ft"]),
            "labels_right": _r(s["zoning_labels"]), "zoning_labels": s["n_zoning_labels"],
            "all_labels_right": _r(s["labels"]), "all_labels": s["n_labels"],
            "holes": int(s["holes"]), "hole_ft": _r(s["hole_ft"]), "missed_runs_over_100_ft": int(s["long_runs"]),
            "missed_run_miles": _r(s["long_miles"]),
            "incidents": i["incidents"], "wrong_lots": i["wrong_lots"], "incidents_of_100_lots": int(i["big"]),
            "lots_in_them": int(i["lots_in_big"])}


def top_reading_calibration() -> dict:
    """How well `energy.TOP_READING_FIT` (logit(chance the top reading is right) = a + b logit(trust)) fits this
    build's trust.  Per label as the energy charges it (a print's own readings, not a far-side copy, of the codes
    the table uses) whose lot under the text has a known code: its top reading's trust from the fit that held that
    lot's quarter out, and whether the code is the table's as downloaded there (what the trust learns).  Gives the
    log loss of the energy's (a, b) and of (a, b) fitted afresh by logistic regression, and per trust bin the
    chance the energy states against how often the top reading is right."""
    C = city()
    vocabulary = table_vocabulary(C.lots)
    trust, right = [], []
    for k, q in enumerate(paths.QUARTERS):
        rd = readings(q, vocabulary)
        best, lot_of = {}, {}
        for r in np.flatnonzero(~rd.far):
            label = int(rd.label[r])
            T = best.setdefault(label, np.zeros(len(rd.names)))
            T[rd.code[r]] = max(T[rd.code[r]], rd.trust[r])
            lot_of[label] = int(rd.lot[r])
        for label in sorted(best):
            lot = lot_of[label]
            if C.quarter[lot] != k or not C.known[lot]:
                continue
            top = int(np.argmax(best[label]))
            trust.append(best[label][top])
            right.append(rd.names[top] == C.table_raw[lot])
    t, y = np.array(trust, float), np.array(right, float)
    t_clipped = np.clip(t, 1e-3, 1 - 1e-3)                  # as the energy clips it
    x = np.c_[np.ones(len(t)), np.log(t_clipped / (1 - t_clipped))]
    used = np.array(energy.TOP_READING_FIT, float)
    w = np.zeros(2)
    for _ in range(CALIBRATION_STEPS):
        p = expit(x @ w)
        step = np.linalg.solve(x.T @ (x * (p * (1 - p))[:, None]) + 1e-9 * np.eye(2), x.T @ (y - p))
        w += step
        if np.abs(step).max() < 1e-10:
            break

    def log_loss(weights):
        p = np.clip(expit(x @ weights), 1e-12, 1 - 1e-12)
        return _r(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))) if len(y) else None

    stated = expit(x @ used)
    bins = []
    for lo, hi in zip(CALIBRATION_BINS[:-1], CALIBRATION_BINS[1:]):
        m = (t >= lo) & ((t < hi) if hi < 1 else (t <= hi))
        if m.any():
            bins.append({"from": lo, "to": hi, "labels": int(m.sum()), "stated": _r(stated[m].mean()),
                         "right": _r(y[m].mean())})
    return {"used": list(energy.TOP_READING_FIT), "labels": len(y), "right": int(y.sum()),
            "log_loss_used": log_loss(used), "refit": [_r(v) for v in w], "log_loss_refit": log_loss(w),
            "bins": bins}


def walls_evaluation(walls_dir: Path | None = None) -> dict:
    """The walls of `walls_dir` (default WALLS_DIR), per quarter q: the held-out log loss of q's odds on the pairs
    with both lots known and in q, against the scoring table (as J7's quoted numbers are; here on this build's
    pairs, J7's on its own) and, from their evaluation.json when there is one, against the table as downloaded (the
    fit's own target); and the candidate features they were fitted with."""
    C = city()
    folder = Path(walls_dir or runs.WALLS_DIR)
    line = C.table[C.a] != C.table[C.b]
    scoring = {}
    for k, q in enumerate(paths.QUARTERS):
        held = C.known[C.a] & C.known[C.b] & (C.quarter[C.a] == k) & (C.quarter[C.b] == k)
        p = np.clip(expit(wall_odds(q, folder)[held]), PROBABILITY_FLOOR, 1 - PROBABILITY_FLOOR)
        scoring[q] = _r(-np.mean(np.where(line[held], np.log(p), np.log(1 - p))))
    out = {"held_out_log_loss": scoring, "j7_held_out_log_loss": dict(J7_WALLS_LOG_LOSS)}
    f = folder / "evaluation.json"
    if f.is_file():
        ev = json.loads(f.read_text())
        out.update(candidates=ev.get("candidates", []), features=ev.get("features"),
                   held_out_log_loss_table_as_downloaded={q: v["held_out_log_loss"] for q, v in ev["quarters"].items()})
    return out


def build_report(run: str, placement_summary: dict, hole_causes: dict, walls_dir: Path | None = None,
                 folder: Path | None = None) -> dict:
    """The report of the run `run` (see the module), from its join and data-error list in `folder` (default the
    run's own): `placement_summary` is what `placement.fit` returned, `hole_causes` the run's holes (data errors
    counted) by cause, `walls_dir` the walls it was solved with."""
    C = city()
    folder = Path(folder or runs.solve_dir(run))
    codes = load_run(run, folder)
    errors = data_error_lots(folder)
    wrong = C.scored & (codes != C.table)
    biggest = incidents(codes)["biggest"]
    return {
        "run": run,
        "measures": {"data_errors_left_out": six_measures(codes, errors),
                     "every_lot": six_measures(codes, np.zeros(C.n, bool))},
        "j7": dict(J7),
        "data_errors": {"parcels": int(errors.sum()), "j7_parcels": J7_DATA_ERRORS},
        "wrong_lots_by_quarter": {q: int(n) for q, n in zip(paths.QUARTERS,
                                                            np.bincount(C.quarter[wrong], minlength=4))},
        "largest_incidents": [{"lots": int(n), "table": t, "solved": r, "blocks": list(blocks)}
                              for n, t, r, blocks in biggest],
        "holes_by_cause": {c: int(n) for c, n in sorted(hole_causes.items())},
        "walls": walls_evaluation(walls_dir),
        "placement": _rounded(placement_summary),
        "top_reading_calibration": top_reading_calibration(),
        "definition_changes": list(DEFINITION_CHANGES),
    }


def _value(key: str, v) -> str:
    if v is None:
        return "–"
    return f"{v:.1%}" if key in ("lines_drawn", "drawn_are_table_lines", "labels_right") else f"{v:,}"


def markdown(report: dict) -> str:
    """The report as Markdown."""
    left_out, every = report["measures"]["data_errors_left_out"], report["measures"]["every_lot"]
    de = report["data_errors"]
    out = [f"# The 2000 solve: run {report['run']}", "",
           "Every lot takes its code from the solve that held its quarter out; scored against the April 2000 table "
           "with its by-eye corrections.", "",
           f"| Measure | J7 ({de['j7_parcels']} data errors left out) | This run ({de['parcels']:,} data errors left "
           "out) | This run, every lot |",
           "|---|---|---|---|"]
    for key, name in MEASURES.items():
        out.append(f"| {name} | {_value(key, report['j7'][key])} | {_value(key, left_out[key])} | "
                   f"{_value(key, every[key])} |")
    out += ["",
            f"- {ONE_MEASURE}",
            f"- By length: lines drawn {left_out['lines_drawn_by_length']:.1%}, drawn that are table lines "
            f"{left_out['drawn_are_table_lines_by_length']:.1%} (data errors left out).",
            f"- Labels: {left_out['zoning_labels']:,} zoning labels; all labels right "
            f"{left_out['all_labels_right']:.1%} of {left_out['all_labels']:,}.",
            f"- Holes: {left_out['hole_ft']:,.0f} ft; missed runs over 100 ft: {left_out['missed_runs_over_100_ft']:,} "
            f"({left_out['missed_run_miles']:.1f} mi).",
            f"- Incidents of 100+ lots: {left_out['incidents_of_100_lots']} holding {left_out['lots_in_them']:,} lots. "
            "Wrong lots and incidents count every scored lot in both columns, as J7's did.",
            "- Wrong lots by quarter: " + ", ".join(f"{q} {n:,}" for q, n in report["wrong_lots_by_quarter"].items())
            + ".", "",
            "## The largest incidents", "", "| Lots | Table | Solved | Blocks |", "|---|---|---|---|"]
    out += [f"| {i['lots']:,} | {i['table']} | {i['solved'] or '(none)'} | {', '.join(i['blocks'])} |"
            for i in report["largest_incidents"]]
    out += ["", "## Holes by cause (data errors counted)", ""]
    out += [f"- {c}: {n}" for c, n in report["holes_by_cause"].items()] or ["- none"]
    walls = report["walls"]
    fitted = "candidates" in walls
    out += ["", "## The walls", ""]
    if fitted:
        out += [f"Candidate features: {', '.join(walls['candidates']) or 'none'}; {walls['features']} features.", ""]
    out += ["Held-out log loss per quarter, on the pairs with both lots known and in the quarter:", "",
            "| Quarter | Against the scoring table | J7, the same on its own pairs |"
            + (" Against the table as downloaded (the fit's target) |" if fitted else ""),
            "|---|---|---|" + ("---|" if fitted else "")]
    for q, v in walls["held_out_log_loss"].items():
        out.append(f"| {q} | {v:.4f} | {walls['j7_held_out_log_loss'][q]:.4f} |"
                   + (f" {walls['held_out_log_loss_table_as_downloaded'][q]:.4f} |" if fitted else ""))
    placed = report["placement"]
    out += ["", "## The placement", ""]
    if "held_out" in placed:
        out += [f"Labels {placed['labels']:,}, with an answer {placed['with_answer']:,} (hand leader moves "
                f"{placed['hand_moves']}). Each model on the labels it held out:", "",
                "| Labels | Number | Right | Always stay |", "|---|---|---|---|"]
        out += [f"| {name} | {v['n']:,} | {v['right']:.1%} | {v['always_stay']:.1%} |"
                for name, v in placed["held_out"].items()]
    else:
        out += [f"`{json.dumps(placed)}`"]
    cal = report["top_reading_calibration"]
    out += ["", "## The trust's calibration", "",
            f"The energy reads a label's top reading as right with logit = {cal['used'][0]} + {cal['used'][1]} "
            f"logit(trust), fitted on J7's trust. On this build's held-out trust ({cal['labels']:,} labels, "
            f"{cal['right']:,} right): log loss {cal['log_loss_used']:.4f}; refitted, logit = {cal['refit'][0]:.3f} "
            f"+ {cal['refit'][1]:.3f} logit(trust), log loss {cal['log_loss_refit']:.4f}.", "",
            "| Trust | Labels | Stated | Right |", "|---|---|---|---|"]
    out += [f"| {b['from']}–{b['to']} | {b['labels']:,} | {b['stated']:.1%} | {b['right']:.1%} |"
            for b in cal["bins"]]
    out += ["", "## What differs from J7's definitions", ""]
    out += [f"- {c}" for c in report["definition_changes"]]
    return "\n".join(out) + "\n"


def write_report(report: dict, folder: Path | None = None) -> None:
    """Write report.json and report.md into `folder` (default the run's own, `runs.solve_dir`)."""
    folder = Path(folder or runs.solve_dir(report["run"]))
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "report.json").write_text(json.dumps(report, indent=1, ensure_ascii=False) + "\n")
    (folder / "report.md").write_text(markdown(report))
