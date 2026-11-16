"""The census of a run's incidents: every incident of MIN_LOTS+ lots, and why its wrong answer won, from the run's
own saved costs.

    uv run python scripts/incidents/census.py [--run final] [--min-lots 20]

Per incident (a connected set of wrong scored lots sharing one table code and one wrong code, as measures.py
defines it), in the solve that held its quarter out (runs/{run}-{q}-costs.npz):
  * its lots, blocks, table code, solved code and quarter;
  * the changed set D: the connected lots around it whose solved code differs from the table's.  The table's answer
    on D (every other lot as solved) against the solved answer: the energy change split into its terms (common.split):
    labels on their own lot, the placement's moves, street labels' far-side copies, the tie-break, the walls where
    lots differ with no line and where they share a code across a line.  Positive = that term favours the solved
    (wrong) answer.  The same for the incident's lots alone (I).
  * the solver's own search confined to D, from the solved answer and from the table's: the lowest energy it finds
    and how much of the incident it gets right; and the cheapest answer found right on the incident (its lots held
    at the table's codes; the rest of D, with 0, 1 or RING_HOPS rings of neighbouring lots, searched from the table's
    answer; the lowest of the three), with its split and its own driving term.  The table's exact answer can pay for
    details beside the incident's cause (a table line the print draws a lot over); this one does not;
  * the driving term: "search" when the table's answer costs less, or the search confined to D finds a lower energy
    that gets most of the incident right (the solve's answer is not its own minimum there); else the term group
    (labels / placement = moves + far side / walls / tie-break) that favours the wrong answer most on D;
  * the labels that move the energy most, the walls on the table lines the solve leaves out and on the lines it
    draws that the table lacks (this run's odds and J7's), and the labels whose text lies on the incident;
  * J7: the share of the incident's lots J7 gets right, J7's answer's energy on D under this run's costs, and, when
    the solve with J7's walls exists (common.J7_WALLS_RUN), the share it gets right: when J7 is right, its walls
    alone fix it if that solve is right too, else the rest of J7 (its labels, placement or search) does.
Writes CENSUS_DIR/incidents.json and CENSUS_DIR/census.md (common.CENSUS_DIR), and each incident's cheapest answer
right on it as CENSUS_DIR/answers/cheapest-right-<id>.npz (the lots it changes and their codes, in the solve of its
quarter).  Reads only; CPU, one thread."""
import argparse
import json
from collections import Counter, defaultdict

import numpy as np
from scipy.special import expit

from common import (CENSUS_DIR, J7_DIR, J7_WALLS_RUN, KINDS, blocks, changed_set, energy_on, has_run, incidents,
                    label_charges, load_costs, mask, ring, run_codes, search_on, split)
from sfzmaps.solver import paths, runs
from sfzmaps.solver.city import city
from sfzmaps.solver.measures import data_error_lots
from sfzmaps.solver.trust import load_prints

GROUPS = {"labels": ("labels",), "placement": ("placement", "far_side"), "walls": ("walls",),
          "tie_break": ("tie_break",)}
TOP_LABELS = 8
RING_HOPS = 2               # the cheapest answer right on the incident may also move lots this many steps around D


def r(x, d=3):
    return round(float(x), d)


def stats(lo):
    """median probability, share above 0.5, count, of log odds `lo` (NaN = none)"""
    lo = lo[np.isfinite(lo)]
    if not len(lo):
        return None
    p = expit(lo)
    return {"pairs": int(len(lo)), "median_p": r(np.median(p)), "share_above_half": r((p > 0.5).mean())}


def groups(s):
    return {k: sum(s[t] for t in v) for k, v in GROUPS.items()}


def drive(s, right=()):
    """(driving term, its amount): "search" when the table's answer is cheaper, or when another answer that gets
    most of the incident right (`right`: their energy changes) is cheaper than the solved answer (amount: the lowest
    such energy change); else the group favouring the wrong answer most (amount: its change)"""
    g = groups(s)
    lower = [s["total"], *right]
    if min(lower) < -1e-6:
        return "search", r(min(lower))
    k = max(g, key=g.get)
    return k, r(g[k])


def j7_input(j7_right, j7w_right):
    """which of J7's inputs makes the difference, from J7's share right and the J7-walls solve's"""
    if j7_right is None or j7_right < 0.5:
        return None
    if j7w_right is None:
        return "unknown (no J7-walls solve)"
    return "walls" if j7w_right >= 0.5 else "not the walls alone (labels, placement or J7's search)"


def main(run: str, min_lots: int) -> None:
    C = city()
    a, b = C.a, C.b
    codes = run_codes(run)
    found = incidents(codes, min_lots)
    for k, inc in enumerate(found):
        inc["id"] = k + 1
    blklot = C.lots["blklot"].astype(str).values
    j7 = run_codes("j7") if has_run("j7") else None
    j7w = run_codes(J7_WALLS_RUN) if has_run(J7_WALLS_RUN) else None
    errors = data_error_lots(runs.solve_dir(run))
    prints = load_prints()
    both = C.known[a] & C.known[b]
    tline = both & (C.table[a] != C.table[b])
    drawn = codes[a] != codes[b]
    wrong_any = C.scored & (codes != C.table)
    inc_of = np.zeros(C.n, int)
    for inc in found:
        inc_of[inc["lots"]] = inc["id"]
    out = []
    for qi, q in enumerate(paths.QUARTERS):
        here = [inc for inc in found if max(inc["quarters"], key=inc["quarters"].get) == qi]
        if not here:
            continue
        costs = load_costs(run, q)
        ch = label_charges(run, q, costs)
        names = np.array(costs.names)
        j7_odds = np.load(J7_DIR / f"walls_{q}.npz")["lo_ab"].astype(float) if j7 is not None else None
        if j7_odds is not None:
            j7_odds[j7_odds == 0] = np.nan           # a pair J7 lacks (import_j7.py writes 0)
        own_lot = {}                                  # label -> the lot under its text
        for lab, k_, lot in zip(ch.label, ch.kind, ch.lot):
            if KINDS[k_] in ("own", "stay"):
                own_lot[int(lab)] = int(lot)
        top_of = {int(lab): (int(t), float(tt)) for lab, t, tt in zip(ch.label, ch.top, ch.top_trust)}
        for inc in here:
            I = inc["lots"]
            x0 = costs.codes
            agree = float(np.mean(names[x0[I]] == codes[I]))
            D = changed_set(costs, I)
            xD = x0.copy()
            xD[D] = costs.table_codes[D]
            xI = x0.copy()
            okI = costs.table_codes[I] >= 0
            xI[I[okI]] = costs.table_codes[I[okI]]
            sD, sI = split(costs, ch, x0, xD, D), split(costs, ch, x0, xI, I)
            e_solved, e_table = energy_on(costs, x0, D), energy_on(costs, xD, D)
            ends = {}
            for start, x in (("solved", x0), ("table", xD)):
                y = search_on(costs, x, D)
                ends[start] = {"energy_change": r(energy_on(costs, y, D) - e_solved),
                               "incident_right": r(np.mean(y[I] == costs.table_codes[I])),
                               "changed_set_right": r(np.mean(y[D] == costs.table_codes[D]))}
            # the cheapest answer right on the incident: its lots at the table's codes, the rest of D and of RING_HOPS
            # rings of neighbours searched from the table's answer on D; the lowest found
            sR, hopsR = None, None
            for hops in range(RING_HOPS + 1):
                y = search_on(costs, xD, np.setdiff1d(ring(D, hops), I[okI]))
                s_ = split(costs, ch, x0, y, np.flatnonzero(y != x0))
                if sR is None or s_["total"] < sR["total"] - 1e-6:
                    sR, hopsR, yR = s_, hops, y
            moved = np.flatnonzero(yR != x0)
            (CENSUS_DIR / "answers").mkdir(parents=True, exist_ok=True)
            np.savez_compressed(CENSUS_DIR / "answers" / f"cheapest-right-{inc['id']}.npz", lots=moved,
                                codes=names[yR[moved]], quarter=q)
            moved = moved[C.scored[moved]]
            right_info = {"lots_changed": int((yR != x0).sum()), "rings_searched": hopsR,
                          "scored_lots_changed": int(len(moved)),
                          "wrong_scored_before": int((names[x0[moved]] != C.table[moved]).sum()),
                          "wrong_scored_after": int((names[yR[moved]] != C.table[moved]).sum())}
            dD, amtD = drive(sD, [v["energy_change"] for v in ends.values() if v["incident_right"] >= 0.5]
                             + [sR["total"]])
            dI, amtI = drive(sI)
            dR, amtR = drive(sR)
            best = min([0.0, e_table - e_solved, sR["total"]] + [v["energy_change"] for v in ends.values()])
            # labels moving the energy most on D
            lab_d = defaultdict(float)
            lab_k = defaultdict(Counter)
            for lab, k_, v in zip(*sD["by_label"]):
                lab_d[int(lab)] += v
                lab_k[int(lab)][KINDS[k_]] += 1
            top_labels = []
            for lab in sorted(lab_d, key=lambda t: -abs(lab_d[t]))[:TOP_LABELS]:
                if abs(lab_d[lab]) < 1e-3:
                    break
                lot = own_lot.get(lab, int(prints["lab_lot"].iloc[lab]))
                t, tt = top_of[lab]
                top_labels.append({"label": lab, "x": r(prints["x"].iloc[lab], 0), "y": r(prints["y"].iloc[lab], 0),
                                   "text_lot": blklot[lot], "text_lot_table": str(C.table[lot]),
                                   "top_reading": str(names[t]), "trust": r(tt), "charges": dict(lab_k[lab]),
                                   "favours_solved_by": r(lab_d[lab])})
            # labels whose text lies on the incident
            inI = mask(C.n, I)
            on = [lab for lab, lot in own_lot.items() if inI[lot]]
            reads = Counter("table" if names[top_of[lab][0]] == inc["table"] else
                            "solved" if names[top_of[lab][0]] == inc["solved"] else "other" for lab in on)
            # walls around the incident
            P = np.flatnonzero(inI[a] | inI[b])
            missed, false = P[tline[P] & ~drawn[P]], P[both[P] & ~tline[P] & drawn[P]]
            walls = {}
            for nm, S in (("table_lines_left_out", missed), ("lines_drawn_not_in_table", false)):
                walls[nm] = {"pairs": int(len(S)), "ft": r(C.border_ft[S].sum(), 0),
                             "this_run": stats(costs.odds[S]),
                             "j7": stats(j7_odds[S]) if j7_odds is not None else None}
            rec = {
                "id": inc["id"], "lots_count": int(len(I)), "table": inc["table"], "solved": inc["solved"],
                "quarter": q, "blocks": blocks(I), "lots": [int(i) for i in I], "blklots": [blklot[i] for i in I],
                "quarter_solve_agrees_with_join": r(agree),
                "data_error_lots": int(errors[I].sum()),
                "changed_set": {"lots": int(len(D)), "wrong_scored_lots": int(wrong_any[D].sum()),
                                "lots_outside_quarter": int((C.quarter[D] != qi).sum()),
                                "other_incidents": sorted({int(v) for v in inc_of[D] if v and v != inc["id"]}),
                                "codes": [[t, s, int(n)] for (t, s), n in
                                          Counter(zip(C.table[D], names[x0[D]])).most_common(6)]},
                "energy": {
                    "sign": "table's answer minus solved answer; positive favours the solved (wrong) answer",
                    "changed_set": {k: r(v) for k, v in sD.items() if k != "by_label"},
                    "incident_only": {k: r(v) for k, v in sI.items() if k != "by_label"},
                    "cheapest_right_on_incident": {k: r(v) for k, v in sR.items() if k != "by_label"},
                    "incident_lots_without_a_table_code": int((~okI).sum()),
                },
                "groups": {"changed_set": {k: r(v) for k, v in groups(sD).items()},
                           "incident_only": {k: r(v) for k, v in groups(sI).items()},
                           "cheapest_right_on_incident": {k: r(v) for k, v in groups(sR).items()}},
                "driving_term": dD, "driving_amount": amtD,
                "driving_term_incident_only": dI, "driving_amount_incident_only": amtI,
                "driving_term_cheapest_right": dR, "driving_amount_cheapest_right": amtR,
                "cheapest_right_on_incident": right_info,
                "solved_is_minimum": bool(best >= -1e-6),
                "search_on_changed_set": ends,
                "lowest_found_change": r(best),
                "labels_moving_energy": top_labels,
                "labels_on_incident": {"count": len(on), "top_reading": dict(reads)},
                "walls": walls,
                "j7": None,
            }
            if j7 is not None:
                xj = x0.copy()
                xj[D] = costs.to_index(j7[D], x0[D])
                j7_right = r(np.mean(j7[I] == C.table[I]))
                j7w_right = r(np.mean(j7w[I] == C.table[I])) if j7w is not None else None
                rec["j7"] = {"incident_right": j7_right,
                             "codes_on_incident": [[c, int(n)] for c, n in Counter(j7[I].tolist()).most_common(3)],
                             "answer_energy_change_on_changed_set": r(energy_on(costs, xj, D) - e_solved),
                             "j7_walls_solve_incident_right": j7w_right,
                             "j7_walls_solve_codes_on_incident": ([[c, int(n)] for c, n in
                                                                   Counter(j7w[I].tolist()).most_common(3)]
                                                                  if j7w is not None else None),
                             "input_that_differs": j7_input(j7_right, j7w_right)}
            out.append(rec)
            print(f"#{rec['id']} {len(I)} {inc['table']} -> {inc['solved']} {q}: D {len(D)}; d(E) "
                  f"{sD['total']:+.1f} (labels {sD['labels']:+.1f}, moves {sD['placement']:+.1f}, far "
                  f"{sD['far_side']:+.1f}, walls {sD['walls']:+.1f}, tie {sD['tie_break']:+.2f}; check "
                  f"{sD['check']:.1e}) -> {dD} {amtD}; I only {sI['total']:+.1f} -> {dI}; search ends "
                  f"{ends['solved']['energy_change']:+.1f} / {ends['table']['energy_change']:+.1f}; right on I "
                  f"{sR['total']:+.1f}; J7 right "
                  f"{rec['j7']['incident_right'] if rec['j7'] else '-'}", flush=True)
    out.sort(key=lambda t: t["id"])
    CENSUS_DIR.mkdir(parents=True, exist_ok=True)
    doc = {"run": run, "min_lots": min_lots, "incidents": len(out), "lots": int(sum(t["lots_count"] for t in out)),
           "wrong_lots": int(wrong_any.sum()), "j7_walls_run": J7_WALLS_RUN if j7w is not None else None,
           "terms": {"labels": "label charges on the lot under each label's text (own lot: no placement options, "
                               "or the placement's stay option)",
                     "placement": "label charges on lots a label's placement moves it to (move options)",
                     "far_side": "street labels' far-side copies",
                     "tie_break": "1e-4 x each code's cheapest-path cost (energy.path_costs)",
                     "walls_no_line": "borders whose lots differ where the odds say no line (-odds)",
                     "walls_same_code_across_line": "borders whose lots share a code where the odds say a line"},
           "driving_term_rule": "search when the table's answer on the changed set costs less than the solved "
                                "answer, or the search confined to the changed set ends lower and gets most of the "
                                "incident right; else the group (labels; placement = moves + far side; walls; "
                                "tie-break) with the largest change favouring the solved answer on the changed set",
           "list": out}
    (CENSUS_DIR / "incidents.json").write_text(json.dumps(doc, indent=1) + "\n")
    notes = CENSUS_DIR / "notes.md"
    (CENSUS_DIR / "census.md").write_text(markdown(doc, notes.read_text() if notes.is_file() else ""))
    print(f"wrote {CENSUS_DIR / 'incidents.json'} and census.md: {len(out)} incidents, {doc['lots']:,} lots")


def _pct(v):
    return "–" if v is None else f"{v:.0%}"


def _p(st):
    return "–" if st is None else f"{st['median_p']:.2f}"


def summary(rows, key="driving_term") -> str:
    """counts of incidents and lots by driving term (`key`)"""
    by = defaultdict(lambda: [0, 0])
    for t in rows:
        by[t[key]][0] += 1
        by[t[key]][1] += t["lots_count"]
    return "; ".join(f"{k} {n} ({lots:,} lots)" for k, (n, lots) in sorted(by.items(), key=lambda kv: -kv[1][1]))


def markdown(doc, notes: str = "") -> str:
    """census.md: the counts, `notes` (a hand-written reading of them, kept in CENSUS_DIR/notes.md), the table of
    every incident and the walls and labels of the big ones"""
    rows = doc["list"]
    big = [t for t in rows if t["lots_count"] >= 100]
    mid = [t for t in rows if t["lots_count"] < 100]
    j7_big = [t for t in big if t["j7"] and t["j7"]["incident_right"] >= 0.5]
    j7w_big = [t for t in big if t["j7"] and (t["j7"]["j7_walls_solve_incident_right"] or 0) >= 0.5]
    not_min = [t for t in rows if not t["solved_is_minimum"]]
    out = [f"# Incidents of {doc['min_lots']}+ lots in the solve `{doc['run']}`", "",
           f"{doc['incidents']} incidents hold {doc['lots']:,} of the run's {doc['wrong_lots']:,} wrong lots "
           f"({len(big)} of 100+ lots hold {sum(t['lots_count'] for t in big):,}). Made by "
           "`pipeline/scripts/incidents/census.py` from the run's saved costs; the numbers per incident are in "
           "`incidents.json`.", ""]
    if notes:
        out += [notes.strip(), ""]
    out += ["## Counts", "",
            f"- **Driving term on the table's answer, 100+ lots:** {summary(big)}.",
            f"- **The same, 20–99 lots:** {summary(mid)}.",
            "- **Driving term on the cheapest answer right on the incident, 100+ lots:** "
            f"{summary(big, 'driving_term_cheapest_right')}; 20–99 lots: "
            f"{summary(mid, 'driving_term_cheapest_right')}.",
            f"- **Not the run's own minimum on D** (a lower energy found there): {len(not_min)} incidents "
            f"({sum(t['lots_count'] for t in not_min):,} lots): " + ", ".join(f"#{t['id']}" for t in not_min) + ".",
            f"- **J7** gets {len(j7_big)} of the {len(big)} big incidents right"
            + (": " + ", ".join(f"#{t['id']} ({t['j7']['input_that_differs']})" for t in j7_big) if j7_big else "")
            + f". This build's solve with J7's walls gets {len(j7w_big)} right"
            + (": " + ", ".join(f"#{t['id']}" for t in j7w_big) if j7w_big else "") + ".", "",
           "How to read the columns:",
           "- **D** is the changed set: the connected lots around the incident whose solved code differs from the "
           "table's (in the solve that held the incident's quarter out). Several incidents can share one D.",
           "- **dE** is the table's answer on D minus the solved answer, under the run's own costs; positive "
           "favours the solved (wrong) answer. Labels: label charges on the lot under the text. Placement: charges "
           "the placement moves elsewhere, plus street labels' far-side copies. Walls: borders drawn where the odds "
           "say no line, plus one code across a read line (the two parts in brackets).",
           "- **Driving term** (rule in incidents.json): `search` when an answer right on most of the incident costs "
           "less than the solved one (the table's on D, one the solver's own search confined to D finds, or the "
           "cheapest right on the incident); else the group that favours the wrong answer most on the table's "
           "answer.",
           "- **Min** is whether the solved answer is the lowest energy found on D (the table's answer, the "
           "search confined to D from either answer, and the cheapest answer right on the incident).",
           "- **Right on I** is the cheapest answer found that keeps the incident's lots at the table's codes and "
           "lets the rest of D and up to two rings of neighbours move (searched from the table's answer): its "
           "energy minus the solved answer's, the group that favours the wrong answer most against it, and the "
           "scored lots it changes that are wrong before and after. It leaves out costs of the table's answer "
           "beside the incident's cause, so it is the closer measure of what tipped the incident.",
           "- **J7** is the share of the incident J7 gets right; **J7 walls** the share this build's solve with "
           "J7's walls (on this build's labels and placement, `incidents-j7walls`) gets right.", "",
           "## The incidents", "",
           "| # | Lots | Table → solved | Q | Blocks | D lots (other incidents) | dE on D | Labels | Placement | "
           "Walls (no line / same code) | Driving term | Min | Right on I: dE, term, wrong lots before → after "
           "| J7 | J7 walls |",
           "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for t in sorted(rows, key=lambda t: (t["lots_count"] < 100, -t["lots_count"], t["id"])):
        e, g, cs = t["energy"]["changed_set"], t["groups"]["changed_set"], t["changed_set"]
        others = f" ({', '.join('#' + str(i) for i in cs['other_incidents'])})" if cs["other_incidents"] else ""
        out.append(
            f"| {t['id']} | {t['lots_count']:,} | {t['table']} → {t['solved']} | {t['quarter']} | "
            f"{', '.join(b for b, _ in t['blocks'][:3])} | {cs['lots']:,}{others} | {e['total']:+.1f} | "
            f"{g['labels']:+.1f} | {g['placement']:+.1f} | {g['walls']:+.1f} ({e['walls_no_line']:+.1f} / "
            f"{e['walls_same_code_across_line']:+.1f}) | {t['driving_term']} ({t['driving_amount']:+.1f}) | "
            f"{'yes' if t['solved_is_minimum'] else 'no'} | "
            f"{t['energy']['cheapest_right_on_incident']['total']:+.1f}, {t['driving_term_cheapest_right']} "
            f"({t['driving_amount_cheapest_right']:+.1f}), {t['cheapest_right_on_incident']['wrong_scored_before']:,}"
            f" → {t['cheapest_right_on_incident']['wrong_scored_after']:,} | "
            f"{_pct(t['j7']['incident_right'] if t['j7'] else None)} | "
            f"{_pct(t['j7']['j7_walls_solve_incident_right'] if t['j7'] else None)} |")
    out += ["", "## The walls and labels of the 100+ lot incidents", "",
            "Walls on the borders touching the incident: the table lines the solve leaves out, and the lines it "
            "draws that the table lacks (pairs; median wall probability under this run's odds and J7's). Labels: "
            "those whose text lies on the incident, by top reading; the label that favours the wrong answer most.",
            "",
            "| # | Table lines left out: pairs, ft, p here, p J7 | Lines drawn not in table: pairs, ft, p here, "
            "p J7 | Labels on the incident | Label favouring the wrong answer most | J7: input that differs |",
            "|---|---|---|---|---|---|"]
    for t in sorted(big, key=lambda t: -t["lots_count"]):
        w = t["walls"]
        lo, ld = w["table_lines_left_out"], w["lines_drawn_not_in_table"]
        on = t["labels_on_incident"]
        fav = [lab for lab in t["labels_moving_energy"] if lab["favours_solved_by"] > 0]
        fav_s = (f"{fav[0]['top_reading']} {fav[0]['trust']:.2f} on {fav[0]['text_lot']} (table "
                 f"{fav[0]['text_lot_table'] or '–'}), {'+'.join(fav[0]['charges'])}, "
                 f"{fav[0]['favours_solved_by']:+.1f}") if fav else "none"
        out.append(f"| {t['id']} | {lo['pairs']}, {lo['ft']:,.0f}, {_p(lo['this_run'])}, {_p(lo['j7'])} | "
                   f"{ld['pairs']}, {ld['ft']:,.0f}, {_p(ld['this_run'])}, {_p(ld['j7'])} | {on['count']}: "
                   + (", ".join(f"{k} {v}" for k, v in on["top_reading"].items()) or "–")
                   + f" | {fav_s} | {(t['j7'] or {}).get('input_that_differs') or '–'} |")
    return "\n".join(out) + "\n"


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--run", default=runs.FINAL_RUN)
    ap.add_argument("--min-lots", type=int, default=20)
    ap.add_argument("--markdown-only", action="store_true",
                    help="rewrite census.md from incidents.json (and notes.md, when there is one)")
    args = ap.parse_args()
    if args.markdown_only:
        notes = CENSUS_DIR / "notes.md"
        doc = json.loads((CENSUS_DIR / "incidents.json").read_text())
        (CENSUS_DIR / "census.md").write_text(markdown(doc, notes.read_text() if notes.is_file() else ""))
    else:
        main(args.run, args.min_lots)
