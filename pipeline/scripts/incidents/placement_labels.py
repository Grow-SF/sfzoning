"""The labels behind the census's placement- and label-driven incidents: which printed labels tip each one, what each
reader read there, where the placement puts each label, and whether the hand leader moves, the other hand fixes, a
pen stroke, the owner's code-check answers and the code reader's saved posterior cover it.

    uv run python scripts/incidents/placement_labels.py labels [ID ...]
    uv run python scripts/incidents/placement_labels.py picture ID [--window X0 Y0 X1 Y1]
    uv run python scripts/incidents/placement_labels.py oracle [ID ...] [--all] [--rings 2]
    uv run python scripts/incidents/placement_labels.py try ID EDIT [EDIT ...]
    uv run python scripts/incidents/placement_labels.py affected VARIANT [...] [--unit area|district] [--max-lots N]
    uv run python scripts/incidents/placement_labels.py street

labels   per incident (default IDS: the incidents whose cheapest right answer the census drives by labels or placement,
         with #5, #10 and #24): the change from the solved answer to the census's cheapest answer right on the
         incident (answers/cheapest-right-<id>.npz), split per label and per charge (own lot, stay, move, far-side
         copy; positive favours the wrong answer), over every changed lot and over the incident's own lots.  Per label
         that moves it: its print (sheet, page px, map ft, in a street), every reader's read (Sol, Terra, the third
         read and its leader flag, the re-check, the turned pass), its readings and trusts, the codes of the lots
         under its text box, its placement options (lot, table code, solved code, right code, chance), its far-side
         lot, the placement fit's answer for it (a hand move, the table's, none), the hand rows of label_fixes.csv
         within MATCH_PX of it, the pen strokes starting at it, the owner's code-check answer and the code reader's
         saved posterior (CODEREADER_RECORDS, its top three).  A rule classes each charge (CLASSES).  Writes
         OUT/labels.json and OUT/labels.txt.
picture  the incident's placement at 2x the grid: page ink grey, the table's borders green and the solve's blue, the
         incident's lots yellow and the rest of its changed set orange; each label's options as arrows from its text
         (stay a ring, move a solid arrow, far-side copy a dashed arrow; green when the option lot's table code is the
         label's top reading, red when not; width by chance); the labels that tip the incident boxed with their
         share.  And OUT/pictures/<id>_crops.png: the code reader's crop of each tipping label (the sheet's page px,
         turned to read left to right) with every reader's read under it.
oracle   ceilings without a full solve, the walls agent's protocol (same_code_lines.py oracle): per incident the
         solver's own search confined to its changed set D and RINGS rings of neighbours, from the solved answer and
         from the table's answer on that set, the lower energy kept; scored lots right in the set, with the quarter's
         label costs changed by each variant (quarter_variants).  Oracles (read the table; they bound a fix, never
         feed one): far (drop a far-side copy whose lot's table code is not the reading), place / place_street /
         place_hand (a label charged on its placement answer alone: every label, street labels, hand moves), read /
         read_codereader_right (a reading no nearby lot carries replaced by the table's code under it), far+place,
         all.  Table-free variants: nofar (no far-side copies), far_split (the far lot shares the stay's chance, the
         label charged once), add_codereader (the pre-fix code reader's posterior as more readings), fixes (the
         label_fixes.csv rows as input: code fixes at trust 0.99, deletions, leader moves charged on their box),
         fixes+codecheck (with the owner's code-check answers), one_lot (OneLot: each label stands for one lot,
         chosen with the codes).  --all: every census incident.
try      one incident's confined search with hand-picked label edits (place, drop, read, nofar), each alone and
         together: which label an incident hangs on.
affected every walled area (or solved district, --unit district) of a quarter holding a lot whose label costs a
         variant changes: the confined search with and without it; lots gained and lost, not only on the incidents.
street   every label printed in a street, city-wide: how often its top reading is the table code of the lot under its
         text (side A), of its far-side lot (side B), of both, of neither.

Reads only; writes into data/solver/experiments/incidents-placement/.  CPU, one thread."""
import argparse
import csv
import json
import math
import pickle
from collections import Counter, defaultdict
from dataclasses import replace

import cv2
import numpy as np
from scipy.spatial import cKDTree

from common import (CENSUS_DIR, KINDS, TIE_BREAK, changed_set, energy_on, label_charges, load_costs, mask, ring,
                    run_codes, search_on, split)
from sfzmaps import config
from sfzmaps.solver import energy as E
from sfzmaps.solver import grid, paths, runs
from sfzmaps.solver import placement as PL
from sfzmaps.solver.city import city
from sfzmaps.solver.codes import canon, table_vocabulary
from sfzmaps.solver.labels import readings
from sfzmaps.solver.placement import LINK_LOT_REACH, LINK_REACH, LINK_WINDOW, MATCH_PX, leader_moves, load_options
from sfzmaps.solver.trust import load_prints

OUT = runs.EXPERIMENTS_DIR / "incidents-placement"
IDS = (5, 10, 11, 19, 20, 24, 25, 26, 27, 28, 30, 31, 32, 35, 40, 42)
SHOW = 0.05                 # a label is listed when it moves the energy by at least this (either way)
SCRATCH = ("/private/tmp/claude-501/-Users-sbuss-workspace-sfzoning/eb18d2fc-e22a-48a6-afbc-57c31cd3e134/scratchpad/"
           "codereader/task3/")
CODEREADER_RECORDS = SCRATCH + "full_cpu8.records"   # the pre-fix code reader's full run (evidence per label)
CODEREADER_PRIOR = SCRATCH + "prior.json"
CODEREADER_LABELS = SCRATCH + "labs.pkl"             # its labels: crops, every read, the owner's answers
NONE_MASS, OTHER_MASS = 0.02, 0.03                   # posterior.py's defaults at that run
READ_NAMES = (("sol", "sol_code"), ("terra", "terra_code"), ("third", "third_code"), ("recheck", "recheck_code"),
              ("turned", "turned_code"))
CLASSES = {
    "misread": "the top reading names a code no lot near the label carries (its options, far side, text box): "
               "the print was read wrong",
    "stays_on_other_code": "read right (a lot near it carries the reading) but charged on the lot under its text, "
                           "which carries another code",
    "moved_wrong": "a move option onto a lot whose right code is not the label's reading",
    "far_side_wrong": "a street label's far-side copy onto a lot whose right code is not its reading",
    "right_charge": "the charge is on a lot whose right code is the reading (it pulls the wrong way only through "
                    "the walls or by the share it spends elsewhere)",
}


# --- inputs ---------------------------------------------------------------------------------------------------------

class _Blob:
    """stands for the code reader's classes when its pickles are read (its module is being edited tonight)"""
    def __init__(self, *a, **k):
        pass

    def __setstate__(self, s):
        if isinstance(s, tuple):
            s = s[1] if s[0] is None else {**s[0], **(s[1] or {})}
        self.__dict__.update(s)


class _Unpickler(pickle.Unpickler):
    def find_class(self, mod, name):
        return _Blob if mod.startswith("sfzmaps") else super().find_class(mod, name)


def codereader() -> tuple[dict, dict]:
    """(print index -> the code reader's label: crop, reads, the owner's answer), (print index -> posterior over
    codes, "none", "other"), from the pre-fix run's saved records; empty when they are gone."""
    try:
        labs = _Unpickler(open(CODEREADER_LABELS, "rb")).load()
        prior = json.loads(open(CODEREADER_PRIOR).read())
    except FileNotFoundError:
        return {}, {}
    by_index = {lab.index: lab for lab in labs if lab.index >= 0}
    post = {}
    with open(CODEREADER_RECORDS, "rb") as fh:
        while True:
            try:
                key, _, _, ev, _ = _Unpickler(fh).load()
            except EOFError:
                break
            if key[0] < 0:
                continue
            by_code = defaultdict(list)
            for f, v in ev.forms.items():
                by_code[prior["forms"][f]].append(float(v[0]))
            mass = 1 - NONE_MASS - OTHER_MASS
            logs = {c: math.log(mass * p) + float(np.logaddexp.reduce(by_code[c])) - math.log(len(by_code[c]))
                    for c, p in prior["codes"].items() if by_code.get(c) and p > 0}
            logs["none"] = math.log(NONE_MASS)
            logs["other"] = math.log(OTHER_MASS) + float(ev.other[0])
            m = max(logs.values())
            z = m + math.log(sum(math.exp(v - m) for v in logs.values()))
            post[key[0]] = {h: math.exp(v - z) for h, v in logs.items()}
    return by_index, post


def placement_rows() -> dict:
    """placement._options() (the placement fit's rows: per label its quarter, whether a hand move answers it, its
    options [(features, is the answer, lot)]), cached in OUT"""
    f = OUT / "placement_rows.pkl"
    if f.is_file():
        return pickle.loads(f.read_bytes())
    rows = {r[0]: r for r in PL._options()}
    OUT.mkdir(parents=True, exist_ok=True)
    f.write_bytes(pickle.dumps(rows))
    return rows


def hand_rows() -> list[dict]:
    with open(runs.LABEL_FIXES, newline="") as fh:
        return list(csv.DictReader(fh))


def text_codes(pr, j, C, transforms) -> dict:
    """the table codes of the lots under label j's text box (7 x 3 points over Terra's box, or its point), share by
    code ("street" where no lot)"""
    from sfzmaps.transform import SheetTransform
    row = pr.iloc[j]
    sheet = str(row["sheet"])
    if sheet not in transforms:
        folder = config.DATA_DIR / "warp" / paths.SERIES / sheet
        transforms[sheet] = SheetTransform.from_json(json.loads((folder / "transform.json").read_text()),
                                                     folder=folder)
    if np.isfinite(row["box_x0"]):
        xs = np.linspace(row["box_x0"], row["box_x1"], 7)
        ys = np.linspace(row["box_y0"], row["box_y1"], 3)
        pts = np.array([[x, y] for x in xs for y in ys], float)
    else:
        pts = np.array([[row["page_x"], row["page_y"]]], float)
    cr = np.floor(grid.to_grid(transforms[sheet].to_world(pts))).astype(int)
    lot = C.lot_raster
    got = Counter()
    for c, r in cr:
        v = lot[r, c] if 0 <= r < lot.shape[0] and 0 <= c < lot.shape[1] else -1
        got["street" if v < 0 else str(C.table[v]) or "?"] += 1
    n = sum(got.values())
    return {k: round(v / n, 2) for k, v in got.most_common()}


class Strokes:
    """the pen strokes the placement reads for leader lines (strokes.npz): the ones with an end within LINK_REACH
    of a label, their far end, width and length, and the lots at the far end"""
    def __init__(self):
        S = np.load(paths.LABELS_DIR / "strokes.npz")
        self.xy, self.off, self.w, self.length = S["xy"].astype(float), S["off"], S["wmed"], S["length"]
        self.end1, self.end2 = self.xy[self.off[:-1]], self.xy[self.off[1:] - 1]
        self.n = len(self.w)
        self.tree = cKDTree(np.r_[self.end1, self.end2])

    def links(self, xy, C, lot_tree) -> list[dict]:
        out = []
        grown = C.grown_raster
        for s_ in self.tree.query_ball_point(xy, LINK_REACH):
            s, end = s_ % self.n, s_ // self.n
            far = self.end2[s] if end == 0 else self.end1[s]
            lots = set()
            c_, r_ = C.px(far)
            win = grown[max(r_ - LINK_WINDOW, 0):r_ + LINK_WINDOW + 1, max(c_ - LINK_WINDOW, 0):c_ + LINK_WINDOW + 1]
            for i in lot_tree.query_ball_point(far, LINK_LOT_REACH):
                if (win == i).any():
                    lots.add(int(i))
            out.append({"stroke": int(s), "length_ft": round(float(self.length[s]), 1),
                        "width_ft": round(float(self.w[s]), 1), "far_end": [round(float(v)) for v in far],
                        "far_lots": sorted(lots)})
        return out


# --- the per-label table --------------------------------------------------------------------------------------------

def right_answer(costs, ident) -> np.ndarray:
    z = np.load(CENSUS_DIR / "answers" / f"cheapest-right-{ident}.npz")
    x = costs.codes.copy()
    x[z["lots"]] = [costs.index[canon(c)] for c in z["codes"]]
    return x


def classify(kind, top, right_here, near_codes) -> str:
    if right_here == top:
        return "right_charge"
    if kind == "far":
        return "far_side_wrong"
    if kind == "move":
        return "moved_wrong"
    return "stays_on_other_code" if top in near_codes else "misread"


def labels_cmd(ids) -> None:
    C = city()
    doc = json.loads((CENSUS_DIR / "incidents.json").read_text())
    run = doc["run"]
    pr = load_prints()
    blk = C.lots["blklot"].astype(str).values
    rows = placement_rows()
    fixes = hand_rows()
    moves = leader_moves()
    cr_labels, cr_post = codereader()
    strokes = Strokes()
    lot_tree = cKDTree(C.centroid)
    solved = run_codes(run)
    transforms = {}
    cache = {}
    out, text = [], []
    for ident in ids:
        rec = next(t for t in doc["list"] if t["id"] == ident)
        q = rec["quarter"]
        if q not in cache:
            costs = load_costs(run, q)
            ch = label_charges(run, q, costs)
            rd = readings(q, table_vocabulary(C.lots))
            S = np.load(paths.LABELS_DIR / f"streetside_{q}.npz")
            cache[q] = (costs, ch, rd, load_options(q, runs.solve_dir(run)), S)
        costs, ch, rd, options, SS = cache[q]
        names = np.array(costs.names)
        x0 = costs.codes
        x1 = right_answer(costs, ident)
        moved = np.flatnonzero(x1 != x0)
        s = split(costs, ch, x0, x1, moved)
        per = defaultdict(lambda: defaultdict(float))
        for lab, k, v in zip(*s["by_label"]):
            per[int(lab)][KINDS[k]] += float(v)
        inI = mask(C.n, rec["lots"])          # the same, from the charges on the incident's own lots alone
        on_inc = defaultdict(lambda: defaultdict(float))
        for j in np.flatnonzero(inI[ch.lot]):
            lot = ch.lot[j]
            if x1[lot] != x0[lot]:
                on_inc[int(ch.label[j])][KINDS[ch.kind[j]]] += float(ch.cost[j, x1[lot]] - ch.cost[j, x0[lot]])
        D = changed_set(costs, np.array(rec["lots"]))
        inc = {"id": ident, "lots": rec["lots_count"], "table": rec["table"], "solved": rec["solved"], "quarter": q,
               "blocks": [b for b, _ in rec["blocks"][:4]], "changed_set": int(len(D)),
               "right_answer_lots_changed": int(len(moved)),
               "split": {k: round(v, 3) for k, v in s.items() if k != "by_label"}, "labels": []}
        text.append(f"\n#{ident}: {rec['lots_count']} lots {rec['table']} -> {rec['solved']} ({q}; blocks "
                    f"{', '.join(inc['blocks'])}); cheapest right answer changes {len(moved)} lots: total "
                    f"{s['total']:+.2f} = own {s['labels_no_options']:+.2f} + stay {s['labels_stay']:+.2f} + move "
                    f"{s['placement']:+.2f} + far {s['far_side']:+.2f} + walls {s['walls']:+.2f} (no line "
                    f"{s['walls_no_line']:+.2f}, same code {s['walls_same_code_across_line']:+.2f})")
        order = sorted(per, key=lambda lab: -sum(per[lab].values()))
        inc["on_incident"] = {k: round(sum(v.get(k, 0.0) for v in on_inc.values()), 3) for k in KINDS}
        for lab in order:
            tot = sum(per[lab].values())
            tot_i = sum(on_inc[lab].values()) if lab in on_inc else 0.0
            if abs(tot) < SHOW and abs(tot_i) < SHOW:
                continue
            row = pr.iloc[lab]
            under = int(row["lab_lot"])
            rr = np.flatnonzero((rd.label == lab) & ~rd.far)
            reads_ = sorted({(names[rd.code[i]], round(float(rd.trust[i]), 3)) for i in rr}, key=lambda t: -t[1])
            jj = np.flatnonzero(ch.label == lab)
            top = str(names[ch.top[jj[0]]])
            trust = float(ch.top_trust[jj[0]])
            lotB = int(SS["lotB"][lab]) if (SS["onst"][lab] and not SS["isfw"][lab]) else -1
            opts = [(int(lot), float(pi), bool(st)) for lot, pi, st in options.get(lab, [])]
            near = {str(C.table[lot]) for lot, _, _ in opts} | ({str(C.table[lotB])} if lotB >= 0 else set())
            tc = text_codes(pr, lab, C, transforms)
            near |= set(tc) - {"street"}
            charges = []
            for j in jj:
                k = KINDS[ch.kind[j]]
                lot = int(ch.lot[j])
                v = float(per[lab].get(k, 0.0))
                charges.append({"kind": k, "lot": blk[lot], "lot_index": lot, "chance": round(float(ch.pi[j]), 3),
                                "table": str(C.table[lot]), "solved": str(names[x0[lot]]),
                                "right": str(names[x1[lot]]),
                                "class": classify(k, top, str(names[x1[lot]]), near)})
            # the charge kinds' contributions, each classed by the charge of that kind holding most chance
            by_kind = {}
            for k, v in per[lab].items():
                if abs(v) < 1e-4:
                    continue
                cs = [c for c in charges if c["kind"] == k]
                worst = max(cs, key=lambda c: (c["class"] != "right_charge", c["chance"])) if cs else None
                by_kind[k] = {"favours_wrong_by": round(v, 3), "class": worst["class"] if worst else "?"}
            prow = rows.get(lab)
            answer = None
            if prow is not None and prow[4]:
                ai = int(np.flatnonzero([f[1] for f in prow[3]])[0])
                answer = {"by_hand": bool(prow[2]), "option": "stay" if ai == 0 else "move",
                          "lot": blk[int(prow[3][ai][2])], "table": str(C.table[int(prow[3][ai][2])])}
            hm = []
            for p0, p1, rad, fc in moves.get(str(row["sheet"]), []):
                d = float(np.hypot(row["x"] - p0[0], row["y"] - p0[1]))
                if d <= rad * 3:
                    hm.append({"code": fc, "distance_ft": round(d, 1), "radius_ft": round(rad, 1),
                               "matches": bool(d <= rad and fc == top)})
            fx = [{"code": f["code"], "fix_code": f["fix_code"], "fix_point": bool((f["fix_x"] or "").strip()),
                   "page_px": round(float(np.hypot(float(f["x"]) - row["page_x"], float(f["y"]) - row["page_y"]))),
                   "note": f["note"]}
                  for f in fixes if f["sheet"] == row["sheet"]
                  and np.hypot(float(f["x"]) - row["page_x"], float(f["y"]) - row["page_y"]) <= MATCH_PX]
            links = strokes.links([row["x"], row["y"]], C, lot_tree)
            for li in links:
                li["far_codes"] = sorted({str(C.table[i]) for i in li["far_lots"]})
                li["far_lots"] = [blk[i] for i in li["far_lots"]][:6]
            cl = cr_labels.get(lab)
            post = cr_post.get(lab)
            item = {
                "label": lab, "favours_wrong_by": round(tot, 3), "by_kind": by_kind,
                "on_incident": round(tot_i, 3),
                "on_incident_by_kind": {k: round(v, 3) for k, v in on_inc.get(lab, {}).items() if abs(v) >= 1e-4},
                "sheet": str(row["sheet"]), "page_px": [round(float(row["page_x"])), round(float(row["page_y"]))],
                "map_ft": [round(float(row["x"])), round(float(row["y"]))], "print_kind": int(row["kind"]),
                "in_street": bool(row["street"]),
                "reads": {n: str(row[c]) for n, c in READ_NAMES if str(row[c])},
                "sol_confidence": None if not np.isfinite(row["sol_confidence"]) else
                round(float(row["sol_confidence"]), 2),
                "third_confidence": None if not np.isfinite(row["third_confidence"]) else
                round(float(row["third_confidence"]), 2),
                "third_leader": bool(row["third_leader"]),
                "readings": reads_, "top": top, "trust": round(trust, 3),
                "under": {"lot": blk[under] if under >= 0 else "-",
                          "table": str(C.table[under]) if under >= 0 else "-",
                          "solved": str(names[x0[under]]) if under >= 0 else "-",
                          "right": str(names[x1[under]]) if under >= 0 else "-"},
                "text_box_codes": tc,
                "far_lot": None if lotB < 0 else {"lot": blk[lotB], "table": str(C.table[lotB]),
                                                  "solved": str(names[x0[lotB]]), "right": str(names[x1[lotB]])},
                "options": [{"lot": blk[lot], "chance": round(pi, 3), "stay": st, "table": str(C.table[lot]),
                             "solved": str(names[x0[lot]]), "right": str(names[x1[lot]])}
                            for lot, pi, st in sorted(opts, key=lambda t: -t[1])[:6]],
                "charges": charges,
                "placement_answer": answer,
                "hand_moves_near": hm, "label_fixes_rows": fx,
                "strokes": links,
                "owner_answer": None if cl is None or cl.answer is None else cl.answer,
                "codereader": None if post is None else
                [(h, round(p, 3)) for h, p in sorted(post.items(), key=lambda t: -t[1])[:3]],
            }
            inc["labels"].append(item)
            kinds_txt = ", ".join(f"{k} {v['favours_wrong_by']:+.2f} [{v['class']}]" for k, v in by_kind.items())
            text.append(
                f"  label {lab} {tot:+.2f} (on the incident's lots {tot_i:+.2f}; {kinds_txt}); top {top} "
                f"{trust:.2f}; readings {reads_}; reads "
                f"{item['reads']} (Sol {item['sol_confidence']}, third {item['third_confidence']}"
                f"{', leader' if item['third_leader'] else ''}); in street {item['in_street']}; under "
                f"{item['under']}; text box {tc}; far {item['far_lot']}")
            opts_txt = [(o["lot"], o["table"], o["solved"], o["right"], o["chance"], "stay" if o["stay"] else "move")
                        for o in item["options"]]
            text.append(f"      options {opts_txt}")
            text.append(f"      placement answer {answer}; hand moves near {hm}; label_fixes {fx}; strokes "
                        f"{[(li['length_ft'], li['width_ft'], li['far_codes']) for li in links]}; owner "
                        f"{item['owner_answer']}; code reader {item['codereader']}")
        out.append(inc)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "labels.json").write_text(json.dumps({"run": run, "classes": CLASSES, "incidents": out}, indent=1) + "\n")
    (OUT / "labels.txt").write_text("\n".join(text) + "\n")
    print("\n".join(text))


# --- the oracles ----------------------------------------------------------------------------------------------------

def label_U(costs, rd, options, n, K, drop_far=frozenset(), place=None, read=None, far_split=None, extra=None,
            drop=frozenset(), read_trust=None):
    """U as energy.label_costs builds it (without the tie-break), with the changes of an oracle or a variant:
    `drop_far` the labels whose far-side copies are dropped; `place` label -> lot: the label charged on that lot
    alone (chance 1); `read` label -> code index: the label's readings replaced by that code at its top trust;
    `far_split` label -> far lot: the far-side copy is dropped and the label's stay chance is shared evenly between
    the lot under its text and the far lot (the label charged once); `extra` label -> {code index: trust}: more
    readings (each code's trust the larger of the two); `drop` labels left out altogether; `read_trust`: the trust
    of a `read` reading (default the label's own top trust)."""
    U = np.zeros((n, K))
    groups = {}
    for r in range(len(rd.lot)):
        groups.setdefault((int(rd.label[r]), bool(rd.far[r])), []).append(r)
    for (lab, far), rows in groups.items():
        if lab in drop or (far and (lab in drop_far or (far_split is not None and lab in far_split))):
            continue
        T = np.zeros(K)
        for r in rows:
            T[rd.code[r]] = max(T[rd.code[r]], rd.trust[r])
        if extra is not None and lab in extra and not far:
            for k_, t_ in extra[lab].items():
                T[k_] = max(T[k_], t_)
        if read is not None and lab in read:
            t_ = T.max() if read_trust is None else read_trust
            T = np.zeros(K)
            T[read[lab]] = t_
        top = int(np.argmax(T))
        t = np.clip(T[top], 1e-3, 1 - 1e-3)
        qq = 1 / (1 + np.exp(-(E.TOP_READING_FIT[0] + E.TOP_READING_FIT[1] * np.log(t / (1 - t)))))
        rest = T.copy()
        rest[top] = 0.0
        rest = (rest + 1.0 / K) / (rest.sum() + 1.0)
        P = qq * np.eye(K)[top] + (1 - qq) * rest
        own = int(rd.lot[rows[0]])
        where = [(own, 1.0)]
        if not far and lab in options:
            stay = [o for o in options[lab] if o[2]]
            if stay and stay[0][0] == own:
                where = [(o[0], o[1]) for o in options[lab]]
        if not far and place is not None and lab in place:
            where = [(place[lab], 1.0)]
        if not far and far_split is not None and lab in far_split:
            w2 = []
            for lot, pi in where:
                w2 += [(lot, pi / 2), (far_split[lab], pi / 2)] if lot == own else [(lot, pi)]
            where = w2
        for lot, pi in where:
            U[lot] -= np.log(pi * P + (1 - pi) / K)
    return U


def quarter_variants(q, C, run, rows, cr_labels, cr_post, only=None, log=print):
    """Quarter q's label costs U under each oracle and variant (the module's list; `only`: a subset of their names),
    with the quarter's saved costs and code names: (costs, names, {variant: U}); "_onelot" holds the one-lot
    candidate (OneLot, tie-break).  Checks that the plain rebuild gives the saved U."""
    costs = load_costs(run, q)
    ch = label_charges(run, q, costs)
    rd = readings(q, table_vocabulary(C.lots))
    options = load_options(q, runs.solve_dir(run))
    names = np.array(costs.names)
    K, n = len(names), C.n
    top = {int(lab): int(t) for lab, t in zip(ch.label, ch.top)}
    tie = ch.tie
    base = label_U(costs, rd, options, n, K) + tie
    base -= base.min(1, keepdims=True)
    err = float(np.abs(base - costs.U).max())
    if err > 1e-3:
        raise ValueError(f"{q}: the rebuilt U differs from the saved one by {err:.2g}")
    street_of = load_prints()["street"].to_numpy(bool)
    # far-side oracle: drop a copy whose lot's table code is not the label's top reading
    far_lab = {int(lab): int(lot) for lab, lot, f in zip(rd.label, rd.lot, rd.far) if f}
    drop_far = frozenset(lab for lab, lot in far_lab.items()
                         if C.known[lot] and str(C.table[lot]) != str(names[top[lab]]))
    # placement oracle: a label the placement fit has an answer for (a hand move, or the table's) is charged on
    # that lot alone
    place, place_hand = {}, {}
    for lab, r in rows.items():
        if lab in options and r[4]:
            ai = int(np.flatnonzero([f[1] for f in r[3]])[0])
            place[lab] = int(r[3][ai][2])
            if r[2]:
                place_hand[lab] = place[lab]
    # misread oracle: a label whose top reading no lot near it carries (no option, its far lot, its own lot)
    # gets the table's code of the lot under its text
    read, read_cr = {}, {}
    for lab in top:
        under = int(load_prints_lab_lot()[lab])
        if under < 0 or not C.known[under]:
            continue
        near = {str(C.table[l_]) for l_, _, _ in options.get(lab, [])} | {str(C.table[under])}
        if lab in far_lab:
            near.add(str(C.table[far_lab[lab]]))
        c = str(names[top[lab]])
        if c not in near and canon(str(C.table[under])) in costs.index:
            read[lab] = costs.index[canon(str(C.table[under]))]
            post = cr_post.get(lab)
            if post:
                best = max((h for h in post if h not in ("none", "other")), key=post.get, default=None)
                if best and canon(best) == canon(str(C.table[under])):
                    read_cr[lab] = read[lab]
    split_far = {lab: lot for lab, lot in far_lab.items()}
    extra = {}
    cr_index = {canon(nm): i for i, nm in enumerate(names)}
    for lab in top:
        post = cr_post.get(lab)
        if post:
            ex = {cr_index[canon(h)]: p for h, p in post.items() if canon(h) in cr_index and p >= 0.05}
            if ex:
                extra[lab] = ex
    fix_read, fix_drop, fix_place = human_fixes(costs, C)
    people_read = human_fixes(costs, C, people_only=True)[0]
    cc_read, cc_drop = {}, set()
    for lab, cl in cr_labels.items():
        ans = getattr(cl, "answer", None)
        if ans is None:
            continue
        if ans == "":
            cc_drop.add(lab)
        elif canon(ans) in costs.index:
            cc_read[lab] = costs.index[canon(ans)]
    variants = {"none": {}, "far": {"drop_far": drop_far}, "nofar": {"drop_far": frozenset(far_lab)},
                "far_split": {"far_split": split_far}, "add_codereader": {"extra": extra},
                "place": {"place": place},
                "place_street": {"place": {k: v for k, v in place.items() if street_of[k]},
                                 "drop_far": drop_far},
                "place_hand": {"place": place_hand}, "read": {"read": read},
                "read_codereader_right": {"read": read_cr},
                "far+place": {"drop_far": drop_far, "place": place},
                "fixes": {"read": fix_read, "read_trust": 0.99, "drop": frozenset(fix_drop), "place": fix_place,
                          "drop_far": frozenset(fix_place)},
                "hand_place": {"place": fix_place, "drop_far": frozenset(fix_place)},
                "fixes_people": {"read": people_read, "read_trust": 0.99, "drop": frozenset(fix_drop),
                                 "place": fix_place, "drop_far": frozenset(fix_place)},
                "fixes+codecheck": {"read": {**fix_read, **cc_read}, "read_trust": 0.99,
                                    "drop": frozenset(fix_drop | cc_drop), "place": fix_place,
                                    "drop_far": frozenset(fix_place)},
                "all": {"drop_far": drop_far, "place": place, "read": read}}
    Us = {}
    for nm, kw in variants.items():
        if only is not None and nm not in only:
            continue
        U = label_U(costs, rd, options, n, K, **kw) + tie
        U -= U.min(1, keepdims=True)
        Us[nm] = U
    log(f"{q}: far copies dropped {len(drop_far)} of {len(far_lab)}; placed {len(place)} (by hand "
        f"{len(place_hand)}); misreads {len(read)} (code reader right on {len(read_cr)}); hand fixes: codes "
        f"{len(fix_read)}, deletions {len(fix_drop)}, leader moves {len(fix_place)}")
    Us["_onelot"] = (OneLot(costs, rd, options, K), tie)
    return costs, names, Us


class OneLot:
    """The candidate where each label stands for ONE lot, chosen with the codes: its options are the placement's
    (stay and moves, with their chances) and, for a street label, the far lot as one more option sharing the stay's
    chance evenly (no separate far-side copy).  A label's cost is min over its options of -log chance - log P(code
    there): a min-mixture, the label charged once.  `answer` alternates the solver's own search (each label charged
    on its chosen lot alone) with each label's best option (hard EM), from the solved answer and from the table's,
    and keeps the lower min-mixture energy.  Table-free: it bounds what charging a label once as a choice of lot
    would do, short of a full solve."""

    def __init__(self, costs, rd, options, K):
        self.costs, self.rd, self.options, self.K = costs, rd, options, K
        far_lab = {int(lab): int(lot) for lab, lot, f in zip(rd.label, rd.lot, rd.far) if f}
        self.far_lab = frozenset(far_lab)
        self.P, self.opts = {}, {}
        groups = {}
        for r in range(len(rd.lot)):
            if not rd.far[r]:
                groups.setdefault(int(rd.label[r]), []).append(r)
        for lab, rows in groups.items():
            T = np.zeros(K)
            for r in rows:
                T[rd.code[r]] = max(T[rd.code[r]], rd.trust[r])
            top = int(np.argmax(T))
            t = np.clip(T[top], 1e-3, 1 - 1e-3)
            qq = 1 / (1 + np.exp(-(E.TOP_READING_FIT[0] + E.TOP_READING_FIT[1] * np.log(t / (1 - t)))))
            rest = T.copy()
            rest[top] = 0.0
            rest = (rest + 1.0 / K) / (rest.sum() + 1.0)
            self.P[lab] = qq * np.eye(K)[top] + (1 - qq) * rest
            own = int(rd.lot[rows[0]])
            where = [(own, 1.0)]
            stay = [o for o in options.get(lab, []) if o[2]]
            if stay and stay[0][0] == own:
                where = [(o[0], o[1]) for o in options[lab]]
            if lab in far_lab:
                w2 = []
                for lot, pi in where:
                    w2 += [(lot, pi / 2), (far_lab[lab], pi / 2)] if lot == own else [(lot, pi)]
                where = w2
            self.opts[lab] = (np.array([w[0] for w in where], int),
                              -np.log(np.maximum(np.array([w[1] for w in where]), 1e-12)))

    def choose(self, x, labs):
        return {lab: int(self.opts[lab][0][np.argmin(self.opts[lab][1] - np.log(self.P[lab][x[self.opts[lab][0]]]))])
                for lab in labs}

    def energy(self, x, S, labs, tie):
        C = city()
        walls = energy_on(replace(self.costs, U=np.zeros_like(self.costs.U)), x, S)
        lab_e = sum(float(np.min(self.opts[lab][1] - np.log(self.P[lab][x[self.opts[lab][0]]]))) for lab in labs)
        return walls + lab_e + float(tie[S, x[S]].sum())

    def answer(self, S, tie, rounds=4):
        C = city()
        inS = mask(C.n, S)
        labs = [lab for lab, (lots, _) in self.opts.items() if inS[lots].any()]
        x0 = self.costs.codes
        xt = x0.copy()
        t = self.costs.table_codes[S]
        xt[S] = np.where(t >= 0, t, x0[S])
        best, be = x0, self.energy(x0, S, labs, tie)     # the solved answer competes too, as in the oracles
        for start in (x0, xt):
            x = start
            for _ in range(rounds):
                z = self.choose(x, list(self.opts))
                U = label_U(self.costs, self.rd, self.options, C.n, self.K, place=z, drop_far=self.far_lab) + tie
                U -= U.min(1, keepdims=True)
                y = search_on(replace(self.costs, U=U), x, S)
                if np.array_equal(y, x):
                    break
                x = y
            e = self.energy(x, S, labs, tie)
            if e < be - 1e-9:
                best, be = x, e
        return best, be - self.energy(x0, S, labs, tie)



def oracle_cmd(ids, rings: int, name: str = "", only=None) -> None:
    C = city()
    doc = json.loads((CENSUS_DIR / "incidents.json").read_text())
    run = doc["run"]
    rows = placement_rows()
    cr_labels, cr_post = codereader()
    by_q = defaultdict(list)
    for ident in ids:
        rec = next(t for t in doc["list"] if t["id"] == ident)
        by_q[rec["quarter"]].append(rec)
    out = []
    for q, recs in by_q.items():
        costs, names, Us = quarter_variants(q, C, run, rows, cr_labels, cr_post,
                                            only=None if only is None else {"none", *only})
        if only is not None and "one_lot" not in only:
            Us.pop("_onelot")
        n = C.n
        for rec in recs:
            D = changed_set(costs, np.array(rec["lots"]))
            S = ring(D, rings)
            sc = C.scored & mask(n, S)
            x0 = costs.codes
            xt = x0.copy()
            t = costs.table_codes[S]
            xt[S] = np.where(t >= 0, t, x0[S])
            right0 = int((sc & (C.table == names[x0].astype(str))).sum())
            inc_lots = np.array(rec["lots"])
            r = {"id": rec["id"], "quarter": q, "set_lots": int(len(S)), "scored": int(sc.sum()),
                 "right_solved": right0}
            for nm, U in Us.items():
                if nm == "_onelot":
                    best, de = U[0].answer(S, U[1])
                    right = int((sc & (C.table == names[best].astype(str))).sum())
                    r["one_lot"] = {"energy_change": round(de, 3), "right": right,
                                    "incident_right": round(float(np.mean(names[best[inc_lots]] == C.table[inc_lots])),
                                                            3)}
                    continue
                cm = replace(costs, U=U)
                e0 = energy_on(cm, x0, S)
                best, be = x0, e0
                for start in (x0, xt):
                    x = search_on(cm, start, S)
                    e = energy_on(cm, x, S)
                    if e < be - 1e-9:
                        best, be = x, e
                right = int((sc & (C.table == names[best].astype(str))).sum())
                r[nm] = {"energy_change": round(be - e0, 3), "right": right,
                         "incident_right": round(float(np.mean(names[best[inc_lots]] == C.table[inc_lots])), 3)}
            out.append(r)
            print(f"#{rec['id']} ({q}, {len(S)} lots, {int(sc.sum())} scored, right as solved {right0}): "
                  + "; ".join(f"{nm} {r[nm]['right']} ({r[nm]['incident_right']:.0%} of I)" for nm in r
                              if isinstance(r[nm], dict)),
                  flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"oracle_rings{rings}{name}.json").write_text(json.dumps({"run": run, "rings": rings, "incidents": out},
                                                              indent=1) + "\n")
    summary_oracle(out)



def affected_cmd(variants, max_lots: int, unit: str = "area") -> None:
    """Every walled area (lots joined by the borders the quarter's odds call open, odds <= 0) holding a lot of that
    quarter whose label costs a variant changes: the solver's own search confined to the area, every other lot as
    solved, from the solved answer and from the table's (the lower energy kept), with the saved costs and with the
    variant's; the scored lots of the quarter right in each.  Gains and losses over every such area, not only the
    incidents: what a variant would break as well as fix, short of a full solve.  Areas over `max_lots` lots are
    skipped and counted.  With `unit` "district" the unit is instead a district of the solve (connected lots with
    one solved code): the walled areas join most of the city into a few areas, the solved districts do not."""
    C = city()
    doc = json.loads((CENSUS_DIR / "incidents.json").read_text())
    run = doc["run"]
    rows = placement_rows()
    cr_labels, cr_post = codereader()
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    out = {v: Counter() for v in variants}
    detail = {v: [] for v in variants}
    for qi, q in enumerate(paths.QUARTERS):
        costs, names, Us = quarter_variants(q, C, run, rows, cr_labels, cr_post, only={"none", *variants})
        op = costs.odds <= 0 if unit == "area" else costs.codes[C.a] == costs.codes[C.b]
        _, comp = connected_components(coo_matrix((np.ones(int(op.sum())), (C.a[op], C.b[op])), shape=(C.n, C.n)),
                                       directed=False)
        order = np.argsort(comp, kind="stable")
        bounds = np.r_[0, np.flatnonzero(np.diff(comp[order])) + 1, C.n]
        members = {int(comp[order[i]]): order[i:j] for i, j in zip(bounds[:-1], bounds[1:])}
        x0 = costs.codes
        table_ok = costs.table_codes >= 0
        here = C.quarter == qi
        done = {}

        def right_with(U, S, key):
            if key in done:
                return done[key]
            cm = replace(costs, U=U)
            xt = x0.copy()
            xt[S] = np.where(table_ok[S], costs.table_codes[S], x0[S])
            e0 = energy_on(cm, x0, S)
            best, be = x0, e0
            for start in (x0, xt):
                x = search_on(cm, start, S)
                e = energy_on(cm, x, S)
                if e < be - 1e-9:
                    best, be = x, e
            sc = C.scored[S] & here[S]
            r = int((sc & (C.table[S] == names[best[S]].astype(str))).sum())
            done[key] = r
            return r

        for v in variants:
            if v == "one_lot":           # the lots whose costs change when each label stands on its best lot under
                ol, tie = Us["_onelot"]  # the solved answer (the candidate's first step)
                Uz = label_U(costs, ol.rd, ol.options, C.n, ol.K, place=ol.choose(x0, list(ol.opts)),
                             drop_far=ol.far_lab) + tie
                Uz -= Uz.min(1, keepdims=True)
                diff = np.abs(Uz - Us["none"]).max(1) > 1e-9
            else:
                diff = np.abs(Us[v] - Us["none"]).max(1) > 1e-9
            areas = np.unique(comp[diff & here])
            for ar in areas:
                S = members[int(ar)]
                if len(S) > max_lots:
                    out[v]["areas skipped (too large)"] += 1
                    out[v]["lots skipped"] += int(len(S))
                    continue
                r0 = right_with(Us["none"], S, ("none", int(ar)))
                if v == "one_lot":
                    best, _ = Us["_onelot"][0].answer(S, Us["_onelot"][1])
                    r1 = int((C.scored[S] & here[S] & (C.table[S] == names[best[S]].astype(str))).sum())
                else:
                    r1 = right_with(Us[v], S, (v, int(ar)))
                out[v]["areas"] += 1
                out[v]["scored lots in them"] += int((C.scored[S] & here[S]).sum())
                out[v]["gained"] += max(r1 - r0, 0)
                out[v]["lost"] += max(r0 - r1, 0)
                if r1 != r0:
                    detail[v].append({"quarter": q, "area_lots": int(len(S)), "right_before": r0, "right_after": r1,
                                      "first_lot": str(C.lots["blklot"].iloc[int(S[0])])})
            print(f"{q} {v}: {dict(out[v])}", flush=True)
    res = {v: {"totals": dict(out[v]), "changed_areas": detail[v]} for v in variants}
    OUT.mkdir(parents=True, exist_ok=True)
    doc_out = {"run": run, "unit": unit, "max_lots": max_lots, "variants": res}
    (OUT / f"affected_{unit}_{'_'.join(variants)}.json").write_text(json.dumps(doc_out, indent=1) + "\n")
    for v in variants:
        t = out[v]
        print(f"{v}: areas {t['areas']} ({t['scored lots in them']:,} scored lots), gained {t['gained']}, lost "
              f"{t['lost']}, net {t['gained'] - t['lost']:+d}; skipped {t['areas skipped (too large)']} areas "
              f"({t['lots skipped']:,} lots)")



def try_cmd(ident: int, edits, rings: int) -> None:
    """One incident's confined search (the oracle's protocol) with hand-picked label edits, each alone and all
    together: "place:LABEL:BLKLOT" (the label charged on that lot alone; its far-side copy dropped), "drop:LABEL"
    (the label left out), "read:LABEL:CODE" (its readings replaced by CODE at its own top trust), "nofar:LABEL" (its
    far-side copy dropped).  Shows which label's fault an incident is."""
    C = city()
    doc = json.loads((CENSUS_DIR / "incidents.json").read_text())
    run = doc["run"]
    rec = next(t for t in doc["list"] if t["id"] == ident)
    q = rec["quarter"]
    costs = load_costs(run, q)
    ch = label_charges(run, q, costs)
    rd = readings(q, table_vocabulary(C.lots))
    options = load_options(q, runs.solve_dir(run))
    names = np.array(costs.names)
    blk = {b: i for i, b in enumerate(C.lots["blklot"].astype(str).values)}
    parsed = []
    for e in edits:
        kind, lab, *rest = e.split(":")
        kw = {}
        if kind == "place":
            kw = {"place": {int(lab): blk[rest[0]]}, "drop_far": frozenset({int(lab)})}
        elif kind == "drop":
            kw = {"drop": frozenset({int(lab)})}
        elif kind == "read":
            kw = {"read": {int(lab): costs.index[canon(rest[0])]}}
            if len(rest) > 1:                       # read:LABEL:CODE:TRUST
                kw["read_trust"] = float(rest[1])
        elif kind == "nofar":
            kw = {"drop_far": frozenset({int(lab)})}
        parsed.append((e, kw))
    every = {}
    for _, kw in parsed:
        for k, v in kw.items():
            every[k] = ({**every.get(k, {}), **v} if isinstance(v, dict) else
                        frozenset(every.get(k, frozenset())) | v if isinstance(v, frozenset) else v)
    D = changed_set(costs, np.array(rec["lots"]))
    S = ring(D, rings)
    sc = C.scored & mask(C.n, S)
    x0 = costs.codes
    xt = x0.copy()
    t = costs.table_codes[S]
    xt[S] = np.where(t >= 0, t, x0[S])
    inc_lots = np.array(rec["lots"])
    for nm, kw in [("none", {})] + parsed + ([("all edits", every)] if len(parsed) > 1 else []):
        U = label_U(costs, rd, options, C.n, len(names), **kw) + ch.tie
        U -= U.min(1, keepdims=True)
        cm = replace(costs, U=U)
        e0 = energy_on(cm, x0, S)
        best, be = x0, e0
        for start in (x0, xt):
            x = search_on(cm, start, S)
            e = energy_on(cm, x, S)
            if e < be - 1e-9:
                best, be = x, e
        right = int((sc & (C.table == names[best].astype(str))).sum())
        got = Counter(names[best[inc_lots]].tolist()).most_common(2)
        print(f"#{ident} {nm}: right {right} of {int(sc.sum())} scored in the set; the incident "
              f"{float(np.mean(names[best[inc_lots]] == C.table[inc_lots])):.0%} right ({got})", flush=True)


_LAB_LOT = None


def human_fixes(costs, C, people_only: bool = False) -> tuple[dict, set, dict]:
    """label_fixes.csv as the solver could read it: each row joined to the nearest print on its sheet within MATCH_PX
    page px; (label -> code index of a code fix, labels a deletion ('-') removes, label -> the lot at a hand leader
    move's fix point, through the sheet's corrected warp).  `people_only`: leave out the code fixes whose note says
    the code check took the readers' agreed code ("Terra and Sol both read ...")."""
    from sfzmaps.transform import SheetTransform
    pr = load_prints()
    sheet = pr["sheet"].astype(str).to_numpy()
    px, py = pr["page_x"].to_numpy(float), pr["page_y"].to_numpy(float)
    read, drop, place, tf = {}, set(), {}, {}
    grown = C.grown_raster
    for f in hand_rows():
        d = np.where(sheet == f["sheet"], np.hypot(px - float(f["x"]), py - float(f["y"])), np.inf)
        j = int(np.argmin(d))
        if d[j] > MATCH_PX:
            continue
        fc = (f["fix_code"] or "").strip()
        if fc == "-":
            drop.add(j)
        elif fc and canon(fc) in costs.index and not (people_only and "both read" in f["note"]):
            read[j] = costs.index[canon(fc)]
        if (f["fix_x"] or "").strip():
            if f["sheet"] not in tf:
                folder = config.DATA_DIR / "warp" / paths.SERIES / f["sheet"]
                tf[f["sheet"]] = SheetTransform.from_json(json.loads((folder / "transform.json").read_text()),
                                                          folder=folder)
            w = tf[f["sheet"]].to_world(np.array([[float(f["fix_x"]), float(f["fix_y"])]]))[0]
            col, row = C.px(w)
            if 0 <= row < grown.shape[0] and 0 <= col < grown.shape[1] and grown[row, col] >= 0:
                place[j] = int(grown[row, col])
    return read, drop, place


def load_prints_lab_lot():
    global _LAB_LOT
    if _LAB_LOT is None:
        _LAB_LOT = load_prints()["lab_lot"].to_numpy(np.int64)
    return _LAB_LOT


def summary_oracle(out) -> None:
    """gains in scored lots right over "none", each changed set counted once"""
    seen, tot = set(), Counter()
    for r in out:
        key = (r["quarter"], r["set_lots"], r["scored"], r["right_solved"])
        if key in seen:
            continue
        seen.add(key)
        for nm in r:
            if isinstance(r[nm], dict):
                tot[nm] += r[nm]["right"] - r["none"]["right"]
    print(f"{len(seen)} distinct sets; gain in scored lots right over 'none': "
          + ", ".join(f"{k} {v:+d}" for k, v in tot.items()))


# --- the street labels ----------------------------------------------------------------------------------------------

def street_cmd() -> None:
    C = city()
    pr = load_prints()
    under = pr["lab_lot"].to_numpy(np.int64)
    out = {}
    for q in paths.QUARTERS:
        qi = paths.QUARTERS.index(q)
        rd = readings(q, table_vocabulary(C.lots))
        SS = np.load(paths.LABELS_DIR / f"streetside_{q}.npz")
        names = rd.names
        top = {}
        for lab, code, t, far in zip(rd.label, rd.code, rd.trust, rd.far):
            if not far and t > top.get(int(lab), ("", -1))[1]:
                top[int(lab)] = (canon(names[code]), float(t))
        cnt = Counter()
        for lab, (c, t) in top.items():
            a = under[lab]
            if a < 0 or C.quarter[a] != qi or not SS["onst"][lab] or SS["isfw"][lab]:
                continue
            b = int(SS["lotB"][lab])
            if b < 0 or not (C.known[a] and C.known[b]):
                continue
            ta, tb = canon(str(C.table[a])), canon(str(C.table[b]))
            same = ta == tb
            key = ("sides same" if same else "sides differ") + ": " + (
                "both" if c == ta and c == tb else "A only" if c == ta else "B only" if c == tb else "neither")
            cnt[key] += 1
        out[q] = dict(sorted(cnt.items()))
        print(q, dict(sorted(cnt.items())))
    tot = Counter()
    for v in out.values():
        tot.update(v)
    print("all", dict(sorted(tot.items())))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "street.json").write_text(json.dumps({"by_quarter": out, "all": dict(tot)}, indent=1) + "\n")


# --- pictures -------------------------------------------------------------------------------------------------------

S2 = 2              # image px per grid px
MARGIN = 60         # grid px


def _up(m):
    return np.kron(m.astype(np.uint8), np.ones((S2, S2), np.uint8))


def _wide(m, w):
    u = _up(m)
    return cv2.dilate(u, np.ones((w, w), np.uint8)) > 0 if w > 1 else u > 0


def _legend(img, lines):
    strip = np.full((22 * len(lines) + 8, img.shape[1], 3), 255, np.uint8)
    for i, (txt, col) in enumerate(lines):
        y, x = 20 + 22 * i, 6
        if col is not None:
            cv2.rectangle(strip, (x, y - 12), (x + 22, y + 2), col, -1)
            x += 30
        cv2.putText(strip, txt, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    return np.vstack([strip, img])


def _dashed(img, p0, p1, col, th, dash=10):
    p0, p1 = np.array(p0, float), np.array(p1, float)
    L = float(np.hypot(*(p1 - p0)))
    if L < 1:
        return
    n = max(int(L // dash), 1)
    for i in range(0, n, 2):
        a = p0 + (p1 - p0) * i / n
        b = p0 + (p1 - p0) * min(i + 1, n) / n
        cv2.line(img, tuple(int(v) for v in a), tuple(int(v) for v in b), col, th, cv2.LINE_AA)
    cv2.arrowedLine(img, tuple(int(v) for v in p0 + (p1 - p0) * 0.9), tuple(int(v) for v in p1), col, th,
                    cv2.LINE_AA, tipLength=0.5)


def picture_cmd(ident: int, window=None) -> None:
    C = city()
    doc = json.loads((CENSUS_DIR / "incidents.json").read_text())
    run = doc["run"]
    rec = next(t for t in doc["list"] if t["id"] == ident)
    q = rec["quarter"]
    lab_doc = json.loads((OUT / "labels.json").read_text())
    linc = next((t for t in lab_doc["incidents"] if t["id"] == ident), None)
    tipping = {it["label"]: max(it["favours_wrong_by"], it["on_incident"]) for it in (linc["labels"] if linc else [])
               if max(it["favours_wrong_by"], it["on_incident"]) > 0.2}
    costs = load_costs(run, q)
    names = np.array(costs.names)
    rd = readings(q, table_vocabulary(C.lots))
    options = load_options(q, runs.solve_dir(run))
    SS = np.load(paths.LABELS_DIR / f"streetside_{q}.npz")
    pr = load_prints()
    a, b = C.a, C.b
    codes = run_codes(run)
    both = C.known[a] & C.known[b]
    tline = both & (C.table[a] != C.table[b])
    d_run = codes[a] != codes[b]
    I = np.array(rec["lots"], int)
    D = changed_set(costs, I)
    lot, grown = C.lot_raster, C.grown_raster
    if window:
        x0_, y0_, x1_, y1_ = window
        c0, r0 = np.floor(grid.to_grid([x0_, y1_])).astype(int)
        c1, r1 = np.ceil(grid.to_grid([x1_, y0_])).astype(int)
    else:
        inD = np.zeros(C.n + 1, bool)
        inD[D] = True
        rr, cc = np.nonzero(inD[np.where(lot >= 0, lot, C.n)])
        r0, r1 = max(rr.min() - MARGIN, 0), min(rr.max() + MARGIN, grid.HEIGHT)
        c0, c1 = max(cc.min() - MARGIN, 0), min(cc.max() + MARGIN, grid.WIDTH)
    ink = C.scan()[r0:r1, c0:c1]
    L, G = lot[r0:r1, c0:c1], grown[r0:r1, c0:c1]
    H, W = L.shape
    keys = a.astype(np.int64) * C.n + b
    pair_px = np.full((H, W), -1, np.int64)
    for gA, gB, s1, s2 in ((G[:, :-1], G[:, 1:], (slice(None), slice(0, W - 1)), (slice(None), slice(1, W))),
                           (G[:-1], G[1:], (slice(0, H - 1), slice(None)), (slice(1, H), slice(None)))):
        lo_, hi_ = np.minimum(gA, gB).astype(np.int64), np.maximum(gA, gB).astype(np.int64)
        k = np.minimum(np.searchsorted(keys, lo_ * C.n + hi_), len(keys) - 1)
        ok = (gA != gB) & (keys[k] == lo_ * C.n + hi_)
        for sl in (s1, s2):
            sub = pair_px[sl]
            sub[ok] = k[ok]
            pair_px[sl] = sub
    has, pk = pair_px >= 0, np.maximum(pair_px, 0)
    img = np.full((H * S2, W * S2, 3), 255, np.uint8)
    inI, inDm = np.zeros(C.n + 1, bool), np.zeros(C.n + 1, bool)
    inI[I], inDm[D] = True, True
    Ls = np.where(L >= 0, L, C.n)
    img[_up(inI[Ls]) > 0] = (150, 235, 255)
    img[_up(inDm[Ls] & ~inI[Ls]) > 0] = (150, 200, 255)
    g = _up(ink) > 0
    img[g] = (img[g] * 0.35 + np.array([110, 110, 110]) * 0.65).astype(np.uint8)
    TABLE, RUN = (60, 180, 60), (220, 120, 20)
    img[_wide(has & tline[pk], 7)] = TABLE
    img[_wide(has & d_run[pk], 3)] = RUN

    def to_img(xy):
        c, r = grid.to_grid(xy)
        return (c - c0) * S2, (r - r0) * S2

    top = {}
    for lab, code, t, far in zip(rd.label, rd.code, rd.trust, rd.far):
        if not far and t > top.get(int(lab), ("", -1))[1]:
            top[int(lab)] = (canon(rd.names[code]), float(t))
    GOOD, BAD = (40, 160, 40), (30, 30, 220)
    xy_all = pr[["x", "y"]].to_numpy(float)
    shown = []
    for lab, (c, t) in top.items():
        px, py = to_img(xy_all[lab])
        if not (0 <= px < W * S2 and 0 <= py < H * S2):
            continue
        shown.append((lab, c, t, px, py))
        for lot_, pi, st in options.get(lab, []):
            if pi < 0.02:
                continue
            col = GOOD if canon(str(C.table[lot_])) == c else BAD
            th = max(1, int(round(1 + 5 * pi)))
            if st:
                cv2.circle(img, (int(px), int(py)), int(8 + 14 * pi), col, th, cv2.LINE_AA)
            else:
                ex, ey = to_img(C.centroid[lot_])
                cv2.arrowedLine(img, (int(px), int(py)), (int(ex), int(ey)), col, th, cv2.LINE_AA, tipLength=0.08)
                cv2.putText(img, f"{pi:.2f}", (int((px + ex) / 2) + 3, int((py + ey) / 2) - 3),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, col, 1, cv2.LINE_AA)
        lotB = int(SS["lotB"][lab])
        if SS["onst"][lab] and not SS["isfw"][lab] and lotB >= 0:
            col = GOOD if canon(str(C.table[lotB])) == c else BAD
            ex, ey = to_img(C.centroid[lotB])
            _dashed(img, (px, py), (ex, ey), col, 2)
    for lab, c, t, px, py in shown:
        cv2.circle(img, (int(px), int(py)), 4, (0, 0, 0), -1)
        txt = f"{lab}: {c} {t:.2f}"
        if lab in tipping:
            txt += f"  [{tipping[lab]:+.1f}]"
            cv2.rectangle(img, (int(px) - 10, int(py) - 22), (int(px) + 12 + 9 * len(txt), int(py) + 10),
                          (180, 0, 180), 2)
        for col, th in (((255, 255, 255), 3), ((0, 0, 0), 1)):
            cv2.putText(img, txt, (int(px) + 6, int(py) - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, th, cv2.LINE_AA)
    img = _legend(img, [
        (f"#{ident}: {rec['lots_count']} lots, table {rec['table']}, solved {rec['solved']}; quarter {q}; 2x the "
         f"grid (1 image px = 2 ft), crop {W * 4} x {H * 4} ft; grey: page ink", None),
        ("tint: yellow the incident's lots, orange the rest of its changed set D", None),
        ("table district border (wide)", TABLE), (f"{run}'s border (thin)", RUN),
        ("placement option whose lot's table code is the label's top reading: ring = stay, arrow = move, "
         "dashed = far-side copy; width by chance", GOOD),
        ("the same, the lot's table code is another code", BAD),
        ("label: print row, top reading, trust; boxed: tips the incident [its energy share, + = favours the wrong "
         "answer]", (180, 0, 180))])
    pics = OUT / "pictures"
    pics.mkdir(parents=True, exist_ok=True)
    name = f"{ident}_placement" + ("_window" if window else "")
    cv2.imwrite(str(pics / f"{name}.png"), img)
    print("wrote", pics / f"{name}.png")
    crops_picture(ident, linc, pics)


def crops_picture(ident, linc, pics) -> None:
    """the code reader's crop (page px, turned to read left to right) of each label that tips the incident, with
    every reader's read, the top reading and trust, the owner's answer and the code reader's posterior"""
    if not linc:
        return
    cr_labels, _ = codereader()
    items = [it for it in linc["labels"] if max(it["favours_wrong_by"], it["on_incident"]) > 0.2][:8]
    tiles = []
    for it in items:
        lab = cr_labels.get(it["label"])
        if lab is None or getattr(lab, "crop", None) is None:
            continue
        crop = cv2.cvtColor(lab.crop, cv2.COLOR_GRAY2BGR)
        scale = 2 if max(crop.shape[:2]) < 300 else 1
        crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
        lines = [f"label {it['label']} ({it['sheet'][-3:]} px {it['page_px'][0]},{it['page_px'][1]}) "
                 f"{it['favours_wrong_by']:+.2f} (on the incident {it['on_incident']:+.2f})",
                 f"top {it['top']} {it['trust']:.2f}; under {it['under']['lot']} table {it['under']['table']}",
                 "reads " + ", ".join(f"{k} {v}" for k, v in it["reads"].items()),
                 f"owner {it['owner_answer']}; reader {it['codereader'][:2] if it['codereader'] else None}"]
        w = max(crop.shape[1], 470)
        tile = np.full((crop.shape[0] + 20 * len(lines) + 10, w, 3), 255, np.uint8)
        tile[:crop.shape[0], :crop.shape[1]] = crop
        for i, ln in enumerate(lines):
            cv2.putText(tile, ln, (4, crop.shape[0] + 18 + 20 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 1,
                        cv2.LINE_AA)
        tiles.append(tile)
    if not tiles:
        return
    h = max(t.shape[0] for t in tiles)
    tiles = [np.vstack([t, np.full((h - t.shape[0], t.shape[1], 3), 255, np.uint8)]) for t in tiles]
    rows_ = [tiles[i:i + 3] for i in range(0, len(tiles), 3)]
    wmax = max(sum(t.shape[1] + 8 for t in r) for r in rows_)
    out = []
    for r in rows_:
        line = np.hstack([np.hstack([t, np.full((h, 8, 3), 255, np.uint8)]) for t in r])
        out.append(np.hstack([line, np.full((h, wmax - line.shape[1], 3), 255, np.uint8)]))
    img = _legend(np.vstack(out), [(f"#{ident}: the code reader's crop of each label tipping the incident (the "
                                    "sheet's page px, x2 when small, turned to read left to right)", None)])
    cv2.imwrite(str(pics / f"{ident}_crops.png"), img)
    print("wrote", pics / f"{ident}_crops.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("labels")
    p.add_argument("ids", type=int, nargs="*")
    p = sub.add_parser("picture")
    p.add_argument("id", type=int)
    p.add_argument("--window", type=float, nargs=4)
    p = sub.add_parser("oracle")
    p.add_argument("ids", type=int, nargs="*")
    p.add_argument("--rings", type=int, default=2)
    p.add_argument("--all", action="store_true", help="every census incident (written oracle_rings<R>_all.json)")
    p.add_argument("--only", nargs="+", help="these variants alone (and none); written with their names")
    sub.add_parser("street")
    p = sub.add_parser("try")
    p.add_argument("id", type=int)
    p.add_argument("edits", nargs="+")
    p.add_argument("--rings", type=int, default=2)
    p = sub.add_parser("affected")
    p.add_argument("variants", nargs="+")
    p.add_argument("--max-lots", type=int, default=3000)
    p.add_argument("--unit", choices=("area", "district"), default="area")
    args = ap.parse_args()
    if args.cmd == "labels":
        labels_cmd(args.ids or IDS)
    elif args.cmd == "picture":
        picture_cmd(args.id, args.window)
    elif args.cmd == "oracle":
        if args.all:
            every = [t["id"] for t in json.loads((CENSUS_DIR / "incidents.json").read_text())["list"]]
            oracle_cmd(every, args.rings, "_all" + ("_" + "_".join(args.only) if args.only else ""), args.only)
        else:
            oracle_cmd(args.ids or IDS, args.rings)
    elif args.cmd == "try":
        try_cmd(args.id, args.edits, args.rings)
    elif args.cmd == "affected":
        affected_cmd(args.variants, args.max_lots, args.unit)
    else:
        street_cmd()
