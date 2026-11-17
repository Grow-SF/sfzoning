"""`scripts/wordread/run_choose_pack.py`: the chosen boxes' table, the fold fit's training set, the pack's read function and
the keep rule's numbers, on synthetic versions and reads (no networks, no pages)."""
import importlib.util
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from sfzmaps.solver.textread import model
from sfzmaps.solver.textread import read as R
from sfzmaps.solver.wordread import choose as C
from sfzmaps.solver.wordread import context

SPEC = importlib.util.spec_from_file_location("run_choose_pack", Path(__file__).parent.parent / "scripts" / "wordread"
                                              / "run_choose_pack.py")
RC = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RC)

CODES = R.CodeList({"R1": "R1", "C2": "C2"})
KINDS = ("zoning", "block", "street", "other")


def box(row, x0, x1, y0=20.0, y1=30.0, quarter=1, sheet="s"):
    b = {"row": row, "sheet": sheet, "quarter": quarter, "i": row, "axis": 0.0, "t_length": x1 - x0,
         "t_height": y1 - y0, **{f"p_{k}": 0.0 for k in KINDS}}
    b["p_zoning"] = 1.0
    for i, (x, y) in enumerate([(x0, y0), (x1, y0), (x1, y1), (x0, y1)]):
        b[f"t{i}x"], b[f"t{i}y"] = x, y
        b[f"g{i}x"], b[f"g{i}y"] = x + (-2 if i in (0, 3) else 2), y + (-2 if i < 2 else 2)
    return b


def got(p_code: dict[str, float], q: float = 0.0, other: float = 0.5) -> dict:
    """A read of a crop: each form's probability (both reads alike), "other" `other`, "none" 1e-3."""
    lf = np.log([p_code.get(f, 1e-6) for f in CODES.forms])
    po = other
    return {"reads": [(lf, math.log(1e-3), po)] * 2, "best": ["x", "x"], "p_classifier": [q, q]}


def versions_for(b, others=()):
    """A box with the three versions as found / split / trim, each word read as the table says."""
    ink = np.zeros((60, 200), bool)
    ink[22:28, 10:40] = True
    ink[22:28, 60:90] = True
    cands = C.candidates(b, ink, list(others), {k: 8.0 for k in KINDS})
    return cands


def entry(cands, reads):
    return {"cands": cands, "gots": [[reads[c.name]] * len(c.rects) for c in cands]}


def test_assemble_split_gives_two_rows_that_tile_the_grouped_rectangle():
    b = box(0, 10, 90)
    cands = versions_for(b)
    ent = entry(cands, {c.name: got({}) for c in cands})
    table, gots = RC.assemble(pd.DataFrame([b]), {0: ent}, {0: "split_gap"})
    assert len(table) == 2 and set(table["choice"]) == {"split_gap"} and len(gots) == 2
    g = table[RC.G_COLS].to_numpy().reshape(2, 4, 2)
    assert g[0, :, 0].min() == b["g0x"] and g[1, :, 0].max() == b["g1x"]
    assert g[0, :, 0].max() == pytest.approx(g[1, :, 0].min())          # cut halfway across the gap
    assert (table["row"].to_numpy() == [0, 1]).all()
    assert table["t_length"].sum() < b["t_length"]


def test_assemble_keeps_the_box_as_found_unchanged_and_never_drops_duplicates():
    b1, b2 = box(0, 10, 90), box(1, 10, 90)                             # two boxes alike: both are kept as found
    ents = {r: entry(versions_for(b), {c.name: got({}) for c in versions_for(b)}) for r, b in ((0, b1), (1, b2))}
    table, _ = RC.assemble(pd.DataFrame([b1, b2]), ents, {0: "as_found", 1: "as_found"})
    assert len(table) == 2
    assert table[RC.T_COLS].to_numpy()[0] == pytest.approx(np.array([b1[c] for c in RC.T_COLS]))


def test_a_join_chosen_by_both_boxes_is_one_row():
    a, b = box(0, 10, 50), box(1, 60, 100)
    ink = np.zeros((60, 200), bool)
    w = {k: 8.0 for k in KINDS}
    ents = {}
    for r, (x, y) in enumerate(((a, b), (b, a))):
        cands = C.candidates(x, ink, [y], w)
        ents[r] = entry(cands, {c.name: got({}) for c in cands})
    table, gots = RC.assemble(pd.DataFrame([a, b]), ents, {0: "join_nearest", 1: "join_nearest"})
    assert len(table) == 1 and len(gots) == 1
    xs = table[[f"t{i}x" for i in range(4)]].to_numpy()
    assert xs.min() == 10 and xs.max() == 100
    assert table["src_row"].iloc[0] == 0


def test_the_choice_reads_each_version_by_the_stored_reads():
    b = box(0, 10, 90)
    cands = versions_for(b)
    reads = {c.name: got({}) for c in cands}
    reads["split_gap"] = got({"R1": 0.9, "C2": 0.05})          # the two halves each read clearly
    reads["as_found"] = got({"R1": 0.3})
    ent = entry(cands, reads)
    fn = RC.read_fn_of(ent, CODES)
    lp = fn(cands[0].rects[0], b)
    assert set(lp) == {"R1", "C2", "other", "none"} and lp["R1"] == pytest.approx(math.log(0.3))
    name, info = C.choose(b, cands, fn)
    assert name == "split_gap" and len(info["words"]) == 2
    with pytest.raises(KeyError):
        fn(np.zeros((1, 4, 2))[0] + 5, b)


def test_fold_fit_uses_only_the_other_quarters_outside_the_answered_blocks():
    rng = np.random.default_rng(0)
    out = []
    for box_id in range(60):
        q = box_id % 3                                                    # quarters 0, 1, 2
        for cand, extra in (("as_found", 0.0), ("split_gap", 1.0)):
            s = [float(rng.normal() + extra) / (1 + extra)] * int(1 + extra)
            out.append({"box": box_id, "quarter": q, "cand": cand, "score": sum(s), "word_scores": s, "gap": extra,
                        "across": 0.0, "kind": "zoning",
                        "weight": float(cand == ("split_gap" if box_id % 2 else "as_found")),
                        "answered": box_id < 6, **{h: q == 1 and box_id % 5 == 0 for h in RC.HOLD[:1]},
                        **{h: False for h in RC.HOLD[1:]}})
    rows = pd.DataFrame(out)
    fit = RC.fold_fit(rows, 1)
    held = rows[(rows["quarter"] != 1) & ~rows["answered"]]
    assert fit["boxes"] == held["box"].nunique() and fit["versions"] == len(held)
    assert set(fit) >= {"intercept", "beta", "alpha", "gamma", "gamma_across", "kind", "count1"}
    assert fit["count1"] is None                                           # no box model's column: no count weight
    fit_count = RC.fold_fit(rows.assign(word_count1=[[-1.0] * len(w) for w in rows["word_scores"]]), 1)
    assert fit_count["count1"] is not None
    with pytest.raises(ValueError):
        C.fit_quality(rows.assign(held_out=False).iloc[:10].assign(answered=True))


def _lp(text: str, frames: int = 24, hit: float = 0.97) -> np.ndarray:
    """A CTC read whose frames spell `text` (each char three frames, blanks between), the rest of the mass spread."""
    lp = np.full((frames, model.CLASSES), math.log((1 - hit) / (model.CLASSES - 1)))
    for k, c in enumerate(text):
        for f in (4 * k + 1, 4 * k + 2):
            lp[f, model.encode(c)[0]] = math.log(hit)
    blank = [f for f in range(frames) if not (lp[f] > math.log(0.5)).any()]
    lp[blank, model.BLANK] = math.log(hit)
    return lp - np.logaddexp.reduce(lp, axis=1, keepdims=True)


def test_pack_read_fn_is_the_ink_alone_without_a_prior_and_lifts_a_listed_string_with_one():
    reads = [_lp("R1"), _lp("R1")]
    base = ["R1", "C2"]
    plain = RC.pack_read_fn(lambda c, b: (reads, 0.0), lambda c, b: None, {}, base_strings=base)(None, {})
    ink = context.ink_logp(reads, 0.0, base)
    assert plain["R1"] == pytest.approx(ink["R1"] - np.logaddexp.reduce(np.r_[list(ink.values()),
                                                                        math.log(max(1e-300, 1 - sum(np.exp(
                                                                            list(ink.values())))))]), abs=1e-6)
    prior = context.Prior(kind_p=np.array([0.0, 0.0, 1.0, 0.0]), escape=np.array([1.0, 1.0, 0.2, 1.0]),
                          items={"street name": [(30.0, {"C2": 1.0})]})
    withp = RC.pack_read_fn(lambda c, b: (reads, 0.0), lambda c, b: prior, {"street name": 1.0},
                            base_strings=base)(None, {})
    assert withp["C2"] > plain["C2"] and withp["R1"] < plain["R1"]
    assert context.OTHER in withp
    assert np.exp(list(withp.values())).sum() == pytest.approx(1.0, abs=1e-6)


def test_choose_takes_the_pack_read_function_and_the_place_of_each_version():
    b = box(0, 10, 90)
    cands = versions_for(b)
    seen = []

    def lp_fn(corners, box_):
        seen.append(float(np.asarray(corners)[:, 0].min()))
        return [_lp("R1")] * 2, 0.0

    def prior_fn(corners, box_):
        return None

    fn = RC.pack_read_fn(lp_fn, prior_fn, {}, base_strings=["R1", "C2"])
    name, info = C.choose(b, cands, fn)
    assert name in {c.name for c in cands} and set(info["scores"]) == {c.name for c in cands}
    assert len(set(seen)) > 1                                             # each version's own rectangle was read


def test_make_prior_fn_places_the_version_on_the_map_and_takes_the_box_kinds():
    class Moved:
        def to_world(self, p):
            return np.asarray(p, float) * 2.0

    calls = []
    real = context.prior
    try:
        context.prior = lambda pack, on_map, kp, sp: calls.append((on_map, kp, sp)) or None
        b = box(0, 10, 90)
        RC.make_prior_fn("pack", "spreads", {"s": Moved()})(C.candidates(b, np.zeros((60, 200), bool), [],
                                                                         {k: 8.0 for k in KINDS})[0].rects[0], b)
    finally:
        context.prior = real
    on_map, kp, sp = calls[0]
    assert on_map["x"] == pytest.approx(100.0) and on_map["y"] == pytest.approx(50.0) and sp == "spreads"
    assert list(kp) == [1.0, 0.0, 0.0, 0.0]


def test_decide_pairs_each_arm_with_reading_as_found(tmp_path):
    n = 40
    block = [f"b{k // 4}" for k in range(n)]
    for s in (0, 1):
        d = tmp_path / RC.run_name("t", "ne", s)
        d.mkdir()
        base = np.r_[np.zeros(20), np.ones(20)]
        better = base.copy()
        better[:12] = 1.0                                                  # twelve labels fixed, in many blocks
        for arm, right in (("without", base), ("plain", better), ("fitted", base)):
            pd.DataFrame({"label": range(n), "row": range(n), "right": right, "block": block,
                          "quarter": 1}).to_parquet(d / f"labels_{arm}.parquet")
    out = RC.decide("t", "ne", (0, 1), root=tmp_path)
    assert out["preliminary"] is True
    assert out["plain"]["ci_excludes_zero"] and out["plain"]["mean_of_seeds"]["diff"] == pytest.approx(0.3)
    assert not out["fitted"]["ci_excludes_zero"] and out["fitted"]["mean_of_seeds"]["diff"] == 0.0
    assert len(out["plain"]["by_seed"]) == 2


def test_reader_path_prefers_the_runs_own_reader_else_the_stages(tmp_path):
    (tmp_path / "base-ne-s1").mkdir()
    (tmp_path / "base-ne-s1" / "reader.pt").write_bytes(b"x")
    assert RC.reader_path("ne", "ne", 1, "base", tmp_path) == tmp_path / "base-ne-s1" / "reader.pt"
    assert RC.reader_path("nw", "ne", 1, "base", tmp_path).name == "nw.pt"


def test_feature_rows_weigh_each_version_by_the_ink_of_terras_words():
    b = box(0, 10, 90)
    cands = versions_for(b)
    ent = entry(cands, {c.name: got({"R1": 0.5}) for c in cands})
    ink = np.zeros((60, 200), bool)
    ink[22:28, 10:40] = True
    ink[22:28, 60:90] = True

    def word(x0, x1):
        return {"sheet": "s", "touch_blocks": False, **{h: False for h in RC.HOLD},
                **dict(zip(RC.C_COLS, (x0, 20, x1, 20, x1, 30, x0, 30)))}

    terra = pd.DataFrame([word(10, 40), word(60, 90)])
    rows = RC.feature_rows(pd.DataFrame([b]), {0: ent}, terra, CODES, ink_of=lambda sheet: ink)
    w = dict(zip(rows["cand"], rows["weight"]))
    assert w["split_gap"] > 0.9 and w["as_found"] < 0.1             # two words in the box: only the split holds one each
    assert not rows["answered"].any() and (rows["box"] == 0).all()
    assert (rows["kind"] == "zoning").all() and [len(w) for w in rows["word_scores"]] == [len(c.rects) for c in cands]
