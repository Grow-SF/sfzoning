import json

import numpy as np
import pandas as pd
import pytest
from shapely.geometry import box

from sfzmaps.solver import answers1921 as a1921


def _write(tmp_path, cases, answers, page="1921-likely"):
    """One page: its cases file and its answers directory (one json per answer, wrapped in `data` for one)."""
    ad, _ = a1921.page_inputs(page, tmp_path)
    ad.mkdir(parents=True)
    cf = ad.parent / "cases.json"
    cf.write_text(json.dumps({"cases": cases}))
    for i, ans in enumerate(answers):
        body = {"case": ans["case"], "answer": ans["answer"], "code": ans.get("code", ""), "shown": ans["shown"]}
        (ad / f"{i}.json").write_text(json.dumps({"data": body} if i == 0 else body))
    return ad, cf


def _answers(tmp_path, n_blocks=3):
    cases = [{"id": f"c000{i}", "blklot": f"000{i}001", "blklots": [f"000{i}001"], "block": f"000{i}", "code": "1R"}
             for i in range(1, 5)]
    cases[3]["block"] = "0003"
    answers = [{"case": "c0001", "answer": "ok", "shown": "1R"},
               {"case": "c0002", "answer": "fix", "code": "LI", "shown": "1R"},
               {"case": "c0003", "answer": "unzoned", "shown": "1R"},
               {"case": "c0004", "answer": "canttell", "shown": "1R"}]
    ad, cf = _write(tmp_path, cases, answers)
    return a1921.owner_answers([ad], [cf])


def test_truth_rules(tmp_path):
    df = _answers(tmp_path)
    assert df.sort_values("case")["truth"].tolist() == ["1R", "LI", "2R"]
    assert df.attrs["dropped"] == 1
    assert set(df.columns) >= {"blklot", "block", "truth", "page", "case"}
    assert set(df["block"]) == {"0001", "0002", "0003"}


def _ans(blklots, truths, blocks=None):
    blocks = blocks or [b[:4] for b in blklots]
    return pd.DataFrame({"blklot": blklots, "block": blocks, "truth": truths, "page": "p",
                         "case": [f"c{i}" for i in range(len(blklots))]})


def test_score_counts_and_interval():
    ans = _ans(["0001001", "0002001", "0003001"], ["1R", "2R", "C"])
    pred = pd.Series({"0001001": "1R", "0002001": "2R", "0003001": "LI"})
    s = a1921.score(pred, ans)
    assert s["n"] == 3 and s["right"] == 2
    assert s["ci95"][0] <= 2 / 3 <= s["ci95"][1]
    assert json.dumps(a1921.score(pred, ans)) == json.dumps(s)      # fixed seed: the same bytes (empty split is NaN)


def test_unmatched_reported():
    ans = _ans(["0001001", "0002001"], ["1R", "2R"])
    s = a1921.score(pd.Series({"0001001": "1R"}), ans)
    assert s["n"] == 1 and s["unmatched"] == 1 and s["right"] == 1


def test_stacked_condo_matches_a_sibling_blklot():
    ans = _ans(["0001001"], ["1R"])
    ans["blklots"] = [("0001001", "0001002")]
    s = a1921.score(pd.Series({"0001002": "1R"}), ans)
    assert s["n"] == 1 and s["unmatched"] == 0 and s["right"] == 1


def test_split_lot_listed():
    ans = _ans(["0001001", "0002001"], ["1R", "2R"])
    pred = pd.Series({"0001001": "1R", "0002001": "2R"})
    assert a1921.score(pred, ans, split_lots=["0002001"])["split_lots"] == ["0002001"]
    assert a1921.score(pred, ans)["split_lots"] == []


def test_footprint_fill_gives_stacked_lots_their_footprints_code():
    geoms = pd.Series([box(0, 0, 1, 1), box(0, 0, 1, 1), box(5, 5, 6, 6)], index=["a", "b", "c"])
    pred = pd.Series({"a": "LI", "c": "C"})
    out = a1921.fill_footprints(pred, geoms)
    assert out.to_dict() == {"a": "LI", "b": "LI", "c": "C"}


def test_compare_pairs_against_published():
    ans = _ans([f"{i:04d}001" for i in range(1, 7)], ["1R"] * 6)
    pub = pd.Series({b: "2R" for b in ans["blklot"]})
    better = pd.Series({b: "1R" for b in ans["blklot"]})
    out = a1921.compare({"published": pub, "better": better}, ans)
    assert out["published"]["right"] == 0 and out["better"]["right"] == 6
    d = out["better"]["vs_published"]
    assert d["diff"] == 6 and d["share_diff"] == 1.0
    assert d["ci95"][0] > 0
    assert "vs_published" not in out["published"]


def test_predictions_from_run(tmp_path):
    table = tmp_path / "table.parquet"
    pd.DataFrame({"blklot": ["a", "b", "b", "c"]}).to_parquet(table)
    run = tmp_path / "final.npz"
    np.savez(run, names=np.array(["1R", "RH-1"]), pred=np.array([0, 1, 0, 1]))
    out = a1921.predictions_from_run(run, lots_table=table)
    assert out.to_dict() == {"a": "1R", "b": "RH-1", "c": "RH-1"}      # the first row of a repeated blklot


def test_split_lot_answers_have_their_own_row():
    ans = _ans(["0001001", "0002001", "0003001"], ["1R", "2R", "C"])
    pred = pd.Series({"0001001": "1R", "0002001": "LI", "0003001": "C"})
    s = a1921.score(pred, ans, split_lots=["0002001", "0003001"])
    assert (s["n"], s["right"]) == (3, 2)                       # still in the totals
    assert (s["split"]["n"], s["split"]["right"]) == (2, 1)
    assert s["split"]["share"] == 0.5
    assert a1921.score(pred, ans)["split"]["n"] == 0


def test_missing_answers_dir_is_recorded_not_fatal(tmp_path):
    cases = [{"id": "c0001", "blklot": "0001001", "block": "0001", "code": "1R"}]
    ad, cf = _write(tmp_path, cases, [{"case": "c0001", "answer": "ok", "shown": "1R"}])
    gone, _ = a1921.page_inputs("1921", tmp_path)
    df = a1921.owner_answers([ad, gone], [cf])
    pages = df.attrs["pages"]
    assert pages["1921-likely"] == {"dir_exists": True, "read": 1, "kept": 1, "dropped": 0, "case_file": True}
    assert pages["1921"]["dir_exists"] is False and pages["1921"]["read"] == 0
    assert len(df) == 1


def test_answers_without_a_case_file_raise(tmp_path):
    ad, _ = _write(tmp_path, [], [{"case": "c0001", "answer": "ok", "shown": "1R"}])
    with pytest.raises(FileNotFoundError):
        a1921.owner_answers([ad], [])


def test_spelling_is_one_code():
    ans = _ans(["0001001"], ["NC-2"])
    assert a1921.score(pd.Series({"0001001": "NC2"}), ans)["right"] == 1


def test_truth_is_in_the_one_spelling(tmp_path):
    cases = [{"id": "c0001", "blklot": "0001001", "block": "0001", "code": "NC2"}]
    ad, cf = _write(tmp_path, cases, [{"case": "c0001", "answer": "ok", "shown": "NC2"}])
    assert a1921.owner_answers([ad], [cf])["truth"].tolist() == ["NC-2"]
