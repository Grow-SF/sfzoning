"""The fresh-clone check's comparison rules, on tiny fixtures (CPU only)."""
import importlib.util
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

_spec = importlib.util.spec_from_file_location(
    "compare_outputs", Path(__file__).parents[1] / "scripts" / "compare_outputs.py")
co = importlib.util.module_from_spec(_spec)
sys.modules["compare_outputs"] = co
_spec.loader.exec_module(co)


def _write(root, stage, rel, make):
    p = root / co.STAGE_DIRS[stage] / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    make(p)


def _fill(tmp_path, stage):
    """The stage's required files, the same on both sides, so only the file under test can differ."""
    for side in ("new", "ref"):
        for pat in co.REQUIRED[stage]:
            _write(tmp_path / side, stage, pat.replace("*", "filler"), lambda p: p.write_bytes(b"same"))


def _both(tmp_path, stage, rel, make_new, make_ref, fill=True):
    if fill:
        _fill(tmp_path, stage)
    _write(tmp_path / "new", stage, rel, make_new)
    _write(tmp_path / "ref", stage, rel, make_ref)
    return co.compare_stage(stage, tmp_path / "new", tmp_path / "ref")


def _png(arr):
    return lambda p: cv2.imwrite(str(p), np.array(arr, dtype=np.uint8))


def _json(d):
    return lambda p: p.write_text(json.dumps(d))


def test_equal_content_npz_with_different_bytes_is_identical(tmp_path):
    def a(p):
        np.savez(p, x=np.arange(5), y=np.array([1.0, np.nan]))

    def b(p):
        np.savez_compressed(p, x=np.arange(5), y=np.array([1.0, np.nan]))
    assert _both(tmp_path, "solver-lots", "lots.npz", a, b).status == "identical"


def test_changed_npz_array_differs(tmp_path):
    r = _both(tmp_path, "solver-lots", "lots.npz", lambda p: np.savez(p, x=np.arange(5)),
              lambda p: np.savez(p, x=np.arange(5) + (np.arange(5) == 3)))
    assert r.status == "differs" and "array x" in r.differs[0]


def test_parquet_compared_by_content(tmp_path):
    df = pd.DataFrame({"a": [1, 2]})
    same = _both(tmp_path, "solver-prints", "p.parquet", lambda p: df.to_parquet(p), lambda p: df.to_parquet(p))
    assert same.status == "identical"
    r = _both(tmp_path / "x", "solver-prints", "p.parquet", lambda p: df.to_parquet(p),
              lambda p: df.assign(a=[1, 3]).to_parquet(p))
    assert r.status == "differs"


def test_text_chance_png_within_two_counts_but_not_three(tmp_path):
    ref = [[10, 20], [30, 40]]
    near = _both(tmp_path, "solver-text", "chance/a.png", _png([[12, 18], [30, 40]]), _png(ref))
    assert near.status == "within epsilon"
    far = _both(tmp_path / "x", "solver-text", "chance/a.png", _png([[13, 20], [30, 40]]), _png(ref))
    assert far.status == "differs"


def test_png_in_an_exact_stage_must_match(tmp_path):
    assert _both(tmp_path, "solver-page", "ink.png", _png([[0, 255]]), _png([[1, 255]])).status == "differs"


def test_model_file_is_not_compared_and_alone_is_not_a_pass(tmp_path):
    r = _both(tmp_path, "solver-text", "model.pt", lambda p: p.write_bytes(b"a"), lambda p: p.write_bytes(b"b"),
              fill=False)
    assert r.status == "differs" and r.identical == 0
    assert all("model.pt" not in d for d in r.differs)
    with_chance = _both(tmp_path / "x", "solver-text", "model.pt", lambda p: p.write_bytes(b"a"),
                        lambda p: p.write_bytes(b"b"))
    assert with_chance.status == "identical"


def test_empty_stage_folders_differ(tmp_path):
    for side in ("new", "ref"):
        (tmp_path / side / co.STAGE_DIRS["solver-walls"]).mkdir(parents=True)
    r = co.compare_stage("solver-walls", tmp_path / "new", tmp_path / "ref")
    assert r.status == "differs" and "no files compared" in r.differs


def test_a_required_output_missing_on_one_side_differs(tmp_path):
    _fill(tmp_path, "solver-lots")
    (tmp_path / "ref" / co.STAGE_DIRS["solver-lots"] / "table.parquet").unlink()
    r = co.compare_stage("solver-lots", tmp_path / "new", tmp_path / "ref")
    assert r.status == "differs" and any("table.parquet" in d for d in r.differs)


def test_text_chance_png_that_is_not_uint8_differs(tmp_path):
    big = lambda v: (lambda p: cv2.imwrite(str(p), np.array([[v]], dtype=np.uint16)))
    assert _both(tmp_path, "solver-text", "chance/a.png", big(100), big(101)).status == "differs"


def test_text_scores_json_within_one_percent(tmp_path):
    ok = _both(tmp_path, "solver-text-scores", "e.json", _json({"auc": 0.995, "n": 5}), _json({"auc": 0.9954, "n": 5}))
    assert ok.status == "within epsilon"
    bad = _both(tmp_path / "x", "solver-text-scores", "e.json", _json({"auc": 0.95}), _json({"auc": 0.995}))
    assert bad.status == "differs"
    # the detector's own stage writes no JSON: one there is held byte for byte
    own = _both(tmp_path / "y", "solver-text", "e.json", _json({"auc": 0.995}), _json({"auc": 0.9954}))
    assert own.status == "differs"


def test_json_of_an_exact_stage_is_byte_exact(tmp_path):
    assert _both(tmp_path, "solver-solve", "r.json", _json({"a": 1.0}), _json({"a": 1.0000001})).status == "differs"


def test_missing_and_extra_files_differ(tmp_path):
    _fill(tmp_path, "solver-page")
    _write(tmp_path / "new", "solver-page", "only_new.txt", lambda p: p.write_text("x"))
    _write(tmp_path / "ref", "solver-page", "only_ref.txt", lambda p: p.write_text("x"))
    assert len(co.compare_stage("solver-page", tmp_path / "new", tmp_path / "ref").differs) == 2


def test_after_a_stage_is_compared_the_references_outputs_replace_the_clones(tmp_path, capsys):
    # the clone's text chances differ from the reference's within the epsilon; the stages after it must read the
    # reference's, so each is held to its own epsilon and not to the ones above it as well
    stage = "solver-text"
    _write(tmp_path / "new", stage, "chance/p01.png", _png([[0, 2]]))
    _write(tmp_path / "new", stage, "chance/p99.png", _png([[1]]))            # made only in the clone
    _write(tmp_path / "ref", stage, "chance/p01.png", _png([[0, 0]]))
    _write(tmp_path / "ref", stage, "model.pt", lambda p: p.write_bytes(b"weights"))
    assert co.compare_stage(stage, tmp_path / "new", tmp_path / "ref").status == "differs"
    assert co.main(["--use-reference", stage, str(tmp_path / "new"), str(tmp_path / "ref")]) == 0
    assert "copied" in capsys.readouterr().out
    new, ref = (tmp_path / side / co.STAGE_DIRS[stage] for side in ("new", "ref"))
    assert {p.relative_to(new): p.read_bytes() for p in new.rglob("*") if p.is_file()} == {
        p.relative_to(ref): p.read_bytes() for p in ref.rglob("*") if p.is_file()}
    assert co.compare_stage(stage, tmp_path / "new", tmp_path / "ref").status == "identical"
    # no reference outputs to use: refused, and the clone's left as they are
    assert co.main(["--use-reference", "solver-page", str(tmp_path / "new"), str(tmp_path / "ref")]) == 2
