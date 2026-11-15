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


def _both(tmp_path, stage, rel, make_new, make_ref):
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
    near = _both(tmp_path, "solver-text-nw", "chance/a.png", _png([[12, 18], [30, 40]]), _png(ref))
    assert near.status == "within epsilon"
    far = _both(tmp_path / "x", "solver-text-nw", "chance/a.png", _png([[13, 20], [30, 40]]), _png(ref))
    assert far.status == "differs"


def test_png_in_an_exact_stage_must_match(tmp_path):
    assert _both(tmp_path, "solver-page", "ink.png", _png([[0, 255]]), _png([[1, 255]])).status == "differs"


def test_model_file_is_not_compared(tmp_path):
    r = _both(tmp_path, "solver-text-nw", "model.pt", lambda p: p.write_bytes(b"a"), lambda p: p.write_bytes(b"b"))
    assert r.status == "identical" and r.identical == 0


def test_text_scores_json_within_one_percent(tmp_path):
    ok = _both(tmp_path, "solver-text-scores", "e.json", _json({"auc": 0.995, "n": 5}), _json({"auc": 0.9954, "n": 5}))
    assert ok.status == "within epsilon"
    bad = _both(tmp_path / "x", "solver-text-scores", "e.json", _json({"auc": 0.95}), _json({"auc": 0.995}))
    assert bad.status == "differs"


def test_json_of_an_exact_stage_is_byte_exact(tmp_path):
    assert _both(tmp_path, "solver-solve", "r.json", _json({"a": 1.0}), _json({"a": 1.0000001})).status == "differs"


def test_missing_and_extra_files_differ(tmp_path):
    _write(tmp_path / "new", "solver-page", "only_new.txt", lambda p: p.write_text("x"))
    _write(tmp_path / "ref", "solver-page", "only_ref.txt", lambda p: p.write_text("x"))
    assert len(co.compare_stage("solver-page", tmp_path / "new", tmp_path / "ref").differs) == 2
