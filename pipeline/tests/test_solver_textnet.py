import contextlib
import dataclasses
import gzip
import json

import cv2
import numpy as np
import pytest
import torch
import torch.nn.functional as F

from sfzmaps import config
from sfzmaps.solver import grid, paths
from sfzmaps.solver.textnet import boxes, data, evaluate, model, predict, stage, train
from sfzmaps.transform import SheetTransform

SHEET = "2000-digital-zoning-p01"
SIZE = 640
SPLIT_PX = 319.5          # the quarter split, page px both ways: px 0-319 west / north, 320 on east / south
WORD_A = [100, 100, 180, 130]     # a zoning code read left to right, in nw
WORD_B = [450, 80, 470, 200]      # a street name read bottom to top, in ne
OTHER_NEAR_A = [160, 110, 230, 125]   # another pass's box, half on word A
OTHER_ALONE = [100, 400, 160, 430]    # another pass's box on no Terra box, in sw
SOL_IN_A = (175, 115)             # a Sol label point inside word A's box
SOL_ALONE = (500, 500)            # a Sol label point far from any box, in se


@contextlib.contextmanager
def torch_settings_kept():
    """Put back what `train.settle_torch` and the seeds change for the whole process (torch's threads, its
    deterministic mode and its random state) when the body ends."""
    threads = torch.get_num_threads()
    deterministic = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    rng = torch.get_rng_state()
    try:
        yield
    finally:
        torch.set_num_threads(threads)
        torch.use_deterministic_algorithms(deterministic, warn_only=warn_only)
        torch.set_rng_state(rng)


@pytest.fixture(autouse=True)
def _torch_as_it_was():
    """Every test here leaves torch as it found it, for the tests that run after it in the same process."""
    with torch_settings_kept():
        yield


def test_torch_settings_come_back_after_settle_torch():
    with torch_settings_kept():
        torch.set_num_threads(2)
        torch.use_deterministic_algorithms(False)
        with torch_settings_kept():
            train.settle_torch()
            torch.manual_seed(123)
            assert torch.get_num_threads() == train.THREADS and torch.are_deterministic_algorithms_enabled()
        assert torch.get_num_threads() == 2 and not torch.are_deterministic_algorithms_enabled()


def _page() -> np.ndarray:
    """The synthetic page: white paper, word A's letters and word B's (drawn turned a quarter), a stray mark."""
    pg = np.full((SIZE, SIZE), 255, np.uint8)
    cv2.putText(pg, "RH-1", (102, 128), cv2.FONT_HERSHEY_SIMPLEX, 0.9, 0, 2)
    turned = np.full((20, 120), 255, np.uint8)
    cv2.putText(turned, "ELM ST", (2, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 0, 2)
    pg[80:200, 450:470] = np.rot90(turned)           # reads bottom to top
    pg[410:420, 110:150] = 0                          # ink under the other pass's lone box
    cv2.line(pg, (0, 300), (639, 300), 0, 2)          # a printed line: ink that is not text
    return pg


def _transform_json() -> dict:
    """An affine placement, 4 ft per page px, north up, with the quarter split at page px (SPLIT_PX, SPLIT_PX)."""
    sx, sy = grid.QUARTER_SPLIT
    corners = [(0.0, 0.0), (640.0, 0.0), (640.0, 640.0), (0.0, 640.0)]
    gcps = [{"px": px, "py": py, "wx": sx + 4 * (px - SPLIT_PX), "wy": sy - 4 * (py - SPLIT_PX), "source": "manual"}
            for px, py in corners]
    return {"smoothing": 1.0, "gcps": gcps}


def _terra(text, box, kind, angle):
    return {"text": text, "kind": kind, "box": box, "angle_deg": angle, "edge": 100}


@pytest.fixture
def sheet(tmp_path, monkeypatch):
    """One synthetic sheet laid out as the stage reads it: its page, Sol's labels, its corrected placement, Terra's
    boxes and one other pass's boxes."""
    monkeypatch.setattr(config, "WORK_DIR", tmp_path / "work")
    monkeypatch.setattr(config, "DATA_DIR", tmp_path / "data")
    monkeypatch.setattr(paths, "PRINTS_DIR", tmp_path / "prints")
    monkeypatch.setattr(paths, "SAVED_ANSWERS", tmp_path / "answers")
    monkeypatch.setattr(paths, "SHEETS", (SHEET,))
    work = tmp_path / "work" / paths.SERIES / SHEET
    work.mkdir(parents=True)
    cv2.imwrite(str(work / "page.png"), _page())
    labels = [{"code": "RH-1", "kind": "use", "x": SOL_IN_A[0], "y": SOL_IN_A[1]},
              {"code": "RH-2", "kind": "use", "x": SOL_ALONE[0], "y": SOL_ALONE[1]}]
    (work / "district_labels.json").write_text(json.dumps({"labels": labels}))
    warp = tmp_path / "data" / "warp" / paths.SERIES / SHEET
    warp.mkdir(parents=True)
    (warp / "transform.json").write_text(json.dumps(_transform_json()))
    paths.PRINTS_DIR.mkdir()
    terra = [_terra("RH-1", WORD_A, "zoning code", 0), _terra("ELM ST", WORD_B, "street name", 90)]
    (paths.PRINTS_DIR / "terra_boxes.json").write_text(json.dumps({SHEET: terra, "another-sheet": []}))
    other = tmp_path / "answers" / data.OTHER_PASSES
    other.mkdir(parents=True)
    with gzip.open(other / "some-pass.jsonl.gz", "wt") as f:
        for box in (OTHER_NEAR_A, OTHER_ALONE):
            f.write(json.dumps({"sheet": SHEET, "text": "x", "kind": "other", "box": box, "angle_deg": 0}) + "\n")
        f.write(json.dumps({"sheet": "another-sheet", "text": "x", "kind": "other", "box": [300, 300, 340, 340],
                            "angle_deg": 0}) + "\n")
    return SHEET


def test_targets_mark_boxes_and_unknowns(sheet):
    t = data.sheet_targets(sheet, "ne")
    assert t.ink.shape == (SIZE, SIZE) and t.ink[300, 10] and not t.ink[5, 5]
    # Terra's boxes are text, with their kind and reading angle
    assert t.text[105:126, 105:176].all() and t.text[85:196, 452:469].all()
    assert t.text.sum() == 81 * 31 + 21 * 121                 # the two boxes, their edges included, nothing else
    assert (t.kind[105, 105], t.kind[100, 455]) == (data.KINDS["zoning code"], data.KINDS["street name"])
    assert t.kind[50, 50] == data.NO_KIND
    assert (t.angle[115, 140], t.angle[150, 460]) == (0, 90)
    # unknown: other passes' boxes and discs at Sol's points, minus every Terra box; not another sheet's boxes
    assert t.unknown[115, 200] and not t.unknown[115, 170]       # the other box beyond word A, and inside it
    assert t.unknown[415, 130] and t.unknown[500, 529] and not t.unknown[500, 532]
    assert t.unknown[140, 175] and not t.unknown[115, 150]       # Sol's point in A: its disc is unknown only outside A
    assert not t.unknown[320, 320]
    # held out: the ne quarter (rows 0-319, columns 320-639) by the placement and the split
    assert t.heldout[:320, 320:].all() and t.heldout.sum() == 320 * 320
    assert not t.known[:320, 320:].any()
    assert np.array_equal(t.known, ~t.unknown & ~t.heldout)
    # crops centre on ink or text outside the held-out quarter
    assert t.centre_ok[300, 10] and t.centre_ok[115, 140] and not t.centre_ok[150, 460] and not t.centre_ok[300, 400]
    assert not t.centre_ok[50, 50]


def test_model_output_shape():
    net = model.TextNet()
    out = net(torch.zeros(2, 1, 64, 96))
    assert out.shape == (2, model.OUTPUTS, 64, 96) and model.OUTPUTS == 7
    assert sum(isinstance(m, torch.nn.Conv2d) for m in net.modules()) == 2 * (5 + 4) + 1


@pytest.fixture
def small_crops(monkeypatch):
    """Smaller batches than the real ones, so CPU tests stay quick; nothing else changes."""
    monkeypatch.setattr(train, "CROP", 64)
    monkeypatch.setattr(train, "BATCH", 4)


def test_three_iterations_twice_give_identical_weights_on_cpu(sheet, small_crops):
    t = data.sheet_targets(sheet, "ne")
    first = train.fit([t], iterations=3, device="cpu", seed=0).state_dict()
    second = train.fit([t], iterations=3, device="cpu", seed=0).state_dict()
    torch.manual_seed(0)
    untrained = model.TextNet().state_dict()
    assert first.keys() == second.keys()
    assert all(torch.equal(first[k], second[k]) for k in first)
    assert not all(torch.equal(first[k], untrained[k]) for k in first)


def test_crop_turns_the_reading_angle_with_the_picture(sheet, small_crops):
    t = data.sheet_targets(sheet, "ne")
    flat = train.crop(t, 140, 460, scale=1.0, turns=0)
    turned = train.crop(t, 140, 460, scale=1.0, turns=1)
    assert np.array_equal(turned["ink"], np.rot90(flat["ink"]))
    on = turned["text"] > 0
    assert on.any() and (turned["angle"][on] == -180).all()      # 90 + 90 = 180, read in [-180, 180)
    assert (flat["angle"][flat["text"] > 0] == 90).all()


def _loss_and_grads(net, parts):
    x, t, k, a, w = train.to_tensors([parts], "cpu")
    net.zero_grad()
    loss, _ = train.loss_fn(net(x), t, k, a, w)
    loss.backward()
    return loss.item(), [p.grad.clone() for p in net.parameters()]


def test_heldout_pixels_carry_no_loss(sheet, small_crops):
    t = data.sheet_targets(sheet, "ne")
    torch.manual_seed(0)
    net = model.TextNet()
    # a crop made only of the held-out quarter, on word B: no loss, no gradient
    inside = train.crop(t, 140, 460, scale=1.0, turns=0)
    assert not inside["known"].any() and inside["text"].any()
    loss, grads = _loss_and_grads(net, inside)
    assert loss == 0 and all((g == 0).all() for g in grads)
    # a crop across the split: changing the held-out pixels' targets changes nothing
    across = train.crop(t, 200, 330, scale=1.0, turns=0)
    held = ~across["known"] & ~t.unknown[168:232, 298:362]
    assert held.any() and across["known"].any()
    loss, grads = _loss_and_grads(net, across)
    flipped = {**across, "text": np.where(held, 1 - across["text"], across["text"]),
               "kind": np.where(held, 1, across["kind"]), "angle": np.where(held, 45, across["angle"])}
    loss2, grads2 = _loss_and_grads(net, flipped)
    assert loss2 == loss and all(torch.equal(g, g2) for g, g2 in zip(grads, grads2))


def _boolean_indexed_loss(out, t, k, a, w):
    """The loss as it was before it became masked sums: kind and angle computed on the known text px only, picked out
    by boolean indexing (whose backward mps can't run deterministically)."""
    lt = (F.binary_cross_entropy_with_logits(out[:, 0], t, reduction="none") * w).sum() / w.sum().clamp(min=1)
    m = (t > 0) & (w > 0)
    lk = F.cross_entropy(out[:, 1:5].permute(0, 2, 3, 1)[m], k[m])
    cs = out[:, 5:7].permute(0, 2, 3, 1)[m]
    rad = torch.deg2rad(a[m])
    la = (1 - (cs[:, 0] * torch.cos(rad) + cs[:, 1] * torch.sin(rad)) / cs.norm(dim=1).clamp(min=1e-6)).mean()
    return lt + train.KIND_WEIGHT * lk + train.ANGLE_WEIGHT * la


def test_masked_loss_equals_the_boolean_indexed_loss_even_where_the_direction_output_is_zero():
    g = torch.Generator().manual_seed(0)
    n, h, w_ = 3, 12, 10
    t = (torch.rand(n, h, w_, generator=g) < 0.4).double()
    w = (torch.rand(n, h, w_, generator=g) < 0.7).double()
    k = torch.where(t > 0, torch.randint(0, 4, (n, h, w_), generator=g), torch.full((n, h, w_), data.NO_KIND))
    a = torch.where(t > 0, torch.randint(-180, 180, (n, h, w_), generator=g).double(), torch.zeros(n, h, w_).double())
    out = torch.randn(n, model.OUTPUTS, h, w_, generator=g, dtype=torch.float64)
    off = ~((t > 0) & (w > 0))
    out[:, 5:7] = torch.where(off[:, None], 0.0, out[:, 5:7])     # (cos, sin) exactly (0, 0) off the known text
    grads = []
    for fn in (lambda o: train.loss_fn(o, t, k, a, w)[0], lambda o: _boolean_indexed_loss(o, t, k, a, w)):
        o = out.clone().requires_grad_(True)
        loss = fn(o)
        loss.backward()
        grads.append((loss.detach(), o.grad))
    (masked, g_masked), (indexed, g_indexed) = grads
    assert torch.isfinite(masked) and torch.isfinite(g_masked).all()
    assert torch.allclose(masked, indexed, rtol=1e-12, atol=0)
    assert torch.allclose(g_masked, g_indexed, rtol=1e-10, atol=1e-15)


def test_fit_stops_at_the_first_loss_that_is_not_finite(sheet, small_crops, monkeypatch):
    t = data.sheet_targets(sheet, "ne")
    real, seen = train.loss_fn, []

    def loss_fn(*args):
        loss, parts = real(*args)
        seen.append(parts)
        return (loss * float("nan"), (float("nan"),) * 3) if len(seen) == 2 else (loss, parts)

    monkeypatch.setattr(train, "loss_fn", loss_fn)
    with pytest.raises(FloatingPointError, match="iteration 2"):
        train.fit([t], iterations=5, device="cpu", seed=0)
    assert len(seen) == 2


@pytest.fixture
def solver_dir(tmp_path, monkeypatch):
    """The solver's outputs in the test's folder."""
    monkeypatch.setattr(paths, "SOLVER_DIR", tmp_path / "solver")
    monkeypatch.setattr(paths, "TEXT_SCORES_DIR", tmp_path / "solver" / "text-scores")
    return tmp_path / "solver"


def test_a_quarter_trains_and_scores_in_one_exclusive_turn_and_writes_only_the_model_and_chances(
        sheet, small_crops, solver_dir, monkeypatch):
    turns, calls = [], []

    @contextlib.contextmanager
    def turn(exclusive, device):
        turns.append((exclusive, device))
        yield
        turns.append(None)

    def inside(fn, what):
        def wrapped(*args, **kwargs):
            calls.append((what, turns[-1:] == [(True, "cpu")]))
            return fn(*args, **kwargs)
        return wrapped

    monkeypatch.setattr(stage.gpu, "gpu_turn", turn)
    monkeypatch.setattr(data, "sheet_targets", inside(data.sheet_targets, "targets"))
    monkeypatch.setattr(train, "loss_fn", inside(train.loss_fn, "step"))
    monkeypatch.setattr(predict, "chance", inside(predict.chance, "scored"))
    paths.text_dir("ne").mkdir(parents=True)
    (paths.text_dir("ne") / "evaluation.json").write_text("{}")       # a file no step of the stage writes
    s = stage.build_text("ne", iterations=2, device="cpu")
    # the pages' targets are read before the turn; every step and every sheet scored is inside it
    assert turns == [(True, "cpu"), None]
    assert calls == [("targets", False), ("step", True), ("step", True), ("scored", True)]
    assert sorted(p.name for p in paths.text_dir("ne").iterdir()) == ["chance", "model.pt"]
    assert [p.name for p in paths.chance_dir("ne").iterdir()] == [f"{SHEET}.png"]
    saved = torch.load(paths.text_dir("ne") / "model.pt")
    assert (saved["iterations"], saved["device"], saved["seed"]) == (2, "cpu", 0)
    assert (s["quarter"], s["iterations"], s["device"], s["sheets"]) == ("ne", 2, "cpu", 1)


def test_a_quarter_trains_the_default_steps_unless_told_and_never_zero(solver_dir, monkeypatch):
    steps = []
    monkeypatch.setattr(train, "quarter_targets", lambda q: [])
    monkeypatch.setattr(train, "fit", lambda sheets, iterations, device, seed=0: steps.append(iterations) or "net")
    monkeypatch.setattr(train, "save_model", lambda *args: None)
    monkeypatch.setattr(predict, "predict_sheets", lambda q, device: 0)
    stage.build_text("nw", device="cpu")
    stage.build_text("nw", iterations=5, device="cpu")
    assert steps == [train.ITERATIONS, 5]
    for bad in (0, -1):
        with pytest.raises(ValueError, match="at least 1"):
            stage.build_text("nw", iterations=bad, device="cpu")
    with pytest.raises(ValueError, match="no quarter"):
        stage.build_text("north", device="cpu")
    assert steps == [train.ITERATIONS, 5]


def test_the_text_command_runs_the_stage_and_refuses_zero_iterations(monkeypatch):
    from typer.testing import CliRunner

    from sfzmaps import cli
    got = []

    def build_text(q, iterations=None, device=None):
        got.append((q, iterations, device))
        return {"quarter": q, "iterations": iterations or train.ITERATIONS, "device": device or "cpu", "sheets": 13,
                "seconds": {"targets": 1.0, "train": 2.0, "predict": 3.0}}

    monkeypatch.setattr(stage, "build_text", build_text)
    runner = CliRunner()
    assert runner.invoke(cli.app, ["solver", "text", "nw", "--iterations", "0"]).exit_code != 0
    assert runner.invoke(cli.app, ["solver", "text", "nowhere"]).exit_code != 0
    res = runner.invoke(cli.app, ["solver", "text", "nw", "--device", "cpu"])
    assert res.exit_code == 0 and "13 sheets scored" in res.output
    assert got == [("nw", None, "cpu")]


def test_scoring_without_a_model_leaves_the_last_chances_alone(sheet, solver_dir):
    old = paths.chance_dir("ne")
    old.mkdir(parents=True)
    (old / f"{SHEET}.png").write_bytes(b"the last run's")
    with pytest.raises(FileNotFoundError, match="model.pt"):
        predict.predict_sheets("ne", device="cpu")
    assert (old / f"{SHEET}.png").read_bytes() == b"the last run's"


def test_scores_write_one_evaluation_per_quarter_from_one_set_of_district_lines(solver_dir, monkeypatch):
    made = []
    monkeypatch.setattr(evaluate, "district_lines", lambda: made.append("lines") or np.ones((2, 2), bool))
    monkeypatch.setattr(evaluate, "score_quarter", lambda q, lines: {"quarter": q, "line_px": int(lines.sum())})
    out = evaluate.build_scores()
    assert made == ["lines"] and list(out) == list(paths.QUARTERS)
    for q in paths.QUARTERS:
        assert json.loads((paths.TEXT_SCORES_DIR / f"evaluation_{q}.json").read_text()) == {"quarter": q, "line_px": 4}


def test_district_lines_are_borders_between_known_different_codes():
    grown = np.array([[0, 0, 1, 1, 2, 2],
                      [0, 0, 1, 1, 2, 2],
                      [3, 3, 3, 3, 3, 3]], np.int32)
    codes = np.array(["RH-1", "RH-2", "RH-2", ""])          # lot 3's code is not known
    known = np.array([True, True, True, False])
    lines = evaluate.district_line_px(grown, codes, known)
    # one px per crack between lots of different known codes: the px right of (or below) the crack
    expected = np.zeros(grown.shape, bool)
    expected[0:2, 2] = True                                  # lots 0 | 1; 1 | 2 share RH-2; 3 is not known
    assert np.array_equal(lines, expected)


def _per_sheet(folder: str, names) -> list[str]:
    return [f"{folder}/{sheet}/{name}" for sheet in paths.SHEETS for name in names]


def test_text_stages_depend_on_the_detector_code_only():
    from sfzmaps import build
    table = build.stages()
    for q in paths.QUARTERS:
        st = table[f"solver-text-{q}"]
        deps = build.stage_deps(st)
        assert st.code == ("sfzmaps.solver.textnet.stage",)
        assert {"src/sfzmaps/solver/textnet/stage.py", "src/sfzmaps/solver/textnet/train.py",
                "src/sfzmaps/solver/textnet/predict.py", "src/sfzmaps/solver/textnet/data.py",
                "src/sfzmaps/solver/textnet/model.py", "src/sfzmaps/solver/grid.py",
                "data/solver/prints/terra_boxes.json"} <= set(deps)
        # hours of training must not rerun for a change to the lots, the table, the merged page, the codes, the
        # prints' code, the model client, the command line, the scoring, the walls' or the solve's folders (runs.py)
        # or the GPU's turns (which decide when it computes, never what); of the prints' output only the Terra boxes
        # are read, and the lots not at all
        assert not any(d.endswith(("solver/lots.py", "solver/table.py", "solver/page.py", "solver/frame.py",
                                   "solver/codes.py", "solver/prints.py", "/llm.py", "/cli.py", "textnet/evaluate.py",
                                   "solver/runs.py", "/gpu.py"))
                       for d in deps)
        assert [d for d in deps if d.startswith("data/solver/")] == ["data/solver/prints/terra_boxes.json"]
        # of each sheet only the files the detector reads: its placement and correction, its page, Sol's points;
        # nothing else of data/warp or data/work (another file the warp or the label stage writes retrains nothing)
        assert [d for d in deps if d.startswith(("data/warp", "data/work"))] == (
            _per_sheet(f"data/warp/{paths.SERIES}", ("transform.json", "correction.npz"))
            + _per_sheet(f"data/work/{paths.SERIES}", ("page.png", "district_labels.json")))
        assert st.upstream == ("solver-prints",) and build.stage_outs(st) == [f"data/solver/text-{q}"]
        assert build.stage_commands(f"solver-text-{q}") == [["solver", "text", q]]


def test_the_text_scores_stage_reads_every_quarters_chances_not_the_detector_code():
    from sfzmaps import build
    table = build.stages()
    st = table["solver-text-scores"]
    deps = build.stage_deps(st)
    assert st.code == ("sfzmaps.solver.textnet.evaluate",)
    assert set(st.upstream) == {f"solver-text-{q}" for q in paths.QUARTERS}
    assert {f"data/solver/text-{q}/chance" for q in paths.QUARTERS} | {"data/solver/lots"} <= set(deps)
    assert "src/sfzmaps/solver/textnet/evaluate.py" in deps and "src/sfzmaps/solver/lots.py" in deps
    # the scores read the chances, not the weights or the code that made them
    assert not any(d.endswith(("textnet/train.py", "textnet/predict.py", "textnet/model.py", "textnet/stage.py",
                               "/gpu.py", "model.pt")) for d in deps)
    # of each sheet only the files the scores read: the detector's own, and the warped footprint
    assert [d for d in deps if d.startswith(("data/warp", "data/work"))] == (
        _per_sheet(f"data/warp/{paths.SERIES}", ("transform.json", "correction.npz"))
        + _per_sheet(f"data/work/{paths.SERIES}", ("page.png", "district_labels.json"))
        + _per_sheet(f"data/warp/{paths.SERIES}", ("warped.footprint.geojson",)))
    assert build.stage_outs(st) == ["data/solver/text-scores"]
    assert build.stage_commands("solver-text-scores") == [["solver", "text-scores"]]
    names = list(table)
    assert names.index("solver-text-scores") > max(names.index(f"solver-text-{q}") for q in paths.QUARTERS)


# --- the training data of many maps, held out by an area -------------------------------------------------------------

OLD, OLD_SHEET = "1960", "3"          # a sheet of an older map, at twice the 2000 sheets' px per foot
NEW_CORNER = (319.5, 319.5)           # page px of VALIDATION_AREA's north-west corner on the synthetic 2000 page
OLD_CORNER = (199.5, 449.5)           # ... and on the synthetic 1960 page
WORD_C = [300, 310, 360, 330]         # a block number across the validation area's west edge (2000 page)
WORD_D = [380, 400, 460, 450]         # slanted "other" text inside the area (2000 page)
WORD_E = [440, 430, 520, 470]         # a street name over part of word D, drawn after it (2000 page)
OLD_WORDS = [("R-1-D", [200, 150, 300, 190], "zoning code", 0),
             ("MAIN ST", [400, 300, 480, 380], "street name", -45),
             ("2", [180, 470, 230, 500], "other", 0)]          # across the area's west and north edges (1960 page)


def _placed_at_area(ft_per_px: float, corner) -> dict:
    """An affine placement, `ft_per_px` ft per page px, north up, with VALIDATION_AREA's north-west corner at page
    px `corner` (col, row): the page px east of and below it are inside the area."""
    ax0, _, _, ay1 = data.VALIDATION_AREA
    cx, cy = corner
    gcps = [{"px": px, "py": py, "wx": ax0 + ft_per_px * (px - cx), "wy": ay1 - ft_per_px * (py - cy),
             "source": "manual"} for px, py in [(0.0, 0.0), (640.0, 0.0), (640.0, 640.0), (0.0, 640.0)]]
    return {"smoothing": 1.0, "gcps": gcps}


def _record(series, sheet, page_box, items, window="0,0,640,640"):
    """One saved answer of `text-on-every-sheet`: Terra's items in one tile (`page_box`) of a sheet's page."""
    return {"series": series, "sheet": sheet, "page_box": list(page_box), "answer": {"items": items},
            "script": {"args": ["gpt-6.1-terra", series, sheet, window, "<out>"]}}


def _word(text, box, kind="zoning code", angle=0, tile=(0, 0)):
    """A Terra item as saved, its box (page px) given in its tile's px."""
    x0, y0, x1, y1 = box
    return {"angle_deg": angle, "direction": "left-to-right" if angle == 0 else "slanted", "kind": kind,
            "text": text, "x0": x0 - tile[0], "x1": x1 - tile[0], "y0": y0 - tile[1], "y1": y1 - tile[1]}


def _write_gz(path, lines):
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt") as f:
        for line in lines:
            f.write(json.dumps(line) + "\n")


def _old_page() -> np.ndarray:
    pg = np.full((SIZE, SIZE), 255, np.uint8)
    cv2.putText(pg, "R-1-D", (202, 186), cv2.FONT_HERSHEY_SIMPLEX, 1.0, 0, 2)
    cv2.line(pg, (400, 380), (480, 300), 0, 3)
    cv2.putText(pg, "2", (190, 498), cv2.FONT_HERSHEY_SIMPLEX, 0.9, 0, 2)
    cv2.line(pg, (0, 560), (639, 560), 0, 2)
    return pg


@pytest.fixture
def maps(tmp_path, monkeypatch):
    """Two synthetic sheets laid out as `load_sheets` reads them: the 2000 sheet (4 ft per px) and a 1960 sheet (2 ft
    per px), each with its page, Sol's labels and its corrected placement; Terra's saved answers for both, in tiles;
    one other pass's boxes (of the 2000 sheet, and of a sheet named like the 1960 one)."""
    monkeypatch.setattr(config, "WORK_DIR", tmp_path / "work")
    monkeypatch.setattr(config, "DATA_DIR", tmp_path / "data")
    monkeypatch.setattr(paths, "SAVED_ANSWERS", tmp_path / "answers")
    monkeypatch.setattr(paths, "TEXT_SHEETS", {paths.SERIES: (SHEET,), OLD: (OLD_SHEET,)})
    new_page = _page()
    cv2.putText(new_page, "12", (305, 328), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 0, 2)
    cv2.line(new_page, (385, 440), (455, 405), 0, 3)
    cv2.putText(new_page, "OAK", (445, 465), cv2.FONT_HERSHEY_SIMPLEX, 1.0, 0, 2)
    layout = {
        (paths.SERIES, SHEET): (new_page, _placed_at_area(4.0, NEW_CORNER), [(175, 115), (500, 500), (330, 300)]),
        (OLD, OLD_SHEET): (_old_page(), _placed_at_area(2.0, OLD_CORNER), [(250, 170), (420, 520)]),
    }
    for (series, sheet), (pg, placed, points) in layout.items():
        work = tmp_path / "work" / series / sheet
        work.mkdir(parents=True)
        cv2.imwrite(str(work / "page.png"), pg)
        labels = [{"code": "X", "kind": "use", "x": float(x), "y": float(y)} for x, y in points]
        (work / "district_labels.json").write_text(json.dumps({"labels": labels}))
        warp = tmp_path / "data" / "warp" / series / sheet
        warp.mkdir(parents=True)
        (warp / "transform.json").write_text(json.dumps(placed))
    tile = {(tx, ty): [tx, ty, tx + 320, ty + 320] for tx in (0, 320) for ty in (0, 320)}   # four tiles of 320 px
    terra = [
        _record(paths.SERIES, SHEET, tile[0, 0], [_word("RH-1", WORD_A)]),
        _record(paths.SERIES, SHEET, tile[320, 0], [_word("ELM ST", WORD_B, "street name", 90, (320, 0))]),
        _record(paths.SERIES, SHEET, tile[0, 320], [_word("12", WORD_C, "block number", 0, (0, 320))]),
        _record(paths.SERIES, SHEET, tile[320, 320], [_word("OAK", WORD_E, "street name", 0, (320, 320)),
                                                      _word("X", WORD_D, "other", 30, (320, 320))]),
        _record(OLD, OLD_SHEET, tile[0, 0], [_word(t, b, k, a) for t, b, k, a in OLD_WORDS]),
    ]
    _write_gz(tmp_path / "answers" / "text-on-every-sheet.jsonl.gz", terra)
    _write_gz(tmp_path / "answers" / data.OTHER_PASSES / "some-pass.jsonl.gz",
              [{"sheet": s, "text": "x", "kind": "other", "box": b, "angle_deg": 0}
               for s, b in ((SHEET, OTHER_NEAR_A), (SHEET, OTHER_ALONE), (OLD_SHEET, [300, 300, 340, 340]))])
    return tmp_path


def _reference_page(series: str, sheet: str, hold_out: bool) -> dict[str, np.ndarray]:
    """The whole page's targets drawn the plain way, for comparison: Terra's boxes (`boxes.terra_boxes`), each
    word's rectangle (`data.word_rect`) filled in order on one page-sized raster per target (padded, so no shape is cut
    at the page's edge); the other passes' rectangles (2000 only) and discs at Sol's points, minus text, unknown; the
    px whose map point (`data.page_world`) is in VALIDATION_AREA held out."""
    gray = cv2.imread(str(config.WORK_DIR / series / sheet / "page.png"), cv2.IMREAD_GRAYSCALE)
    h, w = gray.shape
    pad = 400
    text = np.zeros((h + 2 * pad, w + 2 * pad), np.uint8)
    kind = np.full(text.shape, data.NO_KIND, np.uint8)
    angle = np.zeros(text.shape, np.int16)
    other = np.zeros(text.shape, np.uint8)

    def fill(raster, rect, value):
        cv2.fillPoly(raster, [(np.round(rect).astype(np.int32) + pad).reshape(-1, 1, 2)], value)

    for b in boxes.terra_boxes(series, sheet):
        rect = data.word_rect(b["box"], float(b["angle_deg"]))
        fill(text, rect, 1)
        fill(kind, rect, data.KINDS[b["kind"]])
        fill(angle, rect, data.reading_angle(rect))
    if series == paths.SERIES:
        for b in data.other_boxes(sheet):
            fill(other, data.word_rect(b["box"], 0.0), 1)
    for lb in json.loads((config.WORK_DIR / series / sheet / "district_labels.json").read_text())["labels"]:
        cv2.circle(other, (int(round(lb["x"])) + pad, int(round(lb["y"])) + pad), data.READ_POINT_RADIUS, 1, -1)
    cut = (slice(pad, pad + h), slice(pad, pad + w))
    text, kind, angle, other = text[cut], kind[cut], angle[cut], other[cut]
    folder = config.DATA_DIR / "warp" / series / sheet
    x, y = data.page_world(SheetTransform.from_json(json.loads((folder / "transform.json").read_text()), folder),
                           (h, w))
    ax0, ay0, ax1, ay1 = data.VALIDATION_AREA
    held = (x >= ax0) & (x < ax1) & (y >= ay0) & (y < ay1) & hold_out
    return {"ink": gray < grid.INK_BELOW, "text": text, "kind": kind, "angle": angle,
            "known": ~((other > 0) & (text == 0)) & ~held}


def _reference_crop(page: dict[str, np.ndarray], y, x, size, scale, turns) -> dict[str, np.ndarray]:
    """A crop of the whole page's targets: `size` / `scale` px of page centred on (y, x), moved inside the page,
    resized to size x size (nearest px), turned `turns` quarter turns, the reading angle with it."""
    h, w = page["ink"].shape
    src = int(round(size / scale))
    y0, x0 = int(np.clip(y - src // 2, 0, h - src)), int(np.clip(x - src // 2, 0, w - src))
    out = {}
    for k, v in page.items():
        v = v[y0:y0 + src, x0:x0 + src]
        if src != size:
            v = cv2.resize(v.astype(np.uint8) if v.dtype == bool else v, (size, size),
                           interpolation=cv2.INTER_NEAREST).astype(v.dtype)
        out[k] = v
    out["angle"] = np.where(out["text"] > 0, (out["angle"].astype(np.int32) + 90 * turns + 180) % 360 - 180,
                            0).astype(np.int16)
    return {k: np.rot90(v, turns) for k, v in out.items()}


def test_terra_boxes_decode_for_another_series():
    records = [
        _record("1960", "3", [0, 0, 768, 768], [_word("R-1-D", [700, 300, 760, 330])], window="0,0,5257,6600"),
        _record("1960", "3", [600, 0, 1368, 768], [_word("R-1-D", [700, 300, 760, 330], tile=(600, 0)),   # read again,
                                                   _word("MAIN ST", [800, 400, 900, 420], "street name",  # farther in
                                                         tile=(600, 0))], window="0,0,5257,6600"),
        _record("1964", "3", [0, 0, 768, 768], [_word("C-2", [10, 10, 50, 30])]),   # another map's sheet "3"
        _record("1960", "4", [0, 0, 768, 768], [_word("R-2", [10, 10, 50, 30])]),
    ]
    got = boxes.terra_boxes("1960", "3", records)
    # page px: each item moved by its tile's corner; the word read twice is kept once, from the tile it is farther
    # inside (edges 100 and 8 px); farthest from its tile's edge first
    assert [(b["text"], b["box"], b["tile"]) for b in got] == [("MAIN ST", [800, 400, 900, 420], [600, 0]),
                                                                ("R-1-D", [700, 300, 760, 330], [600, 0])]
    assert [b["text"] for b in boxes.terra_boxes("1964", "3", records)] == ["C-2"]
    assert boxes.terra_boxes("1960", "5", records) == []


def test_validation_area_pixels_carry_no_loss_even_inside_a_crop_centred_outside(maps):
    new, = data.load_sheets([paths.SERIES], hold_out_validation=True)
    kept, = data.load_sheets([paths.SERIES], hold_out_validation=False)
    # a crop centred outside the area (rows and columns 268-331 of the page) whose bottom-right corner is in it
    # (rows and columns 320 on): word C crosses into that corner
    c = data.crop_targets(new, 300, 300, 64, 1.0, 0)
    whole = data.crop_targets(kept, 300, 300, 64, 1.0, 0)
    inside = np.zeros((64, 64), bool)
    inside[52:, 52:] = True
    assert c["text"][inside].any() and c["ink"][inside].any()
    assert not c["known"][inside].any() and whole["known"][inside].all()
    assert np.array_equal(c["known"][~inside], whole["known"][~inside]) and c["known"][~inside].any()
    # so they carry no loss: changing their targets changes neither the loss nor any gradient
    torch.manual_seed(0)
    net = model.TextNet()
    loss, grads = _loss_and_grads(net, c)
    flipped = {**c, "text": np.where(inside, 1 - c["text"], c["text"]), "kind": np.where(inside, 2, c["kind"]),
               "angle": np.where(inside, 45, c["angle"])}
    loss2, grads2 = _loss_and_grads(net, flipped)
    assert loss2 == loss and all(torch.equal(g, g2) for g, g2 in zip(grads, grads2))
    # and no crop centres on them, on text or on ink; without the hold-out some do
    for sd, some_inside in ((new, False), (kept, True)):
        on_text, on_ink = data.centres(sd, 4)
        assert len(on_text) and len(on_ink)
        for pool in (on_text, on_ink):
            assert ((pool[:, 0] >= 320) & (pool[:, 1] >= 320)).any() == some_inside
            assert (pool % 4 == 0).all()


def test_crop_targets_equal_full_page_drawing(maps):
    sheets = data.load_sheets([paths.SERIES, OLD], hold_out_validation=True)
    assert [(s.series, s.sheet) for s in sheets] == [(paths.SERIES, SHEET), (OLD, OLD_SHEET)]
    centres = [(115, 140), (140, 460), (320, 330), (430, 450), (5, 630), (639, 0), (330, 300), (470, 200)]
    for sd in sheets:
        page = _reference_page(sd.series, sd.sheet, hold_out=True)
        assert not page["known"].all() and page["text"].any()
        for i, (y, x) in enumerate(centres):
            for size, scale in ((64, 1.0), (64, 0.6), (96, 1.6), (40, 0.37)):
                turns = (i + size) % 4
                got = data.crop_targets(sd, y, x, size, scale, turns)
                want = _reference_crop(page, y, x, size, scale, turns)
                assert sorted(got) == sorted(want)
                for k in want:
                    assert got[k].dtype == want[k].dtype and np.array_equal(got[k], want[k]), (sd.series, y, x, k)


def test_sheet_data_holds_no_full_page_target_rasters(maps):
    for sd in data.load_sheets([paths.SERIES, OLD], hold_out_validation=True):
        fields = {f.name: getattr(sd, f.name) for f in dataclasses.fields(sd)}
        # every field is an array or a plain value: nothing is held where its bytes would not be counted
        assert all(isinstance(v, (np.ndarray, str, int, float, bool)) for v in fields.values())
        arrays = {k: v for k, v in fields.items() if isinstance(v, np.ndarray)}
        assert all(v.base is None or v.base.nbytes == v.nbytes for v in arrays.values())
        ink = arrays["ink_bits"].nbytes
        assert ink <= SIZE * SIZE                                  # at most one byte per page px
        box_bytes = sum(v.nbytes for k, v in arrays.items() if k in data.BOX_FIELDS)
        assert sum(v.nbytes for v in arrays.values()) <= 1.2 * ink + box_bytes
        assert np.array_equal(data.page_ink(sd), _reference_page(sd.series, sd.sheet, True)["ink"])


def test_each_map_is_read_at_its_own_scale_with_its_own_unknowns(maps, monkeypatch):
    new, old = data.load_sheets([paths.SERIES, OLD], hold_out_validation=True)
    assert new.ft_per_px == pytest.approx(4.0) and old.ft_per_px == pytest.approx(2.0)
    # the other passes are of the 2000 sheets only: their lone box is unknown on the 2000 page; on the 1960 page,
    # the box an other pass gives a sheet named "3" is not
    assert len(new.other) == 2 and len(old.other) == 0
    assert not data.crop_targets(new, 415, 130, 64, 1.0, 0)["known"][32, 32]
    assert data.crop_targets(old, 320, 320, 64, 1.0, 0)["known"].all()
    # sheets named in the code but not in the saved answers stop the read
    monkeypatch.setattr(paths, "TEXT_SHEETS", {paths.SERIES: (SHEET, "2000-digital-zoning-p02"), OLD: (OLD_SHEET,)})
    with pytest.raises(ValueError, match="2000-digital-zoning-p02"):
        data.load_sheets([paths.SERIES], hold_out_validation=True)


def test_text_sheet_lists_match_the_saved_answers(tmp_path, monkeypatch):
    real = dict(paths.TEXT_SHEETS)
    monkeypatch.setattr(paths, "SAVED_ANSWERS", tmp_path)
    tiles = [("1960", s) for s in ("10", "2", "1", "2", "10")] + [
        ("1980-1990-zoning", "1980-1990-zoning-p02"), ("1980-1990-zoning", "1980-1990-zoning-p01")]
    _write_gz(tmp_path / "text-on-every-sheet.jsonl.gz", [_record(s, n, [0, 0, 768, 768], []) for s, n in tiles])
    # each map's sheets Terra read, each once, numbers in a name ordered as numbers
    want = {"1960": ("1", "2", "10"), "1980-1990-zoning": ("1980-1990-zoning-p01", "1980-1990-zoning-p02")}
    assert boxes.text_sheets() == want
    monkeypatch.setattr(paths, "TEXT_SHEETS", want)
    boxes.check_text_sheets()
    for wrong, named in (({**want, "1960": ("1", "2")}, "10"),                        # a sheet missing
                         ({**want, "1960": ("1", "10", "2")}, "1960"),                # out of order
                         ({**want, "1964": ("20180102105639444-p02",)}, "1964"),      # a map Terra never read
                         ({"1960": want["1960"]}, "1980-1990-zoning")):               # a map left out
        monkeypatch.setattr(paths, "TEXT_SHEETS", wrong)
        with pytest.raises(ValueError, match=named):
            boxes.check_text_sheets()
    # the code's own lists: every training and test map, each sheet once, in the same order
    assert list(real) == [*paths.TEXT_TRAIN_SERIES, *paths.TEXT_TEST_SERIES]
    assert not set(paths.TEXT_TRAIN_SERIES) & set(paths.TEXT_TEST_SERIES)
    assert all(list(v) == sorted(set(v), key=boxes.sheet_order) for v in real.values())
