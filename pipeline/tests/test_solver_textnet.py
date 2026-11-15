import contextlib
import gzip
import json

import cv2
import numpy as np
import pytest
import torch
import torch.nn.functional as F

from sfzmaps import config
from sfzmaps.solver import grid, paths
from sfzmaps.solver.textnet import data, evaluate, model, predict, stage, train

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
