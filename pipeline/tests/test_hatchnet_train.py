"""The learned reader's district training (`sfzmaps.solver.hatchnet.train`): the read-noise matrix counted from the
owner's random answers, each lot's prediction pooled over its px with the learned weights, each lot counting once in
the loss, and the folds: a fold by quarter learns nothing from that quarter's reads or answers, and a reads-only run
nothing from any answer.  The folds run on synthetic towns (`district_town`): blocks of the six legend patterns
between white streets, each block two lots, each lot in a quarter by where it lies."""
import dataclasses

import numpy as np
import pandas as pd
import pytest
import torch

from sfzmaps.solver.hatchnet import data, model, train

NAMES = model.NAMES
# A small recipe for the CPU: a narrow network on small tiles, three steps, a checkpoint every step (the real
# settings are train.DistrictRecipe()'s).  32 px cells on a 128 px page hold out 3 of 16 cells.
TINY = train.DistrictRecipe(steps=3, batch=2, tile=64, width=4, val_every=1, val_tiles=2, val_cell=32)

PATTERNS = {"1R": lambda yy, xx: (yy + xx) % 6 == 0,
            "2R": lambda yy, xx: np.zeros(yy.shape, bool),
            "C": lambda yy, xx: np.ones(yy.shape, bool),
            "HI": lambda yy, xx: (yy % 6 == 0) | (xx % 6 == 0),
            "LI": lambda yy, xx: ((yy + xx) % 6 == 0) | ((yy - xx) % 6 == 0),
            "U": lambda yy, xx: ((yy // 4) + (xx // 4)) % 2 == 0}


def district_town(seed: int, first: int, shape=(128, 128), block: int = 40, street: int = 8):
    """A page of blocks, each drawn in one of the six patterns with a black outline, between white streets; each
    block's left and right halves are two lots (ids from `first`).  Returns the rasters and one row per lot (lot,
    quarter by the lot's middle on the page, its true district)."""
    rng = np.random.default_rng(seed)
    h, w = shape
    step = block + street
    oy, ox = (int(v) for v in rng.integers(0, step, 2))
    yy, xx = np.mgrid[0:h, 0:w]
    ry, rx = (yy - oy) % step, (xx - ox) % step
    by, bx = (yy - oy) // step + 1, (xx - ox) // step + 1
    inside = (ry < block) & (rx < block)
    code = rng.integers(0, len(NAMES), (int(by.max()) + 1, int(bx.max()) + 1))
    ink = np.ones(shape, np.float32)
    for k, name in enumerate(NAMES):
        ink[inside & (code[by, bx] == k) & PATTERNS[name](yy, xx)] = 0.0
    ink[inside & ((ry == 0) | (ry == block - 1) | (rx == 0) | (rx == block - 1))] = 0.0
    raw = np.where(inside, (by * code.shape[1] + bx) * 2 + (rx >= block // 2), -1)
    ids = np.unique(raw[raw >= 0])
    lot = np.full(shape, -1, np.int32)
    lot[raw >= 0] = first + np.searchsorted(ids, raw[raw >= 0])
    rows = []
    for i, r in enumerate(ids):
        m = raw == r
        cy, cx = yy[m].mean(), xx[m].mean()
        q = ("n" if cy < h / 2 else "s") + ("w" if cx < w / 2 else "e")
        rows.append({"lot": first + i, "quarter": q, "truth": NAMES[int(code[by[m][0], bx[m][0]])]})
    block_t = np.where(inside, 1, 0).astype(np.uint8)
    r = data.SheetRasters(f"town{seed}", ink, block_t, lot, np.ones(shape, bool), 5.0)
    return r, rows


def legend_sheet(shape=(128, 128)):
    """A page with no lots and the six swatches side by side, each its pattern in a box; and the boxes (sheet, code,
    (x0, y0, x1, y1))."""
    yy, xx = np.mgrid[0:shape[0], 0:shape[1]]
    ink = np.ones(shape, np.float32)
    boxes = []
    for k, name in enumerate(NAMES):
        x0, y0 = 4 + 20 * k, 50
        box = (x0, y0, x0 + 16, y0 + 12)
        m = (xx >= box[0]) & (xx < box[2]) & (yy >= box[1]) & (yy < box[3])
        ink[m & PATTERNS[name](yy, xx)] = 0.0
        boxes.append(("legend", name, box))
    r = data.SheetRasters("legend", ink, np.full(shape, data.UNKNOWN, np.uint8), np.full(shape, -1, np.int32),
                          np.zeros(shape, bool), 5.0)
    return r, boxes


def towns(seed: int = 0):
    """Two towns and the legend: the rasters, the lot table (reads right 80% of the time, every third lot answered
    on the random page, every seventh on another) and the swatches."""
    rng = np.random.default_rng(seed + 100)
    a, ra = district_town(1, 0)
    b, rb = district_town(2, len(ra))
    leg, boxes = legend_sheet()
    t = pd.DataFrame(ra + rb)
    n = len(t)
    wrong = rng.random(n) < 0.2
    t["read_code"] = np.where(wrong, rng.choice(NAMES, n), t["truth"])
    t["read_conf"] = rng.random(n)
    t["answer"] = ""
    t["stratum"] = ""
    third = t.index % 3 == 0
    t.loc[third, "answer"] = t.loc[third, "truth"]
    t.loc[third, "stratum"] = "random"
    seventh = (t.index % 7 == 0) & ~third
    t.loc[seventh, "answer"] = t.loc[seventh, "truth"]
    t.loc[seventh, "stratum"] = "likely"
    return [a, b, leg], t.drop(columns="truth"), boxes


def perturb(lots: pd.DataFrame, which: np.ndarray, seed: int, reads: bool = True, answers: bool = True):
    """The lot table with the reads and/or answers (code, page stratum) of the lots `which` drawn at random."""
    rng = np.random.default_rng(seed)
    t = lots.copy()
    n = int(which.sum())
    if reads:
        t.loc[which, "read_code"] = rng.choice(NAMES + [""], n)
        t.loc[which, "read_conf"] = rng.random(n)
    if answers:
        t.loc[which, "answer"] = rng.choice(NAMES + [""], n)
        t.loc[which, "stratum"] = rng.choice(["random", "likely", "lowconf", ""], n)
    return t


def fit(lots, q, with_answers, rasters, boxes):
    d = train.district_data(rasters, lots, boxes, TINY.val_cell)
    return train.train_fold(q, with_answers, seed=0, data=d, device="cpu", recipe=TINY)


def same(a: dict, b: dict) -> bool:
    return a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)


# --- the read-noise matrix ------------------------------------------------------------------------------------------

def test_read_noise_counts():
    """Random-page answers 1R read as 1R three times and as 2R once: T's 1R row is those counts plus one per cell.
    Answers on other pages, of other quarters, or of lots with no read don't count; the other rows, with no answers,
    are the uniform prior."""
    rows = [("nw", "random", "1R", "1R")] * 3 + [("ne", "random", "1R", "2R")]
    rows += [("nw", "likely", "1R", "C"), ("nw", "lowconf", "C", "2R"), ("se", "random", "1R", "C"),
             ("nw", "random", "1R", ""), ("nw", "", "", "2R")]
    lots = pd.DataFrame(rows, columns=["quarter", "stratum", "answer", "read_code"])
    t = train.read_noise(lots, ["nw", "ne", "sw"])
    assert t.shape == (6, 6)
    one = NAMES.index("1R")
    want = np.ones(6)
    want[NAMES.index("1R")] += 3
    want[NAMES.index("2R")] += 1
    np.testing.assert_allclose(t[one], want / want.sum())
    for j in range(6):
        if j != one:
            np.testing.assert_allclose(t[j], np.full(6, 1 / 6))
    np.testing.assert_allclose(t.sum(1), 1.0)
    # no answers to count: the identity (the reads taken as they are)
    np.testing.assert_array_equal(train.read_noise(lots, ["sw"]), np.eye(6))


# --- pooling and the loss -------------------------------------------------------------------------------------------

def test_pool_weights():
    """A lot whose left half says 1R and right half 2R takes the district of the half its weights favour; with equal
    weights its probability is the mean of its px's.  A px of no lot (-1) is not pooled."""
    d = torch.full((9, 6), -4.0)
    d[:4, NAMES.index("1R")] = 4.0           # lot 0's left half
    d[4:8, NAMES.index("2R")] = 4.0          # lot 0's right half
    d[8, NAMES.index("C")] = 9.0             # no lot
    lot = torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, -1])
    left = torch.tensor([3.0] * 4 + [-3.0] * 4 + [0.0])
    lp = train.lot_logprob(d, left, lot)
    assert lp.shape == (1, 6) and int(lp[0].argmax()) == NAMES.index("1R")
    assert abs(float(lp[0].exp().sum()) - 1) < 1e-5
    assert int(train.lot_logprob(d, -left, lot)[0].argmax()) == NAMES.index("2R")
    even = train.lot_logprob(d, torch.zeros(9), lot)
    np.testing.assert_allclose(even[0].exp().numpy(), torch.softmax(d[:8], 1).mean(0).numpy(), rtol=1e-5)
    # two lots apart: each pools its own px only
    two = train.lot_logprob(d, torch.zeros(9), torch.tensor([0, 0, 0, 0, 1, 1, 1, 1, -1]))
    assert int(two[0].argmax()) == NAMES.index("1R") and int(two[1].argmax()) == NAMES.index("2R")


def test_lot_loss():
    """A read k costs -log sum_j p(j) T[j, k]; an answer j costs -log p(j), whatever T."""
    p = torch.tensor([[0.7, 0.1, 0.1, 0.05, 0.03, 0.02]])
    lp = p.log()
    t = np.full((6, 6), 0.02)
    np.fill_diagonal(t, 0.9)
    log_t = torch.from_numpy(np.log(t)).float()
    k = NAMES.index("2R")
    read = train.lot_loss(lp, torch.tensor([train.READ]), torch.tensor([k]), log_t)
    assert abs(float(read[0]) + np.log(float((p[0].double() * torch.from_numpy(t[:, k])).sum()))) < 1e-5
    ans = train.lot_loss(lp, torch.tensor([train.ANSWER]), torch.tensor([k]), log_t)
    assert abs(float(ans[0]) + np.log(0.1)) < 1e-5


def test_each_lot_counts_once():
    """Two lots wholly in the batch count once each, whatever their px; a lot half in it counts a half."""
    labels = train.Labels(kind=np.array([train.READ] * 3, np.int8), code=np.array([0, 1, 2]),
                          px=np.array([100, 10, 20]), quarter=np.array(["nw"] * 3, object), q=None,
                          t=np.eye(6))
    tile = np.full((1, 16, 16), -1, np.int32)
    tile[0].reshape(-1)[:100] = 0
    tile[0].reshape(-1)[100:110] = 1
    tile[0].reshape(-1)[110:120] = 2          # 10 of lot 2's 20 px
    units = train.assemble(tile, labels)
    torch.manual_seed(0)
    out = torch.randn(1, model.OUTPUTS, 16, 16)
    total, weight, _ = train.batch_loss(out, units, labels)
    dl = out[0, model.DISTRICTS].permute(1, 2, 0).reshape(-1, 6)
    wl = out[0, model.WEIGHT].reshape(-1)
    flat = torch.from_numpy(tile[0].reshape(-1).astype(np.int64))
    lp = train.lot_logprob(dl, wl, flat, 3)
    each = -lp[torch.arange(3), torch.tensor([0, 1, 2])]
    assert abs(float(weight) - 2.5) < 1e-6
    assert abs(float(total) - float(each[0] + each[1] + 0.5 * each[2])) < 1e-4


def test_leak_check_raises():
    """A label of the withheld quarter reaching a batch stops the run."""
    labels = train.Labels(kind=np.array([train.READ, train.READ], np.int8), code=np.array([0, 1]),
                          px=np.array([4, 4]), quarter=np.array(["ne", "nw"], object), q="nw", t=np.eye(6))
    tile = np.full((1, 16, 16), -1, np.int32)
    tile[0, 0, :4] = 0
    train.assemble(tile, labels)                              # quarter ne: fine
    tile[0, 1, :4] = 1
    with pytest.raises(RuntimeError, match="nw"):
        train.assemble(tile, labels)


# --- the folds ------------------------------------------------------------------------------------------------------

def test_fold_withholds_quarter():
    """Quarter q's reads and answers drawn at random: train_fold(q) gives the same bits.  The same change in a
    training quarter changes the weights (the test can see a change)."""
    rasters, lots, boxes = towns()
    nw = (lots["quarter"] == "nw").to_numpy()
    assert nw.sum() > 4 and (~nw).sum() > 4
    base = fit(lots, "nw", True, rasters, boxes)
    assert same(base, fit(perturb(lots, nw, 1), "nw", True, rasters, boxes))
    assert same(base, fit(perturb(lots, nw, 2), "nw", True, rasters, boxes))
    se = (lots["quarter"] == "se").to_numpy()
    assert not same(base, fit(perturb(lots, se, 1), "nw", True, rasters, boxes))
    # the all-quarters model learns from every quarter
    assert not same(fit(lots, None, True, rasters, boxes), fit(perturb(lots, nw, 1), None, True, rasters, boxes))


def test_reads_only_ignores_answers():
    """With with_answers False, every answer drawn at random changes nothing; with answers it does."""
    rasters, lots, boxes = towns()
    every = np.ones(len(lots), bool)
    changed = perturb(lots, every, 3, reads=False)
    assert not lots["answer"].equals(changed["answer"])
    assert same(fit(lots, "ne", False, rasters, boxes), fit(changed, "ne", False, rasters, boxes))
    assert not same(fit(lots, "ne", True, rasters, boxes), fit(changed, "ne", True, rasters, boxes))


def test_fold_trains_and_repeats():
    """The fold trains (its weights move from the start) and gives the same bits twice; the read-noise matrix it used
    is counted from its training quarters only; the lots used per quarter leave the withheld one out."""
    rasters, lots, boxes = towns()
    d = train.district_data(rasters, lots, boxes, TINY.val_cell)
    a = train.fit_fold(d, "sw", True, seed=0, device="cpu", recipe=TINY)
    b = train.fit_fold(d, "sw", True, seed=0, device="cpu", recipe=TINY)
    assert same(a.state_dict, b.state_dict)
    torch.manual_seed(0)
    init = model.HatchNet(TINY.width, TINY.inputs).state_dict()
    assert any(not torch.equal(a.state_dict[k], init[k]) for k in init if init[k].is_floating_point())
    np.testing.assert_array_equal(a.t, train.read_noise(lots, ["nw", "ne", "se"]))
    assert a.units_per_quarter.get("sw", 0) == 0 and sum(a.units_per_quarter.values()) > 0
    assert 1 <= a.best_step <= TINY.steps and len(a.curve) == TINY.steps


def test_swatches_train():
    """The legend's swatch boxes are labels of their own: with no lot read or answered, every tile centres on one."""
    rasters, lots, boxes = towns()
    bare = lots.assign(read_code="", answer="", stratum="")
    d = train.district_data(rasters, bare, boxes, TINY.val_cell)
    f = train.fit_fold(d, None, True, seed=0, device="cpu", recipe=TINY)
    assert f.units_per_quarter == {}
    assert f.swatch_units >= TINY.steps * TINY.batch
    np.testing.assert_array_equal(f.t, np.eye(6))


def test_held_out_lots_by_cell():
    """A lot is held out (checkpoint selection only) when most of its px lie in held-out cells; each lot's px are
    counted over every sheet."""
    r = data.SheetRasters("s", np.ones((64, 64), np.float32), np.ones((64, 64), np.uint8),
                          np.full((64, 64), -1, np.int32), np.ones((64, 64), bool), 5.0)
    held = train.held_out(np.arange(64), np.arange(64), 16)
    r.lot[held] = 0                                             # lot 0: all held out
    r.lot[~held] = 1                                            # lot 1: none
    r2 = dataclasses.replace(r, sheet="t", lot=np.where(held, 1, -1).astype(np.int32))  # lot 1's held px elsewhere
    lots = pd.DataFrame({"lot": [0, 1], "quarter": ["nw", "nw"], "read_code": ["1R", "2R"], "answer": ["", ""],
                         "stratum": ["", ""]})
    d = train.district_data([r, r2], lots, [], 16)
    assert d.px.tolist() == [int(held.sum()), int((~held).sum() + held.sum())]
    assert d.held.tolist() == [True, False]
