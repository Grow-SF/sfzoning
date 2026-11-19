"""The learned reader's network and its learned streets (`sfzmaps.solver.hatchnet.model`, `.train`): the network's
shape, CPU training that gives the same bits twice, a street output that learns more than the share of street px
from a synthetic page (`synthetic_towns`), and blocks as the connected areas the street output calls "not street"."""
import numpy as np
import torch

from sfzmaps.solver.hatchnet import model, train
from synthetic_towns import town

# A small recipe for the CPU: a narrow network on small tiles, few steps (the real settings are train.Recipe()'s).
SMALL = train.Recipe(steps=5, batch=4, tile=64, width=4, val_every=50, val_tiles=4, val_cell=64)


def test_shapes():
    net = model.HatchNet(width=4)
    out = net(torch.zeros(2, 2, 256, 256))
    assert out.shape == (2, 8, 256, 256)
    assert model.OUTPUTS == 1 + len(model.NAMES) + 1 and model.NAMES == ["1R", "2R", "C", "HI", "LI", "U"]


def test_deterministic_cpu():
    sheets = [town(1), town(2, ft_per_px=2.72)]
    a = train.train_streets(sheets, seed=0, device="cpu", recipe=SMALL)
    b = train.train_streets(sheets, seed=0, device="cpu", recipe=SMALL)
    assert a.keys() == b.keys()
    for k in a:
        assert torch.equal(a[k], b[k]), k
    torch.manual_seed(0)
    init = model.HatchNet(SMALL.width).state_dict()
    assert any(not torch.equal(a[k], init[k]) for k in init if init[k].is_floating_point())     # it trained


def test_saved_model_loads(tmp_path):
    fitted = train.fit_streets([town(1), town(2)], seed=0, device="cpu", recipe=SMALL)
    f = train.save_model(tmp_path / "streets.pt", fitted, SMALL, seed=0)
    net, saved = train.load_model(f, "cpu")               # torch.load's default, weights only: plain values
    assert saved["device"] == "cpu" and saved["step"] == fitted.best_step and saved["seed"] == 0
    for k, v in net.state_dict().items():
        assert torch.equal(v, fitted.state_dict[k]), k


def test_streets_learnable():
    recipe = train.Recipe(steps=200, batch=4, tile=64, width=4, val_every=50, val_tiles=4, val_cell=64)
    state = train.train_streets([town(3), town(4)], seed=0, device="cpu", recipe=recipe)
    net = model.HatchNet(recipe.width)
    net.load_state_dict(state)
    fresh = town(5)                                     # a page it never saw
    p = np.clip(train.predict_streets(net.eval(), fresh).astype(np.float64), 1e-7, 1 - 1e-7)
    street = (fresh.block == 0).astype(np.float64)
    bce = float(-(street * np.log(p) + (1 - street) * np.log(1 - p)).mean())
    share = street.mean()
    prior = float(-(share * np.log(share) + (1 - share) * np.log(1 - share)))
    assert bce < prior, (bce, prior)


def test_blocks_are_components():
    p = np.zeros((40, 60), np.float32)
    p[:, 28:32] = 0.9                   # a street ridge between two blank areas
    p[:, 0] = 0.5                       # a probability of exactly 1/2 is not a street (the argmax keeps it a block)
    crop = np.ones(p.shape, bool)
    crop[-3:] = False                   # outside the neatline: no block
    b = train.blocks(p, crop)
    assert b.dtype == np.int32
    assert set(np.unique(b)) == {0, 1, 2}
    assert (b[:, 28:32] == 0).all() and (b[-3:] == 0).all()
    assert len(np.unique(b[:-3, :28])) == 1 and len(np.unique(b[:-3, 32:])) == 1
    assert b[0, 0] != b[0, 59]
    # a street one px wide on the diagonal still parts two blocks: px of one block share a side
    d = np.zeros((20, 20), np.float32)
    d[np.arange(20), np.arange(20)] = 0.9
    assert train.blocks(d, np.ones(d.shape, bool)).max() == 2


def test_tiled_prediction_is_one_pass():
    net = model.HatchNet(width=4).eval()
    sheet = town(6, shape=(200, 150))
    whole = train.predict_streets(net, sheet, tile=1024)
    tiled = train.predict_streets(net, sheet, tile=64)
    np.testing.assert_allclose(tiled, whole, atol=1e-5)
