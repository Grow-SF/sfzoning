"""The learned streets' second loss (`sfzmaps.solver.hatchnet.topo`, `.train`): soft skeletons, the clDice recall and
precision, the noise model's posterior weighting of the recall, and the checkpoint kept by the summed held-out loss."""
import numpy as np
import torch

from sfzmaps.solver.hatchnet import topo, train
from synthetic_towns import town

ITERS = 32          # the skeleton depth for a 64 px tile (TILE / 2: train.Recipe.iters); every test shape is wider


def _street(shape, rows, cols, inside=0.95, outside=0.01):
    p = torch.full(shape, outside)
    p[..., rows, cols] = inside
    return p


def test_skeleton_of_thin_line():
    p = torch.zeros(1, 64, 160)
    p[0, 30:35, 20:140] = 1.0                       # a bar 5 px wide and 120 px long
    s = topo.soft_skeleton(p, ITERS)[0]
    assert s.shape == p.shape[1:]
    assert float(s[:31].abs().max()) == 0.0 and float(s[34:].abs().max()) == 0.0     # only its middle 3 px
    kept = (s[31:34] > 0.5).any(0)
    assert int(kept.sum()) >= 0.95 * 120, int(kept.sum())


def test_skeleton_of_soft_line_stays_inside():
    p = torch.zeros(1, 64, 160)
    p[0, 30:35, 20:140] = torch.tensor([0.3, 0.8, 0.95, 0.8, 0.3])[:, None]     # a soft ridge across the bar
    s = topo.soft_skeleton(p, ITERS)[0]
    assert float(s[:30].abs().max()) == 0.0 and float(s[35:].abs().max()) == 0.0
    assert float(s[32, 25:135].min()) > 0.5                                       # its crest is the skeleton


def test_recall_penalises_gap():
    shape = (1, 32, 100)
    centre = torch.zeros(shape)
    centre[0, 16, 10:90] = 1.0
    corridor = torch.zeros(shape)
    corridor[0, 14:19, 10:90] = 1.0
    whole = _street(shape, slice(14, 19), slice(10, 90))
    gap = whole.clone()
    gap[0, 14:19, 50] = 0.01                        # one px column missing across the street
    r_whole, _ = topo.cldice_terms(whole, centre, corridor, ITERS)
    r_gap, _ = topo.cldice_terms(gap, centre, corridor, ITERS)
    assert float(r_gap) < float(r_whole)


def test_precision_penalises_blob_off_centre():
    shape = (1, 128, 200)
    centre = torch.zeros(shape)
    centre[0, 20, 10:190] = 1.0
    corridor = torch.zeros(shape)                   # 0 off the corridor: today's blocks or ignored px alike
    corridor[0, 18:23, 10:190] = 1.0
    street = _street(shape, slice(18, 23), slice(10, 190))
    blob = street.clone()
    blob[0, 70:110, 120:160] = 0.95                 # a 40 x 40 predicted street far from every centre line
    _, before = topo.cldice_terms(street, centre, corridor, ITERS)
    _, after = topo.cldice_terms(blob, centre, corridor, ITERS)
    assert float(after) < float(before), (float(before), float(after))
    # precision is over every px (the region), labelled or not: the blob's px are all ignored here
    region = torch.ones(shape)
    _, after_region = topo.cldice_terms(blob, centre, corridor, ITERS, region=region)
    assert float(after_region) == float(after)


def test_empty_terms_are_one():
    z = torch.zeros(1, 16, 16)
    r, p = topo.cldice_terms(z, z, z, 4)
    assert float(r) == 1.0 and float(p) == 1.0      # no centre px: nothing to recall; no predicted skeleton


def test_printed_narrow_street_still_enforced():
    """A 5 px printed corridor under a centre line, the street predicted along it but for a 1 px gap the network is
    unsure of.  With the noise model's rates the recall still rises when the gap closes, and the summed street loss
    falls: the posterior weighting keeps a printed street the network reads as street."""
    shape = (1, 32, 120)
    centre = torch.zeros(shape)
    centre[0, 16, 10:110] = 1.0
    street = torch.zeros(shape)                      # labels: the corridor street, everything else block
    street[0, 14:19, 10:110] = 1.0
    w, region = torch.ones(shape), torch.ones(shape)
    rates = torch.tensor([0.12, 0.03])
    for p_gap in (0.4, 0.2, 0.05):
        z_whole = torch.logit(_street(shape, slice(14, 19), slice(10, 110), 0.9, 0.02))
        z_gap = z_whole.clone()
        z_gap[0, 14:19, 60] = float(torch.logit(torch.tensor(p_gap)))
        whole = train.street_terms(z_whole, street, w, centre, street, region, *rates, ITERS)
        gap = train.street_terms(z_gap, street, w, centre, street, region, *rates, ITERS)
        assert float(train.recall_of(whole)) > float(train.recall_of(gap)), p_gap
        assert float(train.street_total(whole)) < float(train.street_total(gap)), p_gap
        # and the loss pushes the gap's px towards street
        z = z_gap.clone().requires_grad_(True)
        train.street_total(train.street_terms(z, street, w, centre, street, region, *rates, ITERS)).backward()
        assert float(z.grad[0, 14:19, 60].max()) < 0.0, p_gap


def test_noise_free_rates_are_plain_bce():
    """With theta = phi = 0 the noisy-label likelihood is the plain binary cross-entropy and the posterior is 1."""
    g = torch.Generator().manual_seed(0)
    z = torch.randn(2, 8, 8, generator=g)
    m = (torch.rand(2, 8, 8, generator=g) < 0.5).float()
    nll = train.noisy_nll(z, m, torch.tensor(0.0), torch.tensor(0.0))
    torch.testing.assert_close(nll, torch.nn.functional.binary_cross_entropy_with_logits(z, m, reduction="none"))
    q = train.street_posterior(z, torch.tensor(0.0), torch.tensor(0.0))
    torch.testing.assert_close(q, torch.ones_like(q))


def test_noisy_likelihood_is_the_model():
    """P(m = 1) = p (1 - phi) + (1 - p) theta and P(m = 0) = p phi + (1 - p)(1 - theta); posterior p(1 - phi) / P(1)."""
    p = torch.tensor([0.001, 0.2, 0.5, 0.9, 0.999])
    th, ph = torch.tensor(0.12), torch.tensor(0.03)
    z = torch.logit(p.double()).float()
    one = -torch.log(p * (1 - ph) + (1 - p) * th)
    zero = -torch.log(p * ph + (1 - p) * (1 - th))
    torch.testing.assert_close(train.noisy_nll(z, torch.ones(5), th, ph), one, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(train.noisy_nll(z, torch.zeros(5), th, ph), zero, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(train.street_posterior(z, th, ph), p * (1 - ph) / (p * (1 - ph) + (1 - p) * th),
                               rtol=1e-5, atol=1e-6)


def test_rates_start_where_stated():
    r = train.NoiseRates(0.25)
    v = r.values()
    assert abs(v["theta"] - 0.25) < 1e-6 and abs(v["phi"] - 0.25) < 1e-6
    assert len(list(r.parameters())) == 1 and next(r.parameters()).numel() == 2


def test_selection_uses_same_sum():
    sums = [0.91, 0.42, 0.40, 0.55, float("nan"), 0.40]
    assert train.select_checkpoint(sums) == 2          # the least; the first of a tie; NaN never kept
    assert train.select_checkpoint([float("nan")] * 3) == 2                        # none finite: the last
    # the fit keeps the checkpoint with the least held-out sum, and that sum is the training loss's own terms
    recipe = train.Recipe(steps=8, batch=2, tile=64, width=4, val_every=2, val_tiles=4, val_cell=64)
    fitted = train.fit_streets([town(1), town(2)], seed=0, device="cpu", recipe=recipe)
    vals = [row["val_loss"] for row in fitted.curve]
    assert [row["step"] for row in fitted.curve] == [2, 4, 6, 8]
    assert [c["step"] for c in fitted.checkpoints] == [2, 4, 6, 8]
    assert fitted.best_step == fitted.curve[train.select_checkpoint(vals)]["step"]
    for row in fitted.curve:
        total = row["val_bce"] + (1 - row["val_recall"]) + (1 - row["val_precision"])
        assert abs(row["val_loss"] - total) < 1e-9
    kept = fitted.checkpoints[train.select_checkpoint(vals)]
    for k, v in kept["state_dict"].items():
        assert torch.equal(v, fitted.state_dict[k]), k
    assert fitted.rates == kept["rates"] and set(fitted.rates) == {"theta", "phi"}


def test_v1_still_trains():
    """The first street loss (BCE on today's blocks, early stopping on the held-out BCE) is still reachable."""
    recipe = train.Recipe(steps=4, batch=2, tile=64, width=4, val_every=2, val_tiles=4, val_cell=64,
                          street_loss="v1")
    fitted = train.fit_streets([town(1), town(2)], seed=0, device="cpu", recipe=recipe)
    assert fitted.rates is None and fitted.checkpoints == []
    assert {"val_loss", "val_agree"} <= set(fitted.curve[0])


def test_v2_needs_street_labels():
    t = town(1)
    bare = train.data.SheetRasters(t.sheet, t.ink, t.block, t.lot, t.crop, t.ft_per_px)
    try:
        train.fit_streets([bare], seed=0, device="cpu", recipe=train.Recipe(steps=2, batch=2, tile=64, width=4))
    except ValueError as e:
        assert "street labels" in str(e)
    else:
        raise AssertionError("v2 trained without street labels")


def test_v2_deterministic_cpu():
    recipe = train.Recipe(steps=4, batch=2, tile=64, width=4, val_every=2, val_tiles=4, val_cell=64)
    a = train.fit_streets([town(1), town(2, ft_per_px=2.72)], seed=0, device="cpu", recipe=recipe)
    b = train.fit_streets([town(1), town(2, ft_per_px=2.72)], seed=0, device="cpu", recipe=recipe)
    for k in a.state_dict:
        assert torch.equal(a.state_dict[k], b.state_dict[k]), k
    assert a.rates == b.rates and np.isfinite([r["val_loss"] for r in a.curve]).all()
