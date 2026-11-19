"""The street loss's label-noise model (`sfzmaps.solver.hatchnet.train`, after Mnih & Hinton, ICML 2012) on today's
streets the print doesn't show, fitted by the real training (`fit_streets`: the recipe's steps, learning rate, schedule
and the rates' stated start) on synthetic towns (`synthetic_towns`) whose noise is known.  A small network on the CPU
stands in for the real one (`SMALL`: width, tile and batch only).  The rates' own test is test_hatchnet_noise_rates.py.

The pass checks (+-0.03 on the rate, a ratio under 1/4) are the test's own criteria, not settings of the model."""
import numpy as np
import torch

from sfzmaps.solver.hatchnet import model, train
from synthetic_towns import printed_blocks, town

# Width 8 (the real network's is 16): with the scan as its only input (model.INPUTS), a width-4 network stays
# unsure on hatched blocks (p about 0.1) and takes the noise in, which tests the stand-in, not the noise model.
SMALL = train.Recipe(batch=4, tile=64, width=8, val_cell=64, val_tiles=8)


def test_unprinted_street_not_forced():
    """Today's streets the print doesn't show: 12% of the labelled street px are strips across solid hatched blocks,
    each at a random place in its block (nothing in the ink tells where).  Once the rates are fitted, the loss's push
    towards street on such a strip's centre line, as a share of the push the same loss gives there without the noise
    model (theta = phi = 0: plain cross-entropy, unweighted recall), is under a quarter of that share on a printed
    corridor's centre line.  (The share, not the push itself: on a printed corridor the network already says street
    with p near 1, so the push left there is small whatever the loss; the share is how much of its label's pull the
    noise model keeps.)"""
    towns = [town(s, unprinted=0.3) for s in range(1, 9)]
    strip = np.concatenate([(printed_blocks(s) & (t.street == 1)).ravel() for s, t in zip(range(1, 9), towns)])
    street = np.concatenate([(t.street == 1).ravel() for t in towns])
    share = float(strip.sum() / street.sum())
    assert 0.09 < share < 0.15, share                                     # the set is the one described
    theta = float(strip.sum() / np.concatenate([printed_blocks(s).ravel() for s in range(1, 9)]).sum())
    fitted = train.fit_streets(towns, seed=0, device="cpu", recipe=SMALL)
    assert abs(fitted.rates["theta"] - theta) <= 0.03, (theta, fitted.rates)          # the set's own theta

    fresh = town(99, unprinted=0.3)
    unprinted_line = (fresh.centre == 1) & printed_blocks(99)
    printed_line = (fresh.centre == 1) & ~printed_blocks(99)
    assert unprinted_line.sum() > 50 and printed_line.sum() > 50
    net = model.HatchNet(SMALL.width, SMALL.inputs)
    net.load_state_dict(fitted.state_dict)
    with torch.no_grad():
        z0 = net.eval()(model.inputs(torch.from_numpy(fresh.ink)[None], torch.tensor([fresh.ft_per_px]),
                                      net.inputs))[:, 0]
    labels = [torch.from_numpy(a)[None].float() for a in
              (fresh.street == 1, fresh.street_w.astype(np.float32), fresh.centre == 1, fresh.street == 1,
               np.ones(fresh.ink.shape, bool))]

    def push(theta_, phi_):
        z = z0.clone().requires_grad_(True)
        train.street_total(train.street_terms(z, *labels, torch.tensor(theta_), torch.tensor(phi_),
                                              SMALL.iters)).backward()
        return -z.grad[0].numpy().astype(np.float64)

    noisy, plain = push(fitted.rates["theta"], fitted.rates["phi"]), push(0.0, 0.0)
    kept_u = noisy[unprinted_line].sum() / plain[unprinted_line].sum()
    kept_p = noisy[printed_line].sum() / plain[printed_line].sum()
    p = torch.sigmoid(z0[0]).numpy()
    print(f"set theta {theta:.4f} (unprinted share of street {share:.3f}); fitted (step {fitted.best_step}) "
          f"{fitted.rates}; p on the unprinted line median {np.median(p[unprinted_line]):.4f}, on the printed "
          f"{np.median(p[printed_line]):.4f}; push kept: unprinted {kept_u:.4f}, printed {kept_p:.4f}, "
          f"ratio {kept_u / kept_p:.4f}")
    assert plain[unprinted_line].sum() > 0 and plain[printed_line].sum() > 0
    assert kept_u / kept_p < 0.25, (kept_u, kept_p)

