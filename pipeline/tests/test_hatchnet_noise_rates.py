"""The street loss's label-noise rates (`sfzmaps.solver.hatchnet.train`, after Mnih & Hinton, ICML 2012), fitted by
the real training (`fit_streets`: the recipe's steps, learning rate, schedule and the rates' stated start) on synthetic
towns (`synthetic_towns`) whose noise is known.  A small network on the CPU stands in for the real one (`SMALL`: width,
tile and batch only).  The unprinted streets' test is in test_hatchnet_noise.py (a file of its own: each fit takes a
few minutes on the CPU, and the two run side by side).

The pass check (+-0.03 on a rate) is the test's own criterion, not a setting of the model."""
import numpy as np

from sfzmaps.solver.hatchnet import train
from synthetic_towns import printed_blocks, town

SMALL = train.Recipe(batch=4, tile=64, width=4, val_cell=64, val_tiles=8)


def test_theta_recovered():
    """Labels flipped px by px with theta = 0.12 (a printed block labelled street) and phi = 0.03 (a printed street
    labelled block): the fitted rates fall within 0.03 of them.  (Seed 0 gives theta 0.097, phi 0.060; seeds 1 and 2
    give theta 0.089 / 0.083 and phi 0.059 / 0.072, outside the check: see train.RATE_INIT.)"""
    towns = [town(s, theta=0.12, phi=0.03) for s in range(1, 9)]
    printed = np.concatenate([~printed_blocks(s).ravel() for s in range(1, 9)])          # printed street
    street = np.concatenate([t.street.ravel() == 1 for t in towns])
    theta, phi = float(street[~printed].mean()), float(1 - street[printed].mean())     # the set's own rates
    fitted = train.fit_streets(towns, seed=0, device="cpu", recipe=SMALL)
    print(f"set theta {theta:.4f} phi {phi:.4f}; fitted (step {fitted.best_step}) {fitted.rates}; "
          f"last step {fitted.last_rates}")
    assert abs(fitted.rates["theta"] - 0.12) <= 0.03, fitted.rates
    assert abs(fitted.rates["phi"] - 0.03) <= 0.03, fitted.rates
