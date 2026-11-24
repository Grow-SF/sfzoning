"""The finder's likelihoods (`sfzmaps.find.losses`): dense street-or-not px, and cut statements ("a line crosses this
cut" needs one px on, "none does" needs every px off)."""
import math

import numpy as np
import torch

from sfzmaps.find import losses

ON, OFF = 30.0, -30.0                   # logits of a px surely on and surely off


def nll(p, prior, says, weight=1.0):
    """One cut of len(p) px."""
    n = len(p)
    return float(losses.cut_nll(torch.tensor(p, dtype=torch.float32), torch.arange(n), torch.zeros(n, dtype=torch.long),
                                torch.tensor(prior, dtype=torch.float32), torch.tensor([says]),
                                torch.tensor([weight], dtype=torch.float32)))


def test_dense_nll_ignores_unlabelled_px():
    g = torch.Generator().manual_seed(0)
    logit = torch.randn(1, 8, 8, generator=g)
    target = (torch.rand(1, 8, 8, generator=g) > 0.5).float()
    weight = (torch.rand(1, 8, 8, generator=g) > 0.5).float()
    other = torch.where(weight == 0, torch.randn(1, 8, 8, generator=g) * 50, logit)
    assert float(losses.dense_nll(logit, target, weight)) > 0
    assert float(losses.dense_nll(other, target, weight)) == float(losses.dense_nll(logit, target, weight))


def test_a_crossing_statement_needs_one_px():
    prior = [0.5, 1.0, 0.5]
    assert abs(nll([OFF, ON, OFF], prior, True)) < 1e-6
    assert nll([OFF, OFF, OFF], prior, True) > 10
    assert abs(nll([ON, OFF, OFF], prior, True) + math.log(0.5)) < 1e-6


def test_a_none_statement_needs_every_px_off():
    prior = [0.5, 1.0, 0.5]
    assert abs(nll([OFF, OFF, OFF], prior, False)) < 1e-6
    assert abs(nll([OFF, 0.0, OFF], prior, False) + math.log(0.5)) < 1e-6


def test_statement_chance_is_the_chance_of_the_cuts_own_statement():
    """The same three-px cut: one px surely on where the prior is one half gives "crosses" an even chance; one px at
    an even chance gives "none" an even chance."""
    px, cut, prior = torch.arange(3), torch.zeros(3, dtype=torch.long), torch.tensor([0.5, 1.0, 0.5])
    crosses = losses.statement_chance(torch.tensor([ON, OFF, OFF]), px, cut, prior, torch.tensor([True]))
    none = losses.statement_chance(torch.tensor([OFF, 0.0, OFF]), px, cut, prior, torch.tensor([False]))
    assert crosses.shape == (1,) and abs(float(crosses[0]) - 0.5) < 1e-6
    assert abs(float(none[0]) - 0.5) < 1e-6


def test_a_cut_of_weight_zero_adds_nothing():
    """Two cuts, the second of weight 0: whatever its px say, the loss is the first cut's alone, and it counts as no
    statement."""
    px, cut = torch.arange(6), torch.tensor([0, 0, 0, 1, 1, 1])
    prior = torch.tensor([0.5, 1, 0.5, 0.5, 1, 0.5])
    says, weight = torch.tensor([True, False]), torch.tensor([1.0, 0.0])
    first = nll([OFF, ON, OFF], [0.5, 1, 0.5], True)
    for second in ([OFF, OFF, OFF], [ON, ON, ON], [0.0, 3.0, -2.0]):
        logit = torch.tensor([OFF, ON, OFF] + second)
        for s in (says, torch.tensor([True, True])):
            assert float(losses.cut_nll(logit, px, cut, prior, s, weight)) == first
    assert losses.statements(torch.ones(4, 4), weight) == 17.0


def test_gradients_are_finite_when_every_px_is_surely_off():
    """A "crosses" cut whose px are all far below zero still gives a finite loss and gradient (computed in logs)."""
    logit = torch.full((3,), -200.0, requires_grad=True)
    loss = losses.cut_nll(logit, torch.arange(3), torch.zeros(3, dtype=torch.long), torch.ones(3),
                          torch.tensor([True]), torch.ones(1))
    loss.backward()
    assert np.isfinite(float(loss.detach())) and torch.isfinite(logit.grad).all() and float(loss.detach()) > 10
