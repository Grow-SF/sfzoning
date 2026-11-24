"""The finder's likelihoods (`sfzmaps.find.losses`): dense street-or-not px, and cut statements ("a line crosses this
cut" needs one px on, "none does" needs every px off)."""
import math

import pytest
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


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_a_crossing_cut_read_as_surely_off_still_teaches(dtype):
    """A "crosses" cut whose px the network reads as surely not a line, however surely: a finite loss and a finite,
    non-zero gradient with weight 1, and exactly zero with weight 0."""
    prior = torch.tensor([0.5, 1.0, 0.5], dtype=dtype)
    px, cut, says = torch.arange(3), torch.zeros(3, dtype=torch.long), torch.tensor([True])
    for z in (-10.0, -20.0, -40.0, -60.0, -80.0, -200.0):
        for weight in (1.0, 0.0):
            logit = torch.full((3,), z, dtype=dtype, requires_grad=True)
            loss = losses.cut_nll(logit, px, cut, prior, says, torch.tensor([weight], dtype=dtype))
            loss.backward()
            assert torch.isfinite(loss.detach()) and torch.isfinite(logit.grad).all(), (z, weight, logit.grad)
            if weight:
                assert (logit.grad != 0).all() and float(loss.detach()) > 5, (z, logit.grad)
            else:
                assert (logit.grad == 0).all() and float(loss.detach()) == 0
        chance = losses.statement_chance(torch.full((3,), z, dtype=dtype), px, cut, prior, says)
        assert torch.isfinite(chance).all() and 0 <= float(chance[0]) < 1e-3     # e^-199 is 0 in float32


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_a_none_cut_read_as_surely_on_has_a_finite_gradient(dtype):
    for z in (20.0, 60.0):
        logit = torch.full((3,), z, dtype=dtype, requires_grad=True)
        loss = losses.cut_nll(logit, torch.arange(3), torch.zeros(3, dtype=torch.long),
                              torch.tensor([0.5, 1.0, 0.5], dtype=dtype), torch.tensor([False]),
                              torch.ones(1, dtype=dtype))
        loss.backward()
        assert torch.isfinite(loss.detach()) and torch.isfinite(logit.grad).all() and (logit.grad != 0).all()


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_an_unused_crossing_cut_with_no_prior_adds_nothing(dtype):
    """A "crosses" cut with no px of prior above 0 says something impossible: unused (weight 0) it adds exactly 0 to
    the loss and to the gradient; its statement's chance is 0."""
    px, cut, prior, says = torch.arange(3), torch.zeros(3, dtype=torch.long), torch.zeros(3, dtype=dtype), \
        torch.tensor([True])
    logit = torch.full((3,), -30.0, dtype=dtype, requires_grad=True)
    loss = losses.cut_nll(logit, px, cut, prior, says, torch.zeros(1, dtype=dtype))
    loss.backward()
    assert float(loss.detach()) == 0 and (logit.grad == 0).all()
    assert float(losses.statement_chance(logit.detach(), px, cut, prior, says)[0]) == 0


def test_a_used_crossing_cut_with_no_prior_is_refused():
    with pytest.raises(ValueError, match="1 "):
        losses.cut_nll(torch.zeros(3), torch.arange(3), torch.zeros(3, dtype=torch.long), torch.zeros(3),
                       torch.tensor([True]), torch.ones(1))


def test_an_unused_impossible_cut_beside_others_changes_nothing():
    """Ordinary cuts (a "crosses" cut with one px of prior 0, which cannot carry its statement, and a "none" cut) give
    the same loss and gradients with an unused all-prior-0 "crosses" cut beside them as without it."""
    g = torch.Generator().manual_seed(0)
    base = torch.randn(9, generator=g, dtype=torch.float64) * 5
    px, cut = torch.arange(6), torch.tensor([0, 0, 0, 1, 1, 1])
    prior, says, weight = torch.tensor([0.0, 1.0, 0.5, 0.5, 1.0, 0.5], dtype=torch.float64), \
        torch.tensor([True, False]), torch.ones(2, dtype=torch.float64)
    alone = base.clone().requires_grad_(True)
    loss_alone = losses.cut_nll(alone, px, cut, prior, says, weight)
    loss_alone.backward()
    beside = base.clone().requires_grad_(True)
    loss_beside = losses.cut_nll(beside, torch.arange(9), torch.tensor([0, 0, 0, 1, 1, 1, 2, 2, 2]),
                                 torch.cat([prior, torch.zeros(3, dtype=torch.float64)]),
                                 torch.tensor([True, False, True]), torch.tensor([1.0, 1.0, 0.0], dtype=torch.float64))
    loss_beside.backward()
    assert torch.isfinite(loss_alone.detach()) and torch.isfinite(alone.grad).all()
    assert alone.grad[0] == 0 and (alone.grad[1:6] != 0).all()
    assert float(loss_beside.detach()) == float(loss_alone.detach())
    assert torch.equal(beside.grad[:6], alone.grad[:6]) and (beside.grad[6:] == 0).all()
