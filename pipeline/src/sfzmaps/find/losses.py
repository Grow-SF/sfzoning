"""The finder's likelihoods: minus the log chance of each label under the network's answers, summed.

- A dense label (street or not at a px, or "no block edge" / "no zoning border" at a px): Bernoulli, `dense_nll`.
- A cut statement (`find.cuts`): "a line crosses this cut" holds with chance 1 - prod_i (1 - prior_i p_i), the line
  being at px i with the network's chance p_i where the placement allows it (prior_i); "none does" with chance
  prod_i (1 - p_i).  `cut_nll`, computed in logs.

Every term is multiplied by its label's weight, and `statements` counts the labels (their weights summed); the
trainer divides the summed loss by it.  The three answers add as one likelihood, with no weights between them.
Reads and writes nothing."""
from __future__ import annotations

import torch
import torch.nn.functional as F

TINY = 1e-30        # numerics only: the chance "a line crosses" is never taken below TINY (a loss of 69)


def dense_nll(logit: torch.Tensor, target: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """The sum over px of weight * binary cross entropy; target 0/1, weight 0 where there is no label."""
    bce = F.binary_cross_entropy_with_logits(logit, target.to(logit.dtype), reduction="none")
    return (torch.where(weight > 0, weight.to(logit.dtype) * bce, torch.zeros_like(bce))).sum()


def _log1mexp(s: torch.Tensor) -> torch.Tensor:
    """log(1 - exp(s)) for s < 0, accurate at both ends."""
    return torch.where(s < -0.6931471805599453, torch.log1p(-torch.exp(s)), torch.log(-torch.expm1(s)))


def _log_statement_chance(logit_flat: torch.Tensor, cuts_px: torch.Tensor, cuts_cut: torch.Tensor,
                          prior: torch.Tensor, says: torch.Tensor) -> torch.Tensor:
    """The log of each cut's statement chance (see `statement_chance`), computed in logs."""
    z = logit_flat[cuts_px]
    prior = prior.to(z.dtype)
    n = len(says)
    # log(1 - prior p) with p = sigmoid(z): softplus(z + log(1 - prior)) - softplus(z); log(1 - prior) is -inf at 1
    miss = torch.logaddexp(torch.zeros_like(z), z + torch.log1p(-prior.clamp(max=1))) - F.softplus(z)
    log_none = torch.zeros(n, dtype=z.dtype, device=z.device).index_add(0, cuts_cut, miss)       # log prod(1 - prior p)
    log_off = torch.zeros(n, dtype=z.dtype, device=z.device).index_add(0, cuts_cut, -F.softplus(z))  # log prod(1 - p)
    crosses = _log1mexp(log_none.clamp(max=-TINY))                           # log(1 - TINY) = -TINY
    return torch.where(says.to(torch.bool), crosses, log_off)


def statement_chance(logit_flat: torch.Tensor, cuts_px: torch.Tensor, cuts_cut: torch.Tensor, prior: torch.Tensor,
                     says: torch.Tensor) -> torch.Tensor:
    """(n,): the chance the network gives each cut's own statement: 1 - prod_i (1 - prior_i p_i) for "a line
    crosses", prod_i (1 - p_i) for "none does".  The network agrees with a cut when this is over one half (the more
    likely answer).  Arguments as `cut_nll`."""
    return torch.exp(_log_statement_chance(logit_flat, cuts_px, cuts_cut, prior, says))


def cut_nll(logit_flat: torch.Tensor, cuts_px: torch.Tensor, cuts_cut: torch.Tensor, prior: torch.Tensor,
            says: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """The sum over cuts of weight * minus the log of the cut's statement chance (`statement_chance`).  `logit_flat`:
    the answer's logits, flat; `cuts_px` (m,): the index into it of each cut px; `cuts_cut` (m,): the cut each px is
    in; `prior` (m,); `says` (n,) bool, True "a line crosses"; `weight` (n,)."""
    term = -_log_statement_chance(logit_flat, cuts_px, cuts_cut, prior, says)
    weight = weight.to(term.dtype)
    return torch.where(weight > 0, weight * term, torch.zeros_like(term)).sum()


def statements(*weights) -> float:
    """How many statements the labels hold: the sum of every weight given (each px's dense weight, each cut's weight);
    a label of weight 1 is one statement."""
    return float(sum(float(w.sum()) for w in weights))
