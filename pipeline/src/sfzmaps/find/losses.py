"""The finder's likelihoods: minus the log chance of each label under the network's answers, summed.

- A dense label (street or not at a px, or "no block edge" at a px): Bernoulli, `dense_nll`.
- A cut statement (`find.cuts`): "a line crosses this cut" holds with chance 1 - prod_i (1 - prior_i p_i), the line
  being at px i with the network's chance p_i where the placement allows it (prior_i); "none does" with chance
  prod_i (1 - p_i).  `statement_chance`, and `cut_nll` from it, computed in logs.  The finder's cuts (block edges) all
  say a line crosses.

Every term is multiplied by its label's weight, and `statements` counts the labels (their weights summed); the
trainer divides the summed loss by it.  The two answers add as one likelihood, with no weights between them.
Reads and writes nothing."""
from __future__ import annotations

import torch
import torch.nn.functional as F

def dense_nll(logit: torch.Tensor, target: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """The sum over px of weight * binary cross entropy; target 0/1, weight 0 where there is no label."""
    bce = F.binary_cross_entropy_with_logits(logit, target.to(logit.dtype), reduction="none")
    return (torch.where(weight > 0, weight.to(logit.dtype) * bce, torch.zeros_like(bce))).sum()


def _segment_logsumexp(t: torch.Tensor, seg: torch.Tensor, n: int) -> torch.Tensor:
    """(n,): log sum_i exp(t_i) over the i of each segment (-inf for one with none or with every t_i -inf)."""
    top = torch.full((n,), float("-inf"), dtype=t.dtype, device=t.device)
    top = top.scatter_reduce(0, seg, t.detach(), reduce="amax", include_self=True)
    top = torch.where(torch.isfinite(top), top, torch.zeros_like(top))
    total = torch.zeros(n, dtype=t.dtype, device=t.device).index_add(0, seg, torch.exp(t - top[seg]))
    return top + torch.log(total)


def _log_statement_chance(logit_flat: torch.Tensor, cuts_px: torch.Tensor, cuts_cut: torch.Tensor,
                          prior: torch.Tensor, says: torch.Tensor) -> torch.Tensor:
    """The log of each cut's statement chance (see `statement_chance`), computed in logs throughout.

    "None does": sum_i log(1 - p_i) = -sum_i softplus(z_i).  "A line crosses": with a_i = prior_i p_i taken in any
    order along the cut, 1 - prod_i (1 - a_i) = sum_i a_i prod_{j<i} (1 - a_j) exactly, so its log is a logsumexp of
    log prior_i + log p_i + sum_{j<i} log(1 - a_j): every term the log of a positive number, its gradient finite and
    never zero however surely the px are off."""
    order = torch.argsort(cuts_cut, stable=True)                 # each cut's px together, in their order
    z, cut = logit_flat[cuts_px[order]], cuts_cut[order]
    prior = prior[order].to(z.dtype)
    n = len(says)
    log_off = torch.zeros(n, dtype=z.dtype, device=z.device).index_add(0, cut, -F.softplus(z))     # log prod(1 - p)
    # log(1 - prior p) with p = sigmoid(z): softplus(z + log(1 - prior)) - softplus(z); log(1 - prior) is -inf at 1
    miss = torch.logaddexp(torch.zeros_like(z), z + torch.log1p(-prior.clamp(max=1))) - F.softplus(z)
    upto = torch.cumsum(miss.double(), 0)                         # in float64: a batch's sum runs to millions
    first = torch.ones(len(cut), dtype=torch.bool, device=z.device)
    first[1:] = cut[1:] != cut[:-1]
    start = torch.cummax(torch.where(first, torch.arange(len(cut), device=z.device), 0), 0).values
    before = ((upto - miss.double()) - (upto[start] - miss[start].double())).to(z.dtype)   # sum_{j<i} in its cut
    # a cut with no px of prior above 0 cannot cross: its terms are made finite before the logsumexp (so no NaN forms
    # in backward) and its chance set to 0 after; a px of prior 0 in any other cut adds exp(-inf) = 0
    possible = _possible(prior, cut, n)
    terms = torch.where(possible[cut], torch.log(prior) + F.logsigmoid(z) + before, torch.zeros_like(z))
    crosses = torch.where(possible, _segment_logsumexp(terms, cut, n), torch.full_like(log_off, float("-inf")))
    return torch.where(says.to(torch.bool), crosses, log_off)


def _possible(prior: torch.Tensor, cuts_cut: torch.Tensor, n: int) -> torch.Tensor:
    """Bool (n,): the cuts with a px of prior above 0."""
    return torch.zeros(n, dtype=torch.long, device=prior.device).index_add(0, cuts_cut, (prior > 0).long()) > 0


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
    in; `prior` (m,); `says` (n,) bool, True "a line crosses"; `weight` (n,).  ValueError for a "crosses" cut of
    weight above 0 with no px of prior above 0 (a statement nothing could make true: a malformed label); one of
    weight 0 adds nothing."""
    bad = says.to(torch.bool) & (weight > 0) & ~_possible(prior, cuts_cut, len(says))
    if bad.any():
        raise ValueError(f"{int(bad.sum())} 'a line crosses' cut(s) of weight above 0 have no px of prior above 0")
    term = -_log_statement_chance(logit_flat, cuts_px, cuts_cut, prior, says)
    weight = weight.to(term.dtype)
    return torch.where(weight > 0, weight * term, torch.zeros_like(term)).sum()


def statements(*weights) -> float:
    """How many statements the labels hold: the sum of every weight given (each px's dense weight, each cut's weight);
    a label of weight 1 is one statement."""
    return float(sum(float(w.sum()) for w in weights))
