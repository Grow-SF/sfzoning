"""The CTC sums the reader's posterior is made of, and the reading direction.

A read: one crop's CTC frames as log-probabilities (frames x classes, float64 log-softmax of the logits).
  - p(s | x) for a string s: the sum over every frame path that collapses to s (merge repeats, drop blanks), by CTC's
    exact forward algorithm (`torch.nn.functional.ctc_loss`); -inf where s can't fit the frames.  Distinct strings
    are disjoint events, so their probabilities add.
  - the code list's printed forms each get p(s | x); "none" is the empty string, the all-blank path (spike S2's
    definition: no text read); "other" is every other string, 1 - the forms - none (exact, up to rounding; floored
    at 0).

Read both ways.  A word box gives an axis (modulo 180 degrees); its crop is read along the axis (0 degrees: the
axis's own angle, counter-clockwise from left-to-right) and turned a half turn (180 degrees).  The reading's
posterior is the mixture

    p(h | box) = (1 - q) p(h | x_0) + q p(h | x_180),    q = p(180 degrees | box)

with q the 0/180 classifier's evidence on the measured prior (`flip_probability`).  The classifier was trained on
balanced classes (each crop as cut and turned), so its odds are a likelihood ratio; the prior is the share of the
training words at that axis that read the other way (`axis_prior`).
"""
import numpy as np
import torch
import torch.nn.functional as F

from sfzmaps.solver.textread import model

AXIS_SD = 5.0          # degrees: the axis prior's kernel (a word's axis a few degrees either way says the same)
AXIS_STEP = 0.1        # degrees between the axis prior's table entries
PRIOR_COUNT = 0.5      # Jeffreys' half count of each direction in the axis prior
D_CLIP = 1e-9          # the classifier's probability is kept this far from 0 and 1 (its odds stay finite)


def log_softmax(logits) -> torch.Tensor:
    """A crop's logits (frames x classes, any float) as float64 log-probabilities."""
    return F.log_softmax(torch.as_tensor(np.asarray(logits)).double(), -1)


def string_log_probs(log_probs: torch.Tensor, strings: list[str]) -> np.ndarray:
    """log p(s | x) for each of `strings` (see the module), for one crop's `log_probs` (frames x classes)."""
    n = len(strings)
    lens = [len(s) for s in strings]
    tgt = torch.zeros((n, max(1, max(lens, default=1))), dtype=torch.long)
    for i, s in enumerate(strings):
        if s:
            tgt[i, :len(s)] = torch.tensor(model.encode(s))
    lp = log_probs[:, None, :].expand(-1, n, -1)
    nll = F.ctc_loss(lp, tgt, torch.full((n,), log_probs.shape[0], dtype=torch.long),
                     torch.tensor(lens, dtype=torch.long), blank=model.BLANK, reduction="none", zero_infinity=False)
    return (-nll).numpy()


def best_path(log_probs: torch.Tensor) -> str:
    """The best path's string: each frame's likeliest class, repeats merged, blanks dropped."""
    out, last = [], None
    for k in log_probs.argmax(-1).tolist():
        if k != last and k != model.BLANK:
            out.append(model.VOCAB[k])
        last = k
    return "".join(out)


def read(log_probs: torch.Tensor, forms: list[str]) -> tuple[np.ndarray, float, float]:
    """One read over the printed forms: log p(form | x) for each of `forms`, log p(none | x) and p(other | x)."""
    lp = string_log_probs(log_probs, list(forms) + [""])
    with np.errstate(over="ignore"):
        total = float(np.exp(lp).sum())
    return lp[:-1], float(lp[-1]), max(0.0, 1.0 - total)


def mixture(q: float, reads: tuple) -> np.ndarray:
    """The mixture of the two reads (0 and 180 degrees, each `read`'s output), in linear probability: the forms,
    then "other", then "none"."""
    out = []
    for (lf, ln, po), w in zip(reads, (1.0 - q, q)):
        out.append(w * np.concatenate([np.exp(lf), [po], [np.exp(ln)]]))
    return out[0] + out[1]


def closed_log(q: float, reads: tuple, code_of: np.ndarray, n_codes: int) -> np.ndarray:
    """log p(code | box, a code) over the codes (0..n_codes-1; `code_of` each form's code index): the mixture's
    forms summed by code and renormalised, in log space."""
    lf = np.logaddexp(np.log(max(1.0 - q, 1e-300)) + reads[0][0], np.log(max(q, 1e-300)) + reads[1][0])
    by = np.full(n_codes, -np.inf)
    np.logaddexp.at(by, code_of, lf)
    m = by.max()
    if not np.isfinite(m):
        return np.full(n_codes, -np.log(n_codes))
    return by - (m + np.log(np.exp(by - m).sum()))


# --- the direction --------------------------------------------------------------------------------------------------

def axis_and_turn(angle_deg) -> tuple[np.ndarray, np.ndarray]:
    """A reading angle (degrees, counter-clockwise from left-to-right) as its axis (modulo 180) and whether it reads
    the axis's half turn (the angle modulo 360 is 180 or more)."""
    a = np.mod(np.asarray(angle_deg, float), 360.0)
    return np.mod(a, 180.0), a >= 180.0


def axis_prior(axes: np.ndarray, turned: np.ndarray, sd: float = AXIS_SD, step: float = AXIS_STEP) -> np.ndarray:
    """The share of words that read their axis's half turn, as a table over the axis (0, step, ..., 180 - step):
    each word weighted by a Gaussian of `sd` degrees in the axis, not wrapped at 0 and 180 (the two ends of the axis
    read opposite ways), with PRIOR_COUNT of each direction added.  `axes` in [0, 180), `turned` bool."""
    grid = np.arange(0.0, 180.0, step)
    axes, turned = np.asarray(axes, float), np.asarray(turned, bool)
    w_all, w_turned = np.zeros(len(grid)), np.zeros(len(grid))
    for k in range(0, len(axes), 4096):
        a, t = axes[k:k + 4096], turned[k:k + 4096]
        w = np.exp(-0.5 * ((grid[:, None] - a[None, :]) / sd) ** 2)
        w_all += w.sum(1)
        w_turned += w[:, t].sum(1)
    return (w_turned + PRIOR_COUNT) / (w_all + 2 * PRIOR_COUNT)


def prior_at(table: np.ndarray, axis, step: float = AXIS_STEP) -> np.ndarray:
    """The axis prior (`axis_prior`'s table) at each axis (degrees modulo 180), at the nearest table entry."""
    k = np.rint(np.mod(np.asarray(axis, float), 180.0) / step).astype(np.int64) % len(table)
    return np.asarray(table)[k]


def flip_probability(p_classifier, prior) -> np.ndarray:
    """q = p(180 degrees): the classifier's balanced-class probability as a likelihood ratio on the prior."""
    p = np.clip(np.asarray(p_classifier, float), D_CLIP, 1 - D_CLIP)
    lr = p / (1 - p)
    r = np.asarray(prior, float)
    return r * lr / (r * lr + 1 - r)
