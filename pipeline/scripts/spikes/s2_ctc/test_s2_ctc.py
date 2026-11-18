"""Spike S2's two checks that keep its numbers honest:

  - the CTC code posterior is normalised: the exact CTC scores of every string sum to 1, so "other" (the remainder)
    is a probability and the codes' masses are exact;
  - the hold-outs leak nothing: no training crop sits in the held quarter, the validation area or an answered block,
    and no answered label's block is among the training crops' blocks (a second route: each crop's centre map point
    and the lot under it, not the footprint check crops.py made).

    uv run pytest scripts/spikes/s2_ctc/test_s2_ctc.py
"""
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402


def _log_probs(frames: int, chars: str, seed: int = 0) -> torch.Tensor:
    """Random CTC frames whose mass is on `chars` and the blank only (float64 log-softmax)."""
    g = torch.Generator().manual_seed(seed)
    logits = torch.full((frames, len(C.VOCAB) + 1), -1e9, dtype=torch.float64)
    keep = [C.VOCAB.index(c) for c in chars] + [C.BLANK]
    logits[:, keep] = torch.randn((frames, len(keep)), generator=g, dtype=torch.float64) * 2
    return torch.log_softmax(logits, -1)


def test_every_string_sums_to_one():
    chars, frames = "RH-", 5
    lp = _log_probs(frames, chars)
    strings = [""] + ["".join(t) for n in range(1, frames + 1) for t in itertools.product(chars, repeat=n)]
    total = np.exp(C.string_log_probs(lp, strings)).sum()
    assert total == pytest.approx(1.0, abs=1e-12)


def test_code_posterior_is_exact_and_normalised():
    lp = _log_probs(6, "RH-12P", seed=3)
    forms = {"RH-1": "RH-1", "RH-2": "RH-2", "RH2": "RH-2", "P": "P"}
    post = C.code_posterior(lp, forms)
    assert sum(post.values()) == pytest.approx(1.0, abs=1e-12)
    assert min(post.values()) >= 0.0
    s = np.exp(C.string_log_probs(lp, ["RH-2", "RH2", "", "P", "RH-1"]))
    assert post["RH-2"] == pytest.approx(s[0] + s[1], rel=1e-12)
    assert post["none"] == pytest.approx(s[2], rel=1e-12)
    assert post["other"] == pytest.approx(1 - s.sum(), abs=1e-12)


def _meta():
    f = C.OUT / "crops_meta.pkl"
    if not f.exists():
        pytest.skip("crops.py has not run")
    return C.load_crops().meta


def test_no_training_crop_in_a_held_area():
    from sfzmaps.solver import paths
    from sfzmaps.solver.textnet import data
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import crops as K
    m = _meta()
    train = m[m["split"] == "train"]
    blocks = set(json.loads((paths.READER_TEXT_DIR / "held.json").read_text())["blocks"])
    assert len(train) > 10000
    assert (train["quarter"] != K.HOLD_QUARTER).all()
    x0, y0, x1, y1 = data.VALIDATION_AREA
    inside = (train["mx"] >= x0) & (train["mx"] < x1) & (train["my"] >= y0) & (train["my"] < y1)
    assert not inside.any()
    assert not train["block"].isin(blocks).any()


def test_no_answered_block_in_training():
    m = _meta()
    train = m[m["split"] == "train"]
    answered = m[m["split"] == "answered"]
    assert len(answered) == 668
    theirs = set(answered["lab_block"]) | {b for bs in answered["case_blocks"] for b in bs}
    theirs.discard("")
    assert theirs, "the answered labels carry their blocks"
    assert not (set(train["block"]) & theirs)
    boxes = {(s, tuple(b)) for s, b in zip(answered["sheet"], answered["box"])}
    assert not any((s, tuple(b)) in boxes for s, b in zip(train["sheet"], train["box"]))
