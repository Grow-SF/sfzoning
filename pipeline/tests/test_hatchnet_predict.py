"""The learned reader's lot predictions (`sfzmaps.solver.hatchnet.predict`): each lot's district is its px's district
probabilities pooled with the learned weights (the trained objective, `train.lot_logprob`) over all its px on every
sheet, once; each quarter's lots come from the fold that withheld that quarter."""
import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn as nn

from sfzmaps.solver.hatchnet import data, model, predict, train

NAMES = model.NAMES
K = len(NAMES)


def says(codes, sure: float = 6.0) -> np.ndarray:
    """(6, P) float32 district logits: each px's own district (`codes`, names) `sure` above the other five."""
    idx = np.array([NAMES.index(c) for c in codes])
    z = np.zeros((K, len(idx)), np.float32)
    z[idx, np.arange(len(idx))] = sure
    return z


def lots_of(quarters) -> pd.DataFrame:
    n = len(quarters)
    return pd.DataFrame({"lot": np.arange(n), "blklot": [f"{1000 + i}001" for i in range(n)], "quarter": quarters})


def probs(frame: pd.DataFrame) -> np.ndarray:
    return frame[[f"p_{c}" for c in NAMES]].to_numpy()


def test_lot_pooled_by_weight():
    """A lot whose few px of high learned weight say 1R and whose many px of low weight say U is 1R: the px are pooled
    by the weights softmaxed over the lot, not counted."""
    codes = ["1R"] * 2 + ["U"] * 8
    weight = np.array([4.0] * 2 + [-1.0] * 8, np.float32)
    lot = np.zeros(10, np.int32)
    assert max(set(codes), key=codes.count) == "U"                 # a px majority would say U
    got = predict.lot_districts([(lot, says(codes), weight)], lots_of(["nw"]))
    assert got["code"].tolist() == ["1R"]
    w = np.exp(weight.astype(np.float64) - weight.max())
    w /= w.sum()
    px = torch.softmax(torch.from_numpy(says(codes).T.astype(np.float64)), -1).numpy()
    np.testing.assert_allclose(probs(got)[0], w @ px, rtol=1e-12)
    assert got["p"].iloc[0] == probs(got)[0].max()


def test_lot_on_two_sheets_pooled_once():
    """A lot's px on two sheets give what one pooled set of the same px gives (the trained pooling, `lot_logprob`),
    not the mean of the two sheets' pooled lots."""
    a_codes, b_codes = ["1R"] * 2, ["2R"] * 6 + ["C"] * 2
    a = (np.array([0, 0, 1], np.int32), says(a_codes + ["U"]), np.array([0.5, 0.5, 0.0], np.float32))
    b = (np.array([0] * 8 + [1], np.int32), says(b_codes + ["HI"]), np.array([0.0] * 6 + [1.5, 1.5, 2.0], np.float32))
    one = tuple(np.concatenate([x, y], -1) for x, y in zip(a, b))
    two_sheets = predict.lot_districts([a, b], lots_of(["nw", "ne"]))
    pooled = predict.lot_districts([one], lots_of(["nw", "ne"]))
    np.testing.assert_allclose(probs(two_sheets), probs(pooled), rtol=1e-12)
    lp = train.lot_logprob(torch.from_numpy(one[1].T.astype(np.float64)), torch.from_numpy(one[2].astype(np.float64)),
                           torch.from_numpy(one[0].astype(np.int64)), 2)
    np.testing.assert_allclose(probs(two_sheets), np.exp(lp.numpy()), rtol=1e-12)
    each = [probs(predict.lot_districts([s], lots_of(["nw", "ne"])))[0] for s in (a, b)]
    assert not np.allclose(probs(two_sheets)[0], np.mean(each, 0), atol=1e-3)     # the test tells the two apart
    assert two_sheets["n_sheets"].tolist() == [2, 2] and pooled["n_sheets"].tolist() == [1, 1]
    assert two_sheets["px"].tolist() == [10, 2]


def test_uncovered_lot_has_no_row():
    """A lot with no px on any sheet (all its place street, -1) has no row; the others keep theirs."""
    lot = np.array([[0, 0, -1, 2], [0, -1, -1, 2]], np.int32)
    z = np.zeros((K,) + lot.shape, np.float32)
    got = predict.lot_districts([(lot, z, np.zeros(lot.shape, np.float32))], lots_of(["nw", "ne", "sw"]))
    assert got["lot"].tolist() == [0, 2]
    assert got["blklot"].tolist() == ["1000001", "1002001"] and got["quarter"].tolist() == ["nw", "sw"]


class Says(nn.Module):
    """A stand-in network: every px says the district `code`, `sure` above the rest, with weight 0."""

    def __init__(self, code: str, sure: float):
        super().__init__()
        self.inputs, self.k, self.sure = model.INPUTS, NAMES.index(code), sure
        self.unused = nn.Parameter(torch.zeros(()))

    def forward(self, x):
        out = torch.zeros((x.shape[0], model.OUTPUTS) + tuple(x.shape[2:]))
        out[:, model.DISTRICTS.start + self.k] = self.sure
        return out


def two_sheets():
    """Two 32 x 32 pages: lots 0-3 (nw, ne, sw, se) on the first, lots 4-5 (se, nw) on the second, and lot 6 (ne) on
    both."""
    rasters = []
    for name, ids in (("a", [0, 1, 2, 3, 6]), ("b", [4, 5, 6])):
        lot = np.full((32, 32), -1, np.int32)
        for i, v in enumerate(ids):
            lot[2 + 6 * i:6 + 6 * i, 2:30] = v
        rasters.append(data.SheetRasters(name, np.ones((32, 32), np.float32), np.ones((32, 32), np.uint8), lot,
                                         np.ones((32, 32), bool), 5.0))
    return rasters, lots_of(["nw", "ne", "sw", "se", "se", "nw", "ne"])


def test_lot_predicted_by_its_own_fold():
    """Each lot of quarter q takes the prediction of the fold that withheld q (answers_<q> with answers, reads_<q>
    reads-only), never another fold's or the all-quarters model's; all.parquet's rows are the all-quarters model's."""
    nets = {}
    for i, name in enumerate(train.run_name(q, w) for q, w in train.RUNS):
        nets[name] = Says(NAMES[i % K], 2.0 + 0.5 * i)       # each model its own code and sureness: its own p
    rasters, lots = two_sheets()
    got, notes = predict.predict_folds(nets.__getitem__, rasters, lots, device="cpu")
    for out, kind in (("with_answers", "answers"), ("reads_only", "reads"), ("all", None)):
        f = got[out]
        assert f["lot"].tolist() == list(range(7))
        for _, row in f.iterrows():
            name = f"{kind}_{row['quarter']}" if kind else "answers_all"
            net = nets[name]
            assert row["model"] == name
            assert row["code"] == NAMES[net.k]
            assert row["p"] == pytest.approx(np.exp(net.sure) / (np.exp(net.sure) + K - 1), rel=1e-12)
    assert got["with_answers"].set_index("lot").loc[6, "n_sheets"] == 2
    assert notes["answers_nw"]["sheets"] == ["a", "b"] and notes["answers_sw"]["sheets"] == ["a"]


def test_contact_order_disagreements_first():
    """The contact sheet's tiles: the lots whose prediction differs from the owner's answer first, each group surest
    first, ties by blklot."""
    t = pd.DataFrame({"blklot": ["a", "b", "c", "d", "e"], "code": ["1R", "2R", "C", "1R", "2R"],
                      "p": [0.9, 0.6, 0.6, 0.99, 0.7], "answer": ["1R", "1R", "2R", "1R", "1R"]})
    assert t.loc[predict.contact_order(t), "blklot"].tolist() == ["e", "b", "c", "d", "a"]
