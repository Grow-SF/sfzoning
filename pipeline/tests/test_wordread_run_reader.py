"""`scripts/wordread/run_reader.py`: a run's folder from a 2-step CPU recipe on synthetic crops, the pool shares counted
and not set, and the keep rule's numbers.  The real inputs (pages, crops, boxes) are replaced by a small environment."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from sfzmaps.solver import paths
from sfzmaps.solver.textread import model, store
from sfzmaps.solver.textread import read as R

SPEC = importlib.util.spec_from_file_location("run_reader", Path(__file__).parent.parent / "scripts" / "wordread"
                                              / "run_reader.py")
RR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RR)

NE = paths.QUARTERS.index("ne")
FORMS = {"R1": "R1", "C2": "C2"}


def _crop(rng, w=40):
    return np.where(rng.random((model.HEIGHT, w)) > 0.7, 0, 255).astype(np.uint8)


def _rect(x, y, w=40, h=20):
    return [x, y, x + w, y, x + w, y + h, x, y + h]


class FakeEnv(RR.Env):
    """Four boxes in NE (two on label points), a handful of crops; reads are fixed numbers, training is real."""

    def __init__(self):
        self.boxes_dir = Path("fake")
        self.extra_calls = []
        rng = np.random.default_rng(0)
        self.rng = rng

    def boxes(self):
        rows = []
        for k in range(4):
            r = dict(zip([f"g{i}{a}" for i in range(4) for a in "xy"], _rect(100.0 * k, 0.0)))
            rows.append({"sheet": "s1", "quarter": NE, "row": k, **r})
        return pd.DataFrame(rows)

    def codes(self):
        return R.CodeList(FORMS)

    def answers(self):
        return pd.DataFrame({"sheet": ["s1"] * 3, "point_x": [20.0, 120.0, 5000.0], "point_y": [10.0] * 3,
                             "truth": ["R1", "C2", "R1"], "quarter": [NE] * 3, "block": ["b1", "b2", "b2"],
                             "terra": ["R1", "C2", "R1"]})

    def answered_crops(self):
        rng = np.random.default_rng(1)
        ans = pd.DataFrame({"answer": ["R1", "C2", "R1"], "truth": ["R1", "C2", "R1"], "quarter": [NE] * 3,
                            "crop": [0, 1, 2], "angle": [0.0, 0.0, 180.0]})
        return ans, store.Crops.of([_crop(rng) for _ in range(3)], model.HEIGHT)

    def terra(self, boxes):
        c = dict(zip([f"c{i}{a}" for i in range(4) for a in "xy"], _rect(0.0, 0.0)))
        return pd.DataFrame([{"sheet": "s1", "text": "R1", "kind": "zoning code", "quarter": NE, **c}])

    def training(self, q, drop_terra_words):
        rng = np.random.default_rng(2)
        n = 12 - (4 if drop_terra_words else 0)
        real = store.Crops.of([_crop(rng) for _ in range(n)], model.HEIGHT)
        rows = pd.DataFrame({"text": ["R1", "C2"] * (n // 2), "weight": [1.0] * n})
        syn = store.Crops.of([_crop(rng) for _ in range(6)], model.HEIGHT)
        pool = RR.train.Pool(real, rows, np.arange(n), syn, ["R1"] * 6)
        return pool, rows, 4 if drop_terra_words else 0

    def direction(self, name):
        return model.Direction().eval()

    def extra(self, q, seed, own, sloppy, mask, boxes, rows):
        self.extra_calls.append((own, sloppy, mask))
        rng = np.random.default_rng(3)
        mk = lambda n: [(_crop(rng), "R1", {"mask": np.ones((model.HEIGHT, 40), bool)}) for _ in range(n)]  # noqa: E731
        return {"own": mk(3) if own else [], "sloppy": mk(9) if sloppy else []}

    def read(self, qname, reader, direction, boxes_q, mask, ans_jobs, workers):
        assert isinstance(reader, model.Reader) and not reader.training
        forms = R.CodeList(FORMS).forms

        def one(best):
            lf = np.full(len(forms), -30.0)
            lf[forms.index(best)] = -0.01
            read = (lf, -20.0, 0.001)
            return {"reads": [read, read], "best": [best, best], "p_classifier": [0.0, 0.0]}

        got = {int(r): one("R1" if r % 2 == 0 else "C2") for r in boxes_q["row"]}
        got.update({key: one("R1" if key[1] != 1 else "C2") for key, _, _ in ans_jobs})
        return got


FAST = {"steps": 2, "max_lr": 1e-3, "pct_start": 0.15, "synth": True}


def test_natural_shares_are_counts_and_nothing_else():
    own, sloppy = RR.natural_shares(n_real=100, n_own=300, n_sloppy=600, synth_share=0.4)
    assert own == pytest.approx(0.6 * 0.3) and sloppy == pytest.approx(0.6 * 0.6)
    assert 1.0 - 0.4 - own - sloppy == pytest.approx(0.6 * 0.1)           # the real pool the rest, by its count
    assert RR.natural_shares(10, 0, 30) == (0.0, 0.75)
    with pytest.raises(ValueError):
        RR.natural_shares(0, 0, 0)


def test_a_run_writes_its_folder_and_scores_the_labels(tmp_path):
    out = RR.run("base", "ne", 0, env=FakeEnv(), recipe=FAST, device="cpu", workers=1, out_root=tmp_path)
    assert out == tmp_path / "base-ne-s0"
    for f in ("labels.parquet", "headline.json", "box_numbers.json", "read_numbers.json", "run.json", "reader.pt"):
        assert (out / f).is_file()
    labels = pd.read_parquet(out / "labels.parquet")
    # the box at x 0..40 reads R1 (the label at 20 is R1: right), the box at 100..140 reads C2 (right); the third
    # label has no box (counted wrong, never dropped)
    assert labels["right"].tolist() == [1.0, 1.0, 0.0] and labels["row"].tolist() == [0, 1, -1]
    head = json.loads((out / "headline.json").read_text())
    assert (head["n"], head["right"], head["with_box"]) == (3, 2, 2)
    info = json.loads((out / "run.json").read_text())
    assert info["shares"] == {"own": 0.0, "sloppy": 0.0} and info["recipe"]["mask"] is False
    assert info["pools"]["real"] == 12 and info["pools"]["synth"] == 6
    assert info["terra_reading"]["n"] == 3 and info["terra_reading"]["terra_direction"] == 1.0
    assert model.load(out / "reader.pt") is not None


def test_a_variant_trains_on_the_counted_pools_and_masks(tmp_path):
    env = FakeEnv()
    out = RR.run("b", "ne", 1, own=True, sloppy=True, mask=True, env=env, recipe=FAST, device="cpu", workers=1,
                 out_root=tmp_path)
    assert env.extra_calls == [(True, True, True)]
    info = json.loads((out / "run.json").read_text())
    pools = info["pools"]
    assert pools == {"real": 8, "terra_words_replaced": 4, "synth": 6, "own": 3, "sloppy": 9}
    want = RR.natural_shares(8, 3, 9, train_synth := RR.train.SYNTH_SHARE)
    assert train_synth == 0.4 and (info["shares"]["own"], info["shares"]["sloppy"]) == pytest.approx(want)
    assert info["recipe"]["mask"] is True and info["train_log"]["own"] == 3 and info["train_log"]["sloppy"] == 9
    with pytest.raises(ValueError):
        RR.run("m", "ne", 0, mask=True, env=env, recipe=FAST, device="cpu", out_root=tmp_path)


def test_the_same_seed_trains_the_same_reader(tmp_path):
    a = RR.run("base", "ne", 3, env=FakeEnv(), recipe=FAST, device="cpu", workers=1, out_root=tmp_path / "x")
    b = RR.run("base", "ne", 3, env=FakeEnv(), recipe=FAST, device="cpu", workers=1, out_root=tmp_path / "y")
    sa = torch.load(a / "reader.pt", weights_only=True)["state_dict"]
    sb = torch.load(b / "reader.pt", weights_only=True)["state_dict"]
    assert all(torch.equal(sa[k], sb[k]) for k in sa)


def _fake_run(root, name, seed, right, terra, blocks):
    d = root / RR.run_name(name, "ne", seed)
    d.mkdir(parents=True)
    pd.DataFrame({"label": range(len(right)), "row": range(len(right)), "right": right, "block": blocks,
                  "quarter": NE}).to_parquet(d / "labels.parquet")
    (d / "run.json").write_text(json.dumps({"terra_reading": {"terra_direction": terra}}))
    (d / "headline.json").write_text(json.dumps({"share": float(np.mean(right)), "ci": [0, 1]}))


def test_the_keep_rule_needs_both_seeds_and_a_paired_interval_off_zero(tmp_path):
    blocks = [f"b{k // 2}" for k in range(80)]
    base = [0.0, 1.0] * 20 + [1.0, 0.0] * 20
    gain = [1.0] * 40 + [1.0, 0.0] * 20                 # right wherever base was, plus the first 40 wrong ones
    for seed in (0, 1):
        _fake_run(tmp_path, "base", seed, base, 0.95 - 0.01 * seed, blocks)
        _fake_run(tmp_path, "a", seed, gain, 0.94, blocks)
        _fake_run(tmp_path, "b", seed, base, 0.95, blocks)       # no change at all
    got = RR.decide("ne", root=tmp_path)
    assert got["a"]["headline_rises"] and got["a"]["paired_mean_of_seeds"]["lo"] > 0
    assert got["a"]["terra_seed_noise"] == pytest.approx(0.01) and got["a"]["terra_ok"]    # -0.005 inside the noise
    assert got["a"]["keep_before_the_look"] is True
    assert got["b"]["headline_rises"] is False and got["b"]["keep_before_the_look"] is False
    # one seed rising and the other not is no verdict
    (tmp_path / RR.run_name("a", "ne", 1)).rename(tmp_path / "gone")
    _fake_run(tmp_path, "a", 1, base, 0.95, blocks)
    assert RR.decide("ne", root=tmp_path)["a"]["headline_rises"] is False
    # a fall at Terra's boxes beyond the seed noise stops it
    for seed in (0, 1):
        _fake_run(tmp_path / "t", "base", seed, base, 0.95 - 0.01 * seed, blocks)
        _fake_run(tmp_path / "t", "a", seed, gain, 0.90, blocks)
    t = RR.decide("ne", root=tmp_path / "t")["a"]
    assert t["headline_rises"] and not t["terra_ok"] and not t["keep_before_the_look"]
