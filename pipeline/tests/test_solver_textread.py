"""The text reader (`sfzmaps.solver.textread`): the CTC sums, the direction mixture, the held-out split, the crop of
one of our boxes, and the plain-torch networks against spike S2's weights."""
import itertools

import numpy as np
import pandas as pd
import pytest
import torch

from sfzmaps import config
from sfzmaps.solver import paths
from sfzmaps.solver.textread import crops as K
from sfzmaps.solver.textread import ctc, model, read, store

SPIKE = config.DATA_DIR / "solver" / "experiments" / "spike-s2"


def _log_probs(frames: int, chars: str, seed: int = 0) -> torch.Tensor:
    """Random CTC frames whose mass is on `chars` and the blank only (float64 log-softmax)."""
    g = torch.Generator().manual_seed(seed)
    logits = torch.full((frames, model.CLASSES), -1e9, dtype=torch.float64)
    keep = [model.VOCAB.index(c) for c in chars] + [model.BLANK]
    logits[:, keep] = torch.randn((frames, len(keep)), generator=g, dtype=torch.float64) * 2
    return torch.log_softmax(logits, -1)


def _brute_force(lp: torch.Tensor, chars: str) -> dict[str, float]:
    """p(s | x) of every string, summed over every frame path of the classes `chars` and the blank."""
    classes = [model.VOCAB.index(c) for c in chars] + [model.BLANK]
    out: dict[str, float] = {}
    p = lp.exp().numpy()
    for path in itertools.product(classes, repeat=lp.shape[0]):
        s, last = [], None
        for k in path:
            if k != last and k != model.BLANK:
                s.append(model.VOCAB[k])
            last = k
        key = "".join(s)
        out[key] = out.get(key, 0.0) + float(np.prod([p[t, k] for t, k in enumerate(path)]))
    return out


def test_ctc_forward_over_printed_forms_is_the_sum_over_paths():
    lp = _log_probs(5, "RH1", seed=1)
    brute = _brute_force(lp, "RH1")
    strings = ["RH", "R", "HH", "RH1", "R1H1", "", "RRH", "H1H"]
    got = np.exp(ctc.string_log_probs(lp, strings))
    want = np.array([brute.get(s, 0.0) for s in strings])
    np.testing.assert_allclose(got, want, rtol=1e-10, atol=1e-300)
    assert sum(brute.values()) == pytest.approx(1.0, abs=1e-12)


def test_read_splits_the_mass_into_forms_none_and_other():
    lp = _log_probs(5, "RH1", seed=2)
    brute = _brute_force(lp, "RH1")
    forms = ["RH", "RH1", "R"]
    log_forms, log_none, p_other = ctc.read(lp, forms)
    np.testing.assert_allclose(np.exp(log_forms), [brute[f] for f in forms], rtol=1e-10)
    assert np.exp(log_none) == pytest.approx(brute[""], rel=1e-10)
    rest = sum(v for s, v in brute.items() if s not in forms and s != "")
    assert p_other == pytest.approx(rest, abs=1e-12)


def test_a_form_too_long_for_the_frames_has_no_mass():
    lp = _log_probs(3, "RH", seed=0)
    assert np.exp(ctc.string_log_probs(lp, ["RHRH"]))[0] == 0.0


def _reads(seed: int, forms: list[str]):
    return ctc.read(_log_probs(6, "RH12", seed=seed), forms)


def test_direction_mixture_weights_each_read_by_its_probability():
    forms = ["RH-1", "RH-2", "RH1", "R"]
    forms = [f for f in forms if all(c in "RH12" for c in f)]
    r0, r180 = _reads(3, forms), _reads(4, forms)
    q = 0.3
    mix = ctc.mixture(q, (r0, r180))

    def vec(r):
        return np.concatenate([np.exp(r[0]), [r[2]], [np.exp(r[1])]])
    np.testing.assert_allclose(mix, 0.7 * vec(r0) + 0.3 * vec(r180), rtol=1e-12)
    assert mix.sum() == pytest.approx(1.0, abs=1e-12)
    # the mixture, not the better of the two reads
    assert not np.allclose(mix, vec(r0)) and not np.allclose(mix, vec(r180))


def test_closed_posterior_sums_a_codes_forms_and_renormalises():
    forms = ["RH1", "R1H", "R", "H"]
    code_of = np.array([0, 0, 1, 2])
    r0, r180 = _reads(5, forms), _reads(6, forms)
    q = 0.4
    got = np.exp(ctc.closed_log(q, (r0, r180), code_of, 3))
    f = 0.6 * np.exp(r0[0]) + 0.4 * np.exp(r180[0])
    want = np.array([f[0] + f[1], f[2], f[3]]) / f.sum()
    np.testing.assert_allclose(got, want, rtol=1e-10)


def test_flip_probability_is_the_classifiers_odds_on_the_prior():
    assert ctc.flip_probability(0.8, 0.5) == pytest.approx(0.8)
    assert ctc.flip_probability(0.5, 0.02) == pytest.approx(0.02)
    # odds 4 on prior odds 1/9: posterior odds 4/9
    assert ctc.flip_probability(0.8, 0.1) == pytest.approx(4 / 13)


def test_axis_prior_reads_each_end_of_the_axis_its_own_way():
    # upright words: axis near 0 read as is, near 180 (sloping down) turned; sideways words mostly bottom-to-top
    axes = np.r_[np.full(500, 0.0), np.full(300, 178.0), np.full(90, 90.0), np.full(10, 90.0)]
    turned = np.r_[np.zeros(500, bool), np.ones(300, bool), np.zeros(90, bool), np.ones(10, bool)]
    table = ctc.axis_prior(axes, turned)
    at = lambda a: float(ctc.prior_at(table, a))      # noqa: E731
    assert at(0.5) < 0.01 and at(179.5) > 0.99
    assert at(90.0) == pytest.approx((10 + 0.5) / (100 + 1), abs=0.01)
    assert ctc.axis_and_turn([0, 180, 270, 359])[1].tolist() == [False, True, True, True]
    np.testing.assert_allclose(ctc.axis_and_turn([0, 180, 270, 359])[0], [0, 0, 90, 179])


def _rows(n: int = 40, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({"text": rng.choice(["RH-1", "", None], n, p=[0.6, 0.3, 0.1]),
                       "kind": "zoning code", "weight": 1.0, "quarter": rng.integers(0, 4, n)})
    df.loc[df["text"] == "", "kind"] = "none"
    for q in paths.QUARTERS:
        df[f"touch_{q}"] = False
    for k, q in enumerate(df["quarter"]):        # every crop touches its own quarter, some a neighbour too
        df.at[k, f"touch_{paths.QUARTERS[q]}"] = True
        if rng.random() < 0.2:
            df.at[k, f"touch_{paths.QUARTERS[(q + 1) % 4]}"] = True
    df["touch_blocks"] = rng.random(n) < 0.15
    return df


def test_a_readers_training_rows_touch_neither_its_quarter_nor_an_answered_block():
    df = _rows()
    for q, name in enumerate(paths.QUARTERS):
        idx = store.training_rows(df, q)
        got = df.iloc[idx]
        assert len(got)
        assert not got[f"touch_{name}"].any() and not got["touch_blocks"].any()
        assert (got["quarter"] != q).all() and got["text"].notna().all()
        want = df[~df[f"touch_{name}"] & ~df["touch_blocks"] & df["text"].notna()].index
        assert list(got.index) == list(want)
        words = df.iloc[store.training_rows(df, q, text_only=True)]
        assert (words["kind"] != "none").all() and set(words.index) <= set(got.index)


def test_the_built_split_holds_out_each_quarter_and_the_answered_blocks():
    """A second route on the stage's own crops: the map point of each training crop's centre is outside its reader's
    quarter, and the block under it is no answered label's block."""
    f = store.CROPS_DIR / "crops.parquet"
    if not f.is_file():
        pytest.skip("solver-text-crops has not run")
    rows = pd.read_parquet(f)
    ans = pd.read_parquet(store.CROPS_DIR / "answered.parquet")
    answered = (set(ans["lab_block"]) | set(ans["block"])) - {""}
    cols = ["sheet", "box_x0", "box_y0", "box_x1", "box_y1"]
    answered_boxes = {tuple(r) for r in ans.loc[ans["crop"] >= 0, cols].itertuples(index=False)}
    assert answered_boxes
    for q in range(len(paths.QUARTERS)):
        got = rows.iloc[store.training_rows(rows, q)]
        assert len(got) > 10000
        assert (got["quarter"] != q).all()
        assert not got["block"].isin(answered).any()
        mine = got[got["series"] == paths.SERIES]
        assert not any(tuple(r) in answered_boxes for r in mine[cols].itertuples(index=False))


def test_a_box_is_cut_along_its_axis():
    """A word printed sideways (bottom-to-top) and the same word upright give the same crop, read at their axes."""
    page = np.full((200, 200), 255, np.uint8)
    page[90:110, 60:62] = 0          # an "L": a stem and a foot, asymmetric under every turn
    page[108:110, 60:72] = 0
    page[90:110, 80:82] = 0          # and an "I" to its right
    corners = np.array([[58, 88], [84, 88], [84, 112], [58, 112]], float)
    upright = K.box_crop(page, corners, 26, 24, 0.0)
    turned_page = np.ascontiguousarray(np.rot90(page))           # counter-clockwise: the word reads bottom-to-top
    h = page.shape[1]
    tc = np.array([[y, h - x] for x, y in corners], float)
    sideways = K.box_crop(turned_page, tc, 26, 24, 90.0)
    np.testing.assert_array_equal(upright, sideways)
    np.testing.assert_array_equal(K.box_crop(page, corners, 26, 24, 180.0), np.rot90(upright, 2))


def test_the_label_holding_box_is_the_nearest_that_holds_the_point():
    def box(cx, cy, sheet="s"):
        c = [(cx - 10, cy - 5), (cx + 10, cy - 5), (cx + 10, cy + 5), (cx - 10, cy + 5)]
        return {"sheet": sheet, **{f"g{i}{a}": v for i, p in enumerate(c) for a, v in zip("xy", p)}}
    boxes = pd.DataFrame([box(100, 100), box(104, 100), box(300, 300), box(100, 100, "t")])
    assert read.holding_box(boxes, "s", 103, 101) == 1
    assert read.holding_box(boxes, "s", 99, 100) == 0
    assert read.holding_box(boxes, "s", 200, 200) == -1
    assert read.holding_box(boxes, "u", 100, 100) == -1


def test_a_no_code_answer_is_right_on_none_or_other():
    codes = read.CodeList({"RH-1": "RH-1", "RH1": "RH-1", "P": "P"})
    assert codes.codes == ["P", "RH-1"]
    open_p = {"RH-1": 0.05, "P": 0.25, "other": 0.3, "none": 0.4}
    log_closed = np.log(np.array([0.3, 0.7]))
    s = read.label_scores(open_p, log_closed, codes, "")
    assert s["open_top1"] == 1.0 and s["open_nll"] == pytest.approx(-np.log(0.7))
    assert np.isnan(s["closed_top1"])
    s = read.label_scores(open_p, log_closed, codes, "P")
    assert s["open_top1"] == 0.0 and s["open_top3"] == 1.0 and s["closed_top1"] == 0.0
    assert s["closed_nll"] == pytest.approx(-np.log(0.3))


# --- the networks against spike S2's -------------------------------------------------------------------------------

def _spike(*names):
    files = [SPIKE / n for n in names]
    if not all(f.is_file() for f in files):
        pytest.skip("spike S2's files are not here")
    return files


def test_the_plain_torch_reader_gives_spike_s2s_reader_c_its_outputs():
    weights, crops_f, frames_f = _spike("models/C.pt", "crops.npz", "frames_C.npz")
    torch.set_num_threads(1)
    net = model.load(weights)
    crops = store.Crops.load(crops_f)
    with np.load(frames_f) as z:
        index, cut, off = z["index"], z["cut"], z["cut_off"]
    for k in np.linspace(0, len(index) - 1, 12).astype(int):
        x, n = model.to_input([crops[int(index[k])]])
        with torch.inference_mode():
            got = net(x)[0, :n[0]].numpy()
        np.testing.assert_allclose(got, cut[off[k]:off[k + 1]], rtol=0, atol=1e-4)


def test_the_plain_torch_classifier_gives_spike_s2s_classifier_d_its_outputs():
    weights, crops_f, dir_f = _spike("models/D.pt", "crops.npz", "direction.npz")
    torch.set_num_threads(1)
    net = model.load(weights, "direction")
    crops = store.Crops.load(crops_f)
    with np.load(dir_f) as z:
        index, p_flip = z["index"], z["p_flip"]
    for k in np.linspace(0, len(index) - 1, 12).astype(int):
        x, _ = model.to_input([crops[int(index[k])]])
        with torch.inference_mode():
            got = float(torch.sigmoid(net(x))[0])
        assert got == pytest.approx(float(p_flip[k]), abs=1e-6)


def test_the_reader_shrinks_a_crops_width_by_the_stride_to_one_row():
    net = model.Reader().eval()
    x, n = model.to_input([np.full((model.HEIGHT, 50), 255, np.uint8)])
    with torch.inference_mode():
        out = net(x)
    assert x.shape[-1] == 52 and n == [13]
    assert tuple(out.shape) == (1, 13, model.CLASSES)
