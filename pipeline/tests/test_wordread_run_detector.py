"""scripts/wordread/run_detector.py (Task 8): the detector run with the word heads on the synthetic `maps` fixture (a
2-step CPU training), its word outputs stitched from tiles, the fit's held-out areas, the words placed and kept in
the held-out quarter, the run's folder."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "wordread"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "textlines"))
import numpy as np  # noqa: E402
import pytest  # noqa: E402
import run_detector as RD  # noqa: E402
import torch  # noqa: E402
import variant  # noqa: E402
from test_solver_textnet import OLD, TINY, maps  # noqa: E402,F401

from sfzmaps.solver import paths  # noqa: E402
from sfzmaps.solver.textnet import data, model, predict, reader_stage, train  # noqa: E402
from sfzmaps.solver.wordread import doubt, words  # noqa: E402

WORDS_TINY = variant.dataclasses.replace(TINY, labels="lines-unknown", heads="words")


def _quarter_of_page(sd) -> int:
    counts = [(reader_stage.page_quarters(sd) == i).sum() for i in range(4)]
    return int(np.argmax(counts))


def _stub_score(out, quarter, scored, chances, extra):
    (out / "scores.json").write_text(json.dumps({"stub": True}))


# --- the word outputs, stitched ------------------------------------------------------------------------------------

@pytest.mark.parametrize("half", [False, True])
def test_word_outputs_stitched_from_tiles_are_one_pass_over_the_page(half):
    torch.manual_seed(0)
    net = train.build_net(train.Recipe(width=4, half_size=half, heads="words")).eval()
    ink = np.random.default_rng(5).random((150, 140)) < 0.2
    ink[60:75, 20:120] = True                  # a word-like bar across the tiles' seams at 32 / 64 px
    tile = 2 * net.multiple
    got = predict.word_outputs(net, ink, "cpu", tile=tile)
    with torch.no_grad():
        ph, pw = (-150) % net.multiple, (-140) % net.multiple
        raw = net(torch.from_numpy(np.pad(ink, ((0, ph), (0, pw))).astype(np.float32))[None, None])[0, :, :150, :140]
    # every px the numbers one pass over the whole page gives (the context reaches as far as the network sees)
    assert np.allclose(got.core, raw[model.CORE].numpy(), atol=1e-4)
    assert np.allclose(got.geo[:4], model.distances(raw[model.GEO]).numpy(), rtol=1e-4, atol=1e-4)
    assert np.allclose(got.geo[4:], raw[5:7].numpy(), atol=1e-4)
    assert np.allclose(got.tag, raw[model.TAG].numpy(), atol=1e-4)
    assert np.array_equal(got.chance, np.round(255 * torch.sigmoid(raw[0]).numpy()).astype(np.uint8)) or \
        np.abs(got.chance.astype(int) - np.round(255 * torch.sigmoid(raw[0]).numpy())).max() <= 1
    # PAD alone (the per-ink outputs' context) is not enough for the seams: the context is the network's reach
    assert predict.context(net) > predict.PAD
    # tiles holding none of the px asked for are not a word's
    only = np.zeros(ink.shape, bool)
    only[:tile, :tile] = True
    some = predict.word_outputs(net, ink, "cpu", only=only, tile=tile)
    assert (some.core[tile:, :] == predict.NOT_SCORED).all() and np.allclose(some.core[:tile, :tile],
                                                                             got.core[:tile, :tile], atol=1e-6)
    assert 0.5 * (1 + np.tanh(np.float64(predict.NOT_SCORED) / 2)) == 0.0
    with pytest.raises(ValueError, match="word heads"):
        predict.word_outputs(train.build_net(train.Recipe(width=4)).eval(), ink, "cpu", tile=tile)


def test_the_context_covers_the_networks_reach():
    torch.manual_seed(1)
    for half in (False, True):
        net = train.build_net(train.Recipe(width=4, half_size=half, heads="words")).eval()
        size = 4 * predict.context(net)
        x = torch.zeros(1, 1, size, size, requires_grad=True)
        base = (torch.rand(1, 1, size, size, generator=torch.Generator().manual_seed(2)) < 0.3).float()
        net(base + x)[0, :, size // 2, size // 2].sum().backward()
        ys, xs = np.nonzero(x.grad[0, 0].abs().numpy() > 0)
        reach = max(size // 2 - ys.min(), ys.max() - size // 2, size // 2 - xs.min(), xs.max() - size // 2)
        assert 0 < reach <= predict.context(net)


# --- the detector run ------------------------------------------------------------------------------------------------

def test_a_word_heads_run_writes_every_2000_pages_word_outputs(maps, monkeypatch):  # noqa: F811
    monkeypatch.setattr(variant, "LINES", maps / "lines")
    sheets = data.load_sheets([paths.SERIES, OLD], hold_out_validation=False)
    sd = next(s for s in sheets if s.series == paths.SERIES)
    q = _quarter_of_page(sd)
    out = RD.train_run("w", paths.QUARTERS[q], 0, recipe=WORDS_TINY, device="cpu", sheets=sheets,
                       areas=(data.Quarter(q),), score=_stub_score, root=maps / "runs")
    assert out == maps / "runs" / f"w-{paths.QUARTERS[q]}-s0"
    run = json.loads((out / "run.json").read_text())
    assert run["recipe"]["heads"] == "words" and run["doubt"] is False and run["mps_peak"] is None
    got = variant.load_word_outputs(out / variant.WORDS / f"{sd.sheet}.npz")
    assert got.core.shape == (sd.height, sd.width) and got.geo.shape == (6, sd.height, sd.width)
    assert got.tag.shape == (4, sd.height, sd.width) and got.kind_p.shape == (4, sd.height, sd.width)
    assert (got.geo[:4] > 0).all()                    # distances in px, every px scored (the whole page)
    assert not (out / variant.WORDS / OLD).exists()   # only the 2000 sheets' pages
    saved = torch.load(out / "model.pt", map_location="cpu")
    assert saved["recipe"]["heads"] == "words" and "word_loss" in saved


def test_a_doubt_run_weighs_each_word_by_its_box_weight(maps, monkeypatch):  # noqa: F811
    monkeypatch.setattr(variant, "LINES", maps / "lines")
    sheets = data.load_sheets([paths.SERIES, OLD], hold_out_validation=False)
    weights = {f"{sd.series}/{sd.sheet}": {"words": doubt.words_hash(sd),
                                            "box": np.linspace(0.2, 0.9, len(sd.words)).astype(np.float32),
                                            "string": np.ones(len(sd.words), np.float32)} for sd in sheets}
    seen = []
    real = train.fit

    def fit(held, recipe, device, seed, bank=None):
        seen.extend((sd.sheet, sd.weights.tolist()) for sd in held)
        return real(held, recipe, device, seed, bank=bank)

    monkeypatch.setattr(variant.train, "fit", fit)
    out = RD.train_run("d", "nw", 0, doubt=True, recipe=WORDS_TINY, device="cpu", sheets=sheets,
                       areas=(data.Quarter(0),), score=_stub_score, root=maps / "runs", doubt_weights=weights)
    assert json.loads((out / "run.json").read_text())["doubt"] is True
    by = {sd.sheet: sd for sd in sheets}
    for sheet, w in seen:
        assert w == pytest.approx(np.linspace(0.2, 0.9, len(by[sheet].words)).tolist())
    with pytest.raises(ValueError, match="word heads"):
        variant.run("x", TINY, "nw", 0, device="cpu", sheets=sheets, doubt=True)
    with pytest.raises(ValueError, match="word heads"):
        RD.train_run("x", "nw", 0, recipe=TINY, device="cpu", sheets=sheets)


# --- the fit's areas -------------------------------------------------------------------------------------------------

def test_windows_cover_the_mask_exactly_each_wholly_inside_it():
    rng = np.random.default_rng(4)
    mask = np.zeros((97, 130), bool)
    mask[:, 40:] = True
    mask[30:50, 60:90] = False                        # an answered block
    mask[70:, :] |= rng.random((27, 130)) < 0.3       # ragged px
    cover = np.zeros(mask.shape, int)
    for y0, x0, h, w in RD.windows(mask):
        assert mask[y0:y0 + h, x0:x0 + w].all()
        cover[y0:y0 + h, x0:x0 + w] += 1
    assert np.array_equal(cover, mask.astype(int))
    assert RD.windows(np.ones((5, 7), bool)) == [(0, 0, 5, 7)] and RD.windows(np.zeros((5, 7), bool)) == []


SIDE = 240


def _sheet(rects, angles, name="s"):
    """A synthetic SheetData of SIDE x SIDE px: Terra's words `rects` read at `angles`, a block of ink in each."""
    ink = np.zeros((SIDE, SIDE), bool)
    for r in rects:
        lo, hi = np.min(r, 0).astype(int) + 3, np.max(r, 0).astype(int) - 3
        ink[lo[1]:hi[1], lo[0]:hi[0]] = True
    return data.SheetData(series=paths.SERIES, sheet=name, height=SIDE, width=SIDE,
                          ink_bits=np.packbits(ink, axis=1), words=np.round(rects).astype(np.int32),
                          kinds=np.zeros(len(rects), np.uint8), angles=np.array(angles, np.int16),
                          other=np.zeros((0, 4, 2), np.int32), points=np.zeros((0, 2), np.int32),
                          world=np.zeros((0, 0, 2)), ft_per_px=float("nan"))


def _box(x0, y0, x1, y1):
    return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], float)


def _perfect(sd, like=None):
    """What a net would give on the sheet: inside each word a core logit of 10, the exact distances to its edges
    (`data._geometry`), its direction and a tag 5 k on the first channel; outside -10, distances of 8 px, tags 0."""
    from sfzmaps.solver.wordread import shrink
    h, w = sd.height, sd.width
    wid = RD.word_ids(sd)
    g = data._geometry(shrink.prepare([sd])[0], wid, 0, 0)
    geo = np.zeros((6, h, w), np.float32)
    geo[:4] = np.where(wid > 0, np.moveaxis(g[..., :4], -1, 0), 8.0)
    a = np.deg2rad(g[..., 4])
    geo[4], geo[5] = np.where(wid > 0, np.cos(a), 1.0), np.where(wid > 0, np.sin(a), 0.0)
    tag = np.zeros((4, h, w), np.float32)
    tag[0] = 5.0 * wid
    core = np.where(wid > 0, 10.0, -10.0).astype(np.float32)
    return predict.WordOutputs(np.zeros((h, w), np.uint8), np.full((4, h, w), 0.25, np.float16), core, geo, tag)


SPREADS = words.Spreads(distance=0.5, angle=1.0, tag=0.3, reversed=0.01, cell=1.0, background_distance=8.0,
                        background_tag=1.0, background_tag_mean=(0.0, 0.0, 0.0, 0.0))
PRIOR = words.WordPrior(slope=1.0, log_prior=-60.0)


@pytest.fixture
def halves(monkeypatch):
    """Quarter 0 on the page's left half, quarter 1 on its right."""
    def page_quarters(sd):
        q = np.zeros((sd.height, sd.width), np.int8)
        q[:, sd.width // 2:] = 1
        return q
    monkeypatch.setattr(RD.reader_stage, "page_quarters", page_quarters)


def _placer(dropped=()):
    """Places every box at its centre (quarter by its half of the page, lot 7), but those whose centre is near one
    of `dropped` (off the map)."""
    def place(name, centres):
        keep = [k for k, c in enumerate(centres) if not any(np.hypot(*(c - d)) < 3 for d in dropped)]
        return {"i": np.array(keep, int), "page_x": [float(centres[k][0]) for k in keep],
                "page_y": [float(centres[k][1]) for k in keep],
                "quarter": np.array([int(centres[k][0] >= SIDE // 2) for k in keep], np.int8),
                "x": [float(centres[k][0]) for k in keep], "y": [float(centres[k][1]) for k in keep],
                "lab_lot": [7] * len(keep), "street": [False] * len(keep)}
    return place


def test_the_words_of_a_quarter_come_from_its_px_alone_and_are_placed(halves):
    rects = [_box(20, 40, 100, 70), _box(140, 40, 220, 70), _box(150, 150, 230, 180)]
    sd = _sheet(rects, [0, 0, 0])
    o = _perfect(sd)
    right, counts = RD.sheet_words("run", "s", 1, SPREADS, PRIOR, lambda n: sd, lambda r, n: o,
                                   _placer(dropped=[(190.0, 165.0)]))
    # the left word is not formed at all (its px are not the fold's: no word's); the off-map one is dropped
    assert counts == {"formed": 2, "off_map": 1, "other_quarter": 0, "kept": 1}
    assert len(right) == 1 and (right["quarter"] == 1).all() and (right["lab_lot"] == 7).all()
    assert np.allclose(right[["page_x", "page_y"]].to_numpy(float)[0], [180.0, 55.0], atol=1.5)
    assert right["quarter"].dtype == np.int8 and right["lab_lot"].dtype == np.int32
    left, counts = RD.sheet_words("run", "s", 0, SPREADS, PRIOR, lambda n: sd, lambda r, n: o, _placer())
    assert counts["formed"] == 1 and len(left) == 1
    assert np.allclose(left[["page_x", "page_y"]].to_numpy(float)[0], [60.0, 55.0], atol=1.5)
    none, counts = RD.sheet_words("run", "s", 2, SPREADS, PRIOR, lambda n: sd, lambda r, n: o, _placer())
    assert len(none) == 0 and counts["formed"] == 0
    # the words are the whole page's with the other px no word's: the window of the votes' reach only saves work
    right, _ = RD.sheet_words("run", "s", 1, SPREADS, PRIOR, lambda n: sd, lambda r, n: o, _placer())
    core, _ = RD.masked_inputs(o, np.arange(SIDE)[None, :].repeat(SIDE, 0) >= SIDE // 2)
    whole = words.core_words(core, o.geo, o.tag, data.page_ink(sd), SPREADS, PRIOR, kind_p=o.kind_p, sheet="s")
    cols = ["page_x", "page_y", *RD.wordboxes_corner_cols()]
    assert np.allclose(right[cols].to_numpy(float), whole[cols].to_numpy(float), atol=1e-3)


def test_the_reach_window_holds_every_box_the_areas_votes_can_make():
    area = np.zeros((300, 400), bool)
    area[100:120, 200:230] = True
    geo = np.full((6, 300, 400), 1.0, np.float32)
    geo[2, 110, 210] = 20.0                          # the area's longest distance: reach ceil(20 sqrt 2) = 29
    geo[0, 0, 0] = 500.0                             # a px outside the area reaches nothing of it
    assert RD.reach_window(area, geo) == (100 - 29, 200 - 29, 120 + 29, 230 + 29)
    geo[3, 101, 201] = 150.0                         # cut to the page
    assert RD.reach_window(area, geo) == (0, 0, 300, 400)
    assert RD.reach_window(np.zeros((3, 3), bool), geo[:, :3, :3]) == (0, 0, 0, 0)


def test_sheets_added_to_a_runs_boxes_keep_the_others_made_with_the_same_model(halves, tmp_path):
    sheets = {"s": _sheet([_box(140, 40, 220, 70)], [0], "s"), "t": _sheet([_box(130, 150, 200, 180)], [0], "t")}
    outs = {k: _perfect(v) for k, v in sheets.items()}
    kw = {"workers": 1, "sheet_fn": sheets.get, "outputs_fn": lambda r, n: outs[n], "placer": _placer()}
    RD.form_words(tmp_path, "ne", SPREADS, PRIOR, sheets=("s",), **kw)
    folder = RD.form_words(tmp_path, "ne", SPREADS, PRIOR, sheets=("t",), add=True, **kw)
    boxes = words.pd.read_parquet(folder / "boxes.parquet")
    assert list(boxes["sheet"]) == ["s", "t"]
    assert sorted(json.loads((folder / "summary.json").read_text())["sheets"]) == ["s", "t"]
    with pytest.raises(ValueError, match="other spreads or prior"):
        RD.form_words(tmp_path, "ne", SPREADS, words.WordPrior(1.0, -50.0), sheets=("t",), add=True, **kw)
    # without add, the folder holds the sheets made now alone
    folder = RD.form_words(tmp_path, "ne", SPREADS, PRIOR, sheets=("t",), **kw)
    assert list(words.pd.read_parquet(folder / "boxes.parquet")["sheet"]) == ["t"]


def test_the_fit_reads_only_the_training_quarters_and_never_an_answered_block(halves, monkeypatch, tmp_path):
    rects = [_box(20, 40, 100, 70), _box(140, 40, 220, 70)]
    sd = _sheet(rects, [0, 0])
    o = _perfect(sd)
    blocks = object()

    def in_area(s, area, y0, x0, h, w):                    # the answered block: rows 100-140 of the left half
        assert area is blocks
        yy, xx = np.mgrid[y0:y0 + h, x0:x0 + w]
        return (yy >= 100) & (yy < 140) & (xx < SIDE // 2)
    monkeypatch.setattr(RD.data, "in_area", in_area)
    seen = {"pages": [], "decisions": []}

    def measure(pages):
        seen["pages"].extend(pages)
        return SPREADS

    def decide(core, geo, tag, spreads, word_id, prior=None):
        seen["decisions"].append((core, word_id, prior))
        # a proposal of the area, one made only of px outside it (gain exactly 0, label 0), a split
        return words.pd.DataFrame({"kind": ["word", "word", "split"], "gain": [1.0, 0.0, 0.0],
                                   "label": [1.0, 0.0, 0.0]})

    fits = iter([words.WordPrior(1.0, -2.0), words.WordPrior(0.5, -1.0)])
    monkeypatch.setattr(words, "measure_spreads", measure)
    monkeypatch.setattr(words, "prior_decisions", decide)
    monkeypatch.setattr(words, "fit_word_prior", lambda g, y: next(fits))
    got = RD.fit(tmp_path / "fit", [(tmp_path / "run", (0,))], sheets=("s",), workers=1, blocks=blocks,
                 sheet_fn=lambda n: sd, outputs_fn=lambda r, n: o)
    assert got["prior"] == words.WordPrior(0.5, -1.0)
    allowed = np.zeros((SIDE, SIDE), bool)
    allowed[:, :SIDE // 2] = True
    allowed[100:140, :SIDE // 2] = False
    # the spreads' pages: windows of quarter 0 outside the block, together all of it and nothing else
    assert sum(p["word_id"].size for p in seen["pages"]) == allowed.sum()
    assert sum(int((p["word_id"] == 1).sum()) for p in seen["pages"]) == int((RD.word_ids(sd)[allowed] == 1).sum())
    assert not any((p["word_id"] == 2).any() for p in seen["pages"])        # the word in the held-out quarter
    for p in seen["pages"]:
        assert p["geo"].shape[1:] == p["word_id"].shape == p["tag"].shape[1:]
    # the decisions: on the window of the area's reach (all of the area in it), the px outside the area no word's
    # and holding no Terra word; first no prior, then the first fit
    assert [d[2] for d in seen["decisions"]] == [None, words.WordPrior(1.0, -2.0)]
    y0, x0, y1, x1 = RD.reach_window(allowed, o.geo)
    assert (y0, x0) == (0, 0) and x1 < SIDE and not allowed[:, x1:].any()
    win = (slice(y0, y1), slice(x0, x1))
    for core, wid, _ in seen["decisions"]:
        assert core.shape == wid.shape == (y1 - y0, x1 - x0)
        a = allowed[win]
        assert (core[~a] == predict.NOT_SCORED).all() and np.array_equal(core[a], o.core[win][a])
        assert not wid[~a].any() and np.array_equal(wid[a], RD.word_ids(sd)[win][a])
    for f in ("spreads.json", "prior.json", "decisions.parquet"):
        assert (tmp_path / "fit" / f).is_file()
    # the proposal made only of px outside the area is no decision of the fit; the rest are kept, each round
    kept = words.pd.read_parquet(tmp_path / "fit" / "decisions.parquet")
    assert sorted(zip(kept["round"], kept["kind"], kept["gain"])) == [(1, "split", 0.0), (1, "word", 1.0),
                                                                     (2, "split", 0.0), (2, "word", 1.0)]
    spreads, prior = RD.load_fit(tmp_path / "fit")
    assert spreads == SPREADS and prior == words.WordPrior(0.5, -1.0)


def test_a_run_folder_holds_the_detector_its_outputs_and_its_placed_words(maps, monkeypatch):  # noqa: F811
    monkeypatch.setattr(variant, "LINES", maps / "lines")
    sheets = data.load_sheets([paths.SERIES, OLD], hold_out_validation=False)
    sd = next(s for s in sheets if s.series == paths.SERIES)
    q = _quarter_of_page(sd)
    run = RD.train_run("f", paths.QUARTERS[q], 0, recipe=WORDS_TINY, device="cpu", sheets=sheets,
                       areas=(data.Quarter(q),), score=_stub_score, root=maps / "runs")
    saved = variant.load_word_outputs(run / variant.WORDS / f"{sd.sheet}.npz")
    assert saved.core.shape == (sd.height, sd.width)
    # the words from what the heads give: here a perfect answer drawn from Terra's words (the 2-step net's is noise)
    perfect = _perfect(sd)

    def place(name, centres):
        keep = np.arange(len(centres))
        return {"i": keep, "page_x": centres[:, 0].tolist(), "page_y": centres[:, 1].tolist(),
                "quarter": np.full(len(keep), q, np.int8), "x": centres[:, 0].tolist(), "y": centres[:, 1].tolist(),
                "lab_lot": [3] * len(keep), "street": [False] * len(keep)}
    folder = RD.form_words(run, paths.QUARTERS[q], SPREADS, PRIOR, sheets=(sd.sheet,), workers=1,
                           sheet_fn=lambda n: sd, outputs_fn=lambda r, n: perfect, placer=place)
    assert folder == run / "boxes"
    boxes = words.pd.read_parquet(folder / "boxes.parquet")
    from sfzmaps.solver import wordboxes
    assert list(boxes.columns) == [*wordboxes.COLUMNS, *words.EXTRA_COLUMNS]
    assert (boxes["quarter"] == q).all() and (boxes["sheet"] == sd.sheet).all() and (boxes["lab_lot"] == 3).all()
    summary = json.loads((folder / "summary.json").read_text())
    assert summary["sheets"][sd.sheet]["kept"] == len(boxes) and summary["prior"]["log_prior"] == -60.0
    # every one of Terra's words in the fold's quarter is a box, centred on it (the synthetic page lies in two
    # quarters: word A's px are the other's, so this fold never forms it); the boxes are in page px, the window of
    # the votes' reach moved back
    qmap = reader_stage.page_quarters(sd)
    centres = sd.words.astype(float).mean(1)
    mine = np.array([qmap[int(c[1]), int(c[0])] == q for c in centres])
    assert mine.sum() == len(boxes) == len(sd.words) - 1 and RD.reach_window(qmap == q, perfect.geo)[1] > 0
    got = boxes[["page_x", "page_y"]].to_numpy(float)
    for c in centres[mine]:
        assert np.min(np.hypot(*(got - c).T)) < 2
    assert np.min(np.hypot(*(got - centres[~mine][0]).T)) > 100


# --- the review page of a word-heads run ----------------------------------------------------------------------------

def test_the_page_of_one_quarter_reads_its_boxes_px_as_components_and_fades_the_rest(tmp_path):
    from sfzmaps.solver.wordread import measures, review

    def rect(x0, y0, x1, y1):
        return [x0, y0, x1, y0, x1, y1, x0, y1]
    h, w = 200, 300
    ink = np.zeros((h, w), bool)
    ink[40:60, 30:120] = ink[40:60, 130:220] = ink[140:160, 30:120] = True
    chance = np.where(ink, 230, 0).astype(np.uint8)
    # one box over two touching words: the merge the word heads are to undo
    boxes = words.pd.DataFrame([{"sheet": "s", "quarter": 1, "axis": 0.0, "length": 200, "height": 30,
                                 "mean_chance": 0.9, "ink_px": 100, "p_zoning": 0.9, "p_block": 0.0, "p_street": 0.0,
                                 "p_other": 0.1, **dict(zip(measures.G_COLS, rect(25, 35, 225, 65)))}])
    terra = words.pd.DataFrame([{"sheet": "s", "text": t, "kind": "zoning code", "quarter": 1,
                                 **dict(zip(measures.C_COLS, r))}
                                for t, r in (("RH-1", rect(28, 38, 122, 62)), ("RH-2", rect(128, 38, 222, 62)))])
    reads = {0: {"closed": {"RH-1": 0.9, "RH-2": 0.1}, "top": "RH-1", "text": "RH-1 RH-2"}}
    visible = np.zeros((h, w), bool)
    visible[:100] = True
    d = review.page_data("s", boxes, reads, terra, chance, None, ink=ink, levels=dict.fromkeys(paths.QUARTERS, 0.5),
                         on_map=np.ones(2, bool), forms={"RH-1": "RH-1", "RH-2": "RH-2"},
                         comps=review.box_components(boxes, chance, ink), visible=visible,
                         reason_text=RD.WORD_REASONS, notes=[(150, 150, "SE: " + RD.NOT_TRAINED)])
    assert [m["reason"] for m in d["missed"]] == ["merged_with_word", "merged_with_word"]
    out = review.render(d, tmp_path / "page.html")
    img, text = review.page_image(out), out.read_text()
    assert tuple(img[150, 50]) == review.FADED["faded_ink"] and tuple(img[150, 10]) == review.FADED["faded_paper"]
    assert tuple(img[10, 10]) == review.COLOURS["paper"]
    assert RD.WORD_REASONS["merged_with_word"] in text and RD.NOT_TRAINED in text
    # left out, the page is the old one: the chance's components at the level, six colours, no notes
    plain = review.page_data("s", boxes, reads, terra, chance, None, ink=ink,
                             levels=dict.fromkeys(paths.QUARTERS, 0.5), on_map=np.ones(2, bool),
                             forms={"RH-1": "RH-1", "RH-2": "RH-2"})
    assert not {"faded", "reason_text", "notes"} & set(plain)
    assert int(plain["pixels"].max()) < len(review.COLOURS)
