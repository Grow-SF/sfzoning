"""Map-like synthetic scenes (`wordread.scenes`, item 8): DataSF geometry drawn as the maps draw it, real words pasted
where the maps print labels, exact masks for every head; and `train.Recipe.scenes`, which draws them through the bank
handed to the training (default 0: the batches of today, byte for byte)."""
import dataclasses

import cv2
import numpy as np
import pytest
import shapely

from sfzmaps.solver import grid, paths
from sfzmaps.solver.textnet import data, train
from sfzmaps.solver.wordread import shrink

# --- the recipe ---------------------------------------------------------------------------------------------------

H, W = 240, 320
WORDS = [((100, 50, 200, 80), 0), ((230, 40, 260, 140), 90)]


def _page() -> np.ndarray:
    return np.random.default_rng(7).integers(0, 256, (H, W)).astype(np.uint8)


def _plain_sheet() -> data.SheetData:
    gray = _page()
    rects = [data.word_rect(box, angle) for box, angle in WORDS]
    sd = data.SheetData(
        series=paths.SERIES, sheet="s", height=H, width=W, ink_bits=np.packbits(gray < grid.INK_BELOW, axis=1),
        words=np.round(np.array(rects, float).reshape(-1, 4, 2)).astype(np.int32),
        kinds=np.zeros(len(WORDS), np.uint8), angles=np.array([data.reading_angle(r) for r in rects], np.int16),
        other=np.zeros((0, 4, 2), np.int32), points=np.zeros((0, 2), np.int32), world=np.zeros((0, 0, 2)),
        ft_per_px=float("nan"))
    return shrink.prepare([sd])[0]


def _same_batches(a: list[dict], b: list[dict]) -> bool:
    return len(a) == len(b) and all(
        x.keys() == y.keys() and all(np.array_equal(x[k], y[k]) and x[k].dtype == y[k].dtype for k in x)
        for x, y in zip(a, b))


class _FakeScenes:
    """A bank whose scenes say what they were asked for (a crop of ones, the scale in `known`'s place)."""

    def __init__(self):
        self.asked = []

    def scene(self, rng, size, scale, turns):
        self.asked.append((size, scale, turns))
        rng.random()                                  # a scene draws from the batch's generator
        return {"ink": np.ones((size, size), bool), "text": np.zeros((size, size), np.uint8),
                "kind": np.full((size, size), data.NO_KIND, np.uint8), "angle": np.zeros((size, size), np.int16),
                "known": np.ones((size, size), bool)}


def test_the_default_recipe_has_no_scenes_and_its_name_is_unchanged():
    r = train.Recipe()
    assert r.scenes == 0.0 and r.name() == "default"
    assert train.Recipe(scenes=0.25).name() == "scenes=0.25"
    for bad in (-0.1, 1.5):
        with pytest.raises(ValueError, match="scenes must be between 0 and 1"):
            train.Recipe(scenes=bad)
    with pytest.raises(ValueError, match="scenes must be 0 with input grey"):
        train.Recipe(input="grey", scenes=0.5)
    # the word heads take scenes (they carry every word target), unlike composites
    assert train.Recipe(heads="words", scenes=0.5).scenes == 0.5


@pytest.mark.parametrize("heads", ["all", "words"])
def test_scenes_zero_leaves_the_batches_byte_identical(heads):
    sheets = [_plain_sheet()]
    pools = train.centre_pools(sheets)
    recipe = train.Recipe(batch=8, crop=48, heads=heads)
    rng_a, rng_b = np.random.default_rng(3), np.random.default_rng(3)
    a = [train.draw(sheets, pools, recipe, rng_a) for _ in range(3)]
    fake = _FakeScenes()
    b = [train.draw(sheets, pools, dataclasses.replace(recipe, scenes=0.0), rng_b, fake) for _ in range(3)]
    assert all(_same_batches(x, y) for x, y in zip(a, b))
    assert fake.asked == [] and rng_a.random() == rng_b.random()      # not one draw more


def test_scenes_come_from_the_bank_at_the_crops_scale_and_turns():
    sheets = [_plain_sheet()]
    pools = train.centre_pools(sheets)
    fake = _FakeScenes()
    got = train.draw(sheets, pools, train.Recipe(batch=6, crop=40, scenes=1.0), np.random.default_rng(0), fake)
    assert len(fake.asked) == 6 and all(c["ink"].all() for c in got)
    # the scale and turns are drawn exactly as a real crop's (and a composite's): the same generator, in the same order

    class FakeComposites:
        asked = []

        def crop(self, rng, sd, y, x, size, scale, turns):
            self.asked.append((size, scale, turns))
            return _FakeScenes().scene(rng, size, scale, turns)

    composites = FakeComposites()
    train.draw(sheets, pools, train.Recipe(batch=6, crop=40, composites=1.0), np.random.default_rng(0), composites)
    assert composites.asked == fake.asked
    with pytest.raises(ValueError, match="scenes .* from a bank"):
        train.draw(sheets, pools, train.Recipe(scenes=0.5), np.random.default_rng(0))
    with pytest.raises(ValueError, match="scenes .* from a bank"):
        train.fit(sheets, train.Recipe(iterations=1, batch=1, crop=32, scenes=0.5), device="cpu")


# --- a synthetic city on one page ---------------------------------------------------------------------------------
#
# Blocks of BW x BH px on a grid (streets STREET px wide between them), each split into 4 x 2 lots, its outline printed
# 2 px wide; its number printed at its centre; a code in some blocks; street names along the streets (the vertical ones
# read bottom to top).  The page is placed north up at 4 ft a px with the quarter split at page px (CX, CY), so a
# quarter held out is a quadrant of the page.

FT = 4.0
BW, BH, STREET, MARGIN = 64, 54, 20, 16
NX, NY = 6, 5
PW, PH = MARGIN + (BW + STREET) * NX, MARGIN + (BH + STREET) * NY
CX, CY = (MARGIN + (BW + STREET) * 3) - STREET // 2, (MARGIN + (BH + STREET) * 2) + BH // 2
ACROSS = ("ALPHA", "BRAVO", "CHARLIE", "DELTA")
UPWARD = ("ECHO", "FOXTROT", "GOLF", "HOTEL", "INDIA")
SCENE_SERIES = paths.SERIES
SCENE_SHEET = "2000-digital-zoning-p01"


def _world(xy) -> np.ndarray:
    xy = np.asarray(xy, float)
    return np.stack([grid.QUARTER_SPLIT[0] + FT * (xy[..., 0] - CX), grid.QUARTER_SPLIT[1] - FT * (xy[..., 1] - CY)], -1)


def _block_px(i, j) -> tuple[int, int, int, int]:
    x0, y0 = MARGIN + (BW + STREET) * i, MARGIN + (BH + STREET) * j
    return x0, y0, x0 + BW - 1, y0 + BH - 1


def _glyph_mask(text, scale=0.35) -> np.ndarray:
    (w, h), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
    pg = np.full((h + base + 4, w + 4), 255, np.uint8)
    cv2.putText(pg, text, (2, h + 2), cv2.FONT_HERSHEY_SIMPLEX, scale, 0, 1)
    m = pg < grid.INK_BELOW
    rows, cols = np.nonzero(m)
    return m[rows.min():rows.max() + 1, cols.min():cols.max() + 1]


def _city():
    """The page (bool ink), the line px, the words (rect corners, kind, text) and the geometry frames."""
    import geopandas as gpd
    import shapely
    ink = np.zeros((PH, PW), bool)
    lines = np.zeros((PH, PW), np.uint8)
    words = []

    def put(text, kind, cx, cy, turn=0, angle=0):
        g = np.rot90(_glyph_mask(text), turn)
        h, w = g.shape
        x0, y0 = int(cx - w // 2), int(cy - h // 2)
        ink[y0:y0 + h, x0:x0 + w] |= g
        box = (x0 - 1, y0 - 1, x0 + w, y0 + h)
        words.append((data.word_rect(box, 90 if turn else angle), kind, text))

    lots, blocks = [], []
    for j in range(NY):
        for i in range(NX):
            x0, y0, x1, y1 = _block_px(i, j)
            cv2.rectangle(lines, (x0, y0), (x1, y1), 1, 2)
            number = f"{100 + 10 * j + i}"
            blocks.append(number)
            put(number, "block number", (x0 + x1) // 2, (y0 + y1) // 2)
            if (i + j) % 2 == 0:
                put("RH-1" if j % 2 else "RM-2", "zoning code", (x0 + x1) // 2, y0 + 12)
            xs, ys = np.linspace(x0, x1, 5), np.linspace(y0, y1, 3)
            for a in range(4):
                for b in range(2):
                    lots.append({"block_num": f"0{number}",
                                 "geometry": shapely.Polygon(_world([(xs[a], ys[b]), (xs[a + 1], ys[b]),
                                                                     (xs[a + 1], ys[b + 1]), (xs[a], ys[b + 1])]))})
    streets = []
    for j, name in enumerate(ACROSS, 1):
        y = MARGIN + (BH + STREET) * j - STREET // 2
        streets.append({"street": name, "type": "ST", "geometry": shapely.LineString(_world([(0, y), (PW - 1, y)]))})
        for k, x in enumerate((MARGIN + 30, MARGIN + 30 + 2 * (BW + STREET), MARGIN + 30 + 4 * (BW + STREET))):
            put(name, "street name", x, y, angle=(0, 2, 358)[(j + k) % 3])
    for i, name in enumerate(UPWARD, 1):
        x = MARGIN + (BW + STREET) * i - STREET // 2
        streets.append({"street": name, "type": "AVE",
                        "geometry": shapely.LineString(_world([(x, 0), (x, PH - 1)]))})
        for y in (MARGIN + 27, MARGIN + 27 + 2 * (BH + STREET)):
            put(name, "street name", x, y, turn=1)
    # a code printed only in the north-east quadrant (held out below), and one touching a block's outline
    x0, y0, x1, y1 = _block_px(4, 0)
    put("ZZ-9", "zoning code", (x0 + x1) // 2, y1 - 12)
    x0, y0, x1, y1 = _block_px(0, 3)
    put("RC-3", "zoning code", x0 + 14, y1 - 4)
    ink |= lines > 0
    frames = (gpd.GeoDataFrame(lots, geometry="geometry", crs="EPSG:2227"),
              gpd.GeoDataFrame(streets, geometry="geometry", crs="EPSG:2227"))
    return ink, lines > 0, words, frames


@pytest.fixture(scope="module")
def city():
    return make_city()


def make_city() -> dict:
    """The city as a sheet (placed, with its line px and cores), its geometry and its words' texts."""
    from sfzmaps.solver.wordread import scenes
    ink, lines, words, (lots, streets) = _city()
    step = data.PLACEMENT_STEP
    gx, gy = np.meshgrid(np.arange(0, PW - 1 + step, step, dtype=float), np.arange(0, PH - 1 + step, step, dtype=float))
    rects = np.round(np.array([w[0] for w in words], float)).astype(np.int32)
    sd = data.SheetData(
        series=SCENE_SERIES, sheet=SCENE_SHEET, height=PH, width=PW, ink_bits=np.packbits(ink, axis=1), words=rects,
        kinds=np.array([data.KINDS[w[1]] for w in words], np.uint8),
        angles=np.array([data.reading_angle(r) for r in rects], np.int16), other=np.zeros((0, 4, 2), np.int32),
        points=np.zeros((0, 2), np.int32), world=_world(np.stack([gx, gy], -1)), ft_per_px=FT,
        line_bits=np.packbits(cv2.dilate(lines.view(np.uint8), np.ones((3, 3), np.uint8)) > 0, axis=1))
    sd = shrink.prepare([sd])[0]
    geo = scenes.Geometry.from_frames(lots, streets)
    texts = {(SCENE_SERIES, SCENE_SHEET): [scenes.norm(w[2]) for w in words]}
    return {"sheet": sd, "geometry": geo, "texts": texts, "words": words}


def _held(city):
    """The north-east quarter and the answered blocks: block 131 (in the south-west)."""
    x0, y0, x1, y1 = _block_px(1, 3)
    (c0, r0), (c1, r1) = np.floor(grid.to_grid(_world([(x0, y0), (x1, y1)]))).astype(int)
    mask = np.ones((r1 - r0 + 1, c1 - c0 + 1), bool)
    return (data.Quarter(1), data.GridArea(mask, c0, r0))


def _bank(city, areas=(), seed=0):
    from sfzmaps.solver.wordread import scenes
    return scenes.SceneBank.build([city["sheet"]], areas, seed, geometry=city["geometry"], texts=city["texts"],
                                  footprints={})


@pytest.fixture(scope="module")
def bank(city):
    return _bank(city)


@pytest.fixture(scope="module")
def held_bank(city):
    return _bank(city, _held(city))


def test_a_pasteable_word_is_its_own_ink_and_one_joined_to_a_line_is_not(city):
    from sfzmaps.solver.wordread import scenes
    sd = city["sheet"]
    found = {j: (x0, y0, own) for j, x0, y0, own in scenes.pasteable(sd)}
    texts = [w[2] for w in city["words"]]
    joined = texts.index("RC-3")                 # its glyphs touch the block's outline: a piece runs outside its box
    assert joined not in found
    ink = data.page_ink(sd)
    for j, (x0, y0, own) in found.items():
        h, w = own.shape
        # its own ink is all the ink in its rectangle, and every piece of it lies wholly inside
        rect = np.zeros((h, w), np.uint8)
        cv2.fillPoly(rect, [(sd.words[j] - [x0, y0]).reshape(-1, 1, 2)], 1)
        assert np.array_equal(own, ink[y0:y0 + h, x0:x0 + w] & (rect > 0))
        grown = np.zeros((PH, PW), bool)
        grown[y0:y0 + h, x0:x0 + w] = own
        n, label = cv2.connectedComponents(ink.view(np.uint8), connectivity=8)
        pieces = np.unique(label[grown])
        assert np.isin(label, pieces).sum() == own.sum()
    assert len(found) == len(texts) - 1


def test_a_line_mixture_finds_the_drawn_share_and_spread():
    from sfzmaps.solver.wordread import scenes
    d2 = np.arange(0, 400)
    d = np.sqrt(d2.astype(float))
    n_b = np.full(len(d), 1000.0)                              # px at each distance, alike
    g = np.exp(-d * d / (2 * 1.5 ** 2))
    near, far = g / g.sum(), np.full(len(d), 1.0 / len(d))
    # two kinds of line under one placement (sigma 1.5 px): 70% of the first's skeleton its own, 20% of the second's
    one, two = 5000 * (0.7 * near + 0.3 * far), 3000 * (0.2 * near + 0.8 * far)
    fit = scenes.fit_near([(d, one, n_b), (d, two, n_b), (d, np.zeros(len(d)), n_b)])
    assert fit["a"][0] == pytest.approx(0.7, abs=1e-4) and fit["a"][1] == pytest.approx(0.2, abs=1e-4)
    assert fit["sigma"] == pytest.approx(1.5, abs=1e-3) and fit["a"][2] == 0.0


def test_thickness_shares_explain_a_mixture_of_drawn_thicknesses():
    from sfzmaps.solver.wordread import scenes
    k2, s2 = scenes.thickness_profile(2)
    k5, s5 = scenes.thickness_profile(5)
    keys = np.union1d(k2, k5)
    w = 0.7 * np.interp(keys, k2, s2, left=0, right=0) * np.isin(keys, k2) \
        + 0.3 * np.interp(keys, k5, s5, left=0, right=0) * np.isin(keys, k5)
    got = scenes.thickness_shares(keys, w)
    shares = dict(zip(got["thickness"], got["share"]))
    assert shares[2] == pytest.approx(0.7, abs=0.02) and shares[5] == pytest.approx(0.3, abs=0.02)
    assert got["unexplained"] == 0.0


def test_street_offsets_separate_misreads_far_away():
    from sfzmaps.solver.wordread import scenes
    rng = np.random.default_rng(0)
    d = np.r_[np.abs(rng.normal(0, 10, 800)), rng.exponential(3000, 200)]
    fit = scenes.fit_far(d)
    assert fit["a"] == pytest.approx(0.8, abs=0.03) and fit["sigma"] == pytest.approx(10, rel=0.1)
    assert fit["r"][:800].mean() > 0.95 and fit["r"][800:].mean() < 0.1


def test_the_bank_measures_the_city_it_was_given(bank):
    m = bank.measured[SCENE_SERIES]
    # the outlines are printed; no lot line or centreline is
    assert m.drawn[0] > 0.9 and m.drawn[1] < 0.05 and m.drawn[2] < 0.05
    t, q = m.thickness[0]
    assert t[np.argmax(q)] in (2, 3)
    # every block number lies in its block, near its centre; codes sit in lots
    assert m.block_rate == pytest.approx(1.0) and np.hypot(*m.block_offsets.T).max() < 2 * FT
    assert len(m.code_offsets) == sum(1 for w in bank.sheets[0].kinds if w == data.KINDS["zoning code"])
    assert m.street_rate > 0 and np.isfinite(m.angle_bandwidth)


def _cores_disjoint(rects, shape) -> bool:
    total = np.zeros(shape, np.int32)
    for r in rects:
        one = np.zeros(shape, np.uint8)
        cv2.fillPoly(one, [r.reshape(-1, 1, 2)], 1)
        total += one
    return int(total.max(initial=0)) <= 1


@pytest.mark.parametrize("scale,turns", [(1.0, 0), (0.7, 1), (1.4, 3)])
def test_a_scenes_masks_are_exact(bank, scale, turns):
    rng = np.random.default_rng([5, turns])
    size = 96
    real = data.crop_targets(bank.sheets[0], 100, 100, size, scale, turns, words=True)
    labelled = 0
    for _ in range(12):
        t = bank.crop(rng, size, scale, turns)
        assert t.keys() == real.keys() and all(t[k].dtype == real[k].dtype and t[k].shape == real[k].shape for k in t)
        text = t["text"] > 0
        # text is only ever the pasted words' own ink
        assert not (text & ~t["ink"]).any()
        y0, x0, src = bank.last["window"]
        own = bank.last["own"]
        if scale == 1.0 and turns == 0:
            assert np.array_equal(text, own)
        rects = [p["rect"] for p in bank.last["labels"]]
        assert _cores_disjoint([r + 200 for r in rects], (src + 400, src + 400))
        # a core px is its word's, the paper inside a word's rectangle is unknown, and a line px in it known not text
        assert np.array_equal(t["core"] > 0, t["tag_id"] > 0)
        assert (t["geo_known"][t["core"] > 0] == 1).all()
        assert not t["known"][(t["core"] > 0) & ~t["ink"]].any()
        assert t["known"][~(t["core"] > 0)].all()
        assert (t["kind"][~text] == data.NO_KIND).all() and (t["angle"][~text] == 0).all()
        labelled += len(rects)
    assert labelled > 0


def test_the_same_seed_gives_the_same_scene(city):
    a, b = _bank(city, seed=4), _bank(city, seed=4)
    for k in range(3):
        x, y = a.crop(np.random.default_rng([4, k]), 80, 1.2, k), b.crop(np.random.default_rng([4, k]), 80, 1.2, k)
        assert x.keys() == y.keys() and all(np.array_equal(x[n], y[n]) for n in x)
    assert a.tally == b.tally


def test_only_the_printed_map_is_measured_pasted_and_centred_on(city, bank):
    from sfzmaps.solver.wordread import scenes
    # the sheet's printed map: its left part (the rest, say, a legend)
    edge = MARGIN + (BW + STREET) * 4
    footprint = shapely.Polygon(_world([(0, 0), (edge, 0), (edge, PH), (0, PH)]))
    part = scenes.SceneBank.build([city["sheet"]], (), 0, geometry=city["geometry"], texts=city["texts"],
                                  footprints={(SCENE_SERIES, SCENE_SHEET): footprint})
    w = part.words[SCENE_SERIES]
    assert len(w) and (w.rect.mean(axis=1)[:, 0] < edge).all()
    assert (bank.words[SCENE_SERIES].rect.mean(axis=1)[:, 0] >= edge).any()
    for pool in part.centres:
        assert (pool[0][:, 1] <= edge).all()
    mine, every = part.measured[SCENE_SERIES].summary, bank.measured[SCENE_SERIES].summary
    assert mine["lines"]["block"]["length_px"] < every["lines"]["block"]["length_px"]
    assert 0 < mine["codes"]["words"] < every["codes"]["words"] and mine["codes"]["lots"] < every["codes"]["lots"]


def test_a_held_out_area_never_appears(city, bank, held_bank):
    from sfzmaps.solver.wordread import scenes
    areas = _held(city)
    shapes = scenes.held_shapes(areas)
    geo = city["geometry"]
    sd = held_bank.sheets[0]
    # no word of a held-out area is in the bank: the code printed only in the north-east is not, without areas it is
    assert "ZZ-9" in bank.words[SCENE_SERIES].text and "ZZ-9" not in held_bank.words[SCENE_SERIES].text
    w = held_bank.words[SCENE_SERIES]
    assert not data._touch_held(sd, w.index).any()
    # nor measured: the codes measured are those outside
    outside = [j for j in np.flatnonzero(sd.kinds == data.KINDS["zoning code"])
               if not data._touch_held(sd, np.array([j]))[0]]
    assert len(held_bank.measured[SCENE_SERIES].code_offsets) == len(outside) < len(bank.measured[SCENE_SERIES].code_offsets)
    # and no scene draws a held-out line or prints a held-out block's, lot's or street's label
    seg = shapely.linestrings(geo.seg)
    seen_held = 0
    for name, b in (("held", held_bank), ("all", bank)):
        rng = np.random.default_rng(11)
        for _ in range(40):
            b.crop(rng, 128, float(rng.uniform(0.6, 1.6)), int(rng.integers(4)))
            idx = b.last["segments"]["index"]
            meets = np.zeros(len(idx), bool)
            for s in shapes:
                meets |= shapely.intersects(seg[idx], s)
            if name == "held":
                assert not meets.any()
                for p in b.last["labels"]:
                    assert b.words[SCENE_SERIES].text[p["word"]] != "ZZ-9"
                    item = {"block": geo.block_poly, "code": geo.lot_poly, "street": geo.lines}[p["what"]][p["item"]]
                    assert not any(shapely.intersects(item, s) for s in shapes)
            else:
                seen_held += int(meets.sum())
    assert seen_held > 0          # the test can fail: without the areas, held-out lines are drawn


def test_scenes_train_with_the_word_heads(bank):
    pools = train.centre_pools(bank.sheets)
    crops = train.draw(bank.sheets, pools, train.Recipe(batch=4, crop=64, heads="words", scenes=0.5),
                       np.random.default_rng(2), bank)
    x, *_ = train.batch_tensors(crops)
    core, geo, tag, known, geo_known, groups = train.word_tensors(crops)
    assert x.shape == (4, 1, 64, 64) and geo.shape == (4, 5, 64, 64) and bank.tally["scenes"] > 0


@pytest.mark.skipif(not (paths.LOTS_DIR / "table.parquet").is_file(), reason="needs pipeline/data (the lots table)")
def test_sf_geometry_has_every_kind_of_line():
    from sfzmaps.solver.wordread import scenes
    geo = scenes.Geometry.load()
    counts = np.bincount(geo.seg_kind, minlength=3)
    assert (counts > 0).all() and len(geo.block_number) > 5000 and len(geo.lines) > 17000
    assert "MISSION" in geo.form_lines and len(geo.block_of_form) > 5000
