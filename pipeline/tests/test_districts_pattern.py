import json

import cv2
import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import box

from sfzmaps import config, llm
from sfzmaps.districts import pattern
from synthetic_districts import FAKE_LEGEND, pattern_fill, swatches, write_legend_page, write_sheet


@pytest.fixture
def work(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "WORK_DIR", tmp_path / "work")
    monkeypatch.setattr(config, "OUT_DIR", tmp_path / "out")
    monkeypatch.setattr(llm, "CACHE_DIR", tmp_path / "cache")
    llm.CallCounter.reset()
    return tmp_path


class FakeClient:
    """Reads each mosaic tile back by its layout box and answers by the tile's inner tone:
    dark -> C, else 2R.  Records the tile counts it saw."""

    def __init__(self, conf=0.8):
        self.calls = 0
        self.tiles_seen = []
        self.conf = conf
        self.sent = []

    def call(self, png, prompt, schema):
        self.calls += 1
        self.sent.append((png, prompt))
        img = cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_COLOR)
        n = int(prompt.split("numbered 1 to ")[1].split()[0].rstrip(".,"))
        self.tiles_seen.append(n)
        boxes = pattern.tile_boxes(n, self.legend_h, pattern.TILE_PX)
        items = []
        for i, (x0, y0, x1, y1) in enumerate(boxes):
            t = img[y0:y1, x0:x1]
            c = t[t.shape[0] // 2 - 4:t.shape[0] // 2 + 4, t.shape[1] // 2 - 4:t.shape[1] // 2 + 4]
            items.append({"index": i + 1, "code": "C" if c.mean() < 100 else "2R",
                          "confidence": self.conf})
        return schema.model_validate({"items": items[::-1]})   # out of order on purpose


def _fake(sw):
    c = FakeClient()
    c.legend_h = pattern.legend_height(sw, pattern.TILE_PX)
    return c


def _tile(v, px=pattern.TILE_PX):
    return np.full((px, px, 3), v, np.uint8)


def _items(rows):
    """rows: (blklot, block, gray value of the crop, tone stat)."""
    return pd.DataFrame({"blklot": [r[0] for r in rows], "block": [r[1] for r in rows],
                         "crop": [_tile(r[2]) for r in rows],
                         "stats": [np.array([r[3] / 255, 0.0]) for r in rows]})


def test_mosaic_layout_and_index_map():
    sw = swatches()
    crops = [_tile(0 if i % 3 == 0 else 255) for i in range(20)]
    crops[1] = np.zeros((30, 90, 3), np.uint8)            # non-square: padded, not squashed
    img, numbers = pattern.build_mosaic(crops, sw)
    assert numbers == list(range(1, 21))
    t = pattern.TILE_PX
    assert img.shape[1] <= pattern.COLS * (t + pattern.GAP) + pattern.GAP
    boxes = pattern.tile_boxes(20, pattern.legend_height(sw, t), t)
    assert len(boxes) == 20 and all(b[3] <= img.shape[0] for b in boxes)
    for i, (x0, y0, x1, y1) in enumerate(boxes):
        assert (x1 - x0, y1 - y0) == (t, t)
        mid = img[y0 + t // 2, x0 + t // 2]
        if i == 1:
            assert mid.max() < 50 and img[y0 + 2, x0 + t // 2].min() > 150   # padded top
        else:
            assert (mid.max() < 50) == (i % 3 == 0)
    assert img[: pattern.legend_height(sw, t)].min() == 0      # the solid swatch is in the strip
    with pytest.raises(ValueError):
        pattern.build_mosaic([_tile(0)] * (pattern.MAX_TILES + 1), sw)


def test_uniform_block_shortcut_one_call(work):
    sw = swatches()
    df = _items([(f"0001{i:03d}", "0001", 0, 30 + i) for i in range(30)])
    fake = _fake(sw)
    out = pattern.classify(df, sw, client=fake, block_crops={"0001": _tile(0)})
    assert fake.calls == 1 and fake.tiles_seen == [1]
    assert (out["code"] == "C").all() and len(out) == 30
    assert np.allclose(out["confidence"], 0.8 * 0.9)


def test_mixed_block_per_parcel(work):
    sw = swatches()
    rows = [(f"0002{i:03d}", "0002", 255, 250) for i in range(5)] + \
           [(f"0002{i:03d}", "0002", 0, 30) for i in range(5, 8)]
    fake = _fake(sw)
    out = pattern.classify(_items(rows), sw, client=fake)
    assert fake.calls == 1 and fake.tiles_seen == [8]
    got = dict(zip(out["blklot"], out["code"]))
    assert [got[f"0002{i:03d}"] for i in range(8)] == ["2R"] * 5 + ["C"] * 3
    assert np.allclose(out["confidence"], 0.8)


def test_results_mapped_back_by_index(work):
    sw = swatches()
    rows = []
    for b in range(60):          # 60 mixed blocks of 2 -> 120 tiles -> 3 mosaics
        rows += [(f"{b:04d}001", f"{b:04d}", b % 40, 20), (f"{b:04d}002", f"{b:04d}", 255 - b % 40, 250)]
    fake = _fake(sw)
    out = pattern.classify(_items(rows), sw, client=fake)
    assert fake.calls == 3 and sorted(fake.tiles_seen) == [24, 48, 48]
    got = dict(zip(out["blklot"], out["code"]))
    assert all(got[f"{b:04d}001"] == "C" and got[f"{b:04d}002"] == "2R" for b in range(60))


def test_unknown_code_from_model_is_dropped(work):
    sw = swatches()

    class Bad:
        def call(self, png, prompt, schema):
            return schema.model_validate({"items": [{"index": 1, "code": "ZZ", "confidence": 0.9}]})
    out = pattern.classify(_items([("0003001", "0003", 0, 20)]), sw, client=Bad())
    assert out["code"].isna().all() and (out["confidence"] == 0).all()


class LegendFake:
    def __init__(self, boxes, mb, jitter=4):
        self.boxes, self.mb, self.jitter, self.calls = boxes, mb, jitter, 0

    def call(self, png, prompt, schema):
        self.calls += 1
        img = cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_COLOR)
        s = img.shape[1] / (self.mb["x1"] - self.mb["x0"])
        sw = []
        for code, (x0, y0, x1, y1) in self.boxes.items():
            j = self.jitter
            sw.append({"code": code, "label": code, "x0": int((x0 - self.mb["x0"] - j) * s),
                       "y0": int((y0 - self.mb["y0"] + j) * s), "x1": int((x1 - self.mb["x0"] + j) * s),
                       "y1": int((y1 - self.mb["y0"] - j) * s)})
        return schema.model_validate({"is_legend": True, "swatches": sw})


def test_legend_swatches_from_vision_refined_and_saved(work):
    boxes, mb = write_legend_page("1948", "1")
    fake = LegendFake(boxes, mb)
    sw = pattern.legend_swatches("1948", client=fake)
    assert set(sw) == set(FAKE_LEGEND) and fake.calls == 1
    assert sw["C"].mean() < 20 and sw["2R"].mean() > 235      # border trimmed off
    assert 0.1 < (sw["1R"] < 128).mean() < 0.5
    saved = json.loads((config.WORK_DIR / "1948" / "1" / "district_legend.json").read_text())
    assert saved["sheet"] == "1" and set(saved["swatches"]) == set(FAKE_LEGEND)
    again = pattern.legend_swatches("1948", client=fake)        # artifact reused, no new call
    assert fake.calls == 1 and set(again) == set(sw)


def test_saved_legend_key_and_staleness(work, monkeypatch):
    boxes, mb = write_legend_page("1948", "1")
    fake = LegendFake(boxes, mb)
    pattern.legend_swatches("1948", client=fake)
    p = config.WORK_DIR / "1948" / "1" / "district_legend.json"
    saved = json.loads(p.read_text())
    assert saved["inputs_sha256"] == pattern.legend_inputs_sha256()
    assert saved["edge"] == pattern.LEGEND_EDGE and saved["era_codes"] == sorted(pattern.era_codes("1948"))
    # an artifact without the key (written through 2d1bf49) still matches the current defaults
    for k in ("inputs_sha256", "edge", "era_codes"):
        saved.pop(k)
    p.write_text(json.dumps(saved))
    assert pattern.legend_fresh(p, "1948")
    pattern.legend_swatches("1948", client=fake)
    assert fake.calls == 1
    # a changed prompt makes it stale: read again
    real = pattern.legend_prompt
    monkeypatch.setattr(pattern, "legend_prompt", lambda codes: real(codes) + " Look closely.")
    assert not pattern.legend_fresh(p, "1948")
    pattern.legend_swatches("1948", client=fake)
    assert fake.calls == 2
    monkeypatch.setattr(pattern, "LEGEND_EDGE", 2048)
    assert not pattern.legend_fresh(p, "1948")


def test_cache_only_pattern_miss_never_calls(work):
    sw = swatches()

    class Raising:
        calls = 0

        def call(self, png, prompt, schema):
            Raising.calls += 1
            raise AssertionError("network")

    with llm.cache_only(), pytest.raises(llm.LLMCacheMiss):
        pattern.classify(_items([("0003001", "0003", 0, 20)]), sw, client=Raising())
    assert Raising.calls == 0


def test_legend_fallback_config(work, monkeypatch):
    boxes, mb = write_legend_page("1948", "1")

    class NoLegend:
        def call(self, png, prompt, schema):
            return schema.model_validate({"is_legend": False, "swatches": []})
    cfg = {"1948": {"sheet": "1", "swatches": {c: list(b) for c, b in boxes.items()}}}
    monkeypatch.setattr(pattern, "_legend_config", lambda: cfg)
    sw = pattern.legend_swatches("1948", client=NoLegend())
    assert set(sw) == set(FAKE_LEGEND) and sw["C"].mean() < 20


def test_read_sheet_end_to_end(work):
    """A synthetic pattern sheet: block 0001 all solid (one tile), block 0002 split
    white/solid (per parcel); district_pattern.json written."""
    items = [(box(0, 0, 100, 100), "solid"), (box(100, 0, 150, 100), "blank"),
             (box(150, 0, 200, 100), "solid")]
    write_sheet("1948", "1", 0, 0, 200, 100, fill=pattern_fill(0, 0, 100, 1.0, items),
                paper="transparent")
    parcels = gpd.GeoDataFrame({
        "blklot": ["0001001", "0001002", "0002001", "0002002", "0002003"],
        "block_num": ["0001", "0001", "0002", "0002", "0002"]},
        geometry=[box(0, 0, 50, 100), box(50, 0, 100, 100), box(100, 0, 150, 100),
                  box(150, 0, 200, 100), box(150, 0, 200, 100)],      # 0002003: condo dup
        crs=config.WORK_CRS)
    sw = swatches()
    fake = _fake(sw)
    out = pattern.read_sheet("1948", "1", parcels, swatches=sw, client=fake)
    got = dict(zip(out["blklot"], out["code"]))
    assert got == {"0001001": "C", "0001002": "C", "0002001": "2R", "0002002": "C", "0002003": "C"}
    assert fake.calls == 1 and fake.tiles_seen == [3]      # 1 block tile + 2 unique parcels
    art = json.loads((config.WORK_DIR / "1948" / "1" / "district_pattern.json").read_text())
    assert len(art["parcels"]) == 5 and art["series"] == "1948"


def test_shipped_legend_config_matches_era_codes():
    cfg = pattern._legend_config()
    for series in ("1921", "1948"):
        assert set(cfg[series]["swatches"]) == set(pattern.era_codes(series)) == \
            {"1R", "2R", "C", "LI", "HI", "U"}


def _old_batches(rows):
    """Tile values in the pre-change item order: mixed 2-parcel blocks yield both parcels in
    order; chunked by MAX_TILES."""
    vals = [r[2] for r in rows]
    return [vals[b:b + pattern.MAX_TILES] for b in range(0, len(vals), pattern.MAX_TILES)]


def test_empty_tile_cache_sends_old_batches(work):
    # the mosaics must equal what the pre-change code built: same items, same order, 48 per batch
    sw = swatches(); rows = []
    for b in range(60):
        rows += [(f"{b:04d}001", f"{b:04d}", b % 40, 20), (f"{b:04d}002", f"{b:04d}", 255 - b % 40, 250)]
    fake = _fake(sw); pattern.classify(_items(rows), sw, client=fake)
    expect = [pattern.build_mosaic([_tile(v) for v in vals], sw)[0] for vals in _old_batches(rows)]
    got = [cv2.imdecode(np.frombuffer(p, np.uint8), cv2.IMREAD_COLOR)[..., ::-1] for p, _ in fake.sent]
    assert sorted(x.tobytes() for x in got) == sorted(x.tobytes() for x in expect)
    assert sorted(p for _, p in fake.sent) == sorted(
        pattern.pattern_prompt(len(v), list(sw), None, None) for v in _old_batches(rows))


def test_rerun_after_shift_pays_only_new_tile(work):
    sw = swatches()
    rows = [(f"{b:04d}00{j}", f"{b:04d}", (0 if j == 1 else 255), (20 if j == 1 else 250))
            for b in range(60) for j in (1, 2)]
    first = pattern.classify(_items(rows), sw, client=_fake(sw))
    fake = _fake(sw)
    new = [("9999001", "9999", 128, 120)]          # a new parcel at the front shifts every batch
    out = pattern.classify(_items(new + rows), sw, client=fake)
    assert fake.calls == 1 and fake.tiles_seen == [1]
    m = dict(zip(first["blklot"], first["code"]))
    assert all(m[k] == c for k, c in zip(out["blklot"], out["code"]) if k in m)


def test_all_hits_make_no_call(work):
    sw = swatches(); rows = [("0001001", "0001", 0, 20), ("0001002", "0001", 255, 250)]
    pattern.classify(_items(rows), sw, client=_fake(sw))
    class Never:
        def call(self, *a): raise AssertionError("no call expected")
    out = pattern.classify(_items(rows[::-1]), sw, client=Never())   # new mosaic, same tiles
    assert list(out["code"]) == ["2R", "C"] and llm.CallCounter.tile_hits >= 2


def test_legend_or_prompt_change_misses(work):
    sw = swatches(); rows = [("0001001", "0001", 0, 20), ("0001002", "0001", 255, 250)]
    pattern.classify(_items(rows), sw, client=_fake(sw))
    fake = _fake(sw); pattern.classify(_items(rows), sw, client=fake, year=1999)   # wording changes
    assert fake.calls == 1
    sw2 = dict(sw); k = next(iter(sw2)); sw2[k] = 255 - sw2[k]
    fake2 = _fake(sw2); pattern.classify(_items(rows), sw2, client=fake2)
    assert fake2.calls == 1


def test_omitted_index_not_cached(work):
    sw = swatches(); rows = [("0001001", "0001", 0, 20), ("0001002", "0001", 255, 250)]
    fake = _fake(sw); orig = fake.call
    def drop_first(png, prompt, schema):
        r = orig(png, prompt, schema); r.items = [i for i in r.items if i.index != 1]; return r
    fake.call = drop_first
    out = pattern.classify(_items(rows), sw, client=fake)
    assert out["code"].iloc[0] is None
    again = _fake(sw); pattern.classify(_items(rows[::-1]), sw, client=again)   # new mosaic
    assert again.calls == 1 and again.tiles_seen == [1]


def test_tile_cache_stores_raw_confidence(work):
    # a uniform block (factor 0.9) and a lone parcel with identical pixels share one entry
    sw = swatches()
    block = _items([(f"0001{i:03d}", "0001", 0, 30) for i in range(3)])
    pattern.classify(block, sw, client=_fake(sw), block_crops={"0001": _tile(0)})
    out = pattern.classify(_items([("0002001", "0002", 0, 30)]), sw, client=_fake(sw))
    assert out["confidence"].iloc[0] == pytest.approx(0.8)      # raw, not 0.72


def test_cached_code_outside_legend_maps_to_none(work):
    sw = swatches(); rows = [("0001001", "0001", 0, 20)]
    tile = pattern._as_tile(_tile(0), pattern.TILE_PX)
    llm.tile_put(llm.tile_key(tile, pattern._tile_context(sw, list(sw), None, None)),
                 {"code": "ZZ", "confidence": 0.9})
    class Never:
        def call(self, *a): raise AssertionError("no call expected")
    out = pattern.classify(_items(rows), sw, client=Never())
    assert out["code"].iloc[0] is None and out["confidence"].iloc[0] == 0.0


def test_cache_only_serves_tile_hits(work):
    sw = swatches(); rows = [("0001001", "0001", 0, 20)]
    pattern.classify(_items(rows), sw, client=_fake(sw))
    with llm.cache_only():
        out = pattern.classify(_items(rows), sw, client=None)
    assert out["code"].iloc[0] == "C"


def test_first_run_free_even_with_shared_tile_across_calls(work):
    import shutil

    sw = swatches()
    rowsA = [("0001001", "0001", 7, 20), ("0001002", "0001", 250, 250)]
    # B: 100 mixed-block parcels (3 old mosaics); its first tile repeats one of A's
    rowsB = [("9000001", "9000", 7, 20), ("9000002", "9000", 250, 250)]
    for b in range(1, 50):
        rowsB += [(f"{9000 + b}001", f"{9000 + b}", 30 + b, 20), (f"{9000 + b}002", f"{9000 + b}", 240 - b, 250)]
    pattern.classify(_items(rowsB), sw, client=_fake(sw))            # B's old mosaics are cached
    first = pattern.classify(_items(rowsB), sw, client=_fake(sw))
    shutil.rmtree(llm.CACHE_DIR / "llm" / "tiles")                   # tile store empty again
    pattern.classify(_items(rowsA), sw, client=_fake(sw))            # A stores the shared tile
    class Never:
        def call(self, *a): raise AssertionError("no call expected")
    out = pattern.classify(_items(rowsB), sw, client=Never())
    assert list(out["code"]) == list(first["code"])
