# District Tile Cache Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Cache the pattern reader's vision answers per rendered tile, so a rerun after a
sheet placement change pays only for tiles whose pixels changed.

**Architecture:**
- A small per-tile answer store in `llm.py`: key, get, put-if-absent, plus a `tile_hits`
  counter.
- `pattern.classify` splits its items into tile-cache hits and misses. It batches only the
  misses, in the original order, through the unchanged `llm.ask_json`, then writes each
  answered tile back.
- With an empty store, the batches are exactly the old ones, so the existing mosaic cache
  serves them for free.

**Tech Stack:** Python 3.12, numpy, OpenCV, pydantic, pytest (uv).

**Spec:** `docs/superpowers/specs/2026-10-31-district-tile-cache-design.md`

## Global Constraints

- Tile rendering, mosaic layout, `MAX_TILES = 48`, `pattern_prompt` wording, the model and
  the legend strip stay byte-identical. `pattern_prompt` is not edited.
- Tile store path: `CACHE_DIR / "llm" / "tiles" / <key>.json`. The value is the raw answer
  `{"code": str, "confidence": float}`, before `_match_code` and before `BLOCK_FACTOR`.
- Writes are atomic (temp file + `os.replace`). An existing entry is never overwritten.
  A corrupt entry is ignored and deleted.
- No new dependencies and no new CLI flags. Never print or log the OpenAI key.
- Tests use fakes only, with no network. `cd pipeline && uv run pytest` stays green.

## Review Focus

1. **A tile-cache hit with a code outside the current legend:** it maps to `None` via
   `_match_code`, as a live answer would; no crash. Pinned in Task 2, step 1.
2. **Every item in a run is a hit:** no `ask_json` call is made, not even with an empty
   batch, and no client is constructed. Pinned in Task 2, step 1.
3. **A corrupt tile JSON file:** treated as a miss and deleted, not a crash. Pinned in
   Task 1, step 1.
4. **Concurrent puts of the same key from the `pmap` workers:** the put is safe and the
   first one wins. Covered by put-if-absent through a temp file plus `os.replace`; a check
   in Task 1 shows a second put leaves the first value.
5. **A block tile and a parcel tile with identical pixels:** they share a key. That's
   correct, since the key is the pixels; the block factor is applied at mapping time, not
   stored. Pinned in Task 2, step 1 (`test_tile_cache_stores_raw_confidence`).

---

### Task 1: Per-tile answer store in `llm.py`

**Files:**
- Modify: `pipeline/src/sfzmaps/llm.py`: `CallCounter`, plus new functions after `is_cached`.
- Test: `pipeline/tests/test_llm.py`

**Interfaces:**
- Produces:
  - `tile_key(tile_rgb: np.ndarray, context: str) -> str`: the hex sha256 of the tile's
    PNG bytes (`_encode` of the BGR-converted tile), `context`, and the same
    provider/model/parameter list `_key` uses. Refactor that list into one shared helper,
    so the two can't drift.
  - `tile_get(key: str) -> dict | None`
  - `tile_put(key: str, answer: dict) -> None`: put-if-absent, atomic.
  - `CallCounter.tile_hits: int`, reset by `CallCounter.reset()`. `tile_get` bumps it on
    a hit.

- [ ] **Step 1: Write the failing tests** in `tests/test_llm.py`. Use a fixture that
  monkeypatches `llm.CACHE_DIR` to `tmp_path` and calls `llm.CallCounter.reset()`.

```python
def test_tile_store_roundtrip_and_counter(tmp_path, monkeypatch):
    monkeypatch.setattr(llm, "CACHE_DIR", tmp_path); llm.CallCounter.reset()
    t = np.full((96, 96, 3), 7, np.uint8)
    k = llm.tile_key(t, "ctx")
    assert llm.tile_get(k) is None and llm.CallCounter.tile_hits == 0
    llm.tile_put(k, {"code": "C", "confidence": 0.8})
    assert llm.tile_get(k) == {"code": "C", "confidence": 0.8}
    assert llm.CallCounter.tile_hits == 1
    assert (tmp_path / "llm" / "tiles" / f"{k}.json").exists()
    llm.CallCounter.reset(); assert llm.CallCounter.tile_hits == 0

def test_tile_put_never_overwrites(tmp_path, monkeypatch):
    monkeypatch.setattr(llm, "CACHE_DIR", tmp_path)
    k = llm.tile_key(np.zeros((96, 96, 3), np.uint8), "ctx")
    llm.tile_put(k, {"code": "C", "confidence": 0.8})
    llm.tile_put(k, {"code": "2R", "confidence": 0.1})
    assert llm.tile_get(k)["code"] == "C"

def test_tile_key_changes_with_pixels_and_context():
    a = np.zeros((96, 96, 3), np.uint8); b = a.copy(); b[5, 5, 0] = 1
    assert llm.tile_key(a, "x") != llm.tile_key(b, "x")
    assert llm.tile_key(a, "x") != llm.tile_key(a, "y")
    assert llm.tile_key(a, "x") == llm.tile_key(a.copy(), "x")

def test_tile_get_corrupt_file_is_a_miss(tmp_path, monkeypatch):
    monkeypatch.setattr(llm, "CACHE_DIR", tmp_path)
    k = llm.tile_key(np.zeros((96, 96, 3), np.uint8), "ctx")
    p = tmp_path / "llm" / "tiles" / f"{k}.json"; p.parent.mkdir(parents=True); p.write_text("{nope")
    assert llm.tile_get(k) is None and not p.exists()
```

- [ ] **Step 2: Run them to verify they fail.** Run:
  `cd pipeline && uv run pytest tests/test_llm.py -k tile -q`. Expected: FAIL with
  AttributeError (no `tile_key`).

- [ ] **Step 3: Implement** `tile_key`, `tile_get` and `tile_put`, and add `tile_hits` to
  `CallCounter`, in `llm.py`. `_key`'s existing output must not change: existing cache
  files must still hit. Factor out the parameter list only; don't change its contents or
  order.

- [ ] **Step 4: Run the tests.** Run: `cd pipeline && uv run pytest tests/test_llm.py -q`.
  Expected: all pass, including the existing `_key` / cache tests.

- [ ] **Step 5: Commit**

```bash
git add pipeline/src/sfzmaps/llm.py pipeline/tests/test_llm.py
git commit -m "feat(llm): per-tile answer store (key, get, put-if-absent, tile_hits)"
```

### Task 2: `pattern.classify` uses the tile store; the CLI reports reuse

**Files:**
- Modify: `pipeline/src/sfzmaps/districts/pattern.py` (`classify`, about lines 416–466)
- Modify: `pipeline/src/sfzmaps/cli.py` (the district summary, about line 349)
- Modify: `pipeline/README.md`: one sentence wherever district costs/caching are described
- Test: `pipeline/tests/test_districts_pattern.py`

**Interfaces:**
- Consumes: Task 1's `llm.tile_key`, `llm.tile_get`, `llm.tile_put` and
  `llm.CallCounter.tile_hits`.
- Produces: no new public API. `classify` keeps its signature and output columns.

- [ ] **Step 1: Write the failing tests** in `tests/test_districts_pattern.py`, reusing
  `work`, `FakeClient`, `_fake`, `_tile`, `_items` and `swatches()`. `_mosaics(fake)`
  means the list of (png bytes, prompt) a recording fake saw. Extend `FakeClient` to
  record `self.sent.append((png, prompt))`.

```python
def test_empty_tile_cache_sends_old_batches(work):
    # the mosaics must equal what the pre-change code built: same items, same order, 48 per batch
    sw = swatches(); rows = []
    for b in range(60):
        rows += [(f"{b:04d}001", f"{b:04d}", b % 40, 20), (f"{b:04d}002", f"{b:04d}", 255 - b % 40, 250)]
    fake = _fake(sw); pattern.classify(_items(rows), sw, client=fake)
    expect = [pattern.build_mosaic([_tile(v) for v in vals], sw)[0] for vals in _old_batches(rows)]
    got = [cv2.imdecode(np.frombuffer(p, np.uint8), cv2.IMREAD_COLOR)[..., ::-1] for p, _ in fake.sent]
    assert sorted(x.tobytes() for x in got) == sorted(x.tobytes() for x in expect)
    # _old_batches: the per-parcel item order of the mixed blocks, chunked by 48 (helper in the test)

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
    out = pattern.classify(_items(rows), sw, client=Never())
    assert list(out["code"]) == ["C", "2R"] and llm.CallCounter.tile_hits >= 2

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
    again = _fake(sw); pattern.classify(_items(rows), sw, client=again)
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
```

The existing `test_cache_only_pattern_miss_never_calls` must still pass unchanged.

- [ ] **Step 2: Run them to verify they fail.** Run:
  `cd pipeline && uv run pytest tests/test_districts_pattern.py -q`. Expected: the new
  tests FAIL (calls made on rerun; no `_tile_context`).

- [ ] **Step 3: Implement it in `pattern.py`.**
  - Add `_tile_context(swatches: dict, codes: list[str], names: dict | None, year: int | None) -> str`.
    It returns a string that changes whenever the legend strip or the prompt wording
    changes. Hash the legend strip `build_mosaic([], swatches)[0]`, or the swatch pixels
    and codes, together with `pattern_prompt(0, codes, names, year)`.
  - In `classify`, after building `items`, render each item's tile with
    `_as_tile(crop, TILE_PX)` and key it with `llm.tile_key(tile, ctx)`.
    - Hits supply the raw answer directly.
    - Misses keep their order and are batched exactly as before.
    - When there are no misses, skip `pmap` and `ask_json` entirely.
    - After each batch is answered, call `llm.tile_put` for every tile whose index the
      model returned, storing `{"code": it.code, "confidence": it.confidence}`.
  - Mapping (`_match_code`, the clip, `BLOCK_FACTOR`) applies the same way to hits and
    to fresh answers.

- [ ] **Step 4: Implement the CLI summary.** Next to the "live vision calls" line in
  `cli.py`, print `tile cache: {n} tiles reused`, using the `tile_hits` delta for the run
  in the same way `live0` is used. Print it only when `n` is greater than 0.

- [ ] **Step 5: Update the README.** Add one sentence to `pipeline/README.md` where
  district reading costs or caching are described: pattern answers are also cached per
  tile, so re-placing a sheet re-reads only the tiles that changed.

- [ ] **Step 6: Run the tests.** Run `cd pipeline && uv run pytest -q`. Expected: all pass
  (561+ before, plus the new tests). The one shapely deprecation warning predates this
  branch.

- [ ] **Step 7: Commit**

```bash
git add pipeline/src/sfzmaps/districts/pattern.py pipeline/src/sfzmaps/cli.py pipeline/README.md pipeline/tests/test_districts_pattern.py
git commit -m "feat(districts): pattern reader reuses answers per tile; rerun pays only for changed tiles"
```
