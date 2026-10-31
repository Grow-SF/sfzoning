# Districts: cache pattern answers per tile, not per mosaic

Status: design, written autonomously (the owner asked "do the block caching fix; brainstorm
it to properly plan first (do this fully autonomously)"). Every decision the owner did not
state is marked **(my call)**.

## Why

The pattern reader (`districts/pattern.py`, editions 1921, 1929, 1935, 1948) asks the vision
model about up to `MAX_TILES = 48` block/parcel tiles at a time, packed into one mosaic image
with the legend strip on top. `llm.ask_json` caches the answer keyed by the whole mosaic PNG
and prompt.

When a sheet is added or re-placed, `sampling.assign_sheets` moves a few parcels between
neighbouring sheets. Every later tile in that sheet's list shifts position, so nearly every
mosaic is a new image and is paid for again, though almost every tile in it was already read.

On 2026-10-31, adding four 1921 sheets re-billed five already-read neighbours: about $18
in one afternoon, roughly $3 per sheet. More sheet fixes are coming (Lake Merced, Bayview in
1929/1935/1948), and each would re-bill its neighbours the same way.

## Goal

A tile the model has already classified is never paid for again while it renders to the
same pixels under the same legend, prompt wording and model. A rerun after a placement
change pays only for tiles whose pixels changed: the new sheet's, and the few blocks at
each neighbour's edge whose parcel set changed.

## Non-goals

- No change to what is sent: tile rendering, mosaic layout, `MAX_TILES`, the prompt
  wording, the model and the legend strip stay byte-identical. Any of these changing would
  invalidate the existing cache and cost money.
- No change to the region reader (`districts/regionread.py`, the 1990s code sheets). It
  batches the same way and has the same flaw, but its regions come from a different
  pipeline and it is not in today's spend **(my call: a follow-up if it starts to cost)**.
- No refund or recovery of the $18 already spent.
- No new CLI flags.

## Anti-goals

- **The first run under the new code must cost nothing extra.** With an empty tile cache,
  the reader must build exactly the mosaics the old code built, in the same order, so every
  one is a hit in the existing mosaic cache. That run fills the tile cache for free.
- **A cached answer must never be used for a tile whose pixels differ.** The key is the
  exact rendered tile, never a parcel id, block number or geometry.
- **No change in results for unchanged inputs.** The same tile gets the same code and
  confidence as before.

## Behavior

### The tile key

- The key is the sha256 of:
  - the exact `TILE_PX × TILE_PX` RGB tile as placed in the mosaic (`_as_tile(crop,
    TILE_PX)`), as PNG bytes;
  - the legend identity: the sha256 of the legend strip rendered by `build_mosaic` with no
    tiles. Equivalently, the swatch pixels plus the codes in order; the builder picks
    whichever is simplest and stable;
  - the prompt template: `pattern_prompt` is left untouched. The key hashes
    `pattern_prompt(0, codes, names, year)`: a fixed placeholder count, because the count
    varies by batch and must not split the key. `codes`, `names` and `year` stay in it,
    since changing them changes the wording **(my call, after the builder check)**;
  - the same model parameters `llm._key` folds in: provider, model, reasoning effort,
    max output tokens, detail, and the `PatternItem` schema. The builder may reuse or share
    `llm._key`'s parameter list so the two never drift.
- **Storage:** one small JSON file per tile, `CACHE_DIR / "llm" / "tiles" / <key>.json`,
  holding the raw answer `{"code": <str>, "confidence": <float>}` as the model returned it.
  That is before `_match_code` and before the block factor.
  - **(my call:)** this sits under `cache/llm/`, so the processed bundle
    (`share.py`, pattern `cache/llm/**`) carries it with the existing answers. There is no
    new bundle rule.
  - Writes are atomic: temp file, then `os.replace`, as in `llm.ask_json`.
  - A corrupt file is ignored and deleted, as in `llm.ask_json`.

### `classify()` flow

1. Build `items` exactly as today: block or parcel, the same grouping, the same order.
2. For each item, render its tile and compute the key. Split the items into cached (a
   tile-cache hit) and uncached, keeping their order.
3. Batch only the uncached items, in order, `MAX_TILES` at a time, and ask through the
   existing `llm.ask_json`. Its mosaic cache, cache-only mode, budget guard and counters
   all apply unchanged.
4. For every tile answered, write its tile entry. That covers live calls and mosaic-cache
   hits alike, so the first run backfills the tile cache from the answers already paid
   for.
5. Map answers back to parcels exactly as today: `_match_code`, the confidence clip,
   `BLOCK_FACTOR` for block tiles.

Rules for the edge cases:
- **An index the model left out of a mosaic's answer:** do not write a tile entry, so a
  later run retries it. The parcels stay `code None, confidence 0`, as today **(my call)**.
- **A code outside the legend:** write the raw answer anyway, because it is what the model
  said about these pixels. `_match_code` drops it at mapping time, as today **(my call)**.
- **Cache-only mode:** tile hits are served, and only uncached tiles go to `ask_json`,
  which raises `LLMCacheMiss` as today. So `--cache-only` becomes more useful: a sheet
  whose only changes are at its edges can now be served almost entirely from the cache.

- **Two uncached tiles with identical pixels in one run:** both are sent, to keep the old
  batching. Each parcel gets its own mosaic's answer, as today. The tile store keeps the
  first answer written for a key and never overwrites an existing entry, so answers stay
  stable across runs **(my call, after the builder check)**.

### Why the first run is free

With an empty tile cache, every item is uncached. Step 3 then batches exactly the items
the old code batched, in the same order and the same groups of 48, with the same images and
prompts. So every `ask_json` call hits the mosaic cache if the old code had answered it.
A test must pin this: with an empty tile cache, the mosaics and prompts are byte-identical
to the old batching.

The guarantee covers an empty tile cache only. With a partly filled tile cache, for
example after an interrupted first run, the uncached remainder is re-batched into new
mosaics and paid for. That is the intended trade-off: those mosaics were never answered
anyway, apart from ones that finished concurrently **(my call, after the builder check)**.
An edition whose sheet assignment changed before its first run under the new code pays
once for the shifted mosaics, as it would today. After that, it doesn't again.

### Reuse across mosaics

A cached answer was given with other tiles beside it in a mosaic. Reusing it in a run where
its neighbours differ is accepted. Each tile is judged on its own pixels inside the red
outline, and the existing prompt already says the area outside the outline is "context only".
No prompt text is added or changed **(my call, after the builder check)**.

### Visibility

- Count tile-cache hits with a new `CallCounter` field (`tile_hits`).
- Report them where the run summary already prints live calls and cost, for example "tile
  cache: N tiles reused". **(my call:)** the builder picks the exact wording and place.
- A per-sheet debug log line is fine.
- The counter is reset with the others. It is printed only in the CLI run summary (where
  `cli.py` prints live calls). `district_pattern.json` gains no field. No other code path
  predicts pattern-reader cost: `estimate` covers page and label reads only **(my call,
  after the builder check)**.

## Constraints

- Python 3.12, existing modules only; no new dependencies.
- `cd pipeline && uv run pytest` stays green. Tests use the existing synthetic fixtures
  (`tests/synthetic_districts.py`, `tests/test_districts_pattern.py`) and a fake client;
  there are no real API calls.
- Never print or log the OpenAI key.
- Rendering stays byte-identical to today (see Non-goals).

## Tests the builder must include

- **Free first run:** with an empty tile cache, the mosaics and prompts sent equal the old
  batching. Record calls with a fake client before and after the change, or compare against
  a direct old-style batching.
- **Shifted list:** after a full read, insert one new parcel at the front of a sheet's list.
  The rerun makes exactly one live mosaic call containing only the new tile, and every
  parcel's code matches the first read.
- **Changed pixels:** a changed tile (one pixel different) is a miss.
- **Legend or prompt change:** a different legend swatch, or a different prompt wording,
  is a miss.
- **Omitted index:** an index the model left out is not cached.
- **Cache-only:** tile hits are served without the client; an uncached tile still raises
  `LLMCacheMiss`.

## Left to the builder

- Helper names; whether the tile store lives in `llm.py` (a small `tile_cache_get/put`) or
  in `pattern.py`.
- How the legend identity is hashed.
- The exact log or summary wording.
