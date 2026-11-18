# Task 6 report: composites inside real crops

Status: DONE_WITH_CONCERNS. Commit 8518d6c (base 102f422; the controller's test-only commit 891ce2e came in between).

## What was built

- **`pipeline/src/sfzmaps/solver/textnet/composite.py`** (new; imports `data`, `linegate`, `zones`; no stage imports it).
  - Clean words come from `linegate.clean_word_mask(sd, line)`, where `line` is the unpacked `SheetData.line_bits`, as
    the brief says.
  - **Line pieces.** These are 8-connected pieces of free line ink (`line_bits & ink`) more than 3 px from every Terra
    word, other box, unmatched zone and held-out px. They are cut by a 256 px tile grid, whose start per sheet comes
    from the bank's own `default_rng([seed, 1])`, and kept when their bounding diagonal is at least 100 px.
  - **`Bank.build(sheets, areas, seed, threads=1)`** works on `data.hold_out(sheets, areas)`.
    - It stores references only: `Words` (sheet, rect, kind, angle) and `Pieces` (sheet, box, seed px), per series.
    - It also stores the crossing share per series and kind, measured on its own sheets. Words whose centre is held out
      are left out.
    - A clean word goes in the bank only if its rectangle's bbox plus 3 px has no held-out px and lies on the page.
      Pieces never touch held px.
  - **`Bank.crop(rng, sd, y, x, size, scale, turns)`** returns the keys of `crop_targets`, from the passed rng only. It
    picks (a) or (b) with `A_SHARE = 0.5`. Both work at page scale on `crop_targets(..., src, 1.0, 0,
    labels="lines-unknown")`, then resize and turn as `crop_targets` does.
    - `_page_parts` and `_finish` are tested equal to `crop_targets` byte for byte at 4 scale/turn pairs.
    - **(a) word onto lines.**
      - A clean word of the crop's series is drawn, then on or off the lines with that series' `cross[kind]`.
      - On: a word-rectangle px goes on a free line-ink px, and the rectangle must hold at least 3 `line_bits` px. Off:
        no `line_bits` within 3 px of the rectangle.
      - It may not touch a Terra word, other box, unknown zone or held px within 3 px, and it must fit in the crop.
      - After 20 tries the next word is drawn. After 8 words the crop is returned as it is (`left_as_is`).
      - The word's ink is ORed in and becomes text (known, with its kind and angle); overlap px are text, weight 1.
    - **(b) line onto a word.**
      - The crop is re-centred on a clean word of the series. All of that word's ink becomes text and known.
        *Corrected in fix round 1:* I had claimed this restores the long strokes the label rule left unknown. It
        doesn't on the real maps: no clean code or block number holds a run-rule px in its box (the reviewer's count,
        NE held). What composites teach is that letter px touching or under line ink are text (see the fix report).
      - A real piece of the series, turned a random number of quarter turns, is ORed in. One of its px lands on a
        random px of the word's rectangle.
      - Piece px that land on paper become ink with text 0, and keep the known value they had, so never on held or
        unknown px. Piece px on other ink keep their targets.
    - **Exact masks.** In a composite crop, text is only ever ink. Box paper (text by the box rule) becomes text 0 and
      unknown, and so does the pasted word's rectangle paper.
  - **`measure(sheets, areas=(), threads=1)`** returns:
    - `cross_by_kind`: at least 3 `line_bits` px inside the box, the brief's definition;
    - `cross_by_kind_ink`: the same, counting only line px that are ink;
    - `clean_words`, `clean_word_texture`, and `line_texture`.
  - **`texture_stats(masks)`** gives the isolated share, pinholes per 1,000 ink px and pieces per mask.
  - **`line_texture(pieces)`**
    - **Edges.** Per piece, the px' principal axis is found, the piece is stepped along it 1 px at a time, and the
      farthest px either way at each step is an edge.
    - **Per edge.** The offset from a least-squares straight line, pooled into an SD and lag-1/2/4 autocorrelation,
      by width class (1–5, 6+).
    - **Not straight.** A piece is left out and counted as `not_straight` when a step has two runs (a gap over 1.5 px)
      or is empty, or when an edge strays more than one width from its fit.
- **`train.py`** gains no import.
  - `Recipe.composites: float = 0.0`, validated to lie in [0, 1]. Its name is "composites=p" only when it is set.
  - `draw(..., bank=None)` draws `turns` into a variable in the same order as before. It draws `rng.random() < p` only
    when p > 0, then calls `bank.crop`.
  - `_batches(..., bank)` and `fit(..., bank=None)`: a recipe with composites and no bank raises ValueError before
    training.
- **`scripts/textlines/variant.py`** (Ruling V1).
  - `run()` refuses composites without `labels="lines-unknown"`, and refuses a bank with composites 0.
  - When composites > 0 it builds `composite.Bank.build(prepared, areas, seed, threads=4)` before `gpu.gpu_turn` and
    passes it to `train.fit`.
  - `run.json` gains `bank` (sizes), `composites_made` (the tally) and `minutes.bank`.
  - The CLI's `--composites P` is live and needs `--labels lines-unknown`.
- **`scripts/textlines/composites.py`** (new, committed so the outputs can be regenerated). It writes
  `distributions.json`, `composites.json` (timing, tally, texture check) and `composites.png`, under
  `data/solver/experiments/textlines/`. The data are not committed.

## TDD evidence

- `PINNED_BATCHES_SEED3 = dd2f5a81…7820` is the sha256 of 3 batches of `train.Recipe()` on the `maps` fixture, seed 3.
  It was computed twice, identically, with `train.py` at 102f422, before `draw` changed.
- **Red.** `tests/test_textnet_composite.py` (20 tests: the brief's 5 plus 15) failed first with an ImportError
  (no `composite`).
- **Green.**
  - The 20 composite tests pass.
  - One line in a test was wrong and I fixed it: cv2.line at thickness 2 draws 3 px, so the measure test's width class
    is "3", not "2".
  - The composites=0 batches equal the pin, byte for byte.
- **Tests I added:**
  - (a) on a line: the word lands on line ink and stays more than 3 px from every forbidden px (Euclidean distance
    transform); the line ink off its letters stays not text and known; some letters overlap the line.
  - (a) off the lines: more than 3 px from every line px.
  - (b): the new ink is not text and known; all of the word's ink is text and known; text ⊆ ink; the piece crosses the
    box.
  - (b) on a 3x putText "RM-2": more than 100 long-stroke px are unknown under the label rule and become text in the
    composite. This shows the mechanism on a drawn word only; real clean words hold no such px (fix round 1).
  - A bank built with a `MapRect` over the right half holds no word or piece within 3 px of it, and its (b) crops leave
    every held px unknown.
  - A bank whose words all fail returns `crop_targets` unchanged.
  - `measure` counts; `texture_stats` and `line_texture` on drawn masks (a straight bar has SD 0; iid bumps give
    SD ≈ sqrt(var/2) and |ac1| < 0.25; blocky bumps give ac1 > 0.6; an L shape is `not_straight`).
  - Recipe name and range.
  - `draw` with a fake bank: 0 calls at p = 0, all crops at p = 1, ValueError without a bank.
  - `fit` refuses composites with no bank, and trains with one.
- **The brief's test 5** keeps its name and fixture. I added assertions: every crop gets the small word, and
  `tally["words_failed"]` rises, so the next candidate really is drawn.
- **`tests/test_textlines_variant.py`:** a composites run builds its bank (areas, seed, line px present) before the GPU
  turn; `run.json` records it; labels "boxes" with composites raises.
- **Unlisted mechanical edit:** `test_default_recipe_trains_as_today` pins `asdict(Recipe())`, so it gains
  `"composites": 0.0`. Task 4 made the same edit for `labels`.
- **Results.**
  - The brief's test command: 219 passed, 3 xfailed.
  - `uv run pytest -q`: 1527 passed, 3 xfailed. This includes `test_build.py`; the closure is unchanged.

## distributions.json (2000 sheets, training quarters; at the top NE and the answered blocks held out)

`cross_by_kind`: at least 3 `line_bits` px inside the box.

| kind | words | crossed | share |
|---|---|---|---|
| zoning code | 772 | 416 | 0.539 |
| block number | 3074 | 673 | 0.219 |
| street name | 1775 | 1115 | 0.628 |
| other | 313 | 74 | 0.236 |
| **all (word-weighted)** | 5934 | 2278 | **0.3839** |

- `cross_by_kind_ink`, counting only line px that are ink: code 0.523, block 0.193, street 0.532, other 0.233, all
  0.340.
- p by held quarter: nw 0.398, ne 0.384, sw 0.400, se 0.393.
- Clean words in 2000's training quarters (NE held): code 282, block 1689, street 36, other 123.
- Clean-word texture over 2130 words: isolated share 3.5e-5, 0.64 pinholes per 1,000 ink px, 4.2 pieces a word.

`line_texture` (NE held, 2000):

| width | pieces | edge px | SD (px) | ac1 | ac2 | ac4 |
|---|---|---|---|---|---|---|
| 1 | 4 | 926 | 0.23 | 0.79 | 0.60 | 0.31 |
| 2 | 139 | 38,870 | 0.40 | 0.33 | 0.29 | 0.18 |
| 3 | 366 | 102,808 | 0.42 | 0.70 | 0.57 | 0.41 |
| 4 | 52 | 12,700 | 0.69 | 0.89 | 0.82 | 0.73 |
| 5 | 13 | 4,280 | 0.89 | 0.83 | 0.76 | 0.69 |
| 6+ | 677 | 308,170 | 0.56 | 0.83 | 0.70 | 0.54 |

`not_straight` = 2500 pieces (junctions, corners, crossings): about two thirds of the pieces. No drawn lines are used
tonight; this table is only what drawn lines would have to match.

## The bank per series and quarter (clean words code/block/street/other; pieces)

| held | 2000 | 1980-1990 | 1990-1996 | 1960 | 1964 | 1968 |
|---|---|---|---|---|---|---|
| nw | 295/1648/27/127; 3919 | 228/1814/53/86; 3975 | 269/1633/32/112; 4244 | 228/0/107/101; 9007 | 229/0/311/139; 10962 | 222/0/266/154; 13862 |
| **ne** | 265/1688/28/97; 3799 | 192/1765/41/62; 3781 | 234/1515/21/98; 3916 | 173/0/82/95; 7927 | 187/0/146/124; 7960 | 191/0/150/135; 11384 |
| sw | 206/1626/27/86; 4000 | 166/1767/57/62; 4312 | 202/1539/35/80; 4136 | 190/0/88/79; 8755 | 196/0/263/109; 11197 | 198/0/240/117; 14049 |
| se | 201/1578/26/91; 3703 | 161/1755/60/69; 3926 | 172/1457/34/72; 3706 | 194/0/115/66; 7673 | 200/0/316/102; 10425 | 201/0/297/128; 14622 |

The crossing share the bank uses per series (NE held), code / block / street / other:

- 2000: .54 / .22 / .63 / .24
- 1980-90: .63 / .22 / .61 / .07
- 1990-96: .53 / .20 / .60 / .24
- 1960: .59 / – / .47 / .31
- 1964: .53 / – / .38 / .29
- 1968: .52 / – / .45 / .31

The bank builds in about 35 s with 4 threads (all 87 sheets); `measure` on the 13 2000 sheets takes about 5 s.

## Draw timing (real data; NE held; stage.DEFAULT_RECIPE with lines-unknown; 12 crops of 320; 100 batches; seed 0; CPU, one thread)

| composites | draw ms/batch | draw + batch_tensors ms/batch |
|---|---|---|
| 0.3839 | 33.2 | 38.1 |
| 0 | 27.3 | 32.1 |

- The draw is about 6 ms a batch (about 22%) slower with composites. It is still well under the GPU step's ~95 ms, so
  with prefetch the training is not draw-bound.
- What the 1,200 crops held:
  - 231 (a): 104 on lines, 127 off;
  - 244 (b);
  - 2 left as they were;
  - 130 word draws failed and were redrawn.
- 477 / 1200 = 0.40 of the crops were composites, against p = 0.384.
- The centre pools took 1.9 s.

The texture check: `texture_stats` of text ink, at page scale, against real crops' known text ink at the same centres.

| | composites: isolated share | composites: pinholes / 1,000 | real crops: isolated share | real crops: pinholes / 1,000 |
|---|---|---|---|---|
| (a) | 3.3e-4 | 1.28 | 7.7e-4 | 0.94 |
| (b) | 2.6e-4 | 0.51 | 3.5e-4 | 0.47 |

Both are of the same order. As M7 says, a composite's text ink is copied real ink, so it matches by construction.

## What composites.png shows

`data/solver/experiments/textlines/composites.png`, 24 tiles as the training sees them (scaled and turned). Text is
red, other ink grey, and px with no lesson are tinted blue. The pasted word's frame, or the (b) word's box, is green.

- **Row 1–2, (a) word onto lines.** The pasted words sit in open paper or across real line work, and none touches
  another word, box or blue (unknown) area:
  - block numbers "5916", "2592" (turned), "1426" set on a vertical lot line, "6563" on a street line, "1023";
  - "ST.", "LOS", "O", "AT";
  - on the line placements, the line ink under the word's box stays grey off the letters and red where it overlaps
    them.
- **Row 3–4, (b) line onto a word.** Real line pieces cross clean words:
  - diagonal lines through 1960's "R-1", "M-1" and "C-1";
  - a horizontal line through "NORTH";
  - a junction (a "+") over "EXPRESSWAY";
  - a thin leader-like line running into "OF 13 SHEETS";
  - vertical lines by "ST" and through "SEE SHEET 6".
  - The piece px are grey, the letters red (overlaps too), and the word box's paper blue.
- **Also visible:**
  - Real words in the crops keep their own targets, red, with their in-box line px pale (unknown).
  - A held-out band (1990-96 p25, tile 4) is entirely pale: no lesson.
  - The first two (b) tiles reuse the same 1968 word (198): a 1-in-476 coincidence of two `rng.integers` draws, not a
    bug (checked by replaying the draws).
  - 9 of the 12 (b) tiles happen to come from the 1960s maps, because the crop's series comes from the pool.
  - Some pieces end abruptly in the open, where the 256 px tile grid cut them.

## P for Task 7

*Superseded by fix round 1 (crossing counted on line ink only): P = 0.3396 for NE, 0.3507 for SW.*

- **P = 0.3839 for NE:** `distributions.json` → `cross_by_kind.all.share` (`p.ne`), word-weighted over all kinds, by the
  brief's `line_bits` definition. Pass `--composites 0.3839`.
- For SW (step 4): `p.sw` = 0.4001.

## Concerns

1. **Kind mix.**
   - (a) draws a clean word uniformly. In the 2000, 1980 and 1990 maps, 81% of clean words are block numbers; codes
     are 13% (265) and street names 1% (28), since street names sit between street lines and are rarely clean.
   - So the big bold codes with long strokes, the RM-2 failure, are a minority of (a) composites. (b) has the same mix.
   - Weighting by kind would be an unmeasured parameter, so I did not add it.
2. **The crossing definition.**
   - The brief's `line_bits` count (the run rule grown by 1 px) counts a line within 1 px outside the box. Streets come
     out 0.63, against the red team's 0.47 (raw run-rule px, whole city). The ink-only count gives 0.53 for streets and
     P = 0.340.
   - The run rule's own false lines in long letter strokes also make some big codes "crossed".
   - P uses the brief's definition; both are in the json.
3. **Departures from "the rest of the crop keeps its own targets".**
   - In a composite crop, box paper is unknown, for real words too. The brief's test `text ⊆ ink` needs this, and it
     avoids teaching paper next to letters as not-text in composites while real crops teach it as text.
   - Real words' ink keeps its targets.
4. **Crossing share per series.** The bank uses each series' own measured crossing share, from its training quarters.
   The brief measures on 2000 only; borrowing 2000's share for old maps would have been unmeasured for them.
5. **Clean words** use `line_bits` (grown by 1) in `clean_word_mask`, as the brief says. They are therefore a subset of
   the gate's `m2_clean_words`, which passes raw `line_px`.
6. **Line pieces.**
   - The tile grid cuts lines into pieces that can end abruptly inside a crop.
   - About two thirds of pieces are junctions or corners. They are used as real line work in (b) but left out of
     `line_texture`.
   - The crossing angle comes from the piece's own orientation times quarter turns, not from a measured line-vs-label
     angle distribution.
7. **Unmeasured constants.** A_SHARE 0.5 (an equal split, F3), WORDS 8, TRIES 20 (the brief's) and TILE 256 are stated
   choices.
   - Failed placements redraw the word and its on/off decision, which biases (a) toward off-line when on-line places
     are scarce. The realised on-line share was 104/231 = 45%.
8. **The bank costs about 35 s per run**, before the GPU turn.
9. **Stages.** `train.py` changed, so the six text stages stay stale, as expected; none was built.

---

# Fix round 1 (commit 4aceaa3)

## Changes

1. **The long-stroke claim, corrected.**
   - The report and `composite.py`'s docstring no longer say (b) teaches the long strokes the label rule leaves
     unknown. On the real maps no clean code or block number holds a run-rule px (the reviewer's count, NE held).
   - The synthetic test of that mechanism is renamed and says it is the mechanism only.
   - What composites teach is now counted per composite (`Bank._lessons`, into `Bank.tally` and `Bank.last`):
     - `new_lesson_px`: px that are text, known and line px. These are letter px on or next to line ink, which the
       control never teaches (it leaves every in-box line px unknown; a test asserts the control crop has none).
     - `box_line_not_text_px`: line ink inside the composited word's box taught as not text.
   - In (b) the pasted piece's px, grown by 1 px as `zones.find_lines` grows line px, join the crop's line px, so its
     overlaps count. A test rebuilds the piece from `last["piece_turns"]` and checks the count exactly.
2. **Kind weights**, with no new constants (`_mix_weights`, `_kind_weights`, `Bank.kind_weights`, `Bank.word_chances`,
   `Bank._pick`).
   - Per map, clean words are drawn with probability proportional to their kind's weight. One `rng.random()`
     replaces `rng.integers`.
   - For (a): the kind's share of the map's training-centre words ÷ its share of the bank's clean words.
   - For (b): its share of the crossed words ÷ its share of the clean words. Where no kind with clean words is crossed,
     (b) uses the (a) weights.
   - A kind with no clean words gets weight 0, and the other kinds keep their mix among themselves.
   - `Bank.weighted=False` draws alike (the old behaviour, kept for comparison).
   - `Bank.sizes()` reports the weights.
3. **Crossing counts line ink only.**
   - `crossings()[:, 1]` (line px that are ink, at least 3 inside the box) now drives `cross_by_kind`, the bank's
     `cross`/`crossing` and P. The old count is kept as `cross_by_kind_line_px`.
   - The (a) on-line acceptance counts `page.line & page.ink` inside the rectangle.
4. **`scripts/textlines/composites.py`** writes:
   - `cross_by_kind_line_px`;
   - the new-lesson count on each picture caption, and `picture_new_lessons`;
   - `sample`: 2,000 composites drawn as `train.draw` draws them (rng 21), weighted and alike. Each has the kind mix
     (in all, by (a)/(b), by map), repeats per map and kind, and new lessons (in all, by (a)/(b), by map and kind).
   - `distributions.json`, `composites.json` and `composites.png` were regenerated.

## Tests

```
uv run pytest -q tests/test_textnet_composite.py tests/test_textlines_variant.py tests/test_build.py tests/test_solver_textnet.py
120 passed in 30.19s
uv run pytest -q
1530 passed, 3 xfailed, 1 warning in 97.21s
```

- Three tests are new:
  - `test_kind_weights_draw_clean_words_in_the_words_mix_and_the_crossed_words_mix`: the weights reproduce the
    words' and the crossed words' mixes; a kind with no clean word gets 0; the fallback to the words' mix.
  - `test_the_bank_draws_its_clean_words_by_kind_weight`: exact chances on the fixture (codes 0.6 / block 0.2 / street
    0.2 for (a); codes only for (b)), 4,000 draws within 0.03 of them, and alike gives 0.25 each.
  - `test_new_lessons_are_text_px_on_line_px_which_the_label_rule_never_teaches`.
- `test_composites_zero_leaves_batches_unchanged` still matches the pin (`dd2f5a81…`), byte for byte.

## The new P

- Crossing on line ink only (NE and the answered blocks held out, 2000 training quarters):

  | | code | block | street | other | all |
  |---|---|---|---|---|---|
  | crossed / words | 404/772 | 593/3074 | 945/1775 | 73/313 | 2015/5934 |
  | share | 0.523 | 0.193 | 0.532 | 0.233 | **0.3396** |

- **P = 0.3396 for NE** (`--composites 0.3396`). For SW, 0.3507; nw 0.3481, se 0.3475.
- The old count (`cross_by_kind_line_px`) gave 0.3839.

## The kind mix of composites, before (alike) and after (weighted)

From 2,000 composites each.

| | code | block | street | other |
|---|---|---|---|---|
| before, all | 0.327 | 0.297 | 0.233 | 0.145 |
| after, all | 0.184 | 0.139 | **0.630** | 0.048 |
| after, (a) | 0.145 | 0.176 | 0.648 | 0.030 |
| after, (b) | 0.222 | 0.101 | 0.612 | 0.065 |
| 2000: before / after | 0.12 / 0.16 | 0.85 / 0.45 | 0.00 / 0.36 | 0.03 / 0.02 |
| 1980-90: before / after | 0.07 / 0.17 | 0.90 / 0.41 | 0.02 / 0.42 | 0.01 / 0.00 |
| 1990-96: before / after | 0.11 / 0.14 | 0.85 / 0.45 | 0.02 / 0.37 | 0.03 / 0.04 |
| 1960: before / after | 0.49 / 0.17 | – | 0.27 / 0.77 | 0.24 / 0.06 |
| 1964: before / after | 0.41 / 0.26 | – | 0.38 / 0.69 | 0.20 / 0.05 |
| 1968: before / after | 0.44 / 0.18 | – | 0.37 / 0.76 | 0.19 / 0.06 |

- The weights for (a) and (b) (NE held):
  - 2000: streets 22.2 and 34.8; codes 1.02 and 1.57; blocks 0.64 and 0.36.
  - 1990-96: streets 25.3 and 40.1.
  - 1960s maps: streets 2.3–3.1.
- Each modern map's composites now follow its own words' mix, as ruled.
- **But codes fall overall, from 33% to 18%.** The old maps supply 49 of the 87 sheets, and their words are mostly
  street names (1960: 2535 of 3452). So street names take 63% of all composites, and codes, the RM-2 failure's kind,
  lose share.

## Repeats per clean word (the weighted 2,000-composite sample)

| map, kind | bank words | draws | words drawn | mean per word | max per word |
|---|---|---|---|---|---|
| 2000, street | 28 | 82 | 27 | 2.93 | 6 |
| 1980-90, street | 41 | 86 | 37 | 2.10 | 7 |
| 1990-96, street | 21 | 75 | 21 | 3.57 | 8 |
| 1960, street | 82 | 320 | 82 | 3.90 | 10 |
| 1964, street | 146 | 284 | 125 | 1.95 | 6 |
| 1968, street | 150 | 413 | 138 | 2.75 | 10 |
| 2000, code | 265 | 36 | 33 | 0.14 | 2 |
| 2000, block | 1688 | 102 | 99 | 0.06 | 2 |

Alike (before), street names: 2000 0 draws; 1980-90 0.12; 1990-96 0.24 per word, at most 2; 1960 1.29, at most 5.

A NE run makes about 4000 × 12 × 0.3396 ≈ 16,300 composites, 8.15× this sample. Scaled up:

| map | mean draws per street name | most draws of one |
|---|---|---|
| 2000 | about 24 | about 50 |
| 1990-96 | about 29 | about 65 |
| 1960 | about 32 | about 80 |

Each draw has a different crop, place, scale, turn and line piece, but the same 21–28 modern street names' letter
shapes recur.

## New lessons (text, known and on or next to line px; page scale)

- The 24-tile picture:
  - 19 of 24 composites (79%) have some;
  - 2,602 px, 108 per composite, 18.7% of the composited words' ink;
  - 3,322 px of line ink in the word's box taught not text.
  - Per tile: (a) codes on lines 311, 455, 275; the "MONTEREY BLVD." street on a line 0 (the line runs between its
    letters); the (b) "AVE." under a heavy line 211.
- The weighted 2,000-composite sample:
  - 68.4% have some; 93 px each on average; 17.8% of the composited words' ink;
  - 240,121 px of in-box line ink taught not text.
  - (a): 38.6% have some (the on-line ones), 14.2% of word ink. (b): 97.7%, 20.2%.
- The alike sample: 69.1% have some, 15.6% of word ink.

**Beside Task 4's 25.9%.** Under the control, 25.9% of the in-box ink of 2000 codes is unknown. In the weighted
sample's 2000 code composites (36 of them), 26.7% of the composited code ink is taught as text on or next to line px.
For other kinds: 2000 streets 42.6%, 2000 blocks 32.7%, 1960 codes 12.2%.

These are different populations:

- Task 4's figure is real code words, where line px and long strokes are both unknown.
- The composites' figure is clean words given new line contact. That contact comes only from the run-rule lines
  ((a)-on) and pasted real pieces ((b)), never from long strokes.

## Timing (P 0.3396)

| | draw, ms a batch | draw + `batch_tensors`, ms a batch |
|---|---|---|
| composites at P | 33.0 | 37.7 |
| composites at 0 | 27.1 | 31.9 |

Still well under the 95 ms GPU step.

## The picture

`composites.png` was regenerated and looked at.

- Street names now show up and land on street lines:
  - "GEARY" set between blocks on a street line;
  - "SUNSET" on a heavy vertical line;
  - "ST" and "ST." on lot lines.
- Codes are crossed: "R-2", "R3", "RH-1".
- (b) pieces run through "AVE." (a heavy horizontal line), "SOTELO", "P", "R-5" and "R-3".
- In every tile the letters are red, the line px grey, box paper and held-out areas pale blue, and each caption ends
  with its new-lesson px.

## Concerns after the fix

1. **Codes lose share.** With the ruled weights, street names take 63% of composites and codes fall from 33% to 18%.
   The weighting follows each map's own words, and the old maps' words are mostly street names. If Task 7 wants codes
   (RM-2), that is a further choice; I made none.
2. **Repeats.** A full NE run would draw each modern street name about 24–29 times, and one up to about 65. Each
   placement differs, but the shapes are the same 21–41 words per modern map.
3. **New lessons stop at the line px.**
   - They are letter px at real run-rule lines or at pasted real pieces.
   - Lines the run rule misses (stair-steps, dashes) bring no new lessons in (a), and their px under a word stay as
     they were.
