# Task 13 report: map-like synthetic scenes (item 8), code only

Status: DONE_WITH_CONCERNS (code, tests, NE measurements and contact sheet; no GPU run, as dispatched).

## What I built

- `pipeline/src/sfzmaps/solver/wordread/scenes.py` (new; no stage imports it):
  - `Geometry`: DataSF geometry as arrays. Lots (`lots/table.parquet`: `block_num` and outline only, never a code),
    blocks (the union of each block's lots, 5,302), centrelines (`context.sf_streets`, 17,176) with their printed
    forms (`context.street_forms`). Three line kinds cut into segments: `block` (outline = street edges), `lot`
    (lot outlines less the block's), `street` (centrelines). Built in ~8 s, no cache needed.
  - Held out: `held_shapes` / `usable`. A quarter is a quadrant shape, an answered-blocks `GridArea` is its grid px
    grown by one px and traced through px centres (covers every held px wholly, so a line clipping a corner still
    meets it). A block, and every lot in it, is usable only when its outline meets no held shape; a centreline the
    same. A word is usable when no px of its rectangle is held (`data._touch_held`).
  - `pasteable(sd)`: the controller's ruling. A Terra word is pasteable when it lies wholly on its page and every
    8-connected ink piece touching its rectangle lies wholly inside it. Its own ink is those pieces. There is no
    `CLEAN_PX` ring. A word joined to a line is never pasted.
  - Measures, per map, on each sheet's training px of its printed map. The printed map is the warp's content
    footprint (`warped.footprint.geojson`, the neatline less the legend and mask boxes); I added this after the
    first measure counted geometry projected onto legends and blank margins.
    - `fit_near`: line presence. The line ink is ink outside every Terra word's and other pass's rectangle. The run
      rule's `line_bits` miss a block's short edges: they gave drawn share 0.42 against about 0.8.
      - Each skeleton px goes to the nearest projected kind, at distance d.
      - Mixture model: P(d | own) ~ n_b(d) exp(-d²/2σ²) against P(d | other) ~ n_b(d), where n_b(d) counts the
        page's training px at that distance.
      - One share a per kind; one σ for every kind (the placement is one). Fitted by EM from every decile start,
        plus the a = 1 and a = 0 ends.
    - `thickness_shares`: EM for the shares of `cv2.line` thicknesses that explain the own px' measured thickness
      (2 × distance-to-paper at the skeleton − 1). Each thickness's profile is measured by drawing it at 180
      angles. cv2 draws 3≡4, 5≡6, … identically, so only the distinct drawings are candidates.
    - Code offsets from the nearest lot's centroid, with a rate per lot.
    - Block-number offsets from the centroid of the block their text names (when inside it), with a rate per block.
    - Street names: the nearest centreline whose street prints the text gives a signed offset and an axis angle.
      `fit_far` is a half-normal (own) plus exponential (misread) EM from every decile start. The near rate is per
      ft of centreline. The angle density is a reflected KDE with a leave-one-out bandwidth
      (`context.loo_bandwidth`).
  - `SceneBank.build(sheets, areas, seed, ...)` and `SceneBank.crop(rng, size, scale, turns)`, also `.scene`, the name
    `train.draw` calls. A scene is built in these steps:
    1. Centre: drawn as real crops are (`train.TEXT_SHARE` on text, else ink; `data.centres` of the held-out sheets,
       on the printed map).
    2. Frame: the sheet's placement over the window as one affine map.
    3. Lines: block outlines and centrelines drawn whole with their kind's drawn share; lot lines per segment. Each
       thickness is drawn from the shares.
    4. Labels: block numbers at block centroid + offset, codes at lot centroid + offset, street names as a Poisson
       count per centreline at offset + angle weight. Each label is a real pasteable word of that string from that
       map; with none, the label is skipped. Labels are placed in random order at whole-px shifts (ink copied, never
       resampled); a label meeting a placed rectangle is skipped.
    5. Targets: `data.crop_targets(..., words=True)` of the scene as a `SheetData`, cores = rectangles, so scenes and
       real crops share every target builder. Then made exact: text = the pasted words' own ink only; a line px
       inside a rectangle is known not-text; the paper inside a rectangle is unknown (as the composites' `_exact`).
  - `Banks(composites, scenes)`: one bank object for both.
- `pipeline/src/sfzmaps/solver/textnet/train.py`: `Recipe.scenes = 0.0` (0..1; refused with `input="grey"`).
  `draw` draws a scene with that chance, only when it is above 0, after the composite chance, via `bank.scene(rng,
  crop, scale, turns)`. `fit` and `draw` refuse scenes without a bank (`_need_bank`). Defaults are byte-identical:
  no extra draw at 0, and `name()` is unchanged. Added on top of Task 12's commits 8b83a08 and 6c8de6b. `"scenes":
  0.0` is added deliberately to both pinned Recipe dicts (`test_solver_textnet.py`, `test_wordread_input.py`).
- `pipeline/scripts/wordread/run_scenes.py`:
  - `build --quarter q`: writes `measured.json`, `sheet.png`/`sheet.json` and `density.json` under
    `data/solver/experiments/wordread/scenes/<q>/`.
  - `bank_for`, `scene_recipe`, `measured_share`, and `scene_run(quarter, seed, share=None, ...)`: the run step for
    Task 8's runner.
  - The keep rule (Task 8's), written before any run, is in the module docstring.
- `pipeline/tests/test_wordread_scenes.py`: 17 tests, all on synthetic arrays except one real-data check that skips
  without `pipeline/data`.

## TDD evidence

- The recipe tests were written first and run red: `AttributeError: 'Recipe' object has no attribute 'scenes'`.
  Then `Recipe.scenes` was added and they went green: scenes=0 byte-identical (heads all and words, no extra rng
  draw); scale/turns the same as a composite's; refusals without a bank. GUARD_WEIGHTS and the composite pinned
  batches pass.
- For scenes.py I wrote a first draft of the module and then the tests, so the order was not strict TDD. The tests'
  first run failed on a real bug: the line fit returned a = 0 when every skeleton px sat on the line (no a = 1 end).
  The fix added the a = 1 and a = 0 ends.
- What the tests cover:
  - masks exact at 3 scale/turn settings, 12 scenes each:
    - text ⊆ ink, and text = the pasted own ink at scale 1;
    - the cores pairwise disjoint and core = tag;
    - the paper inside rectangles unknown;
    - kind and angle only on text;
    - the keys and dtypes those of `crop_targets(words=True)`;
  - the same seed gives the same scene (two banks built separately);
  - a held-out area never appears: a code printed only in NE is absent from the held bank and present without the
    areas; no word touches held px; 40 scenes draw no segment meeting a held shape and no label from a held
    block, lot or line; the measured code count is the outside words; the non-vacuity check is that without the
    areas, held lines are drawn;
  - only the printed map (footprint) is measured, pasted and centred on;
  - `pasteable` keeps own pieces and refuses a word joined to a line;
  - `fit_near` recovers a = 0.7 and 0.2 with a shared σ = 1.5;
  - `thickness_shares` recovers a 0.7/0.3 mixture;
  - `fit_far` separates misreads;
  - scenes feed `batch_tensors` + `word_tensors` with heads=words;
  - real geometry has every line kind.
- Suite before the commit: `uv run pytest -q -n 2 --ignore=tests/test_wordread_words.py` gave 1821 passed,
  3 skipped, 3 xfailed. GUARD_WEIGHTS, the composite pinned batches and `test_build` all pass.
  `test_wordread_words.py` (another agent's WIP) did not finish within 170 s on its own, so I left it out.

Commit: a7debd6 (scenes.py, train.py, run_scenes.py, test_wordread_scenes.py, and the two pinned Recipe dicts).

## Measured inputs (fold NE held out with the answered blocks; 87 training sheets of 6 maps; `ne/measured.json`)

Lines: a single spread σ for every kind per map (px): 2000 0.82, 1980 0.82, 1990 0.84, 1960 1.00, 1964 0.99,
1968 0.98.

| map | block drawn | lot drawn | street drawn | block thickness (cv2 t: share) | lot thickness | street thickness |
|---|---|---|---|---|---|---|
| 2000 | 0.789 | 0.035 | 0.087 | 1: .16, 2: .76, 3: .05, 5: .03 | 1: .10, 2: .19, 3: .26, 5: .43 | 1: .14, 2: .23, 3: .22, 5: .36 |
| 1980 | 0.807 | 0.033 | 0.090 | 1: .12, 2: .81, 3: .04, 5: .02 | 2: .20, 3: .29, 5: .41 | 2: .24, 3: .26, 5: .34 |
| 1990 | 0.794 | 0.034 | 0.092 | 2: .85, 3: .07, 5: .03 | 2: .23, 3: .13, 5: .52 | 2: .27, 3: .13, 5: .43 |
| 1960 | 0.721 | 0.028 | 0.063 | 2: .84, 3: .14 | 2: .42, 3: .13, 5: .41 | 2: .28, 3: .15, 5: .33, 9: .07 |
| 1964 | 0.736 | 0.029 | 0.080 | 2: .94, 3: .04 | 2: .43, 3: .22, 5: .32 | 2: .28, 3: .19, 5: .31 |
| 1968 | 0.754 | 0.025 | 0.059 | 2: .83, 3: .15 | 2: .43, 3: .18, 5: .35 | 2: .30, 3: .13, 5: .35 |

The own px' measured thickness, block outlines, 2000 (px): q1 1.0, q10 1.0, q25 1.8, q50 3.0, q75 3.0, q90 3.0,
q99 7.0. Lot lines (the district lines along lots) q50 5.0, q90 7.0, q99 9.0. Every map is in `measured.json`.

Labels (offsets in ft):

| map | code words / lots / per lot | code offset q10/q50/q90/q99 | block words / blocks / per block | block offset q10/q50/q90/q99 | street words, near share, near σ, near per 1000 ft | street offset q10/q50/q90 (signed) | angle q50/q90 (deg), bandwidth |
|---|---|---|---|---|---|---|---|
| 2000 | 431 / 99,680 / 0.0043 | 12.5 / 45.3 / 130 / 347 | 2,610 / 3,124 / 0.836 | 6.3 / 17.7 / 67 / 212 | 1,176, 0.906, 6.7, 0.41 | −7.0 / 1.2 / 8.5 | 1.3 / 5.5, 1.84 |
| 1980 | 445 / 99,441 / 0.0045 | 12.6 / 44.2 / 152 / 759 | 2,662 / 3,126 / 0.852 | 6.4 / 17.8 / 69 / 212 | 1,391, 0.900, 6.9, 0.48 | −7.4 / 0.8 / 8.5 | 1.4 / 5.4, 1.70 |
| 1990 | 354 / 88,301 / 0.0040 | 14.2 / 47.0 / 171 / 821 | 2,176 / 2,771 / 0.785 | 6.6 / 17.8 / 65 / 206 | 908, 0.883, 7.1, 0.34 | −7.7 / 0.7 / 8.5 | 1.0 / 5.0, 0.77 |
| 1960 | 406 / 132,452 / 0.0031 | 14.8 / 56.0 / 136 / 463 | 0 (no block numbers) | — | 1,742, 0.979, 33.7, 0.50 | −54.8 / 0.7 / 28.8 | 1.4 / 7.2, 0.92 |
| 1964 | 456 / 134,893 / 0.0034 | 12.6 / 53.5 / 118 / 651 | 0 | — | 1,809, 0.982, 34.8, 0.51 | −57.2 / −0.5 / 34.8 | 1.7 / 38.4, 0.50 |
| 1968 | 460 / 173,363 / 0.0027 | 13.5 / 55.8 / 109 / 644 | 0 | — | 2,087, 0.983, 35.0, 0.51 | −57.5 / −0.7 / 27.4 | 1.6 / 34.8, 0.49 |

Pasteable words over usable (on the printed map, outside held areas):

| map | pasteable / usable | codes | block numbers | street names |
|---|---|---|---|---|
| 2000 | 2,418 / 5,131 | 81 | 2,075 | 246 |
| 1980 | 2,505 / 5,524 | 77 | 2,133 | 278 |
| 1990 | 2,096 / 4,237 | 68 | 1,816 | 198 |
| 1960 | 515 / 2,809 | 122 | 0 | 374 |
| 1964 | 707 / 2,993 | 142 | 0 | 537 |
| 1968 | 821 / 3,463 | 147 | 0 | 636 |

The share of scene crops (`density.json`, 2,000 crops each at crop 320 with the training's scales and turns):

| | words per crop: mean ± se | q50 | q90 |
|---|---|---|---|
| real crops | 7.69 ± 0.20 | 5 | 19 |
| scenes | 4.04 ± 0.11 | 2 | 10 |

The written share is the pasteable share of usable words, 9,062 / 24,157 = **0.375** (see concern 2).

Scene tally over 2,012 scenes:
- labels placed: block 3,119, code 1,379, street 3,654;
- skipped for no pasteable word of that string: block 31,955, street 12,703;
- skipped for meeting a placed word: 87, 173, 266.

The bank build takes 6.8 min on CPU; it is effectively one core (the GIL), so the threads add little.

## Contact sheet

`pipeline/data/solver/experiments/wordread/scenes/ne/sheet.png`, with `sheet.json`. It shows 12 scenes drawn as
training draws them (random scale 0.6–1.6, quarter turns). Each tile is two panels:
- left: ink in black, text in red, line ink inside a word's rectangle (known not-text) in blue, the unknown paper
  inside rectangles in pale yellow;
- right: cores coloured by word, with a white dot at each word's centre as its core px' geometry puts it.

What the sheet shows:
- Whole 1-bit block outlines (mostly 2-px). Street ground reads as the paired outlines of facing blocks, as on the
  maps. There are a few heavy short dashes along lot lines (the measured district-line rate, about 3% of lot-line
  length) and rare heavy centrelines.
- Real words at their own size and texture: block numbers inside blocks (4759, 1092, 5286A; upside down where the
  turn is 2), "ST."/"AVE."/street names along streets at the street's angle, and codes (R-5, R-3, P).
- Each text mask is exactly the pasted ink (red), never the paper; where a line crosses a word its px are blue (PALOU in
  scene 2, cut at the window's edge).
- The core dots sit at the rectangle centres, so the geometry targets are consistent.
- 5 of 12 scenes hold 0 words. These are 1960s windows (no block numbers on those maps; street words are sparse
  per crop) and one 2000 window in a park/blank area. The scenes are visibly sparser in words than real crops,
  which matches `density.json`.
- An earlier build drew every segment independently. Outlines came out dashed and centrelines drew through street
  names (that fit's street "own" spread covered the whole street). Both are fixed above, and the sheet shows it.

## Concerns

1. **`scene_run` cannot run until `variant.run` accepts a bank for a recipe with scenes.** Today `variant.run` raises
   "a bank is for a recipe with composites" when `recipe.composites == 0`. It needs a one-line change to
   `bank is not None and not (recipe.composites or recipe.scenes)`. That is Task 8's file, so I did not touch it.
   `run_detector.train_run` also passes no bank.
2. **The share "measured to match real word density" is not pinned by density alone.** Scenes centre as the real
   crops do and still hold 4.0 words per crop against 7.7. Most block numbers have no pasteable twin (31,955
   skipped), and the 1960s maps print no block numbers. So the batch's word density is (1−p)·7.7 + p·4.0, and only
   p = 0 matches 7.7. I wrote p = the pasteable share of usable words (0.375) as a measured stand-in. Please rule.
   - Option (a): keep 0.375.
   - Option (b): centre a measured share of scenes on a rendered label so that scene word density equals the real
     one. That reading is "the share of scene crops [centred on text] to match"; it needs rendering a larger
     window first.
   - Option (c): screen p on the held-out outcome as a setting.
3. **Lot lines are drawn segment by segment.** The maps draw district lines as runs along lot lines, so the scenes
   show short heavy dashes at the measured length share. A district model (polygons along lot lines) would be more
   faithful.
4. **The centreline share (0.06–0.09) may still include some untranscribed street text.** The shared σ forces it to
   be a line within about 1 px of the centreline. The first per-kind fit had σ 17–21 px and share 0.37–0.40.
5. Block outlines are drawn on 72–81% of their length. The rest is DataSF blocks not printed as such (parks, new
   blocks, curved blocks that are misregistered). Scenes draw an outline whole or not at all, so about 20% of blocks
   get no outline.
6. The 1964/1968 street-name angle q90 is about 35°: some street words sit across their street. This is measured
   and kept.
7. The KDE bandwidth search (`context.loo_bandwidth`) warns about dividing by zero on the synthetic test city
   (duplicate angles). The real maps give finite bandwidths of 0.49–1.84°.
8. Scenes carry no `weight` key (they are exact, weight 1), so mixing them with doubt-weighted real crops is fine in
   `word_tensors`.

## Fix round 1 (the controller's rulings on sheet.png)

### What changed

1. **Stand-in words when no word of the string exists.**
   - A block-number or street-name label whose string has no pasteable twin now takes a pasteable real word of the
     same kind from that map, at the measured offset and its own printed size. A street name's stand-in is drawn by
     the same angle density at its angle to the centreline. (Codes were already drawn from the whole code pool.)
   - Each label records `standin` (`last["labels"]`, and `*` in `sheet.json`). The tally counts placed labels as
     "its own string" or "stand-in".
   - A label is skipped only when its kind has no pasteable word at all. There are none for block numbers on the
     1960s maps, which print none, and their block rate is 0 anyway.
   - Street labels now go only on named centrelines. The street rate is measured per ft of named centreline; before
     this fix, unnamed lines took draws too.
   - Placed labels over 2,012 scenes:

     | | own string | stand-in | stand-in share |
     |---|---|---|---|
     | blocks | 2,816 | 2,031 | 42% |
     | streets | 3,693 | 403 | 10% |
     | codes | 1,307 (any code word, no string) | — | — |

   - "Skipped: no word" is gone (it was 31,955 block and 12,703 street proposals).
2. **Lines are measured and drawn by whole edge.**
   - An edge is a straight run between the vertices a page can show (`page_edges`). DataSF puts a vertex at every lot
     corner along a block's straight side: 612,549 outline vertices, half within 0.003 ft of the line, 90% within
     0.06 ft. A vertex within `SIMPLIFY_PX` = 0.5 px of the line through its neighbours is dropped, so a block's
     straight side is one edge. 0.5 px is the page's grid, not a decision about the data, but it is a constant:
     please confirm it is acceptable.
   - Measure: each training px's nearest edge (`cv2.distanceTransformWithLabels`, checked against brute force: 99.75%
     agree, the rest are ties).
     - The line-ink mixture is as before, with one σ.
     - Each edge's own ink px are counted out of its px.
     - `fit_edges` then fits the share of a kind's edges that are drawn: a binomial mixture of drawn and not-drawn
       edges, EM from every decile start.
   - Scenes draw each edge whole or leave it out with that share, at a thickness drawn from the shares. The per-item
     "whole outline" rule of round 0 is gone.
   - Per-edge shares, with the length shares for comparison:

     | map | block edges: by edge (by length) | lot | centreline | σ px |
     |---|---|---|---|---|
     | 2000 | 0.716 (0.765) | 0.068 (0.050) | 0.240 (0.093) | 0.85 |
     | 1980 | 0.732 (0.783) | 0.065 (0.049) | 0.228 (0.096) | 0.85 |
     | 1990 | 0.716 (0.772) | 0.068 (0.049) | 0.256 (0.097) | 0.88 |
     | 1960 | 0.592 (0.700) | 0.048 (0.041) | 0.180 (0.066) | 1.03 |
     | 1964 | 0.605 (0.721) | 0.049 (0.039) | 0.194 (0.083) | 1.01 |
     | 1968 | 0.621 (0.737) | 0.041 (0.035) | 0.181 (0.061) | 1.01 |

     The drawn edges' ink coverage (`p_on`) is 0.85–0.89 for blocks, 0.57–0.66 for lots and 0.33–0.41 for
     centrelines.
3. **The share of scene crops is not ruled.** `Recipe.scenes` stays a parameter, and `scene_run(quarter, seed,
   share)` now takes the share as an argument, with no default. `density.json` records `pasteable_share` = 0.375
   (9,062 of 24,157 usable words) as a measurement only.
4. `variant.run`'s bank line is untouched. It still has to accept a bank with scenes before the scene run.

### Realised words per crop (crop 320 px, the training's scales and turns; mean ± se, crops)

| map | real | scene |
|---|---|---|
| all | 7.63 ± 0.19 (2,000) | 5.08 ± 0.14 (2,000); round 0 was 4.04 |
| 2000 | 13.71 ± 0.71 (251) | 11.64 ± 0.51 (204) |
| 1980 | 15.88 ± 0.82 (202) | 10.51 ± 0.46 (268) |
| 1990 | 12.77 ± 0.73 (216) | 9.86 ± 0.46 (209) |
| 1960 | 4.54 ± 0.21 (403) | 2.15 ± 0.12 (389) |
| 1964 | 4.58 ± 0.20 (410) | 2.40 ± 0.12 (412) |
| 1968 | 4.13 ± 0.18 (518) | 2.11 ± 0.09 (518) |

The 1980–2000 maps are now close; the 1960s scenes hold about half the real words. On those maps the scenes'
labels are only codes and street names. The real crops also hold the maps' "other" words, park and place names,
which scenes don't place: no placement for them is measured.

### Contact sheet, regenerated (`pipeline/data/solver/experiments/wordread/scenes/ne/sheet.png`)

What changed against round 0:
- **Words.** The 1980–2000 scenes are full of labels. Scene 3 (1980) holds 12, scene 9 (1990) 14 and scene 10
  (1960) 11, including stand-ins (6795D*, 7555*, 3106*, 1309*, 4314*, 2079* as block numbers; FULTON*, DAY*, ST* as
  street names). They sit where labels belong: numbers in blocks, names along streets at the street's angle. Every
  text mask is still only the pasted ink, and the core dots sit at the rectangle centres.
- **Lines.**
  - Block sides are now left out whole: scenes 2 and 4 show blocks missing a full side, not dashed outlines.
  - Lot lines appear as whole lot-line edges: short where DataSF's edge is short, like the 25-ft spine pieces in
    scene 0, and full lot side lines elsewhere. They are drawn mostly heavy (3–5 px), as the maps draw district
    lines.
  - Centreline edges now show at 18–26% of edges. They are mostly short edges near intersections and curves, so
    scenes 3, 9 and 10 have heavy pieces in the street (see the first concern below).
- **Zero words.** 4 of 12 scenes hold none (1960 ×3, 1964 ×1), down from 5. All are 1960s windows where the codes
  are sparse (0.003 per lot) and the drawn streets have few names.

### Concerns (round 1)

- **Centrelines measured by edge come out 2.5–3× their length share.** Their drawn edges are only 33–41% inked
  (`p_on`). The likely cause is short edges at intersections, where a crossing street's district line or outline
  ink lies within reach; the binomial mixture then reads overdispersed partial coverage as "drawn". If that is
  wrong for training, a length-weighted edge share or an overdispersed (beta-binomial) mixture are the measured
  alternatives. I have not chosen between them.
- `SIMPLIFY_PX` = 0.5 px is the page's px grid; please confirm it is acceptable.
- The 1960s maps' scene word density is about half the real one.
- In `density.json`'s tally, the code labels' key reads "its own string"; for codes it means "any code word".

### Tests

- 20 scene tests. New ones cover:
  - same-kind stand-ins (a stand-in never crosses kinds, is recorded, and is used only when no twin exists);
  - `fit_edges` recovers a 0.7 drawn share and p_on 0.9;
  - a straight side with a vertex at every lot is one edge on the page.
- The held-out test now checks each drawn edge's item.
- Suite: `uv run pytest -q -n 2 --ignore=tests/test_wordread_words.py` gave 1829 passed, 3 skipped, 3 xfailed. I
  left out `test_wordread_words.py` (another agent's WIP) because it was slow last round: it did not finish in
  170 s.

Commit: 7b755ce (scenes.py, run_scenes.py, test_wordread_scenes.py only).

## Fix round 2 (the controller's rulings)

### What changed

- **`SIMPLIFY_PX` = 0.5 px.** The code now says why: a vertex less than half a px off the line through its
  neighbours can't make a visible corner at the page's resolution, a physical limit.
- **Every word's box is out of the line measure.** All Terra words' and other passes' boxes, the box itself grown by
  nothing, are now removed from the measuring area as a whole: from the edges' px (their length), the background
  n_b(d), and the skeleton. Before, only the ink inside boxes was out, so an edge's px inside a box counted as
  un-inked.
- **Lines are drawn by measured per-edge fractions.** Each edge's inked fraction is its own ink px over its px
  outside boxes, at most 1 (`inked_fractions`). A scene draws each edge's fraction from its kind's measured
  distribution (one value per measured edge, resampled), as one contiguous run at a uniform position along the edge
  (`edge_runs`). This replaces the round-1 binomial "whole edge or none" mixture (`fit_edges`, now removed).
- **Codes' tally key** is now `label:code:any code word` (and `proposed:code:any code word`).

### The centreline test

With every word's box excluded, the per-edge inked fraction on NE-held training quarters:

| map | kind | edges | at 0 | q50 | q75 | q90 | q99 | mean per edge | share by length |
|---|---|---|---|---|---|---|---|---|---|
| 2000 | block | 33,722 | 0.08 | 0.91 | 0.97 | 1.00 | 1.00 | 0.755 | 0.826 |
| 2000 | lot | 172,678 | 0.68 | 0.00 | 0.05 | 0.22 | 0.86 | 0.079 | 0.058 |
| 2000 | **street** | 19,541 | **0.43** | **0.00** | 0.16 | 0.43 | 0.88 | **0.117** | 0.118 |
| 1980 | block | 35,004 | 0.08 | 0.90 | 0.97 | 1.00 | 1.00 | 0.752 | 0.826 |
| 1980 | lot | 175,122 | 0.68 | 0.00 | 0.05 | 0.18 | 0.86 | 0.073 | 0.055 |
| 1980 | street | 19,727 | 0.43 | 0.00 | 0.16 | 0.45 | 0.96 | 0.119 | 0.119 |
| 1990 | block | 28,937 | 0.09 | 0.88 | 0.96 | 1.00 | 1.00 | 0.719 | 0.804 |
| 1990 | lot | 154,414 | 0.70 | 0.00 | 0.05 | 0.19 | 0.84 | 0.073 | 0.054 |
| 1990 | street | 17,386 | 0.40 | 0.00 | 0.16 | 0.41 | 0.85 | 0.115 | 0.116 |
| 1960 | block | 56,600 | 0.17 | 0.74 | 0.92 | 0.96 | 1.00 | 0.566 | 0.727 |
| 1960 | lot | 248,741 | 0.65 | 0.00 | 0.04 | 0.11 | 0.80 | 0.054 | 0.043 |
| 1960 | street | 26,729 | 0.51 | 0.00 | 0.06 | 0.32 | 0.82 | 0.084 | 0.076 |
| 1964 | block | 57,855 | 0.17 | 0.77 | 0.93 | 0.97 | 1.00 | 0.585 | 0.745 |
| 1964 | lot | 254,521 | 0.66 | 0.00 | 0.04 | 0.11 | 0.81 | 0.053 | 0.041 |
| 1964 | street | 27,432 | 0.55 | 0.00 | 0.08 | 0.38 | 0.95 | 0.098 | 0.093 |
| 1968 | block | 67,136 | 0.15 | 0.79 | 0.94 | 0.97 | 1.00 | 0.599 | 0.760 |
| 1968 | lot | 321,640 | 0.68 | 0.00 | 0.04 | 0.10 | 0.79 | 0.047 | 0.037 |
| 1968 | street | 30,998 | 0.57 | 0.00 | 0.05 | 0.30 | 0.85 | 0.079 | 0.069 |

The 10-bin histograms are in `measured.json`. 2000 centrelines:
13,492 / 1,786 / 1,203 / 889 / 697 / 754 / 385 / 94 / 55 / 186.

- **Centrelines are not near 0 on the 1980–2000 maps once text is out.** About 12% of each edge is inked on
  average, and 43% of edges hold none.
- **I looked at what that ink is** (display only, `ne/centreline_ink_2000_p08.png`: centrelines green, block
  outlines blue, centreline-near skeleton ink red). On the 2000 map it is mostly the heavy district boundaries,
  which the 2000 map draws down street centres (e.g. along 20th St and between blocks 4105 and 4169), plus the
  freeway and railway lines (I-280), and a few paths. It is not text: the street names' boxes are out.
- **Only the block outlines' distribution is two-humped:** about 70% of edges at ≥ 0.8 and 8–17% at 0, with a thin
  spread between. Lots and centrelines are mostly 0, with a long thin tail.

### Realised against measured (2,000 scenes; realised = inked px / edge px drawn, by kind and map)

| map | block realised / measured mean | lot | street |
|---|---|---|---|
| 2000 | 0.758 / 0.755 | 0.080 / 0.079 | 0.119 / 0.117 |
| 1980 | 0.752 / 0.752 | 0.073 / 0.073 | 0.119 / 0.119 |
| 1990 | 0.720 / 0.719 | 0.073 / 0.073 | 0.113 / 0.115 |
| 1960 | 0.568 / 0.566 | 0.054 / 0.054 | 0.081 / 0.084 |
| 1964 | 0.578 / 0.585 | 0.053 / 0.053 | 0.095 / 0.098 |
| 1968 | 0.599 / 0.599 | 0.046 / 0.047 | 0.079 / 0.079 |

Words per crop, real / scene:

| map | real | scene |
|---|---|---|
| all | 7.23 | 5.23 |
| 2000 | 12.9 | 11.9 |
| 1980 | 15.9 | 11.2 |
| 1990 | 11.9 | 8.9 |
| 1960s | 4.1–4.7 | 2.2–2.3 |

Labels placed: blocks 2,972 own string and 1,996 stand-ins; streets 3,751 and 448; codes 1,402, all "any code
word".

### Contact sheet (`ne/sheet.png`): what changed

- **Block outlines** are now mostly whole, with some partial runs (a side inked over part of its length).
- **New, and wrong: specks inside blocks.** Every lot edge with a small measured fraction (0.01–0.2) now prints a
  short run at a heavy thickness. These show as scattered black dots throughout blocks (scenes 1, 3, 4, 6, 7, 11)
  and along outlines. The real maps' blocks are clean.
- **I tested the cause** on one 2000 sheet (display-only diagnostic, PCA of the skeleton in a 7×7 neighbourhood,
  angle to the nearest edge):

  | kind | small-fraction edges (0 < f ≤ 0.2): px > 45° to the edge | large-fraction edges (f ≥ 0.8) |
  |---|---|---|
  | lot | **93%** | 8% |
  | block | 56% | 2% |
  | street | 56% | 4% |

  So the small fractions are mostly ink crossing the edge (district lines and outlines running across lot lines),
  not part of the edge drawn. The distance-only mixture counts a crossing line's px near the crossing as the edge's
  own.

### Proposed (not built; please rule)

Add the skeleton's local direction to the line mixture's likelihood. Own ink runs along its edge, so its angle to
the edge has a measured narrow distribution; other ink's angle is anything, its distribution measured too. Both are
fitted by the same EM with d. The crossing px then fall to "other", the small fractions shrink towards 0, and the
lot and street distributions should become the two-humped "drawn or not" the maps show.

The local direction needs a neighbourhood to estimate from. The 3×3 skeleton neighbours are the least one (coarse
angles), and a larger one is an estimator scale. That would be the only new constant; please say which.

Until then, the scenes in this round print crossing specks at the measured rate. Training on them would teach that
blocks hold scattered heavy dots, so this round's scenes should not be trained on as they are.

### Tests

- 20 scene tests.
- New: `edge_runs` (a run is the fraction's length, lies on the edge, a whole fraction is the whole edge) and
  `inked_fractions`.
- Changed: the synthetic city's centrelines measure about 0 inked even though its street names sit on them, because
  their boxes are out of the measure; the codes' tally key is checked.
- `fit_edges` and its test are removed.
- Suite: `uv run pytest -q -n 2 --ignore=tests/test_wordread_words.py` gave 1830 passed, 3 skipped, 3 xfailed.
  `test_wordread_words.py` is another agent's WIP and was slow, so I left it out. No `sfzmaps` build was running.

Commit: af3cd9a (scenes.py, test_wordread_scenes.py).

## Fix round 3 (the controller's ruling: orientation at the measured stroke width)

### What changed

- **Orientation.** Each skeleton px's local orientation is the structure tensor's (`_orientation_gap`). The Scharr
  gradients of the line ink have their products integrated by a Gaussian whose width (2σ) is the measured stroke
  width of the px's line kind on that map, so the window reaches both sides of the stroke and no further. The widths
  come from a first pass over the sheets by distance alone (`line_widths`: the median thickness, weighted by the
  distance responsibilities):
  - 2000 / 1980: block 3, lot 4.7, street 5 px;
  - 1990: block 3, lot 5, street 5 px;
  - 1960s: block 3, lot 3, street 5 px.

  The second pass measures each skeleton px's angle to its nearest edge in tenths of a degree. I first used σ =
  width; on two 2000 sheets it gave the block ink κ 442, which dropped px at 5° (P(own) 0.28). σ = width / 2 gives
  κ 177 and P(own) at 5° of 0.95. Both are "the stroke width as the scale"; please confirm σ = width / 2.
- **Mixture** (`fit_angles`), per kind and map, on the px weighted by their distance responsibility:
  - own ink: a von Mises about 0 on the axis, density exp(κ cos 2θ) / ((π/2) I0(κ)) on [0, 90°], κ fitted;
  - crossing ink: uniform on [0, 90°];
  - the own share fitted, by EM from every decile start and from "all own", the likeliest kept.

  A px counts toward its edge as distance responsibility × P(own | angle), with no cut.
- **Specks: a second cause, found from the sheet.** The angle term left about 30% of lot edges with tiny nonzero
  fractions (around 1e-6), and any fraction above 0 drew a thick dot. A run's px are now a whole number whose
  expectation is fraction × length (`run_px`): the whole px, plus one with the chance of the remainder. A run of a
  hundredth of a px inks a px one time in a hundred, as the measure counts it.

### Fitted angle mixture and the new per-edge inked fractions (NE held out; per map, kind)

| map | kind | width px | own share | κ | own sd (°) | edges at 0 | q50 | q75 | q90 | q99 | mean per edge | by length |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2000 | block | 3.0 | 0.878 | 96 | 2.9 | 0.08 | 0.72 | 0.87 | 0.94 | 1.00 | 0.587 | 0.732 |
| 2000 | lot | 4.7 | 0.328 | 186 | 2.1 | 0.68 | 0 | 0 | 0 | 0.76 | 0.033 | 0.019 |
| 2000 | street | 5.0 | 0.708 | 132 | 2.5 | 0.43 | 0 | 0.006 | 0.25 | 0.65 | 0.062 | 0.083 |
| 1980 | block | 3.0 | 0.895 | 85 | 3.1 | 0.08 | 0.75 | 0.88 | 0.94 | 1.00 | 0.602 | 0.745 |
| 1980 | lot | 4.7 | 0.342 | 182 | 2.1 | 0.68 | 0 | 0 | 0 | 0.74 | 0.031 | 0.019 |
| 1980 | street | 5.0 | 0.730 | 124 | 2.6 | 0.43 | 0 | 0.007 | 0.27 | 0.71 | 0.066 | 0.087 |
| 1990 | block | 3.0 | 0.867 | 104 | 2.8 | 0.09 | 0.67 | 0.85 | 0.92 | 1.00 | 0.545 | 0.701 |
| 1990 | lot | 5.0 | 0.317 | 201 | 2.0 | 0.70 | 0 | 0 | 0 | 0.73 | 0.029 | 0.017 |
| 1990 | street | 5.0 | 0.668 | 134 | 2.5 | 0.40 | 0 | 0.008 | 0.23 | 0.61 | 0.057 | 0.077 |
| 1960 | block | 3.0 | 0.911 | 115 | 2.7 | 0.17 | 0.51 | 0.83 | 0.91 | 1.00 | 0.442 | 0.664 |
| 1960 | lot | 3.0 | 0.236 | 216 | 2.0 | 0.65 | 0 | 0 | 0 | 0.59 | 0.018 | 0.010 |
| 1960 | street | 5.0 | 0.723 | 129 | 2.5 | 0.52 | 0 | 0 | 0.13 | 0.67 | 0.045 | 0.055 |
| 1964 | block | 3.0 | 0.914 | 113 | 2.7 | 0.17 | 0.55 | 0.84 | 0.92 | 1.00 | 0.459 | 0.683 |
| 1964 | lot | 3.0 | 0.261 | 234 | 1.9 | 0.66 | 0 | 0 | 0 | 0.62 | 0.019 | 0.011 |
| 1964 | street | 5.0 | 0.734 | 153 | 2.3 | 0.55 | 0 | 0 | 0.16 | 0.73 | 0.053 | 0.068 |
| 1968 | block | 3.0 | 0.916 | 125 | 2.6 | 0.15 | 0.58 | 0.85 | 0.92 | 1.00 | 0.471 | 0.698 |
| 1968 | lot | 3.0 | 0.226 | 249 | 1.8 | 0.68 | 0 | 0 | 0 | 0.58 | 0.015 | 0.008 |
| 1968 | street | 5.0 | 0.731 | 142 | 2.4 | 0.57 | 0 | 0 | 0.11 | 0.64 | 0.043 | 0.051 |

Against round 2:

| map | lot q90 | lot mean | street q90 | street mean | block mean per edge |
|---|---|---|---|---|---|
| 2000 | 0.22 → 0 | 0.079 → 0.033 | 0.43 → 0.25 | 0.117 → 0.062 | 0.755 → 0.587 |
| 1980 | 0.18 → 0 | 0.073 → 0.031 | 0.45 → 0.27 | 0.119 → 0.066 | 0.752 → 0.602 |
| 1960 | 0.11 → 0 | 0.054 → 0.018 | 0.32 → 0.13 | 0.084 → 0.045 | 0.566 → 0.442 |

The block outlines lose some genuine ink. The px within a stroke's width of a corner, a curve's vertex or a
junction turn with it, and the von Mises (sd 2.6–3.1°) gives them to crossing.

### Realised against measured mean per edge (2,000 scenes; inked px / edge px)

| map | block | lot | street |
|---|---|---|---|
| 2000 | 0.592 / 0.587 | 0.033 / 0.033 | 0.061 / 0.062 |
| 1980 | 0.598 / 0.602 | 0.031 / 0.031 | 0.064 / 0.066 |
| 1990 | 0.546 / 0.545 | 0.029 / 0.029 | 0.056 / 0.057 |
| 1960 | 0.439 / 0.442 | 0.018 / 0.018 | 0.045 / 0.045 |
| 1964 | 0.461 / 0.459 | 0.019 / 0.019 | 0.052 / 0.053 |
| 1968 | 0.471 / 0.471 | 0.015 / 0.015 | 0.041 / 0.043 |

Words per crop, real / scene:

| map | real | scene |
|---|---|---|
| all | 7.59 | 5.22 |
| 2000 | 14.0 | 12.1 |
| 1980 | 16.1 | 10.7 |
| 1990 | 12.4 | 10.9 |
| 1960s | 4.2–4.6 | 1.9–2.3 |

### Contact sheet (`ne/sheet.png`)

**The specks are gone.** The blocks' interiors are clean (scenes 4, 6, 7, 8). What remains inside blocks is
occasional short heavy runs along lot lines, the measured district lines, and rare centreline pieces.

What looks wrong now:
- **Block outlines show gaps.** About 55–60% of each 1980–2000 outline edge and 44–47% on the 1960s maps is
  drawn, as a run at a random position, so outlines are visibly broken: scenes 1, 2 and 11 are 1960s windows with
  sparse broken lines, and scene 2 is nearly blank (a park). Before the angle term the mean was 0.72–0.76 and
  0.57–0.60. The loss is the corners, curves and junctions above.
- **Words.** Scenes 3, 7 and 10 hold 37, 13 and 23 labels. Text masks are still only the pasted ink, and the core
  dots sit at the rectangle centres.

### Concerns (round 3)

- The von Mises "own" component is too narrow for real outline ink at corners and curves (sd about 3°). Outlines
  now lose about 15 points of coverage, and the gap is placed anywhere along the edge, where the maps show
  continuous outlines. A heavier-tailed own component (e.g. a von Mises mixture, or a wrapped Cauchy) would keep
  them. I haven't changed it, as the ruling named the von Mises.
- σ = width / 2 against σ = width: please confirm.
- The synthetic test city's outlines are perfectly straight, so their angle is exactly 0. The von Mises κ runs to
  float64's limit there, and corner px fall to crossing: a fully printed synthetic outline measures about 3/4.
  The test states this.
- The round-2 centreline picture is now at `ne-round2/centreline_ink_2000_p08.png`.

### Tests

- 22 scene tests. New:
  - `fit_angles` recovers own share 0.7 and κ 30 against uniform crossings;
  - crossing ink counts toward a crossed lot edge by distance alone (fractions 0.07 and 0.14) and not with the
    orientation (below 0.05). This needed a crossing line and a printed lot line added to the synthetic city.
  - `run_px` keeps the expectation, and a tiny fraction inks a px rarely.
- Suite: `uv run pytest -q -n 2 --ignore=tests/test_wordread_words.py` gave 1823 passed, 7 skipped, 3 xfailed. I
  left out `test_wordread_words.py` (another agent's WIP, slow before). No `sfzmaps` build was running.

Commit: cd9bc5f (scenes.py, test_wordread_scenes.py).
