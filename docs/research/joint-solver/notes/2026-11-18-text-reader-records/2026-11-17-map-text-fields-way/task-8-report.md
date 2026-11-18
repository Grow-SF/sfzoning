# Task 8 report: the detector experiments

Status (19:50): **crude-map lifts the full headline from 286/699 (0.409) to 343/699 (0.491), paired +0.082
[+0.037, +0.126]; NE +0.175 [+0.113, +0.239]** (section "crude-map" below). crude-map is a lower-bound check, not the
word formation: components of P(core) > 1/2, no tags, no votes, so touching words merge. **crude-aff** (the learned
same-word affinity, arm a″, with the measured class prior) gives NE 144/285, +0.014 [+0.003, +0.028] over crude-map on
the same detector: the affinity does not part touching words. **Merged boxes are 227 of crude-map's 356 wrong labels**
(today: 206 of 413; no box 78 → 14, partial 35 → 3); code + block number is 137 of them. crude-tags was stopped before
forming words (the owner prefers learned word identity). The round-5 word formation chain was stopped on NE (the
controller: crude chains are the path now). Arms a′ and a″ (centre fix) learn the centre within the reach.

Earlier status (kept for the record): Phase 1 was blocked at word formation (code committed 2e3964b, test fix
84639d8): the word heads' geometry did not learn (the reading direction's sign never did), so the measured spreads
were degenerate (distance sd 274 px) and `core_words`' proposals merged a page into a handful of basins (p02's first
round: 4 proposals for the whole page in 20 min). The controller's ruling (sign-free heads, Task 5) fixed the axis.

## What I built (Phase 1)

Files (mine): `pipeline/scripts/wordread/run_detector.py`, `pipeline/tests/test_wordread_run_detector.py`,
`pipeline/scripts/textlines/variant.py` (options), `pipeline/src/sfzmaps/solver/textnet/predict.py` (a new function;
the default paths unchanged), `pipeline/src/sfzmaps/solver/wordboxes.py` (the shared placement helper, outputs
unchanged). Also, for the carried items and the page: `textnet/train.py` (the IoU guard) with its pinned test in
`tests/test_wordread_heads.py`, and default-off options in `wordread/review.py`.

- `predict.word_outputs(net, ink, device, only=None, tile=TILE)` -> `WordOutputs(chance, kind_p, core, geo, tag)`:
  the arrays `core_words` takes, the page's tiles stitched. Each tile is seen with `predict.context(net)` px of page
  around it: the network's whole reach (its receptive field, from the architecture: 128 px full size, 256 px at half
  size), not PAD (64). So every px gets the numbers one pass over the whole page gives it (tested against a single
  pass, full and half size), and a word on a tile's edge has one word's votes on both sides. The reach formula is
  tested against the gradient footprint of an impulse. A px not scored has core logit `NOT_SCORED` (-1e4: chance 0
  in double precision, finite). `chance` and `outputs` are untouched.
- `wordboxes.placement(sheet)` and `wordboxes.place(centres, sheet, transform, footprint, lots, mapped)`: factored
  out of `sheet_boxes` (which now calls them). Checked: `sheet_boxes("2000-digital-zoning-p07")` gives identical
  frames at every level, off-map counts and word tables before and after (assert_frame_equal, exact).
- `variant.py`: `--heads words`, `--doubt`, `--root`; with the word heads the training sheets get their cores
  (`shrink.prepare`) and, with doubt, the quarter's "box" weights (`doubt.prepare`); after the chance, in the same
  GPU turn, every 2000 page's word outputs go to `words/<sheet>.npz` (`save_word_outputs`: core, tag, kind_p in
  float16, geo in float32 since a distance can pass float16's range). `run.json` gains `doubt` and `mps_peak`
  (`variant.Peak`, a sampling thread on `torch.mps` allocator counters; a record only).
- `run_detector.py`: `check`, `train`, `fit`, `words`, `read`, `measure`, `page` (see its docstring).
  - `fit`: spreads by `words.measure_spreads` on windows of the fit's area (`fit_area`: the allowed quarters by each
    px's map point, minus the answered blocks; `windows`: an exact quad-tree cover of that mask), then
    `words.prior_decisions` on whole pages with every px outside the area not a word's (core `NOT_SCORED`, word id
    0: such votes weigh exactly 0 and their background terms cancel in every gain), a first prior on the "word"
    decisions, then all decisions with it, refitted (`words.fit_word_prior`). Phase 1 `--own`; Phase 2 `--from`
    other folds' runs, each read on its own held-out quarter only.
  - `words`: `core_words` on each page with the px outside the held-out quarter not a word's; placed with
    `wordboxes.place`; boxes whose map point is in the quarter kept.
  - `read`: `textread.read.build_read` with `wordboxes.BOXES_DIR` and `store.READ_DIR` pointed at the run.
  - `measure`: `measures.headline`, `label_right` (for the paired intervals), `box_numbers`, `read_numbers`, on the
    quarter's answered labels and Terra words.
  - `page`: `review.page_data` + `render` with the run's boxes in its quarter; red = ink whose word-core chance is
    >= 0.5; the miss reasons read the boxes' own px as components (`review.box_components`) with reason texts that
    say so; the other quarters faded and labelled "fold not yet trained"; "before" = today's numbers on the same
    sheet and quarter (`today_numbers`, review/today-ne/numbers.json).
- `review.py` options, all default-off (21 review tests pass unchanged): `page_data(..., comps=, visible=,
  reason_text=, notes=)`, `miss_reasons(..., comps=)`, `box_components`, a FADED palette added only when used.

## Carried items

- **EAST's IoU +1 (Task 5): replaced, not documented.** It is a published constant: on a small box (the fixed test
  batch) it moves the IoU term from 2.0466 to 2.0249. `word_loss` now takes -log(I / U) at the px it reads (I > 0
  there: a word's box has height and length and the distances are above 0); at every other px I and U are taken as
  1 by `torch.where`, so the term there is 0 and passes no gradient (the 0/0 guard). The fixed-value test is
  re-pinned from a float64 numpy reference (iou 2.04660763, total 5.21986738); gradients finite.
- **mps determinism and memory (Task 5): checked.** Two 100-step trainings of the word recipe on the real sheets (NE
  and the answered blocks held out), deterministic mode, mps: identical weights bit for bit (max |diff| 0.0) and
  identical learned loss settings. Peak 2.31 GB allocated by torch (6.85 GB the driver held), batch 12, crop 320.
  (data/solver/experiments/wordread/detector/check-ne.json)
- **Task 15's text-chance channel** comes from fold detectors (page-derived stacking): stated in the Phase 2
  held-out write-up.
- Training settings (steps, lr, batch) are the stage's recipe, untuned.

## Phase 1 results

### crude-map: the first end-to-end number (the controller's request, CPU)

`run_detector.py crude --quarter all` (runs/crude-map-all-s0; NE alone: runs/crude-map-ne-s0). Each quarter's words
come from its own held-out arm (a) seed-0 detector (round 3 heads) and are read by that quarter's current fold
reader; the measures are Task 1's (`measures.headline`, `paired_labels`: whole answered blocks resampled, 2,000
draws). **What crude-map is:** each word a connected component (8 px around) of the px the core head gives even odds
or better (P(core) > 1/2, the MAP decision); its box the smallest rectangle of the component's px along the
core-weighted mean of the learned axis (cos 2θ, sin 2θ); placed by `wordboxes.place`, the boxes whose map point is in
the quarter kept. No tags and no votes, so **touching words merge**: a lower-bound check of the detector's boxes.
3 min for NE, 5 min for all four quarters (18,218 boxes read in 185 s).

Owner's answered code labels, crude-map against today's (the same fold readers, today's boxes):

| quarter | crude-map | today | paired diff [95%] |
|---|---|---|---|
| NW | 91/190 0.479 | 87/190 0.458 | +0.021 [−0.063, +0.099] |
| NE | 142/285 0.498 [0.441, 0.556] | 92/285 0.323 [0.269, 0.377] | **+0.175 [+0.113, +0.239]** |
| SW | 50/94 0.532 | 46/94 0.489 | +0.043 [−0.097, +0.180] |
| SE | 60/130 0.462 | 61/130 0.469 | −0.008 [−0.107, +0.096] |
| **all** | **343/699 0.491 [0.453, 0.528]** | 286/699 0.409 [0.372, 0.446] | **+0.082 [+0.037, +0.126]** |

Labels with a box: 685 vs 621.

Terra words (IoU 0.5), crude-map vs today:

| quarter | matched | missed | geometric "merged" | our boxes |
|---|---|---|---|---|
| NW | 946 vs 960 | 1,224 vs 1,210 | 416 vs 170 | 4,761 vs 1,811 |
| NE | 1,313 vs 850 | 1,392 vs 1,855 | 452 vs 564 | 5,062 vs 2,452 |
| SW | 1,069 vs 1,170 | 1,087 vs 986 | 291 vs 76 | 3,797 vs 2,026 |
| SE | 1,217 vs 1,421 | 1,788 vs 1,584 | 429 vs 160 | 4,598 vs 2,505 |
| all | 4,545 vs 4,401 | 5,491 vs 5,635 | 1,588 vs 970 | 18,218 vs 8,794 |

NE-wide, crude-map's unmatched boxes: 1,405 read as text, 2,345 read as none (small specks); today's 1,080 / 522.
Outside NE the merges rise (no tags), and matched falls in SW and SE.

**p07 review page** (review/crude-map/review.html, 2.5 MB, every quarter from its own detector, miss reasons from the
boxes' own px): Terra 1,106, ours 1,679, **matched 578** (today 447), **missed 528** (659): **merged_with_word 266**
(293), wrong_size 133, outside_map 64, below_level 27, split 24, merged_into_line 14; unmatched read as text 530, as
none 571. Owner's labels on p07: **71/122 = 0.582** [0.496, 0.667] (today 46/122; Terra's strings 107/122). Matched
reads equal to Terra's: 387 of 578 (today 330 of 448). NE alone (review/crude-map-ne/): matched 377 vs 252, missed
332 vs 457, merged_with_word 194 vs 241, labels 46/87 vs 29/87.

**crude-tags (stopped, on hold):** the components split by Gaussian mean-shift of their px' tags, bandwidth the
within-word tag spread measured on each detector's training quarters' Terra words (RMS per channel: NW 0.450, NE
0.421, SW 0.437). Stopped by the controller before any words were formed: the owner prefers a learned same-word
affinity. The code stays behind `crude --tags` (default off).

### crude-aff: the learned same-word affinity (arm a″)

**Arm a″** (`aff-ne-s0`: a′ + `heads="words_aff"`, Task 7f67fca; NE held out, seed 0; 16 min training): final losses
centre 0.284, size 0.049, axis 0.029, core 0.161, tag 0.315, affinity 0.332. Held-out NE per word: axis error median
1.09° (88.7% within 10°), length ratio 1.10, height 1.01, centre error 0.28 heights; centre votes dx slope 1.04 (corr
0.86) at 50-130 px, 0.86 at 0-50 px, 0.15 beyond 256 px: as a′.

| run | where (n) | axis err | within 10 / 20 | length ratio | height ratio | centre err |
|---|---|---|---|---|---|---|
| aff-ne-s0 | px, held (3,682,175) | 0.86° (0.15, 4.4) | 94.6% / 96.5% | 0.90 (0.29, 1.63) | 1.08 (0.77, 1.77) | 0.51 (0.15, 4.08) |
| aff-ne-s0 | words, held (2,704) | 1.09° (0.17, 12.6) | 88.7% / 92.2% | 1.10 (0.77, 2.22) | 1.01 (0.78, 1.57) | 0.28 (0.09, 1.19) |

**crude-aff** (`crude --aff`): the core px (P(core) > 1/2) joined to their right and lower neighbours where the
affinity gives even odds; words the connected components (4-connected), boxes as crude-map's.

- **At the raw logit (balanced loss): 373,503 NE boxes** (crude-map 5,062); not read (about 2.5 h). The cause, tested
  on held-out p07 NE (t8/aff_frag.py): 87,305 words from 1.06 M core px, 71,496 of one px (the words under 10 px hold
  10% of the px). Pairs with logit > 0: same word (both core) 83% right / 81% down; word beside background 7% / 10%;
  word beside another word **41% / 78%** (584 / 1,403 pairs); two core px on no Terra word 78% / 73%. One same-word
  edge in five says "apart" (chips), and the head does not part touching words.
- **With the class prior** (the controller's (b)): the balanced loss's logit is the odds under equal classes; the
  taught pairs of two core px on a″'s training quarters (answered blocks out) are 14,923,278 same and 675,205 apart,
  log odds **+3.10** (every taught pair: +2.80), added before the even-odds call (`--prior core`). NE: **8,839
  boxes; 144/285 = 0.505** [0.449, 0.564]; paired against today **+0.182 [+0.122, +0.246]**, against crude-map (a)
  **+0.007 [−0.045, +0.057]**. Matched 1,322 (crude-map 1,312), missed 1,383 (1,393), geometric "merged" 480 (452);
  unmatched read as text 2,156, as none 5,361.
- **Control, crude-map on a″'s own core head** (runs/crude-map-from-aff-ne-s0): 140/285 = 0.491; against today
  +0.168 [+0.108, +0.231]; **crude-aff + prior minus it: +0.014 [+0.003, +0.028]** (the affinity alone, the same
  detector; about 4 labels). Boxes 5,419, matched 1,289, missed 1,416, merged 494.
- **Merges** (t8/aff_merges.py, NE): at the raw logit 428 Terra word pairs in one word, 211 sharing an edge (208 of
  them with a logit > 0 on it; median share of the edge above 0: 0.92), 217 joined through core px off both words, 38
  touching pairs kept apart. With the prior: 832 pairs, 249 sharing an edge (248 above 0, median share 1.0), 583
  through other core px, **none kept apart**.

### Why labels are wrong: crude-map against today (the controller's measurement)

scripts/wordread/label_causes.py; experiments/wordread/causes/ (causes.json, labels-*.parquet, crude-map-all-s0/ and
today/ with index.html: 5 crops per cause). Each wrong answered code label in one cause; a box holds a Terra word when
more than half of the word's ink is in it; the code word is the Terra word whose rectangle holds the label's point.

| cause | crude-map (a-*-s0) | today |
|---|---|---|
| wrong (of 699) | 356 | 413 |
| 1 no box | 14 | 78 |
| 2 merged (holds 2+ words) | **227** | 206 |
| 3 partial (holds half or less of the code word) | 3 | 35 |
| 4 clean, misread (code word alone, IoU >= 0.5) | 39 (Terra wrong too: 8) | 26 (0) |
| 5 other | 73: loose box 63, no Terra word at the point 10 | 68: 59, 9 |

Per quarter (wrong / no box / merged / partial / clean / other): crude-map NW 99/3/70/2/6/18, NE 143/6/84/0/26/27,
SW 44/0/34/0/1/9, SE 70/5/39/1/6/19; today NW 103/28/27/12/10/26, NE 193/7/144/10/9/23, SW 48/19/15/4/3/7, SE
69/24/20/9/4/12.

Merged by the other held word (crude-map, along / stacked): code + block number 67 / 70 (137; today 57 / 70), code +
code 32 / 13 (45; today 21 / 28), code + street name 20 / 18 (38; today 10 / 7), code + other 1 / 1; holding two other
words, not the code word, 5 (today 11). "Stacked": the other word's centre farther across the code word's axis than
along it.

Today to crude-map: 152 wrong labels made right (77 merged, 30 no box, 22 other, 12 partial, 11 clean); 95 right
labels made wrong (59 merged, 20 other, 14 clean, 2 no box).

### crude-kind: crude-map's components split where the MAP kind changes (the owner's yes)

`run_detector crude --kind map` (runs/crude-kind-map-all-s0): the core px (P(core) > 1/2) joined to their 8 neighbours
only when both have the same MAP kind (the argmax of `kind_p`, the softmax of the per-ink head's channels 1-4 in
`data.KINDS` order: zoning code, block number, street name, other); words the components; boxes as crude-map's. Each
quarter from its own a-*-s0 detector and that quarter's fold reader; no threshold.

**Kind check first** (`run_detector kind-check`, a-ne-s0 on held-out NE, px in Terra's words, MAP kind against the
word's): ink px (1.45 M) agree 0.784; non-ink core px (2.09 M) **0.835** (zoning code 0.760, block number 0.757,
street 0.878, other 0.929); the nearest-ink variant (each px the MAP kind of its nearest ink px, exact EDT) 0.826 on
the non-ink core px. The kind head is no worse off the ink, so the plain MAP kind is used.

| quarter | crude-kind | today | crude-map | vs today | vs crude-map |
|---|---|---|---|---|---|
| NW | 113/190 | 87 | 91 | +0.137 [+0.048, +0.222] | +0.116 [+0.053, +0.179] |
| NE | 167/285 | 92 | 142 | +0.263 [+0.199, +0.326] | +0.088 [+0.044, +0.132] |
| SW | 60/94 | 46 | 50 | +0.149 [+0.011, +0.287] | +0.106 [+0.034, +0.185] |
| SE | 74/130 | 61 | 60 | +0.100 [−0.008, +0.211] | +0.108 [+0.045, +0.173] |
| **all** | **414/699 = 0.592 [0.555, 0.631]** | 286 | 343 | **+0.183 [+0.139, +0.227]** | **+0.102 [+0.073, +0.131]** |

Boxes 80,834 (crude-map 18,218): matched 5,554 (4,545), missed 4,482 (5,491), geometric "merged" 446 (1,588), split
1,049 (195); unmatched read as text 28,657, as none 46,238 (the kind flicker makes fragments the reader mostly calls
none). Labels with a box 688.

Causes (label_causes.py, causes/causes-crude-kind-map-all-s0.json): wrong 285; no box 11, **merged 123** (crude-map
227), partial 21 (3), clean misread 55 (Terra wrong too 9), other 75 (loose box 61, no Terra word 14). Merged by the
other word (along / stacked): code + block number 25 / 49 (74; crude-map 137), code + code 18 / 10 (28; 45), code +
street 10 / 6 (16; 38), code + other 1 / 0, two other words 4. Crude-map to crude-kind: 94 wrong labels made right (69
merged), 23 right made wrong (8 clean, 6 other, 5 partial, 4 merged).

### Code spotting (the owner's "run the test"): below today

a483bc5: recipe `heads="words_code"` (the 16 word channels + 59 class logits: background, another text, the 2000
map's 57 codes), code targets painted by Terra's rectangles (a word is a code's when its normalised string is a form of
it: 2,082 of the 2,150 code-kind words; 20 words of other kinds, streets named like a code, are codes too), a
class-balanced cross-entropy as a 7th Kendall term; `WordOutputs.code` (MAP class) and `code_p`; `run_detector spot`.
Trained NE seed 0 (`code-ne-s0`, a′ recipe, 17 min). The NE fold's training area holds 754 code words (18 codes none,
7 one; P 135, RH-2 99, RH-1 89, RM-1 63, NC-1 54); 11 of NE's 285 labels have a code with 0-1 training words.

**NE: 38/285 = 0.133 [0.098, 0.174]**; paired against today −0.189 [−0.247, −0.125], against crude-map −0.365
[−0.429, −0.304]. Wrong: no code at the point 118 (107 "other text", 11 background), wrong code in its own component
121, wrong code in a merged component 8; right 38 (2 in merged components). **The head under-fits its own training
words** (t8/code_fit.py, the class at each Terra word's centre px): training code words 143/754 right (19%), held-out
NE 62/421 (15%); other-text words 94% right; per code on training: P 57/135, RH-2 46/99, RH-1 36/89, RM-1 3/63, NC-1
1/54, the next five 0. The code term flattened at 1.42 from step ~2,000; its learned Kendall weight ended 0.375 (the
lowest). Not run on the other folds. A p07 page of the NE regions: review/code-spotting/2000-digital-zoning-p07/.

### The detector (arm (a), NE held out, seed 0)

`detector/runs/a-ne-s0`: the stage's recipe (4,000 steps, half size, ink+1px, batch 12, crop 320) with the word
heads and the label rule lines-unknown; 13.9 min training on mps (0.21 s a step), word outputs of the 13 pages ~20 s
a page. Final losses (mean of the last 250 steps): text 0.103, kind 0.202, **angle 1.011**, core 0.156, Dice 0.167,
**IoU 1.099**, **direction 1.007**, tag 0.130. Learned weights at the end: core ~2, Dice ~1.8, IoU ~0.6,
direction ~0.5, tag ~0.8.

**The reading direction never learned, and the box geometry with it.** 1 − cos stayed at 1.0 (random) for the whole
run in both the per-ink angle (channels 5-6 at text px) and the word direction (the same channels at core px). The
control (heads all, the same label rule) learned its angle from about step 1,250 (0.49 at the end). The two
targets agree at every core px (checked on 504,202 core px of p07 crops at every turn and scale: 0 differ), and
Terra's angles are consistent across the maps (horizontal words read at 0 on all six training maps; 180 only on
6-7% of the 1964/1968 words), so it is not a label conflict. On p07:
- NE (held out): mean |cos| 0.72 (random 0.64), mean 1 − cos 1.68: on horizontal words the net says 180 (mean cos
  −0.735). SE (a training quarter) the same: 1.67.
- The distances carry no side information: corr(pred t − b, true t − b) −0.01 (NE) / 0.03 (SE); corr(s − e) 0.02 /
  0.20. Median predicted height / true 1.34, length / true 1.58 (NE).
- The shipped fold detectors' direction (text-reader/direction.npz) is no better on p07: 1 − cos 1.06.
The geometry targets (top, bottom, start, end) are defined by the reading direction, so without its sign they cannot
be learned from the ink; the IoU loss then fits sign-averaged distances.

Peak memory of the full training on mps (sampled): 3.18 GB allocated by torch, 10.06 GB held by the driver.

The per-ink head on the gate (NE, `linegate`, one seed, so no verdict), this run vs control-ne-s0, called at 0.5:
m1a_in_box 0.412 vs 0.393 (more line ink inside boxes called text), m1c 0.062 vs 0.041 (more far line ink called
text), m2 0.978 vs 0.965, m2_clean_words 0.929 vs 0.927.

### The spreads and prior (own training quarters; optimistic)

`measure_spreads` on the NW/SW/SE windows of the 13 pages (120 s): distance sd **274 px**, angle sd **240 deg**,
reversed share **0.88**, cell **653,130 px** (a correlation length of ~322 px), tag sd 3.11, background distance sd
104 px, background tag sd 1.70. These follow from the geometry: with no position information in the distances, a
word's implied edges drift linearly along it, so the within-word variogram never levels off.

(Saved: detector/runs/a-ne-s0/fit/spreads_measured.json.)

### Word formation: stopped

- The prior's first round (`prior_decisions`, no prior: the proposals alone) on the training-quarter area of each
  page, 3 pages at a time (4.2-7.5 GB a process): p01 has no px in the area (0 decisions, 1 s); **p02: 4 decisions
  in 1,226 s**; the others had run 20 min unfinished when I stopped them. A timing probe on a 500 x 500 window of
  p07's NE (11 s) gave one proposal for the whole window: the proposals' smoothing is a Gaussian of the distance sd
  (274 px), so a page's votes fall into a few basins. The second round (the search's splits on those) and
  `core_words` (Task 6: ~12.6 min and 14 GB on a page-sized synthetic) would have taken hours for a few page-sized
  "words". Nothing of the fit was kept but the spreads; no boxes, reads, measures or page were made.
- Today's numbers on the Mission sheet's NE alone (the page's planned "before", review/today-ne/numbers.json): Terra
  709, ours 684, matched 252 (IoU 0.5), missed 457: merged_with_word 241, wrong_size 103, below_level 82,
  merged_into_line 14, split 13, outside_map 4; headline 29 / 87. NE-wide today (baseline/boxes.json): Terra 2,705,
  ours 2,452, matched 850, missed 1,855 (geometric "merged" 564); NE headline 92 / 285 = 0.323.
- Tests: 13 in tests/test_wordread_run_detector.py (stitching equals one pass, full and half size; the context covers
  the impulse footprint; the variant run's word outputs; doubt weights reach the fit; quad-tree windows exact; the
  fit reads only the training quarters outside the answered blocks and drops out-of-area proposals; words come from
  the fold's px alone and match the whole page's; the reach window; add mode; the folder; the page options). Full
  suite: 1767 passed, 3 xfailed (before the commit). The test failure the controller saw (one Terra word with no
  box) was the test's expectation: word A of the synthetic page lies in the other quarter, so the fold rightly never
  forms it; the test now asserts that.

### Axis against sign (the controller's test, CPU only, no training)

At Terra words' core px (their rectangles' px) of the 13 sheets: held = the NE quarter; train = NW, SW, SE outside
the answered blocks. word = a-ne-s0's direction output (channels 5-6). ne = the shipped NE fold detector's per-ink
direction (text-reader/models/ne.pt, run on the CPU; it equals the stage's direction.npz on the NE px to 5e-4).
Axis error = the angle difference modulo 180, in [0, 90]: a uniform guess has median 45, 11% within 10 and 22%
within 20. Reversed = cos < 0. Per word: the circular mean of (cos 2θ, sin 2θ) over its core px, weighted by
sigmoid(core) for word and by chance for ne, against the word's axis; the word's sign is the weighted mean cos;
conc = the mean resultant length. Script: scratchpad t8/axis_sign.py, numbers in t8/axis_sign.json.

| px | n | axis err median (q10, q90) | within 10 / 20 | reversed among within 10 / 20 | reversed, all |
|---|---|---|---|---|---|
| word, held | 3,682,175 | 27.2 (3.7, 83.0) | 25.3% / 41.6% | **99.8% / 99.8%** | 88.7% |
| word, train | 8,203,429 | 29.5 (4.6, 82.2) | 21.1% / 36.8% | **99.9% / 99.9%** | 88.9% |
| ne, held | 3,682,175 | 33.3 (10.2, 60.7) | 9.8% / 23.5% | 44.2% / 43.0% | 66.6% |
| ne, train | 8,203,429 | 30.8 (8.8, 64.3) | 11.7% / 28.1% | 22.8% / 22.7% | 50.7% |

| words | n | axis err median (q10, q90) | within 10 / 20 | reversed among within 10 / 20 | conc median |
|---|---|---|---|---|---|
| word, held | 2,704 | 23.4 (4.8, 64.4) | 21.1% / 41.4% | **100% / 100%** | 0.80 |
| word, train | 5,927 | 25.1 (6.6, 67.9) | 14.9% / 32.2% | **100% / 100%** | 0.83 |
| ne, held | 2,704 | 32.8 (12.1, 55.9) | 7.7% / 20.2% | 72.6% / 72.8% | 0.77 |
| ne, train | 5,927 | 40.7 (12.6, 57.9) | 7.6% / 17.5% | 51.9% / 47.8% | 0.76 |

Read:
- **The word run's sign is not random: it is the same wrong sign everywhere.** Where its axis is right, the
  direction is reversed at 99.8% of px and 100% of words, held out and trained alike. That is what a sign the net
  cannot see gives under the quarter-turn augmentation: an upright word and the same word turned a half turn look
  alike to it, so it puts out one sign per axis orientation (180 for horizontal text), right on half the training
  crops (1 − cos about 1 in training) and wrong on every upright page.
- **Its axis is only partly learned:** 25% of px within 10 (2.3x a uniform guess) and 42% within 20, but the median
  error is 27 and q90 83. Per word, the votes agree with each other (resultant 0.80) yet the word's axis is off by a
  median 23. The 1 − cos loss cannot teach the axis without the sign: with the sign unseen its expectation is 1
  whatever the axis, so whatever axis it has learned came only where the sign was consistent within a turn.
- **The shipped NE detector learned neither** (n is the same px): axis within 10 at chance (9.8% vs 11%), sign near
  chance or worse.
- So "axis right, sign random" holds only in part: the sign is unlearnable here (constant), and the axis is weak
  because the loss gives it no gradient of its own. That supports Option 1's change of target, (cos 2θ, sin 2θ):
  a loss on the axis alone, which the sign no longer cancels, with RBOX distances (to the two long sides and the
  two short sides, unordered) and 0/180 left to the reader's classifier. Whether the axis then learns well is not
  shown by these numbers; a first run with that target would show it.

### Round 3 heads (the controller's ruling: Option 1, continuous form; Task 5 512a1dc)

- `predict.word_outputs` now decodes with `model.word_geometry`: `WordOutputs.geo` is (5, h, w), the offset in px to
  the word's centre (dx, dy), its length and height in px, its axis in degrees in [0, 180). `run_detector.reach_window`
  now takes a vote's reach as |offset| + half its box's diagonal. The tests for the stitching (now against
  `word_geometry` of one pass; the axis compared modulo 180), the variant run and the reach window are adapted and
  pass. The tests that run `core_words` wait for Task 6's adaptation.
- The signed-layout run is kept as detector/runs/a-ne-s0-signed (and check-ne-signed.json).
- Determinism with the new loss: two 100-step mps runs bit-identical (max |diff| 0.0, learned settings equal); peak
  2.02 GB allocated, 6.85 GB driver. In those 100 steps the axis term is already 0.73 (1 − cos 2Δ; 1.0 is random)
  and the per-ink angle term 0.705.

The retrained arm (a) NE seed 0 (detector/runs/a-ne-s0, Task 5's round 3 heads): 13.1 min training, word outputs 4.5
min; peak 2.75 GB allocated, 10.07 GB driver. Final losses (last 250 steps): text 0.107, kind 0.278, angle 0.032,
core 0.165, Dice 0.180, **axis 0.032** (1 − cos 2Δ), **size 0.100**, **centre 1.78 (flat from step 1)**, tag 0.309.

Against Terra at its words' core px (scratchpad t8/axis_size.py, t8/axis_size.json; held = NE, train = the other
quarters outside the answered blocks; per word: core-weighted circular mean of the axis, weighted geometric mean of
the sizes, weighted mean of the votes' centres):

| | axis err median (q10, q90) | within 10° / 20° | length pred/Terra median (q10, q90) | height ratio | centre err, word heights |
|---|---|---|---|---|---|
| px, held | 1.09° (0.17, 6.9) | 92.5% / 95.1% | 1.18 (0.68, 2.62) | 1.06 (0.77, 1.54) | 0.95 (0.32, 4.50) |
| px, train | 1.20° (0.18, 6.1) | 93.7% / 97.1% | 1.14 (0.71, 2.34) | 1.09 (0.77, 1.60) | 1.11 (0.34, 4.88) |
| words, held (2,704) | 1.53° (0.20, 29.5) | 84.3% / 88.5% | 1.44 (0.90, 3.48) | 1.08 (0.86, 1.62) | 0.11 (0.04, 0.26) |
| words, train (5,927) | 1.45° (0.27, 12.4) | 87.9% / 94.3% | 1.25 (0.82, 2.87) | 1.08 (0.81, 1.82) | 0.11 (0.04, 0.40) |

- The axis is learned (it was at chance with the signed target): 92.5% of held-out px within 10°, median 1.1°; per
  word 84% within 10°, with a tail (q90 29°).
- Height is about right (×1.06-1.08). Length runs long: ×1.18 per px and ×1.44 per word on the held-out quarter
  (q90 ×3.5). Long boxes can join neighbours along a line, so this needs watching on the page.
- Each px's centre vote scatters (median 0.95 word heights, q90 4.5): the centre loss did not fall. The core-weighted
  mean of a word's votes still lands within 0.11 heights of Terra's centre (q90 0.26).

The gate's per-ink head, NE, one seed (no verdict), called at 0.5, new run / signed run / control: m1a_in_box 0.457
/ 0.412 / 0.393; m1c 0.067 / 0.062 / 0.041; m2 0.981 / 0.978 / 0.965; m2_clean_words 0.930 / 0.929 / 0.927.

### The four folds and the second seed (arm (a), round 3 heads; queued while waiting for Task 6)

Queue (logs/queue1.sh): SE s0, NW s0, SW s0, NE s1, one at a time in the GPU's exclusive turn; each writes every 2000
page's word outputs (~6.2 GB a run) and is then checked against Terra on the CPU (t8/axis_size.py <arm> <quarter>
<seed>; rows by t8/axis_rows.py). Columns: axis error median (q10, q90); within 10° / 20°; length pred/Terra median
(q10, q90); height ratio; centre error in word heights. held = the run's own held-out quarter.

| run | where (n) | axis err | within 10 / 20 | length ratio | height ratio | centre err |
|---|---|---|---|---|---|---|
| a-ne-s0 | px, held (3,682,175) | 1.09° (0.17, 6.9) | 92.5% / 95.1% | 1.18 (0.68, 2.62) | 1.06 (0.77, 1.54) | 0.94 (0.32, 4.50) |
| a-ne-s0 | words, held (2,704) | 1.53° (0.20, 29.5) | 84.3% / 88.5% | 1.44 (0.90, 3.48) | 1.08 (0.86, 1.62) | 0.11 (0.04, 0.26) |
| a-ne-s0 | px, train (8,203,429) | 1.20° (0.18, 6.1) | 93.7% / 97.1% | 1.14 (0.71, 2.34) | 1.09 (0.77, 1.60) | 1.11 (0.34, 4.88) |
| a-ne-s0 | words, train (5,927) | 1.45° (0.27, 12.4) | 87.9% / 94.3% | 1.25 (0.82, 2.87) | 1.08 (0.81, 1.82) | 0.11 (0.04, 0.40) |
| a-se-s0 | px, held (4,400,049) | 1.08° (0.17, 10.2) | 89.9% / 95.3% | 1.01 (0.56, 2.73) | 1.11 (0.75, 1.73) | 1.13 (0.32, 5.13) |
| a-se-s0 | words, held (3,002) | 1.95° (0.19, 20.7) | 75.7% / 89.4% | 1.25 (0.77, 2.87) | 1.11 (0.82, 2.14) | 0.11 (0.03, 0.41) |
| a-se-s0 | px, train (7,120,749) | 0.87° (0.14, 5.6) | 95.0% / 97.2% | 1.12 (0.61, 2.38) | 1.10 (0.79, 1.63) | 1.04 (0.34, 4.67) |
| a-se-s0 | words, train (5,369) | 1.00° (0.12, 9.6) | 90.5% / 94.4% | 1.34 (0.87, 2.74) | 1.07 (0.83, 1.58) | 0.10 (0.04, 0.29) |

| a-nw-s0 | px, held (2,983,979) | 0.96° (0.15, 5.7) | 94.7% / 97.2% | 1.21 (0.70, 2.91) | 1.11 (0.74, 1.72) | 1.04 (0.33, 4.63) |
| a-nw-s0 | words, held (2,167) | 1.45° (0.23, 10.3) | 89.7% / 94.1% | 1.45 (0.87, 3.27) | 1.08 (0.78, 1.60) | 0.10 (0.04, 0.27) |
| a-nw-s0 | px, train (9,021,249) | 0.99° (0.14, 6.4) | 93.1% / 96.8% | 1.06 (0.63, 2.76) | 1.11 (0.70, 1.71) | 1.10 (0.33, 4.87) |
| a-nw-s0 | words, train (6,343) | 1.65° (0.26, 15.3) | 84.8% / 92.9% | 1.37 (0.79, 3.41) | 1.06 (0.73, 1.78) | 0.11 (0.04, 0.35) |
| a-sw-s0 | px, held (3,594,454) | 0.99° (0.17, 5.1) | 95.3% / 97.5% | 0.94 (0.52, 2.40) | 1.05 (0.73, 1.56) | 1.16 (0.35, 4.92) |
| a-sw-s0 | words, held (2,154) | 1.42° (0.22, 10.2) | 89.7% / 94.7% | 1.16 (0.76, 2.79) | 1.02 (0.73, 1.62) | 0.14 (0.06, 0.44) |
| a-sw-s0 | px, train (7,742,567) | 1.09° (0.17, 6.8) | 93.0% / 96.5% | 1.04 (0.55, 2.56) | 1.07 (0.71, 1.62) | 1.02 (0.33, 4.66) |
| a-sw-s0 | words, train (6,028) | 1.73° (0.26, 15.4) | 85.5% / 92.6% | 1.24 (0.76, 3.03) | 1.02 (0.73, 1.68) | 0.15 (0.06, 0.42) |
| a-ne-s1 | px, held (3,682,175) | 1.18° (0.21, 7.3) | 91.9% / 94.5% | 1.19 (0.57, 2.89) | 1.09 (0.73, 1.65) | 0.94 (0.31, 4.47) |
| a-ne-s1 | words, held (2,704) | 1.66° (0.29, 34.1) | 83.1% / 87.9% | 1.56 (0.87, 4.04) | 1.05 (0.68, 1.69) | 0.14 (0.05, 0.28) |
| a-ne-s1 | px, train (8,203,429) | 1.09° (0.19, 5.7) | 94.1% / 97.0% | 1.01 (0.55, 2.57) | 1.10 (0.70, 1.63) | 1.09 (0.33, 4.82) |
| a-ne-s1 | words, train (5,927) | 1.53° (0.26, 12.3) | 87.4% / 94.0% | 1.42 (0.81, 3.25) | 1.08 (0.67, 1.80) | 0.15 (0.07, 0.43) |

Queue times: SE 14:37-14:58, NW -15:18, SW -15:40, NE s1 -16:04 (about 20-23 min each with the load at 10-40).
Final losses (last 250 steps), SE / NW / SW s0 / NE s1: axis 0.034 / 0.036 / 0.039 / 0.031; centre 1.43 / 1.54 /
1.37 / 1.66; core 0.143 / 0.160 / 0.161 / 0.156. On every fold the axis is learned (89.9-95.3% of held-out px within
10°, per word 76-90%), the height is within ×1.02-1.11, the length runs long per word (×1.16-1.56 held out), and the
votes' mean centre lands within 0.10-0.14 word heights of Terra's (each px's own vote ~1 height off). NE's two seeds
agree (per px 92.5% / 91.9% within 10°; per-word length ×1.44 / ×1.56).

Queue 2 (logs/queue2.sh): control NW s0 16:08-16:19, control SE s0 -16:33 (textlines/runs, heads all, label rule
lines-unknown, as control-ne/sw), then arm (b) (doubt "box" weights) NE s0 -16:58, NE s1, SE s0, NW s0, SW s0.

| run | where (n) | axis err | within 10 / 20 | length ratio | height ratio | centre err |
|---|---|---|---|---|---|---|
| b-ne-s0 | px, held (3,682,175) | 0.83° (0.14, 6.8) | 92.5% / 95.3% | 1.17 (0.67, 2.55) | 1.08 (0.75, 1.56) | 0.96 (0.33, 4.50) |
| b-ne-s0 | words, held (2,704) | 1.04° (0.16, 24.0) | 84.7% / 89.2% | 1.56 (0.91, 3.76) | 1.05 (0.74, 1.66) | 0.14 (0.06, 0.26) |
| b-ne-s0 | px, train (8,203,429) | 0.98° (0.17, 6.9) | 93.3% / 96.9% | 1.13 (0.66, 2.34) | 1.14 (0.80, 1.59) | 1.12 (0.35, 4.88) |
| b-ne-s0 | words, train (5,927) | 1.23° (0.21, 11.6) | 88.4% / 94.5% | 1.46 (0.89, 3.10) | 1.12 (0.76, 1.81) | 0.17 (0.07, 0.38) |
| b-ne-s1 | px, held (3,682,175) | 0.69° (0.11, 5.4) | 93.3% / 95.6% | 1.08 (0.59, 2.37) | 1.06 (0.74, 1.59) | 0.94 (0.31, 4.49) |
| b-ne-s1 | words, held (2,704) | 1.06° (0.15, 17.7) | 86.3% / 90.6% | 1.45 (0.83, 3.44) | 1.02 (0.74, 1.61) | 0.09 (0.03, 0.25) |
| b-ne-s1 | px, train (8,203,429) | 0.67° (0.12, 5.1) | 94.7% / 97.4% | 1.00 (0.60, 2.19) | 1.11 (0.75, 1.65) | 1.10 (0.33, 4.86) |
| b-ne-s1 | words, train (5,927) | 1.10° (0.17, 10.7) | 89.5% / 95.0% | 1.32 (0.80, 2.87) | 1.05 (0.72, 1.81) | 0.11 (0.03, 0.42) |
| b-se-s0 | px, held (4,400,049) | 1.03° (0.18, 13.5) | 87.9% / 93.9% | 1.07 (0.61, 2.61) | 1.13 (0.74, 1.71) | 1.13 (0.32, 5.10) |
| b-se-s0 | words, held (3,002) | 1.80° (0.24, 24.1) | 72.4% / 86.9% | 1.12 (0.76, 2.60) | 1.10 (0.78, 2.13) | 0.18 (0.07, 0.58) |
| b-se-s0 | px, train (7,120,749) | 0.98° (0.17, 5.6) | 93.8% / 96.3% | 1.11 (0.63, 2.30) | 1.08 (0.77, 1.60) | 1.03 (0.33, 4.66) |
| b-se-s0 | words, train (5,369) | 1.22° (0.21, 11.7) | 89.0% / 93.3% | 1.20 (0.83, 2.58) | 1.03 (0.78, 1.55) | 0.18 (0.06, 0.36) |

Arm (b) NW and SW were not trained (the controller's change: arm (a') instead).

### Arm (a′): the centre fixed (Task 5 dadc5d1: no geometry for a word the crop cuts), NE seed 0

detector/runs/af-ne-s0, queue 3 (17:57-18:19). Final losses: centre **0.286** (arm (a): 1.78), size 0.053 (0.100),
axis 0.031, core 0.164, tag 0.286.

| run | where (n) | axis err | within 10 / 20 | length ratio | height ratio | centre err |
|---|---|---|---|---|---|---|
| af-ne-s0 | px, held (3,682,175) | 0.84° (0.14, 5.5) | 93.5% / 95.9% | 0.84 (0.25, 1.47) | 1.04 (0.71, 1.69) | 0.50 (0.16, 4.06) |
| af-ne-s0 | words, held (2,704) | 1.11° (0.17, 15.7) | 86.8% / 90.9% | 1.06 (0.71, 2.16) | 0.97 (0.68, 1.52) | 0.31 (0.12, 1.23) |
| af-ne-s0 | px, train (8,203,429) | 0.85° (0.15, 5.3) | 94.8% / 97.6% | 0.77 (0.22, 1.34) | 1.07 (0.72, 1.70) | 0.62 (0.16, 4.44) |
| af-ne-s0 | words, train (5,927) | 1.20° (0.20, 9.8) | 90.3% / 95.2% | 0.99 (0.70, 1.84) | 1.00 (0.71, 1.66) | 0.30 (0.10, 1.41) |

Centre votes on held-out NE (t8/centre_bands.py af-ne-s0 ne; predicted offset against the true one):

| farther end | share of px | dx corr / slope | dy corr / slope | true median | predicted median |
|---|---|---|---|---|---|
| 0-50 px | 32.1% | 0.587 / 0.777 | 0.341 / 0.275 | 9.7 px | 9.8 px |
| 50-130 | 35.1% | 0.890 / 0.946 | 0.597 / 0.505 | 27.2 | 26.5 |
| 130-256 | 7.5% | 0.725 / 0.653 | 0.527 / 0.434 | 59.8 | 39.0 |
| > 256 | 25.3% | 0.459 / 0.137 | 0.615 / 0.558 | 133.5 | 36.1 |
| all | 100% | 0.461 / 0.183 | 0.563 / 0.485 | 25.5 | 22.1 |

The centre is learned where the word lies within the reach (dx slope 0.95 at 50-130 px); beyond 256 px it falls
short (slope 0.14), as the reach predicts. The per-word length ratio comes to 1.06 (arm (a) 1.44).

(The words' "centre err" column is the core-weighted mean of the votes; see below: it is no evidence the centre was
learned.)

### The centre votes: uninformative, and not because of the receptive field (the controller's test, CPU)

Task 6's implementer found the centre offsets carry no information (a-se-s0: corr 0.04, slope 0.003; the centre loss
flat from step 250; its Kendall weight down to 0.355). Note: my per-word "centre err 0.10-0.14 heights" above is the
core-weighted mean of the votes, which lands on the word's centroid even when each px votes for itself, so it does
not show the centre was learned; the per-px figure (about 1 word height) is the honest one.

- Reach of the half-size recipe (t8/rf_words.py): the architecture's bound is 256 px each way (a square; the analytic
  sum is 245 px, rounded up to the network's multiple). The trained a-se-s0's gradient footprint at word px of p07
  reaches 282 px (Euclidean, inside the square), with 50% of its |mass| within 50 px and 90% within 130 px.
- Terra words on the NE fold's training quarters (NW, SW, SE, answered blocks out; 6,411 words, 8.2 M word px):
  half-length median 17 px, q90 52, q99 367; the farther end from a word px median 90 px, q90 527, q99 739. Share
  of word px whose farther end lies beyond the bound 256: 33.0%; beyond 130 (r90): 42.2%; beyond 50 (r50): 70.7%.
- But the centre fails inside the reach too (t8/centre_bands.py, a-se-s0 on its held-out SE quarter, 4.4 M word px,
  predicted offset (geo 0-1) against the true one, by the farther end's distance):

| farther end | share of px | dx corr / slope | dy corr / slope | true offset median | predicted median |
|---|---|---|---|---|---|
| 0-50 px | 27.0% | 0.083 / 0.016 | 0.055 / 0.016 | 9.6 px | 1.8 px |
| 50-130 | 27.9% | −0.003 / −0.001 | −0.065 / −0.016 | 28.7 | 3.0 |
| 130-256 | 9.4% | 0.081 / 0.007 | −0.050 / −0.012 | 60.7 | 5.1 |
| > 256 | 35.8% | −0.025 / −0.001 | 0.109 / 0.028 | 136.7 | 5.8 |
| all | 100% | −0.111 / −0.005 | −0.022 / −0.005 | 34.4 | 3.6 |

  The same px learned their word's log length (corr 0.819). So the field is not what stops the centre: words that
  lie wholly within 50 px of the px get no centre either; every px votes for about itself. Not tested here: the
  centre term's form (squared offsets in word heights, dominated by long street names whose px sit up to ~15
  heights from the centre, under one learned variance), which is Task 5's to examine.

### Preview page of the heads (the owner's request; scripts/wordread/preview_heads.py)

review/preview-heads/review.html (11.0 MB) and numbers.json: p07 with each quarter from its own held-out detector
(a-ne/a-se/a-nw/a-sw-s0), views Tags (PCA of the core-weighted tags, opacity = core chance), Core (blue tint = core
chance, no level), Ink; axis ticks every 16th px (opacity = core chance); Terra (red) and today's (teal) outlines;
tap a Terra box for its numbers. Top line on this sheet:

| quarter | Terra words | axis err px median · within 10° | axis err word median · within 10° | length / height, word | centre err, word | two tag clusters | today merged_with_word (of missed) |
|---|---|---|---|---|---|---|---|
| NE | 709 | 1.1° · 90% | 1.5° · 86% | 1.36 / 1.07 | 0.10 | 59 | 241 (of 457) |
| SE | 331 | 0.9° · 96% | 1.2° · 89% | 1.09 / 1.08 | 0.11 | 32 | 43 (of 159) |
| NW | 52 | 2.6° · 89% | 1.7° · 85% | 1.59 / 1.06 | 0.10 | 5 | 1 (of 33) |
| SW | 14 | 1.6° · 91% | 2.1° · 79% | 1.01 / 1.09 | 0.26 | 2 | 2 (of 12) |

Two tag clusters, display only: per Terra word, 2-means of its core-weighted
tags; counted when the means' gap is wider than the gap between the word's mean tag and its nearest Terra
neighbour's. A first rule, Ashman's D > 2, counted 559 of 709 NE words: any smooth drift of tags inside a word passes
it, so it was dropped.

### What the direction finding means for Phase 2 (for the controller; not acted on)

Arms (a)-(c) as written train the same heads, so they inherit the unlearned geometry. A fix is a Task 5/6 design
change, the owner's to approve, e.g. EAST's own RBOX convention: the four distances to the rectangle's sides in a
sign-free frame (the axis taken modulo 180, its "start" the end towards the page's left, or its top for a vertical
axis), the axis learned as (cos 2θ, sin 2θ) (no sign to learn), the 0/180 reading left to the reader's classifier
(it reads both ways already); `core_words`' reversed share would then be ~0 by construction. Whether the sign is
learnable at all by these heads is testable cheaply first: NE seed 1 of arm (a) (a Phase 2 run anyway) shows whether
this run's failure is the seed's.

## Phase 2 plan (by the brief; waits for the controller's go)

The keep rule and arms are the brief's, unchanged. What each arm needs:
- **base**: the control detectors (lines-unknown, heads all; NE s0 and s1 exist in textlines/runs), words from
  connected ink at the NE level chosen as `wordboxes` chooses it (on the other three quarters' words, so the
  control's NW, SW (exists) and SE at s0 too), the reader retrained on those boxes with Task 4's kept options.
- **(a)**: NE s0 (done), NE s1, and NW, SW, SE at s0, the last three for the cross-fit of NE's spreads and prior
  (each read on its own held-out quarter; NE s1 uses the same s0 cross-fit). SE first, so the page's SE part follows.
- **(b)**: (a) + `--doubt` ("box" weights), the same five runs.
- **(c)**: the better of (a)/(b) plus Task 15's box model in Task 9's choice.
- Readers: Task 4's `run` on each arm's NE boxes at s0 and s1; the kept variant then on all four folds at s0.
