# Task 7 report: doubt weights for Terra's words

Status: **DONE_WITH_CONCERNS**. The weights are built, fitted per fold and written. They reach the word heads' loss
per px, and the geometry of a word reaching into a held-out area is now unknown. With no weights, everything is
byte-identical. The concerns are about what the fitted weights can see (see "Concerns").

Commit: see the end of this file. Paths:
- `pipeline/src/sfzmaps/solver/wordread/doubt.py` (new);
- `pipeline/src/sfzmaps/solver/textnet/data.py`;
- `pipeline/src/sfzmaps/solver/textnet/train.py`;
- `pipeline/tests/test_wordread_doubt.py` (new).

Outputs (not committed): `pipeline/data/solver/experiments/wordread/detector/doubt.json` (4.2 MB) and
`doubt_features.parquet` beside it.

## How Terra's double reads are represented (checked before building)

- **One run per sheet, in overlapping tiles.** There is no second pass. `text-on-every-sheet` holds 14,829 tile
  answers for 179 sheets (99 tiles on a 1960 sheet). Each tile is 768 x 768 px, placed about 583 px apart, so
  neighbouring tiles overlap by about 185 px. Every tile's answer read what it saw on its own.
- **What `boxes.terra_boxes` does with them:**
  - it keeps the read farthest from its tile's edge;
  - it drops another read whose axis box overlaps a kept one by IoU > 0.3 (`SAME_TEXT_IOU`);
  - it keeps pairs at or below 0.3 as two words. These are Task 5's "floor" pairs, e.g. "OF THE CIT" with "CITY AND".
- **So the "other pass" is the other tile.** `doubt.tile_reads` decodes every tile's reads, the dropped ones too, as
  `terra_boxes` decodes them.
  - A word is **read twice** when another tile's window holds its whole box (a fact of the tiling).
  - Its **other read** is, among those tiles' reads, the one whose rectangle overlaps the word's with the highest IoU.
    This is an argmax with no IoU cut.
- **The 2000 sheets also have other prompts' passes** (`experiments-2000/text-positions`: sideways-text 9,993 boxes,
  close-up-text 876, and three small code passes). They cover the 2000 sheets only and use different questions, so
  they are not used here. They could serve as a further second read on that map.

## How many double reads (the training maps, 87 sheets, 52,218 Terra words)

| | words | read twice | paired (other read exists) | shown twice, read once | strings agree | IoU = 1 | IoU median | forced height | no ink in box |
|---|---|---|---|---|---|---|---|---|---|
| all | 52,218 | 28,303 | **27,219** | 1,084 | 0.868 | 0.052 | 0.877 | 3,337 | 29 |
| 2000-digital-zoning | 10,036 | 6,493 | 6,144 | 349 | 0.853 | 0.063 | 0.882 | 336 | 7 |
| 1980-1990-zoning | 10,463 | 6,556 | 6,291 | 265 | 0.874 | 0.062 | 0.885 | 493 | 7 |
| 1990-1996-zoning | 8,661 | 5,673 | 5,323 | 350 | 0.832 | 0.076 | 0.889 | 269 | 2 |
| 1960 | 5,988 | 2,457 | 2,434 | 23 | 0.910 | 0.021 | 0.865 | 568 | 5 |
| 1964 | 8,036 | 3,345 | 3,311 | 34 | 0.902 | 0.022 | 0.855 | 887 | 5 |
| 1968 | 9,034 | 3,779 | 3,716 | 63 | 0.878 | 0.030 | 0.864 | 784 | 3 |
| street name | 24,933 | 12,592 | 12,051 | 541 | 0.832 | 0.020 | 0.811 | 3,129 | 17 |
| block number | 13,729 | 9,208 | 8,844 | 364 | 0.882 | 0.086 | 0.896 | 57 | 10 |
| zoning code | 10,018 | 5,387 | 5,244 | 143 | 0.952 | 0.052 | 0.930 | 107 | 0 |
| other | 3,538 | 1,116 | 1,080 | 36 | 0.754 | 0.136 | 0.933 | 44 | 2 |

Pair IoU quantiles (all pairs): 1% 0.056, 5% 0.238, 10% 0.496, 25% 0.775, 50% 0.877, 75% 0.936, 90% 0.972, 95% 1.0.

Notes on the counts:
- 1,084 words were shown twice but read once (no overlapping read in the other tile). They have no pair and are not
  in the fit. They are left out because they mix Terra's misses with wrong words.
- An earlier scratch count with axis-box IoU > 0 had 908. Rectangles at near-45 degrees can miss where their axis
  boxes overlap.

## What I built

### `wordread/doubt.py` (new; nothing in a stage imports it)

- `tile_reads(series, sheet, records)`: every tile's reads in page px, before the drop.
- `forced_height(angle)`: `word_rect`'s near-45-degree case. The 0.25 is copied from `word_rect`, because editing
  `boxes.py` would make the prints and walls stages stale. A test pins the two together over every whole degree.
- `rect_iou(a, b)`: shapely polygon IoU, exactly 1 for the same rectangle.
- `doubt_features(sd, records=None, areas=None)`: one row per word in `sd.words` order. It refuses a sheet whose
  words are not `terra_boxes`' words. Columns:
  - text, kind, edge;
  - **ink_outside**: the share of the px of the word's ink pieces (8-connected components of the page ink meeting
    its painted rectangle) that lie outside the rectangle, pooled over the pieces. An empty box gets 1: none of its
    ink is inside. This applies to 29 words.
  - ink_px, **forced_height**, read_twice, other_tiles, paired, other_text, other_iou, agree;
  - one `touch_<name>` bool per area: the word's painted rectangle holds a px of the area.
- `design` builds x as one intercept per kind, ink_outside and forced_height (FEATURES).
- `fit_doubt(features)` fits on the paired rows. It REQUIRES the `held_out` and `answered` columns and refuses any
  true row, mirroring `choose.fit_choice`. The models:
  - **agree**: a logistic model of P(the strings agree | x).
  - **IoU**: a one-inflated beta regression (Ospina & Ferrari):
    - P(IoU = 1 | x) by a logistic (`one`);
    - otherwise Beta(mu phi, (1 - mu) phi), with logit mu = x . `mean` and log phi = x . `precision`.

    I extended the brief's beta regression in two ways; both are measured below:
    - The **inflation** is there because 5.2% of the pairs are exactly 1 (the same rectangle read twice). A plain
      beta has no likelihood there.
    - The **variable precision** is there because one phi for every word fitted every held-out quarter worse and put
    the mean 0.03 below the observed.
  - All of them are fitted by L-BFGS with analytic gradients and a 1e-6 numerical ridge (as `choose.RIDGE`).
- `doubt_weights(sd, params, features=None, records=None)` returns (box_weight, string_weight) per word:
  - box = P(IoU = 1) + (1 - P(IoU = 1)) mu, the expected pair IoU;
  - string = P(agree);
  - no cut and no flag.
- `prepare(sheets, weights)` puts the box weights into `SheetData.weights`. `load_weights(fold)` reads them back from
  `doubt.json`.
- `write_doubt()` (`python -m sfzmaps.solver.wordread.doubt`, one thread, $0, about 1 min):
  - measures the 6 training maps;
  - per fold of `paths.QUARTERS`, fits on the paired words whose rectangle touches neither that quarter nor an
    answered block (the same 715 blocks as the reader);
  - writes the parameters, every word's weights per fold, `fit_counts`, the quantiles per series and kind, and the
    features parquet.

### `textnet/data.py` (no new import)

- `SheetData.weights`: float32 (n,), default empty, meaning every word weighs 1. Added to `BOX_FIELDS`.
- `crop_targets(words=True)` adds `weight` **only when the sheet holds weights**: the word's weight at its core px
  (looked up through the resized and turned `tag_id`), 1 elsewhere.
  - It raises unless there is one finite weight of at least 0 per word.
  - With `words=False` the weights are ignored, so the targets are unchanged.
- **Held-out geometry (Task 5's review):** `_geometry` zeroes the four distances of every word whose painted
  rectangle holds a held-out px (`_touch_held`, through the sheet's `held` areas). The angle is kept. The loss then
  reads no geometry there.
  - Zero distances, rather than a new key, encode "no box" (a real core px always has top + bottom > 0). The reason
    is that Task 5's tests, which I may not edit, pin both the key set of `crop_targets(words=True)` and the 5-tuple
    of `word_tensors`.
  - The core and the tag of such a word stay known.
- The controller's request: the module docstring and the `SheetData.cores` comment said the cores were shrunk. Both
  now say the core is the word's own labelled rectangle with no shrink, as in Task 5's fd0e71e.

### `textnet/train.py` (no new import)

- `word_tensors`: `known` becomes each px's weight: known x the crop's `weight` (0 where unknown). Tags are only on
  px of weight > 0, so a weight-0 word is untagged. It still returns the same 5-tuple.
- `word_loss`:
  - **core**: the core BCE weighs each positive by its weight, and the hard negatives' number is k = 3 x the
    positives' weight. A fractional k gives the ceil(k)-th place its fraction (`torch.ceil` replaces `long`; the
    same for a whole k).
  - **dice**: weighted.
  - **iou and direction**: weighted means over the core px whose target distances make a box.
  - **pull**: per crop, the words' pulls weighed by their weights.
  - **push**: per pair, by the product of the two weights.
  - **crop weight** (pull, push): each crop counts by its words' or pairs' mean weight.
  - **0 / 0 guard**: the clamps of the divisors go from 1 to MASS_EPS (1e-12). With 0/1 weights every divisor is a
    whole number, so nothing changes.
- The per-ink head (channels 0-6, `loss_fn`, `batch_tensors`) is untouched.

## TDD evidence

`pipeline/tests/test_wordread_doubt.py`, 16 tests, covering:
- `forced_height` against `word_rect` at every whole degree;
- ink_outside: half a bar gives 0.5; a word wholly inside gives 0; letters joined to a line give 50/254 exactly; an
  empty box gives 1;
- a word in two tiles: its other read is tile a's best overlapping read (IoU 0.929 over a worse 0.43 read; the
  strings agree after folding "rh-2 " to "RH-2"). A word in one tile has no pair. The read cut at tile a's edge is
  paired with tile b's whole read at IoU < 0.3, because there is no cut. Identical rectangles give IoU exactly 1.
- a word shown twice but read once is unpaired;
- the touch flags, and refusing a sheet whose words aren't Terra's;
- **the fit recovers known coefficients** on 40,000 synthetic double reads (with 500 unpaired rows mixed in), within
  about 4 standard errors: agree 0.15, one 0.3, mean 0.06, precision 0.15;
- **`fit_doubt` refuses** a missing or true `held_out` / `answered` column;
- **more ink outside gives lower weights, smoothly:**
  - on a 1001-point grid for every kind, forced or not, both weights fall strictly at every step;
  - no step is steeper than the logistics' largest slope (|b|/4) allows;
  - on a page, a box drawn shorter over one bar gets lower weights each time.
- `prepare`: lengths, a missing sheet, and the other fields shared;
- **words=False targets unchanged** by weights, byte for byte. With words=True: no `weight` key without weights,
  otherwise each core px carries its word's weight. A bad shape or a negative weight raises.
- **a word reaching into a held area has no geometry**: distances 0, angle kept, core, tag and known unchanged; a word
  touching nothing held is unchanged;
- `word_tensors` weights: all ones are bit-identical to no weights; weight 0 leaves the word untagged;
- **the losses scale with the weights:**
  - weight 0 gives exactly no gradient at that word's px (all 16 channels), while the others' px have gradient;
  - all-ones weights give a bit-identical loss;
  - iou and direction equal the single words' parts weighed by weight x px;
- pull equals (0.3 a + 0.6 b) / 0.9; push is unchanged by the weights of one pair, and 0 with one word;
- a word with no geometry teaches its core but not its box: no GEO or direction gradient there, and core gradient
  kept;
- OHEM with a positive of weight 0.5: k = 1.5, the second negative at half.

Red and green:
- **Red:** the module didn't exist (ImportError). With `doubt.py` written and before the data/train changes, the 7
  data/train tests failed:
  - no `weight` key;
  - geometry not zeroed;
  - `known` not weighted;
  - gradient at a weight-0 word;
  - pull not weighted;
  - geometry read without a box;
  - OHEM at integer k.
- **Green:** 16 passed.

Mutation check (scratch): each of 8 wrong variants fails a test:
- weights ignored in `word_tensors`;
- held-touching geometry kept;
- integer k;
- unweighted tags;
- geometry read without a box;
- an IoU cut of 0.3 on the other read;
- ink outside averaged per piece instead of pooled;
- one constant precision.

**Defaults byte-identical:**
- **Guards:** `tests/test_solver_textnet.py`, `tests/test_textnet_*.py` and `tests/test_build.py` pass (with
  `tests/test_wordread_heads.py` and the doubt tests: 189 passed, none skipped). That includes `GUARD_WEIGHTS`,
  `GUARD_LOSSES`, crop-target equality, the pinned Recipe fields, the composites' pinned batches and the import
  closure (`dvc.yaml` unchanged).
- **HEAD comparison (scratch):** HEAD's `word_tensors` / `word_loss` against the new ones, with no weights, over 20
  random batches (random turns, partly unknown crops). Tensors, loss, parts and gradients are all **bit-identical**.

**Full suite** (`uv run pytest -q -n 2`, on the final code): 1727 passed, **1 failed**, 3 skipped, 3 xfailed, 498 s.
The failure is `tests/test_wordread_boxmodel.py::test_the_gaps_are_between_consecutive_words_a_box_holds_...`
(0.521 vs 0.625). That is another agent's uncommitted work (Task 15's new `boxmodel.py`), not this task's.

## The fitted coefficients (per fold: the fit leaves out the fold's quarter and the answered blocks)

x = one intercept per Terra kind + ink_outside (0-1) + forced_height (0/1). Logit scale for agree, one and mean;
log scale for the precision.

**agree** (logistic, P(the two strings agree)):

| fold | rows | zoning | block | street | other | ink_outside | forced |
|---|---|---|---|---|---|---|---|
| nw | 16,553 | +4.022 | +2.389 | +2.440 | +1.196 | -1.263 | -0.326 |
| ne | 14,031 | +4.257 | +2.476 | +2.404 | +1.323 | -1.346 | -0.328 |
| sw | 15,550 | +4.206 | +2.456 | +2.470 | +1.264 | -1.260 | -0.366 |
| se | 14,187 | +4.024 | +2.536 | +2.532 | +1.196 | -1.234 | -0.507 |

**one** (logistic, P(pair IoU = 1)):

| fold | rows | zoning | block | street | other | ink_outside | forced |
|---|---|---|---|---|---|---|---|
| nw | 16,553 | -2.209 | -2.185 | -3.461 | -1.639 | -1.411 | -1.125 |
| ne | 14,031 | -2.022 | -2.119 | -3.372 | -1.387 | -1.534 | -13.118 |
| sw | 15,550 | -2.320 | -2.124 | -3.357 | -1.659 | -1.401 | -1.018 |
| se | 14,187 | -2.272 | -2.077 | -3.118 | -1.674 | -1.289 | -1.257 |

**mean** (beta regression's logit mu, for the pairs with IoU < 1):

| fold | zoning | block | street | other | ink_outside | forced |
|---|---|---|---|---|---|---|
| nw | +2.330 | +2.002 | +1.289 | +1.051 | -0.825 | -0.261 |
| ne | +2.294 | +2.005 | +1.163 | +1.062 | -0.804 | -0.296 |
| sw | +2.329 | +1.996 | +1.326 | +1.142 | -0.797 | -0.303 |
| se | +2.368 | +2.036 | +1.488 | +1.138 | -0.782 | -0.503 |

**precision** (beta regression's log phi):

| fold | zoning | block | street | other | ink_outside | forced |
|---|---|---|---|---|---|---|
| nw | +2.505 | +2.853 | +1.721 | +0.748 | -1.068 | +0.079 |
| ne | +2.409 | +2.849 | +1.656 | +0.642 | -1.015 | -0.023 |
| sw | +2.572 | +2.807 | +1.733 | +0.789 | -1.037 | +0.041 |
| se | +2.547 | +2.937 | +1.809 | +0.761 | -1.076 | -0.075 |

Fit-set agreement shares are 0.862-0.881 and IoU = 1 shares 0.051-0.059.

What the coefficients say:
- **Kind:** zoning codes are the surest (P(agree) about 0.98 at no ink outside). "Other" is the least sure (0.77).
- **Ink outside:** more ink outside lowers every part, in every fold:
  - agree about -1.3;
  - P(IoU = 1) about -1.4;
  - mean about -0.8;
  - precision about -1.0, so the IoU spreads wider too.
- **Forced height:** it lowers agreement (about -0.3 to -0.5) and the mean IoU (-0.26 to -0.50).
  - Its P(IoU = 1) term is about -1.1 in three folds.
  - In NE it is -13.1: the 639 forced-height pairs in NE's fit set include no IoU of exactly 1 (7 of about 840-1,080
    in the other folds). That is quasi-separation, held finite by the 1e-6 ridge. It changes a forced word's box
    weight by at most about 0.015.

Held out by quarter (fit on the other three quarters outside the answered blocks; report-only, scratch
`analyse.py`):

| fold | pairs | agree observed / predicted | agree mean log-lik: model / kind only / + tile edge | IoU observed / predicted mean | corr(predicted, observed IoU) |
|---|---|---|---|---|---|
| nw | 3,498 | 0.899 / 0.880 | -0.302 / -0.321 / -0.290 | 0.840 / 0.805 | 0.42 |
| ne | 6,009 | 0.886 / 0.869 | -0.338 / -0.353 / -0.325 | 0.838 / 0.772 | 0.34 |
| sw | 4,539 | 0.859 / 0.868 | -0.374 / -0.394 / -0.354 | 0.797 / 0.788 | 0.49 |
| se | 5,891 | 0.843 / 0.869 | -0.405 / -0.426 / -0.386 | 0.751 / 0.800 | 0.54 |

Beta precision, held out (interior pairs):

| fold | one phi: log-lik a pair | phi by x: log-lik a pair |
|---|---|---|
| nw | 0.679 | 0.839 |
| ne | 0.620 | 0.760 |
| sw | 0.516 | 0.720 |
| se | 0.331 | 0.496 |

## The weights per series and kind (fold NE; quantiles 1% / 10% / 50% / 90% / 99%)

| series | kind | words | box 1 / 10 / 50 / 90 / 99% | string 1 / 10 / 50 / 90 / 99% |
|---|---|---|---|---|
| 2000-digital-zoning | zoning code | 2,150 | 0.821 0.821 0.821 0.919 0.919 | 0.948 0.948 0.949 0.986 0.986 |
| 2000-digital-zoning | block number | 4,630 | 0.774 0.774 0.894 0.894 0.894 | 0.756 0.756 0.922 0.922 0.922 |
| 2000-digital-zoning | street name | 2,634 | 0.516 0.592 0.597 0.770 0.770 | 0.675 0.742 0.750 0.917 0.917 |
| 2000-digital-zoning | other | 622 | 0.586 0.657 0.794 0.794 0.794 | 0.494 0.616 0.790 0.790 0.790 |
| 1980-1990-zoning | zoning code | 1,997 | 0.767 0.821 0.821 0.919 0.919 | 0.930 0.948 0.949 0.986 0.986 |
| 1980-1990-zoning | block number | 4,992 | 0.774 0.774 0.894 0.894 0.894 | 0.756 0.756 0.922 0.922 0.922 |
| 1980-1990-zoning | street name | 3,095 | 0.516 0.592 0.601 0.770 0.770 | 0.675 0.742 0.754 0.917 0.917 |
| 1980-1990-zoning | other | 379 | 0.586 0.671 0.794 0.794 0.794 | 0.494 0.623 0.790 0.790 0.790 |
| 1990-1996-zoning | zoning code | 1,917 | 0.821 0.821 0.821 0.919 0.919 | 0.948 0.948 0.949 0.986 0.986 |
| 1990-1996-zoning | block number | 4,104 | 0.774 0.774 0.894 0.894 0.894 | 0.756 0.756 0.922 0.922 0.922 |
| 1990-1996-zoning | street name | 2,155 | 0.516 0.592 0.599 0.770 0.770 | 0.675 0.742 0.752 0.917 0.917 |
| 1990-1996-zoning | other | 485 | 0.586 0.754 0.794 0.794 0.794 | 0.494 0.739 0.790 0.790 0.790 |
| 1960 | zoning code | 1,210 | 0.821 0.821 0.859 0.919 0.919 | 0.948 0.948 0.966 0.986 0.986 |
| 1960 | street name | 4,289 | 0.516 0.592 0.655 0.770 0.770 | 0.675 0.742 0.819 0.917 0.917 |
| 1960 | other | 488 | 0.587 0.767 0.794 0.794 0.794 | 0.498 0.756 0.790 0.790 0.790 |
| 1964 | zoning code | 1,352 | 0.821 0.821 0.866 0.919 0.919 | 0.948 0.948 0.969 0.986 0.986 |
| 1964 | street name | 5,983 | 0.516 0.592 0.689 0.770 0.770 | 0.675 0.742 0.873 0.917 0.917 |
| 1964 | other | 700 | 0.575 0.677 0.794 0.794 0.794 | 0.494 0.633 0.790 0.790 0.790 |
| 1968 | zoning code | 1,392 | 0.821 0.821 0.867 0.919 0.919 | 0.948 0.948 0.970 0.986 0.986 |
| 1968 | street name | 6,777 | 0.516 0.592 0.669 0.770 0.770 | 0.675 0.742 0.843 0.917 0.917 |
| 1968 | other | 864 | 0.586 0.616 0.794 0.794 0.794 | 0.494 0.552 0.789 0.790 0.790 |

(1960, 1964 and 1968 have one Terra "block number" each: box 0.77-0.80.)

Across all words, the box weight runs from 0.52 (a street name touching line work) to 0.92 (a clean zoning code).
The string weight runs from 0.49 ("other" touching lines, forced) to 0.99. Each kind's weights take few values,
because ink_outside is nearly 0 or 1 (Concern 1).

The folds differ little: the largest change of any word's weight from NE's is box 0.037 / 0.042 / 0.082 and string
0.026 / 0.028 / 0.065 (nw / sw / se).

## Concerns

1. **ink_outside is nearly two-valued, so the weights are too.**
   - The share is pooled over the ink pieces, as the brief defines it. A word touching any line work joins a
     component of thousands of px, so its share is about 1. Quantiles: 50% 0.12, 75% 0.998.
   - Within a kind, the weights take about two values (line-touching or not) plus the forced-height shift.
   - The relation is also not monotone:

     | ink_outside | agree | mean IoU |
     |---|---|---|
     | 0 | 0.931 | 0.893 |
     | (0, 0.1] | 0.895 | 0.803 |
     | (0.1, 0.5] | 0.778 | 0.645 |
     | (0.5, 0.9] | 0.843 | 0.707 |
     | (0.9, 0.99] | 0.784 | 0.651 |
     | (0.99, 1] | 0.809 | 0.767 |

     A box cutting through letters ((0.1, 0.5]) is worse than one touching a line ((0.99, 1]). The single linear term
     can't show this.
   - Possible changes, for the owner: ink outside measured without the line px (`SheetData.line_bits`), or a smooth
     spline of it.
2. **The tile edge is the strongest unused feature.**
   - Terra's distance from its tile's edge, as log(1 + px), improves the held-out agreement log-likelihood in every
     fold, by more than ink_outside and forced height together (e.g. NE: -0.338 to -0.325; kind only -0.353).
   - It speaks directly to seam truncation ("OF THE CIT"). This is report-only and not in the model, because the
     brief names the features. It is one more column for the owner to decide on.
3. **P(string right) is taken as P(the two reads agree).**
   - That is literal to the brief, but agreement is not rightness. Two reads of one model on overlapping images share
     errors (e.g. "ST" vs "ST." differ, while the same misreading twice agrees).
   - Under independent reads P(agree) is about p^2, which would give p = sqrt(P(agree)): 0.86 would become 0.93.
   - I kept the literal reading. The reader's use (a later task) should decide.
4. **What pairs can't see.**
   - The pair IoU measures how much two reads disagree, not a bias they share. A forced-height rectangle comes out
     the same from both reads, and so does a box grown over the next word in both.
   - So forced_height's penalty here is only its read-to-read noise (mean -0.3 to -0.5). The forced 14 px height
     itself is not measured.
5. **IoU calibration across quarters.** The expected IoU's held-out mean is off by +0.05 to -0.07 (NE is
   under-predicted: 0.772 vs 0.838). The quarters differ beyond the three features, though the ranking holds
   (corr 0.34-0.54). Since the loss divides by the summed weight, a shared shift cancels; differences between words
   matter.
6. **Changes beyond the brief's literal model, measured:** the one-inflation (5.2% of pairs at exactly 1) and the
   precision submodel (held-out +0.14 to +0.20 nats a pair, in every fold). Both are still a beta regression on the
   brief's x.
7. **Words shown twice but read once** (1,084) and **words read once** (23,915) get weights from their features only.
   Their own observation isn't used (nor is a paired word's own pair, beyond being a fit row). The brief asks for
   the expected weight from features.
8. **Geometry "unknown" is encoded as zero distances, not a key.** Task 5's tests pin the key set and the 5-tuple, and
   I may not edit them. A real core px always has a box (top + bottom > 0), so the encoding is safe. It is
   documented in `crop_targets` and `word_loss`, and a test covers it.
9. **Not done here: "remove the test that restates HEAD_OUTPUTS".** It is in `tests/test_wordread_heads.py`, which I
   was told not to edit. Task 5's fd0e71e already dropped it ("the HEAD_OUTPUTS restatement dropped").
10. **Stale on paper:** `data.py` and `train.py` are text-stage dependencies, as Task 5's edits were. No stage was
    rebuilt. `boxes.py` was deliberately left unedited, to keep prints and walls clean.
11. **Task 8 must call `doubt.prepare`** (after `shrink.prepare`) with the fold's `load_weights(fold)` for the
    weights to be used. Nothing reads them by default.

## Commit

My four paths are in **38b5e4f**, but **under another agent's message**. I staged them (`git add`), and in the same
moment the readercrops agent ran `git commit`, which committed the whole shared index: its readercrops files and my
four. My own `git commit -- <paths>` then found nothing left to commit.
- The committed contents are exactly my final working tree: `git diff HEAD` on the four paths is empty, and
  doubt.py's sha1 matches.
- I did not rewrite history, since other agents work in this tree.
- To review Task 7: `git show 38b5e4f -- pipeline/src/sfzmaps/solver/wordread/doubt.py
  pipeline/src/sfzmaps/solver/textnet/data.py pipeline/src/sfzmaps/solver/textnet/train.py
  pipeline/tests/test_wordread_doubt.py`.

The commit message I had written: "wordread doubt: Terra's words weighted by fitted doubt, no cut (...); SheetData.weights
(empty = 1, byte-identical) reach the word heads per px (...); a word whose rectangle holds a held-out px has no
geometry target (zero distances); data.py's core docs say the labelled rectangle, no shrink".

# Follow-up round 1 (the controller's rulings 1-6; `wordread/doubt.py` and `tests/test_wordread_doubt.py` only)

Status: **DONE_WITH_CONCERNS**. Commit **1d610d3** holds `doubt.py` and `tests/test_wordread_doubt.py` only. Task 5's
test section is intact: there is no change to it between 6d1af69 and 1d610d3. All six rulings are built. Round 1 fits
every held-out quarter better than round 0 in every part. `data.py` and `train.py` were not touched; Task 5's 6d1af69 owns them and its section of the test file.
My test-file diff against 6d1af69 touches only the doubt tests.

## What changed

1. **Unpaired words are a modelled outcome (zero-inflated).**
   - The fit now runs on every word shown twice (outside the fold's quarter and the answered blocks).
   - P(paired | x) is a logistic over all of them, so the 1,084 shown twice but read once count as doubt.
   - The rest (agree, IoU) is fitted given a pair. The weights carry P(paired):
     - box = P(paired) x (P(IoU = 1) + (1 - P(IoU = 1)) x (P(same) mu + (1 - P(same)) mu_neighbour));
     - string = P(paired) x P(agree).
2. **Tile edge:** x gains log(1 + edge), with no clip. A negative edge is refused; none exists, min 0.
3. **Line-aware ink** (needs `SheetData.line_bits`; `write_doubt` reads them from `zones.LINES_DIR` through
   `zones.sheet_lines`, cached for all 87 sheets; `doubt_features` refuses a sheet without them):
   - `ink_outside`: the pooled share of the word's **letter** ink pieces (components of ink less line px) that lie
     outside its rectangle.
   - `line_share`: the share of the rectangle's px that are line ink. The other form, the share of the rectangle's
     ink that is line (`line_of_ink`), is kept as report-only. The rectangle form won held out in all 4 folds,
     though by little (+0.002 to +0.012 a word).
4. **Angle, continuous:** `axis` = |cos^2 a - sin^2 a|, and the fit reads axis and axis^2. Measured held out, every
   fold, sum per word (single-beta IoU for the comparison):

   | form | sum per word, mean over folds |
   |---|---|
   | boolean forced (round 0's) | 0.079 |
   | axis | 0.117 |
   | skew from the nearest quarter turn | 0.132 |
   | **axis + axis^2** | **0.148** |

   The pairs' IoU is U-shaped in the angle:

   | axis | mean IoU |
   |---|---|
   | <= 0.25 (near 45 degrees, forced height) | 0.666 |
   | (0.25, 0.5] | 0.501 |
   | (0.5, 0.75] | 0.508 |
   | 1 | 0.862 |

   A height solved from an ill-conditioned box at 20-35 degrees disagrees more than the forced 14 px at 45.
   `forced_height` stays as a report-only column.
5. **Pairing.** Any positive overlap pairs (the argmax, no cut), as documented in the module. So a pair may be the word
   and a neighbour, when the other tile missed or cut the word, and such pairs enter the IoU fit.
   - The interior IoU is now a **mixture of two beta regressions**: the same word's (mean and precision by x), weighed
     by P(same | x), and a neighbour's (one mean, one precision).
   - Held out it beats the single beta in every fold, by +0.09 to +0.14 a pair: nw 0.648 to 0.753, ne 0.657 to 0.800,
     sw 0.558 to 0.687, se 0.456 to 0.549. These were measured with the first round-1 forms (line share of the ink,
     linear axis), before the forms below were chosen.
   - The fitted neighbour component: mean 0.33-0.37, phi 2.9-3.3.
6. **Small changes:**
   - `fit_folds` (split out of `write_doubt`) is tested for its quarter exclusion: each fold's fit gets exactly the
     words touching neither its quarter nor an answered block, and every word gets weights.
   - `load_weights(fold)` refuses a fold that is not one of `paths.QUARTERS`, or not in the file.
   - `prepare(sheets, weights, which="box")` checks each sheet's `words_hash` (sha1 of `SheetData.words`), stored per
     sheet in doubt.json, as well as the length. `load_weights` returns {"words", "box", "string"} per sheet.

## Held-out log-likelihood per word, against round 0

The fit is on the other three quarters outside the answered blocks. It is scored on the words shown twice that touch
only the held quarter. The columns are the paired outcome (all words), agree (pairs), and the one-inflated IoU density
(pairs).

| fold | words / pairs | model | paired | agree | iou | sum |
|---|---|---|---|---|---|---|
| nw | 3,562 / 3,498 | kind only | -0.0996 | -0.3209 | 0.4461 | 0.0256 |
| nw | | round 0 | -0.0838 | -0.3021 | 0.5314 | 0.1455 |
| nw | | round 1 without edge | -0.0709 | -0.2917 | 0.7329 | 0.3703 |
| nw | | **round 1** | **-0.0693** | **-0.2801** | **0.7594** | **0.4100** |
| ne | 6,232 / 6,009 | kind only | -0.1597 | -0.3532 | 0.4341 | -0.0788 |
| ne | | round 0 | -0.1289 | -0.3376 | 0.5246 | 0.0581 |
| ne | | round 1 without edge | -0.1153 | -0.3305 | 0.8186 | 0.3729 |
| ne | | **round 1** | **-0.1120** | **-0.3199** | **0.8464** | **0.4145** |
| sw | 4,764 / 4,539 | kind only | -0.1904 | -0.3936 | 0.3666 | -0.2174 |
| sw | | round 0 | -0.1564 | -0.3742 | 0.4656 | -0.0650 |
| sw | | round 1 without edge | -0.1459 | -0.3678 | 0.6901 | 0.1765 |
| sw | | **round 1** | **-0.1416** | **-0.3466** | **0.7184** | **0.2302** |
| se | 6,217 / 5,891 | kind only | -0.2083 | -0.4262 | 0.2249 | -0.4096 |
| se | | round 0 | -0.1713 | -0.4046 | 0.3128 | -0.2631 |
| se | | round 1 without edge | -0.1642 | -0.4025 | 0.6151 | 0.0484 |
| se | | **round 1** | **-0.1611** | **-0.3850** | **0.6391** | **0.0931** |

Notes on the table:
- Round 0 had no paired model. Its "paired" column is a logistic on round 0's x, for comparison.
- **Round 1 against round 0:** +0.26 to +0.36 a word per fold, better in every part and every fold.
- **The edge's held-out gain per fold** (round 1 against round 1 without edge): nw +0.040, ne +0.042, sw +0.054,
  se +0.045.

## Monotone effects of the line-aware features (words shown twice; report-only bins)

**ink_outside** (letters only), against round 0's non-monotone with-lines share:

| ink_outside | 0 | (0, .02] | (.02, .1] | (.1, .25] | (.25, .5] | (.5, .75] | (.75, .99] | (.99, 1] |
|---|---|---|---|---|---|---|---|---|
| paired | 0.982 | 0.977 | 0.951 | 0.933 | 0.921 | 0.931 | 0.896 | 0.359 |
| agree | 0.911 | 0.891 | 0.837 | 0.818 | 0.752 | 0.766 | 0.763 | 0.565 |
| IoU | 0.881 | 0.831 | 0.758 | 0.711 | 0.654 | 0.630 | 0.512 | 0.192 |

**line_share** (line ink over the rectangle):

| line_share | 0 | (0, .01] | (.01, .02] | (.02, .05] | (.05, .1] | (.1, .2] | (.2, .4] | (.4, 1] |
|---|---|---|---|---|---|---|---|---|
| paired | 0.987 | 0.979 | 0.981 | 0.963 | 0.968 | 0.944 | 0.831 | 0.593 |
| agree | 0.896 | 0.862 | 0.884 | 0.853 | 0.873 | 0.827 | 0.705 | 0.539 |
| IoU | 0.828 | 0.764 | 0.776 | 0.747 | 0.783 | 0.766 | 0.741 | 0.673 |

What the bins show:
- **ink_outside:** IoU now falls at every bin. Paired and agree fall except one small bump at 0.5-0.75.
- **line_share:** paired falls (one bump of 0.002). Agree and IoU fall overall, with bumps of up to 0.04 in the
  middle (0.01-0.1). The big fall is above 0.2.
- **Fitted coefficients:** negative for both features in every part and every fold.

## The fitted coefficients (x: kind intercepts, ink_outside, line_share, axis, axis^2, log(1 + edge))

**paired** (P(the other tile read something there)):

| fold | zoning | block | street | other | ink_outside | line_share | axis | axis2 | log_edge |
|---|---|---|---|---|---|---|---|---|---|
| nw | +2.734 | +2.023 | +2.690 | +3.527 | -2.015 | -9.038 | -1.611 | +2.082 | +0.364 |
| ne | +3.079 | +2.162 | +2.717 | +3.660 | -1.913 | -9.069 | -2.062 | +2.665 | +0.350 |
| sw | +3.196 | +2.447 | +3.153 | +3.904 | -1.924 | -9.134 | -3.275 | +3.402 | +0.374 |
| se | +2.522 | +1.783 | +2.394 | +3.320 | -2.338 | -8.922 | -0.944 | +1.832 | +0.397 |

**agree** (given a pair):

| fold | zoning | block | street | other | ink_outside | line_share | axis | axis2 | log_edge |
|---|---|---|---|---|---|---|---|---|---|
| nw | +1.098 | -0.686 | -0.418 | -0.083 | -0.825 | -5.411 | -1.407 | +1.477 | +0.574 |
| ne | +1.044 | -0.872 | -0.704 | +0.088 | -0.669 | -5.819 | -1.592 | +1.717 | +0.618 |
| sw | +1.208 | -0.604 | -0.372 | -0.103 | -0.912 | -5.298 | -1.749 | +1.859 | +0.567 |
| se | +0.861 | -0.709 | -0.627 | -0.417 | -0.959 | -5.768 | +0.346 | -0.024 | +0.574 |

**one** (P(IoU = 1 | a pair)):

| fold | zoning | block | street | other | ink_outside | line_share | axis | axis2 | log_edge |
|---|---|---|---|---|---|---|---|---|---|
| nw | -3.380 | -3.321 | -4.371 | -2.454 | -1.825 | -5.678 | -8.128 | +8.523 | +0.136 |
| ne | -3.996 | -4.020 | -4.942 | -3.011 | -2.093 | -6.637 | -7.493 | +8.853 | +0.099 |
| sw | -2.926 | -2.609 | -3.656 | -1.905 | -2.095 | -5.122 | -9.928 | +9.882 | +0.096 |
| se | -3.456 | -3.128 | -4.135 | -2.511 | -1.778 | -4.941 | -9.948 | +10.505 | +0.091 |

**same** (P(the pair is the same word | interior IoU)):

| fold | zoning | block | street | other | ink_outside | line_share | axis | axis2 | log_edge |
|---|---|---|---|---|---|---|---|---|---|
| nw | -2.280 | -1.584 | -2.804 | -0.978 | -2.605 | -3.661 | -3.848 | +4.501 | +1.114 |
| ne | -3.385 | -2.714 | -4.028 | -0.939 | -2.494 | -4.338 | -3.717 | +4.284 | +1.349 |
| sw | -2.028 | -1.524 | -2.663 | -1.152 | -2.611 | -3.865 | -3.752 | +4.399 | +1.090 |
| se | -2.439 | -1.814 | -3.164 | -1.399 | -2.384 | -4.733 | -5.198 | +5.858 | +1.158 |

**mean** (the same word's logit mu):

| fold | zoning | block | street | other | ink_outside | line_share | axis | axis2 | log_edge |
|---|---|---|---|---|---|---|---|---|---|
| nw | +3.755 | +3.268 | +3.132 | +3.312 | -0.591 | -1.619 | -5.143 | +5.191 | -0.249 |
| ne | +3.784 | +3.321 | +3.115 | +3.444 | -0.728 | -1.409 | -4.806 | +4.924 | -0.271 |
| sw | +4.100 | +3.715 | +3.543 | +3.783 | -0.657 | -1.437 | -5.648 | +5.531 | -0.303 |
| se | +3.855 | +3.397 | +3.244 | +3.558 | -0.563 | -1.429 | -4.622 | +4.794 | -0.294 |

**precision** (the same word's log phi):

| fold | zoning | block | street | other | ink_outside | line_share | axis | axis2 | log_edge |
|---|---|---|---|---|---|---|---|---|---|
| nw | +3.748 | +3.580 | +3.219 | +3.089 | -0.571 | -1.547 | -1.899 | +1.982 | -0.083 |
| ne | +3.966 | +3.848 | +3.352 | +3.300 | -0.730 | -1.330 | -1.485 | +1.489 | -0.121 |
| sw | +3.790 | +3.678 | +3.270 | +3.160 | -0.678 | -1.452 | -1.613 | +1.690 | -0.103 |
| se | +3.985 | +3.796 | +3.397 | +3.427 | -0.456 | -1.430 | -2.263 | +2.226 | -0.103 |

**neighbour** (logit mean / log precision):

| fold | logit mean | log precision | mean | phi |
|---|---|---|---|---|
| nw | -0.711 | 1.156 | 0.33 | 3.18 |
| ne | -0.666 | 1.092 | 0.34 | 2.98 |
| sw | -0.704 | 1.198 | 0.33 | 3.31 |
| se | -0.549 | 1.047 | 0.37 | 2.85 |

Fit rows:

| fold | shown twice | paired | P(paired) |
|---|---|---|---|
| nw | 17,327 | 16,553 | 0.955 |
| ne | 14,646 | 14,031 | 0.958 |
| sw | 16,167 | 15,550 | 0.962 |
| se | 14,703 | 14,187 | 0.965 |

## The weights per series and kind (fold NE; quantiles 1% / 10% / 50% / 90% / 99%)

| series | kind | words | box | string |
|---|---|---|---|---|
| 2000-digital-zoning | zoning code | 2,150 | 0.323 0.745 0.889 0.921 0.921 | 0.469 0.867 0.967 0.984 0.988 |
| 2000-digital-zoning | block number | 4,630 | 0.364 0.753 0.879 0.887 0.889 | 0.205 0.717 0.907 0.930 0.938 |
| 2000-digital-zoning | street name | 2,634 | 0.150 0.406 0.713 0.848 0.850 | 0.079 0.494 0.867 0.933 0.947 |
| 2000-digital-zoning | other | 622 | 0.052 0.507 0.877 0.930 0.936 | 0.019 0.340 0.847 0.966 0.973 |
| 1980-1990-zoning | zoning code | 1,997 | 0.431 0.782 0.890 0.920 0.921 | 0.643 0.911 0.967 0.984 0.988 |
| 1980-1990-zoning | block number | 4,992 | 0.307 0.744 0.879 0.888 0.889 | 0.155 0.729 0.905 0.930 0.938 |
| 1980-1990-zoning | street name | 3,095 | 0.131 0.403 0.688 0.847 0.850 | 0.075 0.530 0.861 0.932 0.947 |
| 1980-1990-zoning | other | 379 | 0.081 0.613 0.916 0.934 0.937 | 0.029 0.530 0.944 0.971 0.976 |
| 1990-1996-zoning | zoning code | 1,917 | 0.333 0.745 0.888 0.921 0.921 | 0.410 0.851 0.965 0.985 0.988 |
| 1990-1996-zoning | block number | 4,104 | 0.386 0.760 0.880 0.887 0.889 | 0.250 0.749 0.907 0.930 0.938 |
| 1990-1996-zoning | street name | 2,155 | 0.139 0.418 0.717 0.848 0.850 | 0.066 0.519 0.867 0.933 0.947 |
| 1990-1996-zoning | other | 485 | 0.127 0.472 0.894 0.931 0.936 | 0.051 0.297 0.812 0.970 0.974 |
| 1960 | zoning code | 1,210 | 0.415 0.847 0.906 0.920 0.921 | 0.739 0.948 0.975 0.986 0.988 |
| 1960 | street name | 4,289 | 0.292 0.464 0.741 0.848 0.850 | 0.229 0.739 0.880 0.931 0.946 |
| 1960 | other | 488 | 0.102 0.524 0.900 0.934 0.937 | 0.019 0.420 0.812 0.963 0.974 |
| 1964 | zoning code | 1,352 | 0.496 0.843 0.908 0.920 0.921 | 0.836 0.953 0.976 0.986 0.988 |
| 1964 | street name | 5,983 | 0.281 0.439 0.769 0.849 0.850 | 0.235 0.741 0.889 0.934 0.946 |
| 1964 | other | 700 | 0.034 0.504 0.887 0.933 0.937 | 0.003 0.394 0.812 0.963 0.974 |
| 1968 | zoning code | 1,392 | 0.340 0.839 0.908 0.921 0.921 | 0.653 0.948 0.976 0.985 0.988 |
| 1968 | street name | 6,777 | 0.286 0.454 0.794 0.849 0.850 | 0.218 0.726 0.889 0.934 0.947 |
| 1968 | other | 864 | 0.050 0.498 0.873 0.933 0.937 | 0.006 0.376 0.783 0.963 0.975 |

(1960, 1964 and 1968 have one "block number" each: box 0.26 / 0.19 / 0.69.)

All words, fold NE:

| | min | 1% | 10% | 50% | 90% | 99% | max |
|---|---|---|---|---|---|---|---|
| box | 0.002 | 0.251 | 0.518 | 0.845 | 0.914 | 0.930 | 0.937 |
| string | 0.000 | 0.158 | 0.675 | 0.905 | 0.972 | 0.986 | 0.989 |

The box weight now takes 6,665 distinct values (to 4 decimals). Round 0 gave about two per kind.

Across folds, the largest change of any word's weight from NE's is box 0.11-0.13 and string 0.12-0.19 (round 0:
0.03-0.09).

## Weights outside the fit's support (edge)

The fit's words (shown twice) reach 250-251 px from their tile's edge, not about 185 as the brief guessed. About 8,000
words lie beyond that, all read once and up to 369 px; the edge term is extrapolated for them.

| fold | words beyond | box 10 / 50 / 90%, beyond vs within | string 10 / 50 / 90%, beyond vs within |
|---|---|---|---|
| nw | 8,008 | 0.588 0.848 0.896 vs 0.514 0.852 0.908 | 0.808 0.924 0.970 vs 0.642 0.895 0.961 |
| ne | 8,151 | 0.581 0.842 0.899 vs 0.505 0.846 0.916 | 0.816 0.932 0.977 vs 0.644 0.900 0.971 |
| sw | 8,008 | 0.580 0.842 0.888 vs 0.517 0.849 0.906 | 0.814 0.930 0.973 vs 0.652 0.901 0.966 |
| se | 8,008 | 0.588 0.848 0.898 vs 0.513 0.854 0.914 | 0.821 0.938 0.973 vs 0.656 0.909 0.965 |

Beyond the support, box weights are the same in median. String weights are about 0.03 higher at the median and 0.17
higher at the 10% quantile.

But inside the support the data turn down at the far end, by edge bin (words shown twice):

| edge | 50-100 | 100-150 | 150-200 | 200-260 |
|---|---|---|---|---|
| P(paired) | 0.947 | 0.980 | 0.974 | 0.950 |
| agree | 0.899 | 0.896 | 0.879 | 0.855 |

The log term can only rise. So the extrapolated words' string weights are probably too high.

## TDD evidence (round 1)

- **Order:** the features' forms were chosen by the held-out measurement first, as the ruling asks for the measured
  form, and the tests written for them after.
- `tests/test_wordread_doubt.py` now holds 20 tests: the round-1 doubt tests and Task 5's committed section. New or
  rewritten:
  - axis against `word_rect`'s forced case;
  - letters-only ink outside and the line share, with exact px counts, with and without line px; a sheet without
    line px is refused;
  - **recovery of a known zero-inflated mixture model** on 60,000 synthetic words shown twice. The mean error of
    every part sits within about 3x the largest of three seeds' measured errors, and so do the weights.
  - the weights carry P(paired), by formula, and a never-paired word weighs 0;
  - smoothly lower with more ink outside or more line ink (strict steps, bounded slope);
  - **edge** read as log(1 + edge), with no clip, rising beyond the fit's edges; a negative edge is refused;
  - `prepare` checks the words hash (moved words with the same count are refused) and `which`;
  - `load_weights` refuses "NE" and a fold missing from the file;
  - `fit_folds` gives each fold's fit exactly the words touching neither its quarter nor an answered block.
- **Mutation check:** all 10 mutants fail a test.
  - P(paired) left out of the box weight;
  - unpaired words dropped from the fit;
  - no hash check;
  - no fold check;
  - answered blocks kept;
  - line px counted as letters;
  - line share of the ink;
  - one component;
  - no edge term;
  - edge clipped.
- **Runs:**
  - doubt + heads + build tests together: 82 passed;
  - full suite (`uv run pytest -q -n 2`, once, on the final code): **1744 passed, 0 failed**, 3 skipped, 3 xfailed,
    328 s.

## Concerns (round 1)

1. **Edge extrapolation.** log(1 + edge) rises without end, while agreement and pairing fall over 200-260 px. 15% of
   the words, all read once, are scored beyond the fit's edges (251 px). Their string weights are probably a little
   high. A form able to turn down, such as adding (log(1 + edge))^2, would be the measured fix. I kept the ruling's
   form.
2. **The folds differ more:** up to 0.13 (box) and 0.19 (string) for a word. The cause is more parameters plus the
   mixture's P(same). The held-out quarters still score better in every fold.
3. **Low tails:**
   - A box weight runs as low as 0.002, and 1% of words are below 0.25: P(paired) and the neighbour mixture compound.
   - "Other" words' string weights reach 0.003-0.05 at the 1% quantile.
   - These are fitted values, not cuts, but they nearly remove those words from training.
4. **Agree intercepts are not comparable to round 0's.** They now sit near 0 or below, because log(1 + edge) (about
   4.9 at the median) carries the level.
5. **line_share's agree and IoU have bumps** (up to 0.04) between 0.01 and 0.1. The fitted effect is monotone
   (negative), so those bumps are smoothed over.
6. **P(string right) is still taken as P(paired) x P(agree),** as round 0 took P(agree) (round 0's Concern 3).
7. **The test file's Task 5 section:** 6d1af69 committed its rewrite of the loss tests in this file. My working copy
   already held it (I edit only the doubt section), and `git diff HEAD` on the file touches only my section, so the
   commit keeps both.
