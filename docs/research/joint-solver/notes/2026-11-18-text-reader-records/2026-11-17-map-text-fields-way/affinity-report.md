# Same-word affinity output (heads="words_aff"): report

The owner approved it ("Yes try affinity"). The network now learns where one word ends and the next begins, as CRAFT
and affinity-based instance segmentation do. Nothing was trained (Task 8 trains it). CPU tests only, $0.

## What was built

- **Recipe value:** `train.Recipe(heads="words_aff")`, added to `train.CHOICES["heads"]`. `train.WORD_HEADS` is
  `("words", "words_aff")` and every former `heads == "words"` check uses it. The recipe refuses composites, as
  "words" already does.
- **Channels:** 18 (`model.WORD_AFF_OUTPUTS`, `model.HEAD_OUTPUTS["words_aff"]`). Channels 0-15 keep the layout of
  "words". `model.AFF = slice(16, 18)`:
  - ch 16 is the logit P(this px and the px to its RIGHT are in one word);
  - ch 17 is the logit P(this px and the px BELOW are in one word).
- **Targets:** `data.affinity_targets(tag_id, known)` gives `aff_right` and `aff_down` (uint8 (h, w)) and `aff_known`
  (uint8 (2, h, w): [0] the right pairs, [1] the down pairs). `data.crop_targets(..., words=True, affinity=True)` adds
  them. The rule for a pair:
  - target 1 when both px carry the same non-zero word id, else 0;
  - taught only when both px are known and at least one is in a word;
  - not taught when both px are background, either px is unknown or held out, or the pair has no second px (the last
    column for right pairs, the last row for down pairs).
- **Turns:** the targets are computed after the crop is scaled and turned, from the turned `tag_id` and `known`. So
  they are exact for every turn by construction. A test checks both ways: against a recomputation pair by pair from
  the turned map, and against the unturned crop's pairs moved by the turn. After one turn, a right pair becomes an up
  pair (the down pair of the px above).
- **Overlaps:** where word rectangles overlap, `tag_id` holds the later word (painted later over earlier, as before).
  The pairs follow `tag_id`, so the earlier word ends where the later one's rectangle starts.
- **Loss:** `train.word_tensors(crops, affinity=True)` returns two more tensors: the targets and each pair's weight,
  both (n, 2, h, w).
  - The term is `_balanced_bce(out[:, model.AFF], aff, aff_weight)`: each class weighted by the inverse of its weight
    in the batch, as the core's term is, with both directions pooled.
  - It is the 7th part (`train.AFF_PARTS = WORD_PARTS + ("affinity",)`), Kendall-weighted by its own learned
    log-variance (`WordLoss(affinity=True)` has 7). No hand-set weights.
  - With Terra doubt weights, a pair weighs the chance its target is right. That is its word's weight when both px
    are in one word or one px is background, and the product of the two words' weights when they are two words.
- **Crops without the keys** (scenes, whose bank changes `known` after `crop_targets`): `word_tensors` derives the
  targets with the same `data.affinity_targets` from the crop's final `tag_id` and `known`.
- **Prediction:** `predict.WordOutputs` has a new last field, **`aff`**: float32 (2, h, w), [0] the right logit, [1]
  the below logit. It is None for a "words" network, and it defaults to None, so 5-argument construction still works.
  - Tiles are stitched like the other outputs.
  - The value is `NOT_SCORED` (-1e4) on tiles that were not scored, and on the page's last column ([0]) and last row
    ([1]), where the pair has no second px.
  - `word_outputs` accepts 16- or 18-channel networks.
  - `run_detector.py` and `variant.py` were not touched; Task 8 owns them and has added `words_aff` there.

## Defaults unchanged

- `heads="words"` still has 16 outputs, 6 learned terms, the same `word_tensors` call and the same tensors; old models
  build and load.
- `heads="all"` is untouched.
- `crop_targets` without `affinity` returns the same keys and bytes.
- No new imports in `data.py` or `train.py`.

## Tests

- New file `tests/test_wordread_affinity.py` (23 tests). It covers:
  - touching words: 0 across their shared edge, 1 inside each;
  - word beside background: 0;
  - background beside background, or a pair with an unknown px: not taught;
  - overlaps painted later over earlier;
  - exactness for every turn at three scales;
  - the loss: near 0 when right, higher when wrong, higher when only the borders between two words are wrong, and
    unchanged by untaught pairs;
  - the balanced cross-entropy matches a direct reference;
  - Kendall's sum has 7 log-variances;
  - pair weights under doubt;
  - the fallback for crops without the keys;
  - the recipe, and that old "words" models save, load and predict with `aff` None;
  - stitching equals one pass over the whole page (full and half size);
  - a 2-step fit on CPU repeats bit for bit and learns the "affinity" weight.
- `tests/test_wordread_heads.py`: the pinned `CHOICES["heads"]` tuple was updated to include "words_aff".
- Iterate set (wordread_*, solver_textnet with `GUARD_WEIGHTS` / `GUARD_LOSSES` and the pinned batches,
  solver_textread, test_build, textnet_composite) with `-n 2`: 495 passed, 4 failed, none caused by this change:
  - `test_wordread_scenes.py::test_a_scenes_masks_are_exact` (3 cases) also fails on HEAD's textnet code, checked
    against a copy of HEAD's data/train/model/predict. Its `geo_known` assertion predates `dadc5d1`'s rule for words
    the crop cuts.
  - Full suite (`uv run pytest -q -n 2`): 1900 passed, 3 xfailed, the same 4 failed. Commit `7f67fca`.
  - Task 8's `test_wordread_run_detector.py::test_crude_aff_splits_touching_words_the_affinity_parts` (now committed
    in `658d390`, still failing after `7f67fca`)
    builds `types.SimpleNamespace(**o._asdict(), aff=...)`. Now that `WordOutputs` has a real `aff` field,
    `_asdict()` already holds `aff` (None), so the keyword is passed twice. The fix is in their test:
    `o._replace(aff=same)`.

## Concerns

1. Class 0 pools word-beside-background pairs with word-beside-word pairs. The second kind is what separates merged
   labels, but it is a small share of class 0. The balanced cross-entropy evens out the two classes, not these two
   kinds within class 0, so the term may mostly learn word outlines, which the core already gives. Weighting the kinds
   by their measured frequencies (a third class) would be the principled fix if the end-to-end run shows that.
2. Word-beside-word pairs exist only where Terra's rectangles touch or overlap. Labels with a background gap between
   them are taught as two word-to-background boundaries.
3. The edge value `NOT_SCORED` on the last column and row is a choice. Task 8's word formation should treat it as "no
   pair".
