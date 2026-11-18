# Task 6 report: words from cores, votes and tags

Status: **DONE_WITH_CONCERNS**.

Commit: ac8c0f4 (branch `solver`). It holds two new files and touches nothing else:
- `pipeline/src/sfzmaps/solver/wordread/words.py`
- `pipeline/tests/test_wordread_words.py`

The controller's correction arrived mid-task (no hand-set thresholds; evidence as probabilities). I made it before the
commit, so there is no follow-up commit. The rule "tags more than 3 apart" from the brief is gone.

## What I built

### `core_words(core, geo, tag, ink, level, r, refine, *, tag_sigma, split_prior, merge_iou=None, kind_p=None, sheet="")`

**Inputs.** It works on one page's arrays:
- `core`: the core chance (h, w);
- `geo`: (5, h, w) or (6, h, w). Either form starts with the four distances in px. Then comes the reading
  direction:
  - 5 channels: the angle in degrees, as `crop_targets(words=True)` gives it;
  - 6 channels: (cos, sin) of any length, as the net's channels 5-6 give it;
- `tag`: (k, h, w);
- `ink`: bool (h, w).

Turning the net's outputs into these arrays is left to Task 8, as ruled. `textnet/predict.py` is untouched.

**1. Words.** The 8-connected components of `core >= level`. `level` is required and has no default constant.

**2. Split, as a likelihood (the controller's correction).**
- The model: a word's px tags are Gaussian about its mean, with sd `tag_sigma` on every channel.
- `two_word_llr(tags, sigma)` is public. It returns log L(two-part mixture) − log L(one Gaussian):
  - the mixture's weight and both means are fitted by EM;
  - EM starts deterministically, from the tag farthest from the mean and then the tag farthest from that.
- A component is proposed for a split when `llr + split_prior > 0`, where `split_prior` is the log prior odds of two
  words against one. Each px goes to its likelier part, the 8-connected pieces of each part are found, and every
  piece is tested again (recursively).
- **Then the pieces are decided by the same model.** Touching pieces are joined while one word is at least as likely
  as two, likeliest join first. The test is `cost + split_prior <= 0`, where cost is the two-means-against-one
  log-likelihood ratio, nₐn_b/(nₐ+n_b)·|μₐ−μ_b|²/2σ².
- Why the join step exists: without it, the noisy three-word chain gave **51 words**. The two-part fit of three words
  leaves about 10% of the middle word's px on the wrong side, scattered in one-px pieces.
- Only `tag_sigma` and `split_prior` decide a split. No distance between tags is fixed anywhere.

**3. Boxes.**
- **`refine="unclip"`:**
  - `cv2.minAreaRect` of the word's core px, taken at (col, row);
  - laid with its first edge along the side nearest the word's reading direction (the chance-weighted sum of its px
    directions);
  - grown by `shrink.unclip(rect, r)`.
- **`refine="votes"`:**
  - Each core px gives a vote: its word's rectangle, from its four distances along its own direction. This is
    `data._geometry` inverted exactly.
  - Each vote's corners are rotated so its first edge lies nearest the word's direction. Without this, a vote read
    the opposite way (180°) would average the corners into a collapsed box.
  - The votes are merged by score-weighted locality-aware NMS, with each px's core chance as its score:
    - in px order, a vote merges into the running box when IoU > `merge_iou`;
    - the boxes set aside are then merged, in falling score, into the first kept box they meet at that IoU.
  - The word's box is the one with the most score, so a minority of bleed votes is outvoted rather than averaged in.
    It is then squared up to a rectangle.
  - `merge_iou` is a required argument for votes. A call without it raises.
- **Page edge:** a box running off the page keeps its axes and becomes the extent of its on-page part. Boxes are never
  wrapped or dropped.

**4. Table.** The columns are `wordboxes.COLUMNS`, with the same meaning, then `EXTRA_COLUMNS = (form, core_px,
tag_spread, p_word, split_llr)`.
- **`g` corners:** in `word_rect` order (first edge along the reading direction, second edge towards the letters'
  feet), in the frame of Terra's rectangles.
- **`t` corners (trimmed box):** the extent of the ink px inside `g` along g's axes, each px a px wide, kept within
  `g`. When there is no ink, `t` is `g`.
- **`axis`:** modulo 180.
- **`mean_chance` = `p_word`:** the mean core chance over the word's core px. Boxes are weighed by it, never dropped.
- **`form`:** `"unclip"` or `"votes"`, with `"+split"` added when the word was cut from a larger component.
- **`tag_spread`:** the root mean square of the word's own tags about their mean. This is the one-word model's fitted
  σ, which is what `tag_sigma` is measured from.
- **`split_llr`:** for a cut word, the log-likelihood ratio of it standing apart from its likeliest touching word.
  NaN for a whole component.
- **Not placed:** `quarter`, `x`, `y`, `lab_lot` and `street` are -1, NaN, NaN, -1 and False. See concern 4.
- **Types:** as `wordboxes.build_boxes` types them.

### `choose_level(boxes, reads, terra, forms=None)`

- Each quarter's level is chosen on the other three quarters by Task 1's headline:
  - `measures.label_right`: read right at our boxes, with the holding box chosen by the T3 rule;
  - computed one sheet at a time;
  - over Terra's code words: the truth is `_code_of(text, forms)`, the point is the rectangle's centre;
  - non-code words are left out;
  - a word with no box counts wrong.
- Ties go to the higher level.
- The levels are whatever the caller passes. The owner's answers are not an input.
- It returns `level`, `right_on_others` and `words_on_others`.

## TDD evidence

1. **Red:** `ImportError` (no `words` module).
2. **Rewritten after the correction:** the split tests now use noisy tag clouds and `tag_sigma`, `split_prior` and
   `merge_iou` as test values (0.3, −20, 0.5).
3. **Red, then green:**
   - The first implementation gave 51 words on the three-word chain and on the two-cloud case. Adding the join step
     fixed both.
   - Unclip on the hand-made touching cores gave IoU 0.53 against the words. The cause was the test, not the code:
     those cores are not r-shrunk. The test now compares unclip with `shrink.unclip(core, R)` (IoU > 0.9), and votes
     with the word itself (exact to 1e-3 px).
4. **Green:** 24 tests in `tests/test_wordread_words.py`. The targets are drawn by Task 5's own
   `data.crop_targets(words=True)`. The tests cover:
   - two words whose ink joins become two boxes (both refines; votes exact, unclip within 1 px);
   - touching cores with two tag clouds 2 apart at σ 0.3 give two words. The same clouds give one word at σ 3, and
     one word at prior −1e9;
   - a three-word chain gives three words (both refines);
   - a line along the foot and a line crossing the middle neither extend the box nor split it (both refines). The
     trimmed box stays in the word;
   - votes keep the word when the core bleeds 60 px along a line above the level, with the bleed votes pointing along
     the line;
   - a core at the page edge (left and bottom) gives a clipped box (both refines): not dropped, not wrapped, still
     read left to right;
   - a rotated word (30, 90, 150, −45, 180°) comes back from votes **within 1 px per corner**, with the reading
     direction kept. Unclip gives IoU > 0.85 with the right direction;
   - the 6-channel (cos, sin) geo gives the same frame as degrees;
   - the column meanings and `kind_p` means;
   - errors for a bad `refine` and for votes without `merge_iou`;
   - an empty page gives an empty table;
   - `choose_level`:
     - nw → 0.7 although nw's own words favour 0.5;
     - sw and se tie at 3/6 and go to 0.7. If the word with no box were dropped instead, 0.5 would win at 3/5;
     - flipping Terra's strings flips the choice.
5. **Mutation checks** (scratch script; each guarded behaviour removed in turn):

   | Behaviour removed | Tests that failed |
   |---|---|
   | page clip | 4/4 edge |
   | LANMS (replaced by a plain weighted mean) | the bleed test |
   | split | 2/2 touch, 2/2 chain |
   | direction-laid unclip | 5/5 rotated |
   | missed boxes counted wrong (dropped instead) | the level test |
6. **Suites:**
   - the iterate set plus `test_build.py`: 96 passed;
   - the full `uv run pytest -q -n 2`: **1694 passed, 3 xfailed**.
7. **Cost and null check:** a 2000×2000 synthetic page with 168 words (codes and street names, 0/30/45/90°, tag noise
   0.3):
   - unclip took 1.0 s (6 ms a word) and votes 2.0 s (12 ms a word), on one thread;
   - no false split.

   With the right σ, the two-word LLR on a single cloud has a median of about 2 and a maximum under 10, for n from 20
   to 1500.

## Concerns

1. **Three parameters must be measured in Task 8, on the training quarters' predictions:**
   - `tag_sigma`: from `tag_spread` over Terra-matched words;
   - `split_prior`: fitted from `two_word_llr` on components holding one Terra word versus two;
   - `merge_iou`: LANMS needs some threshold. Measure it where votes of one Terra word and votes of two touching words
     are equally likely.

   The level is chosen by `choose_level`.
   - If σ is under-estimated, the null LLR grows, causing false splits.
   - PAN's pull loss is a hinge at 0.5, so within-word tags are not truly Gaussian. Treat the fitted σ as an
     approximation and check it by looking.
2. **The split uses two likelihoods.**
   - The soft mixture proposes. The hard two-means cost decides the joins. Both use only σ and the prior.
   - A cut survives only when both favour it.
   - A purely hard criterion would over-split noise: it gains about n/π nats on one cloud. A purely soft one scatters
     px.
3. **Votes give one box per component.** A strong minority vote cluster is discarded, not emitted. For example, two
   words with the same tags would yield only one box. Task 9's split candidate could use that signal.
4. **Boxes are not placed on the map.** `read.py` picks the reader by `quarter`, so Task 8 must place the boxes as
   `wordboxes.sheet_boxes` does (`prints._place`, `grid.quarter_of`) before reading. I left this out to stay in scope.
   Also, `read.py`'s `mask=True` path rebuilds wordboxes components from `chance.png` and checks `i` and `ink_px`, so
   it does not fit these boxes. Read them with `mask=False`, or adapt that path (it is Task 3's file).
5. **Unclip's known weaknesses:**
   - thin cores of small text, where px rounding matters most;
   - cores cut by the page, which grow less (the test allows 2 px);
   - a core that bleeds along a line extends the box (votes resist this).
6. **Coordinates.**
   - Core px stand at (col, row), as the targets measured them. The boxes therefore match Terra's rectangles in
     Terra's (edge) frame.
   - The targets carry a 0.5 px offset from `fillPoly` on edge-coordinate corners. That comes from Task 5 and is not
     corrected here.
   - A rotated box clipped at the page edge can poke a small corner past the page. Axis-aligned clips are exact.
7. **`choose_level` details:**
   - It uses `measures._code_of`, which is private (like Task 2's deferred minor).
   - It does not remove Terra words in answered blocks. They are Terra's words, not the answers. Leaving them out is
     the caller's call if wanted.
8. **Fixed numbers left in the module:** only numerical ones, `EM_STEPS = 100` and `EM_TOL = 1e-9` per px. The EM step
   cap is most of unclip's cost.
9. **Process limit:** my first two pytest runs used the repo's `addopts -n auto` on my own file. That briefly spawned
   more than 2 idle xdist workers. Every later run used `-n 2` with one thread.


# Fix round 1: words from the core-weighted vote mixture (owner's amendment eb0cf7e)

Status: **DONE_WITH_CONCERNS**. The amended brief is implemented and tested. Removed:
- the level;
- the merge IoU;
- r and unclip;
- one box per component;
- the split-then-rejoin;
- `choose_level`;
- the old columns `form`, `core_px` and `split_llr`.

Synthetic maps use `shrink.prepare([sd])`. Nothing uses r. `wordread/words.py` and `tests/test_wordread_words.py`
were rewritten whole.

## What I built

### The model: one likelihood for every decision

- **Votes.** Each px with core chance w > 0 casts a vote: its 4 distances, its reading direction and its tag.
- **Word px (chance w).** The vote belongs to one of the words, by the words' mixture weights. A word is a direction
  φ, its 4 edges along φ's axes, and a mean tag. At px p the word implies:
  - the distances from p to its edges along its own axes, with Gaussian errors of sd `distance`;
  - its direction, with an error of sd `angle`;
  - its tag, with an error of sd `tag`.

  A vote may read the word the other way round (start ↔ end, top ↔ foot), each reading with probability ½.
- **Background px (chance 1 − w).** Each distance is half-normal with sd `background_distance`, the direction is
  uniform, and the tag is Gaussian about 0 with sd `background_tag`.
- **Effective votes.** The log likelihood is divided by `cell`: n votes carry n / cell independent votes. The score
  is LL / cell + K · `word_prior`.
- **`Spreads` has six fields:** distance, angle, tag, cell, background_distance, background_tag.

**Why the errors are modelled on distances and the direction, not on corners.** A vote's 8 corner coordinates come
from 4 distances and 1 direction, so they are not 8 independent errors. Under the corner model, one noisy word
(2,525 votes, noise on the distances as the net makes it) had a two-word gain of **493 nats**. With iid corner noise,
which is what that model assumes, the gain was **13**. So the corner model's likelihood grows with word size: the
reviewer's finding. In the distance/direction model, the null two-word gains in the tests stay under 20.

### The fit (my choice, as asked)

1. **Proposals.**
   - Each px adds its core chance at its vote's box centre (at the page's own px).
   - The sum is smoothed by a Gaussian with sd = the distance spread.
   - Steepest ascent over 8-neighbours gives basins. The votes landing in one basin are one proposal.
   - This is vectorized over the page.
2. **Each proposal** is fitted as one word by EM, with the background in the model. All basins are fitted at once,
   each stopping on its own. A proposal becomes a word when gain / cell + word_prior > 0.
3. **Greedy moves by the same score.** While a move raises the score, the best one is made:
   - join two words whose boxes' bounds meet (one word against the two-word mixture, on their px);
   - or split a word (the two-word mixture, started at the word and at its worst-explained vote, against one word).
4. **Output.**
   - Each word's box comes from its fitted parameters. It is oriented by the chance-weighted sum of its votes'
     directions, then clipped to the page in float64 (Sutherland-Hodgman, then the extent along the box's axes).
   - `p_word` = Σ w·ρ / Σ ρ: the word's share of the weighted vote mass, where ρ is each px's posterior of voting
     for it.
   - `core_mass` = Σ w·ρ.
   - `tag_spread` = the weighted root mean square of the tags about the word's mean tag.
   - `mean_chance` = `p_word`. `level` is NaN. The placement columns are left unplaced, for Task 8.

### For Task 8

- **`measure_spreads(pages)`** takes dicts of core, geo, tag, word_id (the targets' `tag_id`) and words (Terra's
  rectangles). It measures:
  - each spread as a root mean square, with each vote read the way nearer its word;
  - `cell` as the mean over words of n·|mean distance error|² / (4·distance²).
- **The gains `fit_word_prior` needs:**
  - `word_gain(core, geo, tag, spreads, mask)`: one word over none;
  - `extra_word_gain(core, geo, tag, spreads, mask)`: two words over one, at least 0.
- **`fit_word_prior(gains, real)`:** the decision as a probability, P(real) = sigmoid(gain + prior). It returns the
  maximum-likelihood prior: the log likelihood is concave, solved by `brentq` on its slope. It raises when every
  decision is of one kind.
  - Task 8 should give it word_gain on Terra words (real) and on proposals away from them (not real).
  - It should also give it extra_word_gain on single Terra words (not real) and on touching Terra pairs (real).

**Numerical constants only:**
- `EM_STEPS` = 200 and `EM_TOL` = 1e-7 per vote (loosened from 1e-9: tiny components hit the step cap);
- `CHUNK_PX`, which only bounds memory (the results do not depend on it);
- OpenCV's 4-sd Gaussian extent.

## TDD evidence

- **Red:**
  - the new tests against the old module fail at collection (no `Spreads`);
  - the first implementation (corner model) failed 4 tests: 26 boxes for 3 noisy words, 797 for one noisy word,
    and a two-word gain of 410 on one word. The cause was confirmed as above (493 against 13) before the rewrite;
  - after the rewrite, one failure remained: 9 junk two-vote boxes. That was my generator's fault: the background
    px all had direction exactly 0, which contradicts the model's uniform background direction.
- **Green:** 25 tests in `tests/test_wordread_words.py`, one per failure kind:
  - two touching words → two exact boxes;
  - three noisy touching words (the old 51-piece chain) → 3 boxes within 0.5 px;
  - a noisy single word among background votes at chance 0.001 → 1 box, p_word ≈ 1;
  - a long word whose votes scatter (5° direction noise: 12 proposals) → 1 box, IoU > 0.99;
  - votes correlated in 4×4 blocks → 1 word at cell 16, more than 1 at cell 1;
  - a third of a word's px reading it backwards → 1 word, read as the majority reads it;
  - a line touching a word → the word's box exact, the line's votes their own box with p_word 0.6;
  - a strong minority cluster (a code centred on a street name: 15% of the px, the same centre and the same tag) →
    two exact boxes;
  - page edge (left and bottom) → clipped exactly, not dropped, not wrapped;
  - unrounded votes at 30, 90, 150, −45 and 180° → corners within 1e-6 px, with the reading direction;
  - Task 5's own targets (`prepare([sd])`, `crop_targets(words=True)`) → each rounded rectangle exactly;
  - the cos/sin geo form gives the same table as degrees;
  - the table's columns and meanings (p_word = 0.8 = core chance, core_mass = 0.8·n, kind means);
  - bad inputs refused; an empty page gives an empty table;
  - `measure_spreads` recovers distance 1.5, angle 2°, tag 0.3 and background 8. It gives cell ≈ 1 for iid votes
    and ≈ 9 for 3×3 blocks;
  - `fit_word_prior` recovers −6 from 3,000 simulated decisions (±0.4), and refuses decisions of one kind;
  - `extra_word_gain` is > 1000 on two words and < 20 on one;
  - `word_gain` is > 1000 on a word and < 20 on background px.
- **Mutants** (scratch script; each behaviour removed in turn):

  | Behaviour removed | Tests that failed |
  |---|---|
  | splits | the minority test |
  | the whole search | 1/2 noisy, the minority test |
  | joins | the long-word test (14 boxes instead of 1, checked separately: no earlier test exercised joins, so I added it) |
  | background (proposals judged) | 2/2 noisy |
  | cell in the search | 1/2 noisy |
  | the backwards reading | 1/1 |
  | page clip | 2/2 edge |

  Not caught: cell in the proposals' acceptance alone (0/2). No test isolates that.
- **Page-like synthetic.** 1200×1200 px (1.44 Mpx), every px voting, background at 0.001, 64 words at 0/30/90°,
  noisy votes:
  - 13.1 s (9 µs a px) after making the proposals' EM per-basin; before, 60.5 s;
  - peak RSS 1.35 GB, including the test maps;
  - all 64 words found (IoU ≥ 0.994).

- Suites: `test_build.py` 43 passed; the full `uv run pytest -q -n 2` 1734 passed, 3 skipped, 3 xfailed. Commit f7819c9 (my two paths only).

## Concerns

1. **The spreads changed from "corner" to distance + angle.** The reason is above. The brief says "corner and tag
   spreads"; I read that as "the measured spreads of a vote", and say so here.
2. **Junk boxes.** At background chance 0.001 and prior −20, the 1.44 Mpx page also had 140 background boxes with
   p_word ≈ 0.001: about 1 per 10⁴ px, so about 1,700 on a real page. They are kept with their chance, by the brief.
   How many appear depends on `word_prior` and on the background model. So Task 8 must fit the prior on the 0→1
   decisions (`word_gain` on background proposals), not only on splits.
3. **The background model is assumed:** half-normal distances, any direction, tags about 0. The net's geometry is
   untrained off the cores, so Task 8 should compare its real background votes with this model. A mismatch shows up
   as junk words.
4. **Cost on a real page (5100×3300 = 17 Mpx).** At 9 µs a px, that is about 2.5 min. Memory is about 2-3 GB:
   page-sized basin arrays, plus vote indices for every px with w > 0 (every px, since nothing cuts them).
   Run one page per process.
5. **The search is greedy.** Join candidates are words whose box bounds meet. In the search, px ownership is hard:
   - px of rejected basins stay background;
   - a split divides px by their posterior.

   Each word's output responsibilities come from its own px alone, so `p_word` and `core_mass` near a touching
   neighbour are approximate.
6. **One prior for both decision kinds** (0→1 and 1→2), as the brief says. The mix of decisions Task 8 feeds
   `fit_word_prior` shapes the fit.
7. **No longer here:** `choose_level`, the placement and tile stitching all belong to Task 8.


# Fix round 2: the re-review's nine rulings

Status: **DONE_WITH_CONCERNS**. Files: `wordread/words.py` and `tests/test_wordread_words.py` only. Commit
36ab064.

## What changed, ruling by ruling

1. **cell and the spreads, from each word's own fit (not Terra's box).**
   - `measure_spreads` fits each Terra word itself: the votes' mean direction (each vote read the nearer way), the
     mean implied edges, and the mean tag. Terra's direction only starts the fit.
   - **distance and cell come from a variogram of the implied edges within words, over every lag:**
     - only differences of px of one word enter, so a word's offset from Terra cancels;
     - half the mean squared difference at lag δ is σ²(1 − ρ(δ));
     - ρ is Gaussian in the lag with scale ℓ, fitted by least squares weighted by the pair counts (one bounded scalar
       search for ℓ, with σ² in closed form);
     - distance = σ, and cell = (Σ_k exp(−k²/2ℓ²))², the lattice sum of ρ.
   - **angle and tag** are within-word root mean squares. Each word's degrees of freedom are n less its correlated
     share (Σ over pairs of ρ / n, computed from its own mask), which is exact for finite words.
   - **`fit_word_prior` now fits a logit:** slope × gain/cell + log_prior. It is a fractional logistic regression by
     Newton steps with a backtracking guard, and returns `WordPrior(slope, log_prior)`.
   - Test: a per-word offset of sd 3 px gives distance 1.5 (the noise), cell ≈ 1 (±0.25), angle 2 and tag 0.3. The
     round-1 estimator (mean error against Terra's box) fails that test: mutant checked.
   - Test: noise alike in 3×3 blocks gives cell ≈ 9 (±35%).
2. **The reversed share is measured, not ½.**
   - `measure_spreads` counts the reversed readings: q = (count + ½)/(votes + 1) (Krichevsky–Trofimov, so never 0 or 1).
   - The model uses log(1 − q) for one reading and log q for the other; the likelier reading is decided inside the
     likelihood.
   - Tests: a reversed share of 0.2 is measured as 0.2 (±0.02); a word whose px a third read backwards stays one word,
     read as the majority reads it.
3. **The core logit.**
   - `core_words(core_logit, ...)` reads log σ(x) and log σ(−x), so no chance is 0 or 1 and nothing is −∞.
   - Every px votes; there is no "w > 0" set any more.
   - New tests:
     - two noisy touching words sharing a tag, at chance 0.9 among background at 0.001: two boxes, p_word 0.9 ± 0.02,
       logodds > 100;
     - a noisy single word at 0.9 among background: one box, p_word 0.9 ± 0.02.
4. **Ownership.**
   - **One global soft EM** runs after the search over all words plus the background. A word's votes are those within
     PRECISION_SDS = 8.57 distance spreads of its box's bounds, the distance at which its density falls below double
     precision of its peak. A vote is shared among its candidate words by their masses.
     - `p_word` and `core_mass` come from its soft responsibilities.
     - New column `word_logodds` = slope × (−Σ log(1 − r))/cell + log_prior: leave-one-out, the other words held.
       log(1 − r) is computed exactly: each vote's top pair is excluded by name, never by log1p(−r).
   - **Absorb.**
     - Under the score, absorbing a free proposal into a word never lowers it: a word only adds probability to votes
       the background explains.
     - Built as separate moves, every free basin in a word's bounds offered a minuscule positive gain. The search took
       hundreds of noise-level moves (±1e-10, caught by the score test) and was slow (one test took 91 s).
     - So each word takes them in one batch when it is made: each free basin with votes in a word's bounds goes to
       the word it gains most under. The gain is Σ(logaddexp(a, b) − b), never below 0, and is booked exactly.
     - The word then keeps a refit only if it is higher than the booked likelihood.
5. **Termination.**
   - Split and join moves are scored by their realized change: the parts or the union are refitted before the move
     is weighed.
   - A move is made only if it raises the tracked total score.
   - Test (`scores` hook, the score recomputed from the configuration after each move): it rises strictly with every
     move, on a long word whose scattered proposals are joined.
6. **The direction step.**
   - The angle update now accounts for the distances' dependence on φ: a Gauss-Newton step on id·Σr·dev² + ia·Σr·err²,
     using dpa/dφ = pb and dpb/dφ = −pa.
   - It is also kept only if the expected complete log likelihood rises; otherwise the best of "edges only" and
     "no change" is kept (generalised EM).
   - Test: a 200 px word whose direction head is 10° off while its distances agree with 0°:
     - the trace never falls;
     - the fit settles between, at about 0.5° (in (0, 5) as asserted), as the precisions weigh the two terms.
   - The old angle-only step (with the guard) fails this test: it is stuck at 0°.
   - The guard itself is not exercised here: Gauss-Newton never overshot. It is kept as the monotonicity guarantee.
7. **The prior fitted on the decisions `core_words` makes.**
   - `prior_decisions(core_logit, geo, tag, spreads, word_id, prior=None)`:
     - "word" for every proposal, with gain (ll1 − ll0)/cell and as label the share of its core-weighted vote mass
       inside Terra words;
     - with a prior, also "split" and "join" for every split and join the search weighs. The label is the chance the
       two parts are different Terra words: m_A·m_B·(1 − Σ_t a_t·b_t), from core-weighted vote-mass shares, with no
       cut.
   - Task 8 fits a first prior on "word" decisions alone, then on all of them.
   - `word_gain` and `extra_word_gain` are gone.
   - Tests:
     - proposals on words outscore the background's;
     - each lone word's split is labelled < 0.05;
     - the touching pair's join (kept apart) is labelled > 0.95;
     - `fit_word_prior` recovers slope 0.4 and log prior −6 exactly from fractional labels, and within 15% from 0/1
       draws;
     - it refuses labels a threshold parts. That check is exact: the MLE is infinite exactly then.
8. **Background tag mean:** measured (`Spreads.background_tag_mean`, one per channel). The background tag spread is
   taken about it. Test: (0.5, 0, −0.5, 0) is recovered.
9. **Cost on a page-sized synthetic:** see below.

**Test prior.** The tests' prior is now `WordPrior(1, −60)`, with the reason in the test file. Under the model, a
background vote far in its tail is best explained as a word of its own: its gain is log w + c + χ²₈/2. At log prior −20
a 240² page made 2-18 one-vote "words" with p_word ≈ 1e-6. They are kept with their chance, by design, but they broke
the tests' counts. At −60 that would need χ²₈ > 83. Task 8 fits the prior on decisions where these proposals are
labelled not words.

## TDD evidence

- **Red:** the rewritten tests against the round-1 module fail at collection (no `WordPrior`, no `reversed`).
- **Then, in order:**
  - 6 failures from tail junk (prior −20), cured by the stated test prior;
  - the score test's ±1e-10 steps, cured by batch claims;
  - a score test with no moves, moved to a case that has them.
- **Green:** 27 tests in `tests/test_wordread_words.py`.
  - The failure kinds: touching (exact); three noisy touching words; a noisy pair sharing a tag at w 0.9 among
    background; a noisy single word; a long scattering word; block-correlated votes needing their cell; a backwards
    third; a line; a strong minority; page edges; rotated words exact.
  - The fit: GEM monotonicity under disagreement; the search's score rises with each move.
  - Inputs: Task 5's targets; cos/sin; the table with `word_logodds`; bad inputs.
  - Task 8's tools: `measure_spreads` ×2, `fit_word_prior` ×2, `prior_decisions`.
- `test_build.py`: 43 passed. Full suite: 1 failed, 1757 passed, 3 skipped, 3 xfailed. The failure is in Task 8's uncommitted `tests/test_wordread_run_detector.py`: one Terra word has no box within 2 px. Its own perfect answer, fed to `core_words` directly, gives all 5 words exactly (nearest box 0.0 px, word_logodds 1.8e5 to 5.5e5), so the miss is downstream of `core_words` in its pipeline (perhaps the quarter filter). Not blocking, by the brief; reported.

## Cost (item 9)

A page-sized synthetic: 5100 × 3300 px, every px voting, the background at 0.001. It holds 1,638 words (codes,
street names, 0/90/30°, many touching pairs) and 2,000 fragments (8×8 patches at core 0.3). Votes are noisy
(1.5 px, 3°, tags 0.3), drawn per 2×2 px, with spreads to match (cell 4) and prior (1, −60). One process, one thread:

| Stage | Time | Peak RSS |
|---|---|---|
| The synthetic arrays alone | | 2.3 GB |
| Proposals: 265,073 basins | 44 s | 5.8 GB |
| Search: 3,593 words | 495 s | 5.8 GB |
| Global soft EM: 12.0 M (vote, word) pairs | 19 s | 11.5 GB |
| `core_words` end to end | 757 s (12.6 min) | 14.1 GB (cumulative over the script) |

- The boxes by p_word (display only): 1,668 near 0.95 and 1,925 at 0.3 (the fragments, kept with their chance).
- That is 30 more boxes near 0.95 than words painted. I have not looked into which, and give no cause.
- The search is the time and the global EM's per-pair arrays (about 15 floats a pair, plus temporaries) are the
  memory. Task 8 must run one page per process. Chunking the global EM by groups of words would cut its memory;
  not done here.

## Concerns

1. **PRECISION_SDS (8.57 sd) bounds each word's votes in the global soft EM.** It is numerical (below double precision
   of the word's peak), not a decision; the underflow-exact 38.6 sd would multiply the pairs by about 20. This needs a
   ruling if even that counts as a window.
2. **The correlation shape is assumed Gaussian** (one scale ℓ, a model family). The block test (a tent-shaped
   correlation) recovers cell 9 ± 35%. The angle and tag degrees of freedom use the distances' correlation.
3. **The tail junk.** A light-tailed background makes far-tail background votes into one-vote words unless the prior
   is low enough. Their number scales with px count (about 7 per 57,600 px at chance 1e-6 and log prior −20). Task 8's
   fit on "word" decisions must include enough background proposals, labelled 0, to place the prior. If real
   background votes are heavier-tailed than half-normal, it gets worse; measuring the tail is Task 8's.
4. **Absorb is batch claims, not separate moves:** absorbs only add, and as separate moves they were noise-level and
   slow.
5. **The guard on the direction step is not exercised by any test.** The Gauss-Newton step did not overshoot on the
   cases here.
6. **The search is greedy.** Joins are weighed only for words whose bounds meet. Splits are seeded at the
   worst-explained vote.

# Fix round 3: the spatial prior, Task 5's sign-free geometry, the measured biases

Status: DONE_WITH_1. **The page synthetic's numbers are not in.** The full run was stopped at the controller's wrap-up. The cause count
   of round 2's 30 extra boxes on the page is not established (see 7). One run of the committed script gives it.
2. **Time.** Each add, split and join is a full fit, and the page has 9,248 proposals. The fit is exact per move, so
   the remaining cost is the number of moves. The next exact lever is an analytic Hessian; I have none in place.
3. **Domain and numerical bounds that look like numbers.**
   - A box is ≥ 1 px each way (log sizes ≥ 0). Without it, a dead word in a split shrank toward a px's own square
     for 190 steps.
   - TRUST = 1 is a trust region on the Newton step (box lengths, log sizes, radians). The steps repeat, so it does
     not change where they end. Without it, a near-flat direction sent a step to e^700 px.
   - The weights' numerical 0 is sqrt(eps) of the largest eigenvalue.
   - The controller should rule on whether these count as hand-set.
4. **Exact boxes are no longer exact to 1e-6.** The spatial terms move an edge by hundredths of a px where px
   straddle it. The tests check them to EDGE_TOL = a tenth of the edge spread (0.05 px).
5. **The ECM test's claim changed:** from "2-8°, about half way" to "strictly between, at the likelihood's peak".
   The fit now finds about 1.3°, where the likelihood peaks.
6. **The full suite's 17 failures** are another task's uncommitted `choose` work, which does not import `words.py`.
   Also, `pyproject`'s `--dist loadfile` means `-n 2` runs a single file on one worker.. The controller called a wrap-up before the page synthetic finished.

Files:
- `src/sfzmaps/solver/wordread/words.py`
- `tests/test_wordread_words.py`
- `scripts/wordread/page_synthetic.py` (new)

Commits:
- 894ffdc: the round.
- e58759f: the fit's crawls cut. The tags take their exact CM step first, and a box is ≥ 1 px each way.
- 307b5d3: a move's evaluator sums the other words' parts once. Same likelihood to 2e-16.
- 4aa6409: the box's Newton step has an exact gradient. Its Hessian is the finite differences of that gradient
  (6 gradient evaluations instead of 21 values). Checked against central differences of the likelihood: relative
  error 5e-6, at the differences' own precision. The small page's boxes are the same to 1e-4 px.

## The signature (Task 8 calls it)

```python
core_words(core_logit, geo, tag, ink, spreads: Spreads, prior: WordPrior, *, kind_p=None, sheet="") -> pd.DataFrame
```

- `geo` is **(5, h, w), exactly as `textnet.model.word_geometry(out)` decodes the heads**: the centre offset x and y
  in px, the length and height in px, and the axis in degrees in [0, 180).
- `tag` is (k, h, w) and `core_logit` is the CORE logit. The argument names are unchanged.
- **`Spreads` fields:** centre, length, height, angle, tag, cell, edge, core_share, outlier, centre_bias (x, y, in
  heights), length_bias and height_bias (log), angle_bias (degrees), background_centre, background_length (mean, sd),
  background_height (mean, sd), background_tag, background_tag_mean.
  - Gone: distance, reversed, background_distance.
- `measure_spreads(pages)` takes the same pages (geo decoded, tag, word_id, words). It now also returns core_share,
  outlier and the four biases.
- `prior_decisions(core_logit, geo, tag, spreads, word_id, prior=None)` has the kinds "proposal" and, with a prior,
  "word", "split" and "join".
- `WordPrior(slope, log_prior)` and `fit_word_prior(gains, labels)` are unchanged.
- The table's columns are unchanged.

## The model now

- **A vote** follows Task 5's sign-free heads: the centre it points to (an offset in the word's heights), its log
  length, log height and axis (mod 180°), and its tag. Each dimension first has its measured mean bias taken off.
  - There is no reading sign. The reversed share and the two-reading sum are gone.
  - A box's first edge is its axis taken in (−90, 90].
- **Where a word's px lie:** q_k(p) is the word's box grown by a px's half width, convolved with a Gaussian of sd
  `spreads.edge` (measured) on each edge. It is 0 beyond EDGE_SDS = 8.21 sd, where it falls below double precision
  of its value inside. That is the word's support.
  - A box is at least a px long and high (see the speed fix).
- **The core head is evidence of where boxes are.**
  - A px is in some box with chance C = 1 − Π(1 − q_k).
  - Its output is w / core_share as likely in a box, and (1 − w) / (1 − core_share) out of every box (core_share
    measured).
  - So a box over px that the head calls background pays for each of them.
- **Which word's, in a box:** by fitted weights, a_k q_k / Σ a_j q_j.
  - The vote is then that word's with chance 1 − outlier, or an outlier with the background's density (outlier
    measured).
  - Out of every box, the vote is the background's.

How I got there. Each step was traced on the failing case, not guessed.
1. **The controller's m_k q_k f_k, normalized.**
   - With no per-px normalization, it is unbounded in the mass.
   - With normalization against 1 − w, a copy of a line's word paid 80 nats.
2. **A class prior w·C.** The minority case got 0 boxes. Inside the street name's box, the code's px had to be the
   name's or nobody's (add gain −5,840).
3. **Core evidence plus an outlier share.** The code's votes can now sit in the name's box as outliers until a code
   word takes them.
4. **Shares fixed at q_k / Σq.**
   - Copies of the code nested in the name each took share from the name: +166, +61 and +29 nats, so 4 boxes for
     2 words.
   - With fitted weights, a copy's weight only splits its original's, so a copy gains nothing from the shares.

## The fit

Each fit step:
- an E-step;
- the tags' exact CM step (each to its votes' mean);
- a Newton step on the weights (exact gradient and Hessian; checked against finite differences, relative error
  3e-10 and 4e-6);
- per word:
  - a Newton step on its box (gradient and Hessian by finite differences; eigenvalues taken as their size, and no less
    than the votes' curvature; within a trust region);
  - then a move towards its votes' own update.

Each move goes to the first of its full step, half step, quarter step and so on that raises the likelihood of the
whole model. A fall beyond LL_NOISE raises.

**Why the Newton step on the box.**
- Before it, each word moved only toward its votes' own update, which never sees the px.
- In the length test (votes ×1.4, uncorrected), the box stayed at 138 px against the painted 100.
- The likelihood itself preferred 100: 14k nats of background px in the box against 1.5k nats of votes.

## The rulings

1. **The breach is fixed with a spatial prior.**
   - It replaces PRECISION_SDS: the support above, with the same model in the search, the final fit and
     `word_logodds`.
   - **It pulls the box to its core px against long length votes** (the controller's request), through the box's
     Newton step.
   - Test: length votes ×1.4 with sd 0.3 (the spreads matching the noise, length and height).
     - Uncorrected (length_bias 0): one box, length within 3 px of the painted 100, IoU > 0.95.
     - Corrected (length_bias = log 1.4): the same.
     - The first version of the test gave 3 boxes. Its height spread said 0.05 while the generator's heights
       scattered 0.3. Three co-located boxes of different heights explained that scatter. The test was misspecified,
       not the model; fixed.
2. **Candidates by the support.**
   - Joins are weighed for words whose supports share area: rotated boxes, plus the px half width, plus EDGE_SDS edge
     spreads.
   - There are no claims. The shares do it.
3. **`word_logodds` is the true leave-one-out:** slope × (score with − without, the others held, every share
   recomputed) / cell + log prior, on the word's support.
   - The test checks it against `_score` for each word.
4. **`measure_spreads`:**
   - Each Terra word's own fit is the model's on its votes, by EM over own-or-outlier. The outlier share is measured
     from it.
   - The variogram and the RMS spreads are weighted by each vote's chance of being the word's own.
   - With Task 5's geometry the offsets are in the page's frame, so only the axis votes bear on the axis.
   - Test: axes 10° off give neither an inflated centre spread nor an inflated cell.
   - The biases are measured first and taken off every vote, as `core_words` takes them off.
5. **Monotone fits.** See "The fit".
   - Test: a 200 px word whose axis votes are 10° off its px.
     - No step falls.
     - The axis settles strictly between 0° and 10°, where the likelihood peaks along it: no higher ±0.1°.
       It is about 1.3°.
   - The old bound, 2-8° "about half way", was the peak along the votes' segment, not the likelihood's. With the box's
     Newton step the fit finds the likelihood's own peak, so the test now checks that.
6. **Small items.**
   - The score test recomputes the total from the parameters after every move (`_total`). It rises strictly.
   - No NaN where a mass underflows:
     - own_update returns None at no mass;
     - the basins' steps are masked;
     - log-space shares are −inf, never NaN.
   - Joins are fitted from both starts.
   - The test-prior comment's arithmetic is redone: peak +14.4, normaliser 7.1, logit(0.05) −2.9. So the gain is
     logit(w) + 24.4 + χ²₈/2.
   - The numerical guards are labelled in the docstring: EM_STEPS, EM_TOL, LL_NOISE, FIT_STEPS, SPREAD_ROUNDS,
     SPREAD_TOL, SHAPE_STEPS, LINE_TOL, FD_STEP, TRUST, the weights' numerical 0, CHUNK_PX, EDGE_SDS, the variogram
     bounds, the lattice span, and OpenCV's 4 sd.
   - A split's seed is the support's vote most left to the background, by core chance.
7. **The page synthetic and the extra boxes:** Not finished: the controller called a wrap-up.
   - The script is committed (894ffdc). It times core_words' stages inside one end-to-end run, matches boxes to the
     painted words, measures each box's share on other painted objects (claims across neighbours), and re-weighs
     the join of every pair of boxes on one painted word (weighed or not, and its exact score).
   - The full page (e58759f, seed 0) was stopped in the search after 45 min. It got through: 3,638 painted (1,638
     words, 2,000 fragments); proposals 29 s, peak 4.79 GB, 479,632 basins, 9,248 of them proposed as adds.
   - **The 30 extra boxes: not established on the page.** Round 2's 30 came from the old global soft EM, which is
     gone. What the traced cases showed:
     - **(a) duplicates never joined.** Real, under the q/Σq shares. Copies of a nested code each took share from
       the street name (+166, +61, +29 nats), and the same holds at any overlap. Fixed by fitted weights: a copy's
       weight only splits its original's (mutant-checked by the minority test).
     - **(b) splits of one painted word.** Seen only with misspecified spreads: the length test's heights scattered
       0.3 against a spread of 0.05, and 3 co-located boxes explained it. Not a fault of the model.
     - **(c) claims across neighbours.** None seen. The touching tests keep their boxes within 0.016 px.
   - The page count is still to be confirmed with the script (one run), on the faster code below.
8. **Cost:** Chunking and per-component factorisation are exact.
   - `_Votes.centres` and `measure_spreads` go CHUNK_PX votes at a time.
   - There is no global soft EM; `_final` refits each connected set of overlapping words alone.
   - The full page's peak was 4.79 GB after the proposals, against round 2's 11.5-14.1 GB.
   - **Time is the cost now.** Every one of the page's 9,248 proposals is weighed as an add by a full fit. On a small
     page (500 × 260, 14 words, 10 fragments, 40 proposals, 113 fits), one process, under the machine's load:

     | Version | Time |
     |---|---|
     | 894ffdc | 280 s (34 fits at EM_STEPS) |
     | e58759f | 39-57 s |
     | 307b5d3 | 43 s |
     | 4aa6409 | 28.5 s |

   - At 4aa6409's rate the full page would be roughly an hour, extrapolated from about 2.8 fits per proposal.
     That is not measured.

## The measured biases (controller's request 2)

- `measure_spreads` measures each vote dimension's mean error against Terra's word, over every word px:
  - centre_bias: the offset in heights less (c − p)/H, per axis;
  - length_bias and height_bias: the log size less the word's;
  - angle_bias: the axis less the word's, mod 180.
- They are measured on the training quarters' pages that Task 8 passes.
- `core_words` takes them off every vote before anything else. The spreads are then measured about the corrected
  votes.
- Test: length votes ×1.4 give length_bias = log 1.4 ± 0.01, with the others 0. The edge is still the px
  quantisation's √(1/12), so the bias does not stretch it.
- On the real NE numbers (×1.18 per px, ×1.44 per word), this takes off the mean log error, about log 1.18 per px. What
  remains of the per-word ×1.44 is the box-level pull above: the core px end where the word does.

## Exact boxes and the tests' tolerance

- Exact votes still give the votes' box. But the box's edges now sit where the whole likelihood peaks. Three things
  move them by hundredths of a px:
  - a neighbour's px inside its soft edge (two touching words: 0.005 and 0.016 px);
  - the code's ring inside the street name;
  - px straddling a rotated edge.
- The tests that checked corners to 1e-6 now check to a tenth of the edge spread (`EDGE_TOL` = 0.05 px).
- The rotated axis is checked to the angle that moves its ends by `EDGE_TOL`.

## TDD evidence

- **Red, in order** (each traced on its case, not guessed):
  - the round-2 tests fail on the new module at collection;
  - the first spatial build timed out (40 min, 2 tests done);
  - in the new vote space, a join sign error left 1 box for 2 touching words;
  - the split seed gave a minority with no second word;
  - the class priors (above) let duplicates pay, then left the code no box;
  - the Q cliff stopped rotation;
  - edge px at q = ½ over-split noisy words;
  - fixed shares let nested copies pay (4 boxes);
  - the votes-only M-step left the long box at 138 px;
  - two crawls, 171 and 190 steps:
    - the votes' line moved the box and the tag together, so each step a tiny share for the tag dragged the box
      back against the Newton step;
    - a dead word in a split shrank towards a px's own square forever.
    - Fixed by the tags' exact CM step first, and boxes ≥ 1 px.
- **Green:**
  - 32 tests in `tests/test_wordread_words.py` with Task 8's `test_wordread_run_detector.py`: 45 passed on 4aa6409 (418 s, one process); the words file alone took 434 s on e58759f, the longest the two ×1.4 length cases at
    159 s and 124 s.
  - `test_build.py` (43) and Task 8's `test_wordread_run_detector.py` (12) also pass on e58759f.
  - Full suite, once, on 894ffdc: 17 failed, 1832 passed, 7 skipped, 3 xfailed. All 17 failures are in
    `test_wordread_choose.py` (15) and `test_wordread_run_choose_pack.py` (2). Both are another task's uncommitted
    work (`choose.py` and both test files modified in the worktree), and neither imports `words.py`. Reported, not
    mine.
  - Note: `pyproject` has `-n auto --dist loadfile`, so `-n 2` runs one test file on one worker.
- **Mutants** (scratch scripts), each caught by its test:
  - holding the weights (shares fixed at q/Σq): the minority test;
  - no Newton step on the box: the ×1.4 length test;
  - the old class prior: the copy test;
  - no px half width: the edge-px test;
  - the responsibility-weighted split seed: the minority test;
  - no move, and the plain update: the ECM test.

## Concerns

1. **The page synthetic's numbers are not in.** The full run was stopped at the controller's wrap-up. The cause count
   of round 2's 30 extra boxes on the page is not established (see 7). One run of the committed script gives it.
2. **Time.** Each add, split and join is a full fit, and the page has 9,248 proposals. The fit is exact per move, so
   the remaining cost is the number of moves. The next exact lever is an analytic Hessian; I have none in place.
3. **Domain and numerical bounds that look like numbers.**
   - A box is ≥ 1 px each way (log sizes ≥ 0). Without it, a dead word in a split shrank toward a px's own square
     for 190 steps.
   - TRUST = 1 is a trust region on the Newton step (box lengths, log sizes, radians). The steps repeat, so it does
     not change where they end. Without it, a near-flat direction sent a step to e^700 px.
   - The weights' numerical 0 is sqrt(eps) of the largest eigenvalue.
   - The controller should rule on whether these count as hand-set.
4. **Exact boxes are no longer exact to 1e-6.** The spatial terms move an edge by hundredths of a px where px
   straddle it. The tests check them to EDGE_TOL = a tenth of the edge spread (0.05 px).
5. **The ECM test's claim changed:** from "2-8°, about half way" to "strictly between, at the likelihood's peak".
   The fit now finds about 1.3°, where the likelihood peaks.
6. **The full suite's 17 failures** are another task's uncommitted `choose` work, which does not import `words.py`.
   Also, `pyproject`'s `--dist loadfile` means `-n 2` runs a single file on one worker.


# Round 4: the mask, the curve guard and the outlier floor (urgent, for Task 8)

Status: **DONE**. Commit 814ab51, on top of 4aa6409 (the earlier implementer's last round-3 commit). Files:
`words.py` and `tests/test_wordread_words.py` only.

## The mask
- `measure_spreads` reads an optional `mask` (bool, the page's shape) in each page dict:
  - a word with any px outside the mask is left out whole;
  - the background is `word_id == 0 & mask`;
  - `core_share` is counted on the mask's px. The cut words' px inside the mask count as word px.
- With no mask, the spreads are the same as before, bit for bit (checked against HEAD's module on `_pages()`).
- The word loop groups the px by id once. A stable sort keeps each word's px in row order. Before, it scanned the
  whole page once per word, which is too slow on whole pages.
- `prior_decisions` takes arrays, not page dicts, so it is unchanged. Task 8's `masked_inputs` route (NOT_SCORED
  outside the area) covers it.

## The curve guard
- In `own_update`, where the height's curvature is 0 or the step is not finite, the height keeps its value. The test
  is exact: `curve > 0` and a finite step, with no tolerance.
- The 1 px fragment case raised `ZeroDivisionError` at HEAD and now passes. In that case the vote's chance underflows
  to the least double and its offset sits on its own px.

## Tests (red at HEAD, green after)
- A masked page gives exactly the cropped page's spreads (the words outside removed).
- A word cut by the mask is left out whole: the spreads are those with it wholly outside, apart from the core share.
- A mask of every px gives the page's spreads, and a mask of the wrong shape is refused.
- The 1 px fragment keeps its height and does not raise.
- Suites: `test_wordread_words.py` and `test_build.py`, 79 passed.

## The outlier floor (Task 8's 2e-308)
- The floor is reached even on the test file's synthetic pages, which hold no outliers. There it is right: the
  share's maximum-likelihood value is 0.
- On the real page (a-se-s0, p07, SE, 213 whole words) the main cause was upstream: the run's centre offsets carry
  no information.
  - x: true offset rms 66.6 px, predicted rms 4.6 px, correlation 0.04.
  - y: correlation -0.05.
  - train-a-se-s0.log's centre loss stays at 1.34 to 1.51 from iteration 250 to 4000.
  - So the "centre sd 2.1 heights" was the words' own extent: each word's residual is its px's position (slope 0.995
    along, 0.979 across; 0.10 heights left after that trend).
  - The 8,759 px cell is that smooth ramp read as correlation.
- The outlier term is also mis-specified. Its density is the background's for the whole vote, tag included, so a
  word vote whose geometry is off but whose tag is its word's is never likelier as an outlier.
- Both points went to the controller. Round 5 replaced the outlier (below).


# Round 5: calibrated votes, a heavy-tailed vote, shared errors, the edge from the core px, the reach, the search

Status: see the end. Commits: 0b718a6 (A, B and C together: the items share `words.py`'s vote and spatial terms).
Files: `words.py`, `tests/test_wordread_words.py`, `scripts/wordread/page_synthetic.py`.

## A1. Each vote dimension calibrated against Terra
- The model, as the controller agreed: the forward fit pred = α + β·true + error. The likelihood reads the
  calibrated vote (pred − α)/β at scale s/|β|. In code it is the density of pred about α + β·true, so β = 0 needs no
  division.
  - At β ≈ 0 the term is flat in the word's parameters: the dimension carries no information, by its measurement.
  - The inverse fit, a + b·pred (the true value's mean given one vote), is used only for point placements: the
    proposals' centres and the split seeds. With the true values' spread unknown (sd inf, the default), the placement
    is the calibrated vote itself.
- Dimensions:
  - offset x and y, in heights, true = (c − p)/H: α `centre_bias`, β `centre_slope`;
  - log length (`length_bias`, `length_slope`) and log height (`height_bias`, `height_slope`);
  - axis: `angle_bias` only. A slope is not defined on an axis modulo 180°.
- `measure_spreads` fits α and β by least squares over every word px against Terra. The true values' mean and sd
  (`*_truth`) feed the placements. It raises where the true values do not vary (no slope).
- The cell is measured on the calibrated deviations within words, so a ramp across a word is no longer read as
  correlation.
- `calibration_report(spreads)` gives, per dimension: α, β, s, dof, the shared error's sd, the inverse a, b, σ, and
  R² = β²·var(true) / (β²·var(true) + s² + shared²).

## A2. The outlier replaced by a Student-t per dimension
- Each vote dimension's own error is a Student-t (offset x and y, log length, log height, axis, tag), with its scale
  and its dof measured. The tag stays the word's.
  - The separate outlier share is gone: the t takes the tails.
  - In `_mixture` a px in some box has its word's vote. Out of every box it has the background's.
- Measuring: per dimension, the t of the deviations about each word's location, by maximum likelihood.
  - The dof is searched on its log between DOF_BOUNDS (numerical). Where the Gaussian (dof inf) is likelier, it is
    taken.
  - Each word's location is the t-weighted mean.
  - The degrees of freedom are corrected by each word's correlated share.
  - The locations and the t are fitted in turn. SPREAD_ROUNDS = 200, and it **raises** if the log likelihood has not
    settled: nothing is returned silently.
- The fit: every vote update is a Gauss-Newton step weighted by the t's weights, (dof + 1)/(dof + z²). This is an
  exact CM step of the t as a scale mixture of Gaussians. It is used in `_own_updates` (shared by `own_update` and the
  basins' fits), in `vote_curvature`, and in the exact gradient.

## A3. The edge from the core px; per-word vote error as a shared error
- **edge** = sqrt(1/12 + b²): a px's own extent plus b, the spread of the core px's ends about each word's box.
  - Each word's box is fitted to its core px (L-BFGS-B), from Terra's rectangle. The likelihood is `core_words`'
    own: q convolved with the edge, against the core head's evidence w / core_share in a box and
    (1 − w)/(1 − core_share) out of one.
  - Then b is fitted over every word, and the two in turn until settled (raises otherwise).
  - The page dict takes `"core"`, the logit. Without it, Terra's own px are the core and the edge is √(1/12).
  - Each word's px: Terra's rectangle grown by its height, in the mask, in no other Terra word.
  - A word whose reach, at the edge found, passes those px is left out, as a word cut by the mask is, and the edge is
    found again without it. `words.MEASURED` holds the counts.
- **The shared error (random effect).** Per word and dimension (offset, log length, log height, axis), the votes share
  an error u, Gaussian about 0 with sd `*_word`.
  - Measured as the words' locations' mean square, less their noise's: s² (dof + 3)/(dof + 1) × Σρ/n², at least 0.
  - In `core_words`, each word fits its u by an exact CM step after each E-step, with u's prior counted once per
    word (cell times, in the votes' units).
  - The score counts the prior's log ratio to u = 0. There is no Occam factor: a fitted u gains at most about half a
    log of (1 + τ²·precision) per dimension, and the fitted logit takes that up.
  - The data show one: see the real page below.

## A4. The reach, and log(1 − q) exact
- The support's reach is where q, times the most any vote of the page can favour a word, falls below 2^-53. That
  most is the vote's core evidence times a word's peak density over the background's, the largest K over the page.
  It is computed once per page. EDGE_SDS is gone.
  - Test: at the reach, log Φ(−reach/edge) + log K_max = log 2^-53, and a vote the background finds less likely grows
    the reach.
- **log(1 − q) is exact everywhere** (`_log_out`: each axis's two tails, combined as (1 − q_a) + q_a(1 − q_b)).
  - Without the outlier term, a px deep in a box whose vote the box cannot explain was mis-scored: 1 − q under 1e-308
    rounded q to 1, so its likelihood fell by hundreds of nats.
  - Its gradient also overflowed (inf × 0) and stalled the fit.
  - The spatial terms now carry (log q, log(1 − q)) on a last axis. The exact gradient reads the out-of-box term
    through d log q where q < ½ and through each axis's tail where q ≥ ½. Every factor is then bounded.
  - Test: the gradient matches central differences to 1e-5 of its size on that case (a code's box half off its
    name), with Gaussian votes and with dof 3.
- Each basin's fit now starts at its votes' placed centres' mean weighted by core chance. With β = 0 a corner px
  start left the Newton fit stuck on half a word.

## B5. The search's cost
- Moves are weighed once and kept until a move made changes their votes (their region meets the changed one) or
  takes one of their words.
- Words never change in place, so a pair whose supports do not meet is tested once.
- Words (`_Words`) and moves are found through grids of INDEX_CELL px. That is speed only: the results equal a full
  scan.
- The best move comes from a heap. `_near` takes a `_Words` view, and `_final` and the leave-one-out loop use it.
- The full page: see C12.

## C. Smaller items
6. `_final` holds every other word as it stands.
7. `_fit`'s region covers every point of each word's lines: the bounds of each end's reach.
8. `_inputs` (and `measure_spreads`, per word) refuse non-finite geo and non-positive sizes.
9. `RECT_FLOOR` (1e-12) is labelled numerical.
10. The proposals' smoothing is the centre placement's spread at the core-weighted median of the placed heights.
11. Tests:
    - a per-word ±10° axis turn: the angle spread stays 2°, angle_word is 10°, the edge is unchanged;
    - after[t] == before[t+1] in the monotone test. The trace is now the words' gain, region-free;
    - shared errors: offsets shifted per word by 0.5 heights, and sizes by 0.2, are measured as the shared sds with
      the noise apart. In `core_words`, a word read with its shared error keeps its box on its core px while the
      votes are shifted (centre error under a third of reading it without).

## Tests (45 in the file; with test_build, 88 passed, 7.7 min)
- Calibration: length α 1, β 0.6 and an offset with β 0 are recovered, with the report's R² and b consistent.
- Biased head: length ×1.4 and axis 10° off are measured as intercepts, slopes 1.
- Shared errors (above); heavy tails: dof 3 recovered in every dimension (±20%), and the scales.
- Edge: exact words give √(1/12) (±2%); ragged core ends (sd 1 px, logits ±20) give 1.0 (±15%).
- A dimension with β = 0, its density the background's: the boxes are identical whether its votes point at the
  centres or are noise (1e-6), and they are the words.
- A known dimension sharpens: the same box, and moving it 1 px costs the votes' own precision more
  (n (β/Hs)²/2, ±20%).
  - The first version compared centre errors under weak core evidence. It failed because the core px alone place a
    symmetric word exactly (blind error 0.000 px), so precision is what differs.
- The exact gradient (above), the reach (above), and refused sizes (0, −1, NaN, inf).
- The round-4 mask and guard tests still pass.

## The real page (a-se-s0's predictions on p07, SE, answered blocks out; whole page, mask and core; 117 s)
214 whole words; 3 left out of the edge measure.

| Dimension | α | β | s (t scale) | dof | shared sd | R² (one vote) | inverse a, b, σ |
|---|---|---|---|---|---|---|---|
| offset x (heights) | −0.014 | 0.0026 | 0.090 | 10.3 | 0 | 0.006 | 0.006, 2.33, 2.67 |
| offset y (heights) | −0.002 | 0.0026 | 0.090 | 10.3 | 0 | 0.0005 | −0.001, 0.21, 0.80 |
| log length | 1.35 | 0.75 | 0.28 | 10.0 | 0.38 | 0.68 | 0.27, 0.91, 0.52 |
| log height | 1.35 | 0.60 | 0.20 | 12.7 | 0.32 | 0.68 | −0.46, 1.13, 0.52 |
| axis (degrees) | −1.16 | — | 1.46 | 1.83 | 13.4 | — | — |
| tag | — | — | 0.49 | 4.1 | — | — | — |

- cell 1,065 px (was 8,759 before calibration); edge 0.289 (b at its floor); core_share 0.108.
- Read with care:
  - **The offset carries next to nothing (R² 0.006).** The proposals' centres therefore come from the core px (each
    basin starts at its core-weighted middle).
  - **The words share large errors in size and axis.** The axis's 13.4° shared sd sits on a t of dof 1.8, so a few
    words have very wrong axes. A Gaussian shared error is likely too light-tailed for them.
  - **The edge at √(1/12)** means each word's box, fitted to its core px, takes up the core head's blur. The box then
    sits where the head's evidence puts it, not necessarily on Terra's rectangle.

## C12. The page synthetic
- **The full page (5100 × 3300, 3,638 painted) was not finished.**
  - Proposals: 55.6 s, 106,740 basins, peak 4.95 GB.
  - The search was still running after 33 min. I stopped it to profile.
- **A page of 1500 × 1000 (seed 1: 122 words, 178 fragments, the same density):**
  - 2,233 s end to end (search 2,210 s, final fit 17.5 s), peak 1.4 GB, 2,004 fits;
  - **every painted word has exactly one box**: no extra boxes, none missing, no box on nothing;
  - 170 of the 178 fragments have a box (8 fragments none);
  - 292 boxes in all.
  - Round 2's 30 extra boxes do not appear at this size. The full page is still to confirm it.
- **Where the time goes** (profile, 600 × 400 page, 420 s): the fits themselves, about 1 s each.
  - One-word fits: 0.82 s, 7.8 steps on average.
  - Two-word fits (every word's split is weighed): 4.6 s, 35 steps, a few at EM_STEPS.
  - About 680 likelihood evaluations a fit: the line searches halve down to LINE_TOL (21 tries) when no point rises.
- **Extrapolated, not measured:** the full page at about 7 h, from 300 objects at 37 min.
- The search's own overhead (B5) is no longer the cost. The next levers would be fewer halvings once a fit has
  settled, and splits weighed only where the votes disagree. Neither is done: both change the fit and need a ruling.
- The script fix: its `area` used `np.cross` on 2-D vectors, which NumPy 2 refuses. It also now writes the boxes
  before the matching.

## Concerns
1. **Speed.** A real page will take hours per page per process (above). Task 8 should know before running every
   sheet.
2. **The shared error is fitted (MAP), with no Occam factor.** It is bounded and taken up by the fitted logit, but it
   tilts "word" decisions by up to about half a log of (1 + τ²·precision) per dimension.
3. **`edge` changed meaning**, so the reach changed with it:
   - Old tests built with edge 0.5 still pass.
   - Real data give √(1/12), so supports are tighter and the reach is driven by K.
   - 3 of 214 words were left out of the edge measure (`words.MEASURED`).
4. **The bleed case.** Core evidence at chance 1 − 1e-9 along a line past a word, with the word's votes, makes the
   search slow (two-word fits at EM_STEPS) and drifts boxes onto the bleed. It is a synthetic extreme, not a test.
5. **Task 8's uncommitted `tests/test_wordread_run_detector.py:173`** still builds Spreads with `outlier`. Old
   spreads.json files need a refit.
6. **The dof search bounds (DOF_BOUNDS) and the edge's b search bounds are numerical.** A fit at the upper dof bound
   is replaced by the Gaussian when that is likelier.
