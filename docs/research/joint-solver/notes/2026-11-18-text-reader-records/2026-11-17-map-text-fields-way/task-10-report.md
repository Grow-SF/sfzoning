# Task 10 report: context packs and the reader's posterior with them

Status: DONE_WITH_CONCERNS. Commits (branch `solver`): `cd2bf9b` (packs, prior, posterior, fit, script, tests),
`028b943` (measured P(form | code), refit from the kept ink, likelihood profiles).

Files: `pipeline/src/sfzmaps/solver/wordread/context.py`, `pipeline/scripts/wordread/build_pack.py`,
`pipeline/tests/test_wordread_context.py`. Outputs (not committed) in `pipeline/data/solver/experiments/wordread/packs/`:
- `sf-2000.parquet`: the pack;
- `sf-2000.json`: its counts and spreads;
- `sf-2000-fit.json`: the fits, profiles and held-out scores;
- `sf-2000-fit-words.parquet`: 254 MB, the 4,402 read words with their CTC ink, kept for refits and for Task 11.

## What was built

**`Pack`** is one parquet per map. It holds no lot code: `build_sf_pack(2000, lot_codes=True)` raises before it loads
anything, and a pack's frames refuse any column but their own.
- **streets:** DataSF centrelines (`streetside.load_streets`), as name, base name, type, active and LineString, in
  EPSG:2227.
- **blocks:** the number and outline, the union of `lots/table.parquet` lots by `block_num`. Only `block_num` and
  `geometry` are read, never `zoning`.
- **codes:** the 2000 list (`table_codes`) with the reader's printed forms and their measured P(form | code)
  (`text-crops/prior.json` `forms` and `form_weights`, learned on the training labels).
- **spreads:** measured per fold.

Printed forms:
- **Streets:** each word and the whole name; ordinals without zeros and spelled out ("03RD" gives "3RD" and "THIRD";
  "23RD" gives "TWENTY-THIRD"); the type with and without a full stop; the name with the type.
- **Blocks:** the number without leading zeros. 3 of 13,700 Terra block words carry the zero.

**`prior(pack, box_map, kind_p, spreads)`** gives log masses over the listed strings plus `escape`.
- **Codes:** the era list, flat over codes, each code's mass split by P(form | code).
- **Streets and blocks: Bayes on where the box sits, every number measured.**
  - Streets: P(j | box) ∝ (1−m) f(d) g(θ) / 2L against P(none) ∝ m / (90·A).
  - Blocks: P(b | box) ∝ (1−m_b)/N × (p_in / area_b inside, else (1−p_in) f_out(d) / perimeter_b), against
    P(none) ∝ m_b / A.
  - f, f_out and g are Gaussian kernel densities: of log(1+d) by Scott's rule, and of the angle reflected at 0 and
    90.
  - m and m_b are the measured shares of words that don't name what they sit by.
  - L is the centreline length and A the blocks' convex hull.
- **Unclaimed mass** goes to the escape: the share of a kind no item claims, the `other` kind, and kinds with no list.
- **Listing floor:** strings with less than 1e-12 of the mass are not listed. Their lift would be at most about 1e-5
  at the smallest e searched. The search reach follows from the same floor.

**`posterior(ink_logp, prior, w, e)`** normalises log p(s|ink) + w·log prior′(s) over the named strings plus `other`.
- prior′ = E for every string and E + (1−e)·prior(s) for a listed one, with E = (1−e)·prior(escape) + e.
- So the pack only ever lifts strings. One it hardly favours reads exactly as by the ink alone.
- No pack, e = 1 or w = 0 gives the ink alone.
- It raises if a listed string has no ink, or if e ≤ 0.

**Other functions:**
- `ink_logp(reads, q, strings)`: exact CTC (`ctc.string_log_probs`) mixed over the two reading directions as
  `ctc.mixture` mixes them.
- `boxes_on_map` / `box_on_map`: page corners to the map's centre and axis.
- `measure_spreads`, `fit_prior`, `truth_loglik`.
- `refuse_answered`: answered blocks, `touch_blocks`, the held-out quarter. Both the fit and the spreads use it.

**Fit:** a grid in (w, logit e), then L-BFGS-B; deterministic. w is searched in [0, 64] and e in logit [−16, 16].

**Script** (`scripts/wordread/build_pack.py build | fit | refit`), CPU only, 2 processes × 1 thread:
- `build` makes the pack and the spreads ("all", and each quarter left out, from words touching no part of it).
- `fit` takes our boxes matched to Terra's words at IoU ≥ 0.5 (Task 1's `_overlaps` and `_best_word`). Each is read by
  its quarter's reader both ways, and the ink is computed for every listed string of its four fold priors plus Terra's
  string. Then per quarter q it fits on the other three quarters' words (none touching q or an answered block) and
  scores on q's own words.
- `refit` reuses the kept ink.
- **Reading check:** our ink of all code forms equals `text-read/reads.npz`'s mixture over 418,190 comparisons, max
  difference 1.1e-16. The reading is the stage's.

## Design changes from my first draft (measured, then changed)

1. **Survival p-values changed to Bayes.** I first gave each street a compatibility of S(d)·S(θ), the measured
   survival. Two problems:
   - The offsets have a second population: 8% of street words and 6% of block words name a same-text thing 100 ft to 6
     miles away (misreads, renamed streets). Its tail gave every street within kilometres about 2% compatibility.
   - S(median) = 0.5, so even a perfectly placed label sent 75% of its kind's mass to the escape.

   The Bayes form above fixes both. The far population is split off without a parameter: a word names what it sits by
   when its own street (block) is also the nearest of all. The rest make m and m_b.
2. **Posterior floor.** Under a prior of (1−e)·prior(s), a listed string with a tiny mass (a street a mile away) scored
   below any unlisted string: a filter. Now every string keeps E.
3. **Reach.** The kernel-width reach for blocks was the whole city (only 6 to 12 measured words sit outside their
   block). The reach and listing now follow the 1e-12 floor. Priors take about 2.8 ms a box and list about 200 strings
   (q90 243; 95 of them code forms).
4. **Code forms.** A flat split over a code's forms, raised to w ≈ 5, penalised codes printed several ways (NCDs) by
   up to (1/4)^5. The split is now the measured P(form | code), which raised every fold's log-likelihood by 4 to 20 nats.

## TDD evidence

- **RED:** `pytest tests/test_wordread_context.py` gave an ImportError (no `context`).
- **GREEN:** 19 passed.
  - The five brief tests: off-list strong ink wins; no pack equals the ink alone; no street in range falls to the
    escape (and a faintly listed street reads as the ink alone); `build_sf_pack(2000)` refuses lot codes; `fit_prior`
    refuses answered-block words, `touch_blocks` and the held-out quarter.
  - The street prior equals the Bayes formula and is near-zero across the street.
  - Street forms (ordinals, spelled, multi-word, UNNAMED).
  - The block prior equals the formula inside and 15 ft out, with escape only far away.
  - Codes flat with kinds mixed; measured form shares.
  - `posterior` refusals (missing ink, e = 0).
  - `fit_prior` raises w where the list helps, and escapes where it misleads.
  - `truth_loglik` equals `posterior` exactly.
  - The spreads measured on synthetic words (here / away / unnamed).
  - Pack round trip.
  - `box_on_map`.
  - `ink_logp` equals the CTC mixture, and −inf for a string too long for the frames.
  - KIND_ORDER and VOCAB match `textnet.data.KINDS` and `textread.model.VOCAB`.
  - The built pack holds no lot codes (skips without data).
- `tests/test_build.py`: 43 passed.
- **Full `uv run pytest -q`:**
  - before `cd2bf9b`: 1634 passed, 3 xfailed;
  - before `028b943`: 1664 passed, 1 failed, 3 xfailed. The one failure is
    `tests/test_wordread_choose.py::test_label_of_is_the_version_that_best_matches_terras_words`, another agent's file,
    with uncommitted WIP edits to `choose.py` and that test in the tree. Not mine.
- `pgrep -fl 'sfzmaps (build|stage)'` was empty before both commits.

## The SF 2000 pack

| | |
|---|---|
| street centrelines | 17,176 (16,378 active); 2,713 street names, 2,625 printing something (not "UNNAMED"); 7.03 M ft |
| blocks | 5,302 |
| codes | 57 codes, 95 printed forms (with measured P(form \| code)) |
| map area A | 1.44e9 ft² (the blocks' convex hull) |
| answered blocks refused | 715 |

Spreads ("all"; the folds are within a few % of it):
- **Street words:** 1,936, of which 1,854 print a DataSF street's form and 1,693 sit by it, so m = 0.126.
  - Distance: q50 3.6 ft, q90 11.7 ft, q99 43 ft, max 171 ft.
  - Angle: q50 1.2°, q90 5.0°, q99 17.9°.
- **Block words:** 3,983, of which 3,895 print a block number and 3,674 sit by it, so m_b = 0.078. p_in = 0.997; 12
  words sit outside their block (1 to 38 ft).
- **Reach:** streets 400 to 700 ft; blocks 4,300 to 25,600 ft (see concerns).
- **Street texts naming no DataSF street:** 74 distinct, e.g. FREEWAY, O'FARRELL (DataSF "OFARRELL"), JAMES LICK,
  GREAT HIGHWAY, ARMY, "AV.", "SCENIC W.".

## Fitted w and e (per held-out quarter, on the other three; Terra-matched words)

Matched words: 4,402 (3,159 block numbers, 650 zoning codes, 572 street names, 21 other).

| held out | w | e | log-lik pack | log-lik ink alone | train n (truth listed) |
|---|---|---|---|---|---|
| nw | 4.03 | 1.1e-7 (floor) | −4,279 | −6,640 | 2,878 (2,800) |
| ne | 4.89 | 0.0025 | −4,873 | −8,076 | 2,978 (2,862) |
| sw | 5.90 | 0.0026 | −4,120 | −7,623 | 2,581 (2,500) |
| se | 5.17 | 1.1e-7 (floor) | −3,970 | −6,850 | 2,382 (2,301) |

- **w is interior in every fold.** The profiles peak between 4 and 8 and fall steeply past 16.
- **e is at its floor in nw and se.** The likelihood there is flat below 1e-5 and 15 nats worse at 1e-3. In ne and sw
  the optimum is about 0.0026, 14 to 17 nats better than the floor. The data put e between 0 and about 0.003.

Held-out quarter's own words: top-1 agreement with Terra's string. "pack" decodes as the likeliest listed string, or the
best path when `other` wins. "ink" is the same decode at w = 0 over the same strings. Mean log p is of Terra's string.

| held out | kind | n | top-1 pack | top-1 ink | best path | mean log p pack / ink |
|---|---|---|---|---|---|---|
| nw | all | 730 | 0.792 | 0.553 | 0.552 | −2.15 / −4.25 |
| nw | block number | 621 | 0.825 | 0.573 | 0.570 | −2.05 / −4.27 |
| nw | zoning code | 82 | 0.683 | 0.512 | 0.524 | −2.05 / −2.70 |
| nw | street name | 23 | 0.391 | 0.217 | 0.217 | −4.44 / −8.90 |
| ne | all | 630 | 0.822 | 0.670 | 0.665 | −1.48 / −2.63 |
| ne | block number | 426 | 0.930 | 0.784 | 0.784 | −0.70 / −1.31 |
| ne | zoning code | 84 | 0.548 | 0.441 | 0.441 | −2.10 / −3.69 |
| ne | street name | 110 | 0.655 | 0.427 | 0.400 | −3.37 / −6.08 |
| sw | all | 1,030 | 0.872 | 0.719 | 0.719 | −1.66 / −2.04 |
| sw | street name | 142 | 0.676 | 0.479 | 0.479 | −3.65 / −5.82 |
| se | all | 1,229 | 0.874 | 0.732 | 0.734 | −1.50 / −2.36 |
| se | street name | 190 | 0.679 | 0.453 | 0.463 | −3.79 / −5.47 |

The rows not shown (sw and se block numbers and codes, and the `other` kind) are in `sf-2000-fit.json`. The `other`
kind is unchanged, as it should be: it has no list.

**For Task 11:**
- Read with `context.ink_logp(reads, q, strings)` over the listed strings of
  `prior(pack, box_on_map(corners, data.load_transform(sheet)), kind_p, pack.spreads[<held-out quarter>])`.
- Then `posterior(ink, prior, w, e)` with that fold's (w, e) from `sf-2000-fit.json`.
- `sf-2000-fit-words.parquet` already holds the ink of every matched box.

## Concerns

1. **The headline will barely move by construction.** The era list is flat over codes, so the closed posterior
   (code given a code, which the headline uses) changes only where:
   - a code form is also a nearby street form (BROADWAY, CASTRO, POLK, VALENCIA, UNION, SACRAMENTO, HAIGHT, …);
   - or the measured form split matters.

   The pack's gains above are in open decoding, mostly block numbers and street names. Task 11's keep rule asks the
   headline to rise. Expect that only if its decode uses the open posterior (pack decides code vs other/none).
2. **e is at its floor in two folds; w ≈ 4 to 6.** With e ≈ 1e-7, an off-list string in a box whose kind is confidently
   known keeps only E ≈ the pack's own escape share, raised to w. In a sure code box that is a margin of about 20 nats
   the ink must beat, close to a filter there (still never zero). One w for every kind is what the brief specifies, but
   2/3 of the training words are block numbers, so w is mostly the block fit. A per-kind w, or the reader's CTC
   overconfidence calibrated first, would be the next refinement. I have not built either.
3. **These are agreement with Terra, not truth.** Where Terra misread a block number, the pack, picking the block the
   box sits in, is counted wrong. The owner's answers are never used here (refused by code).
4. **The out-of-block sample is tiny** (6 to 12 words), so f_out is a wide kernel. Block reach runs 4,300 to 25,600 ft,
   and blocks far off are listed with masses near 1e-12. That is harmless to the result (the floor), but it is reading
   cost: about 100 extra strings a box.
5. **Choices that are mine, not fitted:**
   - the background uniform over the blocks' convex hull (water included);
   - streets labelled in proportion to their length;
   - distance and angle independent;
   - Scott's rule;
   - MIN_MASS = 1e-12;
   - TANGENT_FT = 1 ft;
   - the parameter-free "sits by its own" split.

   All are documented in the module. The fit's likelihood is the only check of them.
6. **Street-name coverage.** 4.2% of 2000 street words print no DataSF form: apostrophes (O'FARRELL), abbreviations
   (AV., W.), FREEWAY, old names. No other era's code list is held, so a 1980 pack has an empty code list (the codes'
   kind goes to the escape). Its old-name mismatch is for Task 11 to count.
7. **Another agent's uncommitted `textread/crops.py` edits were in the tree during the fit.** The reading check matched
   `reads.npz` to 1e-16, so `box_crop`'s default was unchanged.

## Fix round 1 (commit e159690)

Rulings taken: the ink calibrated apart from the context, the escape per kind measured, w applied to a code before its
forms share it, form weights per fold, the IoU screen, EM for the spreads on our boxes, `prior(..., spreads=)`
required, leave-one-out bandwidths, and `refuse_answered` refusing words touching the held-out quarter. The
no-answer-key guards are unchanged.

### What changed (`context.py`, rewritten)

**Posterior.** log p_tau(s | ink) + log lift(s), normalised over the named strings plus OTHER.
- lift(s) = Σ_k c_k Lift_k(s) / Σ_k c_k, with c_k = kind_p[k]·E_k.
- Lift_k(s) = 1 + Σ_items P(s | item)·((1 + odds)^w_k − 1).
- Each item's odds are tempered whole, then shared by its forms (ruling 3). In the test, four NORTH BEACH forms
  together get exactly one code's lift at any w, never (1/4)^w each.
- A string off every list has lift 1, a listed one more: the pack never pushes a string below its ink.

**Ink.** `ink_logp(reads, q, strings, tau)`: each frame's log-probabilities × τ, renormalised, then exact CTC, mixed
over the two directions.

**Escape per kind (ruling 2).** E_k = 1 / (1 + Σ odds), with nothing set by hand:
- **codes:** odds (1 − e_code)/(N·e_code), e_code being the share of training-quarter Terra code words off the list;
- **streets:** m and the densities f(d), g(θ) from EM;
- **blocks:** m_b;
- **other:** 1.

**Listing.** An item is the escape's when 1 + odds == 1 in float64 (it lifts nothing), and the reach follows from that.
Kernel densities are exactly 0 past KERNEL_ZERO bandwidths (float underflow). No MIN_MASS, no TANGENT_FT (a
centreline's direction is now its exact segment), no W_MAX, no e floor. τ and w are searched unbounded (log τ by Brent,
log w by BFGS).

**Spreads (ruling 6), per fold, on OUR boxes matched to Terra's words.**
- **Streets:** EM of a near component (weighted kernel densities in d and θ, reflected, leave-one-out bandwidth)
  against a far one (exponential d, uniform θ). It starts from every decile split, keeps the likeliest fixed point,
  and stops when the log-likelihood stops rising. On real folds it converges in 5 to 94 iterations; far words get
  responsibility exactly 0.
- **Codes:** e_code and P(form | code) (ruling 4) from Terra's code words of the training quarters.

**Blocks: a choice I made, for the controller to rule on.** A block word counts as near only inside the block it
names; outside it counts as escape (lift 1, read by the ink alone).
- I ran the same EM with a free near-outside density.
- On our boxes only 2 or 3 of about 2,300 found block words sit just outside their block. The other 80 to 120
  outside words are misreads 75 ft to 6 miles off.
- The likeliest fixed point over every start put the "near" density on those misreads (bandwidth 1,340 ft, reach 8
  miles).
- So near-outside cannot be identified from these words.
- The alternative would be a spill model constrained by continuity at the block's edge.

**Fit.** τ first on the ink alone (the reader's p(Terra's string | ink) at τ), then w per kind at that τ. The joint fit
is also reported. w_other is never fitted: the other kind has no list, so its weight has no effect.

**Script and data.** `build_pack.py build | read | fit`. The fit runs resumable per-job files and a
ProcessPoolExecutor. The first full run lost a Pool worker silently and hung.
Data in `data/solver/experiments/wordread/packs/` (not committed):
- `sf-2000.parquet`: the pack with the IoU-0.5 spreads per fold and "all";
- `sf-2000-words.parquet`, `sf-2000-reads.npz`: 5,744 boxes at IoU ≥ 0.3, read;
- `sf-2000-fit.json`;
- `sf-2000-fit-jobs/`.

### Fitted numbers

Fitted on the other three quarters, scored on the held-out one.

| IoU | fold | τ | w code | w block | w street | joint τ | joint w code / block / street |
|---|---|---|---|---|---|---|---|
| 0.3 | nw | 0.630 | 22.7 | 3.53 | 2.97 | 0.426 | 30.1 / 3.54 / 3.36 |
| 0.3 | ne | 0.614 | 16.8 | 3.95 | 2.96 | 0.418 | 22.5 / 3.71 / 3.46 |
| 0.3 | sw | 0.605 | 19.4 | 4.17 | 3.05 | 0.409 | 26.0 / 4.06 / 3.53 |
| 0.3 | se | 0.614 | 17.6 | 4.07 | 3.08 | 0.415 | 23.5 / 3.94 / 3.58 |
| 0.5 | nw | 0.736 | 21.7 | 2.56 | 2.42 | 0.509 | 25.8 / 2.49 / 2.67 |
| 0.5 | ne | 0.702 | 13.2 | 3.08 | 2.28 | 0.482 | 17.3 / 2.85 / 2.80 |
| 0.5 | sw | 0.685 | 17.6 | 3.37 | 2.51 | 0.468 | 22.0 / 3.05 / 2.84 |
| 0.5 | se | 0.693 | 15.4 | 3.22 | 2.61 | 0.472 | 19.1 / 2.96 / 3.05 |
| 0.7 | nw | 0.855 | 18.8 | 1.71 | 2.08 | 0.571 | 20.2 / 1.74 / 2.06 |
| 0.7 | ne | 0.781 | 13.2 | 2.29 | 1.94 | 0.501 | 15.4 / 2.18 / 2.32 |
| 0.7 | sw | 0.750 | 15.6 | 2.92 | 2.18 | 0.480 | 18.4 / 2.46 / 2.55 |
| 0.7 | se | 0.744 | 11.9 | 2.69 | 2.15 | 0.492 | 14.6 / 2.48 / 2.37 |

- τ < 1 everywhere: the reader's ink is overconfident, and more so on looser boxes. Calibrating raises the ink-only
  log-likelihood in every fold (e.g. 0.5 ne: −8,076 as read, −7,090 at τ, −3,571 with the pack).
- No search hit an edge. BFGS reported "precision loss" at its optimum in 5 of 12 fits; the joint fit converged in
  all.
- The joint fit trades a flatter ink (τ ≈ 0.45) for larger w. It is reported, not used.

Measured per fold at IoU 0.5:
- m = 0.066 to 0.092; near street words have median d 3.1 ft and median θ 1.1°;
- bandwidths 0.7 to 2.2 ft and 0.34 to 0.62°;
- m_b = 0.051 to 0.059;
- e_code = 0.029 to 0.036;
- streets are looked for within 27 to 61 ft. Blocks need only containment (reach 0).

### Held-out scores (agreement with Terra's string; IoU 0.5)

| fold | n | top-1 pack | top-1 ink at τ (same strings) | best path | mean log p pack / ink at τ / ink as read |
|---|---|---|---|---|---|
| nw | 730 | 0.799 | 0.551 | 0.552 | −1.66 / −3.55 / −4.25 |
| ne | 630 | 0.844 | 0.671 | 0.665 | −1.00 / −2.32 / −2.63 |
| sw | 1,030 | 0.888 | 0.718 | 0.719 | −1.18 / −1.91 / −2.04 |
| se | 1,229 | 0.883 | 0.733 | 0.734 | −1.11 / −2.13 / −2.36 |

- By kind (0.5): codes 0.70 to 0.93 against 0.44 to 0.62; blocks 0.82 to 0.93 against 0.57 to 0.80; streets 0.48 to
  0.68 against 0.22 to 0.46.
- The `other` kind (no list; 1 to 10 words) is unchanged or slightly worse in log p. The pack lifts the code reading
  of an `other` box whose kind_p is partly code.

**Reliability.** Bins are display only; the posterior is the decode's, pooled over folds at IoU 0.5. It is
overconfident:

| decode posterior | n | agreement with Terra |
|---|---|---|
| 0.9 to 1.0 (mean 0.996) | 3,202 | 0.913 |
| 0.8 to 0.9 (mean 0.85) | 106 | 0.58 |
| 0.7 to 0.8 (mean 0.75) | 85 | 0.48 |
| 0.6 to 0.7 (mean 0.65) | 100 | 0.39 |
| 0.5 to 0.6 (mean 0.56) | 100 | 0.44 |

Part of the gap is Terra's own misreads. Part is OTHER decodes, whose p is the whole off-list class while agreement
needs the best path to equal Terra's string. IoU 0.3 and 0.7 show the same shape (top bin 0.885 and 0.932 agree).

**The brief's example** (every listed string's ink 1e-6, an off-list string's 0.95), on each held-out code box's real
prior and kind_p:
- IoU 0.5: the off-list string wins in all 266, minimum margin 2.5 nats;
- IoU 0.7: all 73, minimum margin 6.5 nats;
- IoU 0.3: 428 of 434. The 6 losses, down to −4.3 nats, are on a string lifted by two kinds at once, a code form
  that is also a street form nearby.

**Real off-list cases** (Terra's string off the list at its place, its ink at τ beating every listed string):
- IoU 0.5: 19 of 62 keep their reading (last round 14 of 31; the count differs because the ink is now at τ and the
  lists changed).
  - by kind: block numbers 11 of 49, streets 5 of 8, other 3 of 4, codes 0 of 1;
- IoU 0.3: 20 of 85;
- IoU 0.7: 17 of 33.

The block cases are boxes inside a block whose number Terra didn't read. Either Terra misread, and the pack's override
is right, or the label belongs to the next block, and it is wrong. Agreement with Terra can't tell those apart. w_block
≈ 3 makes the containing block's lift about 13 nats.

### IoU screen (ruling 5): the levels do not agree

τ rises with the boxes' fit to Terra's words: about 0.61, then 0.70, then 0.78. w_block falls: about 4.0, then 3.2,
then 2.4. w_street falls: 3.0, then 2.5, then 2.1. w_code shows no trend (12 to 23). The held-out gains hold at every
level. I kept 0.5 for the pack's spreads and report this for the controller to rule.

My reading: at read time every one of our boxes is read, the poorly matched ones included. So the 0.3 set, or τ as a
function of box quality, may describe read time better than 0.5.

### Tests

- `tests/test_wordread_context.py`: 21 passed. They cover:
  - the five brief tests; refusal of touch_q;
  - the exact Bayes formulas for streets and blocks; the code escape;
  - form shares under w; lift ≥ 1;
  - `ink_logp` at τ;
  - the EM split (far misreads at r < 1e-6, unnamed at 0);
  - kernel density exactly 0 past its top and the leave-one-out bandwidth at its optimum;
  - the vectorised fit likelihood equal to `posterior`; fit ≥ ink alone;
  - the pack round trip with per-fold form weights; the built pack.
- `tests/test_build.py`: 43 passed.
- Full `uv run pytest -q`: 1,761 passed, 3 xfailed, 2 failed, both in `tests/test_wordread_run_detector.py`
  (another agent's new untracked work in progress).
- pgrep was empty before the commit.

### Concerns

1. **Block words outside the block they name are escape.** This is a modelling choice, explained above, and needs the
   controller's ruling.
2. **The IoU levels disagree on τ and w.** Needs a ruling.
3. **The pack still overrides off-list ink often.** Mainly block numbers (38 of 49 at 0.5), the cases where the box
   sits inside a block whose number Terra didn't read. It is either right or wrong; only the owner's answers
   (Task 11) can say which.
4. **The decode's posterior is overconfident against Terra** (0.996 against 0.913 in the top bin). τ calibrates the
   ink alone, not the combined posterior.
5. **w_code is large (12 to 23).** With codes flat among themselves, it only sets code against off-list and other
   kinds. It compensates for the escape's flat-within-class level against single strings. The brief's example still
   passes at 0.5 and 0.7.
6. **EM:** each start runs until its log-likelihood stops rising, and the likeliest start is kept. The cap
   (EM_MAX_ITER 2,000) was never reached after the fix. Before the fix, one even start ran 2,000 iterations without
   settling: its bandwidth jittered at every step.

## Fix round 2

Rulings taken: blocks outside the block they name stay escape (documented in the module); the IoU selection is
removed; joint against sequential is chosen by held-out NLL, with a held-out-fitted temperature on the combined
posterior.

### What changed

**No IoU anywhere: pairs weighted by the chance the box holds the word.**
- Every pair of one of our boxes and a Terra word on its sheet carries a weight: the product of the word's share of
  its ink inside our box's trimmed rectangle and our box's share of its ink inside the word's rectangle.
  `boxmodel.ink_shares` gives both, called both ways on the page's ink, as Task 15 uses it.
- 9,291 pairs with weight > 0, over 7,861 boxes and 7,899 words, total weight 5,259. Every paired box is read once.
- The weight enters everything:
  - the EM: its mixture share, its kernel weights r·v, its far scale and its log-likelihood; the starts are v-weighted
    deciles;
  - m and m_b;
  - the fit: Σ v log p;
  - the held-out scores: weighted means.
- Tests: weight 2 equals the row twice; a weight-0 pair leaves the EM unchanged; one box's rows share its ink.

**Impossible pairs: the one hard rule.**
- 83 pairs (weight 5.45, max 0.22) ask a box to emit a string longer than its CTC frames can hold, all of them
  fragments (e.g. "USE" of "COM. INDU./ MIX USE"). At any τ their log p is −∞.
- They are left out and counted (`readable`, `ctc_fits`: the characters plus a blank between doubled letters must
  fit the frames). This is the physically impossible case.
- The low-weight pairs that remain are mostly fragments too: weight < 0.5 covers 4,135 pairs, median log p of
  Terra's string −22 to −28. They pull τ down.

**The fit reports both modes.**
- Sequential: τ on the ink alone, then w.
- Joint: τ and w together.
- The mode is picked by the pooled weighted held-out NLL.
- Then one temperature T on the combined log posterior (`fit_temperature`, `tempered`). For quarter q it is fitted
  on the other three quarters' held-out predictions, so q's targets never set it.

### Fitted numbers

Pairs of the other three quarters, held-out scores on the fourth.

| fold | sequential τ | w code / block / street | joint τ | w code / block / street | T (from the others) |
|---|---|---|---|---|---|
| nw | 0.599 | 19.2 / 3.63 / 2.95 | 0.382 | 29.2 / 3.93 / 3.68 | 3.94 |
| ne | 0.588 | 15.2 / 3.79 / 3.00 | 0.371 | 23.6 / 3.79 / 3.90 | 3.85 |
| sw | 0.578 | 17.2 / 4.25 / 3.03 | 0.366 | 26.8 / 4.39 / 3.89 | 3.75 |
| se | 0.579 | 15.9 / 4.10 / 3.06 | 0.367 | 24.6 / 4.28 / 3.95 | 3.86 |

- w_other has no list and is not fitted. All τ searches converged. BFGS reported precision loss at its optimum in 6
  of 8 w fits.
- Measured: m = 0.081 to 0.100, m_b = 0.062 to 0.064, e_code = 0.029 to 0.036; street bandwidths 0.9 to 1.4 ft and
  0.6 to 1.0°.

**Mode, by pooled weighted held-out NLL:** sequential 2.886, joint 2.473, so **joint is kept**. With the cross-fitted
temperature, joint + T reaches **1.085**.

### Held-out results (weighted by the pair weight)

| fold | weight | NLL seq / joint / joint+T | top-1 joint pack | top-1 ink at τ | best path | codes | blocks | streets (pack vs ink) |
|---|---|---|---|---|---|---|---|---|
| nw | 819 | 2.95 / 2.26 / 1.16 | 0.783 | 0.489 | 0.489 | 0.58 vs 0.38 | 0.87 vs 0.54 | 0.26 vs 0.09 |
| ne | 953 | 3.51 / 3.10 / 1.42 | 0.735 | 0.551 | 0.551 | 0.48 vs 0.29 | 0.84 vs 0.68 | 0.52 vs 0.26 |
| sw | 1,109 | 2.86 / 2.48 / 0.95 | 0.841 | 0.660 | 0.660 | 0.74 vs 0.45 | 0.92 vs 0.76 | 0.54 vs 0.31 |
| se | 1,311 | 2.42 / 2.14 / 0.91 | 0.815 | 0.665 | 0.665 | 0.56 vs 0.28 | 0.91 vs 0.77 | 0.49 vs 0.33 |

Sequential's top-1 is close to joint's: 0.752 / 0.752 / 0.830 / 0.818. The temperature leaves the decode unchanged.

**Reliability** (the decode's posterior in display bins, pooled; weight, mean p, agreement):

| bin | sequential | joint | joint + T |
|---|---|---|---|
| 0.9 to 1.0 | 3,491, p 0.995, agree 0.884 | 3,345, 0.995, 0.895 | 1,893, 0.938, 0.967 |
| 0.8 to 0.9 | 176, 0.855, 0.395 | 198, 0.852, 0.515 | 701, 0.864, 0.938 |
| 0.7 to 0.8 | 138, 0.753, 0.359 | 156, 0.752, 0.417 | 212, 0.755, 0.869 |
| 0.6 to 0.7 | 163, 0.646, 0.335 | 190, 0.648, 0.401 | 152, 0.651, 0.781 |
| 0.5 to 0.6 | 158, 0.547, 0.367 | 187, 0.550, 0.418 | 270, 0.539, 0.573 |
| 0.4 to 0.5 | 32, 0.456, 0.143 | 63, 0.457, 0.272 | 209, 0.456, 0.452 |
| 0.3 to 0.4 | | | 159, 0.352, 0.362 |
| 0.2 to 0.3 | | | 197, 0.244, 0.548 |
| 0.1 to 0.2 | | | 313, 0.149, 0.393 |

- Both modes are overconfident, so per the ruling the temperature was added.
- With it, the NLL halves (2.47 → 1.09). The reliability turns mildly underconfident above 0.6 (0.94 claimed against
  0.97 agreement at the top) and below 0.3; it is about right from 0.3 to 0.6.
- One temperature can't fix both ends.

**Off-list strings with stronger ink** (pooled):
- Sequential: 17 of 89 pairs keep their reading (weight 12.6 of 65.1).
- Joint: 6 of 97 (weight 4.4 of 69.6). Of these, block numbers 3 of 75, streets 2 of 17, other 1 of 4, codes 0 of 1.

**The brief's example** (listed ink 1e-6, off-list 0.95, on each held-out code box's real prior):

| fold | sequential: wins / boxes (min margin) | joint: wins / boxes (min margin) |
|---|---|---|
| nw | 217 / 217 (+0.3) | 208 / 217 (−5.6) |
| ne | 278 / 285 (−4.9) | 276 / 285 (−12.4) |
| sw | 111 / 119 (−4.9) | 107 / 119 (−12.2) |
| se | 162 / 173 (−2.7) | 148 / 173 (−9.2) |

The fragment-heavy weighting raised w_code, and joint's larger w makes the pack override stronger ink in 4 to 14% of
code boxes. This is the cost of the NLL choice; see concern 2.

### The block numbers the pack overrules

Joint mode, held out: block-number pairs whose Terra string is off the list, whose ink beats every listed string, and
that the pack overrules. 72 pairs, weight 53.4. Written to `experiments/wordread/packs/block_disagreements/`
(`summary.json`, `index.json`, and 10 example crops, each `*-crop.png` ours and `*-around.png` the page with our box in
red).

- **Edge distance:** they are not near a block edge. Distance from the box centre to the edge of the block holding it:
  q10 39 ft, median 79 ft, q90 112 ft; 1 within 25 ft, 12 within 50 ft.
- **How Terra's number relates:**
  - 46 name another block that exists today, median 2,630 ft away (q10 113 ft, q90 29,000 ft);
  - 23 name no block today;
  - 3 are the same number with another suffix;
  - in 9, the pack's block has a lettered sibling (a split block).
- **33 of the 72 (weight 24.7) are one character from the containing block's number:** 70 against 76, 966 against 968,
  2964 against 2984, 3903 against 3923, 3287 against 3267, and 4178 against 417B.
- **The two crops I looked at agree with the pack, not Terra.**
  - Box 726, p02: the print reads 417B; Terra wrote 4178 (B read as 8).
  - Box 5095, p08: the print reads 4382, between 4352 and 4381; Terra wrote 4362.

  So many of these "losses" are the pack correcting Terra's misreads. The owner can check the 10 crops.

### Tests

- `tests/test_wordread_context.py`: 25 passed. New:
  - the impossible truth is left out and counted;
  - weight equals repeated rows, and one box's rows share its ink;
  - a weight-0 pair leaves EM unchanged;
  - a temperature undoes synthetic overconfidence.
- `tests/test_build.py`: 43 passed.
- Full `uv run pytest -q -n 2`: 1,769 passed, 3 skipped, 3 xfailed, 5 failed. All 5 are in
  `tests/test_wordread_words.py` and `tests/test_wordread_run_detector.py`, other agents' uncommitted work in progress.
- Commit `675d8ab`; pgrep was empty before it.

### Concerns

1. **Fragments dominate the low-weight pairs.** Weight < 0.5 covers 4,135 pairs, median log p −22 to −28. They pull τ
   down (0.58 sequential, 0.37 joint) and w_code up. This is what weighting by the holding probability gives.
2. **Joint wins on NLL but overrides stronger ink more.** It keeps 6 of 97 off-list readings, and in the brief's
   example the off-list string loses in up to 14% of code boxes, with margins down to −12 nats. Sequential is a
   little worse on NLL (2.89 against 2.47) and on top-1, but keeps more (17 of 89; brief example 768 of 794 boxes).
3. **One temperature on the combined posterior halves the NLL** but leaves the reliability uneven (underconfident at
   the top and below 0.3).
4. **The block overrides look mostly like Terra's misreads** (one-character differences, crops agreeing with the
   pack). Only the owner's answers (Task 11) score them as truth.

## Fix round 3

The ruling: a pair's target is the part of Terra's string inside our box.

### What changed

I built the expectation, not the "mostly inside" substring.
- **Characters' extents:** a word's characters are spread evenly along its rectangle. No per-character widths are
  measured.
- **Each character's share inside our box** (`char_shares`): the share of its ink pixels inside our box. Each of the
  word's ink pixels belongs to the character it projects onto along the reading. A character with no ink (a space)
  counts by the share of its extent's area.
- **The targets** (`substring_targets`): every contiguous run of characters, with
  P(run) ∝ Π share (inside) × Π (1 − share) (outside), conditioned on a non-empty contiguous run.
  - Spaces at a run's ends are dropped, and equal strings add.
  - A run with probability exactly 0 is no target. So a word wholly inside (every share 1) keeps its whole string,
    unchanged.
- **The pair's likelihood** (`_Words.loglik`): log Σ_t P(t)·post(t), under the same posterior. The temperature fit
  and the held-out NLL use the same expectation (`target_logp`).
- **The pair weight** stays the product of the two ink shares.
- **Counts:** 4,229 of the 9,291 pairs (weight 2,967 of 5,259) hold their whole word. With parts as targets, no pair
  is impossible any more (0 left out, against 83 last round).
- **Held-out scoring:**
  - top-1 is the expected agreement, Σ_t P(t)·[decode = t];
  - the off-list and keep facts are those of the likeliest part;
  - the block analysis is restricted to pairs that hold their whole word.

### Fitted numbers

| fold | sequential τ | w code / block / street | joint τ | w code / block / street | T (from the others) |
|---|---|---|---|---|---|
| nw | 0.787 | 13.9 / 2.52 / 1.17 | 0.674 | 14.5 / 2.44 / 1.26 | 3.95 |
| ne | 0.807 | 8.5 / 2.10 / 1.15 | 0.689 | 9.1 / 2.04 / 1.24 | 3.90 |
| sw | 0.750 | 11.2 / 3.15 / 1.14 | 0.649 | 11.9 / 2.98 / 1.24 | 3.95 |
| se | 0.757 | 10.8 / 3.03 / 1.11 | 0.651 | 11.4 / 2.89 / 1.22 | 3.94 |

The cause is confirmed: with parts as targets, τ rose from 0.58 to about 0.78 (sequential) and from 0.37 to about 0.67
(joint). w_code fell from 15–29 to 9–14, and w_street from about 3 to 1.2. The two modes are now close.

**Mode, by pooled weighted held-out NLL:** sequential 2.109, joint 2.059, so joint is kept. With the cross-fitted
temperature it reaches 1.012.

### Held-out results (weighted; top-1 = expected agreement with the part of Terra's string inside the box)

| fold | NLL seq / joint / joint+T | top-1 seq / joint | ink at τ | best path | joint codes / blocks / streets (vs ink) |
|---|---|---|---|---|---|
| nw | 2.28 / 2.20 / 1.13 | 0.668 / 0.670 | 0.515 | 0.516 | 0.42 vs 0.27 / 0.76 vs 0.58 / 0.17 vs 0.19 |
| ne | 2.74 / 2.59 / 1.20 | 0.663 / 0.662 | 0.565 | 0.562 | 0.32 vs 0.28 / 0.80 vs 0.68 / 0.40 vs 0.36 |
| sw | 1.72 / 1.72 / 0.86 | 0.726 / 0.725 | 0.640 | 0.641 | 0.61 vs 0.37 / 0.83 vs 0.74 / 0.31 vs 0.29 |
| se | 1.87 / 1.87 / 0.93 | 0.713 / 0.711 | 0.660 | 0.661 | 0.44 vs 0.19 / 0.81 vs 0.77 / 0.37 vs 0.34 |

Top-1 is lower than last round because parts are now targets: a fragment like "MISSI" is rarely on any list.

**Reliability** (pooled; weight, mean p, agreement):

| bin | sequential | joint | joint + T |
|---|---|---|---|
| 0.9 to 1.0 | 3,573, 0.993, 0.768 | 3,519, 0.993, 0.772 | 1,909, 0.938, 0.930 |
| 0.8 to 0.9 | 178, 0.854, 0.312 | 197, 0.856, 0.347 | 667, 0.864, 0.826 |
| 0.7 to 0.8 | 143, 0.745, 0.269 | 129, 0.750, 0.327 | 275, 0.753, 0.612 |
| 0.6 to 0.7 | 131, 0.652, 0.306 | 157, 0.649, 0.282 | 239, 0.651, 0.496 |
| 0.5 to 0.6 | 127, 0.555, 0.276 | 145, 0.551, 0.274 | 264, 0.546, 0.381 |
| 0.4 to 0.5 | 26, 0.458, 0.215 | 30, 0.463, 0.186 | 333, 0.452, 0.323 |
| 0.3 to 0.4 | | | 248, 0.354, 0.230 |
| 0.2 to 0.3 | | | 145, 0.256, 0.184 |
| 0.1 to 0.2 | | | 103, 0.155, 0.114 |

Both modes are overconfident. With the temperature, the top two bins are close (0.94 against 0.93; 0.86 against 0.83),
but the middle and lower bins stay overconfident (0.45 against 0.32). One temperature can't fit every bin.

**Off-list strings with stronger ink** (pooled; most are now fragments, which are off every list):
- **Sequential:** 612 of 1,077 pairs keep their reading, weight 251.5 of 458.1 (**55%**).
  By kind: codes 14 of 77, blocks 231 of 454, streets 356 of 531, other 11 of 15.
- **Joint:** 548 of 1,083, weight 227.1 of 457.7 (**50%**).
  By kind: codes 11 of 80, blocks 216 of 454, streets 310 of 533, other 11 of 16.

**The brief's example** passes in every held-out code box in both modes:
- sequential: 796 of 796, minimum margins 5.0 / 6.1 / 2.1 / 2.4 nats (nw / ne / sw / se);
- joint: 796 of 796, minimum margins 5.4 / 6.4 / 3.1 / 2.9 nats.

**Plainly:** the NLL choice (joint) still overrides strong off-list ink somewhat more than sequential (50% kept
against 55% by weight; 548 against 612 pairs), on a narrow NLL margin (2.06 against 2.11). The brief's example no
longer separates them: both pass everywhere. I haven't changed the selection rule.

**Block disagreements** (whole-word pairs, joint): 43 pairs (weight 28.8).
- Median 86 ft from the block's edge; 5 within 50 ft.
- 27 name another block today, 14 name no block today, 2 differ in suffix; 15 are one character apart.
- The examples and crops in `block_disagreements/` were rebuilt, and are the same heaviest pairs as before (e.g. 4178
  against 417B).

### Tests

- `tests/test_wordread_context.py`: 27 passed. New:
  - a box holding part of a word is asked only for that part: whole word unchanged; "ABCD" with shares (1, 1, 0.5, 0)
    gives AB 0.5 and ABC 0.5; a space at a run's end merges; character shares from ink, with a space by its area;
  - the fragment likelihood equals log Σ P(t) post(t).
- `tests/test_build.py`: 43 passed.
- Full `uv run pytest -q -n 2`: 1,777 passed, 3 xfailed, 7 failed. All 7 are in `test_wordread_run_detector.py` and
  `test_wordread_words.py`, other agents' work in progress.
- Commit `2e099e6`; pgrep was empty before it.

### Concerns

1. **No per-character widths are measured.** The characters are spread evenly, so in narrow-letter words (I, 1) a part's
   chances shift a little.
2. **Joint against sequential is close on NLL; joint keeps 5 points less off-list ink.** For the controller and owner.
3. **The combined posterior is still overconfident in the middle bins** even with the temperature.

## Fix round 4

Rulings taken:
- T is fitted on the read-time vector and persisted per fold, with a read recipe;
- multi-word boxes are measured, then get one target a box;
- the intent check runs on code, block and street boxes;
- the minor items.

Everything below is held out by quarter, weighted, and pooled over the four folds unless a fold is named.

### What changed

**The read-time temperature.**
- At read time a box's posterior names the strings its prior lists, the reader's best path and OTHER
  (`posterior` already takes extra strings).
- T is fitted on that very vector. Each target is scored at its slot: its own when named, else OTHER's
  (`slot_targets`). A run that no read can be (too long for the frames) also goes to OTHER's slot.
- Per quarter q and per mode, T is fitted on the other three quarters' held-out vectors, leaving out every row that
  touches q. q keeps the mode with the lower NLL there after that mode's own T.
- `fit_temperature(rows, words, weights, held_blocks, quarter)` now runs `refuse_answered` like `fit_prior` does.
- The module says in one sentence that T is cross-fitted: q's targets never enter T(q)'s likelihood, though they did
  help fit the other folds' τ and w.

**Persisted per fold, with a loader.**
- `ReadFit(mode, tau, w, temperature)` is stored in the pack parquet as layer "fit", one per quarter, next to that
  quarter's spreads.
- The file carries `LAYOUT = 2`. `Pack.load` raises `PackLayoutError` on any other layout.
- The built-pack test skips only on `PackLayoutError`. Any other loader error fails it. It now runs against the new
  pack and checks each fit.

**The read recipe (for Task 11):**
`context.read_posterior(pack, fold, ink_logp_fn, box, strings=()) -> dict[str, float]`
- `box` is {x, y, axis, kind_p}.
- `ink_logp_fn(names, tau)` returns {name: log p}, e.g. `lambda n, t: ink_logp(reads, q, n, t)`.
- `strings` are the extra names, such as the best path.
- It runs prior with `pack.spreads[fold]` → ink at τ → `posterior` at w → `tempered` at T.
- `fold` must be the box's own quarter. "all" and a quarter with no fit are refused. With no pack it returns the ink
  alone.

**One target a box (`box_targets`).** Rows are now boxes, not (box, word) pairs (`build_pack.py rows` writes
`sf-2000-rows.parquet`). The spreads are still measured on pairs.
- Character shares are measured along the reading only (`char_shares`, between the box's two ends). Each word's share
  across is its own (`across_share`).
- The literal ruling came first: every overlapped word in the box's line by its share across, concatenated in reading
  order. I fitted it (run "4a", kept as `sf-2000-fit-round4a.json`). Multi-word boxes then went from weighted median
  log p −5.6 as pairs to −24.1 (top-1 0.10), and τ fell to 0.58–0.63 (joint).
- The crops showed why: most multi-word boxes are stacked labels. A block number sits with a code printed over or
  under it, inside one tall box, or a small number is printed inside a big code (box 2913: "1265" inside "RH-3").
  The reader reads one line of them ("RH-3", "1860B", "P"), never "RH-3 1265".
- So the box is read as one line, using only physical facts and measured ink:
  - the line's place across the box is drawn from the box's ink inside the words;
  - the line holds the words whose own ink spans that place (own ink = ink in no other word's rectangle, because
    Terra's rectangles are loose and overlap);
  - one line can't hold two words at one place along it, so words that overlap along the box are alternatives. The
    largest sets of words apart along the box (`_apart`, Bron-Kerbosch) are weighted by their own ink in the box;
  - each word is held by its share across, each character by its share along;
  - the line's words are read in order with a space between them (its share is that of the gap between them);
  - the target is the expectation over contiguous runs.

  "MISSION ST." boxes still get "MISSION ST.".
- A run no read can be (too long for the box's CTC frames) is counted as `dropped`, not listed. Its mass is followed
  exactly as the run grows.

**Box weight.** P(the box holds a non-empty run of its words' text) × (the box's ink share inside the union of their
rectangles).
- For one whole word this is the old pair weight with the across share in place of the 2-D share. Along coverage now
  lives in the target, not counted twice.
- Total weight is 5,930 over 7,861 boxes (pairs: 5,259).
- 793 boxes hold two or more words (weight 478). No box is left without a readable target (run 4a had 73).

**Leave-one-out bandwidths** now leave out every row of the same box (`loo_density`/`loo_bandwidth(..., groups)`,
through `_em` and `measure_spreads`).

### Multi-word measurement (before the change; round 3's held-out pairs, joint)

| pairs whose box holds | pairs | weight | weighted median log p | median | weighted mean |
|---|---|---|---|---|---|
| no other pair | 5,463 | 3,940.5 | −0.03 | −0.40 | −1.64 |
| another pair (weight > 0) | 1,168 | 254.4 | −5.63 | −13.3 | −8.62 |
| other pairs' weight ≥ 0.1 (display) | 982 | 179.4 | −8.05 | −14.4 | −10.3 |
| ≥ 0.5 (display) | 332 | 39.2 | −17.3 | −21.3 | −18.8 |

- Of 7,861 boxes, 793 held two or more pairs (up to 25).
- In 46 boxes the pair weights summed above 1 (max 1.52), because Terra's rectangles overlap.

After the change (final fit, held out):

| | boxes | weight | weighted median log p | mean | top-1 |
|---|---|---|---|---|---|
| one word | 5,463 | 4,425.5 | −0.06 | −2.06 | 0.688 |
| two or more words | 337 | 221.7 | −4.72 | −6.70 | 0.276 |
| two or more, literal concatenation (run 4a) | 320 | 237.0 | −24.1 | −27.4 | 0.098 |

### Fitted per fold (kept in the pack)

Every quarter kept joint. "Others' NLL" is per weight on the other three quarters' read-time vectors, after each
mode's own T.

| fold | mode | τ | w code / block / street | T | others' NLL seq / joint | sequential τ, w (its T) |
|---|---|---|---|---|---|---|
| nw | joint | 0.704 | 14.31 / 2.27 / 1.13 | 2.340 | 0.5959 / 0.5860 | 0.793, 13.89 / 2.34 / 1.06 (2.528) |
| ne | joint | 0.721 | 8.69 / 1.93 / 1.10 | 2.216 | 0.5663 / 0.5587 | 0.810, 8.29 / 1.97 / 1.04 (2.382) |
| sw | joint | 0.682 | 11.26 / 2.79 / 1.09 | 2.187 | 0.6421 / 0.6292 | 0.762, 10.86 / 2.91 / 1.03 (2.377) |
| se | joint | 0.678 | 11.05 / 2.70 / 1.08 | 2.319 | 0.6651 / 0.6510 | 0.761, 10.67 / 2.79 / 1.01 (2.536) |

- Every τ search converged.
- BFGS reported precision loss at its optimum in 5 of 8 w fits, as in earlier rounds.
- No held-out box was impossible.

### Held-out results (each fold at its kept mode and T; agreement with Terra, not truth)

- Top-1 is the likeliest named string. "ink" is the ink at τ over the same strings.
- NLL is of the target in string space. NLL slot is on the read-time vector at T.

| fold | weight | NLL | NLL slot | top-1 pack / ink / best path | codes | blocks | streets (pack vs ink) |
|---|---|---|---|---|---|---|---|
| nw | 925 | 2.48 | 0.683 | 0.616 / 0.484 / 0.475 | 0.37 vs 0.23 | 0.75 vs 0.60 | 0.13 vs 0.14 |
| ne | 1,026 | 2.80 | 0.764 | 0.654 / 0.556 / 0.545 | 0.31 vs 0.25 | 0.83 vs 0.70 | 0.38 vs 0.36 |
| sw | 1,238 | 2.00 | 0.540 | 0.699 / 0.618 / 0.615 | 0.59 vs 0.37 | 0.83 vs 0.74 | 0.33 vs 0.31 |
| se | 1,457 | 2.04 | 0.508 | 0.687 / 0.639 / 0.638 | 0.40 vs 0.19 | 0.81 vs 0.77 | 0.37 vs 0.34 |

Pooled:
- weight 4,647;
- NLL 2.28;
- slot NLL 0.785 at T = 1, 0.608 at T;
- top-1 0.669 (pack) / 0.584 (ink) / 0.579 (best path).

### Reliability on the read-time vector

Pooled, each fold at its own T. The bins are display only. The p is the likeliest slot's (OTHER included); "agree" is
the chance the target is in that slot.

| bin | weight | mean p | agree | at T = 1: weight, mean p, agree |
|---|---|---|---|---|
| 0.9–1.0 | 2,507.7 | 0.980 | 0.924 | 3,403.6, 0.990, 0.870 |
| 0.8–0.9 | 297.8 | 0.851 | 0.785 | 335.7, 0.852, 0.605 |
| 0.7–0.8 | 327.9 | 0.748 | 0.745 | 266.8, 0.748, 0.555 |
| 0.6–0.7 | 418.2 | 0.649 | 0.647 | 273.1, 0.650, 0.475 |
| 0.5–0.6 | 531.1 | 0.547 | 0.539 | 253.2, 0.548, 0.417 |
| 0.4–0.5 | 345.1 | 0.457 | 0.439 | 85.2, 0.457, 0.348 |
| 0.3–0.4 | 155.9 | 0.359 | 0.392 | 24.8, 0.360, 0.327 |
| 0.2–0.3 | 42.1 | 0.251 | 0.334 | 4.0, 0.260, 0.447 |
| 0.1–0.2 | 20.6 | 0.164 | 0.447 | 0.8, 0.136, 0.0 |

- The middle bins are now calibrated: round 3's overconfidence there came from fitting T on another vector than the
  one it was applied to.
- The top two bins are still somewhat overconfident (0.98 claimed against 0.92 agreement; 0.85 against 0.78).
- The bottom bins are underconfident.
- Per item 5, nothing further was added.

### The brief's example

Every listed string's ink is 1e-6 and an off-list string's is 0.95, on each held-out box's real prior and kind_p. The
box's kind is its heaviest word's. The margin is in nats, before T (T divides every margin and changes no winner).

| fold | codes: wins / boxes (min margin) | blocks | streets |
|---|---|---|---|
| nw | 202 / 202 (6.06) | 742 / 742 (3.86) | 185 / 185 (7.43) |
| ne | 233 / 233 (7.76) | 755 / 755 (5.36) | 296 / 296 (5.91) |
| sw | 105 / 105 (4.25) | 925 / 925 (0.70) | 467 / 467 (7.03) |
| se | 158 / 158 (4.69) | 1,143 / 1,143 (1.56) | 518 / 518 (6.93) |

- The off-list string wins in every box of every kind and fold.
- Block margins are the thinnest: 0.70 nats in sw, and 0.095 under sequential's w there. These are small blocks, with
  large odds raised to w_block ≈ 2.8.

### Off-list strings with stronger ink

Terra's likeliest target is off the list and its ink at τ beats every listed string.

| | boxes kept / all | weight kept / all | codes | blocks | streets | other |
|---|---|---|---|---|---|---|
| whole words (target certain) | 92 / 156 | 75.1 / 121.7 (62%) | 1 / 11 | 45 / 83 | 41 / 57 | 5 / 5 |
| parts | 447 / 756 | 312.3 / 532.1 (59%) | 7 / 35 | 164 / 319 | 271 / 395 | 5 / 7 |

Every whole-word case is a one-word box.

### Tests

- `tests/test_wordread_context.py`: 38 passed (the built-pack test now runs). New tests cover:
  - along-only character shares;
  - two words in reading order, and any axis sense;
  - a grazing word held by its share across;
  - stacked labels read one line at a time;
  - loose rectangles that don't join lines;
  - a small label inside a big one read apart from it;
  - a part no read can be, dropped and counted;
  - leave-one-box-out density and bandwidth;
  - the temperature's refusals (answered blocks, blocks of a row's other words, the held-out quarter);
  - slot targets;
  - the read recipe equal to the chain, refusing "all" and a quarter with no fit;
  - fits round-tripping through the file, and another layout refused.
- `tests/test_build.py`: 43 passed.

### Concerns

1. **The line model goes past the literal ruling.** Concatenating every overlapped word was measured first and was
   worse on every count (above). The line model rests on two physical facts, one line read across and no two words
   at one place along it, and on measured ink. Two choices are mine:
   - alternatives within a line are weighted by their own ink;
   - a word's span is its own ink's.
2. **The box weight is not the sum of its pairs' weights.** That sum exceeds 1 in 46 boxes and counts along coverage
   twice. The weight is P(non-empty run) × the box's ink share inside the words.
3. **Some boxes' axis runs across the text.** 4.5% of pair weight (236 of 5,259; 89 of the multi-word boxes' 484) sits
   on boxes whose axis is 45° or more off their word's reading, e.g. box 1016's "P 625", read along a vertical box.
   No read can produce their target. This is a detector failure, left in with its weight.
4. **Code boxes rarely keep stronger off-list ink:** 1 of 11 whole words, 7 of 35 parts. w_code is 8.7 to 14.3. The
   brief's example still passes with at least 4.2 nats.
5. **Thin block margins in sw** (0.70 nats joint) and the top bins' remaining overconfidence, both above.

Outputs (not committed), in `data/solver/experiments/wordread/packs/`:
- `sf-2000.parquet` (layout 2, with the fits);
- `sf-2000-rows.parquet`;
- `sf-2000-fit.json`;
- `sf-2000-fit-jobs/`.

Round 3's results are kept as `sf-2000-fit-round3.json` and `-jobs-round3/`, run 4a's as `sf-2000-fit-round4a.json`
and `-jobs-round4a/`.


## Fix round 5

Commit `be2acd9`. `pgrep -fl 'sfzmaps (build|stage)'` was empty before it.

The ruling: the line model and the box weight are replaced by the exact share model. Each pair of one of our boxes
and a Terra word is its own observation.

### What changed

**The pair weight** (`context.pair_weights`): w_bi = s_i · Π_{j≠i} (1 − s_j), the chance box b holds word i and no
other word.
- s_i is word i's share of its ink inside b's trimmed rectangle (`boxmodel.ink_shares`, as since round 2). The product
  runs over every Terra word with ink in b.
- A box's pairs sum to P(it holds exactly one word), `boxmodel.poisson_binomial(s)[1]`. So the total is at most 1 by
  construction; on the real rows the sum matches to 1e-16 in every box.
- A box holding two words whole (s = 1 each, e.g. "MISSION" and "ST.") observes neither alone and weighs 0.
- Round 2's second factor, the box's share of its own ink inside the word, is gone, as the ruling's formula has it.

**The pair's target** (`context.part_targets`): word i's inside part, as round 3 built it. Each character is inside
by its share along the reading (`char_shares`), independently; the box holds one contiguous run; spaces at its ends
print nothing; the result is conditioned on a non-empty run. A run too long for the box's CTC frames is `dropped`
(counted, scored at OTHER's slot), as in round 4. `part_targets` replaces `substring_targets` and uses the same run
recursion (`_runs`).

**Removed:** `_apart`, the cliques, line membership, the own-ink binary, the line fallbacks, the reading-order sort,
`across_share`, `box_targets` and the product-form box weight.

**No argmax per box.** Each pair carries its own word's kind, block and quarter (`build_pack.py rows`). Refusal stays
per box: a pair is refused when any word with ink in its box is in or touches an answered block or the held-out
quarter (`blocks`, the touches), because the box's reads hold that word's ink too.

**The spreads** are measured on the same pair rows with the same weights. Before, they used round 2's pair weights.

**Pairs of weight exactly 0 are left out** of every fit and score: they add nothing. This is exact, not a cutoff.

**Small items:**
- `fit_temperature(..., *, quarter)`: keyword-only and required. Leaving it out raises TypeError; None or an index
  outside paths.QUARTERS raises ValueError.
- `read_posterior`: a box missing any of `BOX_KEYS` (x, y, axis, kind_p) raises a ValueError naming the key, with or
  without a pack. A box may carry its map's `year`; it must equal `pack.year`. The signature is unchanged, and Task
  11's boxes carry no year, so their calls are unaffected.
- `Pack.save` writes a temporary file and swaps it in, so a reader such as Task 11's run never loads half a pack.
- The built-pack test still skips only on `PackLayoutError` or when no pack is built. The layout is unchanged (2).

### The pairs (`build_pack.py rows`)

- 9,294 pairs over the 7,861 read boxes (round 2's pair set plus 3 pairs whose box held none of the word's ink in its
  rectangle).
- Total weight 5,363.3. This is Σ over boxes of P(exactly one word). The rest of the boxes' mass: P(no word) 1,935.7,
  P(two or more) 562.0. For comparison, round 4's box weight was 5,930 and round 2's pair weight 5,258.7.
- 1,529 pairs (in 598 boxes) weigh exactly 0: another word lies wholly in their box (their own share has median 1.0).
  They are left out.
- By the words with ink in the box (pairs, weight): 1 word 7,067, 5,177.6; 2 words 1,108, 172.5; 3 words 378, 12.9;
  4 or more about 0.3.
- By kind (weight): block 3,636, street 906, code 772, other 49.
- 2 pairs have no readable target; 0.26 weight is dropped (runs too long for their frames).
- Spreads on these pairs:
  - m = 0.077 to 0.106, m_b = 0.063 to 0.071, e_code = 0.029 to 0.036;
  - street bandwidths 0.67 to 2.15 ft and 0.63 to 1.06°;
  - EM converged in 10 to 37 iterations.

### Fitted per fold (kept in the pack), against round 4

Every quarter kept joint again. "Others' NLL" is per weight on the other three quarters' read-time vectors, after each
mode's own T.

| fold | mode | τ | w code / block / street | T | others' NLL seq / joint | sequential τ, w | round 4: τ, w, T |
|---|---|---|---|---|---|---|---|
| nw | joint | 0.642 | 12.45 / 3.21 / 1.34 | 2.942 | 0.6586 / 0.6485 | 0.746, 11.74 / 3.37 / 1.23 | 0.704, 14.31 / 2.27 / 1.13, 2.340 |
| ne | joint | 0.676 | 7.84 / 2.24 / 1.30 | 2.731 | 0.5536 / 0.5519 | 0.791, 7.24 / 2.31 / 1.21 | 0.721, 8.69 / 1.93 / 1.10, 2.216 |
| sw | joint | 0.621 | 10.36 / 3.56 / 1.33 | 2.792 | 0.6996 / 0.6870 | 0.714, 9.65 / 3.73 / 1.22 | 0.682, 11.26 / 2.79 / 1.09, 2.187 |
| se | joint | 0.621 | 10.10 / 3.48 / 1.31 | 2.900 | 0.7123 / 0.6982 | 0.721, 9.41 / 3.66 / 1.20 | 0.678, 11.05 / 2.70 / 1.08, 2.319 |

Against round 4:
- τ fell by 0.05 to 0.06.
- w_block rose by 0.3 to 0.9 (0.8 to 0.9 in nw, sw and se).
- w_street rose by about 0.2.
- w_code fell by 0.8 to 1.9.
- T rose by 0.5 to 0.6.

Every τ search converged. BFGS reported precision loss at its optimum in 6 of 8 w fits, as in earlier rounds. One
held-out pair (sw, weight 0.19) was impossible.

### One measure on the same pairs and weights: round 5's recipe against round 4's

Both recipes read the same 5,754 held-out pairs, weight 4,298.96. Round 4's recipe is `sf-2000-round4.parquet`: its
own spreads, τ, w and T per quarter. NLL is of the target in string space; "slot" is on the read-time vector at the
recipe's T. Top-1 is the expected agreement of the likeliest named string with the part of Terra's word the box holds.

| | weight | NLL | NLL slot (at T = 1) | top-1 pack | ink at τ | best path |
|---|---|---|---|---|---|---|
| **pooled, round 5** | 4,299.0 | **2.165** | **0.649** (1.007) | **0.717** | 0.611 | 0.604 |
| pooled, round 4's recipe | 4,299.0 | 2.183 | 0.689 (1.022) | 0.713 | 0.610 | 0.604 |
| codes, round 5 / round 4 | 302.3 | 3.83 / 3.81 | 1.76 / 1.73 | 0.496 / 0.496 | 0.273 / 0.267 | 0.253 |
| blocks | 3,250.5 | 1.54 / 1.59 | 0.498 / 0.560 | 0.808 / 0.803 | 0.698 / 0.698 | 0.693 |
| streets | 707.4 | 4.11 / 4.00 | 0.862 / 0.832 | 0.400 / 0.401 | 0.367 / 0.366 | 0.363 |
| one word in the box | 4,220.5 | 2.05 / 2.05 | 0.621 / 0.651 | 0.723 / 0.720 | 0.617 / 0.617 | 0.611 |
| two or more (292 pairs) | 78.4 | 8.25 / 9.29 | 2.16 / 2.71 | 0.356 / 0.317 | 0.239 / 0.229 | 0.238 |
| nw | 794.8 | 2.11 / 2.22 | 0.632 / 0.682 | 0.704 / 0.696 | 0.539 / 0.537 | 0.525 |
| ne | 1,030.3 | 2.93 / 3.14 | 0.943 / 1.097 | 0.665 / 0.654 | 0.536 / 0.536 | 0.526 |
| sw | 1,119.5 | 1.80 / 1.67 | 0.527 / 0.509 | 0.755 / 0.756 | 0.658 / 0.658 | 0.655 |
| se | 1,354.4 | 1.91 / 1.86 | 0.536 / 0.530 | 0.732 / 0.731 | 0.670 / 0.669 | 0.669 |

- **Pooled, round 5 is a little better on every measure.** The gain is in nw and ne. In sw and se round 4's recipe is
  slightly better.
- The gain on multi-word boxes is the largest (slot NLL 2.71 to 2.16, top-1 0.317 to 0.356), but they weigh 78 of
  4,299. Round 4's −4.7 against −24 compared different target sets; this comparison is on one set.
- Round 4's recipe is scored with spreads and T fitted on other weights and targets. That is the recipe as it ships,
  so this compares the two recipes, not each weighting on its own data.

### Reliability on the read-time vector

Pooled, each fold at its own T. Bins are display only; p is the likeliest slot's (OTHER included).

| bin | round 5: weight, mean p, agree | round 5 at T = 1 | round 4's recipe at its T |
|---|---|---|---|
| 0.9–1.0 | 2,506.9, 0.977, 0.927 | 3,347.6, 0.993, 0.863 | 2,489.7, 0.980, 0.929 |
| 0.8–0.9 | 266.1, 0.854, 0.771 | 242.8, 0.856, 0.575 | 274.1, 0.852, 0.783 |
| 0.7–0.8 | 165.2, 0.753, 0.714 | 182.9, 0.750, 0.440 | 277.9, 0.749, 0.711 |
| 0.6–0.7 | 231.1, 0.646, 0.603 | 218.7, 0.650, 0.389 | 338.3, 0.649, 0.616 |
| 0.5–0.6 | 369.9, 0.544, 0.566 | 214.8, 0.551, 0.374 | 425.6, 0.547, 0.490 |
| 0.4–0.5 | 419.8, 0.455, 0.452 | 78.2, 0.459, 0.321 | 302.0, 0.457, 0.394 |
| 0.3–0.4 | 224.1, 0.358, 0.367 | 11.2, 0.363, 0.283 | 135.2, 0.361, 0.308 |
| 0.2–0.3 | 84.0, 0.261, 0.275 | 2.7, 0.268, 0.004 | 36.6, 0.254, 0.248 |
| 0.1–0.2 | 31.8, 0.157, 0.347 | | 19.3, 0.153, 0.362 |

- From 0.2 to 0.7, round 5 is calibrated within about 0.04. Round 4's recipe is 0.05 to 0.06 overconfident from 0.3
  to 0.6 on these pairs.
- The top bins stay a little overconfident in both: 0.977 claimed against 0.927 agreement.

### The brief's example (listed ink 1e-6, an off-list string's 0.95; each held-out pair's real prior)

Each pair is counted with its own word's kind. Margins are in nats, before T.

| fold | codes: wins / pairs (min margin), round 5 / round 4 | blocks | streets |
|---|---|---|---|
| nw | 169/169 (2.69) / 169/169 (6.25) | **767/769 (−0.51)** / 769/769 (3.86) | 192/192 (5.70) / 192/192 (7.43) |
| ne | 149/149 (6.26) / 149/149 (7.49) | 787/787 (3.94) / 787/787 (5.36) | 297/297 (4.35) / 297/297 (5.91) |
| sw | **100/101 (−0.16)** / 101/101 (4.04) | **850/929 (−2.92)** / 929/929 (0.70) | 477/477 (1.81) / 477/477 (4.20) |
| se | 120/120 (2.77) / 120/120 (5.23) | **1,055/1,166 (−2.23)** / 1,166/1,166 (1.56) | 526/526 (2.18) / 526/526 (4.91) |

**Round 5 fails the example in 193 of 5,682 pairs.** These are block pairs (2 in nw, 79 in sw, 111 in se, weight
178.9) and one sw code pair. Round 4's recipe passed everywhere.

I measured the cause on the losing block pairs:
- They sit in small blocks: median area 38,000 to 53,000 ft², against 112,000 to 144,000 ft² for the winners.
- Their block's odds are e^4.3 to e^4.6 (winners' e^3.3 to e^3.5).
- Raised to w_block ≈ 3.5, the lift exceeds the example's 13.8-nat ink gap.

This is round 4's thin sw margin (0.70 nats, small blocks), now past zero because w_block rose from 2.8 to 3.5.

### Off-list strings with stronger ink

The likeliest target is off the list and its ink at τ beats every listed string. Pooled.

| | round 5: kept / all (weight kept / all) | round 4's recipe | round 5 by kind (codes, blocks, streets, other) |
|---|---|---|---|
| whole words | 70 / 159 (40.2 / 96.7, **42%**) | 94 / 160 (55.2 / 97.9, **56%**) | 2/12, 28/85, 35/56, 5/6 |
| parts | 343 / 750 (148.6 / 338.1, **44%**) | 449 / 759 (197.4 / 341.9, **58%**) | 7/37, 104/314, 228/393, 4/6 |

Round 4's recipe by kind:
- whole words: 2/12, 46/85, 41/57, 5/6;
- parts: 7/36, 163/315, 274/402, 5/6.

The difference is mostly block numbers (28 against 46 whole) and streets. The counts differ by a few because "off the
list" and "stronger" depend on each recipe's lists and τ.

### Tests

`tests/test_wordread_context.py`: 34 passed. The built-pack test ran against the round-5 pack.
- New or rewritten:
  - the pair weight: one word gives its share; two whole words give 0 each; a grazing word leaves the whole one at
    1 − its share; the sum equals `boxmodel.poisson_binomial(s)[1]` for random shares; shares outside [0, 1] raise;
  - `part_targets`: every run by Π inside × Π outside, conditioned non-empty; whole word unchanged; spaces at ends;
    too few shares raise; dropped runs counted against the frames, a doubled letter included;
  - `fit_temperature` requires `quarter` (TypeError when left out, ValueError for None and for 4 or −1);
  - `read_posterior` names a missing box key, with or without a pack, and refuses a box of another year.
- Removed: the five line-model tests.

`tests/test_build.py`: 43 passed.

Full `uv run pytest -q -n 2 --ignore=tests/test_wordread_words.py`: 1,823 passed, 7 skipped, 3 xfailed, 0 failed (7.5
min). `test_wordread_words.py` is another agent's slow work in progress and was left out, as allowed.

### Concerns

1. **The exact share model fits a stronger block weight, and it now overrides strong off-list ink in small blocks.**
   - w_block rose from 2.27 to 3.21 (nw), 2.79 to 3.56 (sw) and 2.70 to 3.48 (se), and τ fell to about 0.62 to 0.64.
   - The brief's example now fails in 193 of 5,682 held-out pairs: 2 nw, 79 sw and 111 se block pairs, and 1 sw code
     pair, down to −2.9 nats. Round 4 passed in all of them.
   - Kept off-list readings with stronger ink fall from 56% to 42% (whole words, by weight) and from 58% to 44%
     (parts).
   - The losing pairs are, measured, in small blocks. Both modes fit w_block 3.5 to 3.7 in sw and se, so the mode
     choice is not the cause.
   - The posterior never reaches zero (the pack only lifts), but in a small block it acts close to a filter. This needs
     the controller's ruling. Only the owner's answers (Task 11) can say whether these overrides are right; round 2's
     crops suggested many block overrides correct Terra's misreads.
2. **The gain over round 4 is small and split by fold.**
   - On one measure, pairs and weights, pooled: NLL 2.183 to 2.165, slot NLL 0.689 to 0.649, top-1 0.713 to 0.717.
   - Round 5 is better in nw and ne, slightly worse in sw and se (NLL 1.67 against 1.80, 1.86 against 1.91).
   - The clearest gains are the multi-word boxes (slot NLL 2.71 to 2.16; weight 78) and the calibration from 0.2 to 0.7.
3. **The ruling's weight drops round 2's second factor** (the box's own ink inside the word). A box holding one whole
   word plus ink no Terra word claims (lines, unlabelled print) now weighs that word's full share. Two whole words in
   one box weigh 0 each (1,529 pairs left out at weight exactly 0).
4. **Refusal stays per box.** A pair is refused when any word with ink in its box is answered or touches the held-out
   quarter, though its kind, block and quarter are its own word's. I kept this because the box's reads hold the other
   word's ink.
5. **Round 4's recipe was scored with its own spreads and T,** fitted on other weights and targets. The comparison is
   between the two shipped recipes, not between each weighting on its own data.
6. BFGS reported precision loss at its optimum in 6 of 8 w fits, as in every round.

### Outputs

Not committed, in `data/solver/experiments/wordread/packs/`:
- `sf-2000.parquet` (layout 2, round 5's spreads and fits);
- `sf-2000-rows.parquet` (one row per pair);
- `sf-2000-fit.json` (with `ref`: round 4's recipe on the same pairs);
- `sf-2000-fit-jobs/`.

Round 4's are kept as:
- `sf-2000-round4.parquet` and `sf-2000-round4.json`;
- `sf-2000-rows-round4.parquet`;
- `sf-2000-fit-round4.json`;
- `sf-2000-fit-jobs-round4/`.

`Pack.save` swaps the file in whole, so Task 11's run, which loads `sf-2000.parquet`, gets round 4's pack or round
5's, never half of one. A run started before 17:32 holds round 4's fits.
