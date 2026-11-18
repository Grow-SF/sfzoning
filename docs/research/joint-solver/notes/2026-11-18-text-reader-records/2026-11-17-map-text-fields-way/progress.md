# SDD ledger — plan: docs/superpowers/plans/2026-11-17-map-text-fields-way.md

Spec: docs/superpowers/specs/2026-11-17-map-text-fields-way-design.md (2c51657; owner approved "Ok"). Plan 928aa5a.
Owner chose subagent-driven (2026-11-17 ~08:40). Branch `solver`, worktree saved-reads. BASE 928aa5a.
Sibling plans: 2026-11-16-text-reader-labels-and-publish (T4/T5 on hold) — not this ledger's business.

## Pre-flight scan

### Pairs sharing a file or an interface
| Tasks | Produces → consumes | Found |
|---|---|---|
| 1 × 2 | T1's `box_numbers.missed_reasons` (described as "the reason codes from Task 2") ← T2's `miss_reason` | Order conflict: T1 runs before T2. T1's own codes are "split"/"merged" (two Terra words in one box); T2's are below_level / merged_into_line / split / wrong_size / outside_map. Ruling R1. |
| 1 × 4, 8, 11, 14 | `headline`, `box_numbers`, `read_numbers`, `compare`, `paired_labels` | Consistent. |
| 2 × 4, 8, 11, 13, 14 | the review page, rebuilt after each kept step | Consistent; the controller publishes. |
| 3 × 4 | READER options own/sloppy/mask; `own_box_crops`, `sloppy_crop`, `mask_outside` | Consistent. Editing textread/train.py makes solver-text-train stale on paper (expected; not rebuilt). |
| 4 × 8 | Task 8 retrains the reader through Task 4's `run` | Consistent. |
| 5 × 6 × 7 × 8 | heads=words targets/outputs → `core_words` → doubt → runs | Sequential edits of data.py/train.py; consistent. Base per-ink rule for Task 8: Ruling R2. |
| 9 × 10 | `choose(prior_fn=None)` first, prior from Task 10 later | Consistent (ink-only default). |
| 10 × 11 | `Pack`, `prior`, `fit_prior`, `posterior` | Consistent. |
| 12, 13 × 5 | `Recipe.input`, `Recipe.scenes`; `crop_targets(words=True)` keys | Consistent. |

### Each task against itself
| Task | Found |
|---|---|
| 1 | Consistent; the baseline 0.411 matches text-read's "over every label" figure. |
| 2 | Test helpers `red_px_after_overlay` / `red_px_of` are the module's own; fine. |
| 3 | Pins a hash before changing `fit`. Consistent. |
| 4 | 8 reader trainings on NE (about 20 min each) + confirmations: long but within rules. |
| 5 | Consistent. |
| 6 | Consistent. |
| 7 | Consistent. |
| 8 | "today's recipe" for the base arm is ambiguous (r328's 328-px discs are a known error). Ruling R2. |
| 9-14 | Consistent. Task 10 needs DataSF street names; if absent, the implementer reports BLOCKED. |

### Rulings (pre-flight)
- R1. Ruling: Task 1's `box_numbers` computes its own geometric reasons ("split", "merged" = two Terra words in one
  box) and takes an optional `reason_fn` so Tasks 2/4+ can pass `review.miss_reason` (codes below_level,
  merged_into_line, split, wrong_size, outside_map) — T1 runs first and can't import T2 — cost if wrong: a rename.
- R2. Ruling: the per-ink head in every Task 8 arm (base included) is trained with `labels="lines-unknown"` (the
  control: 84-px zones only around unmatched points, in-box line ink unknown), and the base arm forms words from
  connected ink as today — r328's 328-px discs are a proven error, and the control is the best per-ink run on the gate;
  adoption of either stays the owner's — cost if wrong: base numbers differ from the shipped r328 (reported side by
  side with the Task 1 baseline).
- R3. Ruling: implementer models — opus for Tasks 2, 3, 5, 6, 8, 10, 13, 14; sonnet for 1, 4, 7, 9, 11, 12 —
  cost if wrong: turns.

## Tasks
Task 1 BASE 928aa5a. Implementer (sonnet) a988c3823e656a4b1.
Task 1: DONE_WITH_CONCERNS 2ad9644. Headline baseline 0.4092 (286/699 code labels; block CI [0.372, 0.446]); the
nearest-centre rule gives 0.4106 (stored 0.411); 1-label gap from the highest-posterior tie rule. Concerns: n = 699
(the 20 no-code labels excluded, as text-read does); many misses get reason "other"; matched_to_missed interpreted
as unmatched boxes meeting a missed word at IoU >= 0.1 (likely wrong: the owner meant matched : missed). Reviewer
(sonnet) a75a2edc5ec4a4648.
Task 1 review: spec ✅ (n = 699 matches read.py's "every label"; tie rule as specified), Needs fixes: 1 Important —
matched_to_missed must be plain matched/missed (near-miss count renamed ours_near_missed). -> fix round 1 (+ cheap:
paired_labels sign test, block-column check).
Task 1: minor (deferred): holding_boxes re-implements read.holding_box's point-in-polygon (needed for multi-hit; add a comment/shared helper).
Task 1: minor (deferred): no test covers ci.
Task 1: note: reason "other" is 2,153 of 5,635 misses at IoU 0.5 — Task 2's reason_fn refines.
Task 1: fix round 1/5 (2 addressed, 0 open; commits 2ad9644..5b0b9d6). matched_to_missed plain: all words 0.781
(IoU 0.5) / 1.307 (0.3); codes 0.433 / 0.93. Headline 0.4092 (286/699).
Task 1: complete (commits 2ad9644..5b0b9d6, review clean)
Task 2 BASE 5b0b9d6. Implementer (opus) a1b54e4e166de999a. Calls in the dispatch: unmatched-none = thin grey
outlines; non-code text = open best path from the fold reader (CPU); Terra-box reading for misses by the fold reader;
red = r328 fold chance; page published by the controller as an Artifact (contract rules passed on).
Owner (~09:05): "Could we be doing more?" -> Ruling: run implementers in parallel on disjoint files (same tree): Task
2 (wordread/review + scripts/wordread/review_page), Task 3 (textread/* + wordread/readercrops), Task 10
(wordread/context + scripts/wordread/build_pack); each runs its own tests while iterating, full suite before commit
(failures only in another agent's WIP test file are reported, not blocking), commits serialized (pgrep, index.lock
retry) — the owner asked for more throughput; the SDD default of one implementer is a conflict guard, and these
file sets are disjoint — cost if wrong: a transient test failure or index.lock retry.
Task 3 BASE (parallel) 5b0b9d6+. Implementer (opus) a0717ede50147439c.
Task 10 BASE (parallel) 5b0b9d6+. Implementer (opus) a6229127470fa9347.
Task 2: DONE_WITH_CONCERNS d627640. Page review/baseline/review.html (2.11 MB) PUBLISHED to
https://claude.ai/artifact/Ckx4CuXsFwgne4TsNkzJEN (update later runs with url= that link). Sheet p07 (16,117 ft of
MISSION ST). Terra 1,106; ours 995; matched 447 (IoU .5), missed 659; ratio 0.68; green 384, grey 163; read: 313 same,
135 differ; owner answers on p07: 46/122 right (Terra 107). Miss reasons: wrong_size 255, merged_into_line 160,
below_level 156, outside_map 64, split 24. Concerns: merged_into_line can't tell word-merge from line-merge; "ST"
vs "ST." purple; red covers 68% of ink. Reviewer (sonnet) a8cf8b6ec5aad77c1.
Task 2 review: spec ✅ (numbers are Task 1's; colours after norm_text; shading under red tested; contract OK), Needs
fixes: (1) blunt merge reason; (2) the read panel isn't read_numbers. Ruling: add merged_with_word (checked first:
another Terra word in the component) before merged_into_line; wrong_size = remainder — the owner needs to tell
word-merges from line-merges at a glance — cost if wrong: one more code. Ruling: panel keeps all-kinds line (labelled)
+ a "zoning-code forms only" line from read_numbers — one measure when comparing — cost: none. Cheap extras taken:
trailing "." fold for colour, 0.5 threshold test, "binarised ink" note. -> fix round 1 (resume a1b54e4e166de999a).
Task 2 fix 1: ce14d96 + 0473912 (Ruling: reason order outside_map → below_level → merged_with_word →
merged_into_line → split → wrong_size — off-map and mostly-below-level are the primary reasons a box never formed —
cost: none). Reasons now: merged_with_word 293, below_level 156, wrong_size 116, outside_map 64, split 18,
merged_into_line 12. Read panel: all kinds 330 same / 118 differ; codes 83 / 16. Page republished (version 2).
Owner (~09:40): "there are some huge boxes ... putting two words into the same box" -> answered: merged_with_word is
the top miss reason; Tasks 5-8 (cores, votes, tags) target it. Re-review (sonnet) ab671ec585751d5bb.
Task 2 re-review: all addressed. Controller checked: grid imported (review.py:59); box_numbers counts the new reason
key (numbers.json); delta absent at baseline (no previous run) — CHECK at the first kept run that "vs last run"
carries a codes entry.
Task 2: fix round 1/5 (3 addressed, 0 open; commits ce14d96..0473912).
Task 2: complete (commits d627640..0473912, review clean)
Task 5 (parallel, disjoint: textnet/model,data,train + wordread/shrink) BASE 0473912. Implementer (opus)
aec3bc757cff1e5ca.
Ruling: Task 5 fills per-word core rectangles into SheetData (`cores`, default empty) via `wordread/shrink.prepare`
called before training, and `crop_targets(words=True)` paints them with the existing `_paint`; geometry targets come
from `sd.words` in data.py — data.py may not import wordread (closure rule) — cost if wrong: a field rename.
Task 3: DONE_WITH_CONCERNS 59b9cf7. Defaults pinned unchanged; 1613 passed. offsets (NE out): 2,978 pairs; our
trimmed boxes start +0.096 / end −0.104 word heights inside Terra's; angle +1.8° (sd 8.1); height ratio 0.93;
line_share 0.20; neighbour_share 0.03. Own-box crops nw 2,850 / ne 2,959 / sw 2,550 / se 2,357 (~80% block numbers;
codes 160-196 per quarter). Concerns: composite line pieces thicker than real crossings (6 vs 4 px median; 18% >
p99), some hide words; pasted neighbour can land on letters; crops.py edit makes solver-text-crops/-read stale on
paper too. Reviewer (sonnet) a81ac9cb7998e12d7.
Task 3 review: spec ✅, Needs fixes (4 Important): line pieces hide words; neighbour on letters; quarter=None unsafe;
no kind weighting. Rulings: pieces filtered to the measured crossing thickness (<= real p99 ~11 px) + reject draws
covering > measured p99 of word ink, redraw to keep line_share; neighbour drawn jointly with the pair's offsets, overlap
with Terra's rect capped at the measured ~0, placed along/beside the axis; quarter required (raise); piece_patches
checks the bank's held areas; kind_weights="terra" option (measured Terra kind shares of training quarters); under
mask=True Terra and sloppy crops masked too (components overlapping Terra's rect) — all measured, no guessed knobs —
cost if wrong: a few options to tune in Task 4. -> fix round 1 (resume a0717ede50147439c).
Task 5: DONE_WITH_CONCERNS 87f3d82 (shrink.py; data/model/train words=True; 15 tests; guards unchanged). CORE=7,
GEO=8-11, TAG=12-15, WORD_OUTPUTS=16; distances = 8 px · exp(clamp ±20). r: brief's rule unanswerable (12-14% of
close pairs overlap at every r: Terra errors). Concerns: mps determinism unchecked (Task 8: two short runs); tag
one-hot up to ~240 MB; predict.outputs refuses 16 ch; tag pull on core px only.
Ruling: r = new overlaps beyond the grid's first value (0.3), pooled over folds, ≤ 1% → 0.40 (pooled 241/25,256 =
0.95%; per fold 0.89-1.01%) — the floor pairs overlap at any r (Terra errors, Task 7's doubt candidates) and the rule's
intent is that shrinking must not merge words Terra keeps apart; one r for a general model — cost if wrong: cores a
few px larger/smaller; Task 8 can screen 0.35. -> resume aec3bc757cff1e5ca (shrink.py/json/test only).
Ruling: word-head prediction (16-ch net → core/geo/tag arrays over a page) is Task 8's; Task 6's core_words works on
arrays and choose_level is tested synthetically — Task 6 can't have a trained net — cost: none.
Task 5 reviewer (opus — subtle losses/geometry) a00d527219541834d. Task 10 committed cd2bf9b (report pending).
Ruling: Tasks 6 and 9 start now in parallel (new files only); Task 7 (edits data.py) waits for Task 5's review —
avoids data.py conflicts with a Task 5 fix round — cost: Task 7 starts later.
Task 5 r ruling applied: da3a149 (r = 0.40 in shrink.json; floor 3,191 = 12.6%). Goes in the Task 5 fix-round re-review.
Task 6 BASE cd2bf9b. Implementer (opus) ad4c7998d4dc9a2c3. Task 9 BASE cd2bf9b. Implementer (sonnet)
afe082cdb3c842809 (call: a split candidate scores as the sum of its words' log posteriors).
Task 9: DONE edb6f33 (choose.py; 18 tests; 1654 passed). Char widths px/char pooled: zoning 22.5, block 8.25, street
9.33, other 19.67 (per-held-out variant in char_width.json). Call: candidate score = sum of its words' log posteriors
(splits penalised by construction). Reviewer (sonnet) a84c26e349311596a — main check: candidate fairness.
Task 9 review: spec ✅, Needs fixes: (1) a confident "other" on a merged crop beats a correct split; (2) any 1 px gap
offers a split, and sums of posteriors across crops aren't comparable; (3) silent pooled widths default (fold leak).
Ruling: the choice is a fitted conditional logit over a box's candidates, P(c) ∝ exp(β·score + α_kind + γ_kind·gap/
char_width), fitted (fit_choice) on the training quarters' Terra-matched boxes (label = candidate best matching
Terra's word(s)), run in Task 11; params=None keeps today's argmax — probabilities, not hand-set gap gates; "other"
is also block numbers / street names, so it can't simply be dropped — cost if wrong: a few fitted params. Widths
required; unknown kind → no trims; join excludes self/overlaps; 90° test. -> fix round 1 (resume afe082cdb3c842809).
Task 3 fix 1: b19c933 (thickness filter drops 632/3,782 pieces at q99 10.8 px; 4.2% line draws redrawn; neighbours
with pair offsets; kind_weights; masks). Deviation accepted — Ruling: neighbour overlap cap 0, not the measured q99
1.0 — that q99 comes from loose Terra rects (block numbers inside slanted codes' rects); intent is never on the word —
cost: none. Controller looked at crops_ne.png: lines still hide EDGAR/1547/7239A/7064A; mask drops "6" of 6795A
(detector called the "/6" not-text; 24% of Terra words lose >25% ink).
Ruling: line pieces match the whole measured thickness AND cover distributions (importance weights + rejection), with
real cover re-measured as line ink in components extending outside the word's rect (letter strokes excluded) —
synthetic numbers drawn from real measurements — cost: a few redraws. Ruling: the mask keeps every binarised-ink
component overlapping the crop's own rect (no text-chance level), blanking only components wholly outside — the
mask must not depend on detector text calls or blank letters — cost: weaker mask (only margin clutter removed).
-> fix round 2 (resume a0717ede50147439c).
Task 5 review: Approved (spec ✅; no Critical/Important). Minors carried: (1) geometry at known core px of words
straddling held areas reveals box extent (not an answer key) — close in Task 7 (drop m for words touching held px);
(2) tag one-hot ~241 MB — record peak memory in Task 8; (3) mps determinism of sort + bmm — Task 8's first step; (4)
test_wordread_heads.py:203 restates HEAD_OUTPUTS (owner: no trivial assertions) — remove in Task 7; (5) PAN++
departures documented. da3a149 (r rule) checked by the controller (diff read: floor/new/pool, test) — OK as code.
OWNER (~10:35, capitals): "I KEEP TELLING YOU TO DO THIS AND NOT DO ARBITRARY THRESHOLDS". Audit of decision cutoffs:
- T3: q99 thickness filter + q99 cover cap (MY round-1 ruling) → whole-distribution matching (round 2, in flight);
  neighbour cap 0 (accepted deviation) → neighbour placement drawn from real pairs incl. real ink overlap (sent).
- T6 (running): tags "> 3 apart" → two-word vs one-word likelihood with measured tag_sigma and fitted split_prior;
  boxes carry p_word; level required, fitted (sent).
- T9 (fix in flight): join "within one word height" window removed; distance is a logit feature (sent).
- T5 r: "≤ 1% new overlaps" is a cutoff → PROPOSED: r screened in Task 8 by the held-out outcome. Awaiting owner yes.
- T7 (HELD): q99 cut + boolean doubtful + IoU ≥ 0.5 contradiction → PROPOSED: per-word weight = P(Terra word right),
  fitted on words Terra read twice (agreement as the label) from ink-outside share, forced height, kind. Awaiting yes.
- T10: check context.py for distance windows when its report lands.
- Display-only (kept): T1 IoU 0.5/0.3 top-line; T2 reason labels; review page red ≥ 0.5.
Root cause found: the spec's anti-goal had reworded the rule as "thresholds are fitted or measured" (licensing
measured cutoffs) and the plan's Global Constraints omitted it, so no brief or review carried it. Fixed 26985e3: spec
anti-goal restated; Global Constraints lead with "No hand-set cutoffs" (reviewers treat a breach as Important; a brief
giving a cutoff = stop and report). All 14 briefs regenerated. EVERY reviewer dispatch from here adds the check: "any
decision made by a fixed number or a cut read off a distribution = Important". Bug report drafted (owner asked).
Task 10: DONE_WITH_CONCERNS cd2bf9b + 028b943. Pack: 17,176 centrelines (2,713 streets), 5,302 blocks, 57 codes (95
forms), no lot codes; m = 0.126 streets / 0.078 blocks; street words median 3.6 ft, 1.2° from their street. Fit on
4,402 Terra-matched boxes: w 4.03/4.89/5.90/5.17 (nw ne sw se); e 0.0025-0.0026 (ne, sw), AT ITS FLOOR (nw, se).
Held-out agreement with Terra: all kinds 0.55-0.73 → 0.79-0.87; streets 0.43-0.48 → 0.65-0.68. Concerns: flat code
list (closed headline barely moves; Task 11 should use the open posterior); w≈5 with e at floor ≈ a filter.
Reviewer (opus, incl. the no-cutoff check) ac18955cfdd9199f4.
Task 9 fix 1: d4b62c2 (conditional logit choice_features/label_of/fit_choice/choose(params); widths required; no
window, no min gap, AXIS_OK/MIN_CHARS/MATCH_IOU removed; 1665 passed). Re-review (sonnet) a27e976df37002e2b (also:
cross-axis offset as a join feature?).
Task 9 re-review: all addressed. Follow-ups -> fix round 2: join along and across gaps as separate fitted features
(hypot conflates next-line with same-line); fit_choice requires held_out/answered columns; ridge on β too; Cand.gap
explicit (lost on pickle); label_of rotated IoU. (resume afe082cdb3c842809)
Task 9 fix 2: ea52c04 (along/across features; held_out/answered required; ridge on β; frozen dataclass candidate;
shapely rotated IoU — shapely>=2 already a dependency). Controller checked: the 4 new tests named for each item.
Signatures changed: choose/choice_features no longer take widths (normalised inside candidates).
Task 9: fix round 2/5 (5 addressed, 0 open; commits edb6f33..ea52c04).
Task 9: complete (commits edb6f33..ea52c04, review clean)
Task 10 review (opus): ❌ partial, Needs fixes. Critical: at fitted w≈5 the pack is near a filter (brief's off-list
example loses in all folds; 14/31 real stronger-ink off-list words kept). Important: e's floor (logit −16) decides e
in nw/se; w per form penalises multi-form codes; form weights not held out by quarter; IoU 0.5 selection of fit words.
Ruling: posterior ∝ ink^τ × prior^w_kind with escape mass = the measured per-kind off-list share (codes too); τ and
w_kind fitted held out; reliability table; no bound decides a parameter — w≈5 compensates overconfident CTC ink —
cost if wrong: 4-8 more fitted params. Ruling: w on the code's mass, then split by P(form|code); form weights per
fold; IoU selection screened at 0.3/0.5/0.7; EM for m/m_b/spreads; spreads measured on OUR matched boxes; spreads
required; LOO bandwidth; refuse touch_q. -> fix round 1 (resume a6229127470fa9347).
OWNER (~11:10): "The R core size measure is still arbitrary numbers that you've picked, not a learned distribution"
(re: screening r = 0.3/0.4/0.5). PROPOSED (awaiting yes): drop r — core target = the word's own rectangle (no
shrink); touching words split by the learned tags (Task 6 likelihood split, measured tag_sigma); boxes from geo votes
only (no unclip); overlapping Terra rects handled by Task 7 weights. Task 7 weights proposal also awaiting yes.
OWNER (~11:15, thinking out loud): learn what a box holds (e.g. text-ink density, ink centroid vs box centre) — "Don't
just hardcode these ... There is a hidden function that you should be able to learn automatically." PROPOSED
(awaiting yes): a learned box model on rectified box crops (ink + text chance + word-head outputs, with margin) →
P(word count 0/1/2/3+) and P(boundary) along the box; labels from training-quarter boxes vs the Terra words they hold
+ joined real neighbour pairs; feeds Task 9's candidate scores; with tags + votes replaces r. Judged by held-out box
numbers (Mission merged_with_word 293 today).
Task 6: DONE_WITH_CONCERNS ac8c0f4 (words.py; 24 tests; 1694 passed; tag split by likelihood, p_word, split_llr).
Concerns: merge_iou cutoff in votes NMS; a rejoin step added after the split broke a noisy chain into 51 words; votes
keep one box per component; quarter/x/y/lot not filled (Task 8). Reviewer (sonnet, with cutoff inventory)
af79b8cb28b37f837. Ruling: Task 6's fix round waits for the owner's answer on the learned box model / r removal, so
it changes once — cost: Task 6 completes later.
OWNER (~11:25): "Yes, that sounds better. Add it to the proposal." → spec + plan amended eb0cf7e: no r (cores =
labelled rects; measure_r/unclip a record only); Task 6 = core-weighted vote mixture (measured spreads, effective-n,
fitted word_prior; no level / merge_iou / one-box-per-component / split+rejoin) — the level removal is the
controller's application of the same rule (a level picked from a grid is a picked number), told to the owner;
Task 7 = fitted weights from double reads (logit P(strings agree), beta regression of pair IoU); Task 8 arms
amended; Task 9 gains box-model features later; NEW Task 15 learned box model. Briefs regenerated (1-15).
Task 6 review (sonnet af79b8cb28b37f837): Needs fixes — merge_iou cutoff; one box per component; split/rejoin two
models (iid tags, llr ∝ px); level a cut; no noisy-single-word test. Folded into the amendment.
Dispatched in parallel (disjoint files): Task 5 follow-up (resume aec3bc757cff1e5ca: prepare = own rects; drop the
trivial HEAD_OUTPUTS assertion); Task 6 fix round 1 (resume ad4c7998d4dc9a2c3: rework); Task 7 BASE eb0cf7e
implementer (opus) a616746a89fdc5c00; Task 15 BASE eb0cf7e implementer (opus) aaa1478ae32e62aa0.
Task 5 follow-up: fd0e71e (prepare(sheets) = own rects; record-only r; trivial assertion removed; 59 passed). Breaks
Task 6's committed tests (17, TypeError) — Task 6's rework removes them (told). data.py docstring stale — Task 7 fixes.
Task 5 reports hand-set loss constants: OHEM_RATIO 3, PAN PULL 0.5 / PUSH 3.0, TAG_WEIGHT 0.25, GEO_ANGLE_WEIGHT 10
(train.py); GEO_UNIT 8 px, ±20 clamp (model.py). Ruling (pending until Task 7 commits train.py): loss weights →
learned homoscedastic-uncertainty weights (Kendall et al. 2018); OHEM 3:1 → BCE weighted by the measured class
frequencies + Dice; PAN margins → a softmax over the crop's word means (px tag classified to its own word vs others
and background) with a learned temperature; GEO_UNIT and clamp are parameterisation (numerical), kept — the plan's
first constraint says published constants are not exempt — cost: a loss rewrite before Task 8's runs.
Task 3 fix 2: 5ec4ced (thickness matched exactly; cover q10/50/90/99 0.05/0.25/0.52/0.70 vs measured
0.02/0.23/0.51/0.69; 3/5,403 gave up; mask keeps every cluster touching the rect: 0/1,434 Terra words lose in-rect
ink, 0.24% blanked; round-1 filter and cap removed). Neighbours sequentially conditioned → Ruling: draw from the
empirical joint (one real pair's exact placement) — no density choice — cost: none. -> fix round 3.
Task 3 fix 3: 2c2b9a9 (empirical-joint neighbour draws; 98 neighbours in 5,393 crops; zero-touch 3.2% vs 2.9%).
Residual: placement by distance from the nearer end → box share median 0.09 vs 0.32, ink-on-letters q90 0.53 vs 0.30.
Ruling: park as minor unless the re-review finds it load-bearing — neighbours are < 2% of crops and Task 4 screens
sloppy crops as a whole — cost: slightly heavier neighbour overlap in 2% of crops. Re-review of rounds 1-3 (sonnet).
Pre-dispatch cutoff check of Task 4's brief: own=0.3 / sloppy=0.3 were picked shares. Ruling (2cfac43): arms base,
(a) the read-time distribution (every training Terra word cut with measured offsets + real own crops at natural
counts, kind_weights="terra"), (b) (a) + mask — the reader only reads our boxes; shares derived from counts, not set
— cost: one fewer arm. Keep rule's paired 95% interval is an inference convention for keeping, not a model cutoff.
Task 3 re-review (rounds 1-3): all addressed. Residual neighbour placement parked (re-reviewer: not load-bearing,
1.8% of crops; anchor by fractional position later if needed — discuss first). Two small items → resumed
a0717ede50147439c: remove LINE_TRIES cap (loop until accepted); neighbour_pose returns None on empty cells. Ruling:
resume the same implementer for this 2-line round 4 rather than a fresh, stronger model — the items are new and tiny,
not a stuck loop — cost: none.
Task 3 fix 4: 38b5e4f (LINE_TRIES gone; words with no ink in their rect get no line — cover undefined, a physical
case; ≤ 15 draws on every real NE word; empty-cells guard). 1730 passed.
Task 3: fix round 4/5 (all addressed, 0 open; commits 59b9cf7..38b5e4f).
Task 3: complete (commits 59b9cf7..38b5e4f, review clean)
Task 7: DONE_WITH_CONCERNS — files in 38b5e4f (shared-index race under Task 3's message; contents identical).
Double reads = Terra's overlapping 768 px tiles: 27,219 pairs (agree 0.868, IoU median 0.877). Logistic P(agree) +
beta regression of pair IoU (with point mass at 1 and feature-dependent precision). Box weights 0.52-0.92; string
0.49-0.99. Concerns: ink_outside near-binary; tile-edge distance predicts better but unused; held-out geometry marked
by zero distances (sentinel?). Reviewer (sonnet) dispatched.
Task 7 review (sonnet abe2ad95156532d47): Approved, no cutoff decides a weight. Zero-distance held encoding is safe
(sums test) but implicit; direction lost for held-touching words. Selections noted: unpaired words dropped; pairing
at any positive overlap. Ruling (owner's rule): follow-up round 1 in doubt.py — P(paired) modelled (zero-inflated),
log(1+edge) tile-edge feature with support check, line-aware continuous ink shares, continuous forced height,
documented/mixture pairing, write_doubt mask test, fold-name and word-identity checks — cost: one more round.
Ruling: Task 5 follow-up round 2 (train/model/data): Kendall uncertainty weights replace TAG_WEIGHT/GEO_ANGLE_WEIGHT;
class-frequency-weighted BCE + Dice replaces OHEM 3:1; softmax-over-word-means tag loss with learned temperature
replaces PAN margins; GEO_UNIT and clamp kept as parameterisation; explicit geo_known; direction mask core·known;
_touch_held per sheet; a fixed-value word_loss test. Commits must stage+commit right after git status (index race).
Task 6 fix 1: f7819c9 (core-weighted vote mixture; spreads in distance+angle space; measure_spreads, word_gain,
extra_word_gain, fit_word_prior for Task 8; 25 tests; 1734 passed). Concerns: greedy search with hard px ownership;
140 junk boxes / 1.44 Mpx at p_word ≈ 0.001 (word_prior must be fitted on word-or-not too); ~2.5 min, 2-3 GB per
page. Re-review (opus) dispatched, incl. a cutoff inventory.
Task 15: DONE_WITH_CONCERNS 7d2f553 (boxmodel.py, run_boxmodel.py, tests; 19 tests; 1734 passed). Examples per
held-out quarter nw 5,063+128 / ne 4,742+106 / sw 4,633+164 / se 4,336+135, K = 6. Today's 8,794 boxes expected
count 0/1/2/3+: 2,869 / 5,363 / 381 / 181 (Mission p07: 304/577/71/44). Same-line gaps median 7.8-10.2 word heights.
Concerns: same-line joins are far words (real merges are 2-D clusters); few real merges (85-133 two-word / fold);
count 0 also covers Terra misses. Ruling (controller, before review): joined examples drawn from real neighbour
pairs ∝ a fitted P(today's boxes merge the pair | relative geometry), exact relative placement, union box crops —
the measured distribution of the merges the model must catch — cost: a fit. -> fix round 1 (resume
aaa1478ae32e62aa0); review after.
Task 6 re-review (opus a508b142e6c291af8): 6/6 original addressed; new: ½ reading prior (BREACH), cell estimator
measures per-word bias (Important), core = 1.0 → −∞ background (Important); minors: hard ownership loses fragments,
no termination guarantee, p_word not a probability, approximate direction step, prior fitted on other decisions,
background tag mean fixed, cost unmeasured. Rulings → fix round 2 (resume ad4c7998d4dc9a2c3): residual-based
spreads/cell + fitted slope; measured reversed share q; logits not probabilities; final global soft EM + absorb
move + word_logodds; accept moves only if the total score rises; prior fitted on core_words' own decisions with soft
vote-mass labels; measured background tag mean; dense-page cost measured.
Task 5 follow-up round 2: 6d1af69 (Kendall weights; per-batch inverse-frequency BCE + Dice; softmax tag loss with
learned temperature + background mean; geo_known; held-touch per sheet; word_tensors 6 values; fixed-value test;
guards 157 passed). It committed edits to test_wordread_doubt.py via update-index while Task 7 edits it → told Task 7
to merge HEAD's version before committing. Re-review (sonnet) dispatched.
Task 5 round-2 re-review (sonnet a977e33bb659eb40e): all addressed (Kendall ½ is the likelihood form, not a factor;
background tag mean measured per crop; per-batch class balance handles no-positive batches). Residual flagged:
EAST's IoU (I+1)/(U+1) smoothing — a published +1 → carry to Task 8 dispatch (replace by a px-area guard or
document as numerical). Controller verified on a CLEAN extract of 6d1af69 (scratchpad, src first on sys.path):
doubt + heads + textnet guards + composite + build: 152 + 43 passed (the 2 first failures were the extract lacking
dvc.yaml).
Task 5: complete (commits 87f3d82..6d1af69 incl. da3a149, fd0e71e; reviews clean)
Task 7 follow-up 1: 1d610d3 (P(paired) modelled; log(1+edge); letter-only ink outside + line ink share; continuous
angle (axis, axis²); pair IoU as a same-word/neighbour mixture; hash/fold checks; quarter-exclusion test; 1744
passed, 0 failed). Held-out LL/word r1 vs r0: nw 0.41/0.15, ne 0.41/0.06, sw 0.23/−0.07, se 0.09/−0.26. NE weights:
box median 0.845 (1% 0.25), string 0.905 (1% 0.16). Concerns: edge extrapolation (pairs ≤ 251 px, read-once to 369);
fold-to-fold movement up to 0.13/0.19. Scoped re-review (sonnet) dispatched.
Task 15 fix 1: 600d723 (merge logistic over 1.3-2.0M pairs/fold, soft labels from ink shares; 8 features: gap and
overlap splits (deviation from the 4 signed offsets, which mis-specified: 24-26% of mass beyond 5 heights); joined
drawn ∝ P(merge), union box, exact targets; K now 8-12 (25-33% of joined hold a third word's ink); drawn medians
match today's merges within 0.1-0.2 heights). Joined per fold nw 128 / ne 106 / sw 164 / se 135. Review (sonnet)
dispatched (incl. how the joined count is decided, MAX_WIDTH).
Task 7 follow-up re-review (sonnet a4fe995e7d983ac5f): all addressed, no cutoff breach; Task 5's test section
preserved. Controller grep: no callers of doubt.load_weights/prepare outside docstrings (Task 8 is the first caller:
load_weights(fold) → {"words","box","string"}; prepare(dict, which)).
Task 7: minor (parked): edge effect as a fitted saturating curve (1 − exp(−edge/τ) or Hill, τ by ML, chosen held
out) — ≤ 0.05 in probability beyond 251 px; the in-support turndown at 200-260 px stays unmodelled.
Task 7: minor (parked): no-ink words form a jump a linear logit can't represent.
Task 7: complete (commits 38b5e4f (Task 7 paths), 1d610d3; reviews clean)
Task 15 review (sonnet a4f96aa9821048671): Approved, no cutoff breach (joined count = Σ P(count ≥ 2), measured).
Minors → round 2 (resume aaa1478ae32e62aa0): ordinal count head (cumulative logits, ordered θ; box_logp adds
P(count > K)) — K set by slivers left upper classes untrained; held_words required; squeezed counts recorded;
merge-model form chosen by held-out NLL; trivial assertion removed. Noted for Task 8: RECIPE/architecture/BN momentum
are picked training settings; the chance channel comes from fold detectors (page-derived stacking) — state it.
Task 4 interim (s0 only, NE 285 code labels): base 32.3% (92), (a) 33.7% (96) — no verdict from one seed.
OWNER (~12:45): "Do we have an updated map ... with the detections?" → no; offered to put the detector training ahead
of Task 4's mask arm. Owner: "Yes". Ruling: Task 4 finishes base s1 + (a) s1, then pauses before (b); Task 8 starts
now (Phase 1: heads=words NE s0 → words → page, NE shown, other quarters faded "fold not yet trained"; p07 is
684/995 boxes NE, 273 SE). Phase 1 spreads from the NE detector's own training quarters (optimistic, flagged); Phase 2
cross-fits spreads/prior from other folds' held-out predictions — avoids fitting on the scoring fold — cost: the
Phase 1 page's boxes may be slightly over-merged or over-split. Task 8 BASE HEAD. Implementer (opus) dispatched.
Task 15 round 2: 36af02a (ordinal head P(≥j) = σ(f − θ_j), box_logp adds P(> K), collapse_counts for alignment;
held_words required; 0 squeezed crops in every fold; merge form chosen held out: 8 features beat 4 offsets in every
fold (NLL total 1143.0 vs 1382.2); trivial assertion gone; 1749 passed). Note for Task 8: the ordinal head learns
slowly in short fits (lr matters) — watch its training curve.
Task 15: complete (commits 7d2f553..36af02a; review approved, round-2 minors addressed)
Task 4 INTERIM (run_reader.py + test committed; 1754 passed): NE 285 labels — base 0.3228/0.3088 (mean 0.316), (a)
0.3368/0.3298 (mean 0.333); paired +0.0175 [−0.012, +0.048] → (a) NOT kept by the rule. Reading at Terra's boxes
within noise. Base s0 reproduces the stage's NE reader exactly. Deviation found: (a)'s pools = real Terra 14,630
(clean, share ~0.78) + own 2,959 + sloppy ~5,400 — not "every training Terra word cut with measured offsets"; the
synthetic share stays at today's 0.4. Finding: the gap is mostly boxes (NE: 850 of 2,705 Terra words matched at IoU
0.5). Ruling: (b) and a corrected (a') (every training Terra word cut with measured offsets, per series where own boxes
exist) run later with Task 8's reader retraining on the new detector's boxes — GPU goes to the detector first, and the
reader options matter on the boxes the final system reads — cost: Task 4 completes later. Implementer
a2cfe38e27fbf0973 paused.
Task 6 fix 2: 36ab064 (all 9 rulings; measured reversed share q; 27 tests; 1757 passed + Task 8's WIP failure).
Cost on a 5100×3300 synthetic page (1,638 words + 2,000 fragments): 12.6 min, peak 14.1 GB (global EM 12M pairs) →
one page per process. Concerns: 8.57 sd vote bound (numerical?); batch claim replaced absorb moves; test prior −60;
30 more strong boxes than painted words, cause unknown. Task 8 told. Re-review (opus) dispatched.
Task 10 fix 1: e159690. τ (CTC temperature) 0.74/0.70/0.69/0.69; w code 21.7/13.2/17.6/15.4, block 2.6-3.4, street
2.3-2.6; escape per kind measured (no floor); per-fold form weights/spreads; LOO bandwidths. Held-out top-1 vs Terra
0.80/0.84/0.89/0.88 (ink alone at τ 0.55-0.73). Brief's off-list example wins 266/266 (min 2.5 nats); real
stronger-ink off-list: 19/62 kept, 38/43 losses = block numbers inside a block Terra didn't name. Posterior
overconfident (top bin 0.996 vs 0.913). Rulings → round 2 (resume a6229127470fa9347): blocks-outside = escape
accepted (measured, 2-3/2,300); block disagreements characterised + 10 example crops for the owner; IoU selection
REMOVED → pairs weighted by ink-share products (no IoU in the fit); τ/w joint vs sequential chosen by held-out NLL,
plus a held-out temperature on the combined posterior if still overconfident. Re-review rounds 1+2 together after.
Task 6 re-review r2 (opus a2ed9b35bb1091c0d): 1-3, 5-9 addressed; 4 open — BREACH: PRECISION_SDS (8.57 sd) decides
shares via candidate sets; Important: word_logodds not leave-one-out (overstated on overlaps); measure_spreads fits
direction on direction votes only (offset leaks into σ/cell); _global's m_k = Σr is not the local-share model's
M-step (non-monotone EM). Claim candidates by axis-aligned bounds (window); joins by bounds meeting. 30 extras: likely
(a) unjoined duplicate proposals, (b) splits of one word, (c) claims across neighbours. Ruling (fix round 3, resume
ad4c7998d4dc9a2c3): a per-word SPATIAL prior (px inside the box, measured edge spread) in shares, search and global
EM — the generative model of where votes come from; makes the window numerical — cost: one more measured spread.
Plus true leave-one-out logodds; direction fitted on both terms in measure_spreads; exact M-step; joins from both
starts; NaN guard; page-synthetic script committed; 30-extras cause found and fixed; exact cost reductions.
Task 8 PHASE 1 BLOCKED (2e3964b, 84639d8 committed: predict.word_outputs, variant --heads words/--doubt/--root,
run_detector.py, wordboxes.place, page options; EAST +1 → 0/0 guard). a-ne-s0 trained 13.9 min mps; reading
direction never learned (1−cos ≈ 1.0 per-ink AND word direction; control reached 0.49); core/dice/tag learned;
distances carry no side information (corr ~0); boxes 1.34× tall, 1.58× long; spreads degenerate (distance 274 px,
angle 240°, reversed 0.88, cell 653k) → no page. Shipped fold detectors' direction also near random on p07 (1−cos
1.06). mps determinism: bit-identical; peak 3.18/10.06 GB. Gate vs control (1 seed): m1a_in_box 0.412/0.393, m1c
0.062/0.041, m2 0.978/0.965. Proposals: (1) sign-free RBOX frame + axis (cos 2θ, sin 2θ) + 0/180 to the reader's
classifier; (2) seed check. Controller: test the cause first (free) — axis error mod 180° and reversed share on
existing predictions; the earlier 41%-reversed finding fits "axis right, sign random". Task 6 round 3 WIP hangs
test_wordread_words.py.
Axis vs sign (Task 8, free check): a-ne-s0 word direction axis error median 27° (25% within 10°), and where the
axis is right it points 180° WRONG at 99.8% px / 100% words (constant sign: one sign per axis orientation under the
quarter-turn augmentation); shipped NE detector learned neither (axis at chance). Ruling: sign-free word geometry
for heads=words — axis (cos 2θ, sin 2θ) on ch 5-6 with 1 − cos 2Δθ; GEO 8-11 = (dx, dy) to the word centre in word
heights, log length, log height (continuous at every angle; no side order); 0/180 to the reader's classifier —
measured: the sign is unlearnable here and its loss starved the axis — cost: Task 5 round 3, Task 6 vote space
change (q dropped), Task 8 decoding + one re-run. Dispatched: Task 5 (aec3bc757cff1e5ca), Task 6 heads-up
(ad4c7998d4dc9a2c3), Task 8 waits then re-runs NE s0 (a16c58c91a3a8ff32).
Task 5 round 3: 512a1dc — sign-free layout (ch 5-6 axis cos2t/sin2t; 7 core; 8-9 dx,dy ÷ height; 10-11 log length,
log height; 12-15 tag); model.word_geometry; loss terms core, dice, centre, size, axis, tag (Gaussian NLL with
Kendall variances; axis 1 − cos 2Δt); per-ink terms given twice the angle; 28 heads tests; guards 157 passed.
Breaks until adapted: predict.word_outputs, words.py, run_detector/variant. Task 8 told to retrain NE s0 now; Task 6
told the exact layout. Re-review of round 3 folded into Task 8's training check (axis error, size ratios).
Task 10 fix 2: 675d8ab — no IoU (pairs weighted by ink-share products, 9,291 pairs, weight 5,259; 83 impossible
CTC-length fragments excluded, counted). Sequential τ 0.58-0.60; joint τ 0.37-0.38, w code 23.6-29.2, block 3.8-4.4,
street 3.7-4.0; joint wins pooled held-out NLL (2.47 vs 2.89); final temperature T 3.75-3.94 → NLL 1.09. Top-1 vs
Terra joint 0.73-0.84 (ink alone 0.49-0.67). Cost: off-list stronger ink kept 6/97 joint vs 17/89 sequential; brief's
example 739/794 vs 768/794. Block overrides (72): 33 one-char misreads (pack right in the 2 checked: 417B→4178,
4382→4362); crops in packs/block_disagreements/. Ruling (round 3): the fragment pairs ask a part-word box for the
whole string — physically impossible targets dominate τ/w — targets become the expected substring inside our box
(character extents along Terra's box, shares, no cut); refit both modes; choose by held-out NLL; report off-list
behaviour; a remaining conflict goes to the owner — cost: one round.
Task 8 retrain (sign-free, a-ne-s0): deterministic; peak 2.75/10.07 GB. Final losses axis 0.032 (was 1.0), size
0.100, core 0.165, tag 0.31, centre flat 1.73→1.78. Held-out NE vs Terra: axis median 1.1° px (92.5% within 10°),
per word 1.5° (84%); length ×1.18 px / ×1.44 word (q90 3.5); height ×1.06/1.08; centre per px 0.95 heights, per
word 0.11. Gate vs control (1 seed): m1a_in_box 0.457/0.393, m1c 0.067/0.041, m2 0.981/0.965, m2_clean 0.930/0.927.
Waiting on Task 6's core_words. Told Task 6: spatial prior must pull length to the core px extent (synthetic test with
×1.4 length votes) and measure per-dimension vote bias vs Terra.
OWNER (~14:35): "Can we do more at once" → Ruling: (1) Task 8 queues arm (a) SE s0, NW s0, SW s0, NE s1 on the idle
GPU (four folds for cross-fitting + the second seed), page first when Task 6 lands; (2) Task 11 (sonnet) builds and
runs item (5) box choice on TODAY's boxes as a labelled PRELIMINARY (CPU reads), pack path built and tested only; (3)
Task 12 (sonnet) code only, no predict.py until Task 8 commits; (4) Task 13 (opus) scenes.py first, Recipe.scenes
after Task 12's commit. Cutoff check: Task 13's clean_word_mask uses a fixed CLEAN_PX ring → Ruling: paste each
word's own ink pieces (touching and wholly inside its box) — the structural half, no number — cost: none.
Task 10 fix 3: 2e099e6 — fragment pairs scored against the expected substring (exact expectation over contiguous
runs). τ rose (seq 0.75-0.81, joint 0.65-0.69); w code 8.5-14.5; w street ~1.2; 0 impossible pairs. Joint wins pooled
held-out NLL 2.059 vs 2.109; T 3.90-3.95 → 1.012. Top-1 (part inside the box) 0.66-0.73 vs ink 0.52-0.66. Off-list
stronger ink kept: joint 50% vs sequential 55% by weight; brief's example wins 796/796 in both. Ruling: keep joint —
the measure was chosen in advance and both satisfy the owner's off-list example — report the 50/55 difference to the
owner — cost: slightly more pack overrides of off-list strings. Middle bins overconfident (0.45 vs 0.32) — re-review
to advise. Re-review (opus) of rounds 1-3 dispatched.
Task 10 re-review r1-3 (opus a9b03aabd45a3c5c8): 8/8 original addressed, no cutoff breach. New Important: (1) T fitted
on the targets-named vector, applied to the read-time vector, persisted nowhere; (2) boxes holding 2+ Terra words get
per-word targets (mirror of fragments); (3) brief's example run on code boxes only (block lift may reach 7-14 nats).
Minor: mode chosen with the scored quarter; fit_temperature unguarded; built-pack test skips on loader errors; LOO
twins; char shares 2-D. Ruling: fix round 4 by a FRESH opus implementer (the prior one carried ~840k tokens over 3
rounds) — T on the read-time vector + named best path + per-fold persistence + read_posterior recipe; multi-word
targets (measure first); example on block/street boxes; whole-word off-list counts; minors; middle bins parked.
grid.INK_BELOW = accepted pipeline constant.
Task 12: DONE 8b83a08 + 6c8de6b (grey input; full size = half_size off; 27 tests). Note: variant.py must pass the uint8
grey page for a grey run (Task 8's file) — carry to the Task 12 comparison runs. Review package review-task12.diff.
Task 12 review (sonnet a850e1ce28a13339b): Approved (code scope). Blocker for the comparison runs: variant.py:245/256
pass page_ink to predict → a grey net raises; fix in Task 8's runner when Task 12's runs happen (pass sd.grey_bits for
a grey recipe). Minor: bilinear scaling of 1-bit scans in training only (note in the decision); docstring reflow.
Task 12: code complete (8b83a08, 6c8de6b; review clean); comparison runs pending the kept detector.
Task 11 PRELIMINARY (10df504; today's boxes/readers): NE fitted choice −0.033 [−0.057, −0.010] (s0 77 vs 92, s1 84
vs 88); plain −0.105. Fitted picks as_found 2124 / join 328 only. 17/18 lost labels (s0) held by a chosen join box.
Page experiments/wordread/review/choose-prelim/ (merged_with_word 293→297; page headline 0.377→0.287) — not
published. Concerns: label_of imitates Terra rects (1212/2340 NE labelled join/trim); a join overlaps its neighbour's
box and takes its label. Controller: test the cause first (overlapping outputs; label theft via T3's
highest-posterior rule; label_of correctness; counterfactual with joins consuming neighbours) before ruling on a
choose.py fix (joint/partition choice).
Task 11 cause (NE s0): the neighbour always stays (261 as found, 67 own join) but the counterfactual removing it gives
76/285 (−0.004 vs fitted) → overlap is NOT the cause; the joined box reading two words is (17/17 wrong codes, 15/17
meet 2+ Terra words). label_of: of 697 join labels only 203 (29%) had a Terra word split across our boxes → labels
by rect IoU reward joining another word because our boxes run short. Ruling (Task 9 fix round 3, resume
afe082cdb3c842809): soft labels = Poisson-binomial P(candidate holds exactly one word) from boxmodel.ink_shares; a
soft-target conditional logit; no IoU — the reading unit is one whole word, measured by ink — cost: one round. Then
Task 11 reruns the preliminary.
Task 9 fix 3: fb02e3b (label_weights: P(exactly one Terra word) by ink shares; soft-target conditional logit; a box
counts min(1, Σ weights) — structural: one box is at most one observation; choose.py imports boxmodel (torch)). Task 11
told to switch to weight= and rerun the NE preliminary.
Task 11 rerun with ink labels (ea969cf): WORSE — NE s0 46 vs 92 (−0.161 [−0.212, −0.105]), s1 62 vs 88 (−0.091);
fit picks join 1559/2452 (64%) while join label weight is 745/4450 (17%); β 0.837, α_join +0.243; matched 850→498,
merged 560→915. Controller: test why first (in-sample calibration and convergence, score gap by label, train vs
held-out feature shift, candidate duplication splitting as_found's probability).
Task 11 "Why joins": fit at optimum (refit identical, grad 0.064), calibrated in-sample (P join 0.277 vs target 0.239);
reader score not higher on joins; NE joins nearer (along 0.14 vs 0.42-0.52 chars). Cause: as_found + 2 trims are
near-duplicate crops of one word, so competing shares split three ways; join stands alone (83.5% of join-chosen boxes
had as_found+trims > join). Ruling (Task 9 fix round 4, FRESH opus implementer): per-candidate fractional logistic of
each candidate's own quality ŵ_c = P(exactly one whole word) on its features (+ box-model log P(count=1) when
present); choose argmax ŵ_c; no competing shares — a decision rule over duplicates can't split a vote — cost: a
rewrite of fit/choose; runs the NE preliminary after.
Task 9 fix 4 (fresh opus ab4a08aced0d0c102): 91ad2a1 — fit_quality/predict_quality per candidate; choose by ŵ_c;
1849 passed. NE preliminary (today's boxes): s0 81 vs 92 (−0.039), s1 86 vs 88 (−0.007), mean −0.023 [−0.041,
−0.004]; joins 191/164 (was ~1500); 11 of 13 lost labels held by joins. Ruling: item (5) NOT kept on today's boxes;
re-test with Task 15's box-model P(count=1) feature on Task 8's boxes (the missing pixel-level signal) — cost: (5)
decided later. Task 9: code complete (edb6f33..91ad2a1; four fix rounds; the experiment lives in Task 11).
OWNER (~16:00): "Can I see what we have so far? Can you make any kind of map?" → Task 8 builds a PREVIEW page of p07
from saved word-head outputs (core chance shading, tag PCA colours × core chance, sparse axis ticks, Terra + today's
outlines, per-quarter held-out detectors only) at experiments/wordread/review/preview-heads/; controller publishes.
PREVIEW published: https://claude.ai/artifact/SqpLkTJ1rRtfR2Uwt2RH9g (p07, all four quarters held out, a-*-s0
sign-free). NE: axis 1.1° px median (90% within 10°), word length ×1.36, height ×1.07, centre 0.10 heights; 59 NE
Terra words with two tag clusters (display-only rule) vs today's 241 merged_with_word. SE/NW/SW similar (axis
0.9-2.1°). Task 6 committed 894ffdc (round 3 — review pending). Task 8 Phase 1 page underway with cross-fitted spreads
(SE/NW/SW detectors on their own quarters); NE s1 training.
Task 13: DONE_WITH_CONCERNS a7debd6 (scenes.py, Recipe.scenes=0.0, run_scenes.py; 17 tests; 1821 passed). Measured:
block outlines drawn 0.72-0.81 of length; codes 0.0027-0.0045/lot (offset q50 44-56 ft); block numbers 0.79-0.85/block
(offset q50 18 ft); street names 0.34-0.51 per 1000 ft; words/crop real 7.7 vs scene 4.0; pasteable 0.375. Controller
looked at sheet.png: sparse (5/12 no words), lot lines dashed, rotations OK. Rulings → fix round 1 (resume
a404ddd47d2020c81): same-kind stand-in words when no same-string twin (the detector never uses the string); partial
drawing as whole edges at the measured per-edge rate, not dashes. PENDING OWNER: the scene share in batches (no
measured quantity defines it) — ask at the scene run. variant.run bank line (Task 8's file) at run time.
Task 8 queue done: arm (a) SE/NW/SW s0 + NE s1; every fold axis 90-95% px within 10° (76-90% words); height
×1.02-1.11; length ×1.16-1.56 per word; centre 0.10-0.14 heights; NE seeds agree. 0884d89 (preview_heads.py, tests).
Phase 1 page: cross-fitted spreads measuring. GO: base controls NW/SE s0, then arm (b) NE s0/s1 + SE/NW/SW s0.
Task 15 resumed: train the box model per fold on GPU (channels from the detector that held out each example's
quarter: page-derived stacking), report calibration/boundaries/p07 merged counts.
Task 13 fix 1: 7b755ce (same-kind stand-ins: blocks 42%, streets 10%; whole-edge drawing from a drawn/undrawn
mixture; scene_run share required; words/crop 5.08 vs real 7.63 (2000: 11.6 vs 13.7)). SIMPLIFY_PX 0.5 px accepted —
sub-pixel corners are invisible at page resolution (physical) — cost: none. Centrelines drawn too often (the 2000 map
prints none; the measured ink is likely street-name text) → Ruling round 2: re-measure per-edge inked fraction with
Terra words' boxes excluded; draw each edge's inked fraction from the measured per-edge distribution (contiguous run).
Task 6 round 3 committed 894ffdc (words.py +1483/−; page_synthetic.py). Task 8 blocked: measure_spreads needs a
pixel mask (answered blocks cut pages into windows; fragments distort spreads) and divides by zero on 1-2 px
fragments (words.py:353); first real page: centre sd 2.1 heights, angle 11.8°, cell 8,759, background share at float
floor 2e-308 (suspect). Ruling: a FRESH opus implementer (Task 6 round 4) commits the mask + exact curve==0 guard
first, checks the floor; a parallel opus re-review of 894ffdc; its findings become round 5 — urgent unblock vs a
long-context implementer — cost: two fresh contexts. Task 8 arm (b) NE s0 training; controls NW/SE done.
Task 13 fix 2: af3cd9a (word boxes excluded from line measure; each edge one contiguous run drawn from its kind's
measured per-edge inked fraction; realised matches measured, e.g. 2000 blocks 0.758/0.755). My guess was WRONG:
centreline ink (12%/edge with text out) is heavy district boundaries down street centres + freeways/railways, not
text. New defect: heavy specks from crossing ink counted as a lot edge's own (93% of "inked" px on small-fraction lot
edges run > 45° to the edge). Ruling round 3: structure-tensor orientation at the MEASURED stroke width; von Mises
"own" + uniform "crossing" mixture fitted per kind/map; px count by P(own | angle) — no picked scale or cut.
COLLISION: the old Task 6 implementer (ad4c7998d4dc9a2c3) kept working after 894ffdc (e58759f, 307b5d3 + uncommitted
exact-gradient hunks; page_synthetic pid 20651) while the fresh one (affa8608a83a65c42) started. Ruling: old commits
sound work (or reverts its own unsound hunks) and stops; fresh waits for a clean words.py, then adds mask + guard on
HEAD, normally committed (no `git apply --cached`). The re-review (ac15fd944777e9cb9) of 894ffdc alone is incomplete →
send it the later commits' diff once the old agent's final commit lands. Lesson: before spawning a fresh implementer
on a task, confirm the old one has stopped.
Task 10 fix 4 (fresh opus abc6f35b276a2ddd0): 4273cdf. Joint kept every fold; τ 0.68-0.72; w code 8.7-14.3, block
1.9-2.8, street ~1.1; T 2.19-2.34. Pooled held-out top-1 vs Terra: pack 0.669, ink 0.584. Read-time reliability:
middle bins calibrated; top still overconfident (0.98→0.92). Brief's example wins in every code/block/street box
(min margins codes 4.2, streets 5.9, blocks 0.70 nats sw). Off-list kept: whole words 62% by weight (code boxes 1/11).
Multi-word pairs: a "line model" (box read as one line; same-place words as alternatives) −4.7 vs −24 literal.
read_posterior(pack, fold, ink_logp_fn, box, strings=()). Rulings: line model and box-weight form accepted pending
re-review; axis-across-text 4.5% left (detector failure); code boxes' strict off-list behaviour → tell the owner.
Re-review (sonnet) dispatched; Task 11 runs the pack preliminary on today's NE boxes.
Task 10 re-review r4 (sonnet abe192d10df7912cb): 1, 3, 4 addressed; 2 OPEN — the line model has hard rules (_apart
≤ 0, own-ink span windows, binary own ink, fallbacks, cliques, sorts, argmax kind) and the −4.7 vs −24 comparison
wasn't like for like; the box weight is a picked product. Ruling (round 5 = last, FRESH opus): exact share model —
pair weight w_bi = s_i Π_{j≠i}(1 − s_j) (Poisson-binomial "only word i"); target = word i's inside part; Σ_i w_bi =
P(count=1) ≤ 1 structurally; remove the line model; per-pair kind/block/quarter; refit; report under one measure on
the same pairs — no rules, no picked forms — cost: τ/w/T shift. Small: fit_temperature quarter required; read_posterior
ValueError on a missing key + year check.
FINDING (fresh Task 6, p07 SE, a-se-s0): the centre head learned NOTHING (pred vs true corr 0.04, slope 0.003;
centre loss flat 1.34-1.51 from iteration 250; Kendall weight → 0.355; arm b the same). Each px votes for itself; the
"0.1 heights per-word centre" was the mean of px positions; the measured centre sd 2.1 heights and cell 8,759 are word
extent and a ramp. Axis, sizes, core and tags learned. Ruling: measure_spreads calibrates each vote dimension (true ≈
a + b·pred, σ) and the likelihood uses calibrated votes, so b≈0 dims drop out by measurement; cell measured after
calibration — cost: one change in words.py. Task 8: test the cause (receptive field vs word half-lengths) and use the
calibrated spreads for Phase 1.
Task 6 round 3 final (old implementer, stopped): 894ffdc, e58759f, 307b5d3, 4aa6409. Signature core_words(core_logit,
geo, tag, ink, spreads, prior, *, kind_p, sheet); Spreads gains biases, outlier, edge, core_share, background stats.
Fitted weights fixed the duplicate copies (the 30-extra cause on test cases); full page memory 4.79 GB (was 14.1);
search ~1 h/page estimated. Rulings: 1-px minimum box (physical), TRUST = 1 (optimizer path), sqrt(eps) eigen floor
(numerical) — accepted. EDGE_SDS → the reviewer judges. Reviewer sent the later commits (review-task6-fix3b.diff).
Task 6 re-review r3 (opus ac15fd944777e9cb9, on 894ffdc): 1-6 addressed (minors); 7 (30 extras) and 8 (cost) not;
9 undercut. Important: outlier = background density → share → float floor (Task 8's 2e-308 explained); edge
mis-measured (votes vs Terra; softens q → duplicate route via C); search cubic (pairs re-tested; _near rebuilt);
EDGE_SDS ignores K. Ruling: Task 6 round 5 (LAST) to the fresh implementer affa8608a83a65c42, ordered: A
calibration, Student-t votes with measured dof, edge from core px, EDGE_SDS with K; B cubic fix + full-page
measurement; C _final held words, _fit region, size guard, labels, core-weighted smoothing, tests, 30-extras run.
Task 8 RF test: reach 256 px (effective 90% within 130); the centre fails INSIDE the reach too (words wholly within
50 px: corr 0.08/0.06, slope 0.016; pred median 1.8 px vs true 9.6) while log length learned (corr 0.82). Task 8
corrected its earlier "0.10-0.14 heights" (a centroid artefact). Controller: test the cause first — Task 5's agent runs an
8-crop overfit test (with and without Kendall) and checks target/loss wiring before any fix. Phase 1 proceeds on the
stopgap fit; calibration drops the centre by measurement.
Task 6 round 4: 814ab51 (mask in measure_spreads page dicts; exact curve guard; no-mask bit-identical; 79 passed). Task 8
told to use it. Ruling: calibration as a FORWARD measurement model pred = α + β·true + η (likelihood on (pred − α)/β,
sd s/|β|) — flat at β ≈ 0, no prior counted per vote; inverse fit only for point placements — cost: none.
Task 6 re-review r3 updated for e58759f/307b5d3/4aa6409: the later commits are correct (tags-first EM step monotone;
_move same likelihood; exact gradient re-derived). EDGE_SDS = Minor (does not decide shares; reach should come from
K_max: −ndtri(2^-53/K_max) ≈ 11.7 sd) — already in round 5 A4. Open items = round 5 as sent (outlier, edge, cubic,
30-extras page count, NaN size guard, weak tests). Nit: task-6-report.md Status line garbled.
Task 11 PACK PRELIMINARY (5116bed; today's NE boxes, read_posterior): headline s0 92→89 (−0.0105 [−0.024, 0.000]),
s1 88→87 (−0.0035); code labels s0 3 R→W, 0 W→R; s1 2 R→W, 1 W→R. Reads vs Terra (matched 0.5): streets 65→88/82,
blocks 371-375→431-433, codes 99-101→108-111. p07 text matching Terra 330→373 of 448. By the written rule (4)'s
headline condition fails (not kept for the headline); street/block gains noted for the solver — owner's decision
later. Page experiments/wordread/review/pack-prelim/ (not published).
OWNER (~17:10, frustrated): "I can't believe you don't have any intermediate things you can show me ... you're
scoring things so you must have intermediate results." → Published: pack-prelim page
https://claude.ai/artifact/VQra75hKcAEdMJBquyy6x3 (owner: looks like before; all lines still red — it is today's
detector, my mistake to offer it as new); progress page https://claude.ai/artifact/PnbodJhmTTyTJjPKXgKzfE (all
numbers, verdicts, contact sheets, links). Task 8 asked for a DRAFT boxes page now (814ab51 + current spreads).
Lesson: publish intermediate results as they land (a progress page), don't wait for finished work.
OWNER (~17:20): "You're telling me that after all of that work today we're at 40.9%?" / "you absolutely should have
validated the ideas before spending 8 hours on them." → Ruling: END-TO-END FIRST. The draft new-box run (Task 8:
new detector boxes → today's reader → NE headline) is the only priority; no more polish rounds on parts whose value is
unmeasured. Memory updated (principled-loop).
Task 13 fix 3: cd9bc5f (structure tensor at measured stroke width; von Mises own + uniform crossing mixture; specks
gone; integer run lengths with the fraction's expectation; drawn matches measured within 0.004). New: outline gaps
at corners/junctions (von Mises too narrow). PAUSED here (parked: heavier-tailed own component; σ = width/2; gaps) —
resume only if the end-to-end run shows training data is the bottleneck.
CENTRE HEAD ROOT CAUSE (Task 5 agent, tested): targets/wiring correct (match 2.3e-7); both weightings overfit 8 crops;
on fresh crops nothing learns, but restricted to words wholly inside the crop it learns fast (corr 0.87/0.86 at 200
steps). At the 320-px crop, 49.6% of core px belong to words cut by the crop border and carry 85.3% of the predict-0
loss. Ruling: geo_known = 0 for words reaching outside the crop window (exact geometric test; axis kept) — it fixes the
main lever cheaply — cost: one NE retraining. Task 8: after the current arm (b) SE s0 job, train a' (fix) NE s0 instead
of arm (b) NW/SW. Draft page stays first. Task 8 draft: spreads oscillate (dead centre); told to kill the slower fit
and report the NE headline number first.
Task 10 fix 5 (fresh opus a5e6b1db6539679e6): be2acd9 — exact share pair weights s_i Π(1 − s_j); per-pair
kind/block/quarter; line model gone; refit (τ 0.62-0.68, w_block 2.24-3.56, w_code 7.8-12.5, T 2.73-2.94); same 5,754
pairs, one measure: NLL 2.183→2.165, top-1 0.713→0.717; reliability within 0.04 (0.2-0.7). Concern: the brief's off-list
example FAILS in 193/5,682 pairs (small blocks, w_block ≈ 3.5; down to −2.9 nats); off-list kept 56→42% whole words.
Breaker (round 5 of 5) → adjudication: not load-bearing now — the pack was not kept for the headline (Task 11
preliminary) → PARK: Task 10 complete with this concern recorded; fix before any adoption (owner's call).
Task 10: complete (commits cd2bf9b..be2acd9; 5 rounds; open concern parked: small-block override of off-list ink)
Centre fix committed dadc5d1 (cut words geo_known 0, axis kept; tests at the window edge, turns, scale 2; guards 157
passed; it also adjusted 3 Task 7 doubt tests whose crops cut words). Task 8 told to train a' NE s0 next.
Task 15 trained (b27aa1c): 4 folds s0, 9 channels (stacked held-out detector outputs). Count CE held out 0.48-0.58
vs prior 0.72-0.82; P(0) calibrated; over-predicts P(2) 2-9×; boundary head weak (NE/SE ~45% recall at 0.5; NW/SW
none). P(≥2) > P(=1) on p07 merged_with_word boxes: 108/123; on held-out boxes with ≥2 targets 136/180; false flags
176/6,107. Use through Task 9's fitted choice (not raw). Not further polished (end-to-end first).
OWNER: load guidance only prevents ~100 and contention; don't kill to hit 20 (memory cap-cpu-load updated; Task 8
told). Task 5 implementer stopped (its work committed).
Task 6 round 5 (affa8608a83a65c42, WIP, committing after tests): Spreads drops outlier; adds forward calibration
(slope/bias), Student-t dof, per-word shared error, truth stats; edge = core px about the fitted box; page dicts take
"core". p07 SE a-se-s0: centre β 0.003 (drops out), log length β 0.75 / log height β 0.60 (R² ~0.85-0.88), axis t
scale 1.46° (dof 1.8), cell 1,065 px (was 8,759), edge 0.289, fixed point settles. Task 8 told: finish the 814ab51 draft,
then re-run the draft on round 5, then on the centre-fixed a' NE s0.
Task 8 draft on 814ab51: 0 words formed (spreads oscillating: centre sd 10.5 heights, cell 3.7M; prior from 4
decisions). draft_r5.sh waits for Task 6's round-5 commit. Ruling (end-to-end first): a CRUDE chain in parallel now —
MAP core components (P(core) > 0.5), min-area rects along the core-weighted axis, no tags/votes → NE headline vs
92/285 — a fast lower-bound test that new boxes can move the headline; MAP (even odds) is a decision rule, not a picked
level — cost: a rough number with merges. a' NE s0 trains when the GPU frees (b-se-s0 done).
Task 6 round 5 committed 0b718a6 (88 passed; exact log(1 − q); bounded gradient forms; basin starts at core-weighted
centres; indexed search). Task 8 told to start draft_r5 now alongside crude-map. Task 6 full-page synthetic count and
real-page measure running; full report to follow (then breaker adjudication).
FIRST END-TO-END (crude-map, a-ne-s0, NE, today's NE fold reader): 142/285 = 0.498 [0.441, 0.556] vs today 92/285 =
0.323; paired +0.175 [+0.113, +0.239] → the new detector's boxes move the headline. Matched 850 → 1,312; missed
1,855 → 1,393; geometric merged 560 → 452; boxes 2,452 → 5,062 (many specks read as none). Chain: 3 min. Progress page
v3 published. Next: crude-map on SE/NW/SW → the full 699-label headline vs 0.4092; p07 crude-map page; round-5 chain;
a' centre check.
FULL HEADLINE (crude-map, all four quarters, own held-out detectors and fold readers): 343/699 = 0.491 [0.453, 0.528]
vs 0.409; paired +0.082 [+0.037, +0.126]. NE +0.175; NW +0.021; SW +0.043; SE −0.008 (merges up outside NE: no tags).
Boxes with a label 621 → 685. p07 NE page published https://claude.ai/artifact/2xzg3uJiXy9xxN7botpGmf (matched 377 vs
252, merged_with_word 194 vs 241, labels 46/87 vs 29/87). Progress page v4. Round-5 first prior degenerate (slope
−0.0001, 30 proposals). Next: "crude-tags" (tag split within MAP components, measured tag spread) on all quarters.
p07 crude-map, all quarters (own detectors): matched 578 vs 447; missed 528 vs 659; merged_with_word 266 vs 293; owner's
labels on p07 71/122 = 0.582 [0.496, 0.667] vs 46/122 (Terra 107). Published to https://claude.ai/artifact/2xzg3uJiXy9xxN7botpGmf
(v2, whole sheet; replaces the NE-only v1).
OWNER (~18:35): "It sounds like you're going to split them based on some kind of arbitrary cutoff. Shouldn't you instead
learn that an individual word is an individual word? ... What's causing you to expand the search outwards in radius?"
Explained: no radius search — core target = whole rect, neighbouring rects touch, core blur bridges gaps → one MAP
component. PROPOSED (awaiting yes): a learned same-word AFFINITY head (P(px and right/down neighbour in the same word),
exact targets from Terra rects), words = MAP-linked px; retrain NE once; score vs today and crude-map. crude-tags HELD.
a' (af-ne-s0, centre fix dadc5d1), held-out NE: centre loss 0.286 (a: 1.78), size 0.053 (a: 0.100); centre corr by
far-end distance 0-50 px 0.59/0.34, 50-130 0.89/0.60, 130-256 0.73/0.53, >256 0.46/0.62 (short of truth beyond reach);
per word: axis 1.1° (87% within 10°), length ×1.06 (a: 1.44), height ×0.97, centre 0.31 heights. crude-tags stopped
before forming words (code kept uncommitted behind --tags). GPU idle awaiting the owner's affinity decision.
OWNER: "Yes try affinity". Dispatched: a fresh opus implementer adds heads="words_aff" (18 ch: +16 right, +17 down
same-word logits; targets from the word id map; bg-bg pairs unknown; balanced BCE + Kendall; predict field;
heads="words" unchanged); Task 8 prepares crude-aff (MAP core px linked by MAP same-word edges → components → boxes)
and trains a″ = a′ + words_aff NE s0 on commit, scoring vs today (92/285) and crude-map (142/285), paired.
Task 8 crude-aff ready (aff_labels: MAP core px, 4-connected MAP same-word edges, csgraph components; vs_run pairs
runs; train --heads words_aff; save/load all WordOutputs fields); queue_aff.sh waits for the words_aff commit.
Implementer told: predict field `aff` (2, h, w), ch0 right, ch1 down.
AFFINITY committed 7f67fca (heads="words_aff", 18 ch, AFF = 16-17, WordOutputs.aff (2,h,w), loss "affinity" 7th
Kendall part; 23 tests; guards pass). Known failures to clean before the final review: test_wordread_scenes
test_a_scenes_masks_are_exact (geo_known assertion predates dadc5d1; scenes paused) and Task 8's crude-aff test
(SimpleNamespace double aff → o._replace) — Task 8 told. Concerns: "apart" mostly word-bg pairs; word-word pairs only
where Terra rects touch; cross-word pair doubt weight = product (implementer's choice). Task 8 runs queue_aff now.
AFFINITY RESULT (a″ aff-ne-s0; final affinity loss 0.332): crude-aff shatters NE into 373,503 boxes (p07 NE: 87,305
words, 71,496 single px). P(same) > 1/2: same-word core pairs 83%/81% (1 in 5 chipped); word-bg 7%/10% (learned);
word-word 41%/78% "same" (NOT learned: rare in training). Merges: 428 Terra pairs merged; 211 share an edge, 208 of
those via logit > 0. Ruling: (b) Bayes prior correction (add log(n_same/n_apart), measured) — not a cutoff — run now;
(a) and (c) held. PROPOSED to the owner (awaiting yes): centre-vote grouping (px grouped by predicted centre, bandwidth
= measured centre-vote spread) using the now-working centre head; alternative: retrain the affinity with synthetic
touching-word pairs.
crude-aff + Bayes prior (+3.10 log odds, measured both-core pairs): NE 144/285 = 0.505; vs today +0.182; vs crude-map
+0.007 [−0.045, +0.057] → no gain; boxes 8,839; merges: 249 edge-sharing pairs all joined (logit > 0); 583 joined via
off-word core px. Affinity does not part touching words. Round 5 chain stopped. Running: crude-map on a″'s core (control
for the centre fix) and a failure breakdown of crude-map's and today's wrong labels (no box / merged by kind & layout /
partial / clean-but-misread / other), with example crops. OWNER (~19:40): "Do some superpowers brainstorming on next
approaches. You're not thinking" → brainstorming; my reframes sent (kind/height split, code-tolerant reader, code
spotting, measure first); awaiting the owner's answer.
Task 6 final (affa8608a83a65c42): 814ab51 (round 4), 0b718a6 (round 5), 21f4f06 (page_synthetic fix); 88 passed.
1500×1000 synthetic: every painted word exactly one box, no extras/missing. Full page: proposals 55.6 s (4.95 GB);
search est. ~7 h (stopped). Real page p07 SE calibration as reported (centre β 0.003; length/height R² 0.68). Breaker
(5 rounds) → adjudication: correct but too slow for use; end-to-end crude chains are the path → PARKED (speed-ups need
rulings: stop halving once settled; weigh splits only where votes disagree).
Task 6: complete (commits ac8c0f4..21f4f06; parked for speed)
FAILURE BREAKDOWN (crude-map, 699 labels; wrong 356 vs today 413): no box 14 (78), merged 227 (206), partial 3 (35),
clean-misread 39 (26), other 73 (68; loose box 63). Merged by partner: code+block 137 (67 along / 70 stacked), code+code
45, code+street 38, code+other 2, box without the code 5. Today→crude-map: 152 fixed, 95 broken (59 merged). Control:
crude-map on a″ 140/285 (a: 142); crude-aff+prior over crude-map-from-a″ +0.014 [+0.003, +0.028] (~4 labels). Causes page
published https://claude.ai/artifact/SC6Ut8UjjpNSVHR6iBANLk. Brainstorm: the merges (64%, code+block 60% of those) are
the lever — kind/size split and code-tolerant reading both target it; awaiting the owner.
OWNER (~20:05): "Why not go direct to the 'detect straight from pixel'? Why is that the biggest build? We're currently
stringing together lots of small parts and struggling to find solutions. A different approach may not be harder?"
→ Agreed (my sizing was wrong: it's one more head on TextNet). PROPOSED (awaiting yes): per-pixel CODE head (2000
code classes + other text + background; targets painted from Terra code words; balanced CE by measured class
frequency; doubt string weights), label = connected same-code MAP px holding the answered point; train NE once;
score vs today 92/285 and crude-map 142/285, paired.
OWNER: "Run the test" (code spotting). Dispatched to Task 8 (a16c58c91a3a8ff32): heads="words_code" (16 words
channels + C = 2000 codes + other text + background); targets painted from Terra words (2000 sheets only; other series
masked); balanced CE (measured frequency) as a Kendall term; predict field `code`; train NE s0; score: MAP class per px,
8-connected same-code components, the component at the label's point (no windows); paired vs today and crude-map; wrong
labels split (no code / wrong code / merged); rare-class counts. If better: all four folds + a p07 page.
CODE SPOTTING (a483bc5; code-ne-s0): NE 38/285 = 0.133; vs today −0.189 [−0.247, −0.125]; vs crude-map −0.365. Wrong:
no code at point 118 (107 "other text"), wrong code 121, merged wrong 8. Under-fits its own training words: 19% right
(held out 15%); other text 94% right → it separates code from other text but can't name codes; guesses RH-2/P/RH-1;
code term flat at 1.42 from step ~2,000, Kendall weight 0.375. 754 training code words; 18 codes unseen (explain only
11 NE labels). NOT KEPT. Next PROPOSED (awaiting yes): the hybrid — kind-split of crude-map components by the per-px
learned kind argmax (existing outputs, no training), then read each part with the CTC reader.
OWNER: "Yes" (kind split). Dispatched crude-kind to Task 8: MAP core px; 8-neighbour edges only between same MAP kind
(argmax of kind channels); components → boxes along the axis; all four quarters; sanity check of kind on ink vs
non-ink core px (variant: kind of nearest ink px by exact distance transform); full headline paired vs today 286 and
crude-map 343; five-way breakdown again.
CRUDE-KIND (MAP kind split of MAP core components; a-*-s0; fold readers): 414/699 = 0.592 [0.555, 0.631]; vs today +0.183
[+0.139, +0.227]; vs crude-map +0.102 [+0.073, +0.131]; every quarter above crude-map (NW +11.6, NE +8.8, SW +10.6,
SE +10.8). Kind check (held-out NE): MAP kind agrees with Terra 0.784 on ink, 0.835 on non-ink core px (codes 0.75-0.76).
Boxes 80,834 (fragments where kind flickers; reader mostly calls them none); matched 5,554 vs 4,545; merged 446 vs
1,588; split 1,049 vs 195. Progress page v5 published. Five-way causes running.
crude-kind causes (285 wrong): no box 11, merged 123 (code+block 74 = 25 along / 49 stacked; code+code 28; code+street
16; other 1; no code word 4), partial 21 (kind flicker), clean misread 55 (NE 36; Terra wrong 9), other 75 (loose 61).
crude-map → crude-kind: 94 fixed (69 merged), 23 broken. Causes page updated in place (SC6Ut8…).
Next PROPOSED: reader retrained on crude-kind boxes per fold (own-box crops at natural counts; Task 4 re-run) — targets
clean misreads 55, loose 61, partial 21 and some merged — 4 GPU trainings.
OWNER (~22:45): "That text reader progress artifact doesn't show me any boxes or anything that's useful for me" →
Task 8 builds the p07 crude-kind review page (boxes coloured by assigned kind; toggle to crude-map / today; faint
"none" fragments) → publish at https://claude.ai/artifact/2xzg3uJiXy9xxN7botpGmf (Mission New Boxes). Reader
retraining waits for the owner's yes. Lesson: the owner judges by looking at boxes on the map; numbers pages are not
enough — every result gets its map page.
crude-kind p07 page (2661bf3; 13.2 MB; kind-coloured outlines; toggle crude-kind / crude-map / today; faint "none"
fragments) published at https://claude.ai/artifact/2xzg3uJiXy9xxN7botpGmf v3. p07 labels: crude-kind 79/122, crude-map
71, today 46; matched 688 / 578 / 447; merged_with_word 131 / 266 / 297. Some vertical codes coloured street (RM-2, RH-2).
OWNER (~22:50, screenshot of the crude-kind page on phone): "Bro this is unusable. Apply some damn UI principles" →
fresh opus agent redesigns the review page renderer: calm default (grey ink + kind-coloured outlines only), layer
chips, upright collision-avoided reading pills, tap → bottom sheet, a mistakes navigator, version segmented control,
compact header; verified by screenshots at 390/1280 px; numbers unchanged. Lesson saved to memory.
Review page redesign ae1d18e (review.py rewritten; build_kind_page via the renderer; readings never rotated (test);
numbers byte-identical). crude-kind page 7.25 MB; controller looked at phone-390 fit and NE 3x screenshots: calm,
upright pills, navigator "43 wrong labels", version control. Published v4 at https://claude.ai/artifact/2xzg3uJiXy9xxN7botpGmf.
Remaining: labels can sit on other boxes in dense clusters; today's big rotated boxes put labels far from outlines.
OWNER (on the redesigned page; RM-2 + CAMP ST. + ALBION merged): "Explain to me why the text isn't clearly detected"
→ explained (touching patches; kind flicker; one averaged-axis rect per patch; crude chain ignores per-px predicted
extent). "But they're all at different orientations" / "'agrees' sounds like something that should be learned" / "Try
it" → dispatched "learned-links" to Task 8: an MLP on pairs of neighbouring core px ([f_a, f_b, f_a − f_b, f_a ⊙ f_b] +
direction; all 16 detector outputs, stacked held-out inputs) trained on the training quarters' crude-map components
(labels from Terra word ids); Bayes-corrected even odds; components → boxes; 699-label headline vs crude-kind 414 and
today 286; causes; a 4th version on the review page.
learned-links MLP (pairs: 37.95 M same / 2.36 M different, of which 120,820 word–word): balanced-call held-out accuracy
same ~0.78-0.80 / different ~0.84-0.86; with the Bayes prior (+2.8) word–word cut only 1-2% → expect ≈ crude-map.
Controller: the per-pair call is the wrong aggregation (one surviving link merges; weak per-pair evidence) → diagnostic
asked: boundary-summed LLR for merged Terra-word pairs vs same-word halves; if separable → region merging with summed
boundary evidence and a measured segment-level prior.
learned-links 351/699 = 50.2% (vs crude-kind −9.0 pt [−12.0, −6.1]); not adopted. Boundary diagnostic running.
OWNER (11-18 ~00:30): "is this an unnecessary tangent? ... brainstorm what we should do next (finishing the model,
pivoting to a CNN, or using the Terra reads to ship)". Analysis: real gains came from the simple chain (40.9 → 59.2);
six elaborate add-ons failed; the map (T4: our reads → solve → wrong lots vs published 35,455 / saved reads 4,543)
was NEVER measured. Recommended: T4 with crude-kind now + T5 publish wiring in parallel; stop new box research
until the map says boxes are the bottleneck. Awaiting the owner's answer; no new runs started.
OWNER CHOSE: ship with the saved Terra/Sol reads; save the research + full write-up. PLAN PARKED. Task 8 stopped
(e24aa90: learned-links + boundary diagnostic, AUC 0.695; 2,536 merged pairs joined only through non-word core px).
Branch text-reader-research pushed at e24aa90 (re-push after the write-up commit). Shipping plan:
docs/superpowers/plans/2026-11-18-ship-2000-saved-reads.md (b23e280).
Task 4 BASE 2cfac43. Implementer (sonnet) dispatched — runs base, (a), (b) on NE s0/s1 on the GPU (exclusive turns;
test sessions hold shared turns, so trainings may wait for gaps).
Task 3: minor (deferred): a box exactly 90° from Terra's angle is not turned (_turned uses cos < 0).
Task 2: minor (deferred): review.matching reads measures' private _overlaps/_best_word (make public wrappers).
Task 2: minor (deferred): the script's stop on missing best paths should name textread.read.
NOTE for review packages in parallel: build each task's package from ITS OWN commits (git log --author/paths), not
a BASE..HEAD range that includes the other tasks' commits.

TASK 8 FINAL (11-18, stopped at the owner's pivot to shipping the saved Terra/Sol reads). All 699 answered code labels,
each quarter from its own held-out a-*-s0 detector and fold reader, paired by whole blocks:
- crude-map 343 (+0.082 vs today 286); crude-kind 414 = 59.2% (+0.183 [+0.139, +0.227] vs today, +0.102 vs crude-map);
  learned-links 351 = 50.2% (+0.093 [+0.048, +0.138] vs today, -0.090 [-0.120, -0.061] vs crude-kind; per quarter
  NW 91, NE 150, SW 50, SE 60; causes: wrong 348, no box 14, merged 221 (code+block 130), partial 3, clean 36,
  other 74). Code spotting (heads words_code) NE 38/285 (the head under-fits its training codes: 19% at word centres).
- learned-links pairs: 37.95 M same / 2.36 M different (120,820 word-word); MLP held-out balanced call same ~0.79,
  different ~0.85; with the Bayes prior (+2.8) only 1-2% of word-word pairs cut.
- Boundary diagnostic (finished; held-out quarters, the balanced logit summed over each boundary): 866 touching
  merged Terra-word pairs vs 9,752 same-word halves; AUC sum 0.695, mean 0.717, length-matched 0.766; code+block 0.63,
  code+street 0.67, code+code 0.58 (other+other 0.85); 2,536 further merged pairs touch through no-word core px only.
- Outputs (pipeline/data/solver/experiments/wordread/): detector/runs/{crude-map-all-s0, crude-kind-map-all-s0,
  crude-links-all-s0, crude-aff-prior-core-ne-s0, code-ne-s0, a-*-s0, af-ne-s0, aff-ne-s0}; detector/links/ (pairs,
  models, counts.json, boundaries.json/.parquet); causes/ (causes-<run>.json, labels-<run>.parquet, crops);
  review/crude-kind/review.html (4 versions: crude-kind, crude-map, today, learned-links); review/crude-map/;
  review/code-spotting/. Report: task-8-report.md.
