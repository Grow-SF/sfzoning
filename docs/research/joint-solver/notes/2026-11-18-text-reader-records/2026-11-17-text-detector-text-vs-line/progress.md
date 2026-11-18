# SDD ledger — plan: docs/superpowers/plans/2026-11-17-text-detector-text-vs-line.md

Spec: docs/superpowers/specs/2026-11-17-text-detector-text-vs-line-design.md (be37735). Plan red-teamed
(.superpowers/sdd/redteam-text-vs-line-plan.md) and corrected before execution. Owner asleep; overnight GO
(memory decisions-log 2026-11-17 ~00:35). Branch `solver`, worktree saved-reads, shared with the text-reader plan's
T3 (agent a98872a5bcddaf07c), which is building stages here.

HARD GATE before Task 1: no `sfzmaps build|stage` running (`pgrep -fl 'sfzmaps (build|stage)'` empty) AND the T3
ledger shows T3's builds recorded. The plan file itself is still uncommitted; commit it at that point.

## Pre-flight scan

### Pairs sharing a file or an interface
| Tasks | Produces → consumes | Found |
|---|---|---|
| 1 × 2 | `linerules.line_px`, `rule_accuracy` → the gate | Consistent. |
| 1 × 4 | `line_px` → `zones.prepare` `line_bits` | Consistent. |
| 2 × 3 | `gate`, `folder_source`, `write` → baselines | Consistent. |
| 2 × 5, 7 | `blocks.npz`, `paired` → Task 5's value check, `decide.verdict` | Consistent. |
| 3 × 5 | panel, audit, `baselines/` → the variant comparisons | Consistent. |
| 4 × 5 | `Recipe.labels`, `zones.prepare` → `variant.run` | Consistent. |
| 4 × 6 | `line_bits`, `crop_targets(labels=)` → `Bank.crop` | Consistent. |
| 5 × 6 × 7 | `variant.run(bank=)` written in Task 5; Bank exists only after Task 6; Task 7's CLI uses `--composites P` | Gap: who builds the Bank in the CLI. Ruling V1. |
| 7 × 8 | the verdict → the preview | Consistent. |

### Each task against itself
| Task | Found |
|---|---|
| 1 | Consistent. The chain rule is measured, not used. |
| 2 | Consistent. The `crossed_word` fixture with a FakeGeo is the implementer's. |
| 3 | Consistent. The old quarter models' chance paths come from the red team; verify they exist. |
| 4 | Consistent. Pins `RECIPE_NAME_BEFORE` before the change. |
| 5 | Consistent. Its stop rule uses `linegate.paired` directly. |
| 6 | Consistent. Pins `PINNED_BATCHES_SEED3` before the change. |
| 7 | Consistent. |
| 8 | Consistent. Preview only. |

### Rulings (pre-flight)
- V1. Ruling: Task 6's implementer extends `scripts/textlines/variant.py`'s CLI so that `--composites P > 0` builds
  `composite.Bank.build(...)` before the GPU turn and passes it to `run(..., bank=)` — Task 5 cannot import a module
  that does not exist yet — cost if wrong: a small edit in Task 7.
- V2. Ruling: the composite share P is the word-weighted `cross_by_kind` over all kinds (Task 7) — the spec says
  "the measured share of line-crossed words", one value — cost if wrong: a p sweep later.
- V3. Ruling: implementer models — opus for Tasks 2, 4, 6 (geometry/mapping, the label rule in hot paths,
  composites); sonnet for 1, 3, 5, 7, 8 — cost if wrong: turns.

## Tasks
Gate passed ~01:15: T3's builds recorded (T3 agent stopped after its report), no build running. Plan committed
aac94be.
Task 1 BASE aac94be. Implementer dispatched (sonnet), agent ab977a002712b5f4c.
Task 1: DONE d008d0a. Run rule: line recall 1.00 on straight cases + hatching; glyph false-line ≤1.4% mean
(putText), learned font 0-1.2% mean, worst RM-4 8.9% (line through) / 11.4% (hatching). Misses stair-step (9%)
and dashed (2%) by design (M1a's job). Chain rule: stair-step 96% but glyph false-line 7-32% (51% hatching): not
used. Concerns: hatching spacing read as perpendicular 10 px; chain run=27; rules.json written by a scratchpad
script. Reviewer (sonnet) ae4813b660b9be50b.
Task 1 review: spec ✅, Approved, 2 Important: rules.json from a scratchpad script (no repo regenerator); the
stair-step case doesn't cross the word on seeds 0-2. -> fix round 1 (resume ab977a002712b5f4c).
Task 1: minor (deferred): the copy-equality test pins one length; assert RUN_ANGLES and LONG_LINE_FT equal too.
Task 1: minor (deferred): rule_accuracy can return glyph_false_line None (no far glyph px); tables don't say so.
Task 1: minor (deferred): no test pins the hatching's perpendicular 10 px spacing.
Task 1 fix 1: 97ec46b (scripts/textlines/rules.py; the staircase crosses the word). Stair-step now: run rule recall
0.08-0.09, glyph false-line 3.0% putText / 3.9% learned font (worst 9.8%); chain 0.97 recall, 26-34% false-line.
Full suite 2 failures from T3's fixer's uncommitted edits (shared tree), 1465 passed before. Re-review afd96a8aab167f1c4.
Task 1: fix round 1/5 (2 addressed, 0 open; commits d008d0a..97ec46b).
Task 1: complete (commits d008d0a..97ec46b, review clean)
Task 2 waits for T3's fix round (commit + solver-text-read build): a new untracked file under pipeline/src would
stop T3's build from starting (build.py:665 dirty-tree check).
T3 fix round done (7509ac7..8157a97; read stage rebuilt and recorded; no build running). Task 2 BASE 8157a97.
Implementer dispatched (opus), agent a47d25c677f7f4d6d.
Task 2: DONE_WITH_CONCERNS 1bba63c. 10/10 tests incl. real round trip (p02 all within 1 px; rare misses up to 7 px
on p06 outside the frame). gate ≈ 60 s on 13 sheets x 4 quarters, 7 GB peak. All-sheets m1b_in_box called_05 0.804
(red team ~0.82). w(d) touching 0.83/0.75/0.43/0.10 at d=0/3/5/8; across streets 0.74 at 0, ~0 at 6, bump 0.14-0.24
at 8-12. Concern: M1a-in-box: 31% weight from d>=7, 44% on M2's base (letters) -> diluted. Reviewer (opus)
a19fda7c49bfcf920.
Task 2 review (opus): spec ✅; Needs fixes. Important: (1) M1a-in-box ~31% weight from the profile tail d>=7, two
thirds of it glyph ink called at M2's rate -> a correct detector is penalised (r328 − all-sheets −0.313 overall vs
−0.414 on d<=6); (2) the across-street bump at d=8-12 is two-sided block outlines (context), removed by (1)'s fix.
Ruling: M1a's weight ends where each class's excess f_D − f_L first falls to its measured far-field level (stated
rule, measured from the profile, ~6-7 touching / ~5 across); the tail is reported as m1a_in_box_context (and
m1a_context), a diagnostic; Task 7's verdict uses the core — the reviewer measured it (69% of weight; all-sheets
0.573, r328 0.987) — cost if wrong: less sensitivity at stair-steps beyond the core.
Ruling: fold two minors into the fix round: #4 each measure states what it misses in scores.md (a spec requirement:
"a gate states what it would miss") and #3 paired() raises when px differ on a common unit (it guards the decision)
— cost: small.
Task 2 fix 1: 7fc310d. Cuts: touch d<=7, across d<=5 (+ a second stop where the excess stops falling: the across
far-field level is negative). m1a_in_box core: r328 0.9874, all-sheets 0.5794; paired all-sheets − r328 −0.408
[−0.429, −0.388]. 1480 passed. Re-review (sonnet) a724cc29c3f1f58d0.
Task 2: fix round 1/5 (4 addressed, 0 open; commits 1bba63c..7fc310d). Real across cut d=6 (< the 8-12 bump).
Task 2: parked — the second stop (e(d+1) > e(d)) has no noise tolerance and the "outer third" far-field window is a
fixed fraction; an early wiggle on another map's noisy profile would cut the core short — Ruling: acceptable for the
2000 map (cuts printed in scores.md, visible); revisit for other maps (smooth e or a noise-based tolerance).
Task 2: minor (deferred): markdown() raises KeyError on an older scores.json without far_level/cut.
Task 2: minor (deferred): MISSES' m1b recall figures (0.08, 0.02) are hardcoded text.
Task 2: complete (commits 1bba63c..7fc310d, 1 parked)
Task 3 BASE 7fc310d. Paths checked: old quarter models data/solver/text-{nw,ne,sw,se}/chance/<sheet>.png; trial
data/solver/experiments/detector-overcall/chance/trial/ (also r_{q} folders there).
Task 3: implementer dispatched (sonnet), agent af4d80f7367b63fcd.
Task 3: DONE 837c994. Baselines (called>=0.5, m1a_in_box / m1a / m2): r328 0.987/0.668/0.999; all-sheets
0.579/0.046/0.983 (m1b_in_box 0.804); trial 0.643/0.059/0.985; old quarters m1a_in_box 0.63-0.73, m2 0.965-0.983.
r328 m1c 0.366. Audit: r328 0/64 text pieces called... (missed by eye 0/64 [0, .056]); old ~4/64.
Controller looked at panel.png: r328 colours nearly every line red; the r30 columns colour letters red and most
lines grey, but lines crossing/touching labels stay red (RH-2, UNION NC, RM-2 crops). The failure is visible.
Reviewer (sonnet) a20f3f6a26b3056f9.
Task 3 review: spec ✅, Approved, no Critical/Important. Reviewer ran paired(r328, old-ne): works on NE units (654-
1514 common blocks), px equal. ⚠️ old models held out? -> controller: detector-overcall/results.md lists them as
"old quarter models (r30, out)": held out by quarter. Resolved.
Task 3: minor (deferred): summary rows mix bases (r328 all quarters vs old-q own quarter); add per-quarter rows or say "use paired".
Task 3: minor (deferred): panel rendered by monkeypatching panel.page_ink globally.
Task 3: minor (deferred): copy_today skips an existing destination without checking it matches.
Task 3: minor (deferred): audit gives counts only, not the scaled-up estimate (~290 text pieces for old models).
Task 3: minor (deferred): "hatched" window is small hatch-filled lots.
Task 3: complete (commits 837c994, review clean)
Task 4 BASE 837c994. Implementer dispatched (opus), agent a1fdcd95036a90e1d.
Task 4: DONE_WITH_CONCERNS 97dbfdc. 1500 passed; boxes guards unchanged; Recipe().name() "default"; test_build ok.
zone_radius 84 px (q50 85, q90 114.4, q99 169; 6197 words); 177/8246 points keep a zone. In-box ink unknown 21.8%
overall (2000 codes 25.9%). Concern: run rule marks long letter strokes of big/bold labels (P, M stems) as line:
12% of 2000 code words lose >= half their ink (unknown, not wrong). Controller looked at labels_rule_high_share.png:
real lines through words AND some long letter strokes are red (unknown). Accepted for the control (composites teach
exact masks). Reviewer (sonnet) adbf897b897507ed4. Six stages now stale on paper (expected; not built).
Task 4 review: spec ✅, Approved, no Critical/Important.
Task 4: minor (deferred): an unlisted, mechanical test edit (a monkeypatch lambda forwards **k) beyond the brief's two.
Task 4: minor (deferred): read Task 5's M2 for 2000 code words beside the per-word unknown-share table (12.1% of words >= 0.5).
Task 4: minor (deferred): 113 points matched only by a non-code word lose their zone (beside the 177 zones).
Task 4: complete (commits 97dbfdc, review clean)
Task 5 BASE 97dbfdc. Implementer (sonnet) a9601a03a06b0c32d.
Task 5: DONE 318ad8b. Stop rule passed; 3 seeds. Paired NE (3-seed mean, run − baseline): vs r328 m1a_in_box −0.588
[−.620,−.554], m1b_in_box −0.424, m2 −0.032, m1c −0.495; vs old-ne m1a_in_box −0.225 [−.257,−.194], m1b_in_box
−0.272 [−.291,−.245], m2 −0.005 [−.023,+.010], m1c +0.027 [+.017,+.037]. Noise: m1a_in_box .018, m1b_in_box .083,
m1c .021, m2 .014. 1990-96 NE: m1b_in_box .53-.60, m1c .04-.07, m2 .96-.98.
Controller looked at control-ne-s0/panel.png: lines through labels grey (rm2b, RH-2, RC-3) = the fix works; BUT the
big vertical "RM-2" (rm2a) is grey (missed) and parts of "UNION NC" grey; the dashed pier-head line is red (new
overcall).
Ruling: the gate gains m2_clean_words = all ink of Terra words with no line_px within 3 px (the clean-word definition
Task 6 uses), no line rule inside the word; it guards the decision (Task 7: must not fall beyond noise) — M2's base
leaves out ink near run-rule lines, so it cannot see long letter strokes missed (the Task 4 reviewer's warning) —
cost if wrong: a measure that only adds information. Baselines and control runs re-gated with it. Dispatched as
"Task 2b" before Task 6. Task 2b agent ac193afad7664078f.
Task 5 review (sonnet a16eca806492d0280): spec ✅, Approved, no Critical/Important; paired numbers reproduced to the
digit; NE hold-out and prediction mask = gate's quarter_of.
Task 5: minor (deferred): real score() never run in a test (stubbed); a tiny ExtraGeo synthetic test would be cheap.
Task 5: minor (deferred): the test doesn't assert run.json's device/minutes/git_status_src.
Task 5: minor (deferred): run.json is written after score(); a scoring failure leaves a trained run without it.
Task 5: complete (commits 318ad8b, review clean)
Task 2b: DONE 102f422 (linegate.clean_word_mask + m2_clean_words; re-gated baselines and control seeds; other
measures asserted unchanged). Paired NE m2_clean_words (3 seeds): vs old-ne −0.0266 [−0.0401, +0.0025]; vs r328
−0.0545 [−0.0806, +0.0006]; seed noise 0.0082. Values: control .927/.935/.928, old-ne .957, r328 .995. Clean
words: street 126/2634, block 2657/4630, code 559/2150, other 290/622. The vertical RM-2 (p02) is NOT clean -> only
the eye sees it.
Task 6 BASE 102f422. Implementer (opus) a49e817fa18b60c39. Task 2b reviewer (sonnet) a23fc6dff710f5281.
Task 2b review: spec ✅, Approved, 2 Important: (1) "other measures unchanged" claim covers 5 measures' px/called_05
only; (2) the over-run test doesn't isolate the piece test. Ruling: (1) correct the claim's wording, no re-gate
(the change is additive) — cost if wrong: an undetected change in an untouched measure, unlikely; (2) fix the test.
-> fix round 1 (resume ac193afad7664078f; test file only).
Task 2b: fix round 1/5 (2 addressed, 0 open; commits 102f422..891ce2e). Re-review clean.
Task 2b: complete (commits 102f422..891ce2e, review clean)
Task 6: DONE_WITH_CONCERNS 8518d6c (composite.py, scripts/textlines/composites.py, train.py, variant.py V1, tests).
1527 passed; composites=0 batches match the pinned hash. P: NE 0.3839, SW 0.4001 (ink-only crossing def: 0.340).
Draw 33.2 ms at P vs 27.3 at 0 (< 95 ms GPU step). Concerns: 81% of 2000 clean words are block numbers (codes
13%, streets 1%); crossing def gives streets 0.63 vs red team 0.47; box paper unknown in composites; per-map rates;
line pieces cut on a 256-px grid; the 50/50 (a)/(b) split unmeasured. Controller looked at composites.png: exact
red masks on real words from many maps; real line pieces across clean words; held-out areas pale. Reviewer (opus)
a621e5428fdcc3779.
Task 6 review (opus): spec ✅, Approved; leakage, masks, determinism, closure, timing clean. Important: (1) composites
do NOT restore long strokes of real codes (no clean code has unknown px; the run rule flags code strokes only when
connected to crossing lines) — what they teach: letter px touching/under lines are text ((a)-on, (b)); (2) street
names ~17x under their share on modern maps (2000: 29.9% of words, 1.7% of clean words), though the most crossed.
CORRECTION to my earlier note: composites "restore long strokes" was wrong for real maps; Task 7 expectations follow.
Ruling: fix round 1 — (2) per-series kind weights: (a) share(all words)/share(clean words) per kind; (b) the crossed
words' kind mix; repeats per word reported — measured, no new constants — cost if wrong: few street names repeated
often (28/41/21 on modern maps); (1) a per-composite tally of text&known&line px (the lessons the control lacked),
reported; crossing rate switched to the ink-only definition (P NE 0.3396) — cost if wrong: none.
Task 6 fix 1: 4aceaa3. P NE 0.3396 (SW 0.3507). Kind mix all maps code .33->.18, block .30->.14, street .23->.63;
2000 code .12->.16, block .85->.45, street .00->.36. Repeats: 2000 street ~24 draws per word in a full run (max ~50).
New lessons: 68% of composites, 17.8% of word ink (2000 codes 26.7%). 1530 passed; composites=0 pinned unchanged.
Re-review (sonnet) a93a9cb521fe3da54.
Task 6: fix round 1/5 (3 addressed, 0 open; commits 8518d6c..4aceaa3). Re-review clean; controller checked _finish
does not read page.line (targets unaffected by the grown line).
Task 6: complete (commits 8518d6c..4aceaa3, review clean)
Task 7 BASE 4aceaa3. Ruling: the verdict also requires m2_clean_words not lower beyond its noise (Task 2b's guard).
Task 7: implementer (opus) a0cffcdef58baedbe; it stops at pending-eye with the panel paths; the controller looks
and replies eye_ok + notes; then SW.
Task 7 NE (f5bc3aa decide.py, 9 tests, 1539 passed): variant − control: m1a_in_box +0.0063 [−.0060, +.0184];
m1b_in_box −0.0042 [−.0119, +.0018]; m2 +0.0017; m1c −0.0094; m2_clean_words +0.0005 -> verdict `control` (the
variant gains on neither line measure). Adoptable as written: neither (both fail m2 vs r328; control fails m1c vs
old-ne +0.027 vs noise 0.021; both fail m2_clean_words vs old-ne ~−0.027).
CORRECTION (controller error): the big vertical RM-2 (rm2a) is RED in control-ne-s0's panel and all six runs; my
Task 5 note "grey" misread a downscaled picture. The m2_clean_words ruling still measures a real cost (−0.027).
Controller eye (compare-ne.png): lines through labels grey in both arms, letters red; dashed pier line red in all six
runs (new over-call; r328/old-ne grey); control s0 parts of UNION grey; variant no visible difference. eye_ok True,
verdict control. Noted post hoc: the adoptable rule's "M2 vs r328" can never pass (r328 calls all ink text, M2
0.9994): uninformative; against old-ne alone both arms still fail one guard narrowly. Not adoptable -> Task 8 skipped.
SW: control s0 running, then the note.
Task 7: DONE d489e38 (decide.py final + docs/research/joint-solver/notes/2026-11-17-text-vs-line-results.md).
SW control s0 vs old-sw: m1a_in_box −0.319 [−.384, −.255], m1b_in_box −0.275 [−.297, −.214]; m2 −0.0142 (noise
.0136), m2_clean_words −0.072 (noise .049), m1c +0.001. Not adoptable (vs r328 also fails m2, m2_clean_words, audit
3/64 vs 0/64). Post hoc: code letters away from lines drop −0.033 NE / −0.018 SW; SW street names −0.048; the
clean-word guard is dominated by a few big "other" words. Reviewer (sonnet) ae09f9ce3ffd9eb57.
Task 7 review: spec ✅, Approved; four paired numbers reproduced.
Task 7: minor (deferred -> final fix wave): the note says SW control "not adoptable" on one seed; word as "does not pass".
Task 7: minor (deferred -> final fix wave): the note states a cause for the dashed over-call without direct evidence; soften.
Task 7: minor (deferred -> final fix wave): "miss about 3x as many such pieces" is from one person's audit count; say so.
Task 7: complete (commits f5bc3aa..d489e38, review clean)
Final review (opus a7979f5fa42e2f7f5): needs fixes (note + handoff); code paths sound, defaults byte-identical,
closure unchanged, fresh-clone clean. Important: (1) control vs old-ne mixes recipe+zones+in-box rule: trial shows >
half of m1c's +0.027 is the recipe's, m1a_in_box gain is not; (2) the dashed-line cause doesn't fit (0 box/zone px in
that window; dashes grey in all-sheets and trial too); (3) "3x" holds on NE only (SW 1.8x); (4) "both recipes fail
on SW" false (only control ran SW); "cannot pass" -> "did not pass, unlikely to"; (5) SW one-seed wording; (6) the
macOS pin never ran on CI; (7) six stale stages not in the handoff. Ruling: ONE fix wave — the note (1-5 + the
minors on its wording), linegate.py docstring, and the pin keyed to Darwin arm64 (skip elsewhere; the in-process
boxes-vs-lines test covers the invariant on every platform) — cost if wrong: the pin's check missing on Linux.
Final fix wave: a2b1dc7 (note items 1-7, linegate docstring, pin keyed to Darwin arm64 + a new all-platform
test_composites_zero_draws_what_no_bank_draws). Re-review (a961e5afcaa1518b5): all addressed.
Final: parked — the exact "composites=0 equals pre-composite draws" pin runs only on Darwin arm64; the new test
compares the current code with itself — Ruling: acceptable; the owner's machine is macOS arm64, and on Linux
GUARD_LOSSES would shift if draw consumed an extra rng value — cost if wrong: a Linux-only regression unseen in CI.
FINAL REVIEW COMPLETE. Morning report published (scratchpad morning/letters-not-lines.html).
Task 8: SKIPPED — no adoptable recipe by the rule written in advance (plan: "Otherwise skip, and say why").
Task 6: minor (deferred): on/off redraw bias untallied (words_failed not split on/off).
Task 6: minor (deferred): line pieces cut on a 256-px grid end abruptly (cut from >= 533-px windows instead).
Task 6: minor (deferred): A_SHARE 0.5 unmeasured; realised crossed share ~0.72 vs 0.34-0.38 real; first ablation if mixed.
Task 6: minor (deferred): texture check compares unlike sets (known vs all text ink).
Task 6: minor (deferred): PINNED_BATCHES_SEED3 computed on macOS arm64; CI is ubuntu x86_64 — watch CI.
Task 6: minor (deferred): variant.run(bank=...) doesn't check the bank's areas.
Task 6: minor (deferred): stray kind/angle on non-text px (masked by the loss; harmless).
Task 2b: minor (deferred): no assertion on tallies/blocks.npz or a paired call in the tests.
Task 2b: minor (deferred): a word with no ink counts as clean (inflates the clean/all counts).
Task 2b: minor (deferred): a muddled test comment (~275); a docstring names "Task 6".
Task 2b: minor (deferred): extra/ folders of the control runs not re-gated (no m2_clean_words there).
Task 2: minor (deferred): Task 3 should gate all sources in one process with one CityGeo (7.7 GB peak; 4 at once ~28 GB).
Task 2: minor (deferred): no skip-able test of CityGeo._rasters' block/weight lookup at sampled ink px.
Task 2: minor (deferred): LEARNED_RULES read without a linerules hash key.
Task 2: minor (deferred): grid_px repeats City.px.
Note for Task 2's gate: the run rule's learned-font glyph false-line reaches 8.9-11.4% on some texts (RM-4); the gate
prints rule accuracy beside every number, as designed.
