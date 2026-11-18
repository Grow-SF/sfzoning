# SDD ledger — plan: docs/superpowers/plans/2026-11-16-text-reader-labels-and-publish.md

Started 2026-11-16 ~21:05. Branch `solver`, worktree saved-reads. Plan base: ee2a59e.
No spec file: the plan argues from the owner's decisions (in the plan's "The owner's decisions" section and in
memory decisions-log). Rulings made without a spec are provisional.

Owner, at the go (21:00): "Ok go on the plan. For release, we will regenerate all maps when we're ready."
Owner (21:10): the heights spike runs first, then T0 ("Both, spike first"). Spike = a separate agent, not a plan task;
its script goes in pipeline/scripts/spikes/heights_spike.py.

## Pre-flight scan

### Pairs sharing a file or an interface
| Tasks | Produces → consumes | Found |
|---|---|---|
| T0 × T4 | T0: the solve's own labels location (read by trust.load_prints, labels.py, streetside.py, placement.py) → T4 writes new prints/readings there | Consistent. T0 must leave room for a new producer without moving the scoring set (ruling P1). |
| T0 × T1, T4 | T0: criteria_check without the energy criterion across sources → T1, T4 comparisons | Consistent. |
| T0 × T5 | T0's fixed scoring set → published_compare (T5's measure) | Gap: published_compare also scores "labels right" and must read the same fixed set (ruling P2). |
| T1 × T2 | both read the per-quarter chance from solver-text-reader (data/solver/text-reader/<sheet>/chance.png) | Consistent. T1 adds the source switch in textgrid; T2 reuses it. |
| T1 × T5 | a walls change moves walls_digest → placement refits → the solve changes → T5 reads the solve | Sequential; fine. |
| T2 × T3 | T2: boxes (corners, axis mod 180°, quarter) → T3 reads both ways along the axis, with the 0/180 classifier | Consistent. |
| T2 × T4 | T2: x, y, lab_lot, street, kind probs, mean chance → T4: prints and calibration features | Consistent (T4's "detector p(zoning code)" comes from T2's kind probabilities). |
| T3 × T4 | T3: posterior over printed forms + other + none → T4 features (closed mass, margin, p(other), p(none)) | Consistent. |
| T4 × T5 | T4 keeps `names` (canon spellings) → T5 maps spellings to the published forms | Consistent. |

### Each task against itself
| Task | Found |
|---|---|
| T0 | Consistent: no behaviour change, so a byte-identical report.json is the right measure. |
| T1 | Two changes in one task (`text_weighted`; the per-quarter chance). The measure takes log loss, then the six measures. Ruling P3: report the log loss for each change separately if cheap. |
| T2 | Consistent. The level is a grouping threshold carried from S3 (accepted method). |
| T3 | The synthetic inputs (font_real.npz, prior3.json, labs.pkl) are made by no stage; the plan says "made by a stage". Ruling P4. |
| T4 | "Real mass" (boxes → prints; codes written per label) is a cutoff unless defined. Ruling P5. |
| T5 | The owner's release note changes the "First" step. Ruling P6. |

### Rulings (pre-flight)
- P1. Ruling: T0 adds one path constant for the solve's labels, which for now points at today's chain outputs; the scoring set keeps reading today's chain explicitly. T4 later points the constant at the new stage. — Why: the smallest change that gives a byte-identical solve and separates the two sources. — Cost if wrong: a rename in T4.
- P2. Ruling: T0 also points published_compare's "labels right" at the fixed scoring set. — Why: one measure when comparing. — Cost if wrong: the published map's score drifts once T4 lands.
- P3. Ruling: T1 reports the walls' held-out log loss for `text_weighted` on today's chance and on the per-quarter chance, if each walls fit is minutes and not hours; else the final combination only. — Why: two changes, and each should be visible. — Cost if wrong: one extra walls fit.
- P4. Ruling: a stage builds the template reader's synthetic inputs (font, code forms, prior) from the training crops outside the owner's answered blocks: one font, not one per quarter. — Why: the lettering is the same across the map, so per-quarter fonts leak only glyph shape, never a label; the answers stay clean. — Cost if wrong: held-out per-quarter reader numbers slightly optimistic.
- P5. Ruling: "real mass" is a numerical truncation fixed before measuring. A box becomes a print when its closed-list mass is ≥ 0.05; a code is written as a reading when its posterior is ≥ 0.01; the rest folds into the leftover mass. T4 reports how many answered labels' boxes each truncation drops, and lowers the level if it drops more than 1% of them. — Why: the owner's rule is probabilities, not cutoffs; p(other) and p(none) carry the not-a-label evidence into the calibration, and the truncation only bounds compute (placement and far_side scale with the label count). — Cost if wrong: a few labels lost, measurable.
- P6. Ruling: T5 builds the period-2000-digital wiring (codes from the solve, lot map, duplicate IDs, spellings) with tests, and the page of differences, at $0. The release rebuild waits for the owner: the 16 stale district stages, `sfzmaps ship`, heights and the 1996-2000 twin. If `sfzmaps build period-2000-digital` would rebuild stale upstream stages or want paid reads, T5 makes the new 2000 map for the differences page by running the stage's code into an experiment folder, and leaves the release stages' lock entries alone. — Why: the owner, "For release, we will regenerate all maps when we're ready", plus no spending. — Cost if wrong: the wiring is first exercised through `sfzmaps build` at release.

## Heights spike (not a plan task)
Done d9e3987. The detector + S3 boxes find 218/221 saved height labels; reader C reads 2/221 right (8% with a
height grammar). It never saw "NN-X"/"OS"; it reads "OS" upside down as "SO". Needs height forms in the synthetic
crops + direction by word list. Page sent to the owner. Outputs: pipeline/data/solver/experiments/spike-heights/.

## Tasks
T0 BASE d9e3987. Implementer dispatched (sonnet), agent acfe4e422e772a287.
T0: implementer DONE_WITH_CONCERNS, commits a19ca2d, 01bbd83. Report task-0-report.md. Real rebuild of
solver-labels/walls/solve: report.json byte-identical, 31 files unchanged. Concerns (observations): two solve
constants (SOLVE_PRINTS_FILE, SOLVE_LABELS_DIR) in runs.py, not paths.py (paths.py is a dep of every solver stage);
streetside still reads/writes readings in LABELS_DIR (carry to T4); the block recount covers labels only.
T0: reviewer dispatched (sonnet), agent a36b252dfebf15512, package review-d9e3987..01bbd83.diff.
T0 review: spec ✅, quality Approved, 1 Important: check_block_split recounts from the same pb arrays per_block used
(self-referential in criteria_check/published_compare); tie it to an independent per-label source. -> fix round 1.
T0: minor (deferred): runs.py docstring first sentence ~150 chars, rewrap.
T0: minor (deferred): check_block_split takes 6 positional args; the two callers repeat the pb[...] unpacking.
T0: minor (deferred): labels_right ratio assert divides by split["zoning"].sum() (ZeroDivisionError on an empty set).
T0: fix round 1/5 (1 addressed, 0 open; commits 54e0d08..adcfbfa). Re-review: addressed; out-of-scope: per_block and
score still duplicate the formula (the check guards drift); incidents/wrong lots keep the totals check only.
T0: minor (deferred): per_block and measures.score duplicate the label-right formula; the check guards drift only.
Task 0: complete (commits a19ca2d..adcfbfa, review clean)
T1 BASE adcfbfa. Implementer dispatched (sonnet), agent a7e1b1e6618f1e069.
Ruling: T1 is kept even if worse beyond noise, with its worst changes reported — the no-LLM rule needs it, and
"better is enough" means no chasing — cost if wrong: a worse wall input carried into T4 (T4's measure catches it).
Ruling: baseline for T1-T4 comparisons = copy of today's final at data/solver/experiments/text-reader-plan/final-before/
— the plan compares against today's final, and the stage overwrites it — cost if wrong: none (a copy).
T1: implementer DONE_WITH_CONCERNS, commits 700f40c, 7b61feb, 1e6b64d. solver-text-reader stale on paper only ->
dvc commit --force. Walls log loss nw/ne/sw/se: (a) .0978/.1027/.0452/.0611; (b) text_weighted on solver-text
.0982/.1026/.0452/.0612; (c) on per-quarter chance .1011/.1069/.0487/.0658. Six measures (c) vs final-before:
labels right 94.79 -> 94.05 [-1.43,-0.11]; incidents 862 -> 921 [+31,+89]; wrong lots 4,543 -> 7,001; lines
drawn 96.43 -> 95.69; table lines 96.65 -> 96.13; holes 48 -> 46. 3 patches (sw, se, ne; ~1,800 lots) lose a table
border. Added criteria_check --baseline. Brought (b) vs (c) to the owner before review (changes an approved method).
Owner (22:35): no preference on (b)/(c); pushed d9e3987 to origin/solver (#62), on the owner's "Push now".
T1: Ruling: the walls read `text_weighted` on the single solver-text detector's chance (b), not the per-quarter
chance (c) — (b) is Terra-free at inference (Terra only trained the detector), ties today's walls' log loss, and (c)
cost +2,458 wrong lots, likely through the reader detectors' known over-calling of ink as text; the per-quarter path
stays as a non-default option — cost if wrong: the 2000 scores are optimistic for an unseen map (the published 2000
map itself is not hurt). Applies to T2's boxes too (same chance), for one source. T1 fix round 1 = switch to (b),
rebuild, measure.
WITHDRAWN 22:37: the owner: "that wasn't no preference. I just didn't have an answer yet." Implementer stopped
before any change (tree clean, head 1e6b64d). The walls choice (b)/(c)/retrain WAITS for the owner. T1 stands at (c).
Owner's follow-on (22:45): wrong lots jumping is not a verdict ("small errors can result in huge shifts in the
number of wrong lots") => keep (c), the plan as written. T2-T4 use the per-quarter chance too. Next: T1 review.
T1 review: spec ✅, quality Approved, no Critical/Important. ⚠️ dvc commit --force solver-text-reader evidence
circumstantial -> controller checked: uv.lock diff ce828df..HEAD changes only +absl-py +immutabledict +ortools
-pymaxflow (no torch/numpy change); pyproject same swap. Resolved: on paper only.
T1: minor (deferred): the textgrid test checks file choice, not the quarter split (that lives in reader_stage.page_quarters, tested in R1).
T1: minor (deferred): criteria_check --baseline still draws the stage's own final in pictures; only help text warns (--no-pictures).
T1: minor (deferred): test_build.py asserts paths.SHEETS twice.
T1: note: scratch copies experiments/t1-walls, t1-before not yet deleted (data; harmless).
Task 1: complete (commits 700f40c..1e6b64d, review clean)
T2 BASE 1e6b64d.
Ruling: T2 records each box's ink-trimmed extent beside the grouped box — the heights spike found the detector's
boxes run wider than the ink (glyphs reach the reader at ~2/3 trained size; trimming took block numbers 4% -> 28%)
— cost if wrong: two extra columns.
Ruling: T2's level is chosen by Terra code-word recall on the other quarters (the solve-measure route needs T3/T4,
so it is not cheap now) — cost if wrong: a level re-chosen in T4.
T2: implementer dispatched (sonnet), agent afd83dc1236c12db3.
T2: DONE_WITH_CONCERNS, commits 5e5b0c5, 6d00181. Levels nw .995 ne .99 sw .995 se .995 (per-quarter chance
saturated near 1.0; S3's 0.5-0.9 found 23-53%). Recall at IoU 0.3 (axis within 10°): nw 73.5 (55.2), ne 46.8 (30.5),
sw 78.8 (53.2), se 69.0 (44.1). Answered: 418/668 with a Terra box met at IoU 0.3; 641/731 have a box holding
their point. 8,794 boxes. Misses mostly merges/splits (72-91% touched at IoU 0.1). Likely cause (implementer): the
unknown disc around Sol points when solver-text-reader trained. cli.py edit re-recorded 6 warp stages (outputs same).
T2: reviewer dispatched (sonnet), agent a180d4015b0e5b7c0.
T2 review: spec ✅, Approved, no Critical/Important. ⚠️ outputs vs lock -> controller: dvc status solver-boxes up
to date; summary.json matches the report.
T2: minor (deferred): tests never run sheet_boxes (placement via _place, Terra word quarter wq untested).
T2: minor (deferred): no test pins the uint8 threshold at 0.995 / 1.0.
T2: minor (deferred): answered_with_box hard-codes a 400 px window (undocumented).
T2: minor (deferred): cli.py calls _in_one_thread before logging.basicConfig (harmless).
Task 2: complete (commits 5e5b0c5..6d00181, review clean)
T3 BASE 6d00181.
Ruling: T3 crops our boxes at the ink-trimmed extent (T2's t* corners); the Terra-box comparison uses the spike's
crop recipe — the heights spike showed loose crops shrink glyphs — cost if wrong: one crop choice to swap.
Ruling: T3's posterior over direction is a mixture of the two readings weighted by the 0/180 classifier's
probability, not the max of the two — prior x likelihood, the owner's rule — cost if wrong: none measurable lost;
T4's calibration sees the mixture.
T3: implementer dispatched (opus), agent a98872a5bcddaf07c.
Owner (~23:20): "Did you reach for 'likely cause' without validating your hunch?" — yes; the unknown-disc cause was
the T2 implementer's guess. Retrain recommendation withdrawn. Owner: "Do the cheap test and report back. We will need
to plan again when you have results." Diagnostic agent a00e5f1a6dff6be59 (opus, read-only, CPU) -> results in
data/solver/experiments/detector-overcall/results.md, script scripts/spikes/detector_overcall.py.
HOLD: T4 does not start until the owner re-plans with those results. T3 continues (the reader trains on Terra-box
crops, independent of the held-out detectors).
Diagnostic done: ba36061 (script), results data/solver/experiments/detector-overcall/results.md. The only recipe
difference: unknown disc radius 328 px (held-out) vs 30 px (all-sheets, the recipe trial, the old quarter models);
328 = the single longest Terra "zoning code" word ("SH", 1968 sheet p18); code words: median 85, p99 169 px.
Non-text line ink called text (>=0.5): held-out 31.2% inside discs, 1.2% ring, 0.4% far; in-sample r328 30.6%;
r30 models 0.1-0.3%. 95% of ink outside Terra words / 98% of district-weight line ink lies inside 328-px discs =
never trained either way (7% at 30 px). Negatives seen: 209-243 M (held-out) vs 749 M (all-sheets). Open: no
radius-only run; which radius is safe; whether it explains all box merges. Controller checked the tables, held.json
and picture 0: consistent. Sent picture 0 to the owner. Awaiting the re-plan.
Owner (~00:10): the detector "learned the location of a label, not the label text"; it flags walls, block lines,
street lines behind the text. Wants: synthetic training with learned letter shapes + drawn map-like lines so the
detector learns text px vs line px; literature; AUTONOMOUS brainstorm (no question rounds); and why the mistake got
in. Brainstorm running: literature (a3e179114bfc1c016 -> docs/research/joint-solver/notes/2026-11-16-text-graphics-
separation-literature.md), root cause (a18ebb48230a538ca -> .../2026-11-16-unknown-disc-root-cause.md), recon
(a16314790b05e848f). Then a design doc in docs/superpowers/specs/, builder check, to the owner. Nothing built
before approval. T3 continues; T4/T5 on hold.
Recon back: font = 31 code chars only (no 0 J Q X Z, no lowercase), p(ink) templates + noise model
(codereader/glyphs.py Font; textread/synth.py renders word crops with real line-window overlays). strokes.npz:
widths (wmed p50 16 ft = 4 px; district 24-32 ft), lengths; NO dash/spacing/angle measures; no text-free region
finder (but textread NONTEXT windows / Overlays rows reusable). Seam: textnet/train.py draw() l.189 (no hook; same
rng). One detector = ~6-7 min at 4,000 steps (the 4 held-out ~25-30 min), NOT 85 min. reader_stage scores.json
ink_outside_boxes_called_text nw 0.0237 (masked measure?).
Literature back (uncommitted file docs/research/joint-solver/notes/2026-11-16-text-graphics-separation-literature.md).
Controller verified a 2nd cause: textnet/data.py:469 text = (word > 0) — the whole Terra word rect is "text", so
line ink crossing a word is taught as text (both detectors). Waiting on root cause, then the design.
Owner (~00:30, crop of "RM-2" with block lines red): the mistake to understand is NOT how 328 got in, but "not
realizing the background ink isn't text". Root-cause agent redirected to the box-derived text target: the decision,
why measures (scores.py text ink found / outside boxes / long lines) hid it, pictures and reviews, earlier warnings;
file renamed ...-text-target-root-cause.md. The 328 disc becomes a secondary section.
Owner to bed (~00:35) with an overnight GO: finish debugging, check it, autonomous brainstorm, red team, plan, red
team, implement (see memory decisions-log). Root cause done (uncommitted note ...-text-target-root-cause.md); fact-
check agent a79306a28f5414c1e. Design written: docs/superpowers/specs/2026-11-17-text-detector-text-vs-line-design.md
(uncommitted). Builder check a3d5a6c372838b154, red team ace16ee53ecf32f5d (-> .superpowers/sdd/redteam-text-vs-line-
design.md). New plan gets its own SDD ledger.
00:10 design corrected after builder check + red team (be37735; red team findings + outputs copied to
data/solver/experiments/redteam-text-vs-line/, scripts to scripts/spikes/redteam_text_vs_line/). Plan written:
docs/superpowers/plans/2026-11-17-text-detector-text-vs-line.md (uncommitted); plan red team aae48b3373e9646f2 ->
.superpowers/sdd/redteam-text-vs-line-plan.md. Then SDD with its own ledger.
T3: implementer DONE_WITH_CONCERNS. Commits 90b1069, 11de9e5, e48d3b7, 92c8215, c070d80 (dvc commit --force for
crops/train after HEAD moved under them: the controller's commits; deps diff empty). Closed top-1 on owner's answers:
at Terra's boxes 0.962 [0.947, 0.976] (662); at our boxes 0.464 [0.425, 0.503] on 633 found (86 missed), 0.412
counting misses. Box error dominates (merged-with-lines boxes read 0.185). Choices to review: 3 stages; a direction
prior by axis (5° kernel, unfitted); C's synthetic faults kept; temperature 1. Agent stopped after its report.
T3: reviewer dispatched (opus), agent aa17009d78192de9e, package review-6d00181..c070d80.diff (skip ba36061, dc32995,
be37735: controller's).
T3 review (opus): spec ✅ except direction; Needs fixes. Important: (1) ctc.py:117 wraps axes >= 179.95° to the
axis-0 prior (23 boxes, 2 answered); (2) reading maths (ctc.py) is in solver-text-train's closure, so any read fix
forces a 77-min retrain; (3) the axis prior (5° kernel) was not asked for; classifier alone scores 0.4622 vs 0.4638.
Ruling: T3 uses the 0/180 classifier alone (the pre-flight ruling's method; the measured difference is one label)
and the axis prior is removed — fixes (1) by removal — cost if wrong: one label; a fitted prior can come later.
Ruling: the reading maths moves out of solver-text-train's closure; solver-text-train is recorded on paper (R8: the
training code path unchanged, moves only, closure test updated) with dvc commit --force, no retrain; solver-text-read
rebuilt (~2 min) — cost if wrong: a 77-min retrain later.
T3: minor (deferred): read.py NEAR = 400 hard window (derive from boxes' reach).
T3: minor (deferred): five answers (NOE VALLEY, NC, 3RD, CMNC, MB-OS) dropped without labels.code_of/ALIAS mapping.
T3: minor (deferred): build_train(quarters=subset) writes a partial set into the stage folder (trap for trials).
T3: minor (deferred): crops.py:303 slanted-word height floors are map-wide (include own quarter); not in leakage notes.
T3: minor (deferred): recording evidence should also show git status --porcelain of dep paths.
T3: minor (deferred): crops.py per-sheet seed from the sheet name's last 8 bytes, not the series.
T3 fix round 1: commits 7509ac7, a6fce98, d40baae, 2f01c4b, 8157a97. Classifier alone; axis prior removed;
ctc.py out of the train closure (best_path to model.py); solver-text-train and solver-text-crops re-recorded on
paper (dvc commit --force; crops' build was started by mistake and stopped, its output untouched); solver-text-read
rebuilt. Closed top-1: Terra's boxes 0.961 [0.944, 0.974]; ours 0.462 [0.424, 0.501] (633 found, 86 missed).
Full suite 1467 passed. Re-review (sonnet) a2bc5f055a38764a2, package review-c070d80..8157a97.diff.
T3: fix round 1/5 (3 addressed, 0 open; commits 7509ac7..8157a97). Re-review clean; controller: dvc status of
solver-text-crops/train/read up to date.
Task 3: complete (commits 90b1069..8157a97, review clean after 1 fix round)
T4, T5: ON HOLD for the owner (box error dominates; the detector plan 2026-11-17 runs first).
T3 -> owner/T4: box error dominates (0.464 at our boxes vs 0.962 at Terra's): decided by the detector plan.
Naming (owner asked): "the held-out detectors" (solver-text-reader, 4 quarter models) and "the all-sheets detector"
(solver-text). Don't say "per-area".
Owner question raised (not blocking T3): retrain the per-quarter detectors with solver-text's recipe? Same root
likely under T1's 3 missed borders and T2's merges. Waits for the owner's words.
T0 -> T1 carry: `solver-text-reader` shows stale in dvc status (pyproject/uv.lock, data/warp, data/work), predating
this plan. T1 reads its chance: find why before using it (R8 if on paper only).
T0 -> T4 carry: trust.build_readings uses load_prints() (follows SOLVE_PRINTS_FILE).
T0 -> T4 carry: streetside.build_streetsides still reads/writes paths.LABELS_DIR (streetside.py:153-156); switching
the solve constants won't redirect it. placement.py:94,99 loads strokes.npz/lotsize.npz from LABELS_DIR (not labels; fine).
