# Red team: "Teaching the text detector letters from line ink" (spec 2026-11-17)

Target: `docs/superpowers/specs/2026-11-17-text-detector-text-vs-line-design.md` ("the design"; line numbers refer to it).
Read-only. CPU, one thread. Every measurement below was run for this review; the scripts and outputs are listed at the
end, in the scratchpad.

**Verdict.** The direction is sound: exact-label synthetic data, corrected real labels and measures independent of the
labeller. The literature backs it. But as written, the gate cannot see the failure it is built for:
- M1's line rule misses most line ink where letters touch lines.
- M2 counts that ink as text.
- So the keep rule would **reject a detector that fixes RM-2** (F1).
- And the synthetic glyphs as rendered today can be told from real ink by texture alone, so M3 can pass while the
  real map does not change (F3).

Fix the measures before any training run.

---

## F1. CRITICAL: M1 cannot see the RM-2 case, M2 rewards it, and the keep rule would reject the correct fix

**Failure.**
- M1's line ink is skeleton strokes ≥ 200 ft (design l.128). `strokes.py` cuts a stroke at every junction
  (`strokes.py:3-8`), and a letter touching a line is a junction.
- So a line running through or along touching letters becomes short strokes, and never enters M1.
- The clause "continue at least 10 px beyond every box they cross" (l.128) removes still more: a piece that ends at a
  letter never leaves the box.
- That ink then sits in M2's base, "ink inside Terra word boxes that is NOT M1 line ink" (l.133), where calling it
  text counts as right. This is the root cause again (box ink = text), now inside the gate.
- The repo already documents this: `codereader/labels.py:41-44` says "where a letter touches a line, much line ink
  near a label is missed".

**Evidence.**
- **Exact-truth synthetic cases.** Learned font; RM-2, RH-1(D), NC-3, C-3-O, RM-4 and P at caps of 20 and 27 px, at
  3.98 ft/px. Share of the in-box line ink each rule calls line:

  | case | stroke rule (M1) | run rule (`labels.run_lines`) |
  |---|---|---|
  | 6-px line through the middle | **27%** (12-65) | 100% |
  | 4-px line along the baseline | 44% (17-98) | 100% |
  | block corner (L of 4-px lines touching) | **26%** (11-59) | 100% |
  | two street lines, word between | 75% (38-99) | 100% |
  | vertical line through the 2nd letter | 82% | 100% |

  In `synth-all.png`, blue is the in-box ink only the run rule catches: exactly the line ink touching the letters.
- **Real, all 13 sheets of the 2000 map.**
  - The in-box ink on a straight 200-ft run (run rule) is 1.10 M px. **81% of it is not on a stroke-rule long line.**
  - Per word holding such ink, the stroke rule catches less than half of it in 78% of code words, 76% of block
    numbers and 56% of street names.
  - In `rm2-zooms.png`, the heavy line crossing the R of "RM-2" is blue (run rule only).
- **What today's detectors do with that ink.** The r30 detectors (all-sheets; the recipe trial) call **85-87%** of it
  text.
  - It is **19.5% of M2's base**.
  - A detector that gets it right therefore loses up to ~17 points of M2 (a little less: some run ink is
    glyph-collinear, see F2).
  - The seed-to-seed spread of the stage recipe is about 1 point (words found 0.940-0.949 over 3 seeds at 4,000 steps,
    0.936-0.950 at 2,000; `experiments/text-half_size-...-seed=*/scores.json`).
  - The keep rule (l.153, "no loss beyond noise on M2") rejects it.
- **The reverse also passes.** M1 sees only the in-box ink on both rules (211 k px, called text 61%). A detector that
  stops calling only unbroken crossing lines text scores an M1 win while still calling every junction-cut piece text.
  It appears fixed and isn't.
- **The true baseline.** Today's r30 detectors call about **82%** of in-box straight-line ink text. That is the
  number to move, not the 66-72% the root-cause note quotes (see F2).

**Correction.**
1. **M1a is the owner's own measure, restored.** Ink on the zoning table's district borders: `borders.border_paths()`
   lot borders where the table's codes differ, carried to page px by the placement.
   - R&D's `evaluate_detector.py` (46f8ec1) did exactly this, "away from any box"; drop that exclusion.
   - Model ink-to-border distance from the placement's measured residuals (a likelihood, not a band).
   - It depends on no ink rule and no Terra box. It also covers the stair-step district lines that no ink rule catches
     (F2).
2. **M1b is the run rule** (junctions don't cut a run), plus a chain rule for bent lines.
3. **Validate every M1 rule on the synthetic cases before using it.** Print its line-ink recall and glyph-ink
   false-line rate (the table above) next to every M1 number.
4. **M2's base:** in-box ink on no M1 rule and not within 1 px of one.
   - Report the overlap/adjacent band on its own, never in M2.
   - The keep rule reads M2 only on that clean base.
5. **Redo §6's baselines** with these rules.

## F2. HIGH: the stroke rule also calls glyph ink "line", and misses district lines; the quoted baselines are mostly glyph ink

**Failure.**
- 27-px code letters have skeletons over 50 px between junctions ("2", "S", "C"). So the stroke rule calls them line.
- M1 would then penalise correct text calls wherever its continuation clause doesn't drop them: missed text outside
  every box, and letters that overhang Terra's box (cap/word height up to 1.25 for a tenth of words,
  `glyphs.py` RATIOS comment).
- Meanwhile the stair-stepped district lines (lot jogs under 50 px) are on neither rule.

**Evidence.**
- **Code words with no run-rule line ink within 3 px** (718 words, glyph ink alone): the stroke rule calls **30.1%** of
  their ink long line, and 55% of them hold some. For "other" text it is 21%, for block numbers 0.1% and for street
  names 1.1%.
- **`rm2-zooms.png`:** the whole "2" of RM-2 is red (stroke rule only).
- **In-box ink that is stroke-long but not on a run** (683 k px, 76% of all in-box stroke-long ink): the r30
  detectors call it text at 96%, the same rate as the rest of the glyph ink (97.9%). So it is glyph ink.
  - The root-cause note's "16.4% of in-box ink is on long lines" and "66-72% of it called text" are therefore about
    three-quarters glyph strokes.
  - The real failure is worse than quoted (F1: ~82%).
- **Ink more than 10 px from every box:** 5.3% is on neither rule, and the grey stair-steps in
  `rm2-window-rules.png` are district lines. The stroke rule alone misses 30% of this ink.

**Correction.**
- As in F1.
- The "district" split of M1 comes from the table's district borders, not from stroke width.
- Report any stroke-rule number only with its synthetic false-line rate.

## F3. HIGH: synthetic glyphs as rendered today can be told from real ink by texture alone

**Failure.**
- The design renders "the learned font's p(ink) templates". The existing synthetic crops do this by drawing each px as
  ink independently with p(ink | distance) (`textread/synth.py:172`).
- The result is speckled ink that no real stroke has. Every rendered px is text, and every real background px is not.
- A U-Net can learn "speckled = text, solid = not text". M3 (synthetic) would then pass while the real in-box lines
  stay text.
- This is most likely while real labels still say "line in box = text" (old series, failed alignments: F4, F5). The
  texture is then the only way to satisfy both label sources.

**Evidence.** I compared 178 real code words with no line ink (p02, p05, p12) against the same number rendered with
the learned font and its noise (`texture.json`, `iid-rendered.png`):

| statistic | real | i.i.d.-rendered |
|---|---|---|
| ink px with at most 1 ink neighbour | 0.02% | 4.3% |
| pinholes per 1,000 ink px | 1.8 | 102 |
| ink pieces per word | 3.2 | 15.9 |

- The noise model's p(ink) on a stroke is 0.955-0.972, so there are holes everywhere.
- It was fitted on reader crops resampled to 32 px tall (`textread/crops.py:4-7`), not on page px.
- Drawn lines (`cv2.line`) are also too clean next to real jagged district lines.

**Correction.**
1. **Use real glyph ink as the text source.**
   - Terra words whose ink pieces lie wholly inside the box, with no line within 3 px, are exact text masks with real
     texture. On 2000 there are 718 codes, 3,397 block numbers and 604 street names. No font is needed.
   - Composite with OR (1-bit overprint is a union):
     - clean real words onto real line windows;
     - real line pieces, and drawn lines, onto clean real words.
   - Both directions are needed, so that texture predicts neither label.
2. **Where the font must render** (rare codes, missing characters): render solid at p ≥ 0.5, then apply a spatially
   correlated edge degradation fitted on page px. The generator must match the three statistics above before any
   training.
3. **A domain check:** a small real-vs-synthetic classifier on ink patches, or the detector's chance on real against
   synthetic ink of the same class.
4. **M3 is diagnostic only.** The gate is held-out real M1/M2 (corrected) and the pictures.

## F4. HIGH: the alignment labeller (§3) mislabels touching ink, has no accuracy measure, and can quietly bring the blind spot back

**Failure modes.**
1. **The template footprint is not the ink.**
   - The template is an EM mean over about 800 kept labels; real strokes are 3-4 px wide.
   - A 1-px edge disagreement mislabels 25-50% of a stroke's px.
   - Neither the design nor `glyphs.py` measures per-px fit accuracy.
2. **Touching line ink becomes glyph halo.**
   - Only run-rule px are unobserved (`glyphs.py:25`, `RULES = "run"` at `:91`). Non-run line ink touching a glyph
     (stair-steps, bent lines, stubs) is observed.
   - The noise model gives it p(ink) 0.29-0.47 within a px of the edge, so "ink the glyphs explain" claims it.
   - `glyphs.py:34-39` itself records that fits "swell to take in more ink" and go to other ink.
3. **"Poor fit" is too loose to catch px-level misalignment.**
   - After round 0, the "good" position spread of box labels is 0.6-0.9 word heights (`text-crops/summary.json`
     `font_trace`), so a fit shifted by a whole character counts as good.
   - 12-13% of box labels are judged bad. Point labels sit at exactly 0.5 bad in every round, so that mixture is not
     informative.
4. **Hard cases go unknown.**
   - 62% of 2000 code words (1,338/2,150) hold ≥ 3 px of straight-line ink inside the box; so do 47% of street names
     and 20% of block numbers.
   - If alignment fails where lines touch every letter, real data again teaches nothing at the crossings. That is the
     old blind spot, smaller. The only teacher left is synthetic (F3's shortcut).
5. **The boundary band (l.67-68), weighted up on real labels,** amplifies exactly the px the labeller is least sure
   of.

**Correction.**
- **The labeller's own gate, before its labels are used:** run it on F3's composites (exact truth). Report per-px
  precision and recall of "text", and the unknown rate, by case: line through, along, touching, corner, heavy.
- **Label actual ink, not the template.**
  - Assign each ink px to its stroke (the `strokes`/`_stroke_ink` assignment).
  - A stroke is glyph by the posterior share of its points inside the rendered glyph.
  - Near-edge px get soft targets from the noise model's posterior.
  - Only overlap px are decided by the template.
- **The boundary-band weight applies on synthetic only.**
- **Every build reports:**
  - the unknown share by kind × line-touch × quarter;
  - M1 in-box separately for boxes left unknown, which shows whether the hard case was learned at all.
- **Move it after the cheap step** (F10): build it only if a measured gap remains.

## F5. HIGH: other years. 86% of training text is not 2000, and the design says nothing about it

**Failure.**
- Text ink by series (`detector-overcall/pages.json`):

  | series | share of training text ink |
  |---|---|
  | 2000 | 14% |
  | 1980-90 | 13% |
  | 1990-96 | 13% |
  | 1960 | 16% |
  | 1964 | 19% |
  | 1968 | 26% |

- The only font is 2000 codes: 29 characters (" ()-1234ABCDEFGHIKLMNOPRSTUVWY", with no 0 or 5-9) learned from 959
  labels (`summary.json`).
- §2 gives other series fonts "only where their alignment fit is good" (l.79) and says nothing about the rest.
- **If the rest keep box targets,** 86% of positives still teach "line in box = text" and contradict the synthetic
  data. That is the incentive for F3's shortcut.
- **If they go unknown,** their positives vanish while their negatives stay. The detector turns against old
  lettering, which defeats the end goal (any old map in).

**Correction.**
- **A defined fallback for every word without a validated alignment:** box = text, except straight-line ink inside it
  (run rule and chain rule), which is unknown (no loss).
  - This needs no font.
  - It removes the contradiction.
  - It never teaches the run rule as "not text", so M1b stays independent.
- Clean-word sprites per series (F3) give exact positives in each series' own texture.
- Report label-source shares per series (aligned / box minus lines / unknown).
- Score one held-out quarter of an old map too.

## F6. HIGH: the variants don't isolate the cause, and the noise is not measured

**Failure.**
1. **(a) vs r328 changes two things:** the 328-px disc becomes 85-px Sol-only zones, and the target changes. The disc
   alone moved clean-line calls from 31% to 0.2% (`detector-overcall/results.md` tables). Any gain of (a) would be
   credited to the new labels.
2. **One run per variant:**
   - Over 3 seeds of one recipe, words found spread 0.936-0.950, and line ink outside boxes called text 0.48-0.82%
     (2,000 steps).
   - (b) against (c) will likely differ by less than that.
   - "Deterministic" on mps has not been shown by a repeat run.
3. **NE alone is a weak and biased choice:**
   - NE's words sit on 4 sheets (p01, p02, p07, p08), so "intervals over sheets" means n = 4.
   - NE is the dense extreme: 41% of answered labels (302/731).
   - A p chosen there may over-suppress sparse text, such as street names along streets in SW/NW.

**Correction.**
- **Add a control (0):** today's target with the new zones and F5's box-minus-lines unknown. Compare every variant to
  (0), not to r328.
- Run (0) at 2 seeds plus 1 same-seed repeat. That spread is "noise" in the keep rule.
- Intervals by bootstrap over blocks (clusters).
- Confirm the top two on SW before choosing.
- Write the decision rule down before the runs.

## F7. MEDIUM-HIGH: the "text-free" windows are far from labels, and Terra misses more text than the weight assumes

**Failure.**
- §2a says the textread NONTEXT windows "already do this". They don't:
  - `textread/crops.py:207-208` rejects any window touching an unknown px;
  - the unknown radius is held.json's 328 px (`crops.py:286`; `summary.json` `unknown_radius` 328).
- 95% of the 2000 map's ink outside words lies inside those discs (`results.md:296`). So the backgrounds come from the
  5% of the map far from any label, which is not where lines meet lettering.
- NE has the fewest windows: 0.15 per word, against 0.20-0.24 elsewhere.
- The windows carry weight 1 − 2.5% for Terra's misses. That rate counts only misses relative to the other LLM passes
  (`crops.py:228-244`).

**Evidence (missed text).**
- On the 13 sheets there are 5,121 letter-sized ink pieces (off run lines) more than 10 px from every box, against
  29,586 inside words.
- Of 64 drawn at random, **7 are plainly text** (legend and title "DISTRICTS", "COMMERCIAL"). Up to 16 are text if the
  doubtful street-name fragments count.
- That is **11-25%**, with 95% intervals of roughly 5-37% (`missed-candidates.png`). In (a) these become negatives.

**Correction.**
- Regenerate the windows under the new zone rule, sampled near labels from the measured distribution of line-to-label
  distances.
- Give each outside-box ink piece a measured p(text): a shape mixture fitted on in-word pieces against straight-line
  pieces. Weight its negatives by 1 − p(text), the background-recalibration idea from the literature note.
- Audit 64 random windows per build by eye, and state the binomial bound beside M1 and M2.

## F8. MEDIUM: "overlap = text" keeps the walls' border loss even with a perfect detector

**Failure.**
- The walls weigh each stroke by its mean text chance over its points (`wallevidence.py:272-275`, levels 0.25/0.5/0.75
  at `:73`). The strokes come from `strokes.npz`, cut at every junction.
- A line through touching letters is cut into short pieces whose end points are overlap px.
- Labelled text, those ends lift a short piece's mean past 0.25, and the piece stops being a wall.
- So the "serves every consumer" goal (l.42) is not checked against how the walls read the chance.

**Correction.**
- Two independent sigmoid outputs, glyph ink and line ink, both 1 on overlap px.
  - Synthetic data gives both exactly.
  - On real data the line output is trained only on synthetic, unknown elsewhere, so M1 stays independent.
  - The walls read the line output; the word boxes and the reader read the glyph output.
- If only one output is kept: change the walls' per-stroke statistic to leave out junction px, or use the median.
- Either way, M5 reports wall evidence on borders inside boxes specifically.

## F9. MEDIUM: gates missing for dashes, hatching and every other non-line ink

**Failure.**
- M1 counts only ink its line rule recognises. The 5.3% of outside-box ink on no rule (stair-steps, dashes, hatching,
  specks) is gated by nothing.
- Dashed lines and hatching are explicit goals (l.38), but only M3 tests them, on synthetic data (F3).
- M2's "words found" is recall-only: the over-caller's measure (root-cause note §8).
- The ≥ 0.5 and ≥ 0.9 cutoffs (l.132) are the only readouts.

**Correction.**
- Keep "all ink outside every box called text" as a gate, with no exclusions.
- Report the mean chance and log loss beside the cutoffs (probabilities, not cutoffs).
- Measure dash and hatch distributions first (l.91 admits they are unmeasured). Put real dashed and hatched windows in
  M4.
- Give a class its own M1 row only when it has a rule validated on synthetic data.

## F10. MEDIUM: order and scope. The riskiest part is a prerequisite, and block numbers and street names are not covered

**Failure.**
- The first variant, (a), needs all of §3 built first: the least-validated piece gates all learning.
- The owner named the method: synthetic letter shapes plus drawn lines ("do the direct thing").
- Block numbers and street names are 72% of 2000 words: 4,630 + 2,634 of 10,036, against 2,150 codes. Their crossing
  rates are 20% and 47%.
- The walls care about block lines at block-number corners. The font cannot render block numbers at all (no 0 or 5-9).

**Correction: the smallest steps that show value, in order.**
0. **$0, CPU, before training:**
   - corrected M1 and M2 (F1, F2), validated on the synthetic cases;
   - baselines for r30, r328 and the trial;
   - the M4 panel.

   This review's scripts already do half of it.
1. **One GPU run (NE): control (0).** Pictures of the RM-2 window.
2. **One GPU run: (0) + composites.**
   - The composites: clean real words OR real line windows near labels, and lines drawn over clean real words. These
     cover codes, block numbers and street names now.
   - p is the measured share of line-crossed words; no p sweep yet.

   The owner looks at the pictures.
3. **Second seed and SW** for the better of the two.
4. **Only then:**
   - the alignment labeller (validated on composites, F4);
   - the font extension;
   - the p sweep.

## F11. LOW: the per-px character head (variant d) is a distraction tonight

- No consumer reads it.
- Per-character targets exist only on synthetic data (F3's shortcut) or from the unvalidated labeller (F4).
- One run on one quarter can't separate its effect from noise (F6).
- The CTC reader already reads.
- **Defer it.** If step 2 leaves in-box line calls high, try it as an auxiliary loss on composites only.

## F12. LOW: smaller points

- The 85-px zone (l.108) is still a fixed disc derived from Terra lengths. Prefer confidence-gated unknowns around Sol
  points with no Terra word (the literature note, rec. 5).
- "Fonts come from crops outside the answered blocks" (l.116): the 2000 font is one font for all quarters
  (`synth.py:5-7`). That is fine for shape, but say so.

---

## The eight attacks, answered

1. **Labeller calls line ink glyph ink?**
   - Yes, for touching non-run line ink (halo), and at stroke edges (template, not ink).
   - No accuracy measure exists, and hard cases go unknown.
   - See F4: validate it on composites; label actual ink; report unknowns by line-touch.
2. **M1 independent and correct?**
   - No. Junction splitting drops 81% of in-box straight-line ink.
   - It calls 30% of line-free code-glyph ink line.
   - It misses stair-step district lines.
   - M2 then rewards the failure (F1, F2).
3. **Domain gap?**
   - Measured, and large: 190× more isolated px, 56× more pinholes, 5× more pieces.
   - See F3: real-glyph composites both ways, a texture-matched renderer, and a held-out real gate only.
4. **Missed text in the windows?**
   - 11-25% of letter-sized pieces outside boxes are text, against the assumed 2.5%.
   - The windows are also biased far from labels (F7).
5. **Keep rule and noise?**
   - The disc is not separated from the labels; there is one seed; NE spans 4 sheets.
   - See F6.
6. **Other years?**
   - 86% of training text has no font and no stated label policy (F5).
7. **Scope and order?**
   - Measures first. Then the control, then composites (F10).
   - Block numbers and street names are covered by sprites.
8. **Character head?**
   - Defer (F11).

## Measurements (scratchpad; CPU, one thread, read-only)

Scripts, run from `pipeline/` with `uv run python <script>`:
- `/private/tmp/claude-501/-Users-sbuss-workspace-sfzoning/eb18d2fc-e22a-48a6-afbc-57c31cd3e134/scratchpad/redteam_measure.py`
  - `synth`: the exact-truth line-rule cases → `synth.json`, `synth-all.png`
  - `real`: the in-box split by rule on 13 sheets → `real.json`, and the run-rule masks `run-<sheet>.npz`
  - `picture`: the RM-2 window → `rm2-window-rules.png`, `rm2-zooms.png`
  - `texture`: real against i.i.d.-rendered glyph ink → `texture.json`, `iid-rendered.png`
- `/private/tmp/claude-501/-Users-sbuss-workspace-sfzoning/eb18d2fc-e22a-48a6-afbc-57c31cd3e134/scratchpad/redteam_outside.py`:
  coverage of outside-box ink by each rule
- `/private/tmp/claude-501/-Users-sbuss-workspace-sfzoning/eb18d2fc-e22a-48a6-afbc-57c31cd3e134/scratchpad/redteam_glyphline.py`:
  the stroke rule on line-free words → `glyphline.json`
- `/private/tmp/claude-501/-Users-sbuss-workspace-sfzoning/eb18d2fc-e22a-48a6-afbc-57c31cd3e134/scratchpad/redteam_m2.py`:
  how the r30 detectors call in-box ink by rule → `m2.json`
- `/private/tmp/claude-501/-Users-sbuss-workspace-sfzoning/eb18d2fc-e22a-48a6-afbc-57c31cd3e134/scratchpad/redteam_missed.py`:
  letter-sized pieces outside boxes → `missed.json`, `missed-candidates.png`

Outputs are in `/private/tmp/claude-501/-Users-sbuss-workspace-sfzoning/eb18d2fc-e22a-48a6-afbc-57c31cd3e134/scratchpad/redteam-out/`.
The scratchpad drops files left untouched for 3 days.

Colours in `rm2-*` and `synth-*`:
- magenta: both rules;
- red: stroke rule only;
- blue: run rule only;
- grey: neither;
- green: Terra words.
