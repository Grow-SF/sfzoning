# Why line ink inside a label was taught, and scored, as text (root cause, 2026-11-16)

This was a read-only investigation. Facts point at a file and line, a commit, or the session transcript. Transcript
times are given in PST; the file stores them in UTC. Anything marked **Inference** is my reading, not a recorded fact.
Section 6 covers the 328-px "unknown" disc, which made the same failure worse near labels.

## The mistake

- **The target.** A px is "text" when it lies inside a Terra word rectangle:
  - `textnet/data.py:4`: "text 1 inside one of Terra's text boxes (every kind), drawn as the rotated rectangle of the
    word in page px";
  - `data.py:469`: `"text": (word > 0)`.
  - The loss reads every known ink px and its 8 neighbours (`ink+1px`, `train.py:12-13`). So a block, street or
    district line crossing a label is taught as text, in both detectors.
- **Its size.** On the three test drawings, 13.5-13.8% of what the scores call "text ink" is ink on long lines.
  - Source: `data/solver/text-scores/scores.json`, computed as `long_lines.in_a_box` × `ink_px` ÷ `text_ink_px`.
  - The all-sheets detector calls 71-72% of that ink text.
  - On the 2000 sheets, the spike found that "16.4% of the ink inside Terra's word rectangles is on long lines. Most of
    that is lines crossing the rectangles" (`data/solver/experiments/detector-overcall/results.md:123-124`).
  - `long_lines` can also pick up the odd letter stroke that touches a line, so these shares are upper bounds on line
    ink alone (`results.md:116-119`).

## 1. Timeline

| When (PST) | Commit or record | What |
|---|---|---|
| before 11-13 14:08 | R&D scratch, committed "as it ran" in `46f8ec1` | `joint_solver/textnet/build_data.py:5`: "text 1 inside a terra text box (every kind), as a rotated rectangle". The line measure looks only at lines "away from any box" (`evaluate_detector.py:7`, `:62`, `:85`). |
| 11-13 14:08 | `46f8ec1`, commit message | "Ink on table district lines called text: 0.55% (NE), 0.22% (NW)": the "away from any box" qualifier is dropped. |
| 11-13 14:08 | `d81bed1`, `notes/one-model-design.md:132-134` | Walls R&D, "Gap 1": "printed text sits on the district line … letter share 0.63 … needs a synthetic case (case C)". |
| 11-15 00:25 | spec `aa9374a`, `solver-build-stages-design.md:214-222` | "a port of `joint_solver/textnet`"; "Targets: the Terra boxes in page px"; "Model and training: as R&D". |
| 11-15 00:28 | plan `39a4f7c`, `solver-inputs.md:394` and `:416` | "drawn as rotated rectangles in page px"; the measure is "the share of ink on table district lines called text" (no qualifier). |
| 11-15 02:09 | `487e633`, `textnet/evaluate.py:14`, `:34`, `:132` | The port keeps "district lines … away from every box and unknown area" (`NEAR_BOX_PX = 7`). |
| 11-15 ~05:44 | ledger `2026-11-15-solver-inputs/progress.md:115` | "district-line ink called text 0.16%" (no qualifier). |
| 11-15 morning | `decisions-log.md:63-66` | The owner: judge every change by "words found, district-line ink called text". |
| 11-15 08:55 | spec `a60e1fc`, `text-detector-training-design.md:64`, `:90-91`, `:97-98` | Targets: "Terra's boxes … the way the 2000 boxes are decoded today". The district-line measure is replaced by "long straight lines" **(my call)**. The quarter models' "0.16%" is quoted. |
| 11-15 09:32 | `b5abd6d` | Today's `data.py:4` and `:469`. |
| 11-15 10:25 / 10:31 | `6f6f4be`, `e7550a4`, `scores.py:26-28` | Long-line ink inside a word is "which a detector calling exactly the words would call". |
| 11-15 ~10:30 | `2026-11-15-text-detector-training/task-3-report.md:70`, `:151-152`; `progress.md:46-48` | In-box line ink is named "the floor a box-perfect detector hits" (6.1-6.9% on 2000). The reviewer approved the task, citing "validation mask shared with training". |
| 11-15 10:59 | `b0c5821`, `trial.py:329-330` | The recipe trials are judged on `long_lines.called_text`, which includes in-box ink. |
| 11-15 12:49 | spec `66e1c9a`, `code-reader-design.md:78-80` | The reader masks long-line px: "A boundary line through a label then neither helps nor hurts any code". The detector's target is not changed. |
| 11-16 ~10:05 | R1 pictures (`experiments/text-reader-pictures/`) | The first pictures of the chance over a dense window. The report says of the walls' detector: "Lines stay grey" (`task-R1-report.md:377`). |
| 11-16 23:21 | the spike's picture 0, sent to the owner | Captioned "The two on the right have the 30-px zone and call only the lettering." |
| 11-16 23:36 | owner (transcript) | "It looks like you're flagging the walls behind the text, the printed block lines, and street lines … You've got to actually learn to discriminate the text from the background ink." |

## 2. The decision

**What the documents say:**
- R&D target (`46f8ec1`, `build_data.py:5`): "text 1 inside a terra text box (every kind), as a rotated rectangle
  mapped from its sheet onto the merged page".
- Build-stages spec (`solver-build-stages-design.md:217`, `:222`): "Targets: the Terra boxes in page px." …
  "Model and training: as R&D."
- Solver-inputs plan (`solver-inputs.md:394`): "Targets are the sheet's Terra boxes … drawn as rotated rectangles in
  page px."
- Detector-training spec (`text-detector-training-design.md:64-65`): "Targets: Terra's boxes for each sheet, decoded …
  the way the 2000 boxes are decoded today."

**Who decided it:**
- **Facts.**
  - The specs mark calls the owner did not make with **(my call)**. None of the target lines carries that mark or an
    owner quote.
  - The owner's recorded decision is only "Our own text detector/reader (CNN + CTC) trained on terra's saved boxes"
    (`decisions-log.md:29`). It doesn't say what a box means px by px.
  - The R&D script ran from the agent's scratch folder (`build_data.py:21`).
- **Inference.** The agent set the target while writing the R&D script. Every later spec carried it as "as R&D" or
  "as today", and no one decided it again.

**Did anyone consider line ink crossing a word?** No detector document does: a search of `docs/superpowers/` and
`docs/research/joint-solver/notes/` for lines through, crossing or inside a label or box finds nothing about the
detector. The overlap was known in four other places:
- **The reader** (`code-reader-design.md:78-80`, 11-15 12:49): "Pixels on long lines are unobserved … A boundary line
  through a label then neither helps nor hurts any code, so a thick line can't turn H into M."
- **The walls' R&D** (`one-model-design.md:132-134`): "printed text sits on the district line. The nearest ink is an
  8-ft stroke with letter share 0.63". The letter share came from Terra's boxes, so line ink inside a box counted as
  letters. Case C was never built: a search of `docs/research/joint-solver` and `scripts/joint_solver` finds no
  mention after this note.
- **The old cleaned page** erased "letters and thin block lines … using terra text boxes" (memory
  `wall-odds-provenance.md`). On 11-06 the owner said to "delete the block lines before erasing letters" (memory
  `do-the-direct-thing.md`).
- **The overcall spike** (`results.md:123-124`) saw the crossing lines, but still tabled in-box ink as "True text"
  (`results.md:163`).

## 3. Why the measures missed it

Every score is computed from the same rectangles as the target (`scores.py:10-11`, "text: the px of Terra's words, as
the detector learns them"). So calling crossing-line ink text counts as right, or isn't counted at all:

| Measure | Where | What it does with line ink inside a box | Effect |
|---|---|---|---|
| Text ink found, AUC, log loss | `scores.py:174`, `:220-223` | It is a positive (`pos` = `scored & text`). | **Rewarded.** Calling it not-text lowers all three. |
| Words found | `scores.py:208-212` | The mean chance over all ink inside the rectangle, line ink included. | **Rewarded.** Calling the line text raises the mean. Also the trials' first gate. |
| Ink outside boxes called text | `scores.py:175`, `:223` | Never in it: negatives are `~text` only. | **Not counted at all.** |
| Long-line ink called text (the trials' second gate) | `scores.py:26-28`, `:225-230`; `trial.py:329-330`; `README.md:707-709` | Counted, but defined as correct ("which a detector calling exactly the words would call"; "the floor a box-perfect detector hits", `task-3-report.md:70`). | **Read as correct, not as an error.** In every validation trial, 66-72% of the in-box long-line ink was called text, while the gate moved by hundredths of a point (`experiments/text-*/scores.json`; derived from `in_a_box`, `called_text`, `rest_called_text`). On the test drawings, 92-95% of the long-line ink called text lies inside boxes. `rest_called_text`, the outside-box part, was computed but never gated. |
| District-line ink called text (R&D, solver-inputs) | `evaluate_detector.py:62`, `:85`; `487e633` `evaluate.py:122`, `:132` | Excluded: px within the 7×7-widened boxes and unknown areas were cut out. | **Not counted at all.** Then quoted without the qualifier: 0.55% / 0.22% (`46f8ec1`), 0.16% (`solver-inputs/progress.md:115`, `text-detector-training-design.md:97-98`). `task-3-report.md:152` noted that "today's 0.16% district-line figure left out ink near boxes", but nothing changed. |
| Unknown px (30-px Sol discs, other passes' boxes) | `scores.py:219-220`; `data.py:432-436` | Left out of every score, and never trained. | **Not counted at all.** **Inference:** line ink just past a word near a Sol point, like the owner's "just past the 2", was neither taught nor scored. |

The 30-px detectors look clean on the spike's measure for the same reason: its "clean not-text" set is line ink at
least 10 px from every Terra word (`results.md:103`).

## 4. Pictures and reviews

- **The R&D picture tool** (`46f8ec1`, `look_pred.py:1-5`) chose its windows by the box measures: Terra words missed
  and unboxed pieces called text. In-box line ink counts as right under both, so it could never be picked.
- **Nothing reached the owner before 11-16 23:21.** The transcript's file sends (`SendUserFile`) from 11-13 to 11-16
  23:21 include no picture of a detector's chance. The two artifact publishes on 11-16 are the incidents report.
- **No detector plan asked for a picture of the chance:** `text-detector-training.md`, its spec, and the
  `solver-inputs.md` detector task.
- **The first dense-window pictures** were R1's, at about 10:05 on 11-16 (`task-R1-report.md:167-169`, `:377`).
  - The report says of the walls' detector: "Lines stay grey", and "The walls' detector does not" call district lines
    text.
  - **Observation** (my look at `chance-walls-vs-reader.png`, left panel): some line ink at labels is red, e.g. a
    block-line corner just left of "RM-1", and the line where "NC-2" sits.
  - These pictures were not sent to the owner.
- **The spike's picture 0** (`detector-overcall/pictures/0-most-inside-p02-1600-1920.png`) went to the owner at 23:21,
  captioned "call only the lettering".
  - **Observation:** in its two 30-px panels, lines crossing labels are red too, e.g. the vertical "RM-2" at the right.
  - The owner named the failure from this picture at 23:36. It was the first time anyone did.
- **The reviews:**
  - The detector-training task-3 report saw in-box line ink and called it a floor (`task-3-report.md:70`, `:151-152`).
    Its review approved it (`progress.md:48`).
  - The final review doesn't raise it (`final-review.md`).
  - The solver-inputs task-6 report copied R&D's near-box exclusion "as R&D" (`task-6-report.md:66-67`).
  - No ledger or review before 11-16 23:21 says that line ink inside a label is called text. R1's concern 1 is about
    the 328-px discs, not the box target.

## 5. Earlier warnings

- **The owner asked for "district-line ink called text"** (11-15, `decisions-log.md:63-66`). It became a long-line
  stand-in (**(my call)**, `text-detector-training-design.md:90-91`), with in-box ink scored as correct.
- **The owner's E7** (`decisions-log.md:55-57`): walls "read from the original scan, text discounted by the
  detector". Telling lettering from line ink was the detector's purpose for the walls.
  - **Inference:** line ink in labels called text is discounted as lettering, which removes wall evidence exactly
    where labels sit on lines. This is the same failure as Gap 1, and possibly behind T1's lost border beside the
    "RH-1(D)" letters (the caption of the T1 picture sent at 22:26 on 11-16).
- **Gap 1, the reader spec, and the owner's 11-06 instruction** (section 2) each showed that letters and lines
  overlap.

## 6. Secondary: the 328-px disc

**Timeline:**
- **The rule.**
  - Spec `b81a063` (11-16 08:47, under **(my call)**): "a measured word length each way along any direction, not
    30 px" (`code-reader-read-along-the-word-design.md:70-75`).
  - Plan `7aab4e7` (08:57) made it "the longest printed code word among training Terra boxes of kind 'zoning code'"
    (`code-reader-read-along-the-word.md:94-95`).
  - `ce828df` (09:29) implemented it as a maximum (`reader_stage.py:96-122`, `:114`).
- **The run** (about 09:30-09:58) gave 328 px, from the "SH" on 1968 sheet p18, a 328 × 293 box.
  - The median code word is 85 px and the 99th percentile 169 px (`text-reader/held.json`).
  - R1's report flagged it as "a Terra mis-box" and named the over-calling (`task-R1-report.md`, "Unknown radius" and
    concern 1).
- **The owner saw it at 10:21, 10:25 and 10:44** (transcript): "The radius was set by one mis-drawn Terra box at
  328 px, so 43% of the map was marked unknown".
  - A fix was proposed: discs only around the 96 Sol points in no Terra word.
  - The owner then redirected the work (11:19, 11:28).
  - The plan was superseded with "R1 … is kept" (`code-reader-read-along-the-word.md:3-5`), with no word on the
    defect.
  - The 13:05 compaction summary doesn't mention it (transcript).
- **`700f40c`** (21:48) recorded the 09:58 outputs in `dvc.lock` after a check that their inputs hadn't changed, not
  that they were good.
- **The new plan** (`2552f04`) called them "the honest per-quarter text chance" (`text-reader-labels-and-publish.md:83`).
  At 22:46 the controller called the disc cause "the T2 implementer's guess". The morning's measurement had been lost.
- **The spike `ba36061`** (23:21) confirmed it.

**Why the stage's measures missed it:**
- `quarter_scores` scores `ink & ~unknown` with the 328-px discs as unknown (`reader_stage.py:182`), so the AUC, log
  loss and outside-box ink leave out exactly the failing px.
- It drops long lines (`reader_stage.py:207`).
- It counts words found on all ink (`:186`), which over-calling wins.
- R1's report said so: "On the px outside the 328-px discs (the stage's own `scores.json` convention) the reader's
  detectors look fine" (`task-R1-report.md:201`).
- The comparison with the all-sheets detector was made only in scratch scripts, not in the stage.

## 7. Process gaps

**The text target:**
1. A per-px target was inherited from an R&D script ("inside a Terra box = text") and carried into three specs as "as
   R&D"/"as today", never decided, even though the reader spec masked lines through labels the same day.
2. Every measure was built from the same box raster as the target. So crossing-line ink called text was scored as
   right (text ink found, AUC, words found), or never counted (outside-box ink, near-box line ink, unknown px).
3. The owner's measure, "district-line ink called text", was swapped for a stand-in that defined in-box line ink as a
   "floor". When the R&D numbers were quoted, the "away from any box" qualifier was dropped, so 0.16-0.55% read as a
   whole-map rate.
4. No picture of the chance over a dense area was required or shown before 11-16 23:21. The one picture tool chose
   windows by the box measures. The two later pictures were captioned "lines stay grey" and "call only the lettering"
   when they showed otherwise.
5. A warning in the walls' notes (Gap 1: text on a district line counted as letters, "needs case C") was never
   followed up or linked to the detector's targets.

**The disc:**
1. A parameter was set from the maximum of an unvetted LLM-labelled set (one Terra mis-box).
2. The stage's measure shared the training mask, and dropped the line measure.
3. A defect known and shown to the owner lived only in the conversation and the SDD ledger. The compaction dropped it,
   and the outputs were later recorded after a check of their inputs only.

## 8. Near-misses of the same kind

- `READ_POINT_RADIUS = 30` is an unmeasured R&D constant (`build_data.py:33` → `data.py:52`). Its discs are left out
  of every score (`scores.py:219-220`).
- `measure_radius` takes the maximum of noisy Terra rectangles (`reader_stage.py:114`).
- The spike's "clean" not-text set excludes 10 px around every Terra word (`results.md:103`). That is the
  blind spot above, again.
- The recipe gate on "words found" (`README.md:707`) and T2's box level, chosen "by Terra code-word recall"
  (`2026-11-16-text-reader-labels-and-publish/progress.md`, T2), are recall-only. An over-calling detector wins both:
  T2's chosen levels sat at 0.995, where the chance was saturated.

## Fact-check (2026-11-17, an independent agent, against files, git and data)

The main chain stands. The target is the whole word rectangle (`data.py:4`, `:469`), carried into the stages
without a recorded owner decision. The scores count in-box line ink as right or ignore it. The unqualified
0.16-0.55% figures came from a measure that left out near-box ink. The warning at `one-model-design.md:134` was not
followed up. The 13.5% and 71-72% figures re-derive from `scores.json`. Corrections to the text above:

- **"In every recipe trial 66-72%"** should read: 30 of 33 trials, 66-72%. The three trials trained on three maps
  only (`2000-digital-zoning+1980-1990+1990-1996`) give 84-86%.
- **The specs' wording:** only `solver-build-stages-design.md` uses "as R&D", and there on the model line
  (`:222`); `:217` says "Targets: the Terra boxes in page px." `solver-inputs.md:394` restates the target without
  "as R&D" or "as today".
- **"The owner's measure was swapped":**
  - the owner asked for "the same held-out scores (words found, district-line ink called text)", which already
    meant the R&D measure that left out near-box ink;
  - so the owner endorsed that measure without knowing about the exclusion;
  - the dropped qualifier stands.
- **Times:**
  - The owner named the failure at about 00:10 on 11-17, per the ledger, not 23:36.
  - Whether no dense-area picture was sent before 11-16 23:21 cannot be checked from files. The repo holds no
    record of a send.
  - The 10:21/10:25/10:44 showings of the 328 px value and the 22:46 time are also unverifiable from files.
- **The disc cause** is supported by four measurements, but no run changes the radius alone
  (`detector-overcall/results.md:388-398`).
- **The long-line measure** (`scores.py:225-230`) keeps in-box ink and drops only unknown px. The near-box
  exclusion is in the district-line measure (`evaluate_detector.py:85`, `evaluate.py:122`).
