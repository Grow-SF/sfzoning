# The 2000 Map's Labels From Our Own Text Reader, Then Publish: Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development. Steps use checkbox (`- [ ]`) syntax.

**Status, 2026-11-16 ~21:00:** the owner said go ("Ok go on the plan"). On the owner questions below: "For release,
we will regenerate all maps when we're ready." So this plan builds the 2000 wiring and the page of differences at $0.
The release rebuild of every map (the stale district stages, heights, the 1996-2000 twin) is the owner's later step.

## Goal

- Publish a 2000 zoning map that is measurably better than today's published one.
- Its printed labels come from our own text detector and the text reader, with no LLM output at inference.

## The owner's decisions (2026-11-16)

- **Better, not perfect:** "It's okay to just be better. It doesn't have to be perfect. Ultimately we need to get to
  a point where we've finished this work and we produce a map that we can publish because the current published map
  has a lot of errors."
- **Hold the publish** until the labels come from the text reader, not the saved LLM reads.
- **Reader names:**
  - **the text reader** is the CTC recogniser, spike S2's "reader C": trained from scratch on crops at Terra's boxes
    plus synthetic code crops;
  - **the template reader** is the old generative reader. It now only makes the synthetic crops.
- **Speed over a near-tie:** the text reader is chosen over the template reader. They tie on accuracy, and the text
  reader is about 800× faster. The owner: "Why would I prefer to be 800 times slower for a fraction-of-a-percent
  improvement in accuracy?"
- **The compound search move is on in the stage.** Today's `final`: 4,543 wrong lots, labels right 94.79%,
  862 incidents.
- **Show the owner where the new map differs from the published one** before publishing.

## Where things stand

Sources: the survey of where LLM output enters (`docs/research/joint-solver/notes/2026-11-16-text-reader-survey.md`,
with file:line references), the spikes' results (`pipeline/data/solver/experiments/spike-s{0,1,2,3}/`), and the
previous plan's results note (`docs/research/joint-solver/notes/2026-11-16-search-move-and-reader-spikes.md`).

**LLM output still used to make the solve's answer** (at inference):
- The prints (`solver-prints`) are positions from Terra's boxes and Sol's points.
- The five saved reads of each print, through the trust fit (`solver-labels`, `trust.py`), become `readings_{q}.npz`.
- The walls' "letter share" draws Terra's boxes (`walls.py:260-265`; 12 of 463 features in `wallevidence.py`).

**Not blocking:**
- Placement and street sides learn from the table and the owner's leader moves (training only).
- The warp's sheet placement came from LLM-read anchors in the old pipeline; leave it for later.
- Heights in the published 2000 map are a separate LLM-read series. **Owner question 1 below.**

**The published map:** `period-2000-digital` publishes `districts_raw.parquet` (the old reader's code on DataSF's
2000-era lots), after unzoned blocks, 5 overrides, heights and parks. Scored like the solve: 35,455 wrong lots
(31,967 wrong codes, 2,202 blank, 1,286 with no matching lot), labels right 65.5%, 3,482 incidents.

## Global constraints

- **No LLM output at inference.** Terra's and Sol's saved answers may train and score. The owner's answers and
  `label_fixes.csv` may train and score.
- **Apache-2.0 code.** No new copyleft dependency. python-doctr cannot be a dependency (its opencv conflicts), so
  the text reader is re-expressed in plain torch. If torchvision is needed, it is BSD.
- **Held out by quarter,** as everything in the solver is. The answered blocks are never trained on.
- **One measure when comparing:** the solve's six measures, every-lot column, with the paired block bootstrap
  (`scripts/incidents/criteria_check.py`), against today's `final`.
- **Better is enough:** each step is kept when it is measurably better, or no worse within noise, on the agreed
  measures. No step chases every miss.
- **Load near 20; one GPU job at a time; deterministic.** Commit locally; the owner merges and ships.

## Tasks

### T0: Fix the scoring labels before the label source changes

- **Why:** "labels right" is computed from today's LLM readings (`measures.Table`, `report.top_reading_calibration`).
  If the solve's labels change, the score's label set must not change with them.
- **What:**
  - The 1,881 scoring labels stay those of today's chain.
  - The solve's own labels get their own location: `trust.load_prints`, `labels.py`, `streetside.py` and
    `placement.py` read from it.
  - `prints.parquet` stays as is: `solver-text-reader` uses it to define its held-out blocks.
- **Also (the previous plan's final review):**
  - `criteria_check.label_lots` repeats the label source in `measures.Table`. Make both read the one fixed scoring
    set.
  - The per-block totals check is weak (it can't catch a regrouping). Strengthen it.
  - The energy criterion ("energy lower in every quarter") means nothing once walls or labels change. Say so in
    `criteria_check`, and leave it out of comparisons across label sources.
- **Measure:** re-solving `final` gives a byte-identical `report.json`.

### T1: The walls without Terra's boxes, from the honest per-quarter text chance

- **What:**
  - The walls' letter share uses the existing `text_weighted` option (the detector's chance in place of Terra's
    share). The README's held-out log loss shows a tie: .0982/.1026/.0452/.0612 against
    .0978/.1027/.0452/.0611.
  - `textgrid` reads the per-quarter chance from `solver-text-reader`. Each px then comes from the detector that never
    saw its quarter; today it comes from `solver-text`, which trained on these very sheets.
  - Drop `terra_boxes.json` from `solver-walls`' reads.
- **Measure:** the walls' held-out log loss, then the six measures.

### T2: Word boxes from our detector (`solver-boxes`)

- **What:** spike S3's plain grouping, run on the per-quarter text chance of every 2000 sheet. Each box records:
  - corners, axis (mod 180°), length, height;
  - mean chance and kind probabilities;
  - quarter;
  - map `x`/`y`, `lab_lot`, `street` (through `prints._place` / `under`).
- **The level:** chosen per quarter on the other quarters, by the solve's measures if cheap, else by code-word
  recall.
- **Measure:** per quarter, the recall of Terra's zoning-code words (IoU 0.3) and the share with axis within 10°;
  how many of the owner's answered labels get a box.

### T3: The text reader as a stage (`solver-text-read`), in plain torch

- **What:**
  - **The model:** MobileNetV3-small features, a 2-layer BiLSTM 128 and CTC, written in plain torch (~150 lines).
    The parameter names stay the spike's, so the spike's weights load as a regression check.
  - **Four readers,** one per held-out quarter, with the answered blocks held out of all four. The same for the 0°/180°
    classifier. About 90 min of GPU in all.
  - **Inputs** the synthetic crops need: the template reader's font, code forms and prior. They are made by a
    stage, not copied from scratch.
  - **Reading:** each box is read by its quarter's reader, both ways along its axis. The output is a posterior over
    the code list's printed forms (exact CTC forward), plus "other" and "none".
- **Measure:**
  - top-1, top-3 and −log p on the owner's answers matched to boxes, box-found and box-missed apart, against the
    spike (0.956–0.973 closed on Terra's boxes) and the LLM chain (96.9%);
  - the same reader on Terra's boxes, so that box error and reader error show separately.

### T4: The solve's labels from boxes and the text reader

- **What:**
  - Boxes where a code has real mass become the solve's prints.
  - A calibration replaces the trust fit. Same target, rows and quarter rule as `trust.py`. Features: closed-list
    mass, margin, p(other), p(none), box height, the detector's p(zoning code), on a street.
  - It writes `readings_{q}.npz` in today's format.
  - Street sides and placement refit as they do today.
  - Fix the `leader_moves` quirk: it matches on the read code, not `fix_code`, in 21 of 101 rows.
  - **`TOP_READING_FIT`** (`energy.py:12,49`) maps a label's top trust through constants fitted to the LLM chain's
    trust. The new calibration fills the same slot. Refit it, held out by quarter, on the new readings, or set it to
    identity on purpose because the calibration is already a probability. Don't leave the old mapping in silently.
  - **Readings per label.** Today a print has about 1.24 readings. The tie-break's path search (`energy.py:66-75`)
    starts from every lot with any reading of a code, and the leftover mass per label (`41-52`) depends on how many
    codes are written. Write only the codes with real mass, and report the number per label. Keep the `names` list,
    which decides the codes the solve can output.
- **Measure:**
  - the six measures against `final`, with intervals (expected somewhat worse; "better than published" is the bar);
  - the calibration's held-out log loss;
  - `top_reading_calibration`.

### T5: Publish the solve's codes on the 2000 map, with a page of differences

- **First (the previous plan's final review):** 16 district stages (`edition-*`, `period-*`, `today`, `shared`) have
  been stale since 11-02, before this branch. Their recorded `warp.py`, `transform.py`, `fresh.py` and `llm.py`
  predate 7 commits, and their lock entries lack the `sheetwarp` dep.
  - Rebuilding `period-2000-digital` would mix those changes into the page of differences.
  - Find what rebuilding them changes, with `sfzmaps estimate` and cache-only: the `llm.py` change may move cache
    keys, so a rebuild might want paid reads. Bring the result to the owner before any rebuild.
- **What:** the `period-2000-digital` stage takes its codes from the solve (`solver-solve` upstream, a DVC link)
  instead of `districts_raw`'s, keeping its other steps (unzoned, overrides, heights, parks). It handles:
  - **lots:** today's table lots are mapped to DataSF's 2000-era lots by majority area;
  - **the 45 duplicated IDs;**
  - **spellings** (`canon`, " NCD", aliases), so that `unknown_codes.json` stays empty.

  **(My call)** over two alternatives:
  - a generated file in `inputs/`: that breaks "inputs are what a person decided";
  - a new publish folder that every reader must follow: larger.
- **The page of differences** for the owner's eye: every connected area where the new map differs from the
  published one, largest first. Each shows the scan, both codes, the table's code and the lots.
- **Measure:**
  - `published_compare` on the new published map (today's: 35,455 wrong lots);
  - all 184,241 lots coded;
  - the owner checks the differences page.
- The owner ships and merges.

## Owner questions

1. **Heights:** the published 2000 map's heights come from a separate LLM-read series. Keep them as published today
   for this release (no new reads), or hold them too?
2. **The 1996-2000 twin scan:** it is published from its own old reads. Leave it as it is for this release?

## Not in this plan

- Debugging the text reader's misses beyond what T3's measures show. Better is enough.
- The table-free lower bound on the search (TRW-S): offered, not yet agreed.
- Other years, and a new map's first placement.
