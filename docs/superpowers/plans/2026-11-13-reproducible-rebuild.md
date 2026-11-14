# Reproducible Rebuild Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** The sheet warper (PR #58), the text detector and the joint solver (PR #57) each rebuild from a fresh clone of
the repo, with outputs that are the same every run (within a stated epsilon). Earlier approaches are simplified where
they were wrong, not copied exactly.

**Architecture:**
- **Fixed inputs** come first, each reachable from a fresh clone: the scans and DataSF data (pipeline), the saved model
  reads (`pipeline/saved-reads`), and decisions people made (git).
- **Then the three parts, in dependency order:** warper → merged page → detector and solver inputs → solver. Each part
  is a package in `pipeline/scripts/` with one entry point.
- **Every output gets a determinism check:** run it twice from the same inputs, then compare.
- **Quality is judged by each part's own measure:**
  - the warper: block-line fit, and the seams by eye;
  - the detector: held-out text found;
  - the solver: the six held-out measures.

  Matching old files is not the test.

**Tech Stack:**
- The pipeline's Python (uv, pixi).
- Two R&D-only libraries, declared under a `rnd` extra: PyTorch 2.15.0 (warper, detector) and PyMaxflow (solver,
  GPL-3.0, which is compatible with AGPL).
- Data sharing: DVC + `sfzmaps ship`/`pull`.

**Spec:**
- The owner's direction of 2026-11-13: "ensure that everything in the map warper, the text detector, and the solver
  are fully reproducible from a fresh clone of the repo, and have *deterministic* outputs (small epsilon
  notwithstanding)"; "Some of the earlier approaches were wrong".
- The decisions log (memory `decisions-log`).
- This replaces `2026-11-13-solver-inputs-rebuild.md`, which aimed at exact rebuilds.

## Global Constraints

- **A fresh clone needs only** `git clone`, `uv sync --extra rnd`, `sfzmaps pull`, `python -m saved_reads
  restore-cache`, and the commands in this plan. No `/tmp`, home-folder or rescued-scratch path is read.
- **No new paid reads** unless the owner asks. Saved reads are fixed input (11-11). Changing a prompt, image or model
  means paying again.
- **Determinism:**
  - Every random choice is seeded.
  - Torch runs with `torch.use_deterministic_algorithms(True)` and a pinned thread count.
  - Each output's epsilon is stated in its task and checked by running it twice.
- **No cleaned page.** Walls are read from the original scan, and text is evidence, not erased (11-13).
- **No priors or areas from earlier solves.** Everything is learned from the table, held out by quarter (11-12).
- **Probabilities, not cutoffs,** for evidence. An existing cutoff that stays is named in its task.
- **One measure for every comparison,** and every score names its held-out area.
- **Code goes on a branch as it is written.** Data goes in `pipeline/data` and is shipped through DVC stages.
  Decisions people made go in git under `pipeline/inputs/`.
- **Nothing replaces what the published maps use without the owner's yes.**

## Decisions needed from the owner

| | Question | Recommendation |
|---|---|---|
| D1 | The judge2 table corrections (888 lots counted as the run's code: helper agents' verdicts) in the solver's score | **Drop them.** Score against the raw table and list the 216 groups for a person. |
| D2 | Which 2000 code reads the warper's code list uses | **The red-mark reruns.** Agreement with the owner's checked codes went 84.8% → 88.3% (Terra). |
| D3 | The owner's uncommitted `label_fixes.csv` rows (the 101 leader moves the solver learns from, plus 266 others) | **Commit them** in their own PR. |
| D4 | Warper simplifications | **One method for every sheet:** prob6 with the same settings. Drop the p06/p07 letter-erased variant and the p09/p10 "plus" special case, unless the fit measure shows they're needed. |
| D5 | What the text detector reads | **Each sheet's page**, not the merged page, so it doesn't depend on the warp and the warper can use it. |
| D6 | How the warper keeps printed text out of its fit (B2) until the detector is ready. Its old way, fitted-and-erased codes, is the cleaned page. | **Down-weight ink inside the saved Terra code boxes** as a stopgap: Terra at use, in the warper only. The other way is to wait for D. |

## Order

```
A fixed inputs ──► B warper ──► C merged page ──► E solver inputs ──► F solver ──► G fresh-clone check
        └────────► D detector (per-sheet pages; feeds B's text removal and E's walls)
```

### A. Fixed inputs, each reachable from a fresh clone

- [x] **A1. Saved model reads** (`saved-reads` branch).
  - 33,194 answers with how each was asked; their images ship through the frozen `saved-reads` stage;
    `restore-cache` puts the answers back in a fresh clone's cache.
  - The red-mark reruns of the code and leader reads are being added now.
  - Then: PR.
- [ ] **A2. The April 2000 table.** A `datasf.fetch_zoning_2000` (data set `cfhj-8c6b`) and a frozen stage
  `zoning-2000` (like `refdata`).
  - Check: the fetched table equals the kept download (`rnd/zon2000/april2000.geojson`) row for row.
  - Test: the stage is frozen and its output is the table's parquet.
- [ ] **A3. Decisions people made, in git.**
  - `label_fixes.csv` (D3).
  - The code-check and leader-check answers: today in `prob-warp:…/codes2000/answers/`; they move to
    `pipeline/inputs/2000-digital-zoning/`.
  - The frame-cut regions picked by eye (`zcuts.py`'s constants) as an input file.
- [ ] **A4. Declared dependencies.** A `rnd` extra in `pipeline/pyproject.toml` (torch==2.15.0, PyMaxflow), locked in
  `uv.lock`.
  - Check: `uv sync --extra rnd` in a fresh clone, then `python -c "import torch, maxflow"`.

### B. The warper (from PR #58's scripts to a package)

- [ ] **B1. Package `pipeline/scripts/warp/`, one entry point: `python -m warp fit|apply SERIES [SHEET...]`.**
  - Steps: block-line points → the prob6 field (`probfit`) → the warped sheet.
  - It reads only the pipeline's sheets, the saved reads and A's inputs.
  - Settings are fixed in one place: LOWMEM on, 3 threads, the same as the approved run.
- [ ] **B2. Text removal before the fit** (prob6 takes printed codes out of the ink it fits to).
  - Today it uses the years chain's fitted-and-erased codes. That is the cleaned-page method the 11-13 decision
    dropped.
  - Replacement, per D6: down-weight ink inside the saved Terra code boxes. Later, after D, use the detector's text
    odds.
  - Gate: the block-line fit (share of points within 1 px; folds) on every sheet is no worse than the approved run
    (`manifest.csv`), and the owner looks at the seams.
- [ ] **B3. One method for every sheet (D4).**
  - Drop the p1-or-prob6 choice unless prob6 folds on a sheet.
  - Drop the p06/p07 variant and the "plus" sheets.
  - Gate: as B2.
- [ ] **B4. Determinism.** Fit each sheet twice; the largest field difference must be ≤ 0.01 px.
  - If torch threading breaks this, pin deterministic algorithms and a single thread for the fit.
- [ ] **B5. A DVC stage per series** (`warp-<series>`), so warps ship and pull like other outputs.

### C. The merged page

- [ ] **C1. Package `pipeline/scripts/merge/`.**
  - Port `zmerge.py`, with the frame from `zframe2.py` and the cuts from A3.
  - Outputs: `raw.png` and `nsheets.png` on the fixed grid (4 ft/px, top-left 5979300, 2124100, 11900 × 9825), with
    no cleaned output.
  - Determinism: byte-identical (it is a union of 1-bit pages).
  - It is a stage after B5.

### D. The text detector

- [ ] **D1. Package `pipeline/scripts/textnet/`.**
  - It trains on each sheet's page (D5), with targets from the saved Terra boxes and "unknown" areas from the second
    passes and label reads.
  - Paths are settings.
- [ ] **D2. Determinism.**
  - Seeded, deterministic algorithms, CPU or pinned MPS.
  - Train the same quarter twice; the predictions must differ by at most 0.01 in probability. If MPS can't meet that,
    train on CPU and say how long it takes.
- [ ] **D3. Finish it.**
  - Train all four held-out quarters.
  - Report held-out text found, by kind, in one table.
  - The owner looks at a page of the pieces Terra didn't box.
- [ ] **D4. A stage, so the models and their predictions ship.**

### E. The solver's inputs (from fixed inputs and B–D only; no old runs)

- [ ] **E1. Quarters from fixed data.** Split at the median centroid of the table's lots inside the mapped area,
  written once as a constant with the code that computed it.
- [ ] **E2. Lot grid** from the table (A2) and the merged page's fixed grid.
  - Check: equal to today's `truth_grid.npz`, or a report of why not.
- [ ] **E3. Mapped area and scored lots** from `nsheets.png`.
  - Scored lots: a known table code, and more than half inside the mapped area.
  - Report what changes against today's 152,351 scored lots.
- [ ] **E4. Label evidence** from the saved reads and lot geometry, fitted held out by quarter.
  - Readings: Sol's labels, Terra's boxes, the turned pass, the third reads.
  - The owner's leader moves and the arrow step stay as named steps.
  - No path costs from old runs.
- [ ] **E5. Leader strokes** from `raw.png` (not the cleaned page); refit placement.
  - Report its held-out hit rate on the 101 leader moves.
- [ ] **E6. Lot-size fits** from the table and E3, per held-out quarter.
- [ ] **E7. Walls inside the solver** (replaces `zodds_Q1x`). Design note first, for the owner's yes:
  - wall odds per lot border, learned from the table;
  - read from `raw.png`, with text ink discounted by D;
  - no chain, no cleaned page, no cutoffs.

### F. The solver on the rebuilt inputs

- [ ] **F1. Solve the four held-out quarters** and report the six measures against J7, with every definition change
  (E1, E3, E4, E5, E7, D1) and its effect listed.
- [ ] **F2. Determinism.** Solve twice; the code of every lot must be identical, since the solve is discrete and
  seeded.
- [ ] **F3. Data errors** rebuilt from F1; update `joint-solver.md`.

### G. Fresh-clone check

- [ ] **G1. `pipeline/scripts/fresh_clone_check.sh`.** Clone into an empty folder, `uv sync --extra rnd`,
  `sfzmaps pull`, `restore-cache`, then run A–F's entry points.
  - Compare each output with the shipped one within its epsilon. Run it before each of these PRs merges.

## Review Focus

1. **A step quietly reading a file that only exists on this machine.**
   - Examples: `pipeline/data/rnd/…`, the rescued scratch, `/private/tmp`.
   - G1 fails it. Each package also has a test that its paths come from settings.
2. **Torch giving different results from run to run.** The B4 and D2 twice-run checks.
3. **A fitted step that saw its own held-out quarter.** Each fitted output records its quarter; F1 refuses one
   without it.
4. **The warp changing under the solver.** C, D and E take the warp's DVC hash as an input, so a new warp makes them
   stale.
5. **New code lists changing the warp (D2).** B2's gate is measured with the code list B actually uses.
