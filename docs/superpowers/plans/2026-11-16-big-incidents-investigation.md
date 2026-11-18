# The 2000 solver's big incidents: an overnight investigation

> **For agentic workers:** the controller runs this plan task by task with subagents (superpowers:subagent-driven-development
> in spirit: one fresh agent per task, an adversarial check of every diagnosis before it is reported). It investigates
> and measures. It never changes the solver.

**Goal:** by the owner's morning, know what causes the solver's biggest mistakes, how many wrong lots each cause
costs, and what could be done about each. The evidence should let the owner choose tomorrow's accuracy work in
minutes.

**Why:** the solve on PR #62 matches the R&D solver's best run (J7) but does not beat it.
- Its mistakes are lumpy: 9 incidents of 100+ lots hold 3,686 of its 7,100 wrong lots.
- The largest, RH-1 read as RH-1(S) behind Mission St, holds 1,838 alone.
- Fixing a few causes may matter more than any general change.

**Already diagnosed** (`/private/tmp/claude-501/-Users-sbuss-workspace-sfzoning/eb18d2fc-e22a-48a6-afbc-57c31cd3e134/scratchpad/incidents/report.md`,
with its scripts in the same folder):
- **451 lots RH-1(D) → RH-1, Junipero Serra.** The map prints the whole roadway as P, but the table's P parcel is only
  the east strip. The west edge's border px fall mid-road, among the street lettering.
- **267 lots RM-1 → NC-1, California St.** A placement near-tie. Three NC-1 labels are printed in the street with leader
  dashes, and `label_fixes.csv` has no leader ends for them.

## What the owner sees in the morning

**A report page (an Artifact),** private by default and linked in chat. It holds:

1. **The cause ranking:** each cause's wrong lots, over every incident of 20+ lots, not only the big ones.
2. **One card per big incident:**
   - a picture of the page ink with the table's, the solve's and J7's district lines;
   - the term that made the wrong answer cheaper (labels, walls, placement, or the search);
   - the cause in plain words;
   - the adversarial check's verdict.
3. **For each top cause:**
   - what an oracle fix would save (the ceiling);
   - what was tried before and why it was dropped;
   - what the literature and published libraries offer;
   - two or three candidate approaches, each with a red team, the measure it would be judged by, and the synthetic test
     case it needs.
4. **The decisions that need the owner,** most important first.

**A written record** (`docs/research/joint-solver/notes/2026-11-16-big-incidents.md`) with the same findings in text.

## Global constraints

- **Read-only on the solver.** No change to product code, `pipeline/inputs/`, or any stage output under
  `pipeline/data/solver/{page,lots,prints,labels,text,text-scores,walls,solve}`.
  - Experiments write to `pipeline/data/solver/experiments/incidents-*/`.
  - Investigation scripts go in `pipeline/scripts/incidents/`, committed on the local branch and not pushed: the owner
    decides in the morning whether they join #62 or a branch of their own.
  - Start from the first inspection's scripts (the scratch folder above), moved into the repo.
- **No fixes overnight,** not even a code bug. A bug gets a minimal reproduction and a failing test written into the
  report, and waits for the owner. A correction of a method is not permission (`wait-for-go-ahead`).
- **No paid or LLM reads,** and no new reads of the scans. Everything is deterministic.
- **The GPU isn't needed.** If it ever is: one exclusive turn (`sfzmaps.gpu.gpu_turn(True, ...)`).
- **Load around 20 on the 10-core Mac.**
  - One thread per worker (OMP/OPENBLAS/VECLIB/MKL_NUM_THREADS=1, `cv2.setNumThreads(1)`).
  - At most two heavy agents at once. The code reader's Task 4 runs beside them.
  - Check `uptime` a minute into any heavy run.
- **One measure:** every oracle and comparison is scored with `sfzmaps solver score`, on the every-lot column, against
  the stage's solve `final`.
- **Pictures say what they are for,** and whether anything in them needs action. They are drawn at the grid's own
  resolution or 2x, never downscaled.
- **Read the owner's rules and past tries before proposing anything.** The memory folder is
  `/Users/sbuss/.claude/projects/-Users-sbuss-workspace-sfzoning/memory/`. In particular:
  - `decisions-log.md`;
  - `whole-line-tried.md`, `spread-solve-tried.md`, `leader-lines-exist.md`, `past-rejections-confounded.md`,
    `special-case-labels.md`, `fix-table-offsets.md`, `parks-always-public.md`;
  - `probabilities-not-cutoffs.md`, `zoning-border-rules.md`, `one-problem-synthetic-tests.md`,
    `tests-from-measured-distributions.md`, `errors-stack-dont-discard.md`, `learn-priors-and-mixture.md`,
    `deterministic-no-llm-answers.md`, `principled-loop.md`;
  - `docs/research/joint-solver/` (the R&D notes).

## How new information is handled

Every rule below is applied by the controller and recorded in the ledger
(`.superpowers/sdd/2026-11-16-big-incidents/progress.md`) as a ruling.

- **A cause outside the taxonomy:** add a class, and re-run the census for it (Task 3).
- **A cause that past tries already cover:** say so and cite the try. Re-propose it only if the old rejection was
  confounded (`past-rejections-confounded`: cutoffs, the cleaned page, lot counts); then the re-test design says how
  it is made fair.
- **A cause with a plausible fix not tried before:** run the research step (Task 5):
  - a literature review (WebSearch, standard mode);
  - a scan of published libraries and workflows, with each one's license (AGPL-3.0-compatible only) and fit;
  - an autonomous brainstorm that writes candidate approaches with a red team.

  None is built.
- **A cheap oracle can bound a fix:** run it (Task 4). An oracle fix of the cause in the solver's inputs, scored, gives
  the most a fix could gain. Oracles use the table only to make a ceiling, never as an input to anything that ships.
- **A library looks like it does the job:** a spike only if it is cheap (under about 30 minutes) and read-only. Label it
  throwaway, put it in the experiments folder, and report what it showed.
- **A finding contradicts an earlier diagnosis:** re-check both with the same measure and report both.
- **One cause dominates** (more than half the big incidents' lots): Tasks 4 and 5 focus on it first. Ledger the
  ruling.
- **Something needs the owner** (a choice between approaches, a rule question, a data error in the table): put it in
  the report's decisions list. Don't stop the night for it.

## Tasks

### Task 1: The census of incidents

One agent. Its outputs are `incidents.json` and a table in `pipeline/data/solver/experiments/incidents-census/`.

- Every incident of the solve `final`: a connected set of wrong lots that share one wrong code, as `measures`
  defines it. Take all of 20+ lots, with their lots, table code, solved code, quarter and blocks.
- **For each incident of 100+ lots** (expect 9):
  - **Energy:** the table's answer against the solved answer over the changed lots, under the run's own saved costs.
    Split it into label, wall and placement terms (and any others the energy has).
  - **Which term made the wrong answer cheaper,** and by how much.
  - **Is the solved answer the run's own minimum on that set?** If not, the search missed it.
  - **The same incident in J7's solve** (`scratchpad/solver/j7_lot_of_new.npy` maps lots, comparisons only): does J7
    get it right, and if so, which of its inputs differs (its walls, its labels)?
- **For incidents of 20–99 lots:** the same energy split, automated, with no pictures.

### Task 2: Diagnose each big incident

Agents in pairs, at most two at once. Each takes three or four incidents of 100+ lots. The two already diagnosed get
only a cross-check against the census, plus the 275-lot RH-1(D) → RH-1 at 6500/6499/6459 (is it the same P-roadway
cause?).

**Per incident:**
- **Pictures:**
  - page ink;
  - the table's, the solve's and J7's lines;
  - the labels with their top reading, trust and placement;
  - the change in wall chance against the default walls where it matters.
- **The cause in plain words.**
- **A cause class:**
  - label misread;
  - label placed on the wrong lots (leader lines, labels in the street);
  - wall missing on a printed line (text over the line, a roadway district, a dashed or thin line, a line off the lot
    lines);
  - wall on a line that is not a district line (lot line, street edge, block outline);
  - table and print disagree (a data error);
  - search (a local minimum);
  - new (described).
- **The evidence for the class:** numbers from the census, and what the picture shows.

### Task 3: Cause weights

The controller, or one agent. Sum the wrong lots per cause class over every incident of 20+ lots. Mid-size incidents
are classed by their energy split and a rule written down per class; mark which were classed by rule and which by
eye. The output is the ranking.

### Task 4: Oracles for the top causes

One agent per cause, at most two at once. Take the top three or four causes by lots. Where an oracle is cheap, build
it as an experiment, never over a stage's output. For example:
- **Label misread:** the table's code for the misread labels.
- **Placement:** the right lot for the labels that went astray, from the table.
- **Missing wall:** the table's border on the incident's lost line, at the walls' own scale.

Solve and score each oracle (`sfzmaps solver solve incidents-oracle-<cause> --walls ...`, or the inputs the oracle
changes). Report the every-lot measures against `final`.

### Task 5: Research for the top causes

One agent per cause, after its oracle.

- **Past tries:** from memory and the R&D notes. What was tried, and why it was dropped.
- **Literature:**
  - historic map digitization and vectorisation;
  - text and feature association, and leader lines;
  - graph-cut / MRF segmentation with priors from printed labels;
  - reading roadway and area symbols.

  Two to five relevant works, each with what it would change here.
- **Published libraries and workflows:** each with its license and fit.
- **Two or three candidate approaches.** For each:
  - what it changes;
  - its red team (how it could fail, which of the owner's rules it could break);
  - the measure, and the oracle's gain as its ceiling;
  - its synthetic test case (`one-problem-synthetic-tests`, numbers drawn from measured distributions);
  - what it costs to build.

### Task 6: Adversarial check

One agent per top cause's diagnosis, fresh, with no access to the diagnosing agent's conclusion until it has formed
its own. Given the incident's census numbers and pictures, does it reach the same cause? Disagreements go in the
report.

### Task 7: The morning report

The controller writes the record (`docs/research/joint-solver/notes/2026-11-16-big-incidents.md`, committed locally)
and publishes the report page. Pictures are embedded, and the page stays under 16 MB. The controller also:
- updates the memory handoff;
- leaves the code reader's Task 4 result beside it, if it has landed.

## Ledger

`.superpowers/sdd/2026-11-16-big-incidents/progress.md`, first line: this plan's path. Each task's dispatch, result
and every ruling goes in it.
