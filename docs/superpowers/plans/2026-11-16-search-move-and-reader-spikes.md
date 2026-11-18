# The Compound Search Move, the Max-Flow Swap and the Reader Spikes: Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development. Steps use checkbox (`- [ ]`) syntax.

**Status, 2026-11-16 ~12:30:**
- **The order is agreed with the owner:** search first, the reader spikes alongside.
- **The owner was asked "Shall I start?" and hasn't said go yet.** Wait for an explicit yes before dispatching.

**Goal:**
- adopt the compound search move in the solver, behind acceptance criteria written before it is measured;
- replace PyMaxflow (GPL) now that the code is Apache-2.0;
- alongside, run four cheap measurements (S0–S3) that decide the reader's rebuild.

**Why this order:**
- The overnight census put search misses at ~2,900 wrong lots and label misreads at ~180.
- The red team showed that while the search misses, label improvements are hard to see in the map's numbers.

The reader's rebuild (a CTC recogniser, below) comes after, as its own spec revision and plan.

**Sources:**
- `.superpowers/sdd/2026-11-16-big-incidents/search-report.md` and `search-adversarial.md`;
- the morning report https://claude.ai/artifact/Xo8GukkZH1ZsnCe9847S4y and
  `docs/research/joint-solver/notes/2026-11-16-big-incidents.md`;
- `.superpowers/sdd/2026-11-16-labels-from-the-scan/redteam.md` and `survey.md`;
- the experiment scripts in `pipeline/scripts/incidents/search_*.py`. These are local commits, not pushed, and they use
  PyMaxflow.

## Global Constraints

- **Code is Apache-2.0** (owner, PR #63, merged to main 11-16). No GPL-family copyleft in shipped code: PyMaxflow and
  thinqpbo are out. Pretrained weights or data with non-commercial terms stay out of shipped code.
- **No LLM output in the pipeline** when it runs on a map. Terra's and Sol's saved outputs train and validate only.
- **Acceptance criteria are fixed before measuring** (owner agreed: "defining your acceptance criteria to avoid bias").
- **Scoring sets are noisy:** intervals, measured set error, and ties broken by looking (`validation-sets-are-noisy`).
- **Value before verification:** quality first, repeat runs after.
- **Load near 20; one GPU job at a time; deterministic.**
- **Commit locally;** the controller pushes to #62 only with the owner's go. The owner merges.

## Task S-A: The max-flow swap (before the move)

- **What:** replace `maxflow` (PyMaxflow) in `pipeline/src/sfzmaps/solver/search.py` `expansion_moves` with OR-Tools
  max flow (Apache-2.0), or SciPy's `maximum_flow` (BSD), integer capacities with a bounded scaling. Pick the faster
  after a timing on one quarter.
- **Check:**
  - the cut energies equal PyMaxflow's within the scaling bound on the four quarters;
  - the solve's six measures unchanged, or differences explained by tie-breaking among equal cuts, shown on a page;
  - the timing.
- **Then** remove `pymaxflow` from `pipeline/pyproject.toml` and the lock, update the README and AGENTS references,
  rebuild the solve, and commit `dvc.lock`.

## Task S-B: The compound move, behind criteria fixed now

**The move:** the census/search experiments' `compound`. A region takes its labels' favoured code; every touching region
holding that code offers each of its lots that lot's next code; a fusion (QPBO on the new max-flow) keeps only the lots
that lower the energy. Built into `search.py` with its synthetic cases first:
- the Mission-strip shape: two codes must change at once;
- the expansion-bound shape of #4, #14 and #18.

`starts-compound` and `all` are not adopted: one made more lots worse outside the incidents studied, the other is the
energy's pick at 4,850.

**Acceptance criteria (fixed 11-16, before the build):**
1. The solver's energy goes down in every quarter. No answer key is needed.
2. Wrong-lot incidents and labels-right are not worse beyond the measured noise: bootstrap intervals, and the table's
   by-eye corrections as its known error.
3. The same two checks on the `1996-2000-zoning` scan, a second scan of the 2000 drawing that Terra never read and the
   same table judges.

Wrong lots and the other measures are reported, not gated. If it passes all three, it becomes the default.

## Task S-C: The reader measurements (alongside, CPU, cheap)

Each decides part of the reader's rebuild:
- **S0:** solve with the existing template reader's saved posteriors as the only reading. Records:
  `scratchpad/codereader/task3/full_fix1*.records` (from the round-1 code) and the partial `full_fix2held.records`;
  or rescore a sample. This is the map's sensitivity to a no-LLM reader.
- **S1:** simulate misread and miss rates on the current readings, and plot the six measures. That gives "somewhat
  worse" (owner) a unit.
- **S2:** a small CTC recogniser spike, as in survey §4.6:
  - **three readers:** docTR's CRNN fine-tuned (baseline only, since its weights' training data is undisclosed); a
    CRNN from scratch; from scratch plus synthetic crops from the glyph templates, noise model and measured line
    overlays;
  - **training:** on crops at Terra's boxes in the training quarters, with Terra's strings as labels;
  - **scored:** on the held-out quarter and the owner's answers, with intervals, beside the c626e46 records and the
    96.7% LLM yardstick;
  - **also:** CPU determinism and crops per second.

  Terra's strings as training labels is how the letter shapes were already learned. The owner noted it "already works
  this way", so no new decision is needed.
- **S3:** word boxes grouped from the existing text chance (DBNet/CRAFT-style), with the axis from each word's second
  moments, against Terra's words: recall and axis error by kind.

## After S-A to S-C: the reader's rebuild (a separate spec revision and plan)

- **The direction from the red team and survey:**
  - keep our U-Net detector, with no new box or axis outputs (they would learn from noisy targets);
  - boxes from the segmentation, as in S3;
  - a CTC recogniser trained from scratch, giving an exact likelihood per candidate;
  - Weinman-style location priors (codes, nearby centreline names, block numbers);
  - "other text" as the leftover mass, and a direction classifier;
  - the template reader becomes the synthetic-crop generator.
- **Also in it:**
  - the walls' default "letter share" feature reads Terra's boxes (`walls.py:72,260`, `wallevidence.py:655`). It must
    switch to the detector for a no-LLM pipeline.
  - The label rule "real mass" becomes a per-box not-a-label probability in the energy.
  - A new map's first placement (today from LLM-read anchors) stays a later decision.
- **Superseded:**
  - `docs/superpowers/plans/2026-11-16-labels-from-the-scan.md` (P1–P6, written before the red team);
  - `2026-11-16-code-reader-read-along-the-word.md` (R1–R4);
  - the spec `2026-11-16-text-pipeline-from-the-scan-design.md` stands for its goal and rules, but its method sections
    (EAST heads, the template reader on boxes) are replaced by the direction above.

## Added 2026-11-16 evening (owner): toward publishing a better 2000 map

Owner: "It's okay to just be better. It doesn't have to be perfect. Ultimately we need to get to a point where we've
finished this work and we produce a map that we can publish because the current published map has a lot of errors."
The owner's answers on the path: finish this plan; turn the compound move on; score the published map; feed the
solve's codes into the published 2000 districts, highlighting where it differs from the published map.

- **Task S-D: the compound move on in the stage.** The owner waived criterion 3 for this ("Turn it on"). The stage
  run `final` solves with `compound_fusion=True`. Rebuild `solver-solve`, then commit `dvc.lock`. Expected to match
  the experiment `search-compound`: 4,543 wrong lots.
- **Task S-E: the published map scored the same way.** Load today's published 2000 map as a run
  (`solver/published.py`). Give its six measures beside the stage's, with the same paired block bootstrap as
  `criteria_check.py`.
- **Task S-F: the solve's codes in the published 2000 districts, with a page of where they differ.** Brief written
  after S-E, from how the 2000 series' districts reach the tiles. The owner checks the differences by eye before
  shipping. The owner ships and merges.
