# The compound search move, the max-flow swap and the reader spikes: results (2026-11-16)

Plan: `docs/superpowers/plans/2026-11-16-search-move-and-reader-spikes.md`. Built by subagents, each task reviewed.
All commits are local on `solver`, between 9358c9b and the head. The outputs are under
`pipeline/data/solver/experiments/`.

## Built

- **The max-flow swap** (c28529d..ed4cf98).
  - `solver/mincut.py` uses OR-Tools' `SimpleMaxFlow` (Apache-2.0). PyMaxflow (GPL-3.0) is gone.
  - The cuts equal PyMaxflow's on every expansion move of the four quarters (max |Δ| 2.3e-10).
  - SciPy's max flow was 26× slower; OR-Tools is about 2× PyMaxflow's time.
  - The stage's answer was unchanged.
- **The compound fusion move** (781e0fd..f33374b; on in the stage since d03ca5d, 7cce81a).
  - The compound proposal fused lot by lot with roof duality (`solver/qpbo.py`), accepted on the exact energy.
  - Acceptance criteria fixed beforehand:
    1. Energy lower in every quarter: passed (−24 to −95).
    2. Incidents and labels right no worse beyond noise: passed, both better (868 → 862 [−12.2, −0.5]; 94.47 →
       94.79% [+0.10, +0.59]).
    3. The 1996-2000 scan: not measurable, because the solver exists only for 2000-digital. The owner waived it
       ("Turn it on").
  - Wrong lots fell from 7,100 to 4,543 [−3,278, −1,793]. 14 lots broke, 12 of them the Mission tip lots the table
    keeps RH-1.
  - The tests are basin tests on five map excerpts (#1, #7, #9, #4, #14). The old moves stay wrong; the new move
    reaches the right answer. They are regression tests on real neighbourhoods, not single-cause synthetic cases: a
    block model could not hold the basin. #18 is a known miss (exact QPBO expansions fix it; not adopted).
- **The published map scored the same way** (`scripts/incidents/published_compare.py`; eaa812a..97a790a).

  | | Published 2000 map | The solve |
  |---|---|---|
  | Wrong lots | 35,455 | 4,543 |
  | Labels right | 65.5% | 94.79% |
  | Incidents | 3,482 | 862 |

  - The published map's wrong lots split into 31,967 wrong codes, 2,202 blanks and 1,286 lots with no matching
    lot.
  - The site's own corrections move this by about 5 lots.

## Measured (spikes; scripts in `pipeline/scripts/spikes/`)

- **S3, word boxes from the detector's text chance.** Chosen on the 2000 validation area and reported on the three
  test drawings: about 90% of zoning codes found (89.9% [88.2, 91.6] at level 0.8), about three quarters with an
  axis within 10°. Short, stacked and one-letter codes flip a quarter turn. The 2000 numbers are not held out: the
  `solver-text` detector trained on those sheets.
- **S2, the text reader** (the CTC recogniser, from scratch plus synthetic code crops).
  - On 662 answered labels with a Terra box, three seeds: top-1 0.956–0.973 and −log p 0.102–0.126, against the
    template reader's 0.949 / 0.186.
  - By the brief's rule it passes at every seed, provisionally, because the template reader sat at its grid's
    temperature edge. On the unflagged answers it ties.
  - It is about 800× faster (0.02 s against 16 s per label) and deterministic on CPU.
  - Owner: adopt it ("Why would I prefer to be 800 times slower...?"). docTR's pretrained baseline stays a little
    ahead, so better synthetic crops are the next lever.
- **S0, the template reader as the only reading** (against the old search's final): labels right 88.78% against
  94.47% (−5.69 pt [−6.99, −4.40]), incidents +116. Wrong lots are not distinguishable.
- **S1, the cost per point of reader error** (one point is about 19 labels; old search; t intervals over 3 seeds):

  | Reader error | Wrong lots | Labels right | Incidents |
  |---|---|---|---|
  | Confident misread | +1,132 [756, 1,508] | −0.82 pt | +14 |
  | Miss | +428 [235, 621] | −0.53 pt | +9 |
  | Misread at the template reader's own sureness | +462 [308, 616] | −0.53 pt | +9.5 |

  How sure a reader is when wrong matters.

## Rulings made by the controller on the owner's behalf

- **R1:** a reusable `mincut.min_cut` for expansion and QPBO.
- **R2:** the experiment scripts switched to it.
- **R3:** the stage rebuilt per task.
- **R4:** #18 named a known miss, not a test.
- **R5 / R5b:** spikes ran alongside the search work, with up to three agents live.
  - Load reached 50–130 at times: over the owner's cap. Lesson: one solve-heavy agent at a time.
- **R6:** spikes compare against a final made by the same search.
- **R7:** the move was off pending criterion 3. Superseded by the owner's "Turn it on".
- **R8 / R8b:** stages stale only for additive edits (cli.py flags, a paths.py constant, an added predict function,
  the generalised hold-outs after a byte-identity check of the training data, the lock files) were recorded
  without rerunning.
- **R9:** OR-Tools accepted at about 2× PyMaxflow's time.
- **R10:** synthetic cases test the basin, not the path.
- **R11:** S2 reported as both readings. The owner then chose the text reader.
- **R12:** the publish wiring moved to the next plan.

## Carried to the next plan

`docs/superpowers/plans/2026-11-16-text-reader-labels-and-publish.md`:
- **T0:** the scoring labels fixed.
- **T4:** `TOP_READING_FIT` refit or set to identity, and readings per label kept small.
- **T5:** resolve the 16 district stages stale since 11-02 before rebuilding `period-2000-digital`.
- **Later:**
  - re-measure the reader costs under the compound search;
  - more seeds at low error rates;
  - the table-free lower bound (TRW-S), offered.
