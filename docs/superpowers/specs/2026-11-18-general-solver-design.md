# A general solver for every San Francisco drawing style

## Why

The joint solver assigns a zoning code to every lot of a scanned zoning map, from the printed labels and from "walls":
the odds, read from the ink, that a district border runs between two neighbouring lots.

- **Where it already works.** The wall model was fitted on the 2000 drawing against the April 2000 zoning table, and
  it works on maps drawn in the same style.
  - 2000: 35,455 wrong lots on the published map, 4,543 from the solver.
  - 1996-2000: 38,507 wrong lots on the published map, 18,651 from the solver.
- **Where it fails.** The model does not carry over to older drawing styles.
  - 1960-1980: printed with thinner pens in another drafting style. Scaling stroke widths lifted 1960 from keeping
    41% to 81% of its own printed labels, against 93% for today's map. On the owner's first 56 answered labels, today's
    map was right where the solver was wrong 19 times, and the reverse 4 times.
  - 1921-1948: districts are drawn as hatching patterns with a legend, not as lines and labels.

The owner wants one general solver for all three styles. It should infer each scan's drawing parameters (line
widths, hatching) from the scan's own statistics, and be trained on the 2000 table plus the owner's answers on the
older maps. Owner: "can we train a general solver on the set of these three and have it infer the parameters of line
width, hatching, etc based on the statistics observed in the scan itself?"

The literature behind this design is in `docs/research/joint-solver/notes/2026-11-18-general-solver-literature.md`.

## The three drawing classes

| Class | Series | Evidence | Labels for training and scoring |
|---|---|---|---|
| A, modern | 2000-digital, 1996-2000, 1990-1996, 1980-1990 (zoning; heights later) | lines + printed codes | the April 2000 table (2000; 1996-2000 for scoring) |
| B, mid-century | 1960, 1964, 1968, 1972, 1975, 1980 | lines + printed codes, thinner pens | the owner's answers on 1960 |
| C, early | 1921, 1929, 1935, 1948 | hatching + legend | the owner's answers on 1921 |

The special-use maps are out of scope: none of their sheets were ever read.

## Goals

1. **One wall model for the three classes, fitted jointly.**
   - The data: the 2000 table's lot pairs, plus the owner's training answers on 1960 and 1921.
   - The weights are shared. Each class may move a few of them (partial pooling). A class gets only as many
     adjustments as its border examples support: roughly ten border examples per fitted number.
   - Keeping the 2000 pairs in the fit keeps 2000 from getting worse. Plain fine-tuning from the 2000 weights does
     nothing here, because the fit is convex.
2. **Each scan measured on its own terms, with no labels.**
   - Line drawings: the evidence is expressed relative to the scan's own lot-line widths (the reference comes from the
     same kind of ink), so a thin pen and a thick pen read the same. A whole-page average is not a usable reference: it
     cost 1996-2000 4,195 lots.
   - The share of lot pairs that are district borders is re-estimated per scan without labels (prior-shift
     estimation).
3. **Hatching evidence for class C.**
   - Each lot's district: the legend swatch whose hatching (orientation, spacing, density) the lot's interior best
     matches, measured per scan from that scan's printed legend.
   - Walls: a change of hatching across a border.
4. **The owner's answers.** The owner will give hundreds of answers per map, on 1960 and on 1921. There are two kinds:
   - Training: small patches of neighbouring lots, answered whole ("every lot in this patch is R-1-D").
   - Scoring: random lots, held out by area and never trained on.

   The pages extend the existing label-check builder (branch `label-checks`) and keep its owner-requested design:
   - the reading drawn on the scan beside the mark;
   - the keyboard-shortcut bar at the bottom;
   - one case at a time, answers saved as they are tapped.
5. **One measure.** Every version of the solver is scored the same way:
   - against today's map, on the owner's held-out random answers for 1960 and 1921, compared lot by lot, with
     intervals from resampling blocks;
   - on the April 2000 table, for 2000 and 1996-2000: the six measures, none worse beyond noise.
6. **Publishing.** When the general solver beats today's map on a class's answered map, it replaces today's maps for
   every year of that class. Owner: "when the general solver starts winning, it should replace as much as it can."
   Class A already publishes from the solver (2000, 1996-2000, 1990-1996, 1980-1990).

## Non-goals

- Our own text reader: parked. Labels come from the saved Terra/Sol reads for now.
- The special-use maps.
- Refitting all 463 wall weights per class.

## Anti-goals (a version that does any of these fails, even if it scores well)

- **A hand-set cutoff:**
  - no picked thresholds, windows, caps or grids;
  - decisions come from probabilities, fitted models and the most-likely answer.
- **The 2000 table's codes in any solve other than 2000's:** only its lot shapes may be used elsewhere.
- **Learning from our own solves:**
  - no pseudo-labels;
  - no labels taken from a neighbouring year's solve;
  - no priors from earlier solves.
- **Any paid or LLM call.**
- **2000 getting worse on the table.**
- **Publishing a class whose map does not beat today's** on the owner's held-out answers.

## Constraints

- $0.
- Stages are built with `sfzmaps build` under the pinned pixi tools and recorded in `dvc.lock`.
- Outputs are byte-identical from run to run: no times or other run-varying values in an output.
- Keep the load under about 80 (one thread per worker), and run one GPU job at a time.
- The owner merges and ships.

## What exists to build on

- **Per-series solver stages** for 17 series, plus `solver-transfer` (the 2000 models applied elsewhere):
  `9da3fb8` and after.
- **Stroke-width scaling** for drawings other than 2000's (`92d918e`).
- **Label-free matching of each wall feature's distribution** to 2000's on the 1960-1980 editions: running now; its
  result goes into this design's first step.
- **The check pages,** published, each saving answers to its own collection:
  - 1960 labels (`answers-1960`);
  - 1960 lots (`answers-1960-lots`: 75 random and 225 where the maps disagree);
  - 1960 low-confidence labels (`answers-1960-lowconf`);
  - 1921 lots (`answers-1921`).
- **The review page of every year's differences:** https://claude.ai/artifact/Hw94SYjGWcFkfd1CmcFbeY

## Left to the builder

- **The form of the joint fit:** feature copying or a penalty toward the 2000 weights. Also which features get
  per-class adjustments, within what the answers support.
- **The hatching texture measures** and how a lot's match to each legend swatch becomes a probability.
- **The patch pages:** patch size and shape, how patches are chosen (spread across the city, not one area), and how
  many of each kind of question to show first.
- **The order of work.** Class B first is the obvious start, since its evidence is lines like 2000's.
- **Synthetic redrawing** of known answers in each era's style, to learn how weights shift with pen width or hatch
  density: optional, only if the measured results call for it.
