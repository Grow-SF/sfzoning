# The street-and-block finder: the first whole chain (2026-11-23)

The first end-to-end run of the finder designed in
`docs/superpowers/specs/2026-11-23-street-and-block-finder-design.md`: one network that looks at a scan and gives, per
spot, three answers (street or not, a block edge, a zoning border); areas closed from the first two by a watershed.

Every number here comes from 14 hand-marked **test** crops (651 blocks) that no run trained on. The pictures named are
local (`pipeline/data/find/`, not in git); they are the model's **results**, not training images.

## What was trained on

- Six sheets labelled automatically from the lots of their year and the April 2000 table:
  `2000-digital-zoning` p05, p07, p11; `2000-digital-height` p07; `2000-digital-su` p10; `1980-1990-zoning` p07.
- Nine hand-marked training crops, all on hatched maps: 1921 x4, 1929, 1935 x2, 1948 x2.
- **No hand crop of 1960-1980** (the lined grey maps): that era is judged without ever being taught.
- Ground under the test and choosing crops is left out of training in every scan. Three training crops lose part of
  their area to that: 29%, 9% and 8%.
- The step kept and the one setting of the area step (the depth) are chosen on 8 **choosing** crops, never on test.

## The four runs

| Run | What differs | Training time | Step kept | Depth |
|---|---|---|---|---|
| A `a-three-answers` | the design: three answers, print-and-scan noise on the tiles | 21 min 42 s | 2500 | 23/255 |
| B `b-no-borders` | the zoning-border answer gets no loss | 20 min 6 s | 2000 | 6/255 |
| C `c-unbroken` | the unbroken-line terms on hand crops' edges | stopped | none | none |
| D `d-no-noise` | as A, without the noise | 25 min 2 s | 3000 | 57/255 |

Run C was stopped by hand after 30 minutes: 250 steps took 12-13 minutes against under 2 for the others, so it would
have hit its time limit near step 1750 with no model. It has no result.

## Where the street and the block boundary are

The owner's measure of this result (2026-11-23): "It's enough to know where the street and block boundary is"; a block
read as several boxes along a zoning border is acceptable.

Per era: the share of hand street px the model calls street; the share of hand block px it calls not street; and, from
each hand block-boundary px, the distance to the nearest px the model calls a block edge (median, and the distance
that holds 90% of them). "Calls" is a chance over one half.

| Era | Run | Street called street | Block called not street | Boundary to model edge: median | 90% within |
|---|---|---|---|---|---|
| 1948, hatched colour (134 blocks) | A | 0.955 | 0.984 | 2.4 ft | 22 ft |
| | B | 0.947 | 0.989 | 2.4 ft | 37 ft |
| | D | 0.949 | 0.991 | 2.4 ft | 7 ft |
| 1921-1935, hatched grey (190) | A | 0.927 | 0.990 | 2.8 ft | 60 ft |
| | B | 0.927 | 0.992 | 2.8 ft | 103 ft |
| | D | 0.936 | 0.992 | 2.7 ft | 41 ft |
| 1960-1980, lined grey (174), never taught by hand | A | 0.940 | 0.970 | 0.0 ft | 14 ft |
| | B | 0.946 | 0.972 | 0.0 ft | 16 ft |
| | D | 0.878 | 0.986 | 2.7 ft | 21 ft |
| 1980s-2000, lined 1-bit (153) | A | 0.988 | 0.963 | 0.0 ft | 3 ft |
| | B | 0.987 | 0.963 | 0.0 ft | 4 ft |
| | D | 0.990 | 0.962 | 0.0 ft | 0 ft |

What the pictures show (run A, `runs/a-three-answers/look/test/`; looked at: 1921 Section 12, 1935 downtown, 1960
sheet 7, 2000 zoning p07):

- Streets come out as solid bands, including Market Street's diagonal and the downtown dead-end alleys.
- Block edges come out as thin, clean outlines that do not follow the heavy zoning borders.
- On the lined maps the model was never shown a 1960-1980 sheet by hand and still places streets and edges there.

**The weak spot is boundary with no street beyond it.** The long tail on the hatched grey maps is one crop, Section 12
at Visitacion and Sunnydale: 41% of its hand boundary px lie over 30 ft from any model edge (1929: 4%; 1935: 5%), two
thirds of them on or beside open land. On the picture (`look/owner/far-edges-1921.png`) the missed boundary is both
lines of the two-line roads, the sides of blocks that face open land instead of a street, and the short lines closing
street ends. Block sides that face a street are found.

## Blocks

`found`: one closed area holds over half of the hand block and over half of that area is the block. That count passes
a block split 60/40, so the stricter `whole` is beside it: the biggest area holds at least 0.9 of the block and the
block is at least 0.9 of that area.

| Run | Found | Whole | Cut up | Joined | Lost | Extra areas on hand street |
|---|---|---|---|---|---|---|
| A | 578 (0.89) | 331 (0.51) | 66 | 6 | 1 | 172 |
| B | 603 (0.93) | 425 (0.65) | 42 | 5 | 1 | 449 |
| D | 593 (0.91) | 357 (0.55) | 46 | 10 | 2 | 291 |

Whole blocks by era (A / B / D): hatched colour 81 / 86 / 88 of 134; hatched grey 131 / 134 / 111 of 190; lined grey
51 / 96 / 91 of 174; lined 1-bit 68 / 109 / 67 of 153.

- **Blocks joined across a street are rare** (5 to 10 of 651): the failure that would move a block boundary.
- **Most blocks that are not whole are split along a zoning border** on the lined maps (picture:
  `look/owner/runA-vs-B-1960-areas.png`). The owner accepts that. On the hatched maps 30-40% of blocks are not whole
  either, with no zoning border there; what splits them has not been looked at.
- The found shares the depth is fitted on (the mean over eras) are 0.89 / 0.93 / 0.91 and hide the splitting. A count
  of whole blocks, or of joins alone, is the one to fit on next time.
- The extra areas (172 to 449) are closed areas lying mostly on hand street. What they are has not been looked at.

## The zoning-border answer

Read cut by cut (a cut is agreed with when its statement's chance is over one half), on the two test crops that have
table labels (2000 zoning p07, 2000 height p07):

| Run | "A zoning border crosses" agreed | "None crosses" agreed |
|---|---|---|
| A | 512 of 8,128 (0.06) | 70,004 of 71,577 (0.98) |
| D | 2,681 of 8,128 (0.33) | 69,631 of 71,577 (0.97) |

On made-up towns drawn from these sheets' measures the same network and loss agreed with 0.79 of the "crosses" cuts.
On real sheets the answer traces the heavy lines only faintly. The cause is not known. Without noise it is stronger.

**Does teaching the zoning-border answer keep the block edge off zoning borders?** No difference here. Of the table's
zoning-border cuts that lie inside a block (no street on the cut; 4,933 cuts), the block-edge answer says "an edge
crosses" on 285 (0.058) in run A and 278 (0.056) in run B. Across a street it says so on 0.99 in both. The block edge
already ignores a zoning border inside a block about 94 times in 100, with or without the third answer; and run B
holds more lined-map blocks whole than run A.

## Noise

Asked for by the owner. One run each, so small gaps mean little.

- On the era never taught by hand (1960-1980), noise helps: streets 0.940 with it against 0.878 without, the boundary
  90% within 14 ft against 21 ft.
- On the eras that were taught, the run without noise places edges a little closer (90% within 7 ft against 22 ft on
  1948; 41 ft against 60 ft on 1921-1935; 0 ft against 3 ft on the 1-bit maps).

## What this run does not show

- Whole sheets. Everything here is crops; reading a page and handing areas to the sheet warp (#69) is not built.
- Anything about lot-level reading. The finder gives streets, block edges and areas; no district is read.
- Whether the hand labels' ragged edges matter. The labels were straightened to within 2 px; the model's own edges
  come out clean in the pictures.
- A second seed. Each run was trained once.

## What to do next (proposals, not decisions)

1. Teach boundary that has open land beyond it: hand crops like Section 12 in the training set (it is a test crop
   today), and check whether the two-line roads follow.
2. Hand-mark a few 1960-1980 crops for training; that era is carried by the 1980s-2000 sheets and the noise alone.
3. Fit the depth on joins and whole blocks rather than on `found`; look at what the extra areas and the hatched maps'
   splits are before changing the area step.
4. Decide whether the zoning-border answer stays in the model: it does not help the block edge and barely works on
   real sheets as trained. Finding out why it works on made-up towns and not here comes first.
5. Read whole sheets and hand the streets and block edges to the sheet warp.

## Reproducing

```
cd pipeline
uv run sfzmaps find labels --series 2000-digital-zoning --sheet 2000-digital-zoning-p05 --sheet 2000-digital-zoning-p07 --sheet 2000-digital-zoning-p11
uv run sfzmaps find labels --series 2000-digital-height --sheet 2000-digital-height-p07
uv run sfzmaps find labels --series 2000-digital-su --sheet 2000-digital-su-p10
uv run sfzmaps find labels --series 1980-1990-zoning --sheet 1980-1990-zoning-p07
uv run sfzmaps find train --run a-three-answers --sheet 2000-digital-zoning/2000-digital-zoning-p05 ... (the six sheets)
uv run sfzmaps find count --run a-three-answers
```

Runs B and D add `--without-borders` and `--no-noise`. The street, boundary, whole-block and A-against-B figures above
were made by one-off scripts on the runs' saved reads (`runs/<run>/reads/test/`); they are not commands yet. $0.
