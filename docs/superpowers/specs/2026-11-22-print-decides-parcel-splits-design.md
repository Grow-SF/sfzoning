# Read the print, then the lots: districts on the hatched maps

**Status:** approved by the owner 2026-11-22 ("I think it sounds right. Yeah let's give it a try."), after three red
teams. Stage 1 starts. Owner questions 2 and 3 (end) are not answered yet; neither blocks stage 1.

## Intent and why

The owner, 2026-11-22:
- On the maps before 2000 the printed map decides. "Read what's printed and then apply it down to the lot", not the
  other way round.
- Not pixel by pixel: "Looking at an individual pixel between two of those lines doesn't tell you that that's a 1R or
  a 2R space. What matters is the context around those pixels so you need to look at blocks as a whole. Of course you
  have blocks that contain multiple zoning codes, like the commercial strips along streets."
- "A small area that differs from everything around it should need strong evidence."
- Lot lines, modern and inactive, are guides to where a border is drawn, never overrides: "if I'm unsure if I should
  draw a line here or 10 pixels to the west and the lot line happens to be 2 pixels to the west, then I have a strong
  prior to draw it at 2 pixels to the west." Probabilities, not hard rules.
- "If the map is drawing a street, you should keep a street."
- The finished map is parcels, split where the print divides them.
- A solve step for the hatched maps is fine. The owner will answer on the label-check page.

**What this replaces in `2026-11-19-one-process-hatched-maps-design.md`:** its "Lots" and "Lot pooling" parts, "no
walls or solve stage", and its anti-goal "learning streets or blocks from today's map, or tracing blocks for
reading" (the owner now asks for blocks read as a whole and streets kept). The rest of that spec stands: one city
image, no sheet-ownership rule, one scan per section, the frame crop, and its rule for done.

## What is wrong today (measured on Section 12, `20180102100525362`)

- The reader answers lot by lot. Lot `6220002` (1.39 million px) got one code over ground 1921 printed as hatched
  blocks, streets and blank land.
- Islands: in blocks where at least 80% of lots read 1R, the reader gives 8% of the lots something else (about 1,800
  lots).
- Street white reads as 2R: a 20 px ring of white paper painted round a 30 px square of sure-1R hatching flips 24 of
  45 squares to 2R. 2R means "a block with nothing printed in it", so it cannot be read without knowing street from
  block.
- Single pixels are noisy: the reader is trained only through a weighted pooling per lot.

## Approaches considered

1. **Read printed areas, each one district or one of a few named divisions, solved with its neighbours.** Chosen.
2. Label pixels with a border cost, then cut parcels (draft 3). Rejected by the owner's ruling: it reads pixel by
   pixel; islands and street stubs are cheap in it.
3. Read lot by lot and smooth afterwards. Rejected: today's method with a patch.

This is the 11-18 design's "streets, blocks, pieces, lots" again. It differs from the abandoned `hatch-regions`
pipeline (hand-built texture measures and masks, tuned on scores with no pictures) in that the evidence is learned
and every stage is judged by pictures first.

## Built in three stages, each shown in pictures before the next

**Stage 1: areas, one district each.** Fixes the islands. Published per lot, no splits yet.
**Stage 2: divided areas** (strips, cross cuts, corners).
**Stage 3: ties between neighbouring areas, and lots split** where they span parts.

### Stage 1

**1a. Areas.** An area is ground enclosed by what the scan draws: streets, printed block outlines, and roads drawn
as thin double lines. Open land with no outline is its own kind of area.

What exists and how it does on Section 12 (measured by the third red team, picture `/tmp/rt3_areas.png`):
- The branch has three street outputs (v1, v2, v2c), each one model over all sheets with one 1,024 px cell in five
  never trained on. The planned printed-street finder of issue #69 is not built.
- With v2c, the hatched blocks inside `6220002` come out as 21 separate areas with their streets kept. But blocks
  bounded only by an ink outline are not cut off: one area is 19% of all area px and holds 42 of today's blocks, and
  merged areas are 43% of area px. v1 separates the outline-bounded blocks in that window.
- So the area finder is the main risk of this design. Stage 1 starts there: outlines and double-line roads as
  boundaries, with a synthetic case for each, and the picture of Section 12's areas goes to the owner before
  anything is read from them.
- Streets stay streets in everything after.

**1b. Texture evidence.**
- The reader's network also sees the area map, so blank paper inside a block and blank paper in a street are
  different inputs. The area map it trains on is made by models that did not train on that ground (shared folds), so
  the reader cannot lean on a finder that is better on its own training ground.
- Trained on ground, not through pooled lots: an in-area px's target is the code of the lot under it (reads through
  the read-error table, answers as exact, legend swatches). Each lot counts once, its px sharing its weight. Ground
  with no read has no target. A lot that lies in more than one area trains nothing: its one read cannot label mixed
  ground.
- Evidence for a piece of ground and a district is the sum over its px of (log probability minus log of the
  district's share in training), times one temperature fitted by likelihood on held-out areas. A single px is never
  an answer.

**1c. One district per area; lots take it.**
- Each area gets the district its evidence favours, with its probability.
- A lot takes the district of the area that holds its centre. Lots that lie in several areas, and lots mostly on a
  printed street (153 on Section 12 are at least 80% on street), are listed, not guessed (owner question 3).
- The least certain areas are shown to the owner first, most doubtful at the top. No cutoff decides the list.

### Stage 2: divided areas

- Named shapes, each with a frequency and a depth distribution: **a strip along one side for its full length**, **a
  cut across**, **a corner**. (Two free lines can "explain" one randomly flipped lot in every pure block, so free
  lines would rebuild the islands. On 109 divided blocks of sure lots, one straight line explains 84% and two 98%.)
- **Where the line sits:** a distribution over its offset from the nearest lot line (modern or inactive), with the
  spread of the placement error of the placement in use. About a third of real dividing lines cross lots (a straight
  mid-block line behind jagged rear lot lines), so a line off the lot lines is allowed and paid for by evidence.
- A lot is on the side that holds its centre; a division needs a lot on each side.
- **The frequencies are not counted from the old reads** (two ways of picking trustworthy reads disagree by a factor
  of two: 52% or 26% of blocks are one district). They are fitted by the held-out likelihood of the texture evidence
  and checked against the owner's answers on whole blocks (owner question 2).
- Open land is never divided by these shapes.

### Stage 3: neighbours, then lots split

- Facing areas and parts are tied across streets, and a strip to the strips of the next areas along its street. A
  tie is how much more often two districts face each other than chance: P(a, b) / (P(a) P(b)), fitted as in stage 2.
- All areas are solved together: each starts at its own best, then they re-choose in a fixed order until nothing
  changes. A case with a weak strip running several blocks is checked against the exact answer for a chain.
- Real lone districts exist but are rare (3 to 5 on Section 12), so the ties must leave them possible; they are
  reported by name every run.
- **Lots split:** a lot whose ground lies in parts with different districts is cut along the straight dividing line
  or the fitted straight side of the area, never along a wobbly edge. Whether it truly spans both is a probability:
  the chance of that overlap depth from placement error alone, against the chance a lot spans two parts at all
  (from the owner's answers). A printed street inside a lot is a piece marked street.
- Output per piece: lot id, piece number, polygon, district or street, probability.

## Cases first (each stage's cases before its real run)

Synthetic, from real Section 12 crops, one thing each, numbers from the measurements above:
- Stage 1: a block bounded only by an outline is its own area; a double-line road bounds an area; a small hatched
  area ringed by street reads 1R (24 of 45 flip today); a block with a few odd lots is one district.
- Stage 2: a C strip one lot deep (28-38 px) on a 1R block, its line on the lot line; a straight mid-block line
  behind jagged lot lines; a pure block with one odd lot stays pure.
- Stage 3: the weak strip run; a lone real district survives; a block shifted by the placement error splits no lot;
  a parcel over several printed blocks, streets and open land comes out as those pieces.

## Score and done

- The 11-19 spec's rule for done stands (the 240 random lots, each predicted by models that never saw its area,
  against the published map).
- The owner's whole-block and point answers, held out by area.
- Reported every run: areas merged or shattered against today's blocks; islands; areas divided; lone districts; lots
  listed; lots split.
- Pictures at the scan's own resolution at every stage: the areas; each area's district and shape; the lots.

## Constraints

- Deterministic, $0, no LLM output at inference. No priors from this reader's own predictions.
- Every prior is fitted or measured, held out by area. No hand-set cutoffs.
- The frame finder and the 2000-and-later maps are not changed.
- First on Section 12 alone; then on the 11-19 city image when 1921 has one.

## Known weaknesses

- Everything rests on the area finder, and today's merges 43% of area px on Section 12.
- Shapes outside the three named ones go to the owner's list.
- Stage 1 alone leaves `6220002` unsplit (listed); it is stage 3 that divides it.
- It is a lot of building: an area finder, a retrained reader, a fitter, a solve, a lot splitter, new answers, and
  later a publish format. The stages are there so each one ships something better on its own.

## Not in this design

- How the published data and the site carry a split parcel or a street piece.
- 1929, 1935, 1948 (same process, later).

## Left to the builder

- The network's size and schedule; the temperature fit's form.
- How an area's sides are taken from its boundary for the strip and cut shapes.
- File formats.

## Questions for the owner

1. **The stage order.** Stage 1 (areas, one district each) fixes the islands and ships first; strips second;
   neighbour ties and split lots third. Yes?
2. **Answers on whole blocks.** On the label-check page: "this block is all one district" or "divided, like this",
   as well as points. It is the cheapest data for how often blocks are divided. Yes?
3. **A modern lot wholly on a 1921 street.** Publish it as street with no district? That would replace the 11-18
   ruling (white is 2R, no unzoned answers) for ground the map draws as street.
