# The street-and-block finder: one model that names every kind of line

**Status:** approved by the owner 2026-11-23 ("approved, write the plan"). Written from the conversation of that
day, after the line-by-line audit (`docs/research/hatch-cnn-code-audit-2026-11-23.md`), a literature review
(`docs/research/street-and-block-finder-literature-2026-11-23.md`) and a builder's check. Nothing is built.
Lines marked *(my call)* are the writer's, not the owner's, and can each be changed.

## Intent and why

The owner, 2026-11-23:

> "What I want to do is detect the blocks and the lines all in one step so that the block finder and the street
> finder don't get confused by ... printed thick-line zoning borders. It should be able to detect the hatched blocks
> and the empty blocks through the same model, as well as detecting the streets."

His decisions that day:

- **One model that names every kind of line, zoning borders included** (approach 1 below).
- **One model, and it is never trained on a label that might not match the print.** ("yes sounds good", to that
  wording. It replaced a first draft that trained on today's map everywhere and kept five leave-out copies of the
  model to undo the damage.)
- **1921-1948 print no zoning borders:** "there are no zoning borders there; only hatching."
- It is built on branch `hatch-cnn` ("Let's build it here") and handed to placement (issue #69) from there.

It is step 2 of the process he set out the same day for every sheet of every series: frame removal, **street and
block finding**, warping and stitching, publishing. Two things wait on it:

- **Placement (#69)** fits today's block sides to the printed lines. It needs the block lines without the zoning
  borders: a fit on 2000 p05 pulled block lines onto thick zoning lines.
- **Reading the hatched maps** (`2026-11-22-print-decides-parcel-splits-design.md`) starts from printed areas. On
  11-22 the owner looked at the areas the branch's street finders give on Section 12 and said "we're back to street
  finding": street ends ran on past the thin line that closes them, blocks drawn only as an outline were not found,
  and roads drawn as two thin lines came out in fragments.

## What the maps draw (looked at on one crop per era, 2026-11-23)

| Maps | Scan | Block lines | Zoning borders | Else |
|---|---|---|---|---|
| 1921, 1929, 1935 | 8-bit grey, 2.7-5.4 ft/px | thin outline; white street between | none: a district is a fill pattern | dead-end alleys inside blocks; roads on open land as two thin lines; street names |
| 1948 | colour | thin outline | none | printed dimensions between blocks; institution names |
| 1960-1980 | 8-bit grey, about 2.7 ft/px | thin (1960) to medium (1975) | one heavy line, down the middle of a street, beside a block's edge, or through a block with jogs | large code labels |
| 1980s-2000, heights and special uses too | 1-bit, about 4 ft/px | thin | heavy, many jogs along lot lines, often drawn on or right beside a block line | large labels over the lines |

Two things follow. **A spot can be both a block edge and a zoning border** (the border is often drawn on top of the
block line), so these are separate yes/no answers, not one pick. And on 1975 the border is only a little heavier
than the block line: **what a line is has to come from what is around it, not from its thickness.**

## Approaches considered

1. **One model that names every kind of line.** Chosen by the owner.
2. One model that knows only street and block (what the branch has). Nothing tells it what a zoning border is, and
   placement still gets no "block lines without border lines".
3. Separate finders in a row (today: a street finder here, border evidence inside the 2000 solver, a text detector,
   the frame finder). Each has to guess what the others know.

## What the finder is

One network that looks at a scan as stored and answers, for every spot inside the frame cut:

- **street or not:** is this ground a street (paper between blocks, a dead-end alley, the strip between a road's two
  lines), whatever ink lies on it;
- **block edge:** does a printed block boundary run here (an outline, the line that closes a street end);
- **zoning border:** does a printed zoning border run here. The borders on the height and special-use maps are
  drawn the same way and count as the same answer.

Each answer is a chance, and the three are independent. From them come the **printed blocks**: the closed areas the
edges and streets enclose, hatched or empty alike (below).

**One model reads every sheet, and it sees only the scan and the frame cut.** No placement, no lots, no saved
reads, no series name, no scale, and no "which way is up": it is trained on every quarter turn of each scan (a
lossless turn), so a sheet scanned on its side is read as it is. That is what lets it read a sheet before the sheet
is placed, and a map of another city.

## What it learns from: only labels that match the print

The audit found that a model trained on today's map draws today's streets on old maps where the print differs, and
a 2025 study in the literature found the same and fixed it with hand labels. So today's data is used as a label
only where it is the map's own year. Labels are used in training only.

| Maps | Street or not, and block edge | Zoning border |
|---|---|---|
| `2000-digital-zoning`, `-height`, `-su` (45 sheets) | The lots of the map's year, joined into blocks, and their outlines; street is the strip round a street centre line between blocks | The April 2000 table's zoning, height and special-use columns: for two neighbouring lots, "a border runs between them" if their codes differ, "none does" if not |
| The other 1980s-2000 series (about 115 sheets) | The lots of the map's year (the lot records reach back to the late 1980s) | Hand crops only. Borders moved between 1980 and 2000, so the 2000 table would teach borders these maps do not draw |
| 1960-1980 | Hand crops only | Hand crops only. (The published maps' districts are not used: on the one 1975 crop looked at, the published map changes district inside blocks where the scan prints no line; cause not tested) |
| 1921-1948 | Hand crops only | "No zoning border" on every spot of every sheet, which is what these maps print (the owner, above). It teaches that a thin outline or the edge of a hatched area is not a border |

What the looks showed: on the 2000 crop the block outlines sit on the thin lines and the table's code changes sit on
the heavy lines. On the 1921 crop at Visitacion and Sunnydale, today's data has two lots where the print draws ten
outlined blocks and two double-line roads.

The labels from lots are laid on each scan through the placement the sheet already has. Each scan is labelled on
its own; the branch's rule that gives a block to one scan is not used.

*(Added while planning, 2026-11-23.)* Lots alone cannot say which ground between blocks is a street and which is
the bay or a park, without a picked width. So on these series street is the strip round a street centre line
between blocks (the branch's street labels, with one set of widths for all streets and no 95% test). The centre
lines are today's: one that crosses a block of the map's year labels nothing, and ground with neither a lot nor a
centre line has no label.

**A zoning-border label from the table is a statement about a place, not about ink.** The table says where a border
must be (between these two lots, or somewhere in the street between these two blocks), not which pixels it is. The
model is asked for "some border on the short cut across here" or "none on this cut", and has to work out which ink
that is. Block edges cross the same cuts whether the codes differ or not, so it cannot answer with the block line.
*(My design. It has no precedent in the repo, so it is tested first, alone, on a made-up case and one real sheet.
Past attempts to link ink to lot borders by rule failed four times; this lets the model learn the link.)*

**Which placement carries the lot labels** *(my call)*: the sheet warp's (`data/warp`), which since the 11-14
decision is the placement everything else uses. It starts from the old corner matching and then slides block sides
onto nearby printed strokes, so where it slid a side onto a zoning border, the edge label sits on the border. Stage
1 draws the labels through it and through the uncorrected placement on border-heavy 2000 sheets, and the one that
sits on the block lines is used. The pictures made while writing this used `data/warp`.

**Nothing in the labels is picked.**
- How far a printed line may sit from its lot label comes from a spread that is measured or fitted, not a band.
- The share of table borders the drawing does not show is fitted. The fit must give the same rate from any starting
  value (the branch's rate fit does not: `hatchnet/train.py:135-148`).
- The three answers' losses add as one likelihood. No hand weights between them.

### Hand labels

Drawn by me, shown to the owner on the scan before anything trains on them, kept in git under `pipeline/inputs/`
like the frame finder's. *(My call, on that precedent.)* They are now the only teacher for every map before the
1980s, so there are many more of them than in the first draft.

- **How:** on a crop I mark seed points and closing strokes; the paper connected to a "street" seed is street,
  what the lines enclose is block. Zoning borders are traced the same way from seeds on the heavy line. The file
  holds the seeds and strokes; preserved code makes the labels from them. A drawing aid for labels only: nothing
  like it runs when a sheet is read.
- **Three sets:**
  - **test crops**, drawn first, used for nothing but the pictures and counts shown to the owner;
  - **choosing crops**, used only to choose the training step kept and the one setting of the area step;
  - **training crops**, the only hand labels the model learns from, added to wherever the pictures show misses.
- **The ground under the test and choosing crops is left out of all training,** in every year's scan. *(1921 and
  1929 draw the same streets, so a training crop on one would give away a test crop on the other.)*
- Each set has crops from every era and every scan style, chosen for what is hard (outline-only blocks,
  double-line roads, street ends, alleys, hatching of each kind, borders beside block lines, labels over lines) and
  for plain ground.
- Each hand-drawn area is marked **block** or **open land** (land with no outline of its own), so open land is never
  counted as blocks that failed to separate.
- My reading of the maps goes to the owner through these pictures: the line closing a street is a block edge; a
  road drawn as two lines is a street; railway lines, shorelines and contours are neither edge nor border. He
  corrects the labels, not a description.

## The printed blocks

A block is a closed area. The branch makes areas by cutting the street chance at one half and taking connected
pieces, so one missing pixel joins two blocks.

- **Which areas are separate is closed by construction.** The chance that a spot is "street or block edge" is
  treated as a height, and the areas are its basins (a watershed). A basin's rim is always closed, so a faint
  stretch of an edge still separates two blocks. This is the step the map-reading literature uses, and there it
  mattered more than the choice of network.
- **How far a block reaches inside its basin** is the ground where "not street" is the more likely answer. That
  trims the block's outline; it cannot join two blocks.
- A road drawn as two lines is not a closed shape; the street answer between its lines is what separates the land
  on either side.
- **The step has one setting:** how deep a dip in a rim must be to count as a gap. It is fitted on the choosing
  crops by the count below, never picked. *(Still a cut level: a weakness, listed below. A step with no setting
  exists in the literature for nerve-cell images and is untried on maps; it is the fallback if this one fails in
  stage 3. No smallest-area setting: the scans differ fourfold in ground per pixel and the finder is not told the
  scale, so tiny areas are reported, not deleted.)*
- **The training step kept** is chosen by the same count on the choosing crops, not by a per-pixel loss, which the
  audit and the literature both found blind to what joins two blocks.

## Score and done

- **By eye first:** the three answers and the blocks on the real scan, whole sheet and close-ups, per era.
  Whole-sheet pictures mark where the training crops are.
- **On the hand test crops, per era, in plain counts.** Every printed block is exactly one of: **found** (most of
  it lies in one area and most of that area is it), **joined** to a neighbour, **cut up**, or **lost** to street.
  Areas found where no block is drawn are counted too. For zoning borders: of the places one is drawn, how many
  were found; and borders found where none is drawn.
- **The one number that chooses** (the step kept, the area setting) is the share of printed blocks found, on the
  choosing crops, each era weighing the same.
- **Block edges in feet:** how far the found edge sits from the hand-drawn one.
- **Done** when the owner has seen the pictures for every era and accepts them, the counts are reported, and
  Section 12's outlined blocks, double-line roads and street ends come out as drawn. No pass mark is set in
  advance: it is an experiment, and what is worse gets looked at.

## Built in stages, each shown on the real scan before the next

1. **The lot and table labels, drawn on the scan,** for the 1980s-2000 series, through both placements where they
   differ. (2000, 1960, 1975, 1921 and a height map were looked at while writing this; three of those pictures went
   to the owner with the first draft.)
2. **Hand crops:** the test set for every era, then the choosing set, then the first training crops. The owner sees
   them. This is the main labelling work.
3. **The crudest whole chain:** a few 2000-era sheets and the first hand crops of 1921 and 1960, one training, the
   three answers and the blocks pictured on test crops. This is where the border labels and the area step are found
   to work or not, before anything is polished. Two comparisons, each one training more: **with the zoning-border
   answer and without it** (does naming borders keep them out of the block edges? this is the owner's question, and
   no published work has tested it), and with and without the unbroken-line loss. It also shows how many hand crops
   an era needs.
4. **Every series.** More hand crops where stage 3's pictures show misses; pictures per era; counts on the test
   crops.
5. **Hand-off:** the block-edge chance to placement (#69); the areas to the hatched-map reading; the zoning-border
   chance stays unused until the solver's border step is next worked on (#68).

## Goals

1. One model and one process for every sheet of every series: no per-year path, no series or scale input.
2. Blocks, hatched and empty, streets, block edges and zoning borders from that one model.
3. A zoning border is never taken for a block edge, nor an edge for a border.
4. Nothing is taught that might not match the print, and nothing is counted on a crop the model trained on.

## Not in this design

- Reading the hatching or any district (the 11-22 design's later stages), and telling open land from blocks, which
  that design names. The finder gives areas; it does not say which kind each is.
- Placement itself, the warp, the frame finder, the 2000 solver, anything published. The 2000 map does not change.
- A lettering answer, or one for fill patterns. *(My call: the three answers already say "not an edge, not a
  border" over letters and hatching. Added only if the pictures show letters or hatch lines taken for edges.)*
- Wiring the zoning-border chance into the solver.
- **Today's map as a label on the older maps.** It is right over most of the city and wrong where streets changed.
  Adding it back is a later experiment, judged on the test crops, not part of this design.
- **A second turn.** #69 will place sheets from this finder's edges, and lot labels laid through those new
  placements would sit better on the print. The owner chose on 11-19 that finder and placement take turns. This
  design is the first turn; a second is its own decision, with pictures.

## Would make it a failure even if it scored well

- More than one model, or a model that is shown anything but the scan and the frame cut.
- A label used in training that might not match the print.
- A hand-set cutoff, window or special case in the reading; erasing or cleaning ink.
- A path that differs by series.
- LLM output when a sheet is read. (Saved reads may only ever be training labels, and this design uses none.)
- Learning from its own reads.
- A score shown before its pictures, or a count or a choice made on a crop the model trained on.

## Constraints

- $0. Deterministic: same inputs and device, same bytes. One GPU job at a time.
- Scans read at their own resolution; no resampling code of our own.
- Duplicate scans are never read (`inputs/<series>/duplicate_scans.csv`); other series are checked for duplicates
  before their sheets are used.
- Outputs under `pipeline/data/` (never committed), with the series in the path. Pictures and counts are kept;
  per-pixel maps of a superseded run are deleted. Free disk is checked before a full run (122 GB on 11-23).
- A new package and command (`sfzmaps find`), beside `frame` and `place`. It reuses the branch's network, and its
  street labels on the 1980s-2000 series only; it does not reuse the per-lot reader or the ownership rule.
- **The unbroken-line loss the owner liked (11-18) is kept as an option, not a default.** On this branch its pull
  toward today's street network drew streets the print lacks, and on scanned maps the literature found no gain
  from such losses on sheets not trained on. Stage 3 trains with it and without it and the pictures decide.
- No GPL-family dependency.

## What the literature says (sources in `docs/research/street-and-block-finder-literature-2026-11-23.md`)

- **Closing shapes matters most.** In the 2021 competition on finding blocks in scanned Paris maps, and the work
  after it, a watershed on the edge chance beat cutting at a level by a wide margin, and the choice of network
  mattered less. Labelling street ground did better than finding edges alone. This design does both.
- **Training on today's map reproduces our failure.** A 2025 study trained on modern labels and got roads and
  buildings the old map lacks; hand labels fixed it. This design follows that.
- **Losses that reward unbroken lines** helped on aerial roads; the one test on scanned maps showed no gain on a
  sheet not trained on.
- **Nobody has tested** giving a confusable line its own answer against leaving it unnamed. That is this design's
  own bet, and stage 3 tests it.
- **Sheets not trained on are where these methods usually fail.** The best published block scores across sheets are
  about half by their measure. This design has far more sheets and a kind of label they lacked, but the first
  pictures should be expected to show misses.
- Later options the literature supports, not in this design: drawing made-up sheets from today's data in each era's
  style (every label exact, borders included), and bending training crops slightly so the model sees more variety.

## Known weaknesses

- **Everything before the 1980s rests on hand crops.** What no crop shows, the model has nothing to learn from.
  How many crops an era needs is not known until stage 3.
- **Table borders come from 45 sheets of one drawing style.** Whether the zoning-border answer carries to 1960-1980
  and to the other 1980s-2000 series is unknown until pictured; hand crops are the only border labels there.
- **The table-border label is new** and may not train. Stage 3 exists to find that out cheaply.
- **The area step has one fitted cut level.** Fitted on labelled crops, not picked, but a cut level all the same.
- **The lot labels ride on the old placements,** so a block side the warp slid onto a zoning border carries a wrong
  edge label, and small shifts remain. Stage 1 looks for the first; the fitted spread covers the second.
- **The 1980s-1990s maps are labelled from lot records that reach back only to the late 1980s.** Where a block
  changed before that, the label is today's, not the print's.

## Left to the builder

- The network's size and schedule (starting from the branch's).
- The form of the losses and of the fitted spread and rate.
- File formats, and how seeds and strokes are stored.
- How many crops each hand set starts with, and how much ground round a test or choosing crop is left out.

## The calls I made, for the owner to check

1. Hand labels are drawn by me with a seed-and-fill aid, in three sets (train, choose, test), and the ground under
   the last two is left out of all training.
2. Lots are used as labels on all the 1980s-2000 series, but the April 2000 table labels zoning borders only on
   the three `2000-digital` series, the maps it is the table of.
3. Lot labels ride the sheet warp's placement, checked against the uncorrected one in stage 1.
4. No lettering answer and no open-land answer in this version.
5. The unbroken-line loss is an option tested in stage 3, not the default.
