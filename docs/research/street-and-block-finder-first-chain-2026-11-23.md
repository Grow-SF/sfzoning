# The street-and-block finder: the first whole chain (2026-11-23, rewritten 2026-11-24)

The finder designed in `docs/superpowers/specs/2026-11-23-street-and-block-finder-design.md`, as it stands after its
first end-to-end runs: one network that looks at a scan and gives, per spot, **two** answers (street or not; a block
edge), and areas closed from them by a watershed. The design's third answer (a zoning border) was dropped on
2026-11-23 (the owner: "We want the blocks and streets"); nothing in the finder reads, makes or stores a zoning border.

Every number under "test" comes from 14 hand-marked test pieces (651 blocks) that no run trained on. Every number
under "choosing" comes from 8 other hand-marked pieces, also never trained on; recipes are compared there. The
block outlines marked by hand on those pieces are the **answer key**: they score a result and are never shown to the
model. Pictures named here are local (`pipeline/data/find/`, not in git).

## Where it stands

The owner's measure (2026-11-23): "It's enough to know where the street and block boundary is"; a block read as
several boxes along a zoning border is acceptable.

The run with both of the night's fixes (`resized-open-land`), on the test pieces:

| Era | Street ground called street | Block ground called not street | Real boundary to the nearest edge drawn: half within, 90% within | Edge drawn to the nearest real boundary: 90% within |
|---|---|---|---|---|
| 1948, hatched colour (134 blocks) | 0.984 | 0.986 | 0.0 ft, 4.9 ft | 4.9 ft |
| 1921-1935, hatched grey (190) | 0.925 | 0.993 | 0.0 ft, 41 ft | 2.8 ft |
| 1960-1980, lined grey (174), never taught by hand | 0.985 | 0.974 | 0.0 ft, 2.7 ft | 5.4 ft |
| 1980s-2000, lined 1-bit (153) | 0.989 | 0.966 | 0.0 ft, 3.4 ft | 6.7 ft |

- The edges the model draws lie on real block boundary in every era (90% within 3 to 7 ft).
- What is still wrong is boundary it leaves out, and nearly all of that is one test piece: Section 12 of 1921
  (Visitacion Valley), where blocks and roads stand in open land. See "Open land" below.
- Blocks joined across a street: 6 of 651.

## What changed since the first write-up, in order

1. **A final review of the whole branch** found a choosing piece lying on test ground (redrawn, two more trimmed; the
   code now refuses it) and that the zoning-border labels made from lots contradicted each other.
2. **The zoning-border answer was removed** (the owner's decision): model, training, labels, its made-up-town test.
3. **The trainer keeps its last step.** It used to keep the checkpoint with the best choosing score; that score
   swings between 0.46 and 0.98 from one checkpoint to the next, and in one run it kept step 250 of 3,000, an
   immature model.
4. **Training tiles are shown at every print size the finder will meet** (see "Print size").
5. **Three training pieces with open land were added** (see "Open land").
6. **The boundary measure got its other half** (edge drawn to nearest real boundary); the first half alone flattered
   a model that calls edges loosely.
7. **The area step's one setting is fitted on panoptic quality** instead of the share of blocks found, which could
   not see stray areas (see "Blocks").

## What is trained on

- Six sheets labelled automatically from the lots of their year: `2000-digital-zoning` p05, p07, p11;
  `2000-digital-height` p07; `2000-digital-su` p10; `1980-1990-zoning` p07 (street or not, "no block edge", and
  short statements that a block edge crosses a cut).
- Twelve hand-marked training pieces on hatched maps: 1921 x6, 1929 x2, 1935 x2, 1948 x2 (nine at first; three with
  open land added on 2026-11-24).
- **No hand piece of 1960-1980** (the lined grey maps): that era is judged without ever being taught by hand.
- Ground under the test and choosing pieces is left out of training in every scan.

## The measures

- **Street ground called street**: of the answer key's street px, the share where the model's street chance is over
  one half. **Block ground called not street**: of its block and open-land px, the share where it is not.
- **Real boundary to the nearest edge drawn**: from each answer-key boundary px, the distance in feet to the nearest
  px whose block-edge chance is over one half. Large values mean boundary the model leaves out.
- **Edge drawn to the nearest real boundary**: the same the other way. Large values mean edges that are not boundary.
- **Blocks**: `found` (one closed area holds over half the block and over half of that area is the block); `whole`
  (the same at nine tenths both ways); `joined`, `cut up`, `lost`; `extra` (a found area lying mostly on street).
- **Panoptic quality**: found areas matched to answer-key areas at an overlap over one half (the least overlap at
  which a match is unique); the matched overlaps summed, over matches plus half the unmatched areas of each side.

## The five runs

All on the two-answer code, 3,000 steps, last step kept, one training each.

| Run | Print sizes | Open-land pieces | Noise |
|---|---|---|---|
| `two-answers` | no | no | yes |
| `two-answers-no-noise` | no | no | no |
| `resized` | yes | no | yes |
| `open-land` | no | yes | yes |
| `resized-open-land` | yes | yes | yes |

**On the choosing pieces** (where a recipe is chosen), all eight together:

| Run | Street called street | Block called not street | Real boundary to edge drawn: 90% within | Edge drawn to real boundary: 90% within | Panoptic quality |
|---|---|---|---|---|---|
| `two-answers` | 0.946 | 0.977 | 12.9 ft | 5.6 ft | 0.714 |
| `two-answers-no-noise` | 0.952 | 0.984 | 12.2 ft | 7.8 ft | 0.644 |
| `resized` | 0.978 | 0.975 | 5.4 ft | 5.4 ft | 0.647 |
| `open-land` | 0.980 | 0.979 | 8.2 ft | 5.6 ft | 0.674 |
| `resized-open-land` | 0.985 | 0.976 | 5.4 ft | 5.4 ft | 0.701 |

`resized-open-land` is the best on the choosing pieces by the owner's measure; on the 1960-1980 choosing pieces its
street share is 0.998 against 0.901 for `two-answers`.

**On the test pieces**, street called street, and the distance within which 90% of real boundary has an edge drawn:

| Era | `two-answers` | `resized` | `open-land` | `resized-open-land` |
|---|---|---|---|---|
| 1948, hatched colour | 0.935, 19.7 ft | 0.984, 4.9 ft | 0.957, 8.8 ft | 0.984, 4.9 ft |
| 1921-1935, hatched grey | 0.915, 65.4 ft | 0.927, 43.5 ft | 0.918, 81.9 ft | 0.925, 41.1 ft |
| 1960-1980, lined grey | 0.935, 13.6 ft | 0.986, 5.4 ft | 0.971, 5.4 ft | 0.985, 2.7 ft |
| 1980s-2000, lined 1-bit | 0.989, 0.0 ft | 0.987, 3.4 ft | 0.991, 3.4 ft | 0.989, 3.4 ft |

**Two trainings of one setup differ** by about this much (the retrained run B of 2026-11-23 against `two-answers`:
same labels and tiles, a different starting head bias): 1960-1980 streets 0.953 against 0.935, its boundary 90%
within 5.4 against 13.6 ft. Gaps of that size between two recipes mean little; the print-size gaps are larger.

## Print size

The owner noticed Market Street on a 1960 piece: "it's wide and so your model doesn't think it's a street?"

- Measured: street ground more than 40 ft from any block was missed far more than the rest, and the missed spots
  were no more often on or near ink than street ground in general.
- The cause, tested: the sheets are scanned at different sizes. The lot-labelled training sheets are 3.7-4.3 ft per
  px; 1960-1975 are 2.7; 1948 is 2.1-2.4; 1921 runs from 2.7 to 5.4. Reading each 1960-1980 test piece shrunk to
  three-quarters size (the learned size) lifted its street share from 0.935 to 0.980 and brought 90% of its boundary
  within 2.7 ft instead of 13.6 ft. The model read print well only at the px size it learned on.
- The finder may not know a sheet's scale when it reads (it is shown only the scan), so the fix is in training:
  each tile is resized from its own sheet's feet per px to that of a sheet drawn from every series (each series
  equally likely, then each of its sheets), labels and cut statements remade at the new size. The sizes are measured
  from the 337 sheets whose placement passes the warp's own check; no size or width is typed in.
- Result: the table above. Market's piece (run `resized`): street 0.904 to 0.962, 90% of its boundary within 2.7 ft instead of 21.9
  ft (`look/owner/result-market-1960-before-after-resize.png`). What remains red there lies under the letters "C-2"
  printed in the street.

## Open land

Section 12 is the long tail of the old grey maps: half its real boundary is over 17 ft from any edge drawn.

- Measured by what lies on the two sides of each boundary px (`two-answers`): a block against a street, half within
  0 ft; a block against open land, 33 ft; open land against a street, 90 ft.
- The nine training pieces held 215,918 px of block-against-street boundary, 2,023 px of open land against a
  street, and **19 px** of a block against open land. The model had never been taught it.
- Three training pieces with open land were drawn (1921 sheet 100525362 away from Section 12; 1921 Bay View Park;
  1929 at the Presidio's edge), each checked on its labels picture.
- Result on Section 12 with both fixes: an edge is the more likely answer on 51% of its boundary against 42%; a
  block against open land is half within 15 ft instead of 33; open land against a street is no better (116 ft).
  **A small gain, not a fix.**
- What the picture shows (`look/owner/result-section12-boundary-before-after.png`): the sides of *hatched* blocks
  facing open land improved (that is what the new pieces show). Still missed is every line with blank paper on both
  sides: the two lines of the roads through open land (Sunnydale, Visitacion) and the open-land side of the row of
  *empty-outlined* blocks. A fourth piece showing exactly that is in "The second try" below.

## Blocks

With the depth fitted on panoptic quality, on the test pieces:

| Run | Depth | Panoptic quality | Found | Whole | Joined | Cut up | Lost | Extra |
|---|---|---|---|---|---|---|---|---|
| `two-answers` | 253/255 | 0.564 | 593 | 378 | 12 | 45 | 1 | 18 |
| `two-answers-no-noise` | 243/255 | 0.449 | 598 | 366 | 7 | 45 | 1 | 60 |
| `resized` | 148/255 | 0.535 | 591 | 377 | 5 | 53 | 2 | 14 |
| `open-land` | 252/255 | 0.543 | 598 | 373 | 10 | 42 | 1 | 26 |
| `resized-open-land` | 153/255 | 0.600 | 600 | 388 | 6 | 44 | 1 | 12 |

- Fitted on the share of blocks found, as before, the same runs gave 147 to 521 extra areas: that score cannot see
  a stray area, and it picked depths of 2 to 90.
- About four blocks in ten are still not `whole`. On the lined maps they are split along zoning borders, on the 1948
  maps along the line where a solid-black district meets blank paper inside a block. The owner accepts such splits.
- **An option measured, not built**: taking the connected pieces of ground the model calls not street as the areas,
  with no watershed and no setting. Measured on the run `resized`: panoptic quality 0.846 on the
  choosing pieces (the watershed: 0.647) and 0.837 on test; 556 blocks whole instead of 377, none cut up; but 62
  blocks joined across a street instead of 5, and 69 extra areas instead of 14. It trades split blocks for joined
  ones; which matters more is the owner's call.

## Noise

Asked for by the owner. On the choosing pieces the runs with and without it tie (street 0.946 and 0.952; boundary
12.9 and 12.2 ft). On test the differences are inside what two trainings of one setup differ by. No measurable win
either way; the later runs keep it on.

## What is still wrong or unsettled

- **Section 12's kind of ground** (above).
- **One run per recipe.** No recipe has been trained twice with different seeds.
- **Labels that disagree between sources.** The lots call lot-free ground "street", including about a million px of
  open land or water on `2000-digital-zoning` p05 that a hand piece would mark as open land; the model does not call
  it street even there (0.135). And where two numbered blocks touch with no street, the lots make one block and
  the hand pieces two.
- **Placements the warp itself doubts** sit under two training pieces (1921 094355582; 1935 p10) and one choosing
  piece (1948 sheet 10). "Off every test piece's ground" is checked through those placements.
- **A strip for the owner to rule on**: on the 1929 Presidio training piece, a narrow strip between the blocks and
  the Presidio is marked street; it may be the Presidio wall
  (`look/owner/training-image-presidio-strip-1929.png`, a training image).
- **The unbroken-line option** (`--unbroken`) is unusable as built (about seven times slower); never compared.
- **Crops only.** Reading a whole sheet and handing streets and block edges to the sheet warp (#69) is not built.
- Both boundary measures are blind at a piece's rim (a block lying just outside the box has no answer-key boundary
  inside it), which inflates their 99% values.

## Decisions waiting for the owner

1. The Presidio strip: road or wall.
2. Areas: keep the watershed (blocks split along zoning borders, few joined) or take plain pieces of not-street
   ground (blocks whole, more joined across streets).
3. Where a lot-free expanse is open land or water, should the lot labels stop calling it street?
4. Whether to hand-mark 1960-1980 training pieces (needs teaching from outline pieces, which is not built), now
   that the print-size fix carries that era to 0.985.

## Reproducing

```
cd pipeline
uv run sfzmaps find labels --series 2000-digital-zoning --sheet 2000-digital-zoning-p05 --sheet 2000-digital-zoning-p07 --sheet 2000-digital-zoning-p11
uv run sfzmaps find labels --series 2000-digital-height --sheet 2000-digital-height-p07
uv run sfzmaps find labels --series 2000-digital-su --sheet 2000-digital-su-p10
uv run sfzmaps find labels --series 1980-1990-zoning --sheet 1980-1990-zoning-p07
uv run sfzmaps find train --run resized-open-land --sheet 2000-digital-zoning/2000-digital-zoning-p05 ... (the six sheets)
uv run sfzmaps find count --run resized-open-land
```

`--no-resize` and `--no-noise` give the other runs; `open-land` and `resized-open-land` were trained at commit
8181fe7, the three before them at f449bbf and 661c674 with nine training pieces, and all five were counted at
946fcc8. The choosing-piece street and boundary figures, the boundary-by-kind figures, the shrink test and the
plain-pieces option were made by one-off scripts on the runs' models; they are not commands. $0.
