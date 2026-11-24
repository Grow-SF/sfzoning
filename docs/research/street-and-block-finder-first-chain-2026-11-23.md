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

The current recipe (every print size, thirteen hand training pieces, noise), trained three times with different
seeds, on the test pieces. Each cell gives the lowest and highest of the three:

| Era | Street ground called street | Block ground called not street | Real boundary to the nearest edge drawn: 90% within | Edge drawn to the nearest real boundary: 90% within |
|---|---|---|---|---|
| 1948, hatched colour (134 blocks) | 0.986-0.987 | 0.982-0.985 | 4.9 ft | 4.9 ft |
| 1921-1935, hatched grey (190) | 0.947-0.958 | 0.991-0.992 | 35-60 ft | 2.8 ft |
| 1960-1980, lined grey (174), never taught by hand | 0.990 | 0.969-0.971 | 3.1-3.8 ft | 5.4 ft |
| 1980s-2000, lined 1-bit (153) | 0.989-0.990 | 0.964-0.965 | 3.7-4.0 ft | 6.7 ft |

- Half of the real boundary has an edge drawn on it (0.0 ft) in every era and every seed.
- The edges the model draws lie on real block boundary (90% within 3 to 7 ft).
- What is still wrong is boundary it leaves out, nearly all on one test piece: Section 12 of 1921 (Visitacion
  Valley), where blocks and roads stand in open land. See "Open land".
- Blocks joined across a street: 4 to 5 of 651.
- Before the night's fixes the same table read: street 0.935, 0.915, 0.935, 0.989; boundary 90% within 19.7, 65.4,
  13.6, 0.0 ft.

## What changed since the first write-up, in order

1. **A final review of the whole branch** found a choosing piece lying on test ground (redrawn, two more trimmed; the
   code now refuses it) and that the zoning-border labels made from lots contradicted each other.
2. **The zoning-border answer was removed** (the owner's decision): model, training, labels, its made-up-town test.
3. **The trainer keeps its last step.** It used to keep the checkpoint with the best choosing score; that score
   swings between 0.46 and 0.98 from one checkpoint to the next, and in one run it kept step 250 of 3,000, an
   immature model.
4. **Training tiles are shown at every print size the finder will meet** (see "Print size").
5. **Four training pieces with open land were added** (see "Open land").
6. **The boundary measure got its other half** (edge drawn to nearest real boundary); the first half alone flattered
   a model that calls edges loosely.
7. **The area step's one setting is fitted on panoptic quality** instead of the share of blocks found, which could
   not see stray areas (see "Blocks").
8. **`find train --seed`**, so that one recipe can be trained more than once (see "Seeds").

## What is trained on

- Six sheets labelled automatically from the lots of their year: `2000-digital-zoning` p05, p07, p11;
  `2000-digital-height` p07; `2000-digital-su` p10; `1980-1990-zoning` p07 (street or not, "no block edge", and
  short statements that a block edge crosses a cut).
- Thirteen hand-marked training pieces on hatched maps: 1921 x6, 1929 x3, 1935 x2, 1948 x2 (nine at first; four
  with open land added on 2026-11-24).
- **No hand piece of 1960-1980** (the lined grey maps): that era is judged without ever being taught by hand.
- Ground under the test and choosing pieces is left out of training in every scan. A training tile that is shrunk
  lets a labelled spot see a little further than the margin allows for: up to about 55 px of a held-out piece's
  print, never one of its labels.

## The measures

- **Street ground called street**: of the answer key's street px, the share where the model's street chance is over
  one half. **Block ground called not street**: of its block and open-land px, the share where it is not.
- **Real boundary to the nearest edge drawn**: from each answer-key boundary px, the distance in feet to the nearest
  px whose block-edge chance is over one half. Large values mean boundary the model leaves out.
- **Edge drawn to the nearest real boundary**: the same the other way, for edge px on labelled ground. Large values
  mean edges that are not boundary.
- **Blocks**, each against the closed area that holds most of it: `found` (that area holds over half the block and
  the block is over half of that area); `whole` (the same at nine tenths both ways); `joined` (most of that area
  lies outside the block: two blocks in one area); `cut up` (neither: the block lies in several areas); `lost` (most
  of the block is in no area); `extra` (a found area lying mostly on street).
- **Panoptic quality**: found areas matched to answer-key areas at an overlap over one half (the least overlap at
  which a match is unique); the matched overlaps summed, over matches plus half the unmatched areas of each side.
  The area step's depth is fitted on the choosing pieces to make this highest, so the choosing value is a fitted one.

## The runs

All on the two-answer code, 3,000 steps, last step kept.

| Run | Print sizes | Hand training pieces | Noise | Seeds trained |
|---|---|---|---|---|
| `two-answers` | no | 9 | yes | 0 |
| `two-answers-no-noise` | no | 9 | no | 0 |
| `resized` | yes | 9 | yes | 0 |
| `open-land` | no | 12 | yes | 0 |
| `resized-open-land` | yes | 12 | yes | 0, 1, 2 |
| `resized-open-land-2` (the current recipe) | yes | 13 | yes | 0, 1, 2 |

**On the choosing pieces** (where a recipe is chosen), all eight together, seed 0:

| Run | Street called street | Block called not street | Real boundary to edge drawn: 90% within | Edge drawn to real boundary: 90% within |
|---|---|---|---|---|
| `two-answers` | 0.946 | 0.977 | 12.9 ft | 5.6 ft |
| `two-answers-no-noise` | 0.952 | 0.984 | 12.2 ft | 7.8 ft |
| `resized` | 0.978 | 0.975 | 5.4 ft | 5.4 ft |
| `open-land` | 0.980 | 0.979 | 8.2 ft | 5.6 ft |
| `resized-open-land` | 0.985 | 0.976 | 5.4 ft | 5.4 ft |
| `resized-open-land-2` | 0.985 | 0.976 | 5.4 ft | 5.4 ft |

**On the test pieces**, seed 0, street called street, and the distance within which 90% of real boundary has an
edge drawn:

| Era | `two-answers` | `resized` | `open-land` | `resized-open-land` | `resized-open-land-2` |
|---|---|---|---|---|---|
| 1948, hatched colour | 0.935, 19.7 ft | 0.984, 4.9 ft | 0.957, 8.8 ft | 0.984, 4.9 ft | 0.986, 4.9 ft |
| 1921-1935, hatched grey | 0.915, 65.4 ft | 0.927, 43.5 ft | 0.918, 81.9 ft | 0.925, 41.1 ft | 0.950, 35.4 ft |
| 1960-1980, lined grey | 0.935, 13.6 ft | 0.986, 5.4 ft | 0.971, 5.4 ft | 0.985, 2.7 ft | 0.990, 3.8 ft |
| 1980s-2000, lined 1-bit | 0.989, 0.0 ft | 0.987, 3.4 ft | 0.991, 3.4 ft | 0.989, 3.4 ft | 0.989, 3.7 ft |

## Seeds

A run repeats exactly: the same recipe trained twice with one seed gave byte-identical weights. With three seeds
each of the last two recipes:

- **The street and boundary measures barely move between seeds.** Street called street stays within 0.003 in three
  eras and within 0.011 on the old grey maps; the 90% boundary distances stay within a foot, except the old grey
  maps' (33 to 73 ft), which is Section 12's tail. So the gains above from print sizes (0.935 to 0.985 on 1960-1980)
  and from the park-drives piece (0.925 to 0.95 on the old grey maps) are far outside what a seed does.
- **The block scores move a great deal between seeds.** Panoptic quality on test: 0.600, 0.385, 0.492 with twelve
  pieces; 0.508, 0.361, 0.454 with thirteen. Blocks whole: 388, 346, 361; and 359, 328, 339. One recipe's seeds
  differ by more than the two recipes do. A difference in block scores between two single runs says nothing; that
  includes every block-score comparison made on 2026-11-23, and the noise comparison below.

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
- Result: the tables above. Market's piece: street 0.904 to 0.971, 90% of its boundary within 2.7 ft instead of 21.9
  ft (`look/owner/result-market-1960-before-after.png`). What remains red there lies under the letters "C-2" printed
  in the street.

## Open land

Section 12 was the long tail of the old grey maps: 0.694 of its street ground called street, half its real boundary
over 17 ft from any edge drawn.

- Measured by what lies on the two sides of each boundary px (`two-answers`): a block against a street, half within
  0 ft; a block against open land, 33 ft; open land against a street, 90 ft.
- The nine training pieces held 215,918 px of block-against-street boundary, 2,023 px of open land against a
  street, and **19 px** of a block against open land. The model had never been taught it.
- **First try**: three training pieces with open land (1921 sheet 100525362 away from Section 12; 1921 Bay View
  Park; 1929 at the Presidio's edge). A small gain: the sides of *hatched* blocks facing open land improved (what
  the pieces show); still missed was every line with blank paper on both sides: the two lines of the roads through
  open land (Sunnydale, Visitacion) and the open-land side of the row of *empty-outlined* blocks.
- **Second try**: a piece of exactly that on Section 12's own sheet. None could be drawn: there the roads and the
  empty blocks beside open land exist only inside the test piece or within 300 px of it.
- **Third try**: one piece of Golden Gate Park's drives (1929 p08): about 12,000 px of two-line road through open
  land. Section 12's street share went to 0.897 (0.847 to 0.934 over three seeds; 0.701 to 0.730 without the
  piece); open land against a street is half within 30 ft instead of 116.
  `look/owner/result-section12-streets-before-after.png` shows Sunnydale turning from missed to found.
- **Still missed**: an edge is the more likely answer on only about half of Section 12's boundary
  (`look/owner/result-section12-boundary-before-after.png`). The model now calls the roads street but draws their
  lines, and the open-land side of the empty blocks, faintly. No place off the test ground was found that shows an
  empty-outlined block's side facing open land.

## Blocks

With the depth fitted on panoptic quality, on the test pieces (seed 0 unless three are given):

| Run | Depth | Panoptic quality | Found | Whole | Joined | Cut up | Lost | Extra |
|---|---|---|---|---|---|---|---|---|
| `two-answers` | 253/255 | 0.564 | 593 | 378 | 12 | 45 | 1 | 18 |
| `two-answers-no-noise` | 243/255 | 0.449 | 598 | 366 | 7 | 45 | 1 | 60 |
| `resized` | 148/255 | 0.535 | 591 | 377 | 5 | 53 | 2 | 14 |
| `open-land` | 252/255 | 0.543 | 598 | 373 | 10 | 42 | 1 | 26 |
| `resized-open-land`, seeds 0, 1, 2 | 153, 126, 150 | 0.600, 0.385, 0.492 | 600, 567, 583 | 388, 346, 361 | 6, 7, 5 | 44, 77, 62 | 1, 0, 1 | 12, 19, 15 |
| `resized-open-land-2`, seeds 0, 1, 2 | 139, 138, 149 | 0.508, 0.361, 0.454 | 577, 579, 574 | 359, 328, 339 | 5, 4, 5 | 68, 66, 70 | 1, 2, 2 | 15, 15, 14 |

- Fitted on the share of blocks found, as before, the first five runs gave 147 to 521 extra areas: that score cannot
  see a stray area, and it picked depths of 2 to 90.
- Steady across seeds and recipes: few blocks joined (4 to 12), few extra areas (12 to 26 with noise). Not steady:
  how many blocks are cut up, and with it the panoptic quality (see "Seeds").
- About four blocks in ten are not `whole`. On the lined maps they are split along zoning borders, on the 1948 maps
  along the line where a solid-black district meets blank paper inside a block. The owner accepts such splits.
- Two fitted depths (252, 253) sit at the top of the range tried (254 is the largest).
- **An option measured, not built**: taking the connected pieces of ground the model calls not street as the areas,
  with no watershed and no setting. Measured on the run `resized` by a one-off script (which gave the watershed
  0.624 on the choosing pieces where the committed count gives 0.647): panoptic quality 0.846 on the choosing
  pieces and 0.837 on test; 556 blocks whole instead of 377, none cut up; but 62 blocks joined across a street
  instead of 5, and 69 extra areas instead of 14. It trades split blocks for joined ones; which matters more is the
  owner's call.

## Noise

Asked for by the owner. One run each, both before print sizes. On the choosing pieces the street and boundary
measures tie (street 0.946 with noise, 0.952 without; boundary 12.9 and 12.2 ft). The run with noise has the higher
panoptic quality (0.714 against 0.644 choosing; 0.564 against 0.449 test) and fewer extra areas (18 against 60), but
that score moves by more than this between two seeds of one recipe. Not settled; the later runs keep noise on.
Settling it takes three seeds without noise on the current recipe.

## What is still wrong or unsettled

- **Section 12's lines** (above): streets found, edges still faint.
- **The area step is the unsteady part**: its scores swing between seeds while the two answers under it do not.
- **Labels that disagree between sources.** The lots call lot-free ground "street", including about a million px of
  open land or water on `2000-digital-zoning` p05 that a hand piece would mark as open land; the model does not call
  it street even there (0.135). And where two numbered blocks touch with no street, the lots make one block and
  the hand pieces two.
- **Placements the warp itself doubts** sit under two training pieces (1921 094355582; 1935 p10) and one choosing
  piece (1948 sheet 10). "Off every test piece's ground" is checked through those placements.
- **Calls on the new training pieces for the owner to rule on**: on the 1929 Presidio piece, a narrow strip between
  the blocks and the Presidio is marked street; it may be the Presidio wall
  (`look/owner/training-image-presidio-strip-1929.png`). On the Golden Gate Park piece, every ring of the Stadium
  and the bands round two lakes are marked street (ways); its strokes along the dashed park lines were laid by a
  helper that follows the printed ink, then checked by eye.
- **The unbroken-line option** (`--unbroken`) is unusable as built (about seven times slower); never compared, and
  not looked at with resized tiles.
- **Crops only.** Reading a whole sheet and handing streets and block edges to the sheet warp (#69) is not built.
- Both boundary measures are blind at a piece's rim (a block lying just outside the box has no answer-key boundary
  inside it), which inflates their 99% values.

## Decisions waiting for the owner

1. The Presidio strip (road or wall) and the Stadium's rings (ways or not).
2. Areas: keep the watershed (blocks split along zoning borders, few joined, scores unsteady) or take plain pieces
   of not-street ground (blocks whole, more joined across streets).
3. Where a lot-free expanse is open land or water, should the lot labels stop calling it street?
4. Whether to hand-mark 1960-1980 training pieces (needs teaching from outline pieces, which is not built), now
   that the print-size fix carries that era to 0.99.

## Reproducing

```
cd pipeline
uv run sfzmaps find labels --series 2000-digital-zoning --sheet 2000-digital-zoning-p05 --sheet 2000-digital-zoning-p07 --sheet 2000-digital-zoning-p11
uv run sfzmaps find labels --series 2000-digital-height --sheet 2000-digital-height-p07
uv run sfzmaps find labels --series 2000-digital-su --sheet 2000-digital-su-p10
uv run sfzmaps find labels --series 1980-1990-zoning --sheet 1980-1990-zoning-p07
uv run sfzmaps find train --run resized-open-land-2 --seed 0 --sheet 2000-digital-zoning/2000-digital-zoning-p05 ... (the six sheets)
uv run sfzmaps find count --run resized-open-land-2
```

`--no-resize` and `--no-noise` give the other recipes. Commits trained at: `two-answers` and its no-noise twin
f449bbf; `resized` 661c674; `open-land` and `resized-open-land` 8181fe7 (twelve training pieces);
`resized-open-land-2` d263361 (thirteen); the seed 1 and 2 runs 576d814, the twelve-piece ones with the park-drives
piece moved out of the folder while they trained. The first five were counted at 946fcc8.

Made by one-off scripts on the runs' models, not by a command: the choosing-piece street and boundary figures; the
boundary-by-kind figures; the feet-from-a-block figures and the shrink test under "Print size"; the 0.135 figure;
the plain-pieces option; the extra-area counts of the old depth fit; the px counts of what the training pieces
hold. $0.
