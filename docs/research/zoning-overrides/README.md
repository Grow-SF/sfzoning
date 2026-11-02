# Zoning overrides: seam and printed-P corrections

`pipeline/data/overrides/` is git-ignored, so this folder keeps a copy. To restore them, copy
`<series>.csv` back there and run `uv run sfzmaps districts --series <S> --republish`, then
`uv run sfzmaps districts --shared`.

- `<series>.csv`: the overrides as applied (`blklot,block,code`).
- `regions.csv`: one row per series, read sheet, crop, old code and new code, with lot counts, blocks
  and the reviewer's note.
- `1975-c2-not-applied.csv`: the lots a 1975 C-2 review left unchanged (see "1975 C-2 misreads").

## What they correct

- **Seam leaks.** A lot read in a sheet's outer margin picked up a code that leaked across blank paper.
  The neighbouring sheet draws the lot in an interior region with another code, and the override uses
  that sheet's code.
- **Printed P.** A park or public lot where the scan prints P inside the lot but the reader recorded
  another code.

## How they were verified

Each candidate pair (series, read sheet, printing sheet) was cropped from the printing sheet's scan
with the proposed codes tinted over it, and a reviewer compared each tinted patch with the code
printed in its heavy-line region. The pair then got a verdict of agree, disagree or unclear. Blocks
that disagreed, sat across a heavy line or had no printed code in view were excluded from the pair.

These were quick, low-resolution checks of whole regions, not block-by-block checks. Neighbouring
years were never used as evidence. Candidates also had to pass these filters: the printing sheet has
ink on the block (at least 0.6), the lot is under 5 acres, the block isn't 9900, and the new code is a
known one and not P. The crop images aren't kept.

## Not applied

- Pairs with a verdict of unclear, and every excluded block.
- `1960 7 11` (349 lots): its evidence was a low-resolution crop only.
- 1975 lots in crop tiles the review notes say weren't viewed (271 lots): `p05 p04` 104, `p07 p13` 67,
  `p06 p07` 56, `p06 p14` 36, `p12 p13` 8.
- Golden Gate Park (1700001) and the Presidio (1300001) in 1960: the scan prints R-1-D there, so they
  stay R-1-D.
- Printed-P candidates with a verdict of disagree or unclear. These include 7560002 and 7560003
  (R-1-D printed in the lot), 7295038 (C-2), 7380036 (RH-1(D)), 4591A090 and 4886009 (mixed), and
  most of the 1975 P candidates. 3899001 in 1948 (Mission Rock) is also unchanged.

## 1975 C-2 misreads

A second pass on 1975 looked at the 8,247 "suspect" lots: C-2 in 1975 but not in 1972 or 1980. Those
years only chose which lots to look at. Each verdict comes from the code printed in the lot's
heavy-line region on the scan, recorded per block and read sheet (per lot where a heavy line splits a
block on p03).

| Verdict | Lots | Result |
| --- | ---: | --- |
| correct | 4,542 | Added to `1975.csv`, with `kind` `c2-suspect` in `regions.csv` |
| correct, moderate confidence | 526 | Stays C-2 |
| unclear | 2,780 | Stays C-2 |
| keep (the scan prints C-2) | 389 | Stays C-2 |
| no verdict | 10 | Stays C-2 |

`1975-c2-not-applied.csv` lists the 3,705 lots that stay C-2, with their verdict, any proposed code
and the reviewer's note. These checks were also by region, so a row can carry a lot or two that
overhang a heavy line; the notes in `regions.csv` say where the reviewer saw that.

C-2 lots went from 30,077 to 25,535, and threePlusShare from 40.9% to 40.4%.

## Still wrong

1975 keeps about 4,500 more C-2 lots than 1972 or 1980. Of the 3,705 lots above, the review confirmed
only the 389 "keep" lots as C-2, and it had no firm evidence to change the rest. The render of Noe also
shows what looks like a misread that isn't C-2, which this pass didn't look for: blocks between Castro
and Noe, such as 3622 to 3624 and 3655, are split between R-2 and R-4 with no heavy line through them
on the scan. 1975's P lots also remain far
below 1972's, because small P regions lost in seams had no candidates to apply.
