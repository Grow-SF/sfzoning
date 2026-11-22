# Map-area labels (issue #71)

Hand-drawn outlines of the printed map area on a few sheets of every edition. They train and score the learned
frame output (issue #71): inside the outline is the printed city; everything outside it (frame bars and rules, ruled-off
legend, title, index and note boxes at the margin, margins, scan background) is not.

One file per sheet: `<series>/<sheet>.json`, the sheet named as in `data/work/<series>/<sheet>/page.png`.

```json
{
 "polygons": [[[x, y], ...]],
 "remove": [[[x, y], ...]],
 "notes": "what was ambiguous and how it was decided",
 "drawn_by": "who, how, date"
}
```

`remove` (optional) lists printed blocks inside the frame that are deleted although they sit on the map area (rule 4):
each a simple outline a few px outside the block's ink. Same format as `polygons`.

Coordinates are `page.png` pixels at full resolution, as stored (x right, y down, the page as scanned, not turned).

## How the outline is drawn

1. **The edge lies on paper between the innermost printed frame rule and the first map ink (owner, 2026-11-20).** The
   innermost rule is the thin rule beside a thick bar, if there is one. The outline touches ink only where map ink
   truly touches the rule, and there it follows the rule's inner edge. **The rule itself always falls outside.**
2. **Corner ornaments are frame (owner, 2026-11-19).** Decorative corner arcs or quarter-circles inside the frame
   (1921–1948) are cut out: at a corner the outline follows the inner edge of the innermost arc (points every ~30 px
   along it), so the arcs and everything between them and the corner are outside. Where map ink runs across an arc
   (1921 blocks do in places), the part beyond the arc is outside too.
3. **Ruled-off boxes at the margin are outside (owner, 2026-11-20).** Every ruled-off box at the margin is outside,
   whether it touches the frame or stands clear of it, cut out together with the blank paper between it and the frame
   (or the page edge, where a side has no frame). That covers legend, title, index inset (e.g. the ZONING MAP INDEX
   inset), sheet-number and notes boxes (the 2000-style legend strip along the bottom, the 1964–1975 title strip down
   one side). The outline runs just outside that box on the map side, as a notch if the box occupies only part of a
   side. "SECTION N" sheet-number badges are not notches: they are `remove` polygons (rule 4).
4. **Removed blocks (owner, 2026-11-20).** Legend/title blocks, compass roses/north arrows and SECTION badges are
   drawn as `remove` polygons wherever they sit, mid-map or at the margin: a simple outline (4–12 points) a few px
   outside the block's ink, on paper; where a block overlaps map ink, the polygon still takes the whole block (say so
   in the notes). Every other floating block (notes, scale bars, cross-references, place names) is inside; no holes.
   Ruled boxes at the margin stay notches (rule 3).
5. **No frame on a side.** Where a side has no frame rule (the scan cut through the map, or the sheet prints none), the
   outline runs along the edge of the page (or of the paper, if the scan shows background beyond the paper).
6. **Several framed maps on one page:** one polygon each.
7. **Skew:** pages are slightly turned; place each corner where it really is, don't square it up.
8. **Points along a side where the rule bows.** Scans (above all the 1921–1948 photocopies) bow: a straight
   corner-to-corner edge can sit 10–20 px inside the rule at mid-side and cut off map. Check every side every
   300–400 px and add a point wherever the edge would sit more than about 5 px inside the rule.

## Tool

`uv run python scripts/frame_label_tool.py` (in `pipeline/`): `overview` (whole page with a coordinate grid),
`zoom` (full-resolution close-ups of one or more points with pixel ticks), `check` (the outline drawn on the overview,
and a close-up of every vertex). Pictures go to `data/research/issue71/labels/<series>/<sheet>/`.
