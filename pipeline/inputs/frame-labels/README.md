# Map-area labels (issue #71)

Hand-drawn outlines of the printed map area on a few sheets of every edition. They train and score the learned
frame output (issue #71): inside the outline is the printed city; everything outside it (frame bars and rules, legend,
title and note boxes ruled off from the map, margins, scan background) is not.

One file per sheet: `<series>/<sheet>.json`, the sheet named as in `data/work/<series>/<sheet>/page.png`.

```json
{
 "polygons": [[[x, y], ...]],
 "notes": "what was ambiguous and how it was decided",
 "drawn_by": "who, how, date"
}
```

Coordinates are `page.png` pixels at full resolution, as stored (x right, y down, the page as scanned, not turned).

## How the outline is drawn

1. **The edge sits just inside the innermost printed frame rule** (the thin rule beside a thick bar, if there is one):
   within about 5 px of the rule's inner edge, on the map side, **never on the rule or outside it**. When unsure, err
   inward. The rule itself must always fall outside.
2. **Corner ornaments are frame (owner, 2026-11-19).** Decorative corner arcs or quarter-circles inside the frame
   (1921–1948) are cut out: at a corner the outline follows the inner edge of the innermost arc (points every ~30 px
   along it), so the arcs and everything between them and the corner are outside. Where map ink runs across an arc
   (1921 blocks do in places), the part beyond the arc is outside too.
3. **Ruled-off boxes are outside.** A legend, title, sheet-number or notes box that is separated from the map by a
   printed rule (the 2000-style legend strip along the bottom, the 1964–1975 title strip down one side) is excluded:
   the outline runs just inside that box's rule on the map side, with a notch if the box occupies only part of a side.
4. **Floating text stays inside.** A title block, legend or note printed over the map area with no rule separating it
   (e.g. 1921's "ZONE MAP" block in a corner of the map) stays inside.
5. **No frame on a side.** Where a side has no frame rule (the scan cut through the map, or the sheet prints none), the
   outline runs along the edge of the page (or of the paper, if the scan shows background beyond the paper).
6. **Several framed maps on one page:** one polygon each.
7. **Skew:** pages are slightly turned; place each corner where it really is, don't square it up.
8. **Points along a side where the rule bows.** Scans (above all the 1921–1948 photocopies) bow: a straight
   corner-to-corner edge can sit 10–20 px inside the rule at mid-side and cut off map. Check every side every
   300–400 px and add a point wherever the edge would sit more than about 5 px inside the rule.

## Tool

`uv run python scripts/spikes/issue71_label_tool.py` (in `pipeline/`): `overview` (whole page with a coordinate grid),
`zoom` (full-resolution close-ups of one or more points with pixel ticks), `check` (the outline drawn on the overview,
and a close-up of every vertex). Pictures go to `data/research/issue71/labels/<series>/<sheet>/`.
