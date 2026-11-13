# Hard cutoffs in the W3 run's code (audit, 2026-11-12)

Categories: **A** modelling cutoff (should become a probability), **B** final decision (fine), **C** numerical safety
(fine), **D** measurement / report only. W3 reads the OLD merged page (mosaic.py), with odds built on zstrokeev5_old.

## The A rows most likely to cause wrong zoning
1. **Solver walls.** zbubble.py:30: a border at odds > 0 passes no push; at <= 0 it passes at least half.
   - zbubble.py:174-175, 189-192: walled areas are connected components over odds <= 0.
   - zpairfit4.py:125-126, znolabel.py:35-36: the same areas are used for training.
2. **Label pinned to one lot, with its code = its top reading** (zbubble.py:165-166, 213-219, 255-256).
   - Second readings never compete.
   - RULE cost keeps the code through merges.
3. **Print placement** (labread2.py:58-59, addprints.py:101/104, labside.py:120/125/143-151, zbubreal.py:61).
   - A print belongs to the single lot under its text-box centre.
   - "On a street" is yes/no.
   - It gets exactly one far-side lot.
4. **The hinge at odds 0** (zcore.py:79-82 with repw 0 at zbubble.py:168; zbubble.py:67-70, 117-123).
   - Crossing a border with odds > 0 costs nothing in the cut.
   - The film features split borders at exactly 0.
   - The same-code film rate RF is assumed, not measured.
5. **Per-sheet cleaning.** Not measured before 2026-11-12.
   - blocklines.py:38-49: all ink on a 2000 block outline is deleted unless zonegraph.py:81 (w >= T) called that edge
     heavy.
   - fitboxes.py:417/577: letters are erased only if on_ink >= .40 and cover >= .6; fitboxes.py:509-538 decides "line vs
     letter" by fixed rules; fitboxes.py:600/624-631 cover the erase and despeckle.
   - mosaic.py:56/65: a sheet's ink is used only where it is > 0.5 inside its drawn footprint.
6. **Whole-wall stage** (zcore.py:163/179, used by zloop_odds.py:53 and zonemodel_cmp2.py:79): areas are merged greedily
   while the summed odds are < 0.  Measured: on raw ink it joins areas across 98% of the SE lines raw lost; it is the
   main reason clear lines are missed (+2.9 when missed vs +4.8 when found).
7. **Single nearest stroke** (zstrokeev.py:73-77, 132-139; zenclose.py:59-65; zinkmeas.py:47-60): each border pixel sees
   only the nearest stroke or closing, so a district line behind a nearer stroke is invisible.

## Other A rows
- **Prints:**
  - addprints.py:59/67, labread2.py:28: only Terra boxes tagged "zoning code".
  - addprints.py:71 (IoU > .3), 74-75 (top turned reading), 78/85 (known codes only).
  - labread2.py:22: only kind "use".
  - labread2.py:35-38: one label point per box.
  - labread2.py:40/45: inside the drawn footprint only.
- **Closings:**
  - zclose.py:27-32: street ground and lot lines only.
  - zclose.py:17/19/71: 512 ft reach.
  - zclose.py:72-76: one closing per loose end.
  - zwholelines.py:135/237-243: nearest stroke within 250 ft.
  - zwholelines.py:174-188: greedy junction pairing.
- **Placement:**
  - zplace.py:136-145: a cut lot goes wholly to one side.
  - zcaseA_offsets.py:56-57: street classes < 150 merged, counted over all quarters.
  - zcaseA_offsets.py:59: cos bins at 1/3, 2/3.
- **Training targets and sets:**
  - zfilmtrain.py:49: truth = more than half of the film's borders.
  - znolabel.py:39: "labelled" = any print.
- **Frame:** zframe2.py:89/101/130/146/153 (search windows, 4 px runs, 70% dark legend row, 60% / 1 px fit rule).
- **Solver:**
  - zbubble.py:376-377: group trust = max.
  - zbubble.py:457-458: insulation only from films with pf > .5.
- **Low impact:**
  - zenclose levels (fitted weights), zinkmeas 6 ft / 12 ft (tree inputs);
  - zstrokes.py:107 spur;
  - zstrokeev.py:45/55/57 text-box shape and placement;
  - NOSUD;
  - zpushcal2 / znolabel / zpairfit4 row filters.

B, C, D rows: see the audit transcript; none change the reading.

## Related measurements (2026-11-12)
- **Gaps are not gaps** (owner, looking at pics/gaps_raw.png): they are lines drawn half to one parcel off (drafting or
  data), lines following a street instead of a right angle, curves, reclassified parcels, condo slivers, or misreads.
- **Ink kinds along the 632 mi of district line** (zoffsetmix.py, a mixture over distance profiles, raw page), with
  expected misses (74 mi):
  - heavy line on the border: 405 mi, 14 mi missed;
  - heavy line about 9 ft off: 104 mi, 13 mi missed;
  - heavy line 16-45 ft off with thin ink nearer: 73 mi, 31 mi missed;
  - nothing within about 370 ft: 23 mi, 15 mi missed.
- **Clear lines missed** (zmisparts.py): the whole-wall stage, and across streets the evidence looks only near the
  street's middle, so a line drawn along a street edge counts against.
