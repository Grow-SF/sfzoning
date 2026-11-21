# Frame crop: find the frame line, delete it and everything outside it (issue #71, revised)

Supersedes the two-scale design in `2026-11-19-learned-frame-design.md` (page pass + pixel pass). Same issue, same labels
(revised as below), simpler model.

## Intent

Every scanned sheet prints a frame around its map: bars, thin rules, corner arcs, ruled-off legend and title strips,
margins, scan background. The goal, in the owner's words: **"detect & delete the frameline and everything outside of
it."** The result is one region per page, the map inside the frame; everything outside it, including the frame ink, is
deleted (painted plain paper in the paper layer, ignored by reading and placement). The cut must be **picture perfect**:
no frame ink left in, no map ink cut off.

## Owner decisions this design rests on

- One learned method for every edition; hand labels allowed because there are few; Claude draws them.
- **One region, no holes.** Floating notes, titles and compass roses inside the frame stay inside (2026-11-20, reversing
  the earlier "floating notes are frame"). Ruled-off strips and insets that touch the frame are cut out, as notches.
- Corner arcs are frame and are cut out.
- **Full resolution in training and prediction.** No downsampled pass ("I don't think we can use low res to train or
  run").
- No made-up limits: no thresholds or penalties set by hand; anything numeric is learned, measured, or an owner cost.
- Judge by eye first; measure with probabilities, sheet-held-out.
- Prediction near instant; training minutes, not hours.

## Goals / non-goals / anti-goals

Goals: per page, a deletion mask (outside the frame) at scan resolution and the chance behind it; a picture of every map
page; held-out scores on every labelled sheet.

Non-goals (this change): wiring the mask into the published paper layer, reading and placement. Each is its own change
after the owner has seen the pictures (publishing needs build + ship).

Anti-goals: a downsampled stage; any hand threshold, window or per-series rule; any stage fitted on outputs of a network
that trained on the same sheets; non-determinism; GPL-family code.

## Is there an open-source tool for this? No.

Checked 2026-11-20:

| Candidate | What it does | Why not |
|---|---|---|
| ICDAR 2021 MapSeg task 2 entries ([report](https://arxiv.org/abs/2105.13265)) | Exactly this task on 1894–1937 Paris atlas sheets (≈10,000 px, 26 training sheets) | Winner (UWB): downsampled whole-page network + recursive Otsu binarization and morphology with parameters tuned on a validation set; **HD95 19 px**, not pixel perfect; code only as "extra material" on a university GitLab, no package or licence we can rely on. The 512 px single network (L3IRIS) scored **126 px** ("overly smoothed masks"): direct evidence that low resolution can't be precise. Rule-based entries (CMM 85 px, IRISA 112 px) were "penalized by unseen frame configurations". |
| dhSegment | Generic historical-document pixel labelling, page extraction | GPL-3.0 per its catalogue listing (licence rule), downsampled input |
| Segment Anything (SAM/SAM2) | Prompted object masks | Works at 1024 px input with a 256 px mask: low resolution; not trained on map frames |
| MapReader | Patch classification of map collections | Classifies patches; no frame boundary |
| OpenCV Hough / LSD, morphology recipes (e.g. Library of Congress blog) | Line finding | Hand rules per frame style; the MapSeg rule-based entries show the generalisation failure |
| mapvision (MIT code) | Compact U-Net with a "frame" class (margins, title boxes, decorations) | 1000 px tiles shrunk to 256 px; trained on 157 Paris tiles under a non-commercial dataset licence; frame-class IoU 0.77, no boundary accuracy |
| Semap (Petitpierre, arXiv 2603.05037) | Mask2Former over heterogeneous map collections; a "background" class covers borders, legends, ornaments | 768 px patches, heavy model; boundary IoU about 41%; code licence not stated. Its synthetic "crop along an edge to imitate the frame" augmentation is the same idea as our edge cutting |
| Fragment-based map evolution (Humanities & Social Sciences Communications 2024) | OCRNet segmenting map content from frame, title and legend | Area IoU 87.7%, far from pixel precise |
| Global Mapper "crop collar" | Crops USGS topographic sheets to the neatline | Proprietary, and built for USGS sheet layouts |
| mapsheet-mosaic ArcGIS tools (Apache-2.0) | Clips georeferenced sheets to a known sheet polygon | Needs the sheet already georeferenced plus the sheet polygon; doesn't find the frame |
