# Assumptions behind the bubble solve (v13) - each with how to check it

| # | assumption | where used | status / check |
|---|---|---|---|
| A1 | The table's code is what the 2000 map prints for the lot | every score | partly false: condo shared parcels (Clarendon "M2"); measure how many scored lots are artifacts |
| A2 | Line odds `lo` per border = calibrated log-odds that the table code differs across it | push, cut, films | CHECK: bin lo (held out) vs observed rate |
| A3 | Borders' evidence along one line is independent (cut sums -lo per border; push per border) | cut, push | FALSE (next border missed 69% vs 3%); fix by judging whole pieces |
| A4 | Seen line (lo > 0) blocks push completely; others pass P(no line) | push | arbitrary step at 0; CHECK effect |
| A5 | Push competition per lot is independent evidence; the cut sums it over lots | cut | SUSPECT: one comparison repeated on every lot (Visitacion leak). CHECK: how evidence scales with area |
| A6 | A label sits on a lot of its own district | push seeds | false for arrows, straddling labels, street prints; MEASURE share |
| A7 | Trust is calibrated (stated 0.9+ -> 97% right) | films, naming | validated |
| A8 | Every district has a label we read | no-label rule | false: 64% of label-less walled areas are their own district |
| A9 | Park chance per lot (size, list, label on lot) calibrated | cut, films | fitted held out; CHECK calibration on lots in big parks vs homes |
| A10 | Lot pairs (from the filled raster) are the real neighbours | everything | 446 pairs cross outside the city; CHECK others through wide non-lot space (freeways, bay) |
| A11 | District boundaries follow lot lines / street centres | design | believed; CHECK ink-to-border offsets |
| A12 | Film judgement fitted on label pairs transfers to bubble films between merged groups | films | distribution shift; CHECK calibration on the films the solve actually judges |
| A13 | Scan aligned to lots within a few ft | all ink reads | checked at county line (6-24 ft); CHECK city-wide (held walls vs table borders offset) |
| A14 | Heavy ink = district line, thin = lot line | line reading | partial: 5-6 px strokes 40% district, 6-9 px 56-75% |
| A15 | Where errors come from: lines vs labels vs structure | choosing what to fix | UNKNOWN - run ceilings first (perfect borders / perfect labels / both) |
| A2 result | calibrated | | VALIDATED 2026-11-12: stated vs observed match in every bin (touching and street). ~1,500 district borders sit at <2% stated: no ink, only labels can find them |
| A15 result | | | perfect borders (from the table) still leave 3,217 wrong (real lines: 4,672): lines ~1,450, labels ~3,200. Split: 1,521 in districts with no label lot inside; 725 naming lost to another label inside; 736 neighbour's code despite a right label (mechanism unknown); 235 misreads |
| A6 result | | | FALSE, the biggest error source: 500 districts (2,890 lots) have no label on their lots; nearest label of their code within 150 ft for 1,062 lots, 150-400 ft for 903, farther/none for 925 |
| A16 | The gate's 152,351 scored lots all have a knowable answer from the map | target 99% | CHECK the unwinnable share (both ceilings) |
| A5 result | | | REJECTED as the 6309 cause (ztestcut: repw/push weight change nothing there); the cause is a thin line read at -0.6..-1.5 |
| A17 | Line odds use the ink only | push, cut, films | FALSE as a best choice: lot character across a border (log-area difference) is real evidence. zborderfit2 held-out log-loss 0.0510 -> 0.0472; run v14lots 4,663 -> 4,502 |
| A18 | Empty walled areas are named after the neighbour across the weakest wall | dark rule | 89% of dark scored lots wrong. The chain's per-box choice model (relink3 link offers) used as prints: v14m0 4,663 -> 4,490 |
| A12 check | | | zfilmcal.py: stated film chance vs observed (codes differ) on the films the solve judges |
