# Task 2b report: m2_clean_words

## Change (linegate.py, tests/test_textnet_linegate.py only)
- `clean_word_mask(sd, line) -> bool per word`. A word is clean when every 8-connected ink piece that touches its box lies wholly inside the box, and no run-rule px (`line`) lies within CLEAN_PX = 3 px OUTSIDE the box (ring of an elliptical dilation minus the box). Run-rule px inside the box do not disqualify the word (long P/M stems count); a line crossing the box also runs out of it, so the piece test and the ring test catch it.
- Measures `m2_clean_words` (all ink in clean words' boxes, no line rule applied) and `m2_clean_words_by_kind:*` (kind of the word drawn last on the px). Same fields (px, called_05, called_09, mean, log_loss, ci_05); tallies in blocks.npz, so `paired` works.
- "What each measure misses": "words touching or crossed by lines are left out; it measures letters, including long strokes, on clean words only".
- Existing measures untouched: in the re-gate, the five measures' (m2, m1a, m1b_in_box, m1c, overlap_band) px and called_05 were unchanged; the code change is additive.

## TDD
- RED: 3 new tests failed with `AttributeError: module 'linegate' has no attribute 'clean_word_mask'` (1 failed, 13 passed with -x).
- GREEN: `tests/test_textnet_linegate.py` 16 passed. Full `uv run pytest -q`: 1504 passed, 3 xfailed.
- Tests: a clean word with a 4x100 px stem the run rule marks line counts fully (m2_clean_words px = all ink; m2 px smaller); a word crossed by a line is left out; a word whose ink runs out of its box, and a word with a line 2 px from its box, are not clean, a plain word is.

## Re-gate
One process, one CityGeo, scripts in scratchpad (regate.py, pair.py). Folders rewritten (scores.json, scores.md, blocks.npz): baselines r328, old-ne/nw/se/sw, all-sheets, trial; runs control-ne-s0/1/2 (NE). `extra/` folders of the runs were not re-gated (they lack the new measure).

m2_clean_words, called_05 (px; called_09; mean; log_loss; ci_05):
- r328 0.9951 (1,719,548; 0.925; 0.9713; 0.037; [0.989, 1.000])
- all-sheets 0.9956 (0.939; 0.972; 0.036)
- trial 0.9967 (0.949; 0.971; 0.035)
- old-nw 0.9944 (292,511), old-ne 0.9565 (360,858; 0.9065; 0.9467; 0.096; [0.936, 0.9995]), old-sw 0.9812 (468,418), old-se 0.9855 (597,761)
- control-ne-s0 0.9271 (360,858; 0.890; 0.899; 0.664; [0.893, 1.000])
- control-ne-s1 0.9352 (0.913; 0.911; 0.520; [0.904, 1.000])
- control-ne-s2 0.9275 (0.848; 0.887; 0.580; [0.893, 1.000])

NE kinds for the controls (s0): zoning code 73,595 px (0.993), block number 103,172 (0.9997), street name 6,058 (1.0), other 178,033 (0.855).

## Paired NE (3 control seeds as one arm; difference = control less baseline, called_05, 713 blocks)
- vs old-ne: -0.0266 [-0.0401, +0.0025]
- vs r328: -0.0545 [-0.0806, +0.0006]
- Seed noise (largest pairwise difference over the 3 seeds): 0.0082 (m2 for comparison: 0.0136).
- For comparison m2: vs old-ne -0.0051 [-0.0231, +0.0104]; vs r328 -0.0317 [-0.0454, -0.0207]. So m2_clean_words shows the control losing about 5x more against old-ne than m2 does, about 3x the seed noise, interval just touching 0.

## Clean words (whole city, 2000 sheets), clean / all
street name 126 / 2634; block number 2657 / 4630; zoning code 559 / 2150; other 290 / 622. Few clean street names and zoning codes: most are touched by lines.

## RM-2 on p02
Not a clean word. The big vertical RM-2 is Terra word 373 (zoning code, box x 2490-2523, y 1807-1900; 1257 ink px, 266 run-rule px in its box): clean False. So m2_clean_words does not see it either (it is left out as touched or joined to a line); the control's regression there is not measured by this new measure. Other words in the rm2a window (words 335, 425, 478, 781, 69, 836) are also not clean.

## Fix round 1
- Corrected the claim above to what was checked (five measures px and called_05 unchanged; additive change). No re-gate re-run.
- Replaced the over-run test: the word 0 piece is now 36 px (under the run rule), asserted not line, so the piece test alone rejects it (RED not needed: test-only change, GREEN).
- Added: line at exactly 3 px from the box (rejected) vs 4 px (clean), both asserted to be run-rule line; a rotated (diamond) box judged by its own edges, with a line 15 px off its edge inside the grown bounding rectangle (clean) and the same line across it (not clean).
- `uv run pytest -q tests/test_textnet_linegate.py`: 18 passed.
