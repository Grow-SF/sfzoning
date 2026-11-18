# Final fix wave report

1. Confound: summary now defines control 0 (zones 84 px + in-box rule) and says every control-vs-old number also carries the recipe change (old Recipe() 8000 steps/full size/all known px vs DEFAULT_RECIPE 4000/half/ink+1px). New section "Post hoc decomposition" (trial baseline, in-sample on NE, linegate.paired on NE blocks). Recomputed (trial-old / control-trial):
   m1c +0.0161 [+0.0131,+0.0195] / +0.0109 [-0.0015,+0.0230]; m1a_in_box -0.0060 [-0.0334,+0.0249] / -0.2193 [-0.2466,-0.1925]; m1b_in_box -0.0147 [-0.0273,-0.0057] / -0.2577 [-0.2806,-0.2275]; m2_clean_words +0.0286 [-0.0012,+0.0421] / -0.0552 [-0.0821,+0.0017]; codes +0.0012 [-0.0017,+0.0045] / -0.0344 [-0.0396,-0.0294]. Matches the reviewer's numbers. Reading as instructed.
2. Dashed over-call: only seen facts (window holds no box/zone/point; red in six runs, grey in r328/old-ne/all-sheets/trial); one untested hypothesis; cause (zones vs in-box rule) unknown.
3. "3x": per-quarter missed-pool ratios recomputed from audit JSON: NE control 1257/449=2.8x, composites 1269/449=2.8x, old 1379/449=3.1x; SW control 799/445=1.8x, old 821/445=1.8x. "They are lines too" attributed to one person's 64-tile count.
4. Only control ran on SW: fixed; "cannot pass" -> "did not pass for any line-aware detector here, and is unlikely to" (two places).
5. SW wording: "does not pass the rule on one seed (M2 -0.0142 vs noise 0.0136)".
6. Minors: "no change beyond the seed noise" with M1c/M2 block intervals and M1c letting the variant pass old-ne; "nearly all in-box ink" (r328 M1c 0.37-0.55); grey lines qualified (panel windows; M1b in box ~0.55); M2 base is the complement of what the control leaves unknown; open list gains the composites' domain check (unlike sets) and the r328 copy_today line.
7. New section "Stages left stale by this plan", options (a)(b)(c), nothing acted on.
8. linegate.py clean_word_mask docstring: no task name; says composites use grown line_bits so their clean words are a subset.
9. Pin: PINNED_BATCHES_SEED3 is now a dict keyed ("Darwin","arm64"); the test skips with a reason elsewhere. The named in-process test (tests/test_textnet_labels_rule.py::test_lines_on_a_sheet_never_change_the_boxes_rules_batches) checks only that lines on a sheet do not change boxes-rule batches; it does NOT cover composites=0 == pre-composite code. So I added test_composites_zero_draws_what_no_bank_draws (all platforms: composites=0 with a bank is byte-identical to no bank and the bank is never asked).

Tests:
- `uv run pytest -q -n 0 tests/test_textnet_linegate.py tests/test_textnet_composite.py tests/test_build.py` -> 85 passed in 26.67s
- `uv run pytest -q` -> 1540 passed, 3 xfailed
