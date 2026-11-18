# Task 12 report (code only; no GPU runs)

Commits: 8b83a08 (data/train/tests/run_input.py), then the predict.py commit (grey reading; Task 8's 29e0286 had landed).

Built
- data.py: SheetData.grey_bits (uint8 page, default (0,0)); sheet_data/load_sheets(grey=False); crop_targets(grey=True) adds
  key "grey" = page window /255, cv2 bilinear when scaled, turned with the crop. Every other key (incl. word targets:
  axis 5-6, core 7, centre 8-9, log sizes 10-11, tag 12-15) is byte-equal with and without grey (tested, words on/off).
- train.py: Recipe.input "ink" (default) | "grey"; composites>0 with grey refused; draw/batch_tensors/to_tensors take it;
  the ink recipe's calls are unchanged (an existing test monkeypatches batch_tensors with 2 args). build_net sets net.input.
- predict.py: chance/outputs/word_outputs feed a grey net uint8 grey/255, padded with white (1), no tile skipped for lacking
  ink (no new cutoff); an ink net is unchanged; ink array given to a grey net is refused.
- Full size: already Recipe.half_size=False (the stage default is True); no code; forward tested at both with grey.
- scripts/wordread/run_input.py: item_recipe(kept, "grey"|"full"), item_run(...) for Task 8's runner (loads sheets with grey=True).
  The keep rule is in its docstring. variant.run must pass the grey page (not ink) to predict for a grey model: its call sites
  (scripts/textlines/variant.py, other agent's file) still pass ink, so a grey run is not valid until that is changed.
- Grey pages: data/work/<series>/<sheet>/page.png exist locally (data.load_page); grey = the unwarped scan, same source as the ink.

Tests: tests/test_wordread_input.py (27). Pinned Recipe dict in test_solver_textnet.py updated deliberately (+"input").
Full suite (before predict commit, test_wordread_words.py excluded): 1766 passed, 1 failed (my batch_tensors signature change vs a
monkeypatch; fixed). Afterwards the heads/input/textnet/composite/run_detector/build/textread set: 198 passed, 4 skipped.
Not done: the two comparison runs and experiments/wordread/input/decision.md (need the kept detector and GPU).
