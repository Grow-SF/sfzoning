# Final fix round: report

- Important 2: "today's recipe" now reads "the old recipe (the quarter models', `Recipe()`)" in cli.py help, trial.py,
  train.py, stage.py and the README (text, tables, tolerance rule). `text-trial --recipe` / `--set` accept `stage` as the
  first word: it starts from `stage.DEFAULT_RECIPE`, the settings after it apply on top. Without `stage`, the base is
  `Recipe()` so folder names are unchanged. CLI help says so. Tests: "stage" is DEFAULT_RECIPE; "stage iterations=2000"
  changes only iterations; no-`stage` unchanged.
- Minor 1: train.py float16 note now says 22% on one fixed batch, 0-8% in the training loop (noise).
- Minor 2: README timing now about 0.1 s a step, about 6-7 min for 4,000 steps; the old recipe's 0.39 s / 3.1 s kept, labelled.
- Minor 3: tolerance sentence rewritten as asked.
- Minor 4: README says the scores stage predicts in one turn and each trial recipe trains and predicts in one.
- Minor 6: `page.load_page` (same bytes as `textnet.data.load_page`) used by textgrid. The walls also reached
  `textnet.data` through `wallevidence.text_rings` (`word_rect`), so `word_rect` and `TYPICAL_HEIGHT` moved to
  `textnet/boxes.py` (data re-exports them). The walls' code deps now hold only `textnet/boxes.py`; dvc.yaml
  regenerated (`--write-dvc-yaml`); test_build asserts that. Note: solver-text's code hash changes (boxes.py, data.py,
  train.py, stage.py edited), so that stage reruns on the next build; outputs should be bit-identical. dvc.lock untouched.
- Minor 8: the three precision tests folded into test_solver_textnet.py; the precision file removed.
- Minor 10: `runs.LINE_MASKS`; `solve_dir("line-masks")` refused; trial uses the constant; test added.
- Left alone: Important 1 paragraph, Minors 5, 7, 9, 11. codereader/*, dvc.lock untouched.
