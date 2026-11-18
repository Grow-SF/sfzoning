# Task 5b report: Recipe.precision

Changed: train.py (`Recipe.precision` "float32"|"float16" in CHOICES and docstring; module docstring sentence; `fit` runs the forward under
`torch.autocast(device_type, float16)`, `.float()` on the output before `loss_fn` outside the autocast, `torch.amp.GradScaler(device_type, enabled=half)`:
scale, backward, step, update). predict.py: one comment in `load_model`. New tests/test_solver_textnet_precision.py (3 tests, pass).

GradScaler finding (torch 2.15.0): enabled on cpu and mps (cuda unavailable here). BUT `scaler.step` calls `.item()` on its found_inf
each step for a non-fused optimizer (AdamW default), so it waits for the device every step on mps. That can cancel part of the 22% gain;
not measurable here (no GPU run). Documented in the Recipe docstring. Option if it costs too much: a fixed loss scale with no check
(no sync, deterministic), or fused AdamW. float32 runs use a disabled scaler (no-op, same weights).
CPU timing (batch 4, 128 px, median of 8 steps): float32 0.32 s; float16 autocast without scaler 6.87 s; with scaler 7.68 s (+12%).
CPU float16 autocast works but is ~20x slower than float32 on CPU (not relevant to mps); scaler overhead there is small.
Repeatability test runs on CPU float16 autocast (supported) and passes.

Tests: new file 3 passed. Full suite: 1101 passed, 3 skipped, 1 failed:
tests/test_solver_textnet.py::test_default_recipe_trains_as_today pins the dict of Recipe fields, which now has "precision": "float32".
That file is not mine to edit; the controller must add `"precision": "float32"` to the dict at line ~493. No test_codereader failures.
