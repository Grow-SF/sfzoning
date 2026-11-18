# Task 2 report: the training recipe and its options

Commit `cb12116` on `solver` (on top of `ccfe2b5`, the other implementer's validation-area script). Status: done.
Full suite: 1075 passed (1068 + 7 new).

## What was built

### `textnet/train.py`

- **`Recipe`:** a frozen dataclass. Its defaults are today's recipe:
  - `iterations=8000`, `max_lr=2e-3`, `pct_start=0.3`, `batch=12`, `crop=320`, `width=24`;
  - `ink_only="off"`, `half_size=False`, `heads="all"`, `weight_average=0.0`, `prefetch=False`, `decoder="full"`.
- **Checks in `__post_init__`:**
  - Types are checked, and ints and floats are coerced, so `weight_average=0` and `0.0` give one name and one saved
    form.
  - Choices are checked, and so are bounds: iterations, batch, crop and width ≥ 1; max_lr > 0; pct_start in (0, 1);
    weight_average in [0, 1).
  - `half_size` with an odd crop raises ValueError ("even").
- **`Recipe.name()`:**
  - The fields that differ from `Recipe()`, sorted by name.
  - Each is written `name=value` (`str()`, so 6e-3 becomes `0.006`); a bool switched on is its name alone.
  - They are joined by `-`. The default recipe is `"default"`.
  - Example: `half_size-ink_only=ink-iterations=2000`.
  - It compares against the class defaults (today's recipe), not `stage.DEFAULT_RECIPE`, so names stay stable when
    Task 5 changes the default.
- **`build_net(recipe)`:** builds a `TextNet` from the recipe's width, outputs (7, or 1 for `heads="text"`),
  half_size and decoder.
- **`fit(sheets, recipe, device=None, seed=0)`:** takes a list of `SheetData`, or `SheetTargets` for the quarter
  path until Task 4 removes it.
  - The two are told apart with `isinstance`, in `centre_pools` and `_crop_at`. This is a temporary bridge.
- **Crop centres:** a `Pool` keeps them sheet by sheet.
  - Each sheet's array is (n, 2) int32, with no single concatenated array. `pool[j]` uses searchsorted on cumulative
    counts.
  - The same j maps to the same (sheet, row, col) as the old concatenated pool, so the random draws are unchanged.
  - `data.centres` runs 8 sheets at a time on threads.
- **`draw(sheets, pools, recipe, rng)`:** uses the same random-number sequence as before.
- **`loss_weight`, `batch_tensors` and `to_tensors(crops, device, ink_only="off")`:**
  - `"ink"`: the loss weight is known & ink.
  - `"ink+1px"`: known & (ink or one of the 8 px around ink). The ring is computed within the crop, in numpy, on the
    CPU or in the prefetch thread.
- **`loss_fn(out, t, k, a, w) -> (loss, parts)`:**
  - `parts` is now a device tensor `[text, kind, angle]`, not floats.
  - With one output channel, the loss is the text BCE alone and `parts` is `[lt, 0, 0]`.
  - The masked-sum arithmetic is unchanged.
- **No sync on every step:** each step adds `parts.detach()` to a device sum.
  - Every `LOG_EVERY` steps, and at the last step, the mean is read with one `.cpu()`, logged as "iterations a-b",
    and checked.
  - A mean that is not finite raises `FloatingPointError("iterations a-b: ...")`.
- **`WeightAverage`:** the moving average of every floating-point state entry, the BN running statistics included.
  - It starts from zero and is updated after every step. At the end it is divided by 1 − d^t, so the steps' shares
    sum to 1 and the untrained weights count for nothing. Integer entries (`num_batches_tracked`) are the net's own.
  - It is written into the net at the end of `fit`.
- **Prefetch (`_batches`):**
  - A daemon thread draws exactly `iterations` batches from the one `rng`, in order, into a queue of `PREFETCH=2`.
    The main thread moves each batch to the device.
  - A producer error is raised in the main thread.
  - On any exit, a stop event is set and the thread is joined.
- **`save_model(out_path, net, recipe, device, seed)`:** writes `state_dict`, `recipe` (asdict), `iterations`,
  `device` and `seed`.
  - It now takes a path, so Task 4 can write to `data/solver/text/`.

### `textnet/model.py`

- `fold` / `unfold` wrap `pixel_unshuffle(2)` / `pixel_shuffle(2)`, with `FOLD = 2`.
- `TextNet(width, outputs, half_size, decoder)`:
  - With half_size, the first block takes 4 channels and the head gives `outputs × 4` channels, which are unfolded.
  - `decoder="light"` gives one conv+BN+ReLU per up block.
  - `net.multiple` is 16, or 32 at half size.
- The default net has the same modules in the same creation order, so the same state-dict keys and the same initial
  weights: 2,758,567 parameters.

### `textnet/predict.py`

- **`load_model(path, device)`:** builds the net from the saved recipe.
  - A file with no `recipe` raises ValueError ("train the detector again"); it never guesses.
  - The signature changed from `(q, device)`, and `predict_sheets` passes `text_dir(q)/model.pt`.
- **`chance`:** pads each window to `net.multiple`, so odd page edges work at half size. TILE and PAD are documented
  as even.

### `textnet/stage.py` (outside the brief's file list, needed to keep the quarter path working)

- `DEFAULT_RECIPE = train.Recipe()` was added now; Task 4 was going to add it anyway.
- `build_text(q, iterations, …)` uses `dataclasses.replace(DEFAULT_RECIPE, iterations=...)`, `fit(sheets, recipe, …)`
  and `save_model(text_dir(q)/"model.pt", net, recipe, …)`.
- The CLI is unchanged.

### `pipeline/README.md`

Two sentences changed so they stay true:

- model.pt holds the recipe.
- The non-finite check runs at each 250-step log.

Task 4 rewrites this section anyway.

## Tests (pipeline/tests/test_solver_textnet.py)

**RED:** the test module failed to import (`train.Recipe` missing). **GREEN:** 29 passed in the file, 5 s serially;
every new test takes under 0.3 s.

**New tests:**

- `test_default_recipe_trains_as_today`:
  - `asdict(Recipe())` equals today's settings, written out literally.
  - `build_net(Recipe())` has 2,758,567 parameters, measured on 78c74c1 before any change.
  - On a fixed batch of 4 crops of real `SheetData` (both synthetic maps, across the validation corner), the weight
    equals `known`, and `loss_fn` is `torch.equal` to `_todays_loss`, a verbatim copy of 78c74c1's `loss_fn` frozen
    in the test.
- `test_half_size_fold_is_lossless_and_shapes_round_trip` (Review Focus 5):
  - Folding gives a permutation of the px, and unfold(fold(x)) == x.
  - The half-size net's first block sees (n, 4, h/2, w/2), and the output has full size.
  - An odd crop is refused.
  - TILE and PAD are even.
  - A 75×101 page is scored exactly as the net on the page padded with paper to 96×128 and cropped.
  - With TILE 32 and PAD 8, the odd-sized edge windows score, and the last row and column are covered.
- `test_ink_only_loss_ignores_paper`:
  - Changing the text, kind and angle targets on paper (for "ink") or beyond the 1-px ring (for "ink+1px") leaves the
    loss and every gradient unchanged.
  - The ring counts for "ink+1px" only.
  - With "off", paper counts.
- `test_prefetch_gives_bit_identical_weights` (Review Focus 4):
  - 5 steps with prefetch on and off give `torch.equal` weights.
  - A spy on `draw` shows 10 draws: the 5 without prefetch on the main thread, the 5 with prefetch off it.
- `test_text_head_only_predicts_text`:
  - The output has 1 channel.
  - The loss equals the text BCE and `parts` = [text, 0, 0].
  - Changing kind and angle leaves the loss and gradients unchanged.
  - It trains, and `chance` scores a page.
- `test_weight_average_is_the_ema_of_weights`:
  - A spy records the net's state at each `update` (4 steps, d=0.6).
  - The last recorded state equals the plain recipe's final weights, so the average doesn't change training.
  - Every float entry equals Σ (1−d)·d^(t−s)·w_s / (1−d^t) (rtol 1e-5).
  - The integer buffers are the plain recipe's.
- `test_recipe_round_trips_through_the_saved_model`:
  - The name rules: `"default"`, the brief's example, and a full 11-option name.
  - A model trained with half_size, text head, light decoder, ink+1px, EMA, max_lr and pct_start is saved, and
    `Recipe(**saved["recipe"]) == recipe`.
  - `predict.load_model` scores identically to the trained net.
  - A file without a recipe is refused.

**Old tests changed for the new API:**

- `small_crops` (which monkeypatched `CROP`/`BATCH`) is gone; tests pass `SMALL = Recipe(batch=4, crop=64)` or
  `size=64`.
- The 3-iteration determinism test now runs on the new `SheetData` path.
- The finite-check test now expects the error at the log ("iterations 3-4", LOG_EVERY=2, the NaN at step 3, 4 loss
  calls).
- The stage tests patch `stage.DEFAULT_RECIPE` and check `saved["recipe"]`.

**Mutation checks** (script: `/private/tmp/claude-501/-Users-sbuss-workspace-sfzoning/eb18d2fc-e22a-48a6-afbc-57c31cd3e134/scratchpad/textnet/mutate.py`).
Each of these 8 breaks fails its test:

- the EMA without its bias correction;
- the EMA started from the untrained state;
- prefetch drawing from another generator;
- prefetch drawing on the main thread;
- ink+1px treated as ink;
- ink_only ignored;
- the text head still reading kind;
- the name including default settings.

## Real-data timing (CPU only, 4 torch threads, M1 Max, load average 2.3 to 4.3 during the run)

`load_sheets(TEXT_TRAIN_SERIES, True)`: 87 sheets in 25.5 s. `centre_pools`: 7,020,022 text and 11,103,644 ink
centres in 1.7 s.

| recipe | s / iteration (20 iterations) | after the first 5 | draw (s / iteration) |
|---|---|---|---|
| default (today's) | **3.13** | 3.10 | 0.009 |
| `half_size=True` | **0.84** | 0.83 | 0.009 |

- Half size is 3.7× faster on CPU.
- The default's 3.1 s per step matches the README's earlier CPU figure for the old code.
- Losses fall in both: text 0.64 → 0.55 for the default, 0.71 → 0.63 at half size.
- Script: `/private/tmp/claude-501/-Users-sbuss-workspace-sfzoning/eb18d2fc-e22a-48a6-afbc-57c31cd3e134/scratchpad/textnet/timing.py`.
- Log: `/private/tmp/claude-501/-Users-sbuss-workspace-sfzoning/eb18d2fc-e22a-48a6-afbc-57c31cd3e134/scratchpad/textnet/timing.log`.

## Concerns / for later tasks

- **Not run on the GPU (as instructed):**
  - `pixel_shuffle`/`pixel_unshuffle` have MPS kernels (checked in torch's dispatch registry, not run) and are pure
    permutations.
  - Their determinism under `use_deterministic_algorithms(True)` on mps is unverified until Task 5's two-run check.
- **The quarter path is kept by an `isinstance` bridge** in `fit` (`SheetTargets` vs `SheetData`). Task 4 should
  delete `_quarter_centres`, `crop` and the bridge along with `sheet_targets`.
- **`predict.load_model` now takes a path and refuses a model.pt without a recipe.**
  - The old `data/solver/text-<q>/model.pt` files on disk can't be loaded.
  - Only the stage reads them, after it retrains, so nothing breaks. Task 4 removes those folders.
- **The weight average also averages the batch norms' running statistics** (timm-style); they are not recomputed
  afterwards.
- **"ink+1px" is dilated within each crop:** paper at a crop's edge next to ink just outside the crop is not counted.
  Deterministic, and negligible.
- **The log line changed** from an exponential running mean ("iteration N") to the mean over the window ("iterations
  a-b").
- **Prefetch on CPU gains almost nothing:** draw is 9 ms of a 3.1 s step. On mps today's step was 0.39 s, so the
  draw is about 2% there, more at half size.
