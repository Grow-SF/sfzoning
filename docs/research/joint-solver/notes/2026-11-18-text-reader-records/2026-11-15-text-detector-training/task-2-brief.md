### Task 2: The training recipe and its options

**Files:**
- Modify: `pipeline/src/sfzmaps/solver/textnet/train.py`, `model.py`, `predict.py`
- Test: `pipeline/tests/test_solver_textnet.py`

**Interfaces:**
- **Consumes:** Task 1's `SheetData`, `load_sheets` and `crop_targets`.
- **Produces:** a frozen dataclass `Recipe`, whose defaults are today's settings:
  - `iterations=8000`, `max_lr=2e-3`, `pct_start=0.3`, `batch=12`, `crop=320`, `width=24`;
  - `ink_only: str = "off"` (`"off" | "ink" | "ink+1px"`);
  - `half_size: bool = False`;
  - `heads: str = "all"` (`"all" | "text"`);
  - `weight_average: float = 0.0` (EMA decay; 0 is off);
  - `prefetch: bool = False`;
  - `decoder: str = "full"` (`"full" | "light"`).

  Also `fit(sheets, recipe, device=None, seed=0) -> TextNet` and `Recipe.name()`, a stable short id used in folder
  names.
- **Details:**
  - **`half_size`:** `pixel_unshuffle(2)` on the ink (4 channels), the same U-Net, a head with 4× the outputs, then
    `pixel_shuffle(2)`. Crops and prediction tiles stay even.
  - **`prefetch`:** a background thread builds batches from the one seeded rng, in order.
  - **No per-step `.item()`:** losses accumulate on the device and are logged every `LOG_EVERY`. The finite check
    runs at log time.
  - **`heads="text"`:** a 1-channel output, with no kind or angle loss.
  - **Prediction** reads `Recipe` from the saved model, so `predict` works for any recipe.

- [ ] **Step 1: Write failing tests**, on synthetic data, CPU:
  - `test_default_recipe_trains_as_today`: the default recipe's loss on a fixed batch equals the current `loss_fn`'s.
  - `test_half_size_fold_is_lossless_and_shapes_round_trip`, including an odd page size at prediction (Review
    Focus 5).
  - `test_ink_only_loss_ignores_paper`: changing the targets on paper pixels changes nothing.
  - `test_prefetch_gives_bit_identical_weights` over 5 iterations (Review Focus 4).
  - `test_text_head_only_predicts_text`.
  - `test_weight_average_is_the_ema_of_weights`.
  - `test_recipe_round_trips_through_the_saved_model`.
- [ ] **Step 2:** Implement; the tests pass. Then real-data timing, CPU only: 20 iterations of the default recipe
  and of `half_size=True`. Report the seconds per iteration for both.
- [ ] **Step 3:** Full suite. Commit.

