### Task R1: The reader's detector (stage `solver-text-reader`)

**Files:**
- Modify: `pipeline/src/sfzmaps/solver/textnet/data.py`
  - a generalised hold-out: `SheetData.held`, a tuple of map-area tests, of which today's validation area is one, so
    trials and the `solver-text` stage behave exactly as before;
  - a wider unknown region around Sol points.
- Modify: `pipeline/src/sfzmaps/solver/textnet/predict.py`: `outputs(net, ink, device) -> (chance, kind, direction)`.
- Create: `pipeline/src/sfzmaps/solver/textnet/reader_stage.py`: `build_reader_text(device=None) -> dict`.
- Modify: `pipeline/src/sfzmaps/build.py` (the stage), `pipeline/src/sfzmaps/cli.py` (`sfzmaps solver text-reader`),
  `pipeline/src/sfzmaps/solver/paths.py` (`READER_TEXT_DIR = data/solver/text-reader`), `pipeline/README.md`.
- Test: `pipeline/tests/test_solver_textnet.py`, `pipeline/tests/test_build.py`

**Interfaces:**
- **Produces** `data/solver/text-reader/`:
  - for each 2000 sheet:
    - `<sheet>/chance.png` (uint8, 0–255);
    - `<sheet>/kind.npz` (uint8 per px, the argmax kind, and float16 probabilities of the 4 kinds);
    - `<sheet>/direction.npz` (float16 cos and sin);
    - each sheet's px take the model whose held-out quarter holds them;
  - `models/<q>.pt`;
  - `held.json`: the held-out blocks, and the unknown radius with its measurement.
- **The unknown radius** around Sol points (today `READ_POINT_RADIUS = 30`): the longest printed code word among
  training Terra boxes of kind "zoning code", in page px, measured and recorded. Use it for every map.
- **Hold-outs are by map area,** through each px's map point (the way `in_validation` works):
  - quarter `q` by `grid.quarter_of`;
  - held-out blocks by the lot under the px (`data/solver/lots`), on every training map at once.
- **The recipe is `stage.DEFAULT_RECIPE`, seed 0, on `paths.TEXT_TRAIN_SERIES`.** `solver-text` and its outputs are
  untouched.

- [ ] **Step 1: Failing tests.**
  - `test_reader_detector_never_trains_on_its_quarter_or_held_blocks`: a fit's drawn crops never centre in, and its
    loss weight is 0 on, quarter q's area and on held-out blocks' lots, on two synthetic maps.
  - `test_unknown_around_sol_points_covers_the_word`: px within the radius of a point are unknown, and px beyond it
    are not.
  - `test_outputs_returns_chance_kind_and_direction`: shapes; probabilities sum to 1; |(cos, sin)| > 0.
  - `test_reader_stage_deps` (test_build): the stage reads the training maps' warps, prints, lots and the code-check
    answers, and its code deps are its import closure.
- [ ] **Step 2: Implement.** The tests pass, and the full suite passes.
- [ ] **Step 3: Real run.** `uv run sfzmaps solver text-reader`, in one exclusive GPU turn: about 4 × 7 min of
  training plus prediction. Report:
  - the minutes;
  - the unknown radius;
  - for each quarter model, its words found on its own held-out quarter (the `scores` measure);
  - a picture of one sheet's direction field over a dense area (say what it shows).

  Then run it twice and check the outputs are identical.
- [ ] **Step 4: Commit.**

