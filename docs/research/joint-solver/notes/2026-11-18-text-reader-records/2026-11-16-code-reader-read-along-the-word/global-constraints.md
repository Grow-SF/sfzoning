## Global Constraints

- **The owner's answers are the test.**
  - Never learn from an answered label or any label in its block (`Label.held_out`, `Label.held_block`,
    `labels.refuse_held`). That holds for the detector's training (R1), every measured spread (R2), and the priors
    (R4).
  - The only exception is Task 4's three calibration constants, cross-fitted.
- **The reader never sees the table's code at a label's spot.** Code names from the table's list are allowed.
- **No brute-force sweep over angles, positions or sizes.** No hard cutoffs as decisions. A truncation is a numerical
  tolerance whose dropped mass is bounded and reported.
- **Deterministic.**
  - Seeded; `torch.use_deterministic_algorithms(True)`.
  - Scoring workers run one thread each (OMP/OPENBLAS/VECLIB/MKL/NUMEXPR_NUM_THREADS=1, `torch.set_num_threads(1)`,
    `cv2.setNumThreads(1)`).
  - The detector trains in one exclusive GPU turn (`sfzmaps.gpu.gpu_turn(True, "mps")`). Never two GPU jobs at once.
- **Machine load near 20.** Size the workers to it and check `uptime` a minute into any long run.
- **Synthetic cases** draw every number (angles, heights, spacings, offsets, line widths, noise) from measurements on
  the training labels for that situation (`tests-from-measured-distributions`), and come before any real run.
- **One measure for every comparison.** On the training labels: top-1 against the training reads' consensus, and mean
  −log p of it, box and point labels separately. The comparison is the saved records of commit c626e46.
- **No paid or LLM reads.** Bootstrap from primary sources: every input is made by code in the branch.
- **Process.**
  - Tests run on the CPU (`cd pipeline && uv run pytest -q`).
  - Commit locally only, and don't push. Stage only your own files.
  - Commit messages end with `Co-Authored-By: <the model you are> <noreply@anthropic.com>` and
    `Claude-Session: https://claude.ai/code/session_01UjTgrnx2j4PhzZErrad25Q`.

## Review Focus

1. **A label printed upside down or sideways reads right,** with no turn search: the detector's direction gives the
   frame. Test in R2 and R3: `test_upside_down_and_sideways_codes_read`.
2. **A heavy line through a letter doesn't change the code.** Line px are unobserved in every hypothesis. Test in R3:
   `test_line_through_h_does_not_make_it_m`, the existing test, rerun against the new engine.
3. **A Sol point off the word's middle by the measured spread reads the full code,** not a prefix or "other". Test in
   R3: `test_off_centre_point_reads_full_code`. Every offset passes, with no xfail.
4. **A code printed beside a street name reads the code.** The fused grouping is one weighted frame among several. Test
   in R2 and R3: `test_code_beside_street_name_reads_code`.
5. **Each label is read with a detector that never saw its quarter, and no answered block is ever trained on.** Test
   in R1: `test_reader_detector_never_trains_on_its_quarter_or_held_blocks`.

---
