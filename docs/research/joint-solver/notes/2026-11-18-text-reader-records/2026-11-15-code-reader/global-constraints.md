## Global Constraints

- **The owner's answers are the test.** `pipeline/inputs/code-check-2000/`: 717 answered codes, 20 "no label",
  593 unflagged as the fair sample.
  - Never train on them, nor on any label in the same block as an answered label.
  - The only exception: the three calibration constants (temperature, "no code" mass, "other" mass), fitted by
    two-fold cross-fitting on the answers.
- **The reader never sees the table's code at a label's spot.** Code names from the table's list are allowed.
- **No hard cutoffs as decisions:** the pose prior is a distribution, and line pixels are unobserved, not deleted.
- **Deterministic:** seeded, `torch.use_deterministic_algorithms(True)`, forward-only pose search on mps, and one
  exclusive GPU turn (`sfzmaps.gpu.gpu_turn(True, device)`) if the GPU is used. Never run two GPU jobs. CPU is fine.
- **Fresh clone only:** no paid reads, no `pipeline/data/rnd`, no scratch inputs, no per-item stamps.
- **The same measures for every comparison:**
  - top-1 and top-3;
  - mean −log p(true);
  - a reliability table (10 bins);
  - the hard subset: box short side < 23 px, or sideways/slanted, or line-touching (ink on the box edge > 0.2).

  Reported for the reader, for each existing reader (Terra, Sol, the red-circle reads, the third read and re-check,
  the turned pass) and for J7's trust, on the same labels.
- **Process:** tests on CPU (`cd pipeline && uv run pytest -q`). Commits end with a `Co-Authored-By:` line naming
  the model you are, then `Claude-Session: https://claude.ai/code/session_01UjTgrnx2j4PhzZErrad25Q`. Stage only your
  own files.
- **Recon to reuse (scratch, read-only):**
  `/private/tmp/claude-501/-Users-sbuss-workspace-sfzoning/eb18d2fc-e22a-48a6-afbc-57c31cd3e134/scratchpad/codereader/`.
  It has the join of the answers to prints (`join.py`, `cases.pkl`), every reader's accuracy (`score.py`), the
  confusions, and `misread_labels.png`. The code goes into the package; the scripts stay scratch.

## Review Focus

1. **Test-set leakage through blocks or calibration.** Test: the training label set excludes every answered label
   and every label in an answered label's block, and the calibration fit never sees the half it scores.
2. **Line pixels.** A label crossed by a heavy line must score its true code no worse than the same label without
   the line. Test on a synthetic H crossed by a bar.
3. **Rotated and sideways labels.** The pose grid covers the box angle and its quarter turns. Test: a code drawn
   sideways reads right.
4. **"No code here".** A window of only lines and background prefers "no code". Test, plus the 20 "no label"
   answers reported.
5. **Aliases.** A printed variant ("NOE VALLEY") maps to its code, not to "other". Test.

---
