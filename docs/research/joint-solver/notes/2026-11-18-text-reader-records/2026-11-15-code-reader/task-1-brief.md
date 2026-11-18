### Task 1: The label set, the code list and the line masks

**Files:**
- Create: `pipeline/src/sfzmaps/solver/codereader/__init__.py`, `labels.py`
- Test: `pipeline/tests/test_codereader.py`

**Interfaces:**
- **Produces:**
  - **`labels.Label`:** sheet, page point, Terra box (if any), angle, crop (page px, straightened by the box angle
    with a margin), the LLM read strings, `answer` (the owner's code or None), `fold` (0/1 for cross-fitting),
    `block`.
  - **`labels.load_labels() -> list[Label]`,** from `prints.parquet` and the code-check answers. The answers join by
    the recon's rule (a case point inside a print's Terra box, or equal to a kind-2 point).
  - **`labels.training(labels) -> list[Label]`** excludes the answered labels and their blocks.
  - **`labels.code_list() -> dict[str, str]`:** printed form → code. It holds the table's code names plus the
    printed variants seen in training reads.
  - **`labels.line_mask(label) -> bool array`** over the crop: the detector's long-line masks
    (`textnet.scores.line_ink`), dilated by 1 px. Also report the coverage of line ink in label windows.

- [ ] **Step 1: Failing tests:**
  - `test_training_excludes_answered_labels_and_their_blocks` (Review Focus 1);
  - `test_alias_maps_to_its_code` (Review Focus 5);
  - `test_crop_is_straightened_by_the_box_angle`.
- [ ] **Step 2:** Implement; the tests pass.
- [ ] **Step 3:** Real run, CPU. Report:
  - the label counts (all, answered, training);
  - the code list size;
  - how much ink near labels the masks cover. If they miss lines broken by junctions, add a width-based line rule
    (spec); choose by Task 3's training likelihood.
- [ ] **Step 4:** Full suite. Commit.

