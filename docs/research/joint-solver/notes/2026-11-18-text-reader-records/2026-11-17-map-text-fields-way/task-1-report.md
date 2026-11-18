# Task 1 report: the end-to-end measures

Commit 2ad9644 (branch solver): `pipeline/src/sfzmaps/solver/wordread/{__init__,measures}.py`, `pipeline/tests/test_wordread_measures.py`.

## Built
`wordread/measures.py`: `headline`, `box_numbers` (with `reason_fn=None`, R1), `read_numbers`, `compare`, `paired_labels`, plus helpers `label_right`, `label_box`, `holding_boxes`, `cluster_interval`, `geometric_reason`, and store loaders `reads_from_store`, `answers_from_store`, `terra_words`.
- A reading is `{"closed": {code: p}, "top": str}`; Terra's words are a frame with corners `c0x..c3y`.
- Headline: label goes to the box whose grouped rectangle holds the point; of several, the highest closed posterior (ties: nearest centre). No box = wrong. CI resamples blocks (a label with no block is its own cluster); `ci_labels` resamples labels.
- `box_numbers`: IoU on grouped boxes (as `wordboxes` does). A box matches only its best Terra word (so a merged box matches one word). Reasons from geometry: split, merged, none (no overlap), other. With `reason_fn(t, ti, bj, iou, best_word)` its codes are counted instead. Also `by_quarter`.
- `matched_to_missed` (my reading of the brief): unmatched boxes that meet a missed word at IoU >= 0.1.
- `read_numbers`: on boxes matched at IoU, our likeliest code vs the code of Terra's string (needs `forms`), plus `codes_right_vs_answers` (ours vs Terra's string, on labels whose box is matched).

## TDD
RED: with `measures.py` moved aside, `pytest tests/test_wordread_measures.py` gave an ImportError (no `measures`). GREEN: 10 passed (the 5 brief tests verbatim, plus reason_fn, unmatched by read, IoU 0.4 level, read_numbers, compare). `tests/test_build.py` 43 passed; full `uv run pytest -q`: 1550 passed, 3 xfailed. pgrep was empty before the commit.

## Baseline (data/solver/experiments/wordread/baseline/{headline,boxes,reads}.json, not committed)
Headline over the 699 answered labels with a code (719 scored minus 20 "no code there"; this is what 41.1% in `measures.json` is computed on, `every_label_closed_top1`): 286 right = **0.4092**, CI (blocks) [0.3723, 0.4460]; with a box 621, right with box 286. With `holding_box`'s nearest-centre rule it is 287 = 0.4106 (the stored 0.411). The one-label difference is the specified highest-posterior rule.

Boxes vs Terra (all 10,036 words; ours 8,794):
- IoU 0.5: matched 4,401, missed 5,635 (other 2,153, none 2,120, merged 970, split 392); ours unmatched 3,403 read as text, 989 as none; matched_to_missed 2,694.
- IoU 0.3: matched 5,686, missed 4,350 (none 2,120, other 1,078, merged 970, split 182); unmatched text 2,125, none 925; matched_to_missed 1,206.
- Zoning-code words only (2,150): matched 650 at 0.5 and 1,036 at 0.3.

Reads (zoning-code words): at 0.5, 650 matched boxes, 524 same as Terra, 121 differs; at 0.3, 1,062 matched, 641 same, 406 differs. Codes right vs answers: 0.5, 259 labels, ours 220, Terra 256; 0.3, 405 labels, ours 266, Terra 394.

## Concerns
- Headline excludes the 20 no-code labels (as the 41.1% does); n is 699, not 719.
- Highest-posterior rule gives 0.4092, not 0.4106, by one label.
- "Other" reasons are large (no box meets at 0.5 but one meets lower and is not split or merged); Task 2's reason_fn should refine them.
- `read_numbers` "same/differs" counts only forms; `matched_to_missed` is my interpretation.
- Baseline location taken as `pipeline/data/solver/experiments/wordread/baseline/` per Global Constraints.

## Fix round 1
- `matched_to_missed` is now matched / missed (None when none missed); the old near-miss count is `ours_near_missed`. Baseline JSONs rewritten. matched_to_missed, all words: 0.781 at IoU 0.5, 1.307 at 0.3; zoning-code words: 0.433 and 0.93.
- `paired_labels` raises ValueError when the arms' `block` columns differ.
- New tests: plain ratio (and None), paired_labels sign (a - b), block mismatch. 13 pass in the file; test_build 43; full suite 1553 passed, 3 xfailed. pgrep empty.
