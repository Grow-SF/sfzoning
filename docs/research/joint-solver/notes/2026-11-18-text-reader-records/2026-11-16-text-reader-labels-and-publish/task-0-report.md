# Task 0 report: fix the scoring labels before the label source changes

Status: DONE_WITH_CONCERNS (concerns below are small).

## Implemented
- `runs.py` (not `paths.py`, see concerns) holds the label locations:
  - `SOLVE_PRINTS_FILE`, `SOLVE_LABELS_DIR`: what the solve reads. T4 changes these two lines.
  - `SCORING_PRINTS_FILE`, `SCORING_LABELS_DIR`: today's chain, fixed.
  - Today all four point at the same files.
- `labels.py`: `Source(prints_file, labels_dir)`, `solve_source()`, `scoring_source()`. `readings(q, vocab, source=None)` defaults to the solve's; `top_readings(q, source)` REQUIRES a source so a score can't silently take the solve's.
- `trust.load_prints(f=None)` reads `runs.SOLVE_PRINTS_FILE`; `placement` reads `readings_{q}` from `runs.SOLVE_LABELS_DIR`. `streetside` reads prints through `load_prints`.
- Scoring: `measures.scoring_labels()` is the one function choosing the score's labels. `Table` uses it (and now keeps `label_lot`). `report.top_reading_calibration` passes `scoring_source()`. `criteria_check.per_block` uses `Table.label_lot`; its duplicate `label_lots()` is deleted. `published_compare` scores through `per_block`, so it is on the same fixed set. `scripts/spikes/reader_spike.py` scoring uses pass `scoring_source()`.
- Per-block check: `measures.check_block_split` recounts per-block label counts label by label (block of the lot under each label) in addition to the totals; `criteria_check` and `published_compare` both call it.
- Energy criterion: documented in `criteria_check` docstring; `--across-label-sources` reports criterion 1 as "not comparable" and leaves it out of the verdict (help text explains).
- `dvc.yaml` regenerated (one added dep line).

## Files changed
src/sfzmaps/solver/{runs,labels,trust,placement,measures,report}.py; scripts/incidents/{criteria_check,published_compare}.py; scripts/spikes/reader_spike.py; tests/test_solver_solve.py, tests/test_solver_trust.py; dvc.yaml, dvc.lock.

## Tests
- New: `test_the_score_keeps_its_labels_when_the_solve_reads_other_ones` (solve source moved to other prints/readings: `labels.readings` follows, `scoring_labels`, `Table`, calibration do not); `test_the_block_split_is_refused_when_labels_are_regrouped_between_blocks_with_the_same_totals` (a swap with equal totals fails; a wrong total still fails).
- Fixtures patch the new `runs` constants.
- Full suite `cd pipeline && uv run pytest`: 1430 passed, 3 xfailed. (Before dvc.yaml regeneration one test, `test_committed_dvc_yaml_is_the_generated_one`, failed as expected; fixed.)

## Byte-identity evidence
- Before any change: `cp data/solver/solve/report.json .../t0-report-before.json`; sha of every file under data/solver/{solve,labels,walls} recorded.
- `pixi run build solver-solve` (after committing code, as the build requires): "built 3 stage(s) in 11.7 min" (labels, walls, solve).
- `cmp data/solver/solve/report.json .../t0-report-before.json` -> identical.
- `diff outputs.before.sha outputs.after.sha` (31 files incl. runs/*.npz and *.log, placement.npz, labels/*.npz, walls) -> no difference.
- Because the stages were really rebuilt (not `dvc commit --force`), R8 was not needed; `dvc.lock` diff is only code hashes plus solver-labels' new dep on runs.py (no output hashes changed).

## DVC actions
Regenerated dvc.yaml (`build --write-dvc-yaml`); built via the pixi build; committed dvc.lock. No `dvc commit --force`, no gc.
`solver-text-reader` shows stale in `dvc status` for reasons that predate this work (pyproject/uv.lock, data/warp, data/work); untouched.

## Commits
- a19ca2d T0: pin the score's label set apart from the labels the solve reads
- 01bbd83 T0: dvc.lock after rebuilding solver-labels, solver-walls and solver-solve on the new code

## Concerns
1. P1 said "ONE path constant"; there are two for the solve (`SOLVE_PRINTS_FILE`, `SOLVE_LABELS_DIR`) because prints.parquet and readings live in different folders. Switching is still two adjacent lines.
2. Constants are in `runs.py`, not `paths.py`: `paths.py` is a dep of every solver stage, so editing it would have made page/lots/prints/text stale for nothing (runs.py's own docstring says this is its purpose).
3. `streetside.build_streetsides` still reads `readings_{q}` from `paths.LABELS_DIR` and writes there (it is a sibling chain stage of `trust`, which writes `LABELS_DIR`); only its prints go through the solve constant. T4 must decide how the chain stages feed the new source.
4. `check_block_split` can't recount incidents or wrong lots per block independently (they come from codes, not labels); those keep the totals check.
5. The `lab_lot` / chain stages (`trust.build_readings`) also use `load_prints()` and so follow the SOLVE prints file; fine today, relevant in T4.

## Fix report, round 1 (review: the block check was tautological)

What changed:
- `measures.score` now also returns its per-label answer: `label_lot`, `label_zoning`, `label_right` (zoning and right).
- `measures.check_block_split(split, blocks, codes, exclude, measured, name)` no longer takes per-label arrays from the caller. It calls `score(codes, Table(exclude))` itself and recounts each block's zoning and right labels from `score`'s lots and flags, with the block of each lot by `blocks`. `per_block` (criteria_check) keeps its own derivation and no longer returns `label_*` arrays, so the two are independent.
- `criteria_check.main` and `published_compare.main` pass the codes and the excluded lots.
- Test (production path): `test_the_block_split_is_refused_when_labels_are_regrouped_between_blocks_with_the_same_totals` imports `scripts/incidents/criteria_check.py`, builds the split with `criteria_check.per_block`, then makes it with one label's lot in another block (same totals); the check refuses it ("grouped into blocks differently"). The genuine split passes; a wrong total is still refused.
- Smoke on the real data: `per_block(final)` + `check_block_split` on `final` passes.

Covering tests and commands:
- `uv run pytest tests/test_solver_solve.py -q -k "block_split or score_keeps"` -> 2 passed.
- Full suite `uv run pytest tests -q` -> 1430 passed, 3 xfailed.
- (no tests exist for the scripts besides this one; criteria_check and published_compare run only the check in their main paths.)

Byte identity (measures.py is a solver-solve dep, so I rebuilt, R8 not needed):
- `pixi run build solver-solve` -> built 1 stage in 5.3 min.
- `cmp data/solver/solve/report.json t0-report-before.json` -> identical.
- `diff outputs.before.sha outputs.after2.sha` (31 files) -> no difference.
Commits: the fix, then dvc.lock (code hashes only).
