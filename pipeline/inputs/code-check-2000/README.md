# The owner's answers on the 2000 code-check pages

Decisions a person made, kept as input. Nothing in the build reads this folder.

| File | What |
|---|---|
| `codecheck_cases_published.json` | The 1,648 code-check cases as published (one per use code the pipeline found on the 2000 sheets: sheet, code, page position, and the order the page showed them in: the models' disagreements first, flagged). Made by `scripts/saved_reads/sources/codecheck_build.py`. |
| `codecheck_answers.json` | The owner's answers to 743 of them, 2026-11-05: `ok` (the case's code is right, 557), `fix` (the right code, 160), `nolabel` (no code there, 20), `unsure` (6). |
| `labeler_cases_published.json` | The 145 leader-check cases: a 640 px crop around a code. |
| `labeler_answers.json` | The owner's answers to 50 of them, 2026-11-05: `point` (where the code's leader points: `px`, `py`; 8), `none` (no leader, 26), `nolabel` (4), `unsure` (12). |

Used by: the saved reads' re-check scoring (`saved_reads.rerun.compare`) and the replay of the code reads
(`scripts/saved_reads/record_all.sh`). These answers moved here from `prob-warp:pipeline/scripts/prob-warp/codes2000/answers/`.
