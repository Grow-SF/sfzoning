---
name: processing-map-corrections
description: Use when a GitHub issue labelled `correction` (or a downloaded sfzoning-corrections file) reports that sfzoning.com shows the wrong district code for lots on a historic map, or when asked to apply, import or review readers' map corrections.
---

# Processing map corrections

## Overview
Readers suggest corrections on the site's lot cards; the tray opens an issue with a fenced
```` ```sfzoning-corrections ```` JSON block (era, series, blklot, shown, proposed, note).
`sfzmaps corrections N` imports it. **The only question a correction answers is what the
old map prints for the lot** — not today's zoning, not what seems likely.

## Steps (from `pipeline/`, on a branch off up-to-date `main`)
1. `uv run sfzmaps corrections N` (or `--file PATH` for a downloaded file). Writes nothing to
   `inputs/`; prints kept rows, skipped rows with reasons, and writes scan crops to
   `data/check/corrections/N/`. Rows for the 1996-2000 / 2000-digital twin scan are added
   automatically (same drawing).
2. Read **every** crop (lot outlined red). Keep a row only if the printed code in that lot's
   district is legibly the proposed code. Drop rows that match today's zoning but not the scan.
3. Decide override vs label fix (table below). For a label fix, add a row to
   `inputs/<series>/label_fixes.csv` (`sheet,x,y,code,fix_code,fix_x,fix_y,note`: x, y = the
   label as read, page px, from `data/work/<series>/<sheet>/district_labels.json`; the lot's
   sheet is `source_sheet` in `data/work/<series>/districts_raw.parquet`) and drop those rows.
4. `--apply` takes every row it reads. If any row is rejected:
   `gh issue view N --json body -q .body > /tmp/corrections-N.md`, delete those rows from the
   JSON block, and use `--file /tmp/corrections-N.md`. Then
   `uv run sfzmaps corrections N --apply [--file ...]`. Writes `inputs/<series>/overrides.csv` (note names the issue, shown on the
   lot card) and review cases in `inputs/district_cases.csv`. Commit `inputs/`.
5. Build from the repo root: `pixi run build -j 8` ($0; no `--dry-run` flag exists).
   If `shared` fails with "park_maps.csv is out of date", run `uv run sfzmaps parks --maps`,
   commit `inputs/park_maps.csv`, build again. A label fix may need a capped paid region read:
   `SFZMAPS_MAX_USD=1 uv run --env-file <main checkout>/pipeline/.env sfzmaps districts --series S`.
6. Look: `uv run sfzmaps districts --map S --against <snapshot dir>` (take
   `--snapshot <dir>` before step 4) and check the corrected spot.
7. `uv run pytest`, commit `dvc.lock`, then
   `SFZMAPS_ENV_FILE=<main checkout>/site/.env.local uv run sfzmaps ship` (a worktree has no env files).
8. PR with the kept/skipped rows and why, and the crops: "Closes #N" if every row is resolved,
   else "Refs #N" (the issue stays open for the rest). CI green → merge.
9. Publish from main's data: in `site/`,
   `node --env-file=<main checkout>/site/.env.local scripts/publish-tiles.mjs` (`--dry-run` first);
   it refuses unless `dvc status` is clean — don't run stages by hand after the build. Commit
   `site/lib/published.json`, PR, check the Vercel preview by eye, merge (auto-deploys).
10. Comment on the issue: what was applied, what was skipped and why.

## Override or label fix?
| Situation | Do |
|---|---|
| A few lots; the label is read right but the region is wrong | Override (step 4) |
| Many lots wrong because one printed label was misread (crop shows the code printed elsewhere read as something else) | `label_fixes.csv` row (recode/move/drop/add) — fixes every lot it feeds and keeps the reader honest |
| Proposed code is today's, scan prints the shown code | Reject |
| Crop illegible / district unclear | Ask on the issue; don't apply |

## Never
- Apply a row without looking at its crop.
- Delete cached answers to force a re-read; edit inputs instead.
- Print, copy or commit anything from `.env` / `.env.local`.
