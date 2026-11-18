# Ship the 2000 map from the solve on the saved reads

**Goal:** update the published 2000 zoning map with the solve's codes, read from the saved Terra/Sol reads
(published map: 35,455 wrong lots; the solve on the saved reads: 4,543).

**Owner, 2026-11-18:** "save all of our work for this tangent, as well as a full write-up of everything we've
attempted and your analysis just now, and then we should pivot to shipping with the saved reads from Terra and Sol.
I want to get the published map updated."

This reverses the 11-16 "hold the publish until labels come from our reader" for this release. The text-reader
research is parked: its code is committed and its write-up is
`docs/research/joint-solver/notes/2026-11-18-text-reader-research.md`.

Builds on `docs/superpowers/plans/2026-11-16-text-reader-labels-and-publish.md`: T0-T3 are done, T4 is dropped
(the solve keeps the saved reads), and T5 is carried here as S4-S6.

## Global constraints

- $0. No paid or LLM calls. `uv run sfzmaps estimate` and `--cache-only` only. If a step would need paid reads, stop
  and bring it to the owner.
- Never print, log, copy or commit `pipeline/.env`, `site/.env.local`, or any Vercel/Blob token.
- Rebuild with `uv run sfzmaps build` (pixi tools), and commit `dvc.lock` afterwards. Never commit `pipeline/data/`.
  Never run a bare `dvc gc`.
- Commits:
  - Never commit while a build runs (`pgrep -fl 'sfzmaps (build|stage)'` must be empty).
  - Run `git status` right before `git add <paths>`, and commit straight after.
- Don't overwrite a published `.pmtiles` in place. A changed archive gets a new path.
- The owner runs `sfzmaps ship` and merges. We open or update the PRs.
- One GPU job at a time. Keep the load well under 100.
- Pictures say what they're for. Pages must be usable on a phone (calm default, details on tap).

## My calls (the owner can override)

- **The walls go back to Terra's boxes** for the letter share, as in `final-before` (4,543 wrong lots, 94.79% labels
  right, 862 incidents). T1's held-out-chance walls (7,001 / 94.05% / 921) existed only for the no-LLM rule, which
  this release sets aside. Keep T1's walls as an option, not the default.
- **Heights** stay as published (the separate LLM-read series). The 1996-2000 scan is left unchanged.
- Main (#63, the Apache-2.0 licence) is merged into `solver` before the release build, so the shipped tree is the
  merged tree.

## Steps

- **S1. Save the research:**
  - all code committed on `solver`;
  - the branch pointer `text-reader-research` pushed to origin;
  - the write-up committed.
- **S2. The solve on the saved reads:**
  1. Restore Terra-box walls as the default (revert 7b61feb's default, keep its option).
  2. Run `dvc status` and list every stale stage, with the reason each is stale.
  3. Rebuild the needed solver stages with `sfzmaps build`.
  4. Check the six measures against `final-before` (`criteria_check --baseline`). They must match, or every
     difference must be explained.
  5. Commit `dvc.lock`.
- **S3. The 16 district stages stale since 11-02** (`edition-*`, `period-*`, `today`, `shared`):
  - find what rebuilding them changes, with `sfzmaps estimate` and cache-only;
  - report whether any paid reads would be needed and which maps besides 2000 would change;
  - the owner decides before any rebuild that touches another map or costs money.
- **S4. Wiring (T5):** `period-2000-digital` takes its codes from the solve (`solver-solve` upstream) instead of
  `districts_raw`, and keeps unzoned, overrides, heights and parks. It handles:
  - the lots: today's table lots are mapped to DataSF's 2000-era lots by majority area;
  - the 45 duplicated IDs;
  - spellings (`canon`, " NCD", aliases), with `unknown_codes.json` left empty.

  Tests go with it.
- **S5. The page of differences,** for the owner's eye:
  - every connected area where the new map differs from the published one, largest first, with the scan, both codes,
    the table's code and the lots;
  - `published_compare` on the new map;
  - all 184,241 lots coded.
- **S6. Release:**
  1. Merge main into `solver`.
  2. Rebuild `period-2000-digital` (plus whatever S3 cleared) and commit `dvc.lock`.
  3. Check that a fresh clone would build: audit the diff for local-only paths.
  4. Push and update the PR (#62 stack: #58 / #60 / #62).
  5. The owner runs `sfzmaps ship` and merges.
