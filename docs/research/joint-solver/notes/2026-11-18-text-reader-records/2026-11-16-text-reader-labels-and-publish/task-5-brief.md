# Brief: task T5 of docs/superpowers/plans/2026-11-16-text-reader-labels-and-publish.md

### T5: Publish the solve's codes on the 2000 map, with a page of differences

- **First (the previous plan's final review):** 16 district stages (`edition-*`, `period-*`, `today`, `shared`) have
  been stale since 11-02, before this branch. Their recorded `warp.py`, `transform.py`, `fresh.py` and `llm.py`
  predate 7 commits, and their lock entries lack the `sheetwarp` dep.
  - Rebuilding `period-2000-digital` would mix those changes into the page of differences.
  - Find what rebuilding them changes, with `sfzmaps estimate` and cache-only: the `llm.py` change may move cache
    keys, so a rebuild might want paid reads. Bring the result to the owner before any rebuild.
- **What:** the `period-2000-digital` stage takes its codes from the solve (`solver-solve` upstream, a DVC link)
  instead of `districts_raw`'s, keeping its other steps (unzoned, overrides, heights, parks). It handles:
  - **lots:** today's table lots are mapped to DataSF's 2000-era lots by majority area;
  - **the 45 duplicated IDs;**
  - **spellings** (`canon`, " NCD", aliases), so that `unknown_codes.json` stays empty.

  **(My call)** over two alternatives:
  - a generated file in `inputs/`: that breaks "inputs are what a person decided";
  - a new publish folder that every reader must follow: larger.
- **The page of differences** for the owner's eye: every connected area where the new map differs from the
  published one, largest first. Each shows the scan, both codes, the table's code and the lots.
- **Measure:**
  - `published_compare` on the new published map (today's: 35,455 wrong lots);
  - all 184,241 lots coded;
  - the owner checks the differences page.
- The owner ships and merges.


## Global constraints

- **No LLM output at inference.** Terra's and Sol's saved answers may train and score. The owner's answers and
  `label_fixes.csv` may train and score.
- **Apache-2.0 code.** No new copyleft dependency. python-doctr cannot be a dependency (its opencv conflicts), so
  the text reader is re-expressed in plain torch. If torchvision is needed, it is BSD.
- **Held out by quarter,** as everything in the solver is. The answered blocks are never trained on.
- **One measure when comparing:** the solve's six measures, every-lot column, with the paired block bootstrap
  (`scripts/incidents/criteria_check.py`), against today's `final`.
- **Better is enough:** each step is kept when it is measurably better, or no worse within noise, on the agreed
  measures. No step chases every miss.
- **Load near 20; one GPU job at a time; deterministic.** Commit locally; the owner merges and ships.

