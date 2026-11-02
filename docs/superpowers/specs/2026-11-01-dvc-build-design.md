# Tracked, parallel pipeline builds (DVC) — design

Owner's request (2026-11-01): track pipeline builds with a build tool, after a full rebuild found
two failures that nothing caught. Owner chose DVC from a sketch and asked for a brainstorm, plan,
red team, fixes and the build, autonomously ("No questions"). Calls marked *(my call)* are mine.

## Why

Pipeline data (~30 GB) lives in git-ignored `pipeline/data/`. Nothing records which version of
it built what is live. On 2026-11-01 that let two things through:

1. **A stale copy published.** A worktree (`.claude/worktrees/cache-maybe`) had its own copy of
   `data/out/`. It built the shared districts from 10-30 height reads, and 4,304 lots' 1980–1990
   lot cards went live with wrong heights (fixed by PR #19).
2. **A missing input went unnoticed.** `data/overrides/` (the zoning corrections from PRs
   #15/#16) existed only in that worktree. Main had none; districts built there would silently
   have dropped the corrections. It was caught by hand.

Also: rebuilds were orchestrated by a throwaway Makefile; "is the data current?" had no answer
short of rebuilding; and hand work is mixed into machine folders (`work/*/*/review.json`), with
some placements existing only as machine output (1921 sections 8, 10, 11, 12 are `transform.json`
with origin "auto"; a `--force` run would replace them).

## Goals

- One committed record (`pipeline/dvc.lock`) of the content hash of every input and output of
  every build step, so a checkout knows whether its data matches the code, and which data built
  what is published.
- Hand-made inputs in git, where every checkout has them.
- `sfzmaps build`: rebuild exactly what is stale, in dependency order, up to N steps in parallel.
- Worktrees get exactly the data their commit names, from one shared cache, at no extra disk.
- A build can never spend money.

## Non-goals

- Replacing the pipeline's own per-sheet freshness checks. DVC works at series granularity;
  `sfzmaps run` keeps skipping fresh sheets inside a step.
- Remote storage. A DVC remote needs a bucket and an account; creating one is the owner's call.
  The design leaves room for it (phase 3, documented), and the Drive bundle/`pull` stay as they are.
- Publishing. `site/scripts/publish-tiles.mjs` stays a separate, deliberate step.
- Tracking the LLM answer cache (574k files) in DVC.

## Anti-goals (a failure even if it works)

- A build that deletes or rewrites the work of a step that didn't need rerunning (DVC deletes
  outputs before running a stage unless `persist: true`).
- A build that silently degrades output (e.g. a cache miss falling back to inheritance).
- Corrupting the DVC cache by editing files in place through hardlinks or symlinks.
- Losing hand-made data in the migration.

## Design

### 1. Hand-made inputs move to `pipeline/inputs/` (git)

| Input | Today | New home |
|---|---|---|
| Reviewed fits (5) | `data/work/<series>/<sheet>/review.json` | `inputs/reviews/<series>/<sheet>.json` |
| Zoning corrections (10) | `data/overrides/<series>.csv` (git-ignored; copy in `docs/research/zoning-overrides/`) | `inputs/overrides/<series>.csv` |
| Page corrections (1: 1990–96 p05 is a text page) | edited `data/work/.../page.json` | `inputs/pages/<series>/<sheet>.json`, fields merged over the machine `page.json` when it is loaded |

- One module, `sfzmaps.inputs`, owns these paths (`review_path`, `overrides_path`,
  `page_patch_path`); every reader and writer goes through it: runner, manifest, editor save,
  cli, share (bundle/pull), assemble.
- The owner's `.points` answer keys stay in `data/raw/` (they come from the Drive tree with the scans).
- `docs/research/zoning-overrides/<series>.csv` become the moved files (git mv); the research
  files beside them (`regions.csv`, `1975-c2-not-applied.csv`, README) stay. *(my call: one
  copy, not two.)*
- Bundle/pull: reviews are in git now, so the bundle stops carrying `review.json` and `pull`
  drops its keep-local-review logic. A bundle from before this change still contains
  `work/*/*/review.json`; `pull` ignores them with a warning naming where reviews live now.
  *(my call.)*
- Precious auto placements (1921 sections 8, 10, 11, 12; crossref placements) are out of scope
  for conversion; DVC versions their `transform.json` (below). Converting them to accepted
  reviews is a follow-up. *(my call: converting changes the fit, which deserves an eye.)*

### 2. DVC stages: one per edition, one per period, today, shared

DVC refuses two stages writing into one tracked directory (verified), and some commands write
across series (a height series' districts write `out/<zoning series>/districts/height.parquet`).
So a stage is the smallest unit whose writes don't cross another's:

| Stage | Runs | Writes |
|---|---|---|
| `edition@<E>` for 1921 1929 1935 1948 1960 1964 1968 1972 1975 1980 | run, paper, districts (E) | `data/work/E`, `data/out/E` |
| `period@<P>` for 1980-1990 1990-1996 1996-2000 2000-digital | run + paper for P-height, P-zoning, P-su; districts P-height then P-zoning | work/out of the three series |
| `today` | districts --series today | `data/work/today`, `data/out/today` |
| `shared` | districts --shared | `data/out/shared` |

- Every output is `persist: true` (DVC must not delete outputs before running), and
  `cache: true` (versioned in the DVC cache).
- Deps: `src/sfzmaps`, `data/raw`, `data/ref`, `inputs`, `data/work/label_scale.json`, `uv.lock`;
  `shared` depends on each zoning series' and today's `data/out/<s>/districts` (granular deps
  inside other stages' outputs work; verified).
- The LLM cache is not a dep: answers are immutable per key, and builds can't add answers.
- A stage's command is one CLI call, `uv run --frozen sfzmaps stage <name>`, so the order of
  the steps inside it is Python, tested, and the same whether DVC or `sfzmaps build` runs it.
- `pipeline/dvc.yaml` is generated from `sfzmaps.build` (`sfzmaps build --write-dvc-yaml`) out of
  `districts.series.SERIES`; a test fails if the committed file differs. One source of truth.

### 3. `sfzmaps build`: parallel, on top of DVC

DVC runs stages one at a time (no `-j`; concurrent `dvc repro` blocks on the repo lock;
verified). `sfzmaps build [-j 8] [--dry-run] [STAGE...]`:

1. `dvc status --json` names the stages whose deps, outs or command changed.
2. The run set is those plus everything downstream (from the generated DAG), limited to the
   requested stages and their upstream when STAGE is given.
3. Stages run as subprocesses, up to `-j` at a time, each when its upstream stages have
   succeeded; logs go to `data/build-logs/<stage>.log`.
4. After a stage succeeds, `dvc commit --force <stage>` records it (commits serialized; DVC
   holds a lock). A failed stage is not committed; its downstream stages are skipped and named.
5. Exit non-zero if anything failed; print a summary (ran / skipped / failed, with log paths).

`dvc repro` still works (sequential) for anyone who prefers it.

### 4. Builds never spend money

`sfzmaps stage` sets `SFZMAPS_MAX_USD=0` for itself. The spend guard runs after the cache
lookup, so cached answers are free and any live call raises `LLMQuotaError`: the stage fails,
names the series, and the summary says to run the paid read deliberately with a cap and then
`sfzmaps build` again. Not `--cache-only`, which makes region reads silently fall back to
inheritance (an anti-goal).

### 5. DVC cache and worktrees

- `cache.type = reflink,copy` in the committed `.dvc/config`: copy-on-write clones on APFS (free),
  plain copies elsewhere. Never `hardlink`/`symlink`: the pipeline rewrites files in place.
- `pipeline/scripts/worktree-data.sh <main checkout>`: in a new worktree, link `data/raw`,
  `data/ref`, `data/cache` to the main checkout, point DVC at the main checkout's cache
  (`dvc cache dir --local`), and `dvc checkout`, so `work/` and `out/` match the worktree's
  `dvc.lock` exactly, as private copy-on-write files.
- DVC is a dev dependency (Apache-2.0, AGPL-compatible) so `uv sync` installs it.

### 6. Adoption

The data on disk was rebuilt from `main` on 2026-11-01 and verified against live. Adoption
records it as-is: `dvc commit --force` every stage (hashing ~20 GB once), commit `dvc.lock`.

### 7. Phase 3 (documented only)

A DVC remote (Cloudflare R2 recommended, GCS fine) would let `dvc push`/`dvc pull` replace the
Drive bundle. Steps documented in `pipeline/README.md`; nothing configured.

## Left to the builder

Exact log formatting, stage naming within the scheme, test structure, wording in docs.

## Red team (2026-11-01) and rulings

An independent review attacked this design against the code. Its verdict was "don't build DVC
as specified"; the owner had already chosen DVC, so each finding is fixed instead. Rulings:

1. **mtimes and auto placements (claimed blocker).** Tested on APFS with `reflink,copy`:
   `dvc commit -f` leaves files in place and `dvc checkout` restores each file's mtime, so DVC
   does not trigger the runner's mtime-based reruns. The underlying risk is real anyway (a
   `--force` run refits placements that came from cross-edition matching or hand seeding), so:
   **the 24 unreviewed sheets whose rough fit has origin `crossref` get their rough fit frozen as
   an input** (`inputs/<series>/seeds/<sheet>.json`). The runner uses a seed instead of the
   automatic rough fit (rewriting `rough.json` only when its content differs), then refines as
   before. The refit pipeline stays the same; the starting point is pinned.
2. **Silent per-sheet failures.** `sfzmaps stage` fails when any sheet of its series ends with an
   `error:` score reason that it didn't have before the stage, or when `paper`/stitching skipped
   a series. Committing a stage requires it to succeed.
3. **Deps changing mid-build / two builds at once.** `sfzmaps build` takes an exclusive lock
   (`data/.build.lock`), and refuses to start when `src/sfzmaps`, `inputs/` or `pipeline/dvc.yaml`
   have uncommitted changes; it rechecks before each commit and stops committing if HEAD or those
   paths changed.
4. **Coarse deps.** Inputs are laid out per series (`inputs/<series>/...`, every series has a
   folder), and each stage depends only on its series' folders. Code deps exclude `editor/`,
   `templates/` and static assets. Documented: a stage rerun means the pipeline's own freshness
   checks decide what is recomputed inside it; re-tiling every series after a code change is the
   known cost, shown by `sfzmaps build --dry-run` before anything runs.
5. **Writes outside outs.** `out/ref/blocks.pmtiles` and `data/ref/zoning_current.parquet` are
   prepared once, serially, before any stage starts (`build.prepare()`); stages never fetch.
   `.dvcignore` excludes editor caches (`dz/`, `preview.*`, `autofill.json`) and stitch
   intermediates (`cut/`, `cut-paper/`, `*.vrt`, VRT `.txt` lists), which are rebuilt on demand.
   `share pull` writing `work/`/`out/` is documented: follow it with `sfzmaps build`.
6. **The stale publish isn't prevented by DVC alone.** `publish-tiles.mjs` refuses to publish
   unless `dvc status` reports the data up to date with the checkout's `dvc.lock`
   (`--skip-dvc-check` exists for emergencies and says so loudly).
7. **Old worktrees after migration.** The old `review.json` copies stay in `data/work` (new code
   ignores them), so an old worktree running old code can't refit the reviewed sheets.
8. **Cache growth / unstable bytes.** Shared archive gzips are written with `mtime=0`; stitch
   intermediates are dvcignored. `dvc gc` is documented as `dvc gc --all-commits --all-branches
   --workspace` only. Remaining byte-unstable fields (`generated_at`, provenance stamps) are
   small files; accepted.
9. **`label_scale.json`** moves to `inputs/label_scale.json`.
10. **Page reads bypassing the patch.** Every `page.json` read goes through one helper that applies
    patches (`pattern.py`, `stitch.py`, `cli.py` included).
11. **Missing key in a worktree** (`LLMAuthError` before the guard) is reported by the stage as
    "needs a paid read", like `LLMQuotaError`.
12. DVC runs via `uvx --from dvc==3.67.1` (pinned, not in `uv.lock`, so upgrading DVC doesn't
    restale every stage); the driver runs DVC with cwd `pipeline/`; the p05 backup exists;
    `pull` also skips old bundles' `overrides/`; the worktree script gets a real acceptance run;
    `-j` defaults to 4.
