# Versioned, shippable pipeline cache — implementation plan

**Goal:** a branch ships exactly the answers and data its commit needs to Vercel Blob when it is
ready to merge; `main` (and anyone) pulls them by commit, at $0.

**Spec:** `docs/superpowers/specs/2026-11-02-versioned-cache-design.md`

## Global constraints

- No live vision calls in any of this work; ship and pull never call the API.
- Never overwrite a Blob object; every name is content-addressed.
- The Blob token comes from `site/.env.local`; never printed, logged, committed or passed in argv.
- Run with the pixi toolchain; `pytest` and site tests green.

## Review focus

1. A fresh stage (skipped as current) must not lose its sidecar: needed answers come from
   sidecars, never from what a run touched.
2. A pack upload that dies halfway must leave nothing recorded (record only after verified upload).
3. CDN 404 caching: no read of a not-yet-uploaded object may be cached at its real URL.
4. Ship on a dirty tree or stale data must refuse.
5. Pull applied twice is a no-op; a corrupt download is detected by sha.

Red team rulings in the spec supersede this plan where they differ (recorder design, artifact
list, replay instead of baseline, merge policy).

### Task 1: `llm.recording()` and answer sidecars
`llm.recording() -> ContextManager[set[str]]` captures keys (as `llm/<key>.json` or
`llm/tiles/<key>.json`) read or written inside it, thread-safe (label reads use worker threads).
Each answer-derived artifact's writer wraps its reads and writes `<artifact>.answers` atomically,
in the same step. Tests: every artifact type gets a sidecar; a fresh rerun keeps it; a reread
that changes keys rewrites it.

### Task 2: packs
`sfzmaps/cachepack.py`: `make_pack(keys) -> (zip path, sha, index)`, deterministic zip
(sorted, fixed dates); `apply_pack(zip)` put-if-absent with sha check; records in
`pipeline/cache-packs/`. Tests: determinism, put-if-absent, corrupt pack rejected.

### Task 0 (separate PR, first): tiling scratch out of tracked outputs
`tiles.make_pmtiles` tiles in a temp dir outside `data/out`; `.dvcignore` gets
`pipeline/data/out/*/sfzmaps-tiles-*/`; the 13 GB of crashed leftovers are removed; re-commit dvc.lock.

### Task 3: Blob I/O
Node `site/scripts/blob-put.mjs` (stdin manifest of files → `put()`, never overwrite, read-back
sha check) reused by the proxy and by pack upload; Python `public_exists(url)` with a
cache-busting query. Tests with a fake `put`.

### Task 4: DVC objects to and from Blob, directly (red team ruling 1)
Enumerate a commit's objects from `dvc.lock` + `.dir` manifests; upload missing ones with the
Node uploader (md5 checked against path, HEAD with cache-bust, never overwrite); download into the
DVC cache with a cache-busting fetcher (md5 verified) and `dvc checkout`. `data/ref` becomes a
`refdata` DVC stage. Real round trip: ship one stage, pull into a fresh worktree, compare bytes.

### Task 5: `sfzmaps ship` and `sfzmaps pull`
As in the spec; `ship --dry-run`; baseline pack command (`sfzmaps ship --baseline`, owner-run
once). Tests for refusal cases, no-op rerun, delta computation against main's packs.

### Task 6: CI `cache-shipped` check
Also recomputes DVC hashes of git-tracked deps against `dvc.lock` (ruling 8) and checks packs
never disagree on a key's content (ruling 7).
`pipeline/scripts/check_shipped.py`, a GitHub workflow on PRs touching `pipeline/`. Tests with
fixture dvc.lock and fake HTTP.

### Task 7: docs and rollout
README/CONTRIBUTING/AGENTS; drop Drive from `pull`. Rollout: $0 provenance replay, explicit
baseline of true mismatches only, first full ship of main's data; owner turns on "require branches
to be up to date" and requires the check (squash not needed: ruling 8).
