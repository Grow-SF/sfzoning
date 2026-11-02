# Versioned, shippable pipeline cache — design

Owner's request (2026-11-02): each branch or worktree can build and share data locally without
publishing it; when a branch is ready to merge, it ships what it built so the next run of `main`
starts with it, at $0. Abandoned or superseded work never ships. Squash merges only, accepted by
the owner as the cost of a cache whose versions are commits ("I don't love squash merges but if
there's a real technological need then I guess"). Calls marked *(my call)*.

## What is cached, and what "needed by a commit" means

Two things make a build expensive:

1. **Vision answers** (`data/cache/llm/`): `llm/<key>.json` (38k batch answers) and
   `llm/tiles/<key>.json` (536k per-tile answers split from batches). A key is a hash of the
   exact request (image bytes, prompt, schema, model). Answers are immutable; the store is
   put-if-absent. These cost money.
2. **Processed data** (`data/work/`, `data/out/`, ~15 GB): rebuildable at $0 from the answers,
   but it takes 30–80 minutes. DVC already versions it per commit (`pipeline/dvc.lock`).

A commit **needs** exactly the answers its processed data was made from, and the data its
`dvc.lock` names. Not the answers a run happened to touch: a stage that was already fresh
reads no answers, so recording "what this run read" would ship too little.

## Design

### 1. Answer provenance travels with the data

Every artifact made from vision answers records the keys it was made from, in a sidecar written
in the same step as the artifact: `<artifact>.answers` (sorted keys, one per line, `llm/...`
relative paths). Artifacts: `page.json` (classify), `labels.json` (label reads),
`district_labels.json`, the district legend reads, `districts_raw.json` (pattern / region reads).
The recording is a context in `sfzmaps.llm` (`with llm.recording() as keys:`) that captures
every key the wrapped code reads or writes (`ask_json`, `tile_get`, `tile_put`), so the request
code isn't duplicated. Sidecars live inside the stage outputs, so DVC versions them with the
data: **the answers a commit needs = the union of the sidecars in its `dvc.lock` outputs**.

Existing artifacts predate sidecars. *(my call)* They get one **baseline pack** of the whole
current cache (165 MB zipped): it is what today's `main` was built from (plus a little junk from
past experiments, once). Sidecars are created as artifacts are next written; `sfzmaps answers
--check` lists artifacts still missing one.

### 2. Packs: immutable bundles of answers on Blob

- A pack is a zip of answer files at their cache paths, named by the sha256 of its bytes:
  `cache/packs/<sha>.zip`, plus a key index `cache/packs/<sha>.keys.gz`.
- Committed record per pack: `pipeline/cache-packs/<sha>.json` (`sha`, `keys`, `bytes`,
  `created`, `index_sha`). One file per pack, so concurrent PRs never conflict.
- Main's answer set = the union of the packs recorded on main.

### 3. Processed data on Blob through DVC

- Pull: DVC's stock HTTP remote over the public Blob URLs (`blob` remote,
  `https://<store>.public.blob.vercel-storage.com/dvc`); no token, no wrapper.
- Push: DVC's HTTP remote can't upload to Blob (Blob takes `PUT
  https://vercel.com/api/blob/?pathname=…` with four headers; verified in `@vercel/blob` 2.8.2),
  so `site/scripts/dvc-blob-proxy.mjs`, a 127.0.0.1-only HTTP server, receives DVC's uploads
  and calls `put()` from `@vercel/blob` (multipart for large files, never overwriting), and
  answers DVC's existence checks from the public URLs (bypassing the CDN cache, as
  `publish-tiles.mjs` does). DVC pushes through the `blob-push` remote, `http://127.0.0.1:<port>/dvc`,
  configured per machine (`.dvc/config.local`). The token stays in `site/.env.local`.

### 4. `sfzmaps ship` (run on a branch when it is ready to merge)

Refuses unless: the working tree is clean (code, inputs, `dvc.lock`), `dvc status` is clean (the
data matches this commit), and the branch is ahead of `origin/main`. Then:

1. Needed answers = union of sidecars in the outputs `dvc.lock` names; missing sidecars fail
   ship (the data's provenance is unknown) unless covered by the baseline *(my call: the baseline
   is recorded as covering artifacts built before sidecars existed)*.
2. New answers = needed minus the keys of every pack recorded on `origin/main` and this branch.
   They become one new pack, uploaded, then its record is committed.
3. `dvc push -r blob-push` (through the proxy) for this commit's outputs.
4. Prints what shipped (keys, bytes, objects) and leaves a commit to push with the PR.

Ship never pays (no vision calls) and never overwrites. Re-running it is a no-op.

### 5. Getting a commit's cache (`sfzmaps pull`)

`sfzmaps pull` (replacing the Drive download): downloads the packs recorded at this commit that
aren't applied locally (applied list in `data/cache/.packs`), unpacks them put-if-absent, then
`dvc pull -r blob` for this commit's data. A fresh checkout is then `pixi install`, `sfzmaps
pull`, and `sfzmaps build` finds nothing to do.

### 6. Local branches and worktrees (the default)

Nothing changes: local work never leaves the machine until `ship`. All checkouts share one
local answer cache (safe: content-addressed, put-if-absent) and one DVC cache; a worktree gets
its commit's data with `worktree-data.sh`.

### 7. Squash merges and the merge gate

- Squash-only: the merged commit's tree is the PR tip's tree, so `main` gets exactly the
  `dvc.lock` and pack records that were shipped. Turning on "squash only" in the GitHub repo
  settings is the owner's action.
- CI check on every PR (`cache-shipped`): every pack recorded exists on Blob; every DVC object
  the PR's `dvc.lock` names exists on Blob (.dir manifests fetched and their files checked);
  and completeness: the sidecars named by `dvc.lock` are fetched (small files) and every key in
  them is in some recorded pack's index. Public reads only, so CI needs no secret.
- A PR whose `dvc.lock` conflicts with `main` must merge `main`, rebuild and ship again (the
  data is a function of both changes).

### 8. Retention

Packs and DVC objects reachable from `main` are kept. Objects shipped by PRs that were closed
unmerged are deletable after 30 days (`sfzmaps ship --gc`, token needed, owner-run).

## Non-goals

Parallel CI builds; shipping from CI (answers and data live on the developer's machine); a
DVC remote other than Blob; per-commit shipping of non-squash history.

## Costs

Blob storage ≈ $0.023/GB-month: ~15 GB data + 0.2 GB answers ≈ $0.35/month, growing with
versions until GC. A contributor's first full pull ≈ 15 GB ≈ $0.75 at $0.05/GB (or covered on a
flat-rate CDN plan); later pulls fetch only changed objects.

## Red team (2026-11-02) and rulings

An independent review read this design against the code. Its verdict: the answer side (sidecars
+ content-addressed packs) is the right shape but incomplete; the data side (DVC HTTP remote +
upload proxy) is overbuilt and carries most of the risk. Rulings, which supersede the sections
above where they differ:

1. **Data side: no proxy, no DVC HTTP remote.** `ship` enumerates the commit's DVC objects from
   `dvc.lock` and the `.dir` manifests in `.dvc/cache/files/md5/`, and uploads missing ones to
   `dvc/files/md5/...` on Blob with a Node uploader (`@vercel/blob`, never overwrite; each object's
   md5 checked against its path before upload; existence by HEAD with a cache-busting query, never a
   download). `pull` downloads missing objects with a cache-busting fetcher into the DVC cache
   (md5 verified) and runs `dvc checkout`. No server, no open upload relay, no CDN 404 poisoning
   (nothing ever requests a real object URL before it exists). Measured size: 64 outputs, 25.8 GB,
   6,268 files (not ~15 GB).
2. **Reference data is shipped too.** `data/ref` becomes a DVC-tracked output (a `refdata` stage),
   shipped like the rest; otherwise a fresh checkout re-fetches today's DataSF data, every stage goes
   stale and the rebuild can differ from main. `data/raw` stays sourced from the owner's Drive scans
   (`sfzmaps fetch`), but its hash stays a stage dep, so a changed scan shows as stale.
3. **Provenance from live artifacts only.** Needed answers = sidecars of a closed list of live
   artifacts: `page.json` (classify), `labels.json` (labels), `district_labels.json`,
   `district_legend.json`, `districts_raw.parquet` (pattern/region reads), and the height series'
   `out/<zoning>/districts/height.parquet`. A sidecar is deleted with its artifact; partial outputs
   (`districts-partial/`, `districts_raw.partial.*`) and artifacts of sheets no longer live
   (reclassified non-map, unassigned) are excluded. Deliberately excluded reads: `calibrate`
   (its result, `inputs/label_scale.json`, is committed) and the `llm.py` smoke read.
4. **Recorder.** A process-global registry with explicit, non-nested scopes, fed by `ask_json`,
   `tile_get`, `tile_put` from any thread (`llm.pmap` workers included; contextvars don't cross
   its executor). An inner scope (legend read, district labels) claims its keys, so outer sidecars
   are stable between identical runs. A tile key is recorded only if its put won (or the stored
   answer was read).
5. **Sidecar before artifact.** Written atomically first; an artifact without a sidecar is never
   "fresh" for provenance: it is replayed (6).
6. **No blanket baseline.** A $0 replay gives true provenance for current data: for the four
   skippable artifact types, the read reruns cache-only under recording; if the result equals the
   stored artifact, only the sidecar is written (artifact mtime kept). Pattern/region keys are
   re-recorded by any districts run. Only true mismatches (e.g. classify/labels whose prompt
   changed since) are listed explicitly in a committed `pipeline/cache-baseline.json` by
   (path, md5) and covered by a pack; nothing is inferred from "no sidecar".
7. **Packs carry content hashes.** Each pack index maps key -> sha256 of the answer; CI fails if two
   packs disagree on a key. Deleting an answer to force a re-ask is forbidden; change the prompt or
   context version instead (the key changes).
8. **Merge policy: up-to-date branches, not squash.** Squash isn't what makes main's tree equal the
   shipped tip; "branch up to date with main before merge" is (GitHub's strict status checks, or a
   merge queue), and it works with ordinary merge commits. Bisect over shipped states with
   `git bisect --first-parent` (merge commits on main are exactly the shipped ones). CI also
   recomputes DVC's hashes of the git-tracked deps (src, inputs, pyproject, uv.lock, pixi files,
   dvc.yaml) and fails if `dvc.lock` doesn't match: commits added or a rebase without rebuild and
   reship can't pass. Turning on strict checks is the owner's action.
9. **Writes race-free.** `ask_json` stores put-if-absent (`os.link`) and returns the stored answer,
   so two worktrees asking the same thing at once agree with what they record.
10. **Key safety.** Every key is validated against `^llm/(tiles/)?[0-9a-f]{64}\.json$` and resolved
    under `CACHE_DIR/llm` before packing; `apply_pack` rejects zip-slip paths. Only cache files and
    DVC objects can ever be uploaded.
11. **Junk out of tracked outputs.** Tiling scratch (`sfzmaps-tiles-*`, 13 GB of crashed leftovers
    found in `data/out/`) moves to a temp dir outside `data/out`, and the pattern goes in
    `.dvcignore` (a fix to the already-merged build, made separately and first).
12. **GC** keeps everything reachable from any commit on main or an open PR head.
13. **Documented:** a build's outputs can depend on which answers are cached (old-mosaic path in
    `pattern.classify`), so builds never run cache-only, and `pull` fetches the commit's packs before
    building.
