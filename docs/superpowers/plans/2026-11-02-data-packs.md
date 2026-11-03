# Data Packs Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `ship` and `pull` move the pipeline's data as a few large zip packs, like the answer packs, instead of one request per DVC object.

**Architecture:** A new `sfzmaps/datapack.py` packs small DVC objects into content-addressed zips. The shipped manifest says where each object lives: in a pack, or alone if it's large. `pull` and the merge gate read either the old or the new manifest format. A one-time repack moves today's objects into packs.

**Tech Stack:** Python 3 (`zipfile`, `hashlib`), the existing node scripts `site/scripts/blob-list.mjs` and `blob-put.mjs`, DVC 3.67.1, pytest.

**Spec:** None. The owner's words: "I thought we were doing option 2 from the beginning. at no point did you tell me it'd be thousands of small files. You only told me it'd be large zip files." Also memory `minimize-remote-requests`: "If 7 requests would get all the data, you should have made 7 requests."

## Measured now (this worktree's dvc.lock, PR #28)

- 10,454 objects, 10.6 GB. 7,623 are under 100 KB (0.05 GB in all); 10,241 are under 8 MB (3.4 GB).
- 213 objects are 8 MB or more (7.2 GB); the largest is 130 MB.
- Today a fresh `pull` makes about 10,450 requests, and shipping PR #28 would add about 3,700.

## Global Constraints

- **Pack rule:** objects under `SINGLE_MIN = 8 MB` go in packs of at most `PACK_MAX = 256 MB`. Larger objects are stored alone, as now. For today's data that's about 14 packs plus 213 single objects, so a fresh pull makes about 230 requests instead of 10,450.
- Names are content hashes, nothing on the store is ever overwritten or deleted, and records are written last, as `ship` does now.
- Requests: one list call per 1,000 names; uploads and downloads at most 2–4 at a time. `ship --dry-run` prints the number of requests it would make, and `ship` refuses more than 500 without `--yes`.
- Old shipped manifests (v1: objects stored alone) must still pull and pass the merge gate, so older commits and `git bisect` keep working.
- Tests: only those listed. Never read or print `site/.env.local`; node reads it.

## Decision for the owner

**Fingerprint files stay one per output.** Once packed, the 3,700 sidecars from PR #28 travel in a pack, at about 1 MB and 0 extra requests. Merging them into one file per map year would need file locking, because sheets are processed in parallel. My recommendation is to leave them per output; the alternative adds a task.

## Review Focus

- An object that's already in a pack on `main`: ship must not pack it again. Test in Task 2.
- A pull interrupted halfway: the next pull resumes, with no corrupt objects in the DVC cache. Test in Task 1 (extraction checks every member's md5, writes atomically and skips present files).
- A v1 manifest from an older commit: pull and the gate still work. Test in Task 3.
- A corrupt or wrong pack: refused before anything is written. Test in Task 1.
- Very large packs: `blob-put.mjs` sends files as one buffer, so check it handles 256 MB, or stream it. Verify in Task 4.

---

### Task 1: `datapack.py`

**Files:** Create `pipeline/src/sfzmaps/datapack.py`; Test `pipeline/tests/test_datapack.py`

**Interfaces (produces):**
- `plan_packs(objs: dict[str, int]) -> tuple[list[list[str]], list[str]]`: given md5 → size, returns (packs as lists of md5s in sorted order, each at most `PACK_MAX` bytes; single objects, at least `SINGLE_MIN`).
- `make_pack(md5s: list[str], cache: Path, out: Path) -> str`: writes an uncompressed (`ZIP_STORED`), deterministic zip (member name = md5, fixed timestamps) and returns its sha256.
- `apply_pack(zip_path: Path, cache: Path, want: set[str] | None = None) -> int`: extracts members into the DVC cache layout. Each member's md5 must equal its name; writes are atomic; members already present are skipped. Returns the number written.

- [ ] Tests: `test_pack_roundtrip_is_deterministic` (pack twice, same sha; applying to an empty cache restores the bytes); `test_apply_refuses_a_member_whose_md5_is_wrong` (nothing is written for that member, and it raises); `test_plan_splits_by_size` (a 9 MB object is single, small ones fill packs up to `PACK_MAX`).
- [ ] Implement; run `uv run pytest tests/test_datapack.py`, which should pass; commit.

### Task 2: `ship` writes packs

**Files:** Modify `pipeline/src/sfzmaps/ship.py` (`ship`, `ShipReport`, manifest); Test `pipeline/tests/test_ship.py` (create if absent; mock `listing` and `upload`)

- Manifest v2: `{"version": 2, "lock_md5", "outs", "objects": {md5: size}, "where": {md5: pack_sha}, "data_packs": {sha: bytes}, "packs": ...}`. An object not in `where` is stored alone.
- Objects that the base's manifest (or any earlier v2 manifest named in `cache-needed.json` on `main`) already places in a pack keep that pack. Only new small objects go into new packs, under `cache/datapacks/<sha>.zip`.
- Report and dry-run: the number of packs, single objects and requests.

- [ ] Tests: `test_ship_packs_new_small_objects_and_uploads_large_alone`; `test_ship_reuses_packs_main_already_has` (a second ship uploads nothing).
- [ ] Implement; run the pipeline suite; commit.

### Task 3: `pull` and the merge gate read both formats

**Files:** Modify `ship.py` (`pull`, `check_shipped`); Test `test_ship.py`

- `pull` (v2): download each needed pack once, 2 at a time, and `apply_pack` it; fetch single objects as now. With v1, behaviour is unchanged.
- `check_shipped`: for v2, every object is either in a pack listed in `data_packs` or stored alone, and `ship` confirmed each pack's size by listing.

- [ ] Tests: `test_pull_v2_applies_packs_and_singles` (a fake client serving a pack and one single object; the DVC cache ends up complete); `test_pull_v1_still_works`.
- [ ] Implement; run the suite; commit.

### Task 4: Repack and ship PR #28

- [ ] Add `sfzmaps ship --repack`: puts every small object of this dvc.lock into packs, including those already stored alone. It runs once; old single objects stay on the store.
- [ ] Check `blob-put.mjs` with a 256 MB pack, and stream it if needed.
- [ ] `uv run sfzmaps ship --repack --dry-run`. Expected: about 14 packs, 0 to 213 single uploads and under 300 requests. Then ship for real.
- [ ] Measure a fresh pull into an empty scratch DVC cache: report the requests, time and whether `dvc checkout` succeeds.
- [ ] Update `pipeline/README.md` ("Shipping the cache") and the `ship.py` docstring with the pack rule and request counts. Commit with PR #28 and push. The Cache shipped check should pass.

### Task 5: Delete what nothing uses (after PR #28 merges)

The owner asked: "be sure to efficiently delete the unused files from blobstore when done".

- [ ] Add `sfzmaps prune --dry-run` and `site/scripts/blob-del.mjs`. "Used" means named by the shipped manifest of `main`, or by an open PR branch's `cache-needed.json`: every listed pack, single object, answer pack, needed list and manifest. "Unused" means anything else under `dvc/` and `cache/`. Tiles and stats the site publishes are out of scope.
- [ ] One listing finds the candidates. Delete in batches of up to 1,000 URLs per `del()` call (the Blob SDK takes an array), one batch at a time. A final listing confirms.
- [ ] Refuse to run when `main`'s manifest is v1, and when any used name is missing from the listing.
- [ ] Old commits shipped before this change can't be pulled afterwards; say so in the README. Report the names and bytes deleted and the requests made.
