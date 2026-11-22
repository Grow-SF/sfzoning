# Downloads: Separate Data and Text Change Checks — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A README/LICENSE-only change never runs GDAL or ogr2ogr, and rebuilding the same data gives byte-identical images.

**Architecture:** Each download item gets two hashes in `index.json`: `data_fingerprint` (sources, that layer's data-writing code, `EXPORTER_VERSION`) and `fingerprint` (`data_fingerprint` + the text/packaging code). `run()` rebuilds data only when `data_fingerprint` changes; a text-only change rewrites citywide sidecars or re-zips vector data from the existing zip. The publish script treats a citywide entry as changed when the image or either sidecar changed.

**Tech Stack:** Python (pytest, GDAL CLI via pixi), Node publish script (vitest).

**Spec:** `docs/superpowers/specs/2026-11-18-downloads-split-change-checks-design.md` (builds on `2026-11-18-gis-download-names-design.md`).

## Global Constraints

- `TIFFTAG_IMAGEDESCRIPTION` = `"<series> edition"` exactly; `TIFFTAG_COPYRIGHT` unchanged.
- Text-only change: no `gdal_translate`, no `ogr2ogr` call (asserted in tests).
- Same sources + same data code → same image SHA-256 (asserted with a real GDAL run on a small fixture).
- `made` = the day the item's files last changed (either check).
- Published files are never overwritten; a citywide entry with any changed file (image, README, LICENSE) is versioned anew, all three under the new date.
- `dvc.yaml`/`dvc.lock` unchanged (`downloads` is outside the build's code deps).
- Checks: `cd pipeline && uv run pytest tests/test_downloads.py -q` while iterating, full `uv run pytest -q -n 8` before committing; `cd site && npm test && npx tsc --noEmit && npm run lint`.

## Review Focus

1. An index written by the previous code (no `data_fingerprint`): every item is fully rebuilt, never "text-only". (Task 1)
2. A text-only change when the output file is missing on disk: full rebuild, not a crash. (Task 1)
3. A vector zip re-made text-only keeps every data member byte-identical and in the same order, with the fixed zip date. (Task 1)
4. A data-code change for one layer (e.g. `vectors.py`) rebuilds only that layer's items, never the citywide images. (Task 1)
5. A citywide entry whose image is unchanged but whose README changed is published as a new version (image, README, LICENSE under the new date); one with nothing changed is untouched. (Task 2)

---

### Task 1: Pipeline: two change checks, text-only paths, date-free image

**Files:**
- Create: `pipeline/src/sfzmaps/downloads/package.py` (zip writing for vectors, README/LICENSE sidecars for citywide), `pipeline/src/sfzmaps/downloads/sheets.py` (move `sheet_entries`, `_natural`, `_md5` out of `rasters.py`; update imports)
- Modify: `pipeline/src/sfzmaps/downloads/__init__.py`, `rasters.py` (image only), `vectors.py` (layers only: returns the data member files, no zip, no README)
- Test: `pipeline/tests/test_downloads.py`

**Interfaces:**
- `DATA_CODE: dict[str, list[Path]]` = `{"citywide": [rasters.py], "lots": [vectors.py], "districts": [vectors.py]}`; `TEXT_CODE = [readme.py, readme-template.txt, package.py]`.
- `data_fingerprint(item, exporter_version) -> str`; `fingerprint(item, exporter_version) -> str` (= hash of data_fingerprint + TEXT_CODE digests).
- `rasters.write_image(item, out: Path) -> None` (COG only); `vectors.write_layers(item, tmp: Path) -> list[Path]`.
- `package.write_zip(out: Path, data_members: dict[str, bytes], item, updated, commit) -> None`; `package.data_members(zip_path: Path) -> dict[str, bytes]` (every member except README.txt/LICENSE.txt); `package.write_sidecars(out: Path, item, updated, commit) -> list[str]` (returns the two index-relative paths).
- Index `files[k]` gains `data_fingerprint`.

- [ ] **Step 1: Write the failing tests**
  - `test_text_only_change_rewrites_sidecars_without_gdal`: run once (fake exporters as in `fake_exports`, plus counting `rasters.write_image`/`vectors.write_layers` calls); change a TEXT_CODE file's hash (monkeypatch the digest or point TEXT_CODE at a temp copy and edit it); run again on a later day → `write_image` and `write_layers` call counts unchanged; citywide README text changed; `.tif` bytes unchanged; `made` = the later day for every item.
  - `test_text_only_vector_rezip_keeps_data_members`: real `package.write_zip` on a fixture zip → after a text-only run, every non-README/LICENSE member's bytes and member order equal the original; README is the new text.
  - `test_old_index_without_data_fingerprint_rebuilds_fully` (Review Focus 1) and `test_text_only_with_missing_output_rebuilds_fully` (Review Focus 2).
  - `test_vector_code_change_does_not_touch_images` (Review Focus 4): change only `vectors.py`'s digest → `write_image` not called; `write_layers` called for every vector item.
  - `test_citywide_image_is_deterministic_and_dateless`: with real GDAL (pattern of `test_citywide_is_cog_with_license_tags`), export the same small fixture twice with different `updated` → identical SHA-256; `gdalinfo` shows `TIFFTAG_IMAGEDESCRIPTION=<series> edition` and no date.
  - Update `test_fingerprint_covers_the_exporter_code` / `test_exporter_code_is_the_real_exporter_files` to the new DATA_CODE/TEXT_CODE split.
- [ ] **Step 2:** `cd pipeline && uv run pytest tests/test_downloads.py -q` → new tests fail.
- [ ] **Step 3:** Implement. In `run()`: if `old` exists, `old.get("data_fingerprint") == dfp`, the output and its extras exist, and the path matches: when `old["fingerprint"] == fp` keep as today; else text-only (citywide → `write_sidecars`; vector → `write_zip(out, data_members(out), ...)`), new `sha256`/`bytes`/`made`. Otherwise full rebuild. Store both fingerprints.
- [ ] **Step 4:** `uv run pytest -q -n 8` → all pass; `git status pipeline/dvc.*` clean.
- [ ] **Step 5:** Commit: `pipeline: downloads split data and text change checks; date-free citywide images`.

### Task 2: Publish: citywide versioned when any of its files changed

**Files:**
- Modify: `site/scripts/publish-downloads.mjs` (`plan()` keep rule; manifest entries gain `extraSha256`), `site/lib/downloadRows.ts` (type)
- Test: `site/tests/publish-downloads.test.ts`

**Interfaces:**
- Consumes: sidecar sha256s already computed in `main()` (`hashSidecars`) and passed to `plan()`.
- Produces: `ManifestFile.extraSha256?: Record<"README.txt" | "LICENSE.txt", string>`; keep rule = `isVersioned(old) && old.sha256 === e.sha256 && sameSidecars(old.extraSha256, e's sidecar sha256s)`.

- [ ] **Step 1: Write the failing tests** (Review Focus 5)
  - Citywide entry with the same image sha256 and the same sidecar sha256s → kept: no uploads, no new versions.
  - Same image sha256, README sha256 changed → new versioned names for image, README and LICENSE under the index `made` date; uploads for all three plus their `.sha256`; one new `Version` (the image).
  - A manifest citywide entry without `extraSha256` (written before this change) → versioned anew.
- [ ] **Step 2:** `cd site && npx vitest run tests/publish-downloads.test.ts` → new cases fail.
- [ ] **Step 3:** Implement; write `extraSha256` on every citywide manifest entry.
- [ ] **Step 4:** `npm test && npx tsc --noEmit && npm run lint` → pass.
- [ ] **Step 5:** Commit: `site: publish a citywide image anew when its README or LICENSE changed`.

### Task 3 (controller): finish the publish

- [ ] **Step 1:** `cd pipeline && pixi run uv run sfzmaps downloads` (full export, $0).
- [ ] **Step 2:** Run the export again immediately → every item logs `same` (nothing rewritten). Then touch only the README template locally (do not commit), run with `--series 1921` → log shows the citywide and vector items rewritten with no gdal/ogr2ogr processes (watch `pgrep`), then `git checkout` the template and re-run `--series 1921` to restore.
- [ ] **Step 3:** Continue the parent plan's Task 7: dry run, show the counts, publish, link spot checks, `npm run build`, commit the JSON files, mark PR #65 ready.
