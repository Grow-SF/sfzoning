# Saved reads

Every answer a vision model gave this project that a later step is built from, and how each was asked for. The later
steps are the sheet warp (PR #58), the joint solver (PR #57) and the text detector. That is 33,194 answers, from Terra
(`gpt-6.1-terra`) and Sol (`gpt-6.1-sol`), asked between 2026-10-30 and 2026-11-11.

The answers are fixed input (decision of 2026-11-11). Nothing here asks a model again unless you choose to (see
[Redo a read with a newer model](#redo-a-read-with-a-newer-model-paid)).

Made by `pipeline/scripts/saved_reads` (`record_all.sh`, then `python -m saved_reads package`).

## Files

| File | What it holds |
|---|---|
| `reads/<read>.jsonl.gz` | One line per answer. Fields are listed under [One record](#one-record). |
| `templates.jsonl` | Each read's prompt, with what varies blanked out: every number becomes `#`, and the code asked about becomes `"{code}"`. A record's `prompt` field names its template. |
| `prompts.jsonl` | Every exact prompt (339), with its sha256 and template. |
| `schemas.jsonl` | Every answer format (JSON schema) the model was held to. |
| `runs.jsonl` | Every recorded run: script and its git location, arguments, settings, and the script's top-level values. |
| `images.manifest.tsv` | Every image's readable name, sha256 and size. |

**The images** are 24,437 files, 2.0 GB, so they are not in git. They live in `pipeline/data/saved-reads/images/<image>`,
and a replay rebuilds them (see [Check it](#check-it-no-cost)). Each name says what the image is:
- `terra-page-boxes/2000-digital-zoning/2000-digital-zoning-p01/tile_1519_0000_768x768.png` is a tile of that sheet's
  `page.png`: its top-left corner at page px (1519, 0), 768 × 768 px.
- `code-reads/leader-map/2000-digital-zoning-p07/at_1400_2998.png` is the crop around the code printed at page px
  (1400, 2998).
- `code-reads/recheck/…/case_r0123_at_….png` is the same, for code-check case r0123.
- `pipeline-label-reads/<series>/<sheet>/tile_X_Y_WxH_at_scale_S.png` is a tile cut from the page scaled by S.
  Its `page_box` in the record is in page px.

## The reads

| Read | Answers | Model, effort | What the model was shown | What it was asked | When (UTC) | Made by | Used by |
|---|---|---|---|---|---|---|---|
| `terra-page-boxes` | 14,829 | Terra, low | 768 × 768 px tiles (160 px overlap) of each whole page: 179 sheets of the 9 series 1960–2000 | every piece of printed text: text, kind (zoning code, block number, street name, other), box, direction, angle | 11-06 … 11-08 | `prob-warp:…/allyears/textboxes.py` | warp: the codes it fits and erases before prob6; solver (2000); detector: training targets |
| `terra-turned-tiles` | 702 | Terra, low | the 2000 tiles turned a quarter clockwise | same as page boxes | 11-06 … 11-11 | `k8/textpass.py cw` | solver: extra readings and prints; detector: "unknown" areas |
| `code-reads` | 6,592 | Terra and Sol, high | a 320 px square around each of the 1,648 use codes on the 2000 sheets, shown at 2×, with the code circled in red | t1, the leader map: what code is circled, and does a leader line carry it elsewhere (and where); t2, the re-check: read the code nearest the centre | 11-05 | `prob-warp:…/codes2000/leaders2.py map`, `reread.py all` | warp: the 2000 code list (`codes_final.json`), with the owner's answers |
| `leader-reads` | 3,296 | Sol, medium and low | the same crops | leader read (t1 medium, t2 low) | 11-05 | `zoning-lines-rnd:…/leaders.py sheet` | the leader-check page's cases |
| `pipeline-label-reads` | 3,866 | Sol, low | the pipeline's tiles of each district map page (245 sheets, 14 series), at the series' district scale | every district label: code, kind, point, confidence | 10-30 … 10-31 | the pipeline's `districts/labels.py` | warp: the 1960–1990 code lists; solver: the second reader; detector: "unknown" areas |
| `terra-place-reads` | 106 | Terra | enlarged tiles where an old solve was unsure, upright and turned | same as page boxes | 11-11 | `k8/placeread.py` | detector: "unknown" areas |
| `terra-place-asks`, `…-split` | 472, 508 | Terra | enlarged crops of an old solve's places, tinted yellow, upright and turned | which codes are printed on or beside the tinted lots | 11-11 | `k8/placeask.py` | detector: "unknown" areas |
| `terra-district-asks` | 390 | Terra | crops of an old solve's districts that hold no print, tinted | which code the map gives the tinted area | 11-11 | `k8/distask.py` | detector: "unknown" areas |
| `third-read`, `third-reread` | 1,865, 418 | Sol (the pipeline's default) | a close crop around each printed code | what code is printed, and does a leader line point it elsewhere | 11-10 | `k8/read3.py`, `read3r.py` | solver: extra readings, and the leader flags in its arrow step |
| `terra-rotation-pilot` | 150 | Terra | a 30-crop pilot of turned reads | — | 11-11 | `k8/rotpilot.py` | nothing |

Not here: the pattern-map reads of 1921–1948 (the pipeline's `districts/pattern.py`). The pipeline ships them with its
build and lists them per output (`.answers` sidecars).

## One record

```
read, series, sheet, image          what was read, and the image's readable name (above)
page_box | at | case | place        where the image sits on the page (page px), the point it is about,
                                    the code-check case, or an old solve's place number
variant, scale                      turned / upright / cw; the label reader's page scale
model, provider, effort, params     the model and call settings
prompt, prompt_sha256               the prompt's template, and the exact prompt (prompts.jsonl)
answer_format                       sha256 of the answer schema (schemas.jsonl)
asked                               when the answer was stored (UTC): the cache file's time
script                              source (branch@commit:path, or the rescued scratch copy), git_blob,
                                    args, env
matched                             "replayed" or "by time" (below)
answer                              the answer, exactly as the model gave it (coordinates in the image's px)
image_sha256, cache_key             the exact image bytes, and the answer's file in pipeline/data/cache/llm
```

## How each record was matched

- **`replayed`** (29,381 answers). The script was run again with the API blocked: every question was answered from
  the cache, and every lookup was recorded. The record's model, prompt, answer format and image hash together to
  exactly its `cache_key`, so they are the ones that produced the answer.
- **`by time`** (3,813 answers). These questions cannot be asked again exactly:
  - the third read placed each code using that day's sheet transforms;
  - the place and district reads took their places from old solves;
  - the place and district scripts were edited after their paid runs.

  So each answer is matched to the paid run that made it by when it was stored, and by its fields. `timed_reads.json`
  (beside the code) gives each run's command, time window and session-transcript line. These records have no image
  or prompt.

  The files those runs wrote hold their decoded results: `pipeline/data/rnd/scratch-2026-11-13/k8/pNN/textboxes2_*.json`
  and `zon2000/read3.npz`, `read3r.npz`.

## Check it (no cost)

**Quick: the package against itself.** This needs only this folder and the images. For every `replayed` record it
rebuilds the cache key exactly as `sfzmaps.llm` does, from the record's own provider, model, prompt text, answer schema,
call parameters and image bytes, and compares it with `cache_key`. It also checks every image against the manifest.
```
cd pipeline/scripts && ../.venv/bin/python -m saved_reads verify
# {'key rebuilt exactly': 29381, 'by time (no image to check)': 3813}   mismatches: 0   (2026-11-13)
```

**Full: everything again from the scans.** From a clone of this branch, in `pipeline/`:

1. **Data.** Set up the pipeline data for the scans and the pipeline's own outputs: `scripts/worktree-data.sh`, or
   `uv run sfzmaps pull`.
2. **Answers.** Put this package's answers into the answer cache. They are not in a shipped answer pack yet, so
   `pull` does not bring them:
   ```
   cd scripts && ../.venv/bin/python -m saved_reads restore-cache && cd ..
   ```
3. **Replay** every script with the API blocked. Every lookup must say `0 not in the cache`.
   ```
   zsh scripts/saved_reads/record_all.sh data/saved-reads/check
   ```
   It needs the branches `prob-warp` and `zoning-lines-rnd` for the scripts. The `k8` reads also need the rescued
   scratch folder `pipeline/data/rnd/scratch-2026-11-13` (see [Gaps](#gaps)).
4. **Package** the replay and compare it with this one:
   ```
   cd scripts && ../.venv/bin/python -m saved_reads package --calls ../data/saved-reads/check \
     --out ../data/saved-reads/check-package --images ../data/saved-reads/check-images && cd ..
   for f in saved-reads/reads/*.jsonl.gz; do
     diff <(gunzip -c $f | jq -c 'del(.asked, .script)') \
          <(gunzip -c data/saved-reads/check-package/reads/${f:t} | jq -c 'del(.asked, .script)') > /dev/null \
       && echo "same  ${f:t}" || echo "DIFFERS ${f:t}"
   done
   diff saved-reads/images.manifest.tsv data/saved-reads/check-package/images.manifest.tsv && echo "images same"
   ```
   `asked` is left out because restored answer files carry the restore time; `script` because the branches' commits
   move.

## Redo a read with a newer model (paid)

1. **Count first.** Each read's answer count above is its number of calls; `terra-page-boxes` is 14,829 tiles.
   Price it before you start. AGENTS.md: changing a model or prompt means paying again.
2. **Run** the read's script live, with another model and a spending cap. `record_all.sh` shows each script's
   arguments and settings.
   ```
   cd pipeline/scripts
   SR_MODEL=<new model> SR_EFFORT=<effort> SFZMAPS_MAX_USD=<cap> \
     uv run --env-file <the API key file> python -m saved_reads run --live --out ../data/saved-reads/<new> \
     <script> <args>
   ```
   - `--live` refuses to run without `SFZMAPS_MAX_USD`.
   - `SR_MODEL` and `SR_EFFORT` override the model even in scripts that hard-code it.
   - New answers get new cache keys; the old ones stay.
3. **Package** the new folder (`python -m saved_reads package --calls ../data/saved-reads/<new> --out <folder>`).
   Compare it read by read with this one; images with the same name show the same piece of the page.

## Gaps

- **The `k8` scripts and their inputs** live only in the rescued scratch folder
  `pipeline/data/rnd/scratch-2026-11-13` (git-ignored) and on the local branch `rnd/scratch-archive`. Their inputs
  include old solves' place files and that day's transforms.
- **The answers are not shipped.** They aren't in an answer pack yet, so `sfzmaps pull` doesn't bring them;
  `restore-cache` does.
- **One second pass has no maker found.** `textboxes2_cwx` was found only as a file, and the detector reads it.
- **`by time` records have no image or prompt** (see above).
- **`asked` is the cache file's time.** For an answer applied from a shipped pack, it's when the pack was applied.
