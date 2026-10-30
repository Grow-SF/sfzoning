# Tuning scripts (Task 13)

Diagnosis and tuning tools behind `docs/superpowers/notes/2026-10-29-tuning.md`. They read the
work tree (`pipeline/data/work/<series>/<sheet>/`) left by `sfzmaps run` and make no API
calls. Images and caches go to `pipeline/data/out/tuning/` (git-ignored). Run from
`pipeline/`. Scripts that plot need matplotlib, which is not a project dependency:
`uv run --with matplotlib python ...`.

First produce the work tree (cached labels, so no API spend once labels exist):

    uv run sfzmaps run --year 1948 --force
    uv run sfzmaps run --year 1960 --force
    uv run sfzmaps eval --year 1948
    uv run sfzmaps eval --year 1960

## Tuning

| script | what | example |
|---|---|---|
| `harness.py` | re-run refine -> classify_changes -> score with patched constants | `uv run python scripts/tuning/harness.py 1948 all --smooth 0.1 --coh 6 --minc 4` |
| `cv.py` | in-sample vs block-held-out error of the final warp vs smoothing | `uv run python scripts/tuning/cv.py 1948 1` |
| `leverage.py` | how far one 40 ft GCP / one 15 ft block pulls the warp vs smoothing | `uv run python scripts/tuning/leverage.py 1948 1` |
| `classify_trace.py` | real-change count per classify_changes round (oscillation) | `uv run python scripts/tuning/classify_trace.py 1948 1` |

## Misplacement

| script | what | example |
|---|---|---|
| `misplace_diag.py` | labels, anchor candidates, rough inliers, rough/final distance to the key | `uv run python scripts/tuning/misplace_diag.py 1948 1960` |
| `anchor_check.py` | how many anchor candidates the owner's key agrees with | `uv run python scripts/tuning/anchor_check.py 1948 10` |
| `scale_prior.py` | street-width scale guess vs the key's ft/px | `uv run python scripts/tuning/scale_prior.py 1948 1960` |
| `detector.py` | misplacement signals vs key distance (optionally on an older work tree) | `uv run python scripts/tuning/detector.py 1948 1960` |

## Residual diagnosis (images)

| script | output (in `data/out/tuning/`) | example |
|---|---|---|
| `resid.py` | `resid_<s>_<sheet>.png` residual vectors and magnitudes; `.npz` for block.py | `uv run --with matplotlib python scripts/tuning/resid.py 1948 1` |
| `block.py` | per-block translation/affine decomposition (text) | `uv run python scripts/tuning/block.py 1948 1` |
| `edges.py` | stroke centre / extracted edge vs modern lot line per edge (text, `.npy`) | `uv run python scripts/tuning/edges.py 1948 1` |
| `xstreet.py` | correlation of edge offsets across streets (text) | `uv run python scripts/tuning/xstreet.py 1948 1` |
| `overlay.py` | `ov_<s>_<sheet>_<x>_<y>.png` scan crop, modern (red), extracted (blue) | `uv run python scripts/tuning/overlay.py 1948 1 1900 3300 350 3` |
| `gallery.py` | `gallery_<s>_<sheet>_<min>.png` crops at residuals in [min, max) | `uv run python scripts/tuning/gallery.py 1948 1 9 16` |
| `terr.py` | `terr_<x>_<y>.png` white-space classes and block territories | `uv run python scripts/tuning/terr.py 1948 1 3230 2847` |
| `changes_plot.py` | `changes_<s>_<sheet>[_tag].png` real change, no-equivalent, residuals | `uv run --with matplotlib python scripts/tuning/changes_plot.py 1948 1 final` |
| `regions.py` | share within 6/8 ft by neighbourhood (text) | `uv run python scripts/tuning/regions.py` |
