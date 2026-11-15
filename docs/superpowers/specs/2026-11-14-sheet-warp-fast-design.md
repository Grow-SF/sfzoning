# Sheet warp, fast

2026-11-14. Branch `prob-warp`, PR #58. Builds on `2026-11-14-sheet-warp-build-stage-design.md`.

## Why

The owner wants the sheet warp "as fast as possible" over all 357 placed sheets on their Mac (Apple silicon, 10 cores,
64 GB). Measured alone on the idle machine, one big sheet (1948 sheet 11) took 442 s; the whole job, at the ~1 sheet
a minute the machine managed running 9 sheets at once, about six hours.

The owner's decisions (2026-11-14):
- The corrections need not stay byte-identical to the approved run. A faster method is judged by the same scores:
  block-side points within 1 and 2 px of the print (`targets.measure`), no folds (smallest cell determinant > 0), the
  control points' miss, and the seams by eye.
- Running it twice must give the same answer.
- It runs on the owner's Mac; Linux (CI) only needs a working CPU path.
- "Fully autonomously using your best judgment and subagent driven development."

## What the time went on (442 s, big sheet, idle machine)

| Step | Seconds |
|---|---|
| probabilistic correction (fit: backward passes, grid sampling, re-blurring 18 maps every stage ~56 s) | 216 |
| block-side points and their placement (`to_pixel`: scipy RBF evaluation inside Newton steps ~61 s) | 72 |
| first correction (fit ~36 s, carrying shifts outward 17 s) | 56 |
| fit targets (skeleton, 18 Euclidean distance maps ~28 s) | 45 |
| block-line weights (skeleton again, stroke-width EM, structure tensor) | 25 |
| warp (plain and paper, GDAL) | 22 |
| the rest | 6 |

## What the spike measured (throwaway code: scratchpad `spike/`)

On 11 sheets (1948 ×2, 1921 ×2, 1929, 1935, 1960, 1964, 1968, 2000 p03, 2000 p09), with the fits copied from
today's code (reproducing all 11 saved corrections bit for bit) as the baseline. Reference code: `/private/tmp/claude-501/-Users-sbuss-workspace-sfzoning/eb18d2fc-e22a-48a6-afbc-57c31cd3e134/scratchpad/spike/fits.py`
(the fits on the GPU, the deterministic field and sampler, the stop rule), `/private/tmp/claude-501/-Users-sbuss-workspace-sfzoning/eb18d2fc-e22a-48a6-afbc-57c31cd3e134/scratchpad/spike/fast_prep.py` and `common.py` (the
faster preparation), `exp5_cpu.py` (its equality checks). Throwaway, never copied into the repo as is.

| | Today (CPU) | GPU, same steps | GPU, probabilistic stages stop when settled |
|---|---|---|---|
| fits, 11 sheets, s | 2013 | 237 | about 85 (first fit 1500 steps kept) |
| within 1 px against today: mean (worst sheet) | — | −0.0001 (−0.001) | −0.0009 (−0.003) on the chosen correction |
| the same choice | — | 11 of 11 | |
| runs repeat bit for bit | yes (CPU) | yes, 11 of 11, every grid | (same formulation) |

- One step, 1948 sheet 11 (515k points): probabilistic 47 ms → 5.6 ms, first 23 ms → 3.8 ms; the stage blur 56 s →
  under 1 s (torch separable convolution on the GPU; OpenCV `sepFilter2D` on the CPU, 1.3 s; both within 7.5e-7 of
  scipy's maps' peak).
- Plain `grid_sample` on the GPU is 3–4× slower than the deterministic formulation, and its backward pass scatters
  2M contributions onto 11k nodes with atomic adds (order-dependent).
- Convergence: each 800-step probabilistic stage settles in 25–75 steps (worst 325). The first fit's loss never
  settles (Adam wobbles) but its within-1 score still creeps up late (up to +0.007), so it keeps all 1500 steps.
- Rejected by measurement: L-BFGS (first fit up to −0.022), coarse maps (no faster: a step costs per point;
  −0.095 on 1968 p21), fewer stages (up to −0.019), a stop rule on the first fit (up to −0.004).
- Where the GPU corrections differ from today's: probabilistic p99 0.01–0.34 px at the points (largest grid node
  1.36 px); first 0.6–6.6 px at worst at the points, nodes up to 8 px (unconverged Adam: any rounding change moves its
  wobble), scores unchanged.
- Preparation, 2000 p03 (today → fast, identical outputs): fit targets 11.4 → 5.1 s, block-line weights 6.5 → 0.9 s,
  held sides 0.9 → 0.15 s, the page's half width 1.25 → 0.55 s. Point placement (`to_pixel`, the largest preparation
  cost: 18–224 s per sheet today) through a 32 px cubic grid plus one exact step: 16.2 → 3.0 s, within 1.3e-4 px,
  no rounded pixel changed (one sheet measured).
- Estimated: 1948 sheet 11 442 s → about 90 s; the whole job about 6.3 h → about 45–60 min (the GPU the floor).

## What gets built

1. **The fits on the GPU.** Both fits run on PyTorch's `mps` device when it is available, on the CPU otherwise (the
   same code; the device is chosen once per process). The loss terms, stages, step counts and learning rates stay
   as they are unless item 5 says otherwise.
2. **A deterministic field and sampler.** The correction field at a fixed set of points is a fixed bilinear
   interpolation of the grid: `u = W G` (4 weights per point), its gradient `dL/dG = Wᵀ dL/du` summed per grid node
   over a fixed, padded list of that node's contributions (nodes bucketed by how many they have), never with atomic
   adds. The density maps are sampled by one gather of each point's 4 neighbours from the 18 flattened maps, with an
   analytic gradient in the point's position (the maps take no gradient). Both match `grid_sample` to float rounding
   (field exact, gradient within 2e-5; sampler within 4e-6) and repeat bit for bit on the GPU.
3. **The stage blur on the device**: a separable Gaussian convolution in torch (or OpenCV on the CPU path) instead
   of `scipy.ndimage.gaussian_filter`.
4. **Faster preparation on the CPU**, each verified to give the same values as today's or a stated, measured
   tolerance:
   - distance maps with OpenCV's exact Euclidean distance transform (`cv2.distanceTransform`, `DIST_MASK_PRECISE`),
     the exact value recovered from its squared integer distance;
   - the crop's skeleton, its ink half width and the ink's direction computed once, shared by the fit targets and
     the block-line weights;
   - point placement for the block-side points (and the other large `to_pixel` batches) through a dense grid:
     `to_world` sampled every 32 px over the page, a cubic spline of it inverted by today's iteration, then one
     exact step of today's `to_pixel` from that start (within 1.3e-4 px on the sheet measured). The exact `to_pixel`
     stays for everything else. Verify on the big sheets (1964 p17, 1948 sheet 11) before relying on it;
   - the stroke-width EM of the block-line weights on the width histogram (`np.unique` counts) instead of per pixel
     (identical float16 output);
   - `propagated_shifts`: each side's points by one sort of the side ids instead of a full-array mask per side
     (identical output; 13–17 s on big sheets today, not yet measured fast).
5. **Fewer optimizer steps**: the probabilistic fit stops each stage when its loss fell by less than 1e-4 (relative) over the
   last 25 steps, after at least 50 steps; 800 stays the cap. The first fit keeps its 1500 steps. Nothing else about
   the optimization changes.
6. **Running sheets side by side**: with the fits on the GPU, a sheet is mostly CPU preparation; several sheets run
   at once (worker processes, as now), all sharing the GPU. The default number at once is set by measurement on
   this Mac.
7. **The device in the fingerprint.** A sheet's freshness records which device fitted it (results differ slightly
   between the GPU and the CPU), and `scores.json` says which.

## How we know it works

- **Scores:** on every sheet with a result from today's method (149 sheets kept in
  `pipeline/data/warp-baseline-2026-11-14/<series>/<sheet>/scores.json` and `correction.npz`: 1921–1968, part of
  1972–1980, and 2000), the new method's chosen correction scores within 0.005 of today's within-1 px share or
  better, never folds, and the mean over those sheets is within 0.001 of today's. Sheets that get worse by more are
  listed with pictures.
- **The same every run:** fitting the same sheet twice on the same device gives bit-identical corrections.
- **Speed:** per-sheet and whole-job times on this Mac, against today's.
- **Tests:** the sparse field against `grid_sample` (same values to float tolerance) and its gradient; the device
  choice; the prep helpers against today's functions on synthetic inputs.
- **Then:** the full build of all 357 sheets with the fast method, the comparisons above, the seams by eye, a fresh
  clone, push, PR description.

## Constraints

As in the build-stage design (no `pipeline/data` commits, no paid calls, old stages frozen, pixi's GDAL, every
placed sheet, street edges and later blocks on 2000 p09/p10 by name). No new dependencies unless the spike shows one
pays (PyTorch, OpenCV, scipy are already in).

## Left to the builder

Module layout for the device and sparse-field helpers; how W is built; logging.
