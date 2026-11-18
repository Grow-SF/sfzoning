## Global Constraints

- **The maps:**
  - **train:** `2000-digital-zoning`, `1980-1990-zoning`, `1990-1996-zoning`, `1960`, `1964`, `1968`;
  - **test:** `1980`, `1972`, `1975`.
  - Only sheets Terra read are used. The sheet lists are fixed constants in code; `dvc.yaml` must not depend on data
    being present.
- **Validation area:** one named rectangle in map feet, about 10% of the mapped area. A page pixel is held out when
  its map point (through `data/warp/<series>/<sheet>/transform.json`) falls inside. Held-out pixels carry no loss,
  and crops don't centre on them.
- **Same answer every run:** two runs predict within 0.01.
  - Seeded; `torch.use_deterministic_algorithms(True)`; masked-sum losses (never boolean-index a tensor that needs a
    gradient on mps).
  - The run takes `sfzmaps.gpu.gpu_turn(True, device)` around training and prediction. Never run two mps jobs.
- **Scores**, the same measure for every comparison, on ink pixels:
  - AUC and log-loss;
  - text ink found;
  - words found, by kind;
  - ink outside every box called text;
  - long straight lines called text: strokes ≥ `LONG_LINE_FT = 200` long and ≤ `THIN_LINE_FT = 12` wide.
- **Keep rule:** validation words found and long-line ink called text are no worse within the tolerance. The
  tolerance is 2 × the two-seed spread of the fast recipe, with a floor of 0.5 points on words found.
- **Clock:** the training loop ≲ 5 min, the whole `solver-text` stage ≲ 10 min. If kept changes stall above that,
  stop and report; never loosen the quality bar.
- **Fresh clone only:** no paid or model reads; nothing from `pipeline/data/rnd`, scratch or another worktree; no
  per-item freshness stamps.
- **Process:**
  - Tests run on CPU (`cd pipeline && uv run pytest -q`; the conftest guard keeps tests off a busy GPU).
  - Never run `sfzmaps warp/build/solver text*` outside the task that says to.
  - Commits end with a `Co-Authored-By:` line naming the model you are, then
    `Claude-Session: https://claude.ai/code/session_01UjTgrnx2j4PhzZErrad25Q`.
  - Stage only your own files.

## Review Focus

1. **The validation area leaking through crop context.** A crop centred outside the area overlaps it, and those
   pixels must carry no loss. Test in Task 1.
2. **Maps at different scales:** the 1960s pages are about twice the pixels. The long-line constants are in feet and
   must convert per sheet through its transform. Test in Task 3.
3. **Memory:** 87 training sheets must not become full-page per-pixel target rasters (13 GB). Test in Task 1
   (target bytes per sheet bounded); measured in Task 5.
4. **Prefetching changes the batches.** With the background thread on, the weights must be bit-identical to off.
   Test in Task 2.
5. **Half-size folding at odd sizes.** The fold needs even sizes, so crops and prediction tiles must be even and the
   page's odd edge handled. Test in Task 2.

---
