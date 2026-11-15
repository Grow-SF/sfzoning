# One text detector for every year, trained in minutes

Design, 2026-11-15. It changes the text detector built by the solver-inputs plan
(`docs/superpowers/plans/2026-11-15-solver-inputs.md`, tasks 6–7). The owner approved the outline in conversation;
calls the owner did not make are marked **(my call)**.

## Why

The detector finds printed text, so the walls can discount it. Today:

- It is four separate models, one per quarter of the city, each trained without that quarter's labels.
- Each model takes about 50 min on the Mac's GPU, and only one GPU job can run at a time, so a retrain takes about
  3.3 hours.
- It learns only from the 2000 map.

The owner wants:

- **Speed, to iterate:** "faster at same quality is good too, to help with iterating on approach"; target "a few
  minutes per quarter".
- **One detector across lettering styles:** "one detector on multiple styles", so later years' maps can use it.

Training on a lower-resolution copy of each page is acceptable if the full-resolution test scores come out the same.

Held-out quarters exist so that no fit learns from the zoning table's answers in the area it is scored on. The
detector never sees the table. It learns only where text is printed, from Terra's text boxes. So one detector trained
on all of a map leaks nothing into the solver's scores. Its own accuracy is measured on whole drawings it never
trained on.

## Goals

- **One detector**, trained on maps of both lettering styles (below), and tested on whole drawings it never saw.
- **A few minutes per training run** on the Mac's GPU, with test scores no worse than today's recipe trained on the
  same data.
- **The same answer every run:** two runs predict within 0.01.
- **One `solver-text` build stage** instead of four. Scoring becomes a separate, cheap stage. The walls read one text
  map.

## Non-goals

- Reading the text, or a detector for the 1920s–1950s maps. Those have no Terra boxes; their lettering differs again.
- Changing what the walls do with the text chance. The walls' candidate loop is rerun on the new chance afterwards,
  as already planned.

## Anti-goals

- Choosing a training setting by its score on the test drawings, then reporting that score as unbiased. Settings are
  screened on a validation sample from the training maps (below). The test drawings report the final result.
- Two GPU jobs at once (silent wrong numbers; `sfzmaps.gpu`).
- New paid reads, or any input outside a fresh clone.

## Data

| Role | Maps (series) | Lettering | Sheets with Terra boxes |
|---|---|---|---|
| train | `2000-digital-zoning` | modern italic | 13 |
| train | `1980-1990-zoning`, `1990-1996-zoning` (scans of the 2000 drawing at earlier dates) | modern italic | 13, 12 |
| train | `1960`, `1964`, `1968` (1964 and 1968 are one drawing) | older serif | 13, 16, 20 |
| test | `1980` (a separate drawing) | modern italic | 32 |
| test | `1972`, `1975` (one drawing) | older serif | 30, 30 |

- **Pages:** `data/work/<series>/<sheet>/page.png`, the original unwarped scans. The pipeline's frozen stages made
  them, and `sfzmaps pull` brings them.
- **Labels:**
  - **Targets:** Terra's boxes for each sheet, decoded from `saved-reads/answers/text-on-every-sheet.jsonl.gz` the
    way the 2000 boxes are decoded today. The decoding is generalised to any series.
  - **"Unknown"** (no loss): 30-px discs at Sol's label points (`district_labels.json`), plus, for 2000 only, the
    other text passes' boxes, as today.
  - Only sheets Terra read are used. The 1990–1996 map has 33 pages but 12 with boxes.
- **Validation:** a fixed area of the city, about 10% of the mapped area, held out on every training map at once.
  It is a named rectangle in map feet, defined in code. **(my call)**
  - A page pixel is in it when its map point (through the sheet's warp transform; every series has one) falls
    inside, the same way pixels are held out by quarter today.
  - During screening runs those pixels carry no loss, and crops don't centre on them; the scores are computed on
    them.
  - Holding out an area, not sheets, keeps the reprinted scans of one drawing (2000 / 1980–90 / 1990–96, 1964 /
    1968) from leaking into validation.
- **The detector the build ships is retrained on everything, the validation area included**, with the chosen
  settings. It is scored on the test drawings. **(my call)**
- **Memory:** 87 training sheets at about 17 Mpx each. The targets must not be full-page rasters per sheet (today
  9 B/px would be 13 GB). Draw them per crop from the boxes, or keep them compact. **(my call: the builder picks)**

## Scores (the same measure for every comparison)

On a sheet's ink pixels:

- AUC and log-loss of text vs not;
- text ink found;
- words found, by kind;
- ink outside every box called text;
- **long straight lines called text:** ink on strokes at least 200 ft long and at most 12 ft wide, named constants.
  This stands in for "district-line ink called text", which needs the zoning table. **(my call)**

Reported per test drawing and for the validation area.

- **Today's 2000-only quarter models are reported for context, not used as a gate** (nw: 96.6% of words found;
  0.16% of district-line ink called text). Their measure differs, and the walls read only the 2000 sheets.
  **(my call)**
- **One check on mixing styles.** Once the recipe is fast, the same recipe is also trained on the modern-lettering
  maps only, and the two are compared on the 1980 test drawing.
  - If the both-styles detector is worse there beyond the tolerance, it goes to the owner before anything ships.
  - Otherwise the both-styles detector ships. **(my call)**

## Making it fast

Start with a baseline: today's recipe on the new training data, scored on validation and test.

Then the candidates, each measured on its own against the current best.

- **Keep rule:** a candidate is kept only if its validation scores are no worse within the tolerance, on words found
  and on long-line ink called text. Then they are combined.
- **The tolerance** is twice the spread between two seeds of the fast recipe, measured once early, with a floor of
  0.5 points on words found. **(my call)**
- **The clock:** "a few minutes" means the training loop at most about 5 min, and the whole `solver-text` stage at
  most about 10 min (reading pages and targets, training, predicting 2000).
  - If the kept changes stall above that, the loop stops and reports. It does not loosen the quality bar: the
    owner asked for "faster at same quality". **(my call)**

The order is the audit's evidence of size:

1. **Loss on ink pixels only** (82% of the loss is now paper nobody reads). A variant: ink dilated by 1 px.
2. **Half resolution, losslessly:** fold each 2×2 block into 4 channels, run the same network at half size, and
   unfold the output. About 4× fewer operations.
3. **A shorter schedule:**
   - lengths 2,000, 4,000 and 8,000 steps;
   - peak learning rate 2e-3 or 6e-3, warm-up 10%;
   - an average of the weights over the last steps.
4. **Drop the kind and reading-angle outputs.** Nothing downstream reads them.
5. **No GPU sync every step,** and the next batch prepared in a background thread from the one seeded generator
   (5–15%).
6. **A lighter decoder** (one conv per up block, or width 16), only if needed to reach the target.

Not tried: half precision, `torch.compile` and MLX, which the review found uncertain or slower on this GPU;
pretrained encoders (licences, little gain); the published map-text models (non-commercial licences).

**Report:** the minutes per training run and the scores, before and after each kept change.

## The build

- **`solver-text`** (one stage) trains the detector on the training maps. It writes the model and the text chance of
  the 2000 sheets, the only ones the walls read now.
  - Reads: the named page, label and box files of the training sheets. The fixed sheet lists live in code, so
    `dvc.yaml` doesn't depend on data being present.
- **`solver-text-scores`** predicts the validation sheets and the test drawings, and writes the scores.
- **The four `solver-text-{q}` stages and the per-quarter assembly in `textgrid` go.** `textgrid` samples one model's
  chance.
- **Determinism:** seeded; deterministic algorithms; the masked-sum losses (no boolean indexing on the GPU); one GPU
  turn per run.

## Afterwards (not this design)

Rerun the walls' text candidates on the new chance. Choose the walls' default features. Then the build, `dvc.lock`,
the fresh-clone check and the PR, as planned.
