# Saved reads

Every answer an OpenAI vision model (Terra or Sol) gave this project that a later step is built from, with exactly how
each was asked: 43,082 answers from 2026-10-30 to 2026-11-13. They were paid for once; keeping them here means nobody
pays to ask again, and the steps built on them (the text detector, the joint solver) start from the same answers every
time.

## What happened, step by step

**1. The questions we paid for.** We showed the models pictures cut from the map scans and asked about them. Each kind
of question has its own answer file in `answers/`:

| Answer file | What the model was shown | What it was asked | Model | Answers | Used by |
|---|---|---|---|---|---|
| `text-on-every-sheet` | each whole sheet of 1960–2000 in 768-px tiles (179 sheets, 9 years) | find every piece of text: what it says, what kind (zoning code, block number, street name, other), a box around it, which way it runs | Terra | 14,829 | text detector (what to learn); joint solver (via its label readings) |
| `zoning-codes-2000` | a square around each of the 1,648 zoning codes on the 2000 map, the code circled | question 1: what code is circled, and does a leader line carry it somewhere else (where)? question 2: read the code at the centre (a re-check) | Terra and Sol | 6,592 | the code-check page you answered was made from these; use the red-circle version from now on |
| `zoning-codes-2000-red-circle` | the same, with the circle shown red as the question says ([why](#the-red-circle-fix)) | the same | Terra and Sol | 6,592 | the 2000 code reads from now on (scored against your answers below) |
| `leader-lines-2000` | the same squares | is there a leader line, and where does it point? | Sol | 3,296 | the leader-check page you answered was made from these; use the red-circle version from now on |
| `leader-lines-2000-red-circle` | the same, circle red | the same | Sol | 3,296 | the 2000 leader reads from now on |
| `district-labels` | the pipeline's own tiles of every district map (245 sheets, 14 years) | every district label: code, kind, where, how sure | Sol | 3,866 | joint solver (its second reader); text detector ("unknown" areas) |
| `experiments-2000/…` | eight experiments on the 2000 map, Nov 10–11 (step 5) | | | 4,611 | joint solver; text detector |

**2. Recording how each answer was asked** (`scripts/saved_reads/replay_all.sh`, $0). Every script that asked a question
is run again with the internet blocked, so every answer comes from the saved copies. For each answer this records the
exact question, picture and model that produced it. The scripts it runs are kept exactly as they ran, in
`scripts/saved_reads/scripts_as_they_ran/` and `red_circle/`.

**3. Packaging** (`python -m saved_reads package`). The recordings become the files in this folder: one line per
answer, holding the answer, the question, the picture's name, the model, and where on the map the picture came from.
The 29,381 pictures (2.1 GB) are too big for git; they ship with the pipeline's data (the build stage `saved-reads`),
`uv run sfzmaps pull` brings them to `pipeline/data/saved-reads/images/`, and `images.manifest.tsv` lists each one.

**4. Checking** (`python -m saved_reads verify`). For every answer that was recorded in step 2, its cache fingerprint is
rebuilt from its own question, picture and model, and must match the one it is filed under: 39,269 rebuilt exactly, 0
mismatches. The other 3,813 are from experiments matched to their runs by time (step 5) and have no picture to check.

**5. The experiments, kept as recorded** (`answers/experiments-2000/`). These questions were built from things that no
longer exist (an old map alignment, old solver runs), so they can't be asked again; their answers are kept exactly as
recorded, and nothing replays them:

| Answer file | What the model was shown | What it was asked | Model | Answers |
|---|---|---|---|---|
| `sideways-text` | the 2000 tiles turned a quarter clockwise, to catch sideways text | as `text-on-every-sheet` | Terra | 702 |
| `close-up-text` | close-ups of places an old solver run was unsure about, upright and turned | as `text-on-every-sheet` | Terra | 106 |
| `codes-on-tinted-lots` | close-ups of an old solver run's places, the lots tinted yellow | which codes are printed on or beside the tinted lots? | Terra | 472 |
| `codes-on-tinted-lots-smaller` | the same, with the places cut into smaller pieces | the same | Terra | 508 |
| `code-of-tinted-area` | areas an old solver run found no print in, tinted | which code does the map give the tinted area? | Terra | 390 |
| `third-code-read` | a close crop around each printed code | what code is printed, and does a leader line point it elsewhere? | Sol | 1,865 |
| `third-code-read-recheck` | the unsure third reads, again | the same | Sol | 418 |
| `sideways-trial` | a 30-picture trial of turned reads | as `text-on-every-sheet` | Terra | 150 (used by nothing) |

Most of these answers were matched to the run that made them by when they were stored (`matched: "by time"`), so
their records don't say where on the sheet each piece of text is. `answers/experiments-2000/text-positions/` keeps that,
one file per experiment that found text, as the experiments wrote it: every box in page pixels, with its sheet. The
text detector reads these. The experiments' code is in `scripts/saved_reads/scripts_as_they_ran/experiments_2000/`.

**6. Your hand answers** (`pipeline/inputs/code-check-2000/`): your 743 answers from the code-check page and 50 from
the leader-check page, with the lists of cases you were shown.

## What's in this folder

```
saved-reads/
  README.md                       this
  answers/                        one file per kind of question; one line per answer (gzipped JSON lines)
    text-on-every-sheet.jsonl.gz
    zoning-codes-2000.jsonl.gz                 zoning-codes-2000-red-circle.jsonl.gz
    leader-lines-2000.jsonl.gz                 leader-lines-2000-red-circle.jsonl.gz
    district-labels.jsonl.gz
    experiments-2000/             step 5: eight experiments, kept as recorded
      sideways-text, close-up-text, codes-on-tinted-lots, codes-on-tinted-lots-smaller, code-of-tinted-area,
      third-code-read, third-code-read-recheck, sideways-trial (.jsonl.gz)
      text-positions/             where the experiments' text sits on each sheet (page pixels)
  how-each-was-asked/
    questions.jsonl               every exact question asked (339), by its sha256
    question-templates.jsonl      each question with what varies blanked out (numbers -> #, the circled code -> {code})
    answer-formats.jsonl          the answer format (JSON schema) each model was held to
    recorded-runs.jsonl           every recorded run: script, where it ran, arguments, settings
  images.manifest.tsv             every picture: readable name, sha256, size
```

Pictures are named for what they show, under the answer file's name: for example
`text-on-every-sheet/2000-digital-zoning/2000-digital-zoning-p01/tile_1519_0000_768x768.png` is the tile of that sheet
whose top-left corner is at page pixel (1519, 0), and `zoning-codes-2000-red-circle/leader-map/2000-digital-zoning-p07/at_1400_2998.png`
is the square around the code printed at page pixel (1400, 2998).

## One answer line

```
read, series, sheet, image          which answer file, which map and sheet, the picture's name (above)
page_box | at | case | place        where the picture sits on the page (page pixels), the point it is about,
                                    the code-check case, or an old solver run's place number
variant, scale                      turned / upright; the label reader's page scale
model, provider, effort, params     the model and its settings
prompt, prompt_sha256               the question's template, and the exact question (how-each-was-asked/questions.jsonl)
answer_format                       sha256 of the answer format (how-each-was-asked/answer-formats.jsonl)
asked                               when the answer was first stored (UTC)
script                              where the asking script ran (branch@commit:path), its arguments and settings
matched                             "replayed" (step 2) or "by time" (step 5)
answer                              the answer, exactly as the model gave it (coordinates in the picture's pixels)
image_sha256, cache_key             the exact picture bytes, and the answer's file in pipeline/data/cache/llm
```

## The red circle fix

The code and leader questions draw a red circle on each picture and ask about "the code circled in red". The scripts
handed the picture to the model with its colour channels in the wrong order, so the model saw the circle blue.
`scripts/saved_reads/red_circle/` holds copies of the three scripts with only that fixed (`check_marks.py` checks the
circle reaches the model red); model, effort, question and picture are otherwise the same. Every one of those pictures
was asked again with your go-ahead (9,888 answers, $95.53 by the spend meter), into the `-red-circle` answer files.

What changed (`python -m saved_reads.red_circle.compare`: each answer against the original from the same model and
effort about the same picture; every differing field counts, so one answer can change twice):

| Question | Model, effort | Answers | Changes | in answers | Details |
|---|---|---|---|---|---|
| what code is circled, and its leader | Sol, high | 1,648 | 201 | 192 | code read differs on 35; leader seen flips on 136 (272 → 288); points elsewhere on 30 |
| what code is circled, and its leader | Terra, high | 1,648 | 61 | 58 | code read differs on 27; leader seen flips on 30 (113 → 121); points elsewhere on 4 |
| leader line | Sol, medium | 1,648 | 152 | 152 | leader seen flips on 129 (215 → 236); points elsewhere on 23 |
| leader line | Sol, low | 1,648 | 154 | 154 | leader seen flips on 144 (202 → 216); points elsewhere on 10 |
| re-check: read the code | Sol, high | 1,648 | 119 | 119 | code read differs on 119 |
| re-check: read the code | Terra, high | 1,648 | 93 | 93 | code read differs on 93 |
| **all** | | **9,888** | **780** | **768** | Sol 626 in 617 of 6,592; Terra 154 in 151 of 3,296 |

"Points elsewhere": both answers see a leader, but they point more than 15 page pixels apart. Only the re-check can be
scored: on the 737 code-check cases you answered, Terra's code matches yours 84.8% of the time with the blue circle and
88.3% with the red; Sol's 86.0% and 87.4%. Use the `-red-circle` files; the originals stay for comparison.

## Check it yourself ($0)

**Quick: the package against itself.** Needs only this folder and the pictures.
```
cd pipeline/scripts && ../.venv/bin/python -m saved_reads verify
# {'key rebuilt exactly': 39269, 'by time (no image to check)': 3813}   mismatches: 0
```

**Full: everything again from the scans, in a fresh clone.** Nothing outside the clone and what `pull` brings is read.
1. In `pipeline/`: `uv sync && uv run sfzmaps pull` (the scans, the pipeline's outputs, these pictures).
2. Replay every question that can be asked again, with the internet blocked. Every line must say `0 not in the cache`.
   It first writes this package's answers into the answer cache, since `pull` brings only the answers the build's own
   outputs were made from.
   ```
   zsh scripts/saved_reads/replay_all.sh data/saved-reads/check
   ```
3. Package the replay over a copy of this folder (it carries the experiments and the first-stored times from it) and
   compare. Every file must be the same but `how-each-was-asked/recorded-runs.jsonl`, which lists the replay's own runs.
   ```
   cp -R saved-reads data/saved-reads/check-package
   cd scripts && ../.venv/bin/python -m saved_reads package --calls ../data/saved-reads/check \
     --out ../data/saved-reads/check-package && cd ..
   diff -r -x recorded-runs.jsonl -x README.md saved-reads data/saved-reads/check-package && echo "same"
   ```

## Ask again with a newer model (paid)

1. **Count first.** Each answer file's count above is its number of calls. AGENTS.md: a new model or question means
   paying again.
2. **Run** the question's script live, with another model and a spending cap. `replay_all.sh` shows each script's
   arguments and settings.
   ```
   cd pipeline/scripts
   SR_MODEL=<new model> SR_EFFORT=<effort> SFZMAPS_MAX_USD=<cap> \
     uv run --env-file <the API key file> python -m saved_reads run --live --out ../data/saved-reads/<new> \
     <script> <args>
   ```
   `--live` refuses to run without `SFZMAPS_MAX_USD`; `SR_MODEL` and `SR_EFFORT` override the model even where a
   script names one; new answers get new cache keys and the old ones stay.
3. **Package** the new folder (`python -m saved_reads package --calls ../data/saved-reads/<new> --out <folder>`) and
   compare it with this one; pictures with the same name show the same piece of the page.

## Limits

- **The experiments can't be asked again,** only checked against themselves (`verify`).
- **Answers matched by time have no picture or question,** only their answer and the run that made them.
- **`asked` is the cache file's time when the answer was first packaged.** For an answer applied from a shipped pack
  before that, it is when the pack was applied.
