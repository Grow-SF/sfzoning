# The reading scripts, as they ran

Byte-for-byte copies of the scripts that asked for the saved reads, so a replay (`../record_all.sh`) needs nothing
outside this branch. Each record's `script.git_blob` is the hash of its file here (`git hash-object FILE`).

| File | Ran from | Made |
|---|---|---|
| `textboxes.py` | `prob-warp@99f85e7:pipeline/scripts/prob-warp/allyears/textboxes.py` | `terra-page-boxes` |
| `leaders2.py`, `leaders2_v2.txt` | `prob-warp@99f85e7:pipeline/scripts/prob-warp/codes2000/` | `code-reads` (t1, the leader map) |
| `codecheck_build.py` | same folder | the code-check page's cases (`pipeline/inputs/code-check-2000`) |
| `reread.py` | same folder | `code-reads` (t2, the re-check) |
| `leaders.py` | `zoning-lines-rnd@8ec7b40:pipeline/scripts/zoning-lines/leaders.py` | `leader-reads` |

They hand their crops to the model with the red mark shown blue. `../rerun/` holds the fixed copies that made the
`-fixed-marks` reads. Don't edit these: they are the record of what ran.
