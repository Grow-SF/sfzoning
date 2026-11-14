# The scripts that asked the questions, as they ran

Byte-for-byte copies of the scripts that asked the models the questions in `pipeline/saved-reads`, so the replay
(`../replay_all.sh`) needs nothing outside this checkout. Each answer's `script.git_blob` is the hash of its script
here (`git hash-object FILE`). Don't edit them: they are the record of what ran.

| Script | Ran from | Asked |
|---|---|---|
| `textboxes.py` | `prob-warp@99f85e7:pipeline/scripts/prob-warp/allyears/textboxes.py` | `text-on-every-sheet` |
| `leaders2.py`, `leaders2_v2.txt` (its question) | `prob-warp@99f85e7:pipeline/scripts/prob-warp/codes2000/` | `zoning-codes-2000`, question 1: what code is circled, and its leader |
| `codecheck_build.py` | the same folder | made the code-check page's cases (`pipeline/inputs/code-check-2000`); asks nothing |
| `reread.py` | the same folder | `zoning-codes-2000`, question 2: the re-check |
| `leaders.py` | `zoning-lines-rnd@8ec7b40:pipeline/scripts/zoning-lines/leaders.py` | `leader-lines-2000` |

These hand their pictures to the model with the red circle shown blue; `../red_circle/` holds the fixed copies that
asked the `-red-circle` questions. Some of the scripts hard-code the R&D session's scratch folder;
`python -m saved_reads run --scratch DIR` reads a work folder in its place, and refuses to run them without it.

`experiments_2000/`: the code of the eight experiments of Nov 10–11 (`answers/experiments-2000/`):
- the scripts that asked: `textpass.py` (`sideways-text`), `placeread.py` (`close-up-text`), `placeask.py`
  (`codes-on-tinted-lots`, and with `SPLIT=1` `codes-on-tinted-lots-smaller`), `distask.py` (`code-of-tinted-area`);
  each matches its answers' `git_blob`. `read3.py` (`third-code-read`), `read3r.py` (`third-code-read-recheck`),
  `rotpilot.py` (`sideways-trial`) are their last versions, edited after their paid runs;
- the modules they import: `lotwalls.py`, `truth.py`, `yard.py`, `lotcode.py`, `lotcode_lib.py`;
- `by_time.py` with `timed_reads.json`, which matched the answers recorded by time to the runs that made them, once.

They read inputs (an old map alignment, old solver runs) that no checkout has, so nothing runs them.
