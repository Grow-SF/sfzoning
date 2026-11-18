# Task 3 fix round 1: re-review (opus)

| Finding | Verdict |
|---|---|
| Critical 1 | ADDRESSED |
| Critical 2 | PARTLY |
| Important 3, Important 4 | ADDRESSED |
| Minors 5, 6, 7, 8 | ADDRESSED |
| Minor 9 | ADDRESSED, except the off-centre range |

**Critical 2.** The off-centre test asserts 6 offsets and leaves out −0.75 and −1.0. In the test's setup:

| Offset | End table on | End table off |
|---|---|---|
| −0.5 | RH-1(D) 0.672 (other 0.301) | other 0.941 |
| −0.75 | **other 0.666** | other 0.977 |
| −1.0 | RH-1 0.901 | RH-1 0.762 |

- **−1.0** is the principled outcome: the measured placement prior beats the measured end evidence.
- **−0.75 is not:** "other", summing partial strings, wins, and that half of the finding is still open.
- **The real map did not improve:** training point labels with a code string go from 23/50 top-1 to 21/50, and mean
  −log p from 2.27 to 2.55.

## New, Important: "other" now beats clean printed codes on box labels (from the Minor 6 change, posterior.py:929-962)

On the 885 training box labels whose training string is a code, committed against now, at the default constants:

| | Committed | Now |
|---|---|---|
| p(other) > 0.5 | 4 | 8 |
| p(other) > 0.1 | 6 | 17 |
| Top-1 | 815 | 811 |

"Other"'s log E rose by a median of 11.5 nats (90th percentile 23.5).

Confident flips from the right code:
- #1414 M-2, 0.975;
- #1440 M-2, 1.000;
- #280 RH-1(D), 0.924;
- #1708 RH-1(D), 0.992.

M-2 is in the code list, so these are regressions. #1414 is a crisp, isolated M-2 on clean paper. Its "other" log E
(20.3) is above M-2's (18.6) although the bigram charges "M-2" −13.9 nats. So "other"'s best string fits the ink at
least 15 nats better than the code's fixed layout. A likely cause is a spaced spelling such as "M -2" (−15.5); there
is no backtrace to confirm it. Pictures: scratchpad/review3fix/other_flips.png.

## Minor

1. **The off-centre test hides −0.75.** Mark −0.75 and −1.0 xfail with the measured reason, or fix them.
2. **`evidence()` takes an exclusive GPU turn on every call.** A caller that wraps it in its own turn, as gpu.py
   advises, would deadlock.
3. **`settle_torch()` has process-wide side effects** when called from a per-label call.
4. **Old pickled labels read `held_block` as False.** `refuse_held` then catches only held-out labels unless
   `mark_held_blocks` is called first.

## Records

Both files are complete and consistent: 731 held-out labels at 4 temperatures, 1,174 others at 32.744, no overlap,
all finite.

## Verdict

- **All findings addressed:** no.
- **Task 4 may proceed:** NO. Fixing the "other" over-strength changes every label's evidence and needs a re-score.
  Deciding it after Task 4 has seen the answers would tune on the test.

Scripts: /private/tmp/claude-501/-Users-sbuss-workspace-sfzoning/eb18d2fc-e22a-48a6-afbc-57c31cd3e134/scratchpad/review3fix/
