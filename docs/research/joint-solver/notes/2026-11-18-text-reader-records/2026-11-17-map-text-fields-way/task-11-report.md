# Task 11 report: box choice (item 5, PRELIMINARY) and the pack (item 4, built, not run)

Script: pipeline/scripts/wordread/run_choose_pack.py; tests: pipeline/tests/test_wordread_run_choose_pack.py (10 pass).
Outputs (not committed): pipeline/data/solver/experiments/wordread/choose-pack/prelim-ne-s{0,1}/ (run.json, params.json,
labels_{without,plain,fitted}.parquet, chosen_boxes.parquet, version_features.parquet).

## What was run (today's boxes, today's readers: PRELIMINARY, no final keep decision)
- Boxes: data/solver/boxes/boxes.parquet (8794). Each box gets its versions (choose.candidates, all other boxes of the sheet
  as neighbours: no window), every version's words are read (CPU, 4 workers x 1 thread) by the reader that never saw the
  box's quarter: NE by runs/base-ne-s{seed}/reader.pt, the other quarters by the stage's <q>.pt; all with the stage's 0/180
  classifier.
- Arms: without (as found; reproduces the base runs' labels exactly: 92 and 88 of 285), plain (sum of log posteriors,
  no fitted weight), fitted (choose.fit_choice per fold, on the other three quarters' boxes outside the answered blocks
  (a box meeting a Terra word that touches an answered block, or the held-out quarter, is out), labelled by label_of from
  Terra's words; the owner's answers are never used).
- Candidate widths: load_char_widths(held_out=<the box's quarter>).

## Fitted parameters, fold NE (boxes used: 4195; the fit is the same at both seeds, the training quarters' reads being the stage readers')
beta 1.473; alpha: split_gap -3.464, join_nearest -0.036, trim_start -3.098, trim_end -3.229; gamma (along): split_gap -11.42,
join_nearest -0.457; gamma_across join_nearest -0.168; nll 3161.7.  Other folds (boxes): nw 4469, sw 3980, se 3676 (params.json).
Which version Terra's words match (label_of, NE boxes): as_found 1119, join_nearest 697, trim_start 313, trim_end 202, split_gap 9.

## NE headline (right of 285), box choice against without
| seed | without | plain | fitted | fitted - without (paired, blocks resampled) | plain - without |
|---|---|---|---|---|---|
| 0 | 92 (0.3228) | 51 | 77 | -0.0526 [-0.0830, -0.0213] | -0.1439 [-0.1869, -0.0990] |
| 1 | 88 | 69 | 84 | -0.0140 [-0.0385, +0.0108] | -0.0667 [-0.1025, -0.0282] |
| mean of seeds | | | | -0.0333 [-0.0567, -0.0104] | -0.1053 [-0.1381, -0.0709] |
The headline falls; neither arm rises. By the written rule (5) would not be kept on these boxes and readers (the review
page's look is the controller's).

## Which version wins (NE boxes, seed 0 / seed 1)
- plain: as_found 1243/1307, join_nearest 690/595, trim_end 272/275, trim_start 239/266, split_gap 8/9.
- fitted: as_found 2124/2122, join_nearest 328/330 (no split or trim ever chosen).
- Fitted seed 0, labels changed: 18 right to wrong, 3 wrong to right; the box that holds 17 of the 18 lost labels is a
  chosen join_nearest (the other 1 as found).

## Box numbers, NE, IoU 0.5 (Terra 2705 words)
| | ours | matched | missed | missed: merged / split / other / none |
|---|---|---|---|---|
| without | 2452 | 850 | 1855 | 560 / 80 / 844 / 371 |
| plain s0 | 2460 | 805 | 1900 | 711 / 141 / 701 / 347 |
| fitted s0 | 2452 | 812 | 1893 | 616 / 118 / 797 / 362 |
Read numbers vs answers (boxes matched to a Terra word): without ours 63 of 77; fitted 54 of 61; plain 32 of 46.
Review page (Mission sheet p07, all quarters' boxes by their own fold's fitted choice, NE's reader at seed 0; before =
baseline): merged_with_word 293 -> 297, matched 447 -> 431, missed 659 -> 675, headline on the page 0.377 -> 0.287.

## Page
pipeline/data/solver/experiments/wordread/review/choose-prelim/review.html (2.2 MB; numbers.json, missed.json beside it).

## Item 4 (the pack)
pack_read_fn(lp_fn, prior_fn, weights, tau, base_strings) gives choose a read_fn that is context.posterior over
ink_logp (the two reads' log-probabilities, q, tau) and the prior at each version's own rectangle; make_prior_fn(pack,
spreads, transforms) builds prior_fn through context.box_on_map and context.prior; read_lp reads a crop's log-probabilities.
Tested on synthetic data (ink alone without a prior; a listed string lifted; normalised; choose runs on it; the version's
own rectangle is placed). `pack` subcommand refuses to run: it waits for Task 10's final fit. Not run on real data.

## Concerns
1. The fit's labels come from IoU with Terra's rectangles, which pick join_nearest/trim for 1212 of 2340 labelled NE boxes:
   Terra's rectangle shapes (tighter, joined) and "reads better" are different targets; the logit learns to imitate Terra's
   rectangle, not to read better. The fit is also per version-count unbalanced (as_found vs 4 alternatives).
2. A chosen join produces a box that overlaps the neighbour's own box; the headline's one-box-per-label rule then lets the
   join take the label (17 of the 18 losses in fitted s0). Whether the join or the pairing rule is at fault is not tested.
3. The training quarters' scores come from the stage readers (and NE's from the seed's reader): a mix of readers; both seeds
   share the same training rows, so the two seeds are not independent for the fit.
4. Grouped rectangles of split/join versions are my construction (grouped_rect: the box's grouped rectangle cut at the gap /
   grown to hold the join), not the pipeline's; a label point in a split's gap is held by neither half.
5. Preliminary: today's boxes and readers; Task 8's detector and readers replace them (rerun with --boxes-dir, --reader-name,
   --reader-root). Four-fold confirmation needs the other seeds' readers (the fit and choice for all four folds are already
   computed at seed 0 in params.json and chosen_boxes.parquet).
6. pytest: the full run excluded tests/test_wordread_words.py (other agent's file).

## Cause (NE seed 0, fitted arm; existing outputs, CPU; analysis script in the scratchpad, cause.py)
1. Overlapping outputs. 328 NE boxes chose join_nearest. The joined neighbour (recomputed by choose's own rule; in NE for
   327) also stays in the output: as its own box as found in 261, and as its own join in 67 (32 of these are mutual
   joins: A joins B and B joins A, kept once by the twin rule). So every join overlaps another output box.
2. The 17 lost labels held by a join. (One of the 18 lost was held by an as_found box.) Note: the box that held the
   label before was in every case itself now a join, never as found; no label's old box "survived" as found.
   - 10: the winner is the same source box, widened by its join (nothing else competed: 1 box holds the point in 8 of
     them, 2 in the rest).
   - 7: the winner is the neighbour's join box; it beat the old box (which was also a join) by T3's highest closed
     posterior in all 7 (e.g. 0.996 vs 0.414, 0.960 vs 0.551). In these the old box's own neighbour pick: 7 as_found,
     10 join (the second figure covers all 17: the winner's neighbour chose join in 10, as_found in 7).
   - The joined read was a wrong code in all 17 (P, RED, SLR, RSD, C-2, CVR...), and the joined rectangle meets 2 or
     more Terra words in 15 of 17 (1 in two cases; 25 and 14 in two big ones), so it reads more than one word.
3. Is the label right that a join was needed? Of the 697 NE boxes label_of marks join, 203 have a best-matching Terra
   word that 2 or more of our boxes meet (IoU >= 0.1: a split word); 494 do not (the word is met by one box). Also
   515 are trims (313 start, 202 end). So about 29% of join labels are Terra words split over our boxes.
4. Counterfactual (diagnostic only): each chosen join consumes its neighbour (neighbour boxes removed, except boxes that
   are themselves chosen joins; 243 rows removed): headline 76 of 285 [0.216, 0.321]; against without -0.056
   [-0.087, -0.024]; against the fitted arm as run -0.004 [-0.011, 0.000]. Consuming the neighbour does not recover the
   loss (it is slightly worse): the loss is not caused by the leftover neighbour box.
Reading: the lost labels come from the joined box reading two or more words (wrong code), and the fitted choice
picking join for 328 boxes, 70% of whose labels are not split words.

## Rerun with ink labels (Task 9 fb02e3b: label_weights + soft fit_choice; runs "ink-ne-s0/s1"; page rebuilt)
run_choose_pack.py now builds `weight` per version with choose.label_weights (page ink per sheet, Terra words meeting any
version; no IoU) and fits fold NE on 4288 boxes (4195 before). Params NE: beta 0.837; alpha split -1.557, join +0.243, trim_start
-0.306, trim_end -0.292; gamma along split -0.350, join -0.223; gamma_across join +0.013. Total label weight on NE boxes:
as_found 1346, join 745, split 168, trim_start 985, trim_end 977.
- Headline (of 285), fitted: seed 0 46 (without 92), seed 1 62 (88). Paired vs without: seed 0 -0.161 [-0.212, -0.105];
  seed 1 -0.091 [-0.137, -0.048]. Worse than with IoU labels (77 and 84); plain unchanged (51, 69).
- Kinds the fit now picks: seed 0 join_nearest 1559, as_found 880, trim_end 8, trim_start 5 (seed 1: join 1484, as_found 948,
  trim_end 14, trim_start 6); no split. Before: join 328.
- Of the joins picked (NE, seed 0), a Terra word split across the two boxes (both meeting it at IoU >= 0.1): 83 of 1559
  (5%); seed 1 80 of 1484. The earlier run: 19 of 328 (6%) by this same test (the 29% figure was a different test: join-labelled
  boxes whose best word is met by two or more boxes).
- Box numbers IoU 0.5 (seed 0): matched 498 (without 850), missed 2207, merged 915, split 279; codes right vs answers where
  matched: 10 of 17 (without 63 of 77). Page (Mission, baseline -> now): matched 447 -> ~ (see numbers.json), headline 0.377 -> 0.164.
- Not tested: why the argmax of the fitted logit picks join for 64% of boxes when the join weights are 745 of ~4,450.

## Why joins (ink-ne-s0 rows and params; CPU; analysis script why.py in the scratchpad, nothing changed in choose.py)
The contradiction dissolves: the fit is at its optimum, and the argmax picks join because it stands alone while as_found's
probability is shared with two near-copies (the trims). The "17% of the label weight" was the raw weight total (745 of ~4450),
not the fit's target (below).
1. In-sample, 4288 training boxes (fold NE's fit rows, refit reproduces params.json to the digit, nll 5484.670):
   mean fitted P vs mean target (target = box's weights over their sum, times min(sum, 1)):
   as_found 0.304 vs 0.271; join 0.277 vs 0.239; split 0.034 vs 0.033; trim_end 0.194 vs 0.181; trim_start 0.191 vs 0.179.
   (Mean P exceeds mean target a little everywhere because boxes with total weight under 1 are discounted in the target but
   not in P: sum of caps 3875 of 4288 boxes; 317 boxes have total weight < 0.5.) Gradient at the solution, from the moment
   conditions: every component under 0.06, norm 0.064 (beta -0.0001); the fit has converged. fit_choice does not return its
   iteration count. Argmax picks in training: as_found 2283, join 1995, trims 10.
2. Features. Training boxes with a join; (score_join - score_as_found), mean / median, share above 0:
   - boxes where as_found holds more label weight: -0.113 / -0.029, 19.6% above 0;
   - boxes where join holds more: -0.036 / -0.001, 44.6% above 0.
   Join's score is on average LOWER than as_found's (all quarters: mean -0.05 to -0.10, 23-38% above 0): no systematic reader
   bias towards joined crops. Both crops read mostly "other" (join 73%, as_found 71% of best hypotheses) or "none" (23%, 23%),
   a code in 4.5% and 6.2% (NE, seed 0 reads). Along (in char widths) median 0.59 where as_found holds more, 0.37 where join holds more
   (means 1.65, 0.93); across medians 0.70 and 0.60.
3. NE against the training quarters: score differences alike (NE mean -0.047, median -0.006, 38% above 0; others -0.070 / -0.075 /
   -0.103 means, 36% / 29% / 23% above 0). NE's join is NEARER: median along 0.14 (training quarters 0.42-0.52), p90 3.3 (3.5-5.4);
   so the fitted along penalty bites less in NE. The as_found scores per quarter are alike (NE mean -0.099; others -0.107, -0.063,
   -0.067). NE mean P by kind: as_found 0.304, join 0.320 (training 0.304, 0.277), i.e. NE joins get a slightly higher share.
4. Mechanics. Candidates per box: 2 (895 boxes), 3 (53), 4 (2818), 5 (5028): usually as_found, join, trim_start, trim_end (+ split).
   as_found and the two trims are near-copies of one crop (trims differ by one character width), so their probability splits
   three ways while join stands alone. NE mean max-probability by chosen kind: as_found 0.335, join 0.374, trim_end 0.280,
   trim_start 0.289. Of the NE boxes that choose join: mean P(join) 0.374, P(as_found) 0.288, P(as_found + trims) 0.609;
   in 83.5% of them P(as_found + both trims) exceeds P(join). So the argmax makes join win the 64% of boxes with a modest
   probability, while the fitted probabilities are calibrated to the targets (mean P(join) 0.28-0.32, not 0.64).
Consequence (a fact, not a ruling): choose's argmax over the versions of a soft fit is not the version-kind's probability; the
decision under the fitted probabilities, with near-duplicate versions, is what drives the picks.

## Pack preliminary (item 4; today's NE boxes as found, today's readers; runs pack-ne-s0/s1; PRELIMINARY)
run_choose_pack.py's pack path now calls context.read_posterior (4273cdf; pack = packs/sf-2000.parquet via Pack.load, the
box's own quarter's fit, strings = the code forms, "" and the reader's two best paths). Every NE box is read once (CPU, 4
workers); the "without" arm is the stage's reading and reproduces the base runs' labels exactly. Pack read of a box: closed =
the code forms' posterior renormalised over the codes; the text = the likeliest named string (None where OTHER is likeliest:
40.7% / 37.3% of NE boxes at seeds 0 / 1).
- Headline (of 285): seed 0 without 92, pack 89, paired -0.0105 [-0.0238, 0.0000]; seed 1 without 88, pack 87, paired -0.0035
  [-0.0171, +0.0071]. No rise.
- Code labels changed (headline labels): seed 0 right->wrong 3, wrong->right 0; seed 1 right->wrong 2, wrong->right 1.
- Read numbers against Terra (boxes matched to a Terra word at IoU 0.5 and reading exactly its string; without = the
  reader's best path in the classifier's direction; pack = the pack's likeliest string):
  | kind | matched | without s0 / s1 | pack s0 / s1 |
  |---|---|---|---|
  | street names | 153 | 65 / 65 | 88 / 82 |
  | block numbers | 478 | 371 / 375 | 431 / 433 |
  | zoning codes | 203 | 99 / 101 | 108 / 111 |
  The older code measure (read_numbers, boxes matched at 0.5 whose Terra string is a printed form): same as Terra 165 -> 162
  (s0), 159 -> 160 (s1); codes right against the owner's answers where matched: 63 -> 61 of 77 (s0), 58 -> 58 (s1).
- Box numbers are not comparable: the pack has no "none" hypothesis but the empty string, so boxes read as text rise (IoU 0.5,
  ours_unmatched_text 1080 -> 1276, ours_unmatched_none 522 -> 326 at seed 0).
- By the written rule (4): the headline does not rise (interval does not exclude zero, upward), streets, blocks and codes rise
  against Terra, codes against the owner's labels do not rise. Not kept on the headline; not decided here.
- Page: pipeline/data/solver/experiments/wordread/review/pack-prelim/review.html (Mission sheet p07; boxes as found; every
  box's text is the pack's likeliest string, the best path where OTHER is likeliest; the NE boxes by the seed-0 reader, the
  other quarters' boxes by the stage readers). On the page matched 447, merged_with_word 293 as in baseline (boxes unchanged);
  text read same as Terra 330 -> 373 (streets 57 -> 63, blocks 215 -> 244, codes 58 -> 66 of 98); headline on p07's 122 labels 46 -> 46.
