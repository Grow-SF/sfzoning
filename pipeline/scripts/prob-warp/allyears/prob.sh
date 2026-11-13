#!/bin/sh
# the probabilistic correction (prob6, as run on 2000 p06/p07) for one sheet of any series, after the old correction
# chain (corr.sh: reg2, dm_sides, pix, dm_today, dm_p1 in run/SERIES/SHEET).  Logged to run/SERIES/SHEET/prob.log.
# usage: prob.sh SERIES SHEET
export SERIES=$1 SHEET=$2
A=/private/tmp/claude-501/-Users-sbuss-workspace-sfzoning/eb18d2fc-e22a-48a6-afbc-57c31cd3e134/scratchpad/allyears
SCR=${A%/allyears}
TPY=$SCR/torchenv/bin/python
O=$A/run/$1/$2
L=$O/prob.log
cd /Users/sbuss/workspace/sfzoning/.claude/worktrees/cache-maybe/pipeline || exit 1
[ -f $O/confit_$2_prob6.npz ] && [ -f $O/dm_$2_prob6.npz ] && { echo "SKIP $1 $2 (done)"; exit 0; }
[ -f $O/reg2_$2.npz ] && [ -f $O/dm_sides_$2.npz ] || { echo "SKIP $1 $2 (no old chain inputs)"; exit 0; }
echo "STAGE $1 $2 start $(date +%H:%M:%S)" | tee $L
.venv/bin/python -W ignore $A/ay_probprep.py >> $L 2>&1 || { echo "FAILED $1 $2 probprep" | tee -a $L; exit 1; }
.venv/bin/python -W ignore $A/ay_regprep3.py >> $L 2>&1 || { echo "FAILED $1 $2 regprep3" | tee -a $L; exit 1; }
PROBDIR=$O/ PERPOINT=1 WMAP=1 SHAPE=1 LOWMEM=1 THREADS=${THREADS:-3} $TPY $SCR/probfit.py $2 prob6 800 >> $L 2>&1 || { echo "FAILED $1 $2 fit" | tee -a $L; exit 1; }
.venv/bin/python -W ignore $A/distmeasure.py prob6 >> $L 2>&1 || { echo "FAILED $1 $2 measure" | tee -a $L; exit 1; }
.venv/bin/python $A/ay_probsum.py $1 $2 >> $L 2>&1
rm -f $O/regw_$2.npz
echo "STAGE $1 $2 done $(date +%H:%M:%S)" | tee -a $L
tail -n 1 $L
