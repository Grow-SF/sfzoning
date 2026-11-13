#!/bin/sh
# the probabilistic correction with added evidence (sides_plus.py: later blocks, street edges) for one sheet, into
# run/SERIES/SHEET/plus (the plain prob6 run's files untouched).  Logged to plus/prob.log.  usage: prob_plus.sh SERIES SHEET
export SERIES=$1 SHEET=$2
A=/private/tmp/claude-501/-Users-sbuss-workspace-sfzoning/eb18d2fc-e22a-48a6-afbc-57c31cd3e134/scratchpad/allyears
SCR=${A%/allyears}
TPY=$SCR/torchenv/bin/python
O=$A/run/$1/$2
P=$O/plus
mkdir -p $P
L=$P/prob.log
cd /Users/sbuss/workspace/sfzoning/.claude/worktrees/cache-maybe/pipeline || exit 1
echo "STAGE $1 $2 start $(date +%H:%M:%S)" | tee $L
.venv/bin/python -W ignore $A/sides_plus.py >> $L 2>&1 || { echo "FAILED $1 $2 sides_plus" | tee -a $L; tail -n 5 $L; exit 1; }
.venv/bin/python -W ignore $A/ay_probprep_plus.py >> $L 2>&1 || { echo "FAILED $1 $2 probprep" | tee -a $L; tail -n 5 $L; exit 1; }
.venv/bin/python -W ignore $A/ay_regprep3.py >> $L 2>&1 || { echo "FAILED $1 $2 regprep3" | tee -a $L; tail -n 5 $L; exit 1; }
ln -sf $O/reg2_$2.npz $P/reg2_$2.npz
mv $O/regw_$2.npz $P/regw_$2.npz
PROBDIR=$P/ PERPOINT=1 WMAP=1 SHAPE=1 LOWMEM=1 THREADS=${THREADS:-5} $TPY $SCR/probfit.py $2 plus 800 >> $L 2>&1 || { echo "FAILED $1 $2 fit" | tee -a $L; tail -n 5 $L; exit 1; }
grep -E "street-edge|start \(today\)|stage 3|stage 4" $L | cut -c1-300
echo "STAGE $1 $2 done $(date +%H:%M:%S)" | tee -a $L
