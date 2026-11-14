#!/bin/sh
# the probabilistic correction with added evidence (extra_evidence_points.py: later blocks, street edges) for one sheet, into
# run/SERIES/SHEET/plus (the plain prob6 run's files untouched).  Logged to plus/prob.log.  usage: extra_evidence_correction.sh SERIES SHEET
export SERIES=$1 SHEET=$2
A=${WARP_STEPS:-$(cd "$(dirname "$0")" && pwd)}                 # this folder (sheet_warp/steps)
PIPE=${WARP_PIPELINE:-$(cd "$A/../../.." && pwd)}              # the repo's pipeline folder
RUN=${WARP_FITS:-$PIPE/data/sheet-warp/fits}                       # outputs, per series/sheet
TPY=${WARP_TORCH_PYTHON:?set WARP_TORCH_PYTHON to the torch environment python (README)}
O=$RUN/$1/$2
P=$O/plus
mkdir -p $P
L=$P/prob.log
cd $PIPE || exit 1
echo "STAGE $1 $2 start $(date +%H:%M:%S)" | tee $L
.venv/bin/python -W ignore $A/extra_evidence_points.py >> $L 2>&1 || { echo "FAILED $1 $2 sides_plus" | tee -a $L; tail -n 5 $L; exit 1; }
.venv/bin/python -W ignore $A/extra_evidence_inputs.py >> $L 2>&1 || { echo "FAILED $1 $2 probprep" | tee -a $L; tail -n 5 $L; exit 1; }
.venv/bin/python -W ignore $A/block_line_weights.py >> $L 2>&1 || { echo "FAILED $1 $2 regprep3" | tee -a $L; tail -n 5 $L; exit 1; }
ln -sf $O/reg2_$2.npz $P/reg2_$2.npz
mv $O/regw_$2.npz $P/regw_$2.npz
PROBDIR=$P/ PERPOINT=1 WMAP=1 SHAPE=1 LOWMEM=1 THREADS=${THREADS:-5} $TPY $A/probabilistic_fit.py $2 plus 800 >> $L 2>&1 || { echo "FAILED $1 $2 fit" | tee -a $L; tail -n 5 $L; exit 1; }
grep -E "street-edge|start \(today\)|stage 3|stage 4" $L | cut -c1-300
echo "STAGE $1 $2 done $(date +%H:%M:%S)" | tee -a $L
