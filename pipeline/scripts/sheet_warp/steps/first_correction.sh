#!/bin/sh
# the correction for one sheet of any series (the 2000 steps and settings; see warp_common.py), logged to run/SERIES/SHEET/corr.log
# with a STEP line (name, end time) after each step.  usage: first_correction.sh SERIES SHEET
export SERIES=$1 SHEET=$2
A=${WARP_STEPS:-$(cd "$(dirname "$0")" && pwd)}                 # this folder (sheet_warp/steps)
P=${WARP_PIPELINE:-$(cd "$A/../../.." && pwd)}                 # the repo's pipeline folder
RUN=${WARP_FITS:-$P/data/sheet-warp/fits}                          # outputs, per series/sheet
TPY=${WARP_TORCH_PYTHON:?set WARP_TORCH_PYTHON to the torch environment python (README)}
O=$RUN/$1/$2
mkdir -p $O
cd $P || exit 1
L=$O/corr.log
[ -f $O/confit_$2_p1.npz ] && [ -f $O/gcpmoved_$2_p1.png ] && { echo "SKIP $1 $2 (done)"; exit 0; }
echo "STAGE $1 $2 start $(date +%H:%M:%S)" | tee $L
st() { echo "STEP $1 $(.venv/bin/python -c 'import time; print(f"{time.time():.2f}")')" >> $L; }
st begin
# E3c: the side points and their pixels (block_side_points.py) beside regprep2; independent of each other
.venv/bin/python $A/block_side_points.py > $O/sides.log 2>&1 &
SIDES=$!
.venv/bin/python $A/fit_targets.py >> $L 2>&1 || { echo "FAILED $1 $2 targets" | tee -a $L; exit 1; }
wait $SIDES || { cat $O/sides.log >> $L; echo "FAILED $1 $2 sides" | tee -a $L; exit 1; }
cat $O/sides.log >> $L
st regprep2+sides
.venv/bin/python $A/measure_fit.py today >> $L 2>&1 || { echo "FAILED $1 $2 measure" | tee -a $L; exit 1; }
st distmeasure_today
.venv/bin/python $A/held_sides.py >> $L 2>&1 || { echo "FAILED $1 $2 held" | tee -a $L; exit 1; }
st heldrule
.venv/bin/python $A/first_correction_inputs.py >> $L 2>&1 || { echo "FAILED $1 $2 inputs" | tee -a $L; exit 1; }
st conprep3
.venv/bin/python $A/propagate_shifts.py >> $L 2>&1 || { echo "FAILED $1 $2 propagate" | tee -a $L; exit 1; }
st propagate
$TPY $A/first_correction_fit.py 1 100 1000 p1 2 1000 >> $L 2>&1 || { echo "FAILED $1 $2 fit" | tee -a $L; exit 1; }
st confit6
.venv/bin/python $A/measure_fit.py p1 >> $L 2>&1
st distmeasure_p1
.venv/bin/python $A/measure_fit.py today p1 >> $L 2>&1
st distmeasure_cmp
.venv/bin/python $A/control_points_picture.py p1 >> $L 2>&1
st gcpmoved
echo "STAGE $1 $2 done $(date +%H:%M:%S)" | tee -a $L
