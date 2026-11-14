#!/bin/sh
# the correction for one sheet of any series (the 2000 steps and settings; see ay.py), logged to run/SERIES/SHEET/corr.log
# with a STEP line (name, end time) after each step.  usage: corr.sh SERIES SHEET
export SERIES=$1 SHEET=$2
A=${PW_CODE:-$(cd "$(dirname "$0")" && pwd)}                 # this folder (prob-warp/allyears)
P=${PW_PIPELINE:-$(cd "$A/../../.." && pwd)}                 # the repo's pipeline folder
RUN=${PW_RUN:-$P/data/prob-warp/run}                          # outputs, per series/sheet
TPY=${PW_TPY:?set PW_TPY to the torch environment python (README)}
O=$RUN/$1/$2
mkdir -p $O
cd $P || exit 1
L=$O/corr.log
[ -f $O/confit_$2_p1.npz ] && [ -f $O/gcpmoved_$2_p1.png ] && { echo "SKIP $1 $2 (done)"; exit 0; }
echo "STAGE $1 $2 start $(date +%H:%M:%S)" | tee $L
st() { echo "STEP $1 $(python3 -c 'import time; print(f"{time.time():.2f}")')" >> $L; }
st begin
# E3c: the side points and their pixels (sides.py) beside regprep2; independent of each other
.venv/bin/python $A/sides.py > $O/sides.log 2>&1 &
SIDES=$!
.venv/bin/python $A/regprep2.py >> $L 2>&1 || { echo "FAILED $1 $2 targets" | tee -a $L; exit 1; }
wait $SIDES || { cat $O/sides.log >> $L; echo "FAILED $1 $2 sides" | tee -a $L; exit 1; }
cat $O/sides.log >> $L
st regprep2+sides
.venv/bin/python $A/distmeasure.py today >> $L 2>&1 || { echo "FAILED $1 $2 measure" | tee -a $L; exit 1; }
st distmeasure_today
.venv/bin/python $A/heldrule.py >> $L 2>&1 || { echo "FAILED $1 $2 held" | tee -a $L; exit 1; }
st heldrule
.venv/bin/python $A/conprep3.py >> $L 2>&1 || { echo "FAILED $1 $2 inputs" | tee -a $L; exit 1; }
st conprep3
.venv/bin/python $A/propagate.py >> $L 2>&1 || { echo "FAILED $1 $2 propagate" | tee -a $L; exit 1; }
st propagate
$TPY $A/confit6.py 1 100 1000 p1 2 1000 >> $L 2>&1 || { echo "FAILED $1 $2 fit" | tee -a $L; exit 1; }
st confit6
.venv/bin/python $A/distmeasure.py p1 >> $L 2>&1
st distmeasure_p1
.venv/bin/python $A/distmeasure.py today p1 >> $L 2>&1
st distmeasure_cmp
.venv/bin/python $A/gcpmoved.py p1 >> $L 2>&1
st gcpmoved
echo "STAGE $1 $2 done $(date +%H:%M:%S)" | tee -a $L
