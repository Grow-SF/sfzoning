#!/bin/zsh
# Is prob6 the same every run?  Fits a series' sheets again from their p1 inputs in $PW_RUN, into
# rebuild-2026-11-13/run-twice, then compares the two fields per sheet (field_diff.py: largest, p99 and median
# difference, page px).  Same code, same inputs, same settings (THREADS=3), so every difference is run-to-run noise.
# usage: [JOBS=4] fit_twice.sh SERIES [SHEET ...]
CK=${0:A:h}
source $CK/rebuild_env.sh
zsh $CK/torchenv.sh || exit 1                        # the torch environment, made if missing
SER=$1; shift
TWICE=$PW_DATA/rebuild-2026-11-13/run-twice
mkdir -p $TWICE/$SER
SH=($@)
[ ${#SH} -eq 0 ] && SH=($PW_RUN/$SER/*(/:t))
one() {
  local sh=$1 d=$TWICE/$SER/$1
  rm -rf $d && /bin/cp -cR $PW_RUN/$SER/$sh $d                         # the p1 inputs (copy-on-write clone)
  rm -f $d/confit_${sh}_prob6.npz $d/dm_${sh}_prob6.npz $d/dm_${sh}_prob6.png $d/prob.log
  PW_RUN=$TWICE THREADS=3 sh $PW_CODE/prob.sh $SER $sh > $d/twice.log 2>&1 && echo "  $sh done" ||
    echo "  FAILED $sh: $(tail -n 2 $d/twice.log)"
}
echo "fit again: ${#SH} sheets $(date +%H:%M:%S)"
i=0
for sh in $SH; do
  one $sh &
  i=$((i + 1))
  [ $((i % ${JOBS:-4})) -eq 0 ] && wait
done
wait
$PW_PIPELINE/.venv/bin/python $CK/field_diff.py $PW_RUN $TWICE $SER
echo "done $(date +%H:%M:%S)"
