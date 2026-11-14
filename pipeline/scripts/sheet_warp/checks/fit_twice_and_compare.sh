#!/bin/zsh
# Is the probabilistic correction the same every run?  Fits a year's sheets again from their first-correction inputs
# in WARP_FITS, into WARP_DATA/fits-again, then compares the two correction fields per sheet (field_differences.py:
# largest, 99th percentile and median difference, page px).  Same code, inputs and settings, so any difference is
# run-to-run noise.  usage: [JOBS=4] fit_twice_and_compare.sh YEAR [SHEET ...]
CK=${0:A:h}
source $CK/../settings.sh
zsh $CK/../make_torch_environment.sh || exit 1
YEAR=$1; shift
AGAIN=$WARP_DATA/fits-again
mkdir -p $AGAIN/$YEAR
SH=($@)
(( ${#SH} )) || SH=($WARP_FITS/$YEAR/*(/:t))
one() {
  local sh=$1 d=$AGAIN/$YEAR/$1
  rm -rf $d && /bin/cp -cR $WARP_FITS/$YEAR/$sh $d                    # the first-correction inputs (macOS clone)
  rm -f $d/confit_${sh}_prob6.npz $d/dm_${sh}_prob6.npz $d/dm_${sh}_prob6.png $d/prob.log
  WARP_FITS=$AGAIN THREADS=3 sh $WARP_STEPS/probabilistic_correction.sh $YEAR $sh > $d/again.log 2>&1 && echo "  $sh done" ||
    echo "  FAILED $sh: $(tail -n 2 $d/again.log)"
}
echo "fit again: ${#SH} sheets $(date +%H:%M:%S)"
i=0
for sh in $SH; do
  one $sh &
  i=$((i + 1))
  [ $((i % ${JOBS:-4})) -eq 0 ] && wait
done
wait
$WARP_PIPELINE/.venv/bin/python $CK/field_differences.py $WARP_FITS $AGAIN $YEAR
echo "done $(date +%H:%M:%S)"
