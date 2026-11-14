#!/bin/zsh
# fit_one_sheet.sh for the given sheets of a year (default: the year's sheets the warp corrects, sheets_to_correct.py),
# JOBS at a time (default 4), each logged to WARP_LOGS/YEAR/SHEET.log.  usage: [JOBS=4] fit_one_year.sh YEAR [SHEET ...]
CK=${0:A:h}
source $CK/settings.sh
zsh $CK/make_torch_environment.sh || exit 1
YEAR=$1; shift
L=$WARP_LOGS/$YEAR
mkdir -p $L
SH=($@)
(( ${#SH} )) || SH=($(cd $WARP_PIPELINE && .venv/bin/python $WARP_STEPS/sheets_to_correct.py $YEAR 2>/dev/null))
(( ${#SH} )) || { echo "$YEAR: no sheets (sheets_to_correct.py $YEAR gave none)"; exit 1; }
echo "$YEAR: ${#SH} sheets, start $(date +%H:%M:%S)"
i=0
for sh in $SH; do
  { zsh $CK/fit_one_sheet.sh $YEAR $sh > $L/$sh.log 2>&1 && echo "OK $sh $(date +%H:%M:%S)" || echo "FAILED $sh: $(tail -3 $L/$sh.log)"; } &
  i=$((i + 1))
  [ $((i % ${JOBS:-4})) -eq 0 ] && wait
done
wait
echo "$YEAR done $(date +%H:%M:%S)"
