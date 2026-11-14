#!/bin/zsh
# rebuild_one.sh for the given sheets of a series (default: the series' sheets the warp corrects, allyears/sheets.py),
# JOBS at a time (default 4), each logged to rebuild-2026-11-13/logs/SERIES/SHEET.log.  Where this machine still has
# the approved run's outputs (PW_RESCUED), each sheet is then compared with them (compare_rebuild.py): p1 matches;
# prob6 differs wherever that run took text out (1960 on).  A fresh clone has no such folder and skips the comparison.
# usage: [JOBS=4] rebuild_series.sh SERIES [SHEET ...]
CK=${0:A:h}
source $CK/rebuild_env.sh
zsh $CK/torchenv.sh || exit 1                        # the torch environment, made if missing
SERIES=$1; shift
L=$PW_DATA/rebuild-2026-11-13/logs/$SERIES
mkdir -p $L
SH=($@)
(( ${#SH} )) || SH=($(cd $PW_PIPELINE && .venv/bin/python $PW_CODE/sheets.py $SERIES 2>/dev/null))
(( ${#SH} )) || { echo "REBUILD $SERIES: no sheets (allyears/sheets.py $SERIES gave none)"; exit 1; }
echo "REBUILD $SERIES: ${#SH} sheets, start $(date +%H:%M:%S)"
i=0
for sh in $SH; do
  { zsh $CK/rebuild_one.sh $SERIES $sh > $L/$sh.log 2>&1 && echo "OK $sh $(date +%H:%M:%S)" || echo "FAILED $sh: $(tail -3 $L/$sh.log)"; } &
  i=$((i + 1))
  [ $((i % ${JOBS:-4})) -eq 0 ] && wait
done
wait
if [ -d $PW_RESCUED/allyears-run/$SERIES ]; then
  echo "COMPARE $SERIES with the approved run $(date +%H:%M:%S)"
  ARGS=()
  for sh in $SH; do ARGS+=($PW_RESCUED/allyears-run/$SERIES/$sh $PW_RUN/$SERIES/$sh); done
  $PW_PIPELINE/.venv/bin/python $CK/compare_rebuild.py $ARGS
fi
echo "REBUILD $SERIES done $(date +%H:%M:%S)"
