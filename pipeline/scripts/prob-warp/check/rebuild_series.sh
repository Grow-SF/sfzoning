#!/bin/zsh
# rebuild_one.sh for the given sheets of a series (default: every sheet the rescued run has), JOBS at a time
# (default 4), each logged to rebuild-2026-11-13/logs/SERIES/SHEET.log; then each sheet's outputs compared with the
# rescued run (compare_rebuild.py) and, for the chain series, its chain pages with the rescued chain where they
# survive.  usage: [JOBS=4] rebuild_series.sh SERIES [SHEET ...]
CK=${0:A:h}
source $CK/rebuild_env.sh
SERIES=$1; shift
L=$PW_DATA/rebuild-2026-11-13/logs/$SERIES
mkdir -p $L
SH=($@)
[ ${#SH} -eq 0 ] && SH=($PW_RESCUED/allyears-run/$SERIES/*(/:t))
echo "REBUILD $SERIES: ${#SH} sheets, start $(date +%H:%M:%S)"
i=0
for sh in $SH; do
  { zsh $CK/rebuild_one.sh $SERIES $sh > $L/$sh.log 2>&1 && echo "OK $sh $(date +%H:%M:%S)" || echo "FAILED $sh: $(tail -3 $L/$sh.log)"; } &
  i=$((i + 1))
  [ $((i % ${JOBS:-4})) -eq 0 ] && wait
done
wait
echo "COMPARE $SERIES $(date +%H:%M:%S)"
ARGS=()
for sh in $SH; do ARGS+=($PW_RESCUED/allyears-run/$SERIES/$sh $PW_RUN/$SERIES/$sh); done
$PW_PIPELINE/.venv/bin/python $CK/compare_rebuild.py $ARGS
for sh in $SH; do
  for f in $sh/noblock.png ${sh}_c1/clean.png; do
    [ -f $PW_RESCUED/allyears-chain/$SERIES/$f ] && [ -f $PW_CHAIN/$SERIES/$f ] && {
      cmp -s $PW_RESCUED/allyears-chain/$SERIES/$f $PW_CHAIN/$SERIES/$f && echo "  chain identical  $f" || echo "  chain DIFFERS    $f"; }
  done
done
echo "REBUILD $SERIES done $(date +%H:%M:%S)"
