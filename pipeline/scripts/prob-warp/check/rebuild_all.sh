#!/bin/zsh
# rebuild_series.sh for each series in turn (default: the 13 series of the original run), each logged to
# rebuild-2026-11-13/rebuild_SERIES.log.  Exits non-zero if any sheet failed or a series rebuilt none.
# usage: [JOBS=3] rebuild_all.sh [SERIES ...]
CK=${0:A:h}
source $CK/rebuild_env.sh
SER=($@)
(( ${#SER} )) || SER=(1921 1929 1935 1948 1960 1964 1968 1972 1975 1980 1980-1990-zoning 1990-1996-zoning 2000-digital-zoning)
bad=0
for s in $SER; do
  L=$PW_DATA/rebuild-2026-11-13/rebuild_$s.log
  zsh $CK/rebuild_series.sh $s > $L 2>&1
  ok=$(grep -c '^OK' $L) failed=$(grep -c '^FAILED' $L)
  echo "$s: $ok ok, $failed failed, $(grep -c 'DIFFERS' $L) files differ $(date +%H:%M:%S)"
  (( failed == 0 && ok > 0 )) || bad=1
done
exit $bad
