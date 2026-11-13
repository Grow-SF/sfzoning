#!/bin/zsh
# rebuild_series.sh for each series in turn (default: the 13 series of the original run), each logged to
# rebuild-2026-11-13/rebuild_SERIES.log.  usage: [JOBS=3] rebuild_all.sh [SERIES ...]
CK=${0:A:h}
source $CK/rebuild_env.sh
for s in ${@:-1921 1929 1935 1948 1960 1964 1968 1972 1975 1980 1980-1990-zoning 1990-1996-zoning 2000-digital-zoning}; do
  zsh $CK/rebuild_series.sh $s > $PW_DATA/rebuild-2026-11-13/rebuild_$s.log 2>&1
  echo "$s: $(grep -c '^OK' $PW_DATA/rebuild-2026-11-13/rebuild_$s.log) ok, $(grep -c '^FAILED' $PW_DATA/rebuild-2026-11-13/rebuild_$s.log) failed, $(grep -c 'DIFFERS' $PW_DATA/rebuild-2026-11-13/rebuild_$s.log) files differ $(date +%H:%M:%S)"
done
