#!/bin/zsh
# fit_one_year.sh for each year in turn, each logged to WARP_LOGS/YEAR.log.  Exits non-zero if any sheet failed or a
# year fitted none.  usage: [JOBS=3] fit_every_year.sh [YEAR ...]      (default: the 13 years the warp corrects)
CK=${0:A:h}
source $CK/settings.sh
YEARS=($@)
(( ${#YEARS} )) || YEARS=(1921 1929 1935 1948 1960 1964 1968 1972 1975 1980 1980-1990-zoning 1990-1996-zoning 2000-digital-zoning)
mkdir -p $WARP_LOGS
bad=0
for s in $YEARS; do
  L=$WARP_LOGS/$s.log
  zsh $CK/fit_one_year.sh $s > $L 2>&1
  ok=$(grep -c '^OK' $L) failed=$(grep -c '^FAILED' $L)
  echo "$s: $ok sheets fitted, $failed failed $(date +%H:%M:%S)"
  (( failed == 0 && ok > 0 )) || bad=1
done
exit $bad
