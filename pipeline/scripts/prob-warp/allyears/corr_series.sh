#!/bin/zsh
# corrections for every fitted sheet of one series, CORR_P at a time (default 3).  usage: corr_series.sh SERIES
A=${PW_CODE:-$(cd "$(dirname "$0")" && pwd)}                 # this folder (prob-warp/allyears)
P=${PW_PIPELINE:-$(cd "$A/../../.." && pwd)}                 # the repo's pipeline folder
RUN=${PW_RUN:-$P/data/prob-warp/run}                          # outputs, per series/sheet
W=${PW_WORK:-$P/data/work}/$1
SH=()
for d in $W/*(/); do
  [ -f $d/transform.json ] && [ -f $d/warped.tif ] && [ -f $d/blocks_px.geojson ] && [ -f $d/refine.json ] && SH+=(${d:t})
done
echo "SERIES $1: ${#SH} sheets, start $(date +%H:%M:%S)"
print -l $SH | xargs -P ${CORR_P:-3} -I{} sh $A/corr.sh $1 {}
echo "SERIES $1 done $(date +%H:%M:%S)"
sysctl vm.swapusage
