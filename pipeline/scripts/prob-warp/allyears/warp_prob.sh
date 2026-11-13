#!/bin/zsh
# warp every corrected sheet in the data_ay2 mirror with its chosen correction (run/choice.json): warped.tif,
# warped_paper.tif and their footprints, 2 at a time.  usage: warp_prob.sh
A=${PW_CODE:-$(cd "$(dirname "$0")" && pwd)}                 # this folder (prob-warp/allyears)
P=${PW_PIPELINE:-$(cd "$A/../../.." && pwd)}                 # the repo's pipeline folder
RUN=${PW_RUN:-$P/data/prob-warp/run}                          # outputs, per series/sheet
PY=$P/.venv/bin/python
cd $P
JOBS=($($PY -c "
import json
c=json.load(open('$RUN/choice.json'))
print(' '.join(k for k,v in c.items() if v['fit']))"))
echo "START ${#JOBS} sheets $(date +%H:%M:%S)"
print -l $JOBS | xargs -P 2 -n 1 zsh $A/warp1.sh
echo "ALL done $(date +%H:%M:%S)"
sysctl vm.swapusage
