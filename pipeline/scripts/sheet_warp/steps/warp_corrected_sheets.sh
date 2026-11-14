#!/bin/zsh
# warp every corrected sheet in the stand-in data folder with its chosen correction (WARP_FITS/choice.json): warped.tif,
# warped_paper.tif and their footprints, 2 at a time.  usage: warp_corrected_sheets.sh
A=${WARP_STEPS:-$(cd "$(dirname "$0")" && pwd)}                 # this folder (sheet_warp/steps)
P=${WARP_PIPELINE:-$(cd "$A/../../.." && pwd)}                 # the repo's pipeline folder
RUN=${WARP_FITS:-$P/data/sheet-warp/fits}                          # outputs, per series/sheet
PY=$P/.venv/bin/python
cd $P
JOBS=($($PY -c "
import json
c=json.load(open('$RUN/choice.json'))
print(' '.join(k for k,v in c.items() if v['fit']))"))
echo "START ${#JOBS} sheets $(date +%H:%M:%S)"
print -l $JOBS | xargs -P 2 -n 1 zsh $A/warp_one_corrected_sheet.sh
echo "ALL done $(date +%H:%M:%S)"
sysctl vm.swapusage
