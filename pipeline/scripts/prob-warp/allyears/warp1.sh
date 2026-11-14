#!/bin/zsh
# warp one sheet (SERIES/SHEET) of the data_ay2 mirror with its chosen correction.  usage: warp1.sh SERIES/SHEET
A=${PW_CODE:-$(cd "$(dirname "$0")" && pwd)}                 # this folder (prob-warp/allyears)
P=${PW_PIPELINE:-$(cd "$A/../../.." && pwd)}                 # the repo's pipeline folder
RUN=${PW_RUN:-$P/data/prob-warp/run}                          # outputs, per series/sheet
k=$1
cd $P
export SERIES=${k%/*} SFZ_CORR=1 PYTHONPATH=$A/../corrpatch SFZMAPS_DATA_DIR=${PW_MIRROR2:-$P/data/prob-warp/data_ay2}
export PATH=${PW_PIXI_BIN:-$P/../.pixi/envs/default/bin}:$PATH
.venv/bin/python $A/warpK.py ${k#*/} both > $RUN/${k%/*}/${k#*/}/warp2.log 2>&1 && echo "ok $k $(date +%H:%M:%S)" || echo "FAILED $k"
