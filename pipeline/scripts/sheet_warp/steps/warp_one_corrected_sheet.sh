#!/bin/zsh
# warp one sheet (SERIES/SHEET) of the stand-in data folder with its chosen correction.  usage: warp_one_corrected_sheet.sh SERIES/SHEET
A=${WARP_STEPS:-$(cd "$(dirname "$0")" && pwd)}                 # this folder (sheet_warp/steps)
P=${WARP_PIPELINE:-$(cd "$A/../../.." && pwd)}                 # the repo's pipeline folder
RUN=${WARP_FITS:-$P/data/sheet-warp/fits}                          # outputs, per series/sheet
k=$1
cd $P
export SERIES=${k%/*} SFZ_CORR=1 PYTHONPATH=$A/../correction_patch SFZMAPS_DATA_DIR=${WARP_STAND_IN_CHOSEN:-$P/data/sheet-warp/stand-in-data-chosen}
export PATH=${WARP_PIXI_BIN:-$P/../.pixi/envs/default/bin}:$PATH
.venv/bin/python $A/warp_with_correction.py ${k#*/} both > $RUN/${k%/*}/${k#*/}/warp2.log 2>&1 && echo "ok $k $(date +%H:%M:%S)" || echo "FAILED $k"
