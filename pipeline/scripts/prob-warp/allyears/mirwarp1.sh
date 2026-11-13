#!/bin/zsh
# one sheet's mirror entry (build_mirror_ay.py SERIES SHEET, real data dir) and its corrected warps (warpK.py, mirror
# data dir + the patch).  usage: mirwarp1.sh SERIES SHEET
A=${PW_CODE:-$(cd "$(dirname "$0")" && pwd)}                 # this folder (prob-warp/allyears)
P=${PW_PIPELINE:-$(cd "$A/../../.." && pwd)}                 # the repo's pipeline folder
RUN=${PW_RUN:-$P/data/prob-warp/run}                          # outputs, per series/sheet
PY=$P/.venv/bin/python
cd $P
env -u SFZMAPS_DATA_DIR -u SFZ_CORR -u PYTHONPATH $PY $A/build_mirror_ay.py $1 $2 || exit 1
SERIES=$1 SFZ_CORR=1 PYTHONPATH=$A/../corrpatch SFZMAPS_DATA_DIR=${PW_MIRROR:-$P/data/prob-warp/data_ay} \
  PATH=${PW_PIXI_BIN:-$P/../.pixi/envs/default/bin}:$PATH \
  $PY $A/warpK.py $2 both > $RUN/$1/$2/warp.log 2>&1 || { tail -3 $RUN/$1/$2/warp.log; exit 1; }
grep -h "copied exactly\|wrote" $RUN/$1/$2/warp.log
