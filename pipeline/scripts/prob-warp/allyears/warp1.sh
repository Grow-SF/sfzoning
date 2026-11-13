#!/bin/zsh
# warp one sheet (SERIES/SHEET) of the data_ay2 mirror with its chosen correction.  usage: warp1.sh SERIES/SHEET
A=/private/tmp/claude-501/-Users-sbuss-workspace-sfzoning/eb18d2fc-e22a-48a6-afbc-57c31cd3e134/scratchpad/allyears
S=${A:h}
k=$1
cd /Users/sbuss/workspace/sfzoning/.claude/worktrees/cache-maybe/pipeline
export SERIES=${k%/*} SFZ_CORR=1 PYTHONPATH=$S/corrpatch SFZMAPS_DATA_DIR=$S/data_ay2
export PATH=/Users/sbuss/workspace/sfzoning/.claude/worktrees/cache-maybe/.pixi/envs/default/bin:$PATH
.venv/bin/python $A/warpK.py ${k#*/} both > $A/run/${k%/*}/${k#*/}/warp2.log 2>&1 && echo "ok $k $(date +%H:%M:%S)" || echo "FAILED $k"
