#!/bin/zsh
# The whole warp, every series, from the pipeline's sheets: rebuild.sh's steps with check/'s settings (rebuild_env.sh;
# the 2000 sheets in SAME2000 mode), into rebuild-2026-11-13/run-notext and its data_ay / data_ay2 mirrors:
#   1. per sheet: p1, its mirror entry, prob6 (rebuild_all.sh -> rebuild_series.sh -> rebuild_one.sh, JOBS sheets at a
#      time; a sheet's finished fits are kept), each series compared with the rescued run
#   2. the choice between p1 and prob6, written into the data_ay2 mirror (mirror_prob.py -> choice.json)
#   3. 2000 p09 and p10 on the "plus" fit (apply_plus.sh)
#   4. every chosen sheet's warp: warped.tif, warped_paper.tif and footprints in data_ay2 (warp_prob.sh)
# usage: [JOBS=3] [PW_PIPELINE=DIR] rewarp_all.sh [SERIES ...]      (default: the 13 series of the original run)
set -e
CK=${0:A:h}
source $CK/rebuild_env.sh
echo "1. fits $(date +%H:%M:%S)"
JOBS=${JOBS:-3} zsh $CK/rebuild_all.sh $@
echo "2. choice $(date +%H:%M:%S)"
(cd $PW_PIPELINE && .venv/bin/python $PW_CODE/mirror_prob.py)
echo "3. plus $(date +%H:%M:%S)"
zsh $PW_CODE/apply_plus.sh
echo "4. warps $(date +%H:%M:%S)"
zsh $PW_CODE/warp_prob.sh
echo "done $(date +%H:%M:%S)"
