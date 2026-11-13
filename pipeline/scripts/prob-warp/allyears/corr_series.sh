#!/bin/zsh
# corrections for every fitted sheet of one series, CORR_P at a time (default 3).  usage: corr_series.sh SERIES
A=/private/tmp/claude-501/-Users-sbuss-workspace-sfzoning/eb18d2fc-e22a-48a6-afbc-57c31cd3e134/scratchpad/allyears
W=/Users/sbuss/workspace/sfzoning/.claude/worktrees/cache-maybe/pipeline/data/work/$1
SH=()
for d in $W/*(/); do
  [ -f $d/transform.json ] && [ -f $d/warped.tif ] && [ -f $d/blocks_px.geojson ] && [ -f $d/refine.json ] && SH+=(${d:t})
done
echo "SERIES $1: ${#SH} sheets, start $(date +%H:%M:%S)"
print -l $SH | xargs -P ${CORR_P:-3} -I{} sh $A/corr.sh $1 {}
echo "SERIES $1 done $(date +%H:%M:%S)"
sysctl vm.swapusage
