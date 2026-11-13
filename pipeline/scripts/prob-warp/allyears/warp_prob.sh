#!/bin/zsh
# warp every corrected sheet in the data_ay2 mirror with its chosen correction (run/choice.json): warped.tif,
# warped_paper.tif and their footprints, 2 at a time.  usage: warp_prob.sh
A=/private/tmp/claude-501/-Users-sbuss-workspace-sfzoning/eb18d2fc-e22a-48a6-afbc-57c31cd3e134/scratchpad/allyears
S=${A:h}
PY=/Users/sbuss/workspace/sfzoning/.claude/worktrees/cache-maybe/pipeline/.venv/bin/python
cd /Users/sbuss/workspace/sfzoning/.claude/worktrees/cache-maybe/pipeline
JOBS=($($PY -c "
import json
c=json.load(open('$A/run/choice.json'))
print(' '.join(k for k,v in c.items() if v['fit']))"))
echo "START ${#JOBS} sheets $(date +%H:%M:%S)"
print -l $JOBS | xargs -P 2 -n 1 zsh $A/warp1.sh
echo "ALL done $(date +%H:%M:%S)"
sysctl vm.swapusage
