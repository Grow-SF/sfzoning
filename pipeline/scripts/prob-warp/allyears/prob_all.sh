#!/bin/zsh
# prob.sh for every sheet of every zoning series that has the old chain's inputs, PROB_P at a time (default 3,
# THREADS 3 each).  Log: run/prob_all.log.  usage: prob_all.sh
A=${PW_CODE:-$(cd "$(dirname "$0")" && pwd)}                 # this folder (prob-warp/allyears)
P=${PW_PIPELINE:-$(cd "$A/../../.." && pwd)}                 # the repo's pipeline folder
RUN=${PW_RUN:-$P/data/prob-warp/run}                          # outputs, per series/sheet
JOBS=()
for s in $RUN/*(/); do
  for d in $s/*(/); do
    [ -f $d/reg2_${d:t}.npz ] && [ -f $d/dm_sides_${d:t}.npz ] && JOBS+=("${s:t} ${d:t}")
  done
done
echo "START ${#JOBS} sheets $(date +%H:%M:%S)"
print -l $JOBS | xargs -P ${PROB_P:-3} -L 1 sh -c 'THREADS=3 sh '$A'/prob.sh $0 $1 | grep -E "^(FAILED|SKIP)|done"'
echo "ALL done $(date +%H:%M:%S)"
sysctl vm.swapusage
