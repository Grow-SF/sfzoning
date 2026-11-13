#!/bin/zsh
# 2000 p09 and p10: the "plus" fit (prob_plus.sh: later blocks and street edges as added evidence), and their
# transforms in the data_ay2 mirror pointed at it (the prob6 ones kept as transform_before_plus.json).  This is the warp
# part of k8/do_plus.sh as run on 2026-11-09; its other steps redid the zoning-lines sheet outputs.  Run after
# mirror_prob.py and before warp_prob.sh.  usage: apply_plus.sh
A=${PW_CODE:-$(cd "$(dirname "$0")" && pwd)}                 # this folder (prob-warp/allyears)
P=${PW_PIPELINE:-$(cd "$A/../../.." && pwd)}                 # the repo's pipeline folder
RUN=${PW_RUN:-$P/data/prob-warp/run}                          # outputs, per series/sheet
M2=${PW_MIRROR2:-$P/data/prob-warp/data_ay2}
W=$M2/work/2000-digital-zoning
for nn in 09 10; do
  sh=2000-digital-zoning-p$nn
  [ -f $RUN/2000-digital-zoning/$sh/plus/confit_${sh}_plus.npz ] || THREADS=${THREADS:-3} sh $A/prob_plus.sh 2000-digital-zoning $sh
  [ -f $W/$sh/transform_before_plus.json ] || cp $W/$sh/transform.json $W/$sh/transform_before_plus.json
  $P/.venv/bin/python -c "
import json
p = '$W/$sh/transform.json'
tj = json.load(open(p))
tj['correction'] = {'file': '$RUN/2000-digital-zoning/$sh/plus/confit_${sh}_plus.npz'}
json.dump(tj, open(p, 'w'))
print('$sh correction ->', tj['correction']['file'].split('/')[-1])"
done
