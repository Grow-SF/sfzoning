#!/bin/zsh
# 2000 p09 and p10: the "plus" fit (extra_evidence_correction.sh: later blocks and street edges as added evidence), and their
# transforms in the stand-in data folder pointed at it (the prob6 ones kept as transform_before_plus.json).  This is the warp
# part of k8/do_plus.sh as run on 2026-11-09; its other steps redid the zoning-lines sheet outputs.  Run after
# choose_correction.py and before warp_corrected_sheets.sh.  usage: extra_evidence_2000.sh
A=${WARP_STEPS:-$(cd "$(dirname "$0")" && pwd)}                 # this folder (sheet_warp/steps)
P=${WARP_PIPELINE:-$(cd "$A/../../.." && pwd)}                 # the repo's pipeline folder
RUN=${WARP_FITS:-$P/data/sheet-warp/fits}                          # outputs, per series/sheet
M2=${WARP_STAND_IN_CHOSEN:-$P/data/sheet-warp/stand-in-data-chosen}
W=$M2/work/2000-digital-zoning
for nn in 09 10; do
  sh=2000-digital-zoning-p$nn
  [ -f $RUN/2000-digital-zoning/$sh/plus/confit_${sh}_plus.npz ] || THREADS=${THREADS:-3} sh $A/extra_evidence_correction.sh 2000-digital-zoning $sh
  [ -f $W/$sh/transform_before_plus.json ] || cp $W/$sh/transform.json $W/$sh/transform_before_plus.json
  $P/.venv/bin/python -c "
import json
p = '$W/$sh/transform.json'
tj = json.load(open(p))
tj['correction'] = {'file': '$RUN/2000-digital-zoning/$sh/plus/confit_${sh}_plus.npz'}
json.dump(tj, open(p, 'w'))
print('$sh correction ->', tj['correction']['file'].split('/')[-1])"
done
