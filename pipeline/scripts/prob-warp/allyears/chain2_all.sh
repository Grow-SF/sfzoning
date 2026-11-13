#!/bin/zsh
# sheetAY2.sh for every sheet of every series that has the earlier chain's Terra boxes, except 2000 (run k8), as one
# queue, CHAIN_P at a time (default 2), NPROC code-fit workers each (default 2); sheets with more boxes first.
# Each sheet logs to chain2/SERIES/SHEET.log.  usage: chain2_all.sh
A=/private/tmp/claude-501/-Users-sbuss-workspace-sfzoning/eb18d2fc-e22a-48a6-afbc-57c31cd3e134/scratchpad/allyears
PY=/Users/sbuss/workspace/sfzoning/.claude/worktrees/cache-maybe/pipeline/.venv/bin/python
$PY - $A > $A/run/chain2_jobs.txt <<'EOF'
import json, sys
from pathlib import Path
A = Path(sys.argv[1])
c = json.load(open(A / "run/choice.json"))
jobs = []
for k, v in c.items():
    ser, sh = k.split("/", 1)
    tb = A / "chain" / ser / sh / "textboxes_gpt-6.1-terra.json"
    if not v["fit"] or ser == "2000-digital-zoning" or not tb.exists():
        continue
    jobs.append((-len(json.load(open(tb))), ser, sh))
for _, ser, sh in sorted(jobs):
    print(ser, sh)
EOF
echo "START $(wc -l < $A/run/chain2_jobs.txt) sheets $(date +%H:%M:%S)"
cat $A/run/chain2_jobs.txt | xargs -P ${CHAIN_P:-2} -L 1 zsh $A/chain2_1.sh
echo "ALL done $(date +%H:%M:%S)"
sysctl vm.swapusage
