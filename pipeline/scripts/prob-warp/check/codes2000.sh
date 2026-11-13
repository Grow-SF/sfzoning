#!/bin/zsh
# 2000's person-checked code list (codes_final.json), rebuilt at $0 with the scripts that made it (codes2000/, run
# unchanged through run_as_ran.py, the R&D scratch folder replaced by $W):
#   1. leaders2.py map: Terra and Sol (high effort, prompt leaders2_v2.txt) read every use code the pipeline found
#      on each sheet, answered from the cache only -> W/leaders3 (maprun.sh is how it ran)
#   2. codecheck_build.py: the code-check page's cases (models' disagreements first, the rest shuffled, seed 7)
#      -> W/codecheck/cases.json, checked against the page as published (answers/codecheck_cases_published.json)
#   3. reread.py: Terra and Sol (high) re-read every case, cache only -> W/reread_*_high_all.json
#   4. the owner's answers on the code-check page and the leader-check page (answers/, from the pages' stores; all
#      given 2026-11-05 before 17:41 UTC, when merge_codes.py last ran before the years chain copied its output)
#   5. merge_codes.py -> W/codes_final.json
# usage: codes2000.sh      (W = rebuild-2026-11-13/codes2000)
set -e
CK=${0:A:h}
source $CK/rebuild_env.sh
C=$CK/../codes2000
W=$PW_DATA/rebuild-2026-11-13/codes2000
mkdir -p $W/leaders2 $W/truth/labels $W/truth2/reads $W/labeler
cd $PW_PIPELINE
PY=(env -u SFZMAPS_DATA_DIR -u SFZ_CORR -u PYTHONPATH -u OPENAI_API_KEY .venv/bin/python $CK/run_as_ran.py)
echo "1. leaders3 $(date +%H:%M:%S)"
THREADS=12 $PY --cache-only --scratch $W $C/leaders2.py map gpt-6.1-terra $C/leaders2_v2.txt high > $W/leaders3_terra.log 2>&1 &
THREADS=12 $PY --cache-only --scratch $W $C/leaders2.py map gpt-6.1-sol $C/leaders2_v2.txt high > $W/leaders3_sol.log 2>&1 &
wait
tail -n 1 $W/leaders3_terra.log $W/leaders3_sol.log
echo "2. code-check cases $(date +%H:%M:%S)"
$PY --scratch $W $C/codecheck_build.py
.venv/bin/python - $W/codecheck/cases.json $C/answers/codecheck_cases_published.json <<'EOF'
import json, sys
new = {c["id"]: c for c in json.load(open(sys.argv[1]))}
pub = {c["id"]: c for c in json.load(open(sys.argv[2]))}
key = lambda c: (c["sid"], c["code"], c["x"], c["y"])
same = sum(i in new and key(new[i]) == key(pub[i]) for i in pub)
print(f"   cases: {len(new)} rebuilt, {len(pub)} published, {same} with the same id, sheet, code and place")
EOF
echo "3. rereads $(date +%H:%M:%S)"
$PY --cache-only --scratch $W $C/reread.py gpt-6.1-terra high all > $W/reread_terra.log 2>&1 &
$PY --cache-only --scratch $W $C/reread.py gpt-6.1-sol high all > $W/reread_sol.log 2>&1 &
wait
tail -n 1 $W/reread_terra.log $W/reread_sol.log
echo "4. answers"
.venv/bin/python - $C/answers $W <<'EOF'
import json, sys
from pathlib import Path
A, W = Path(sys.argv[1]), Path(sys.argv[2])
for name, sub in [("labeler_answers.json", "truth/labels"), ("codecheck_answers.json", "truth2/reads")]:
    for k, v in json.load(open(A / name)).items():
        (W / sub / f"{k}.json").write_text(json.dumps(v))
(W / "labeler/cases.json").write_text((A / "labeler_cases_published.json").read_text())
print("   answers laid out")
EOF
echo "5. merge $(date +%H:%M:%S)"
$PY --scratch $W $C/merge_codes.py
