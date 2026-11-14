#!/bin/zsh
# Does the warper need to know about text?  prob6 on a series' sheets twice: as approved (the codes the years chain
# fitted and erased are taken out of the ink it fits: PW_CHAIN's noblock.png / clean.png), and with nothing taken out
# (an empty PW_CHAIN: ay_regprep3 then reports text_out none).  Same p1 inputs, same settings; only the text differs.
# The "with" run is $PW_RUN (rebuild_series.sh made it, identical to the approved run); "without" goes to
# rebuild-2026-11-13/run-notext.  Then both are summed up with the same measures (ay_probsum: block-line points within
# 1 / 2 px of a printed same-way line, control points' miss, smallest cell determinant).
# usage: [JOBS=4] text_ablation.sh SERIES [SHEET ...]
CK=${0:A:h}
source $CK/rebuild_env.sh
SER=$1; shift
WITH=$PW_RUN
WITHOUT=$PW_DATA/rebuild-2026-11-13/run-notext
EMPTY=$PW_DATA/rebuild-2026-11-13/no-chain-pages
mkdir -p $WITHOUT/$SER $EMPTY
SH=($@)
[ ${#SH} -eq 0 ] && SH=($WITH/$SER/*(/:t))
one() {
  local sh=$1 d=$WITHOUT/$SER/$1
  rm -rf $d && /bin/cp -cR $WITH/$SER/$sh $d                          # the p1 inputs (copy-on-write clone)
  rm -f $d/confit_${sh}_prob6.npz $d/dm_${sh}_prob6.npz $d/dm_${sh}_prob6.png $d/prob.log
  PW_RUN=$WITHOUT PW_CHAIN=$EMPTY THREADS=3 sh $PW_CODE/prob.sh $SER $sh > $d/ablation.log 2>&1 \
    && echo "  $sh: $(grep -o '"text_out": "[^"]*"' $d/prob.log)" || echo "  FAILED $sh: $(tail -n 2 $d/ablation.log)"
}
echo "without text removal: ${#SH} sheets $(date +%H:%M:%S)"
i=0
for sh in $SH; do
  one $sh &
  i=$((i + 1))
  [ $((i % ${JOBS:-4})) -eq 0 ] && wait
done
wait
$PW_PIPELINE/.venv/bin/python - $WITH/probsum.jsonl $WITHOUT/probsum.jsonl $SER <<'EOF'
import json, sys
def rows(f):
    out = {}
    for line in open(f):
        r = json.loads(line)
        if r["series"] == sys.argv[3]:
            out[r["sheet"]] = r                                   # the last summary of each sheet
    return out
a, b = rows(sys.argv[1]), rows(sys.argv[2])
keys = [("within1_prob6", "within 1 px"), ("within2_prob6", "within 2 px"), ("gcp_miss_px_p99_prob6", "ctrl miss p99"),
        ("min_det_prob6", "min det")]
print(f"{'sheet':26s} " + "  ".join(f"{n:>13s} with/without" for _, n in keys))
for sh in sorted(set(a) & set(b)):
    print(f"{sh:26s} " + "  ".join(f"{a[sh][k]:>13} / {b[sh][k]:<7}" for k, _ in keys))
for k, n in keys:
    sa = [a[s][k] for s in sorted(set(a) & set(b))]
    sb = [b[s][k] for s in sorted(set(a) & set(b))]
    print(f"mean {n}: with {sum(sa) / len(sa):.3f}, without {sum(sb) / len(sb):.3f}")
EOF
echo "done $(date +%H:%M:%S)"
