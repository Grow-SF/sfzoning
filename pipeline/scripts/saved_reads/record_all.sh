#!/bin/zsh
# Record every saved read the later steps (sheet warp, joint solver, text detector) are made from, cache-only: nothing is
# paid (no key, a client that cannot call out; a missing answer stops that script).  One record folder per read under
# OUT (default pipeline/data/saved-reads/calls).  Each script runs as it ran, with the arguments and settings of its
# paid run (from the 2026-11 session's commands):
#   terra-boxes-SERIES  Terra, whole-page text boxes, every sheet the years chain read     sources/textboxes.py
#   codes2000-terra/sol 2000 code list: leader map of every use code, then a re-read of     sources/leaders2.py,
#                       every code-check case (Terra and Sol, high effort)                 reread.py
#   leaders-sol         Sol, leader read of every 2000 use code, medium and low effort     sources/leaders.py
#   fixed-marks         the code and leader reads asked again with their marks red          rerun/*.py
#   k8-*                2000 second passes and third reads, as run in k8 with pay.sh       k8/*.py (the scratch copy)
#                       (SFZ_CORR mirror data_ay2): textpass.py cw; placeread.py; placeask.py; placeask.py SPLIT=1;
#                       distask.py; read3.py; read3r.py
#   pipeline-labels     Sol, the pipeline's district label reads, every series             districts/labels.py
# sources/ holds the scripts of the first three exactly as they ran (each record's script.git_blob is the file's hash);
# SR_SOURCE names where each ran.  The code-check cases come from inputs/code-check-2000.
# The k8 scripts write into their scratch folder, so they run on a copy-on-write clone of the rescued one (with k6:
# k8/p06 and k8/p07 are links into run k6).
# usage: [SR_PIPELINE=DIR] [ONLY="terra-boxes codes2000 leaders-sol k8 pipeline-labels fixed-marks"] record_all.sh [OUT]
P=${SR_PIPELINE:-${0:A:h:h:h}}                                # the pipeline folder (data, venv, git)
PKG=${0:A:h:h}                                                # the folder holding this package
OUT=${1:-$P/data/saved-reads/calls}
SRC=$PKG/saved_reads/sources                                  # the reading scripts, as they ran
CASES=${PKG:h}/inputs/code-check-2000/codecheck_cases_published.json
W=$P/data/saved-reads/work
RESCUED=$P/data/rnd/scratch-2026-11-13
mkdir -p $OUT $W
cd $P
PW="prob-warp@99f85e7:pipeline/scripts/prob-warp"             # where each ran (branch@commit:folder)
ZL="zoning-lines-rnd@8ec7b40:pipeline/scripts/zoning-lines"
RUN=(env -u SFZMAPS_DATA_DIR -u SFZ_CORR -u PYTHONPATH -u OPENAI_API_KEY PYTHONPATH=$PKG .venv/bin/python -W ignore -m saved_reads run)
# the package's answers into the answer cache first: a fresh clone's cache has none of them (sfzmaps pull brings only
# the answers the build's outputs were made from).  Answers already there are left as they are.
env -u SFZMAPS_DATA_DIR -u SFZ_CORR -u PYTHONPATH PYTHONPATH=$PKG .venv/bin/python -m saved_reads restore-cache \
  --from ${PKG:h}/saved-reads
want() { [ -z "$ONLY" ] || [[ " $ONLY " == *" $1 "* ]]; }
count() { echo "  $1: $(wc -l < $OUT/$1/calls.jsonl | tr -d ' ') lookups, $(grep -c '"hit": false' $OUT/$1/calls.jsonl) not in the cache"; }

if want terra-boxes; then
echo "terra-boxes $(date +%H:%M:%S)"
for s in 1960 1964 1968 1972 1975 1980 1980-1990-zoning 1990-1996-zoning 2000-digital-zoning; do
  for d in data/work/$s/*(/); do
    [ -f $d/transform.json ] && [ -f $d/warped.tif ] && [ -f $d/blocks_px.geojson ] && [ -f $d/refine.json ] || continue
    sh=${d:t}
    WH=$(.venv/bin/python -c "import cv2; a = cv2.imread('$d/page.png', cv2.IMREAD_GRAYSCALE); print(f'{a.shape[1]},{a.shape[0]}')")
    SR_SOURCE=$PW/allyears/textboxes.py SERIES=$s $RUN --out $OUT/terra-boxes-$s $SRC/textboxes.py gpt-6.1-terra $s $sh 0,0,$WH $W/terra-boxes/$s/$sh \
      > $W/terra-boxes-$sh.log 2>&1 || echo "  FAILED $sh: $(grep -m1 -E 'LLMCacheMiss|Error' $W/terra-boxes-$sh.log)"
  done
  count terra-boxes-$s
done

fi
if want codes2000; then
echo "codes2000 $(date +%H:%M:%S)"
C=$W/codes2000
rm -rf $C && mkdir -p $C/leaders2
SR_SOURCE=$PW/codes2000/leaders2.py THREADS=12 $RUN --out $OUT/codes2000-terra --scratch $C $SRC/leaders2.py map gpt-6.1-terra $SRC/leaders2_v2.txt high > $C/map_terra.log 2>&1
SR_SOURCE=$PW/codes2000/leaders2.py THREADS=12 $RUN --out $OUT/codes2000-sol --scratch $C $SRC/leaders2.py map gpt-6.1-sol $SRC/leaders2_v2.txt high > $C/map_sol.log 2>&1
$RUN --out $W/codes2000-build --scratch $C $SRC/codecheck_build.py > $C/build.log 2>&1
SR_SOURCE=$PW/codes2000/reread.py $RUN --out $OUT/codes2000-terra --scratch $C $SRC/reread.py gpt-6.1-terra high all > $C/reread_terra.log 2>&1
SR_SOURCE=$PW/codes2000/reread.py $RUN --out $OUT/codes2000-sol --scratch $C $SRC/reread.py gpt-6.1-sol high all > $C/reread_sol.log 2>&1
count codes2000-terra; count codes2000-sol

fi
if want leaders-sol; then
echo "leaders-sol $(date +%H:%M:%S)"
L=$W/leaders
rm -rf $L && mkdir -p $L/leaders
for e in medium low; do
  for n in 01 02 03 04 05 06 07 08 09 10 11 12 13; do
    SR_SOURCE=$ZL/leaders.py $RUN --out $OUT/leaders-sol --scratch $L $SRC/leaders.py sheet 2000-digital-zoning-p$n $e > $L/p$n-$e.log 2>&1 \
      || echo "  FAILED p$n $e: $(grep -m1 -E 'LLMCacheMiss|Error' $L/p$n-$e.log)"
  done
done
count leaders-sol

fi
if want k8; then
echo "k8 $(date +%H:%M:%S)"
WS=$W/scratch
if [ ! -d $WS ]; then
  mkdir -p $WS && for f in k8 k6 zon2000 corrpatch data_ay2; do /bin/cp -cR $RESCUED/$f $WS/; done   # macOS clones: no extra disk
fi
K8=(env -u OPENAI_API_KEY SFZ_CORR=1 PYTHONPATH=$PKG:$WS/corrpatch:$WS/k8 SFZMAPS_DATA_DIR=$WS/data_ay2 .venv/bin/python -W ignore -m saved_reads run --scratch $WS)
k8() { local name=$1; shift; SR_SOURCE="rescued scratch copy: pipeline/data/rnd/scratch-2026-11-13/k8/$1" $K8 --out $OUT/k8-$name $WS/k8/$1 ${@:2} > $W/k8-$name.log 2>&1 || echo "  FAILED k8 $name: $(tail -1 $W/k8-$name.log)"; count k8-$name; }
k8 textpass-cw textpass.py cw
k8 placeread placeread.py c_cwb_pfitf1.npz 40
k8 placeask placeask.py c_cwb_pfitf1.npz 0 236
SPLIT=1 k8 placeask-split placeask.py c_cwq_pfitf1.npz 0 2000
k8 distask distask.py cws
k8 read3 read3.py
k8 read3r read3r.py

fi
if want pipeline-labels; then
echo "pipeline-labels $(date +%H:%M:%S)"
SR_SOURCE="saved-reads: pipeline/scripts/saved_reads/pipeline_labels.py; pipeline src/sfzmaps/districts/labels.py @ $(git rev-parse --short HEAD)" $RUN --out $OUT/pipeline-labels $PKG/saved_reads/pipeline_labels.py > $W/pipeline-labels.log 2>&1
grep -c MISS $W/pipeline-labels.log | sed 's/^/  sheets with a missing answer: /'
count pipeline-labels
fi
if want fixed-marks; then
# the code reads and leader reads asked again with their marks red (rerun/rerun.sh, paid), replayed from the cache
echo "fixed-marks $(date +%H:%M:%S)"
F=$W/fixed-marks
R=$PKG/saved_reads/rerun
rm -rf $F && mkdir -p $F/leaders2 $F/codecheck $F/leaders
cp $CASES $F/codecheck/cases.json
FX="saved-reads@0117477:pipeline/scripts/saved_reads/rerun"   # where the replay was first recorded
for m in terra sol; do
  SR_SOURCE=$FX/leaders2.py THREADS=12 $RUN --out $OUT/codes2000-$m-fixed --scratch $F $R/leaders2.py map gpt-6.1-$m $R/leaders2_v2.txt high \
    > $F/map_$m.log 2>&1 || echo "  FAILED leader map $m: $(grep -m1 -E 'LLMCacheMiss|Error' $F/map_$m.log)"
  SR_SOURCE=$FX/reread.py THREADS=12 $RUN --out $OUT/codes2000-$m-fixed --scratch $F $R/reread.py gpt-6.1-$m high all \
    > $F/reread_$m.log 2>&1 || echo "  FAILED re-check $m: $(grep -m1 -E 'LLMCacheMiss|Error' $F/reread_$m.log)"
  count codes2000-$m-fixed
done
for e in medium low; do
  for n in 01 02 03 04 05 06 07 08 09 10 11 12 13; do
    SR_SOURCE=$FX/leaders.py $RUN --out $OUT/leaders-sol-fixed --scratch $F $R/leaders.py sheet 2000-digital-zoning-p$n $e \
      > $F/leaders-p$n-$e.log 2>&1 || echo "  FAILED leader read p$n $e: $(grep -m1 -E 'LLMCacheMiss|Error' $F/leaders-p$n-$e.log)"
  done
done
count leaders-sol-fixed
fi
echo "done $(date +%H:%M:%S)"
