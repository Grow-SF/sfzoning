#!/bin/zsh
# Record the reads the sheet warp (PR #58) is built from, cache-only (nothing is paid), one record folder each under
# OUT (default pipeline/data/saved-reads/calls):
#   terra-boxes      Terra's text boxes, whole page, on every sheet the years chain read (textboxes.py, as terra.sh ran
#                    it: sheets with transform.json, warped.tif, blocks_px.geojson and refine.json)
#   codes2000        2000's code list: Terra and Sol (high effort) on every use code (leaders2.py map), then on every
#                    code-check case (reread.py ... all); codecheck_build.py in between, as check/codes2000.sh runs them
# The scripts are taken from the prob-warp branch as committed (each run records its git blob id).
# usage: [SR_PIPELINE=DIR] record_warp.sh [OUT]
set -e
P=${SR_PIPELINE:-${0:A:h:h:h}}                                # the pipeline folder (data, venv, git)
PKG=${0:A:h:h}                                                # the folder holding this package
OUT=${1:-$P/data/saved-reads/calls}
SRC=$P/data/saved-reads/scripts/prob-warp
W=$P/data/saved-reads/work
mkdir -p $OUT $SRC $W
cd $P
for f in allyears/textboxes.py codes2000/leaders2.py codes2000/leaders2_v2.txt codes2000/codecheck_build.py codes2000/reread.py; do
  git show prob-warp:pipeline/scripts/prob-warp/$f > $SRC/${f:t}
done
RUN=(env -u SFZMAPS_DATA_DIR -u SFZ_CORR -u PYTHONPATH -u OPENAI_API_KEY PYTHONPATH=$PKG .venv/bin/python -m saved_reads run)

echo "terra-boxes $(date +%H:%M:%S)"
one() {
  local s=$1 sh=$2 WH
  WH=$(.venv/bin/python -c "import cv2; a = cv2.imread('data/work/$s/$sh/page.png', cv2.IMREAD_GRAYSCALE); print(f'{a.shape[1]},{a.shape[0]}')")
  SERIES=$s $RUN --out $OUT/terra-boxes-$s $SRC/textboxes.py gpt-6.1-terra $s $sh 0,0,$WH $W/terra-boxes/$s/$sh > $W/terra-boxes-$sh.log 2>&1 \
    || echo "  FAILED $sh: $(grep -m1 -E 'LLMCacheMiss|Error' $W/terra-boxes-$sh.log)"
}
for s in 1960 1964 1968 1972 1975 1980 1980-1990-zoning 1990-1996-zoning 2000-digital-zoning; do
  for d in data/work/$s/*(/); do
    [ -f $d/transform.json ] && [ -f $d/warped.tif ] && [ -f $d/blocks_px.geojson ] && [ -f $d/refine.json ] || continue
    one $s ${d:t}
  done
  echo "  $s: $(wc -l < $OUT/terra-boxes-$s/calls.jsonl) lookups, $(grep -c '"hit": false' $OUT/terra-boxes-$s/calls.jsonl) misses"
done

echo "codes2000 $(date +%H:%M:%S)"
C=$W/codes2000
rm -rf $C && mkdir -p $C/leaders2
THREADS=12 $RUN --out $OUT/codes2000-terra --scratch $C $SRC/leaders2.py map gpt-6.1-terra $SRC/leaders2_v2.txt high > $C/leaders3_terra.log 2>&1
THREADS=12 $RUN --out $OUT/codes2000-sol --scratch $C $SRC/leaders2.py map gpt-6.1-sol $SRC/leaders2_v2.txt high > $C/leaders3_sol.log 2>&1
env -u SFZMAPS_DATA_DIR PYTHONPATH=$PKG .venv/bin/python -m saved_reads run --out $OUT/codes2000-build --scratch $C $SRC/codecheck_build.py > /dev/null
$RUN --out $OUT/codes2000-terra --scratch $C $SRC/reread.py gpt-6.1-terra high all > $C/reread_terra.log 2>&1
$RUN --out $OUT/codes2000-sol --scratch $C $SRC/reread.py gpt-6.1-sol high all > $C/reread_sol.log 2>&1
for r in codes2000-terra codes2000-sol; do
  echo "  $r: $(wc -l < $OUT/$r/calls.jsonl) lookups, $(grep -c '"hit": false' $OUT/$r/calls.jsonl) misses"
done
echo "done $(date +%H:%M:%S)"
