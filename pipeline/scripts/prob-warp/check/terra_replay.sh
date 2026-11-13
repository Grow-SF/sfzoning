#!/bin/zsh
# terra.sh's two per-sheet outputs, at $0: Terra's text boxes replayed from the answer cache only
# (run_as_ran.py --cache-only: a tile not in the cache stops the sheet, no call is made), and the map frame's bottom rule
# (framedetect.py, image only).  Written to $PW_CHAIN/SERIES/SHEET, as sheetAY.sh expects.  If REF is set, each
# sheet's boxes are compared with REF/SHEET/textboxes_gpt-6.1-terra.json.
# usage: [JOBS=4] [REF=DIR] terra_replay.sh SERIES [SHEET ...]   (no sheets: every fitted sheet of the series)
CK=${0:A:h}                            # this folder (inside a function $0 is the function's name)
source $CK/rebuild_env.sh
export SERIES=$1; shift
cd $PW_PIPELINE
WK=$PW_PIPELINE/data/work/$SERIES
SH=($@)
if [ ${#SH} -eq 0 ]; then
  for d in $WK/*(/); do
    [ -f $d/transform.json ] && [ -f $d/warped.tif ] && [ -f $d/blocks_px.geojson ] && [ -f $d/refine.json ] && SH+=(${d:t})
  done
fi
one() {
  local sh=$1 D=$PW_CHAIN/$SERIES/$1 WH msg
  mkdir -p $D
  env -u SFZMAPS_DATA_DIR -u SFZ_CORR -u PYTHONPATH .venv/bin/python $PW_CODE/framedetect.py $sh $D > $D/framedetect.log 2>&1 \
    || echo "FRAME FAILED $sh"
  WH=$(.venv/bin/python -c "import cv2; a = cv2.imread('$WK/$sh/page.png', cv2.IMREAD_GRAYSCALE); print(f'{a.shape[1]},{a.shape[0]}')")
  if env -u OPENAI_API_KEY -u SFZMAPS_DATA_DIR -u SFZ_CORR -u PYTHONPATH .venv/bin/python $CK/run_as_ran.py --cache-only $PW_CODE/textboxes.py \
      gpt-6.1-terra $SERIES $sh 0,0,$WH $D > $D/terra.log 2>&1; then
    msg="$(tail -1 $D/terra.log)"
    if [ -n "$REF" ] && [ -f $REF/$sh/textboxes_gpt-6.1-terra.json ]; then
      cmp -s $D/textboxes_gpt-6.1-terra.json $REF/$sh/textboxes_gpt-6.1-terra.json && msg="$msg | same as REF" || msg="$msg | DIFFERS from REF"
    fi
    echo "TERRA $sh: $msg"
  else
    echo "TERRA $sh FAILED: $(grep -m1 -E 'LLMCacheMiss|Error' $D/terra.log)"
  fi
}
i=0
for sh in $SH; do                      # JOBS sheets at a time
  one $sh &
  i=$((i + 1))
  [ $((i % ${JOBS:-4})) -eq 0 ] && wait
done
wait
