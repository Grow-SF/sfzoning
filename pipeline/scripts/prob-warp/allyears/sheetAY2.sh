#!/bin/zsh
# One sheet of any series up to its cleaned page, on its chosen fit (data_ay2 mirror) with every Terra box's letters
# fitted and erased (as run k8 for 2000; the earlier years chain erased only the codes): raw graph, block outlines
# deleted, walls, all text fitted + erased.  The graph is built afterwards on the series' merged page.
# Terra boxes and the legend cut line from the earlier chain (chain/SERIES/SHEET).  No API key: no paid call.
# usage: sheetAY2.sh SERIES SHEET   -> chain2/SERIES/SHEET, chain2/SERIES/SHEET_c1 (clean.png, fits3.json)
set -e
export SERIES=$1 SHEET=$2
S=/private/tmp/claude-501/-Users-sbuss-workspace-sfzoning/eb18d2fc-e22a-48a6-afbc-57c31cd3e134/scratchpad
A=$S/allyears
export SFZ_CORR=1 PYTHONPATH=$S/corrpatch SFZMAPS_DATA_DIR=$S/data_ay2
PY=/Users/sbuss/workspace/sfzoning/.claude/worktrees/cache-maybe/pipeline/.venv/bin/python
OLD=$A/chain/$SERIES/$SHEET
D=$A/chain2/$SERIES/$SHEET
G=$D/raw
DC=$A/chain2/$SERIES/${SHEET}_c1
PAGE=$SFZMAPS_DATA_DIR/work/$SERIES/$SHEET/page.png
mkdir -p $D $G $DC
cd /Users/sbuss/workspace/sfzoning/.claude/worktrees/cache-maybe/pipeline
stage() { echo "STAGE $SERIES $SHEET $1 $(date +%H:%M:%S)"; }
[ -f $OLD/textboxes_gpt-6.1-terra.json ] || { echo "FAILED $SERIES $SHEET: no Terra boxes"; exit 1; }
[ -f $OLD/frame_bottom.json ] || { echo "FAILED $SERIES $SHEET: no frame line"; exit 1; }
cp $OLD/textboxes_gpt-6.1-terra.json $OLD/frame_bottom.json $D/
WH=$($PY -c "import cv2; a = cv2.imread('$PAGE', cv2.IMREAD_GRAYSCALE); print(f'{a.shape[1]},{a.shape[0]}')")
echo '{}' > $D/stats.json
if [ ! -f $G/zgraph.json ]; then
  stage raw-graph
  cp $PAGE $G/clean.png
  echo '[]' > $G/fits3.json
  $PY $A/vectorize.py $SERIES $SHEET 0,0 $G > /dev/null
  BLOCKS=1 $PY $A/zonegraph.py $SERIES $G > /dev/null
fi
if [ ! -f $D/noblock.png ]; then
  stage block-outlines
  $PY $A/blocklines.py $SERIES $SHEET $G $D > /dev/null
fi
if [ ! -f $D/walls.json ]; then
  stage walls
  WALLS_OUT=$D/walls.json $PY $A/sidealign.py $D > /dev/null
fi
if [ ! -f $DC/fits3.json ]; then
  stage fit-erase-all-text
  cp $D/textboxes_gpt-6.1-terra.json $DC/
  echo '{}' > $DC/stats.json
  CODES=$A/codes_$SERIES.json FITWORD_WHITE_UNIT=grow FITWORD_WALL_INSIDE=ignore SHEETFILES=$D PAGE=$D/noblock.png NPROC=${NPROC:-2} \
    $PY -u $A/fitboxes.py $SERIES $SHEET 0,0,$WH $DC 2>&1 | grep --line-buffered -E "^PROGRESS|Traceback|Error" \
    | sed -u "s/^PROGRESS \([0-9]*\)\/\([0-9]*\)/PROGRESS \1\/\2 $SHEET/" | awk 'NR % 200 == 1 || /Traceback|Error/'
  [ -f $DC/fits3.json ] || { echo "FAILED $SERIES $SHEET fit-erase"; exit 1; }
fi
stage done
