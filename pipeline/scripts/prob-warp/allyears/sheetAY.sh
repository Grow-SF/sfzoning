#!/bin/zsh
# One sheet of any series through run k3's per-sheet chain (sheetK.sh), with its corrected fit (the patch + the
# data_ay mirror): raw graph, block outlines deleted, walls, codes fitted + erased, graph, graph2, labels.
# Terra boxes and the legend cut line come from terra.sh.  No API key is given: nothing here can make a paid call.
# usage: sheetAY.sh SERIES SHEET
set -e
export SERIES=$1 SHEET=$2
S=/private/tmp/claude-501/-Users-sbuss-workspace-sfzoning/eb18d2fc-e22a-48a6-afbc-57c31cd3e134/scratchpad
A=$S/allyears
export SFZ_CORR=1 PYTHONPATH=$S/corrpatch SFZMAPS_DATA_DIR=$S/data_ay
PY=/Users/sbuss/workspace/sfzoning/.claude/worktrees/cache-maybe/pipeline/.venv/bin/python
D=$A/chain/$SERIES/$SHEET
G=$D/raw
DC=$A/chain/$SERIES/${SHEET}_c1
G2=$DC/graph2
PAGE=$SFZMAPS_DATA_DIR/work/$SERIES/$SHEET/page.png
mkdir -p $D $G $DC/graph $G2
cd /Users/sbuss/workspace/sfzoning/.claude/worktrees/cache-maybe/pipeline
stage() { echo "STAGE $SHEET $1 $(date +%H:%M:%S)"; }
[ -f $D/textboxes_gpt-6.1-terra.json ] || { echo "FAILED $SHEET: no Terra boxes"; exit 1; }
[ -f $D/frame_bottom.json ] || { echo "FAILED $SHEET: no frame line"; exit 1; }
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
  stage fit-erase
  cp $D/textboxes_gpt-6.1-terra.json $DC/
  echo '{}' > $DC/stats.json
  CODES=$A/codes_$SERIES.json CODES_ONLY=1 FITWORD_WHITE_UNIT=grow FITWORD_WALL_INSIDE=ignore SHEETFILES=$D PAGE=$D/noblock.png NPROC=${NPROC:-4} \
    $PY -u $A/fitboxes.py $SERIES $SHEET 0,0,$WH $DC 2>&1 | grep --line-buffered -E "^PROGRESS|Traceback|Error" \
    | sed -u "s/^PROGRESS \([0-9]*\)\/\([0-9]*\)/PROGRESS \1\/\2 $SHEET/" | awk 'NR % 200 == 1 || /Traceback|Error/'
  [ -f $DC/fits3.json ] || { echo "FAILED $SHEET fit-erase"; exit 1; }
fi
if [ ! -f $DC/graph/zgraph.json ]; then
  stage graph
  cp $DC/clean.png $DC/graph/clean.png
  echo '[]' > $DC/graph/fits3.json
  $PY $A/vectorize.py $SERIES $SHEET 0,0 $DC/graph > /dev/null
  TEXTBOXES=$DC/textboxes_gpt-6.1-terra.json $PY $A/zonegraph.py $SERIES $DC/graph > /dev/null
fi
if [ ! -f $G2/zgraph.json ]; then
  stage graph2
  cp $DC/graph/graph.json $DC/graph/clean.png $G2/
  SPLIT=$($PY -c "import json; print(json.load(open('$G/zgraph.json'))['split_ft'])")
  THIN=$($PY -c "import json; print(json.load(open('$G/zgraph.json'))['thin_ft'])")
  TEXTBOXES=$DC/textboxes_gpt-6.1-terra.json SPLIT_FT=$SPLIT THIN_FT=$THIN $PY $A/zonegraph.py $SERIES $G2 > /dev/null
  $PY $A/makelabels.py $DC $SHEET > /dev/null
  cp $DC/graph/labels.json $G2/labels.json
fi
stage done
