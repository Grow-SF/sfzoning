#!/bin/zsh
# Run k4, one sheet: run k3's chain from the letter fit on, with every Terra text box fitted and erased from the
# pixels (block numbers, street names, other, zoning codes: no CODES_ONLY), and the graph built without the
# drop-edges-inside-text-boxes rule (no TEXTBOXES).  Raw graph, block outlines, walls, Terra boxes and the legend
# line are run k3's own (stepview/sheetsK/pNN, read only).  Writes k4/pNN_c1 (clean.png, fits3.json, graph, graph2).
# No API key is given: nothing can make a paid call.
# usage: k4sheet.sh NN
set -e
NN=$1
SHEET=2000-digital-zoning-p$NN
S=/private/tmp/claude-501/-Users-sbuss-workspace-sfzoning/eb18d2fc-e22a-48a6-afbc-57c31cd3e134/scratchpad
export SFZ_CORR=1 PYTHONPATH=$S/corrpatch SFZMAPS_DATA_DIR=$S/data_k1
PY=/Users/sbuss/workspace/sfzoning/.claude/worktrees/cache-maybe/pipeline/.venv/bin/python
D=$S/stepview/sheetsK/p$NN
G=$D/raw
DC=$S/k4/p${NN}_c1
G2=$DC/graph2
mkdir -p $DC/graph $G2
cd /Users/sbuss/workspace/sfzoning/.claude/worktrees/cache-maybe/pipeline
stage() { echo "STAGE p$NN $1 $(date +%H:%M:%S)"; }
if [ ! -f $DC/fits3.json ]; then
  stage fit-erase-all-text
  cp $D/textboxes_gpt-6.1-terra.json $DC/
  echo '{}' > $DC/stats.json
  FITWORD_WHITE_UNIT=grow FITWORD_WALL_INSIDE=ignore SHEETFILES=$D PAGE=$D/noblock.png NPROC=${NPROC:-4} \
    $PY -u $S/fitboxes.py 2000-digital-zoning $SHEET 0,0,3300,5100 $DC 2>&1 | grep --line-buffered -E "^PROGRESS|Traceback|Error|^\{|^poor" \
    | sed -u "s/^PROGRESS \([0-9]*\)\/\([0-9]*\)/PROGRESS \1\/\2 p$NN/" | awk 'NR % 50 == 1 || /Traceback|Error|^\{|^poor/'
  [ -f $DC/fits3.json ] || { echo "FAILED p$NN fit-erase"; exit 1; }
fi
if [ ! -f $DC/graph/zgraph.json ]; then
  stage graph
  cp $DC/clean.png $DC/graph/clean.png
  echo '[]' > $DC/graph/fits3.json
  $PY $S/vectorize.py 2000-digital-zoning $SHEET 0,0 $DC/graph > /dev/null
  SHEET=$SHEET $PY $S/zonegraph.py 2000-digital-zoning $DC/graph > /dev/null
fi
if [ ! -f $G2/zgraph.json ]; then
  stage graph2
  cp $DC/graph/graph.json $DC/graph/clean.png $G2/
  SPLIT=$($PY -c "import json; print(json.load(open('$G/zgraph.json'))['split_ft'])")
  THIN=$($PY -c "import json; print(json.load(open('$G/zgraph.json'))['thin_ft'])")
  SHEET=$SHEET SPLIT_FT=$SPLIT THIN_FT=$THIN $PY $S/zonegraph.py 2000-digital-zoning $G2 > /dev/null
  $PY $S/makelabels.py $DC $SHEET > /dev/null
  cp $DC/graph/labels.json $G2/labels.json
fi
stage done
