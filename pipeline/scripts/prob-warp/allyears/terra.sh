#!/bin/zsh
# Terra text boxes for every fitted sheet of a series (whole page), 3 sheets at a time, $4 cap per sheet; and the
# map frame's bottom rule (legend cut).  Both read the page only (no fit).  usage: terra.sh SERIES
A=/private/tmp/claude-501/-Users-sbuss-workspace-sfzoning/eb18d2fc-e22a-48a6-afbc-57c31cd3e134/scratchpad/allyears
W=/Users/sbuss/workspace/sfzoning/.claude/worktrees/cache-maybe/pipeline/data/work/$1
export SERIES=$1
cd /Users/sbuss/workspace/sfzoning/.claude/worktrees/cache-maybe/pipeline
SH=()
for d in $W/*(/); do
  [ -f $d/transform.json ] && [ -f $d/warped.tif ] && [ -f $d/blocks_px.geojson ] && [ -f $d/refine.json ] && SH+=(${d:t})
done
echo "TERRA $1: ${#SH} sheets, start $(date +%H:%M:%S)"
one() {
  local sh=$1 D=$A/chain/$SERIES/$1
  mkdir -p $D
  [ -f $D/frame_bottom.json ] || .venv/bin/python $A/framedetect.py $sh $D > /dev/null 2>&1
  if [ ! -f $D/textboxes_gpt-6.1-terra.json ]; then
    local WH=$(.venv/bin/python -c "import cv2; a = cv2.imread('$W/$sh/page.png', cv2.IMREAD_GRAYSCALE); print(f'{a.shape[1]},{a.shape[0]}')")
    SFZMAPS_MAX_USD=4 uv run --env-file /Users/sbuss/workspace/sfzoning/pipeline/.env --project . python $A/textboxes.py gpt-6.1-terra \
      $SERIES $sh 0,0,$WH $D > $D/terra.log 2>&1
    echo "TERRA $sh $(date +%H:%M:%S): $(tail -1 $D/terra.log)"
  fi
}
i=0
for sh in $SH; do
  one $sh &
  i=$((i + 1))
  if [ $((i % 3)) -eq 0 ]; then wait; fi
done
wait
echo "TERRA $1 done $(date +%H:%M:%S)"
