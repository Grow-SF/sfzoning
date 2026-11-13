#!/bin/zsh
# one sheet's Terra text boxes (whole page, paid unless cached, $4 cap) and its map-frame line.  As terra.sh's one().
# A lock (D/.terra.lock, made atomically) keeps two readers off the same sheet: the second waits for the first and
# then finds its boxes.  usage: terra1.sh SERIES SHEET
A=/private/tmp/claude-501/-Users-sbuss-workspace-sfzoning/eb18d2fc-e22a-48a6-afbc-57c31cd3e134/scratchpad/allyears
W=/Users/sbuss/workspace/sfzoning/.claude/worktrees/cache-maybe/pipeline/data/work/$1
export SERIES=$1
cd /Users/sbuss/workspace/sfzoning/.claude/worktrees/cache-maybe/pipeline
D=$A/chain/$1/$2
mkdir -p $D
until mkdir $D/.terra.lock 2>/dev/null; do sleep 5; done
trap "rmdir $D/.terra.lock 2>/dev/null" EXIT
[ -f $D/frame_bottom.json ] || env -u SFZMAPS_DATA_DIR -u SFZ_CORR -u PYTHONPATH .venv/bin/python $A/framedetect.py $2 $D > /dev/null || exit 1
if [ ! -f $D/textboxes_gpt-6.1-terra.json ]; then
  WH=$(.venv/bin/python -c "import cv2; a = cv2.imread('$W/$2/page.png', cv2.IMREAD_GRAYSCALE); print(f'{a.shape[1]},{a.shape[0]}')")
  env -u SFZMAPS_DATA_DIR -u SFZ_CORR -u PYTHONPATH SFZMAPS_MAX_USD=4 uv run --env-file /Users/sbuss/workspace/sfzoning/pipeline/.env --project . \
    python $A/textboxes.py gpt-6.1-terra $1 $2 0,0,$WH $D > $D/terra.log 2>&1 || { tail -3 $D/terra.log; exit 1; }
  echo "TERRA $2 $(date +%H:%M:%S): $(tail -1 $D/terra.log)"
fi
