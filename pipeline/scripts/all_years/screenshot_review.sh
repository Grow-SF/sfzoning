#!/usr/bin/env bash
# Screenshot the review page (served locally for a minute) at phone and desktop widths: OUT_DIR/review-390.png (the
# first series, light) and OUT_DIR/review-1280.png (series HASH, dark).
#
#   scripts/all_years/screenshot_review.sh OUT_DIR [HASH]
set -u
out=${1:?usage: screenshot_review.sh OUT_DIR [HASH]}
hash=${2:-}
here=$(cd "$(dirname "$0")/../.." && pwd)
cd "$here/data/solver/experiments/all-years/review" || exit 1
timeout 90 python3 -m http.server 8765 --bind 127.0.0.1 > /dev/null 2>&1 &
sleep 1
chrome="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
"$chrome" --headless=new --disable-gpu --hide-scrollbars --virtual-time-budget=8000 --window-size=390,1500 \
  --screenshot="$out/review-390.png" "http://127.0.0.1:8765/index.html" > /dev/null 2>&1
"$chrome" --headless=new --disable-gpu --hide-scrollbars --virtual-time-budget=8000 --window-size=1280,1100 \
  --force-dark-mode --screenshot="$out/review-1280.png" "http://127.0.0.1:8765/index.html#$hash" > /dev/null 2>&1
ls -la "$out"/review-*.png
