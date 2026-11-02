#!/usr/bin/env bash
# Give a git worktree the pipeline data its commit names, from the main checkout, at no extra
# disk (copy-on-write clones on APFS):
#   - data/raw, data/ref and data/cache are links to the main checkout's (inputs that every
#     checkout shares; builds never write raw, and cache answers are immutable);
#   - DVC uses the main checkout's cache, and `dvc checkout` puts this worktree's
#     data/work and data/out in place exactly as its dvc.lock records them.
# Never link data/work or data/out: a worktree with another checkout's work/ or out/ is how a
# stale publish happened (2026-11-01).
#
#   pipeline/scripts/worktree-data.sh /path/to/main/checkout
set -euo pipefail
main=$(cd "${1:?usage: worktree-data.sh MAIN_CHECKOUT}" && pwd)
here=$(git rev-parse --show-toplevel)
[ "$main" != "$here" ] || { echo "run this in a worktree, not the main checkout" >&2; exit 1; }
dvc() { (cd "$here/pipeline" && uv tool run --from dvc==3.67.1 dvc "$@"); }

mkdir -p "$here/pipeline/data"
for d in raw ref cache; do
  t="$here/pipeline/data/$d"
  if [ -e "$t" ] && [ ! -L "$t" ]; then
    echo "pipeline/data/$d exists and is not a link; leaving it" >&2
  else
    ln -sfn "$main/pipeline/data/$d" "$t"
  fi
done
for d in work out; do
  if [ -L "$here/pipeline/data/$d" ]; then
    echo "pipeline/data/$d is a link to another checkout; replacing it with this commit's data" >&2
    rm "$here/pipeline/data/$d"
  fi
done
dvc cache dir --local "$main/.dvc/cache"
dvc checkout
dvc status
