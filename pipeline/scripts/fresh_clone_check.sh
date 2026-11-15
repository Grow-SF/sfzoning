#!/usr/bin/env bash
# The fresh-clone check for the 2000 solver's build stages (README, "The 2000 solver").
#
#   pipeline/scripts/fresh_clone_check.sh [--clone DIR] [--ref PIPELINE_DIR] [--dry-run] [STAGE...]
#
# Clones this branch from origin into DIR (default pipeline/data/fresh-clone, git-ignored; it must not exist or be
# empty), installs the pinned tools, pulls what is shipped, builds the warp of 2000-digital-zoning if the pull did not
# bring it, then every STAGE in order, and compares each stage's outputs with the checkout in --ref (default: the
# pipeline/ this script lives in). Each stage is held to its own epsilon (scripts/compare_outputs.py):
#   byte-identical     the warp (deterministic on the GPU), solver-page/-lots/-prints/-labels/-walls/-solve
#                      (an .npz by its arrays, a .parquet by its table)
#   solver-text-<q>    chance/*.png within 0.01 x 255 (2 counts); numbers in .json within 1%; model.pt not compared
#   solver-text-scores numbers within 1%
# Default STAGEs: solver-page solver-lots solver-prints solver-text-nw solver-labels solver-walls solver-solve.
# Only nw is retrained (it uses the GPU: run when it is free); text-ne/-sw/-se are copied from --ref into the clone
# before the first stage that reads all four chances (solver-walls, solver-text-scores), unless named.
#
# The stages run as `uv run --frozen sfzmaps stage NAME`, not `sfzmaps build --force NAME`: the build plans every
# stale upstream stage as well, and a fresh clone's dvc.lock has no record of the text/labels/walls/solve stages until
# the owner has run `sfzmaps ship`, so building solver-walls would retrain text-ne/-sw/-se. Exit status is 1 if any
# stage differs, 2 for a usage or setup problem.
set -u -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REF="$(cd "$SCRIPT_DIR/.." && pwd)"
CLONE=""
DRY=0
STAGES=()
while [ $# -gt 0 ]; do
  case "$1" in
    --clone) CLONE="${2:?--clone needs a folder}"; shift 2 ;;
    --ref) REF="$(cd "${2:?--ref needs a pipeline folder}" && pwd)"; shift 2 ;;
    --dry-run|-n) DRY=1; shift ;;
    -h|--help) sed -n '2,23p' "$0"; exit 0 ;;
    -*) echo "unknown option $1" >&2; exit 2 ;;
    *) STAGES+=("$1"); shift ;;
  esac
done
[ ${#STAGES[@]} -gt 0 ] || STAGES=(solver-page solver-lots solver-prints solver-text-nw solver-labels solver-walls solver-solve)
[ -n "$CLONE" ] || CLONE="$REF/data/fresh-clone"

known="warp-2000-digital-zoning solver-page solver-lots solver-prints solver-text-nw solver-text-ne solver-text-sw solver-text-se solver-text-scores solver-labels solver-walls solver-solve"
for s in "${STAGES[@]}"; do
  case " $known " in *" $s "*) ;; *) echo "unknown stage $s; known: $known" >&2; exit 2 ;; esac
done
has() { case " ${STAGES[*]} " in *" $1 "*) return 0 ;; *) return 1 ;; esac; }

REPO="$(git -C "$REF" rev-parse --show-toplevel)"
BRANCH="$(git -C "$REPO" rev-parse --abbrev-ref HEAD)"
HEAD_SHA="$(git -C "$REPO" rev-parse HEAD)"
URL="$(git -C "$REPO" remote get-url origin)"

echo "fresh-clone check: branch $BRANCH ($HEAD_SHA)"
echo "  origin   $URL"
echo "  clone    $CLONE"
echo "  compare  $REF"
echo "  stages   warp-2000-digital-zoning (only if the pull does not bring it) ${STAGES[*]}"
for q in ne sw se; do
  has "solver-text-$q" || echo "  note     solver-text-$q is NOT retrained: copied from $REF before the first stage that needs all four"
done
[ "$DRY" = 1 ] && { echo "(dry run: nothing cloned or built)"; exit 0; }

# --- the branch must be pushed -------------------------------------------------------------------
git -C "$REPO" fetch --quiet origin "$BRANCH" 2>/dev/null || { echo "branch $BRANCH is not on origin: push it first" >&2; exit 2; }
REMOTE_SHA="$(git -C "$REPO" rev-parse "origin/$BRANCH")"
if [ "$REMOTE_SHA" != "$HEAD_SHA" ]; then
  echo "origin/$BRANCH is $REMOTE_SHA, but HEAD is $HEAD_SHA: push the branch first (the clone is made from origin)" >&2
  exit 2
fi
if [ -n "$(git -C "$REPO" status --porcelain -- pipeline/src pipeline/inputs pipeline/dvc.yaml pipeline/pyproject.toml pipeline/uv.lock)" ]; then
  echo "warning: uncommitted code or inputs in $REPO are not in the clone" >&2
fi

if [ -e "$CLONE" ] && [ -n "$(ls -A "$CLONE" 2>/dev/null)" ]; then
  echo "$CLONE is not empty: remove it or pick another --clone" >&2; exit 2
fi
mkdir -p "$CLONE" || exit 2
CLONE="$(cd "$CLONE" && pwd)"
git clone --quiet --branch "$BRANCH" "$URL" "$CLONE" || exit 2
[ "$(git -C "$CLONE" rev-parse HEAD)" = "$HEAD_SHA" ] || { echo "the clone is not at $HEAD_SHA" >&2; exit 2; }

# pixi's tools first on the path, as `sfzmaps build` requires (AGENTS.md)
px() { (cd "$CLONE/pipeline" && pixi run --manifest-path "$CLONE/pixi.toml" "$@"); }
step() { echo; echo "== $*"; }

step "pixi install"; (cd "$CLONE" && pixi install) || exit 2
step "uv sync --frozen"; px uv sync --frozen || exit 2
step "sfzmaps pull (what is not shipped yet is built below)"; px uv run --frozen sfzmaps pull || echo "(pull reported the above; continuing)"

run_stage() { step "stage $1"; px uv run --frozen sfzmaps stage "$1"; }

if [ ! -d "$CLONE/pipeline/data/warp/2000-digital-zoning" ] || [ -z "$(ls -A "$CLONE/pipeline/data/warp/2000-digital-zoning" 2>/dev/null)" ]; then
  has warp-2000-digital-zoning || STAGES=(warp-2000-digital-zoning "${STAGES[@]}")
fi

copied=0
copy_other_quarters() {
  [ "$copied" = 1 ] && return
  copied=1
  for q in ne sw se; do
    has "solver-text-$q" && continue
    echo "copying text-$q from $REF/data/solver/text-$q (not retrained)"
    mkdir -p "$CLONE/pipeline/data/solver"
    rm -rf "$CLONE/pipeline/data/solver/text-$q"
    cp -R "$REF/data/solver/text-$q" "$CLONE/pipeline/data/solver/text-$q" || exit 2
  done
}

RESULTS=()
FAILED=0
for s in "${STAGES[@]}"; do
  case "$s" in solver-walls|solver-text-scores) copy_other_quarters ;; esac
  if run_stage "$s"; then
    out="$(cd "$REF" && uv run --frozen python scripts/compare_outputs.py "$s" "$CLONE/pipeline" "$REF")"; rc=$?
    echo "$out"
  else
    out="$s: stage failed"; rc=1; echo "$out"
  fi
  RESULTS+=("$out")
  [ $rc -eq 0 ] || FAILED=1
  [ "$s" = warp-2000-digital-zoning ] && [ $rc -ne 0 ] && break
done

echo; echo "== summary"
printf '%-28s %s\n' STAGE RESULT
for r in "${RESULTS[@]}"; do
  printf '%-28s %s\n' "${r%%:*}" "$(printf '%s' "$r" | head -1 | sed 's/^[^:]*: //')"
done
[ $FAILED -eq 0 ] && echo "fresh-clone check: every stage identical or within its epsilon" || echo "fresh-clone check: SOME STAGE DIFFERS"
exit $FAILED
