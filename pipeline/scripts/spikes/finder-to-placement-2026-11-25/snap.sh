#!/bin/zsh
# One-off check, not pipeline code: lists mtime and size of today's data folders and the all-pieces trial, to prove the rough-check left them untouched. Run from anywhere.
# mtime (ns-ish, fractional s) and size of every file under today's data folders and the all-pieces trial (links not followed)
cd "$(dirname "$0")/../../.."
find data/work data/warp data/out data/find/trial/all-pieces \( -type f -o -type l \) -exec /usr/bin/stat -f "%Fm %z %N" {} + | sort -k3
