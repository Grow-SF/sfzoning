#!/bin/zsh
# Run try_parts.sh for every candidate parts.json side by side.
S=/private/tmp/claude-501/-Users-sbuss-workspace-sfzoning/eb18d2fc-e22a-48a6-afbc-57c31cd3e134/scratchpad
for c in k6/city4a k6/city4 k6/city4x k6/city3a k6/city3 k5/city3a; do
  zsh ${0:A:h}/try_parts.sh ${c//\//_} $S/$c/parts.json &
done
zsh ${0:A:h}/try_parts.sh fresh $PW_DATA/rebuild-2026-11-13/scratch/stepview/city_m2/parts.json &
wait
