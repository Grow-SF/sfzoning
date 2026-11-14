# Environment for rebuilding the prob-warp outputs with these scripts (source this from anywhere).
# Inputs: the pipeline (venv + data/work).  Outputs: pipeline/data/prob-warp/rebuild-2026-11-13 (git-ignored): run-notext
# (prob6 as it is now, with no text removal); run/ keeps the exact rebuild of the approved run, which had it.
_C=${${(%):-%x}:A:h}                                  # this folder (prob-warp/check)
export PW_CODE=$_C/../allyears
export PW_PIPELINE=${PW_PIPELINE:-${_C:h:h:h}}
export PW_WORK=$PW_PIPELINE/data/work
export PW_DATA=$PW_PIPELINE/data/prob-warp
export PW_TPY=${PW_TPY:-$PW_DATA/torchenv/bin/python}
export PW_RESCUED=$PW_DATA/rescued-2026-11-13          # what survived the /tmp cleaner: compared against, never written
export PW_RUN=${PW_RUN:-$PW_DATA/rebuild-2026-11-13/run-notext}
export PW_MIRROR=$PW_DATA/rebuild-2026-11-13/data_ay
export PW_MIRROR2=$PW_DATA/rebuild-2026-11-13/data_ay2
export PW_PIXI_BIN=${PW_PIXI_BIN:-$PW_PIPELINE/../.pixi/envs/default/bin}
