# Where the sheet warp reads and writes (source this from anywhere; each setting can be set beforehand instead).
# It reads the pipeline's data (pipeline/data/work, pipeline/data/ref) and writes only under pipeline/data/sheet-warp.
_C=${${(%):-%x}:A:h}                                                # this folder (pipeline/scripts/sheet_warp)
export WARP_STEPS=$_C/steps
export WARP_PIPELINE=${WARP_PIPELINE:-${_C:h:h}}                    # the pipeline folder
export WARP_WORK=$WARP_PIPELINE/data/work                           # the pipeline's sheets (read, never written)
export WARP_DATA=$WARP_PIPELINE/data/sheet-warp                     # everything the warp writes
export WARP_FITS=${WARP_FITS:-$WARP_DATA/fits}                      # per year and sheet: the corrections and their scores
export WARP_STAND_IN=$WARP_DATA/stand-in-data                       # the pipeline's data with each sheet's first correction
export WARP_STAND_IN_CHOSEN=$WARP_DATA/stand-in-data-chosen         # ... with each sheet's chosen correction; the warps
export WARP_LOGS=$WARP_DATA/logs
export WARP_AREA_2000=$WARP_DATA/measured-area-2000.json            # 2000 fitted as the approved run fitted it
export WARP_TORCH_PYTHON=${WARP_TORCH_PYTHON:-$WARP_DATA/torch-environment/bin/python}
export WARP_PIXI_BIN=${WARP_PIXI_BIN:-$WARP_PIPELINE/../.pixi/envs/default/bin}   # gdal, from pixi install
