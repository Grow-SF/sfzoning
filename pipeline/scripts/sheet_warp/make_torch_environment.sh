#!/bin/zsh
# The torch environment the fits run in (WARP_TORCH_PYTHON, default pipeline/data/sheet-warp/torch-environment):
# Python 3.12 with exactly torch_requirements.txt.  Made if missing, else left as it is.  The fit scripts call this.
# usage: make_torch_environment.sh
CK=${0:A:h}
source $CK/settings.sh
[ -x $WARP_TORCH_PYTHON ] && exit 0
echo "torch environment: making ${WARP_TORCH_PYTHON:h:h}"
uv venv --python 3.12 ${WARP_TORCH_PYTHON:h:h} && uv pip install --python $WARP_TORCH_PYTHON -r $CK/torch_requirements.txt
