#!/bin/zsh
# The torch environment (PW_TPY, default pipeline/data/prob-warp/torchenv): Python 3.12 with exactly
# ../requirements-torch.txt.  Made if missing, else left as it is.  The rebuild scripts call this first.
# usage: torchenv.sh
CK=${0:A:h}
source $CK/rebuild_env.sh
[ -x $PW_TPY ] && exit 0
echo "torch environment: making ${PW_TPY:h:h}"
uv venv --python 3.12 ${PW_TPY:h:h} && uv pip install --python $PW_TPY -r $CK/../requirements-torch.txt
