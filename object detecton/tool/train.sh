#!/usr/bin/env bash

CONFIG=$1
GPUS=$2
PORT=${PORT:-20182}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/train.py $CONFIG --launcher pytorch ${@:3}

# PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
# python $(dirname "$0")//analysis_tools/get_flops.py $CONFIG 

# python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
#     $(dirname "$0")/test.py $CONFIG  ./quad_t3/epoch_36.pth  --eval bbox segm  --launcher pytorch ${@:3}