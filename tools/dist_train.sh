#!/usr/bin/env bash

PYTHON=${PYTHON:-"python"}

CONFIG=$1
GPUS=$2

CUDA_VISIBLE_DEVICES=2,3 $PYTHON -m torch.distributed.launch --nproc_per_node=$GPUS --master_port 8527 \
    $(dirname "$0")/train.py $CONFIG --launcher pytorch ${@:3}
