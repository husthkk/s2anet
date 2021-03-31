#!/usr/bin/env bash

PYTHON=${PYTHON:-"python"}

CONFIG=$1
GPUS=$2

CUDA_VISIBLE_DEVICES=4,5 $PYTHON -m torch.distributed.launch --nproc_per_node=$GPUS --master_port 8528 \
    $(dirname "$0")/train.py $CONFIG --launcher pytorch ${@:3}
