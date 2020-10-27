#!/bin/bash

PARTITION=${1:-"gpu"}
NUM_GPU=${2:-"2"}

set -o xtrace
srun --account=pi_ferraro --partition=${PARTITION} --mem=40000 --nodes=1 --ntasks-per-node=8 --gres=gpu:${NUM_GPU} --pty --preserve-env --qos=medium+  $SHELL -l
