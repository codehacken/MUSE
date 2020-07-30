#!/bin/bash

set -o xtrace
srun --account=pi_ferraro --partition=gpu --mem=40000 --nodes=1 --ntasks-per-node=8 --gres=gpu:2 --pty --preserve-env --qos=medium+  $SHELL -l
