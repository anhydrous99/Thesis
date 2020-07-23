#!/bin/bash
for (( c = 0; c < $(nproc) - 4; c++ ))
do
  d=$(( $c % 4 ))
  tmux new -d "sleep 10; conda activate $1; CUDA_VISIBLE_DEVICES=$d python OrbitDecayEnv.py --optuna"
done