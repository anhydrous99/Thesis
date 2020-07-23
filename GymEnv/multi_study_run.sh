#!/bin/sh
for (( c=1; c<= $(nproc) / 2; c++))
do
    screen -dm bash -c "sleep 10; conda activate $1; python OrbitDecayEnv.py --optuna"
done