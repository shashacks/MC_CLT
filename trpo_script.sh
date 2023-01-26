#!/bin/bash
# ex ./trpo_script.sh Hopper-v2

env=$1

for i in {0..9}
do
    python -m spinup.run trpo --hid "[64,32]" --env ${env} --exp_name ${env}/TRPO --seed 500${i}
done

