#!/bin/bash
# ex ./ppo_script.sh Hopper-v2

env=$1

for i in {0..9}
do
    python -m spinup.run ppo --hid "[64,32]" --env ${env} --exp_name ${env}/PPO --seed 500${i}
done

