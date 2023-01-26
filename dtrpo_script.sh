#!/bin/bash
# ex ./dtrpo_script.sh Hopper-v2
# python -m spinup.run dtrpo --hid "[64,32]" --env Hopper-v2 --exp_name temp --seed 5000
env=$1

for i in {0..9}
do
   python -m spinup.run dtrpo --hid "[64,32]" --env ${env} --seed 500${i} --exp_name ${env}/DTRPO --d_output 8
done
