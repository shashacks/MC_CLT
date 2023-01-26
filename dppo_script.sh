#!/bin/bash
# ex ./dppo_script.sh Hopper-v2

env=$1

for i in {0..9}
do
   python -m spinup.run dppo --hid "[64,32]" --env ${env} --seed 500${i} --exp_name ${env}/DPPO_200 --d_output 8
done
