#!/bin/bash
# ex ./mc_clt_ppo.sh Hopper-v2 128 128 0.9 0.6 True

env=$1
d_h_size=$2
std_h_size=$3
target_weight=$4
min_weight=$5
u_weight_update=$6

for i in {0..9}
do
   if [ "$u_weight_update" == "True" ]; then
      python -m spinup.run rn_ppo --hid "[64,32]" --env ${env} --exp_name ${env}/RN_PPO/h${d_h_size}_s${std_h_size}_t${target_weight}_m${min_weight} --seed 500${i} --d_h_size ${d_h_size} --std_h_size ${std_h_size} --target_weight ${target_weight} --min_weight ${min_weight}
   elif [ "$u_weight_update" == "False" ]; then
      python -m spinup.run rn_ppo --hid "[64,32]" --env ${env} --exp_name ${env}/RN_PPO/noT_h${d_h_size}_s${std_h_size} --seed 500${i} --d_h_size ${d_h_size} --std_h_size ${std_h_size} --u_weight_update ${u_weight_update}
   fi
done