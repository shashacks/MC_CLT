#!/bin/bash
# ex ./trpo_ensemble.sh Hopper-v2 128 0.9 0.6 True

env=$1
h_size=$2
target_weight=$3
min_weight=$4
u_weight_update=$5

for i in {0..9}
do
   if [ "$u_weight_update" == "True" ]; then
      python -m spinup.run trpo_ensemble --hid "[64,32]" --env ${env} --exp_name ${env}/TRPO_ENSEMBLE/${uncertainty_func}_h${h_size}_t${target_weight}_m${min_weight} --seed 500${i} --h_size ${h_size} --target_weight ${target_weight} --min_weight ${min_weight} 
   elif [ "$u_weight_update" == "False" ]; then
      python -m spinup.run trpo_ensemble --hid "[64,32]" --env ${env} --exp_name ${env}/TRPO_ENSEMBLE/noT_h${h_size} --seed 500${i} --h_size ${h_size} --u_weight_update ${u_weight_update} 
   fi    
done