#!/bin/bash
# ex ./mc_clt_trpo.sh Hopper-v2 128 128 0.85 0.4 True

env=$1
d_h_size=$2
std_h_size=$3
target_weight=$4
min_weight=$5
u_weight_update=$6

for i in {0..9}
do
   if [ "$u_weight_update" == "True" ]; then
      python -m spinup.run rn_trpo --hid "[64,32]" --env ${env} --exp_name ${env}/RN_TRPO/h${d_h_size}_s${std_h_size}_t${target_weight}_m${min_weight} --seed 500${i} --d_h_size ${d_h_size} --std_h_size ${std_h_size} --target_weight ${target_weight} --min_weight ${min_weight}
   elif [ "$u_weight_update" == "False" ]; then
      python -m spinup.run rn_trpo --hid "[64,32]" --env ${env} --exp_name ${env}/RN_TRPO/noT_h${d_h_size}_s${std_h_size} --seed 500${i} --d_h_size ${d_h_size} --std_h_size ${std_h_size} --u_weight_update ${u_weight_update}
   fi
done









#!/bin/bash
# ex ./mc_clt_trpo.sh Hopper-v2 500 50
# python -m spinup.run mc_clt_trpo --hid "[64,32]" --env Hopper-v2 --exp_name temp --seed 5000  --mn_std 20 --T 10
env=$1
T=$2
mn_std=$3

for i in {2..2}
do
   python -m spinup.run mc_clt_trpo --hid "[64,32]" --env ${env} --seed 500${i} --exp_name ${env}/MC_CLT_TRPO/T${T}_mn${mn_std} --mn_std ${mn_std} --T ${T} 
done

# for i in {0..9}
# do
#    python -m spinup.run mc_clt_ppo --hid "[64,32]" --env Walker2d-v2 --exp_name  --seed 5000 --exp_name Walker2d-v2_MC_CLT_T10 --mn_std 20 --T ${T} 
# done

