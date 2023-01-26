#!/bin/bash
# ex ./mc_clt_ppo.sh Swimmer-v2 0.01 10
# ex ./mc_clt_ppo.sh Hopper-v2 0.01 500
# ex ./mc_clt_ppo.sh Walker2d-v2 0.01 100
# ex ./mc_clt_ppo.sh HumanoidStandup-v2 0.001 1500
# ex ./mc_clt_ppo.sh Ant-v2 0.5 500
# ex ./mc_clt_ppo.sh BipedalWalker-v3 0.5 5
# ex ./mc_clt_ppo.sh BipedalWalkerHardcore-v3 1.0 50
# ex ./mc_clt_ppo.sh InvertedDoublePendulum-v2 0.001 1000


env=$1
T=$2
mn_std=$3

for i in {0..9}
do
   python -m spinup.run mc_clt_ppo --hid "[64,32]" --env ${env} --seed 500${i} --exp_name ${env}/MC_CLT/T${T}_mn${mn_std} --mn_std ${mn_std} --T ${T} 
done

