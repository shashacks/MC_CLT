#!/bin/bash
# ex ./mc_clt_trpo.sh Swimmer-v2 0.01 10
# ex ./mc_clt_trpo.sh Hopper-v2 0.1 500
# ex ./mc_clt_trpo.sh Walker2d-v2 0.05 100
# ex ./mc_clt_trpo.sh HumanoidStandup-v2 0.001 1500
# ex ./mc_clt_trpo.sh Ant-v2 0.5 500
# ex ./mc_clt_trpo.sh BipedalWalker-v3 0.01 5
# ex ./mc_clt_trpo.sh BipedalWalkerHardcore-v3 0.001 10
# ex ./mc_clt_trpo.sh InvertedDoublePendulum-v2 0.001 1000

env=$1
T=$2
mn_std=$3

for i in {0..9}
do
   python -m spinup.run mc_clt_trpo --hid "[64,32]" --env ${env} --seed 500${i} --exp_name ${env}/MC_CLT_TRPO/T${T}_mn${mn_std} --mn_std ${mn_std} --T ${T} 
done

