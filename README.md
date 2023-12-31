# MC_CLT

## Descriptions
Learning a predictive model of the mean return, or value function, plays a critical role in many reinforcement learning algorithms. Distributional reinforcement learning (DRL) improves performance by modeling the value distribution. For continuous tasks, we study the value distribution and find that the trained value distribution is empirical quite close to normal. We design a method that exploits this property. We employ variances predicted from a variance network, along with returns, to analytically compute target quantile bars representing a normal for our distributional value function. In addition, we propose a policy update strategy based on the correctness as measured by structural characteristics of the value distribution not present in the standard value function. The approach we outline is compatible with many DRL structures. We use two representative on-policy algorithms, PPO and TRPO, as testbeds. Our method yields statistically significant improvements in 10 out of 16 continuous task settings, while utilizing a reduced number of weights and achieving faster training time compared to an ensemble-based method.

## Dependencies
To install the dependencies below:
<pre>
<code>
conda create -n env_name python=3.6
pip install mujoco-py==1.50.1.68
pip install scipy
pip install gym==0.15.7
pip install -e .
pip install box2d-py
git clone https://github.com/benelot/pybullet-gym.git
cd pybullet-gym
pip install -e .
</code>
</pre>

### Hyperparameters
We provide the hyperparameters used in our experiment. Those are the best configuration, so the average scores would be higher than the reported numbers. Please refer to hyperparameters.md.  To exclude the uncertainty weight for the policy update, set u_weight_update as False in script files.

### Train MC_CLT_PPO 
<pre>
<code>
./mc_clt_ppo.sh Hopper-v2 128 128 0.9 0.6 True
</code>
</pre>

### Train PPO_Ensemble 
<pre>
<code>
./ppo_ensemble.sh Hopper-v2 128 0.85 0.4 True
</code>
</pre>


### References 
* https://github.com/openai/spinningup
