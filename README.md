# MC_CLT

## Descriptions
Learning a predictive model of the mean return, or value function, plays a critical role in many reinforcement learning algorithms. Distributional reinforcement learning (DRL) methods instead model the value distribution, which has been shown to improve performance in many settings. In this paper, we model the value distribution as approximately normal using the Markov Chain central limit theorem. We analytically compute quantile bars to provide a new DRL target that is informed by the decrease in standard deviation that occurs over the course of an episode. In addition, we propose a policy update strategy based on uncertainty as measured by structural characteristics of the value distribution not present in the standard value function. The approach we outline is compatible with many DRL structures. We use two representative on-policy algorithms, PPO and TRPO, as testbeds and show that our methods produce performance improvements in continuous control tasks.

## Dependencies
To install the dependencies below:
<pre>
<code>
pip install -e .
</code>
</pre>
* cloudpickle==1.2.1
* gym[atari,box2d,classic_control]~=0.15.3
* ipython
* joblib
* matplotlib==3.1.1
* mpi4py
* numpy
* pandas
* pytest
* psutil
* scipy
* seaborn==0.8.1
* tensorflow>=1.8.0,<2.0
* torch==1.3.1
* tqdm

### Train MC_CLT_PPO 
<pre>
<code>
./mc_clt_ppo.sh Hopper-v2 0.01 500
</code>
</pre>

### Train PPO + $V^{D, \pi}$
<pre>
<code>
./dppo_script.sh Hopper-v2
</code>
</pre>

### References 
* https://github.com/openai/spinningup
