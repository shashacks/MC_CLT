"""
TRPO is almost the same as PPO. The only difference is the update rule that
1) computes the search direction via conjugate
2) compute step by backtracking
"""

import time

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.distributions
from torch.optim import Adam

import spinup.algos.pytorch.rn_trpo.core as core
from spinup.utils.eval import eval
from spinup.utils.logx import EpochLogger
from spinup.utils.mpi_pytorch import mpi_avg_grads, sync_params, setup_pytorch_for_mpi
from spinup.utils.mpi_tools import mpi_fork, proc_id, mpi_statistics_scalar, num_procs
import bisect
import pybulletgym  # register PyBullet enviroments with open ai gym

EPS = 1e-8


class GAEBuffer:
    """
    A buffer for storing trajectories experienced by a PPO agent interacting
    with the environment, and using Generalized Advantage Estimation (GAE-Lambda)
    for calculating the advantages of state-action pairs.
    """

    def __init__(self, obs_dim, act_dim, size, gamma=0.99, lam=0.95, d_output=8, num_net=3):
        self.obs_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(core.combined_shape(size, act_dim), dtype=np.float32)
        self.adv_buf = np.zeros(size, dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.ret_buf = np.zeros(size, dtype=np.float32)
        self.logp_buf = np.zeros(size, dtype=np.float32)
        self.d_val_buf = np.zeros(size, dtype=np.float32)
        self.u_weight_buf = np.zeros(size, dtype=np.float32)

        self.gamma, self.lam = gamma, lam
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size
        self.d_output = d_output
        self.num_net = num_net
        self.last_vals = []
        self.last_diffs = []
        self.ptrs = []


    def store(self, obs, act, rew, logp, d_val):
        """
        Append one timestep of agent-environment interaction to the buffer.
        """
        assert self.ptr < self.max_size     # buffer has to have room so you can store
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.logp_buf[self.ptr] = logp
        self.d_val_buf[self.ptr] = d_val

        self.ptr += 1
    
    def store_u_weight(self, u_weights):
        self.u_weight_buf = u_weights

    def save_cur_ptr_and_last_values(self, last_val=0, last_diff=0):
        self.ptrs.append(self.ptr)
        self.last_vals.append(last_val)
        self.last_diffs.append(last_diff)

    def compute_advantage(self, T, adaptive_temperature, min_weight):
        while len(self.last_vals) > 0:
            path_start_idx = 0 if len(self.ptrs) == 1 else self.ptrs[-2]
            ptr = self.ptrs.pop()
            last_val = self.last_vals.pop()
            last_diff = np.array(self.last_diffs.pop())

            scaled_diff = (T * last_diff).mean()
            last_u_weight = ((1-min_weight)/0.5)*1/(1 + np.exp(scaled_diff)) + min_weight if adaptive_temperature else 1.0

            path_slice = slice(path_start_idx, ptr)
            rews = np.append(self.rew_buf[path_slice], last_val)
            d_vals = np.append(self.d_val_buf[path_slice], last_val)
            u_weights = np.append(self.u_weight_buf[path_slice], last_u_weight)
            # print(u_weights)
            # the next two lines implement GAE-Lambda advantage calculation 
            deltas = rews[:-1] + self.gamma * d_vals[1:] - d_vals[:-1] # with ue reward r' and s_v
            weights = (self.gamma * u_weights[1:] + u_weights[:-1]) * 0.5
            self.adv_buf[path_slice] = core.discount_cumsum_with_uncertainty_weight(deltas, weights, self.gamma * self.lam)
        
            # the next line computes rewards-to-go, to be targets for the value function
            self.ret_buf[path_slice] = self.adv_buf[path_slice] + d_vals[:-1]
    
        
        assert len(self.last_vals) == 0

    def get(self):
        """
        Call this at the end of an epoch to get all of the data from
        the buffer, with advantages appropriately normalized (shifted to have
        mean zero and std one). Also, resets some pointers in the buffer.
        """
        assert self.ptr == self.max_size    # buffer has to be full before you can get
        self.ptr, self.path_start_idx = 0, 0
        # the next two lines implement the advantage normalization trick
        adv_mean, adv_std = mpi_statistics_scalar(self.adv_buf)
        self.adv_buf = (self.adv_buf - adv_mean) / adv_std
        data = dict(obs=self.obs_buf, act=self.act_buf, ret=self.ret_buf,
                    adv=self.adv_buf, logp=self.logp_buf)
        return {k: torch.as_tensor(v, dtype=torch.float32) for k,v in data.items()}


def rn_trpo(env_fn, actor_critic=core.MLPActorCritic, ac_kwargs=dict(), seed=0,
         steps_per_epoch=4000, epochs=250, gamma=0.99, delta=0.01, d_vf_lr=1e-3,
         train_v_iters=80, damping_coeff=0.1, cg_iters=10, backtrack_iters=10,
         backtrack_coeff=0.8, lam=0.97, max_ep_len=1000, logger_kwargs=dict(),
         save_freq=10, algo='dtrpo', d_output=8, T=1, d_h_size=256, std_h_size=128, target_weight=0.9, 
         interval=0.1, u_weight_update=True, min_weight=0.3, uncertainty_func='sig', num_net=3, lr_std=1.0, d_act='relu'):
    """
    Trust Region Policy Optimization
    (with support for Natural Policy Gradient)
    Args:
        env_fn : A function which creates a copy of the environment.
            The environment must satisfy the OpenAI Gym API.
        actor_critic: The constructor method for a PyTorch Module with a
            ``step`` method, an ``act`` method, a ``pi`` module, and a ``v``
            module. The ``step`` method should accept a batch of observations
            and return:

            ===========  ================  ======================================
            Symbol       Shape             Description
            ===========  ================  ======================================
            ``a``        (batch, act_dim)  | Numpy array of actions for each
                                           | observation.
            ``v``        (batch,)          | Numpy array of value estimates
                                           | for the provided observations.
            ``logp_a``   (batch,)          | Numpy array of log probs for the
                                           | actions in ``a``.
            ===========  ================  ======================================

            The ``act`` method behaves the same as ``step`` but only returns ``a``.

            The ``pi`` module's forward call should accept a batch of
            observations and optionally a batch of actions, and return:

            ===========  ================  ======================================
            Symbol       Shape             Description
            ===========  ================  ======================================
            ``pi``       N/A               | Torch Distribution object, containing
                                           | a batch of distributions describing
                                           | the policy for the provided observations.
            ``logp_a``   (batch,)          | Optional (only returned if batch of
                                           | actions is given). Tensor containing
                                           | the log probability, according to
                                           | the policy, of the provided actions.
                                           | If actions not given, will contain
                                           | ``None``.
            ===========  ================  ======================================

            The ``v`` module's forward call should accept a batch of observations
            and return:

            ===========  ================  ======================================
            Symbol       Shape             Description
            ===========  ================  ======================================
            ``v``        (batch,)          | Tensor containing the value estimates
                                           | for the provided observations. (Critical:
                                           | make sure to flatten this!)
            ===========  ================  ======================================
        ac_kwargs (dict): Any kwargs appropriate for the actor_critic
            function you provided to TRPO.
        seed (int): Seed for random number generators.
        steps_per_epoch (int): Number of steps of interaction (state-action pairs)
            for the agent and the environment in each epoch.
        epochs (int): Number of epochs of interaction (equivalent to
            number of policy updates) to perform.
        gamma (float): Discount factor. (Always between 0 and 1.)
        delta (float): KL-divergence limit for TRPO / NPG update.
            (Should be small for stability. Values like 0.01, 0.05.)
        vf_lr (float): Learning rate for value function optimizer.
        train_v_iters (int): Number of gradient descent steps to take on
            value function per epoch.
        damping_coeff (float): Artifact for numerical stability, should be
            smallish. Adjusts Hessian-vector product calculation:

            .. math:: Hv \\rightarrow (\\alpha I + H)v
            where :math:`\\alpha` is the damping coefficient.
            Probably don't play with this hyperparameter.
        cg_iters (int): Number of iterations of conjugate gradient to perform.
            Increasing this will lead to a more accurate approximation
            to :math:`H^{-1} g`, and possibly slightly-improved performance,
            but at the cost of slowing things down.
            Also probably don't play with this hyperparameter.
        backtrack_iters (int): Maximum number of steps allowed in the
            backtracking line search. Since the line search usually doesn't
            backtrack, and usually only steps back once when it does, this
            hyperparameter doesn't often matter.
        backtrack_coeff (float): How far back to step during backtracking line
            search. (Always between 0 and 1, usually above 0.5.)
        lam (float): Lambda for GAE-Lambda. (Always between 0 and 1,
            close to 1.)
        max_ep_len (int): Maximum length of trajectory / episode / rollout.
        logger_kwargs (dict): Keyword args for EpochLogger.
        save_freq (int): How often (in terms of gap between epochs) to save
            the current policy and value function.
        algo: Either 'trpo' or 'npg': this code supports both, since they are
            almost the same.
    """

    # Special function to avoid certain slowdowns from PyTorch + MPI combo.
    setup_pytorch_for_mpi()

    # Set up logger and save configuration
    logger = EpochLogger(**logger_kwargs)
    logger.save_config(locals())

    # Random seed
    seed += 10000 * proc_id()
    torch.manual_seed(seed)
    np.random.seed(seed)

    import random
    random.seed(seed)   
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)       # for multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    env = gym.make(env_fn)
    env.seed(seed)
    obs_dim = env.observation_space.shape
    act_dim = env.action_space.shape
    env_eval = gym.make(env_fn)
    env_eval.seed(seed)

    d_act = nn.ReLU if d_act == 'relu' else nn.Tanh

    # Create actor-critic module
    ac = actor_critic(env.observation_space, env.action_space, num_net, **ac_kwargs, d_ouput=d_output, d_h_size=(d_h_size, d_h_size), std_h_size=(std_h_size, std_h_size), d_activation=d_act)

    # Sync params across processes
    sync_params(ac)

    # Count variables
    var_counts = tuple(core.count_vars(module) for module in [ac.pi, ac.ensemble_d_v])
    logger.log('\nNumber of parameters: \t pi: %d, \t v: %d\n' % var_counts)

    # Set up experience buffer
    local_steps_per_epoch = int(steps_per_epoch / num_procs())
    buf = GAEBuffer(obs_dim, act_dim, local_steps_per_epoch, gamma, lam, d_output, num_net)

    sequence = torch.arange(num_net)
    repeat_num = steps_per_epoch // len(sequence)
    mask = sequence.repeat(repeat_num)
    remaining = steps_per_epoch - len(mask)

    if remaining > 0:
        mask = torch.cat((mask, sequence[:remaining]))

    ones_tensor = torch.ones_like(mask)
    zeros_tensor = torch.zeros_like(mask)

    # z value range
    z_bar_range = np.linspace(-3, 3, num = 2000000)
    z_cdf = np.zeros_like(z_bar_range) 
    cum_prob = []
    for i in range(d_output):
        cum_prob.append((i + 1) / (d_output + 1))
    cum_prob = torch.from_numpy(np.array(cum_prob).reshape(1, -1))

    epsilon = 0.0001

    def compute_standard_normal_locations(n):
        dx = z_bar_range[1] - z_bar_range[0]
        per_sum = 0.00135
        z_cdf[0] = per_sum
        amount = 1.0 / (n + 1)
        target = 1.0 / (n + 1)
        locations = []
        constant = 1.0 / np.sqrt(2*np.pi)
        for i in range(len(z_bar_range)-1):
            per_sum += 0.5 * dx * (constant * np.exp((-z_bar_range[i]**2) / 2.0) + constant * np.exp((-z_bar_range[i+1]**2) / 2.0))
            z_cdf[i+1] = min(per_sum, 1.0) 
            if per_sum >= target and len(locations) < int(n/2):
                locations.append((z_bar_range[i]+z_bar_range[i+1])/2.0)
                target += amount

        z_cdf[len(z_cdf) - 1] = 1.0
        lim = int(n/2)
        if n % 2 == 1:
            locations.append(0.0)
        for j in range(lim):
            i = lim - j - 1
            locations.append(-locations[i])    
        return np.array(locations)
    
    def sig(x):
        return 1/(1 + np.exp(-x))

    def compute_diff(bars):
        av_std = 0.0
        z_values = []
        for i in range(d_output):
            cur_portion = (i + 1.0) / (d_output + 1)
            idx = bisect.bisect_left(z_cdf, cur_portion)
            cand_std = (bars[i] - np.mean(bars)) / (z_bar_range[idx] + 1e-6) 
            z_values.append(z_bar_range[idx])
            if cand_std < 0:
                return 10e4
            av_std += cand_std
        av_std = av_std / d_output
        z_values = np.array(z_values)
        normal_bars = z_values * av_std + np.mean(bars)
        diff = np.sum((bars - normal_bars)**2)
        return diff

    def compute_loss_pi(data):
        obs, act, adv, logp_old = data['obs'], data['act'], data['adv'], data['logp']
        # Policy loss
        _, logp = ac.pi(obs, act)
        ratio = torch.exp(logp - logp_old)
        loss_pi = -(ratio * adv).mean()
        return loss_pi

    def compute_loss_d_v(data, shuffled_mask):
        obs, ret  = data['obs'], data['ret']

        with torch.no_grad():
            pred_std = ac.ensemble_std(obs) # no gradient
            pred_std = torch.max(pred_std, epsilon * torch.ones_like(pred_std))
            pred_std = np.expand_dims(pred_std, axis=1)
            
        ret = np.expand_dims(ret, axis=1)
        loc = torch.as_tensor(pred_std * base_loc + ret, dtype=torch.float32)
        error = ((ac.ensemble_d_v(obs) - data['ret'])**2).detach()

        ratio = 0
        d_v_loss = 0
        for i in range(num_net):
            i_mask = torch.where(shuffled_mask == i, ones_tensor, zeros_tensor)
            
            pairwise_delta = loc - ac.ensemble_d_v.forward_idx(obs, i) 
            abs_pairwise_delta = torch.abs(pairwise_delta)
            huber_loss = torch.where(abs_pairwise_delta > 1, abs_pairwise_delta - 0.5, pairwise_delta ** 2 * 0.5)
            quantile_huber_loss = (torch.abs(cum_prob - (pairwise_delta.detach() < 0).float()) * huber_loss).mean(axis=1)
            
            d_v_loss += i_mask * quantile_huber_loss

            grad_std = ac.ensemble_std.forward_idx(obs, i)
            grad_std = torch.max(grad_std, epsilon * torch.ones_like(grad_std))
            ratio += i_mask * error / (2*grad_std**2) + torch.log(grad_std)
        
        loss = d_v_loss + lr_std * ratio

        return loss.mean()
        

    def compute_kl(data, old_pi):
        obs, act = data['obs'], data['act']
        pi, _ = ac.pi(obs, act)
        kl_loss = torch.distributions.kl_divergence(pi, old_pi).mean()
        return kl_loss

    @torch.no_grad()
    def compute_kl_loss_pi(data, old_pi):
        obs, act, adv, logp_old = data['obs'], data['act'], data['adv'], data['logp']
        # Policy loss
        pi, logp = ac.pi(obs, act)
        ratio = torch.exp(logp - logp_old)
        loss_pi = -(ratio * adv).mean()
        kl_loss = torch.distributions.kl_divergence(pi, old_pi).mean()
        return loss_pi, kl_loss

    def hessian_vector_product(data, old_pi, v):
        kl = compute_kl(data, old_pi)

        grads = torch.autograd.grad(kl, ac.pi.parameters(), create_graph=True)
        flat_grad_kl = core.flat_grads(grads)

        kl_v = (flat_grad_kl * v).sum()
        grads = torch.autograd.grad(kl_v, ac.pi.parameters())
        flat_grad_grad_kl = core.flat_grads(grads)

        return flat_grad_grad_kl + v * damping_coeff

    # Set up optimizers for policy and value function
    d_vf_optimizer = Adam(list(ac.ensemble_d_v.parameters()) + list(ac.ensemble_std.parameters()), lr=d_vf_lr)

    # Set up model saving
    logger.setup_pytorch_saver(ac)

    def update():
        data = buf.get()
        shuffled_mask = mask[torch.randperm(mask.size(0))].detach()

        # compute old pi distribution
        obs, act = data['obs'], data['act']
        with torch.no_grad():
            old_pi, _ = ac.pi(obs, act)

        pi_loss = compute_loss_pi(data)
        pi_l_old = pi_loss.item()
        d_v_l_old = compute_loss_d_v(data, shuffled_mask).item()

        grads = core.flat_grads(torch.autograd.grad(pi_loss, ac.pi.parameters()))

        # Core calculations for TRPO or NPG
        Hx = lambda v: hessian_vector_product(data, old_pi, v)
        x = core.conjugate_gradients(Hx, grads, cg_iters)

        alpha = torch.sqrt(2 * delta / (torch.matmul(x, Hx(x)) + EPS))

        old_params = core.get_flat_params_from(ac.pi)

        def set_and_eval(step):
            new_params = old_params - alpha * x * step
            core.set_flat_params_to(ac.pi, new_params)
            loss_pi, kl_loss = compute_kl_loss_pi(data, old_pi)
            return kl_loss.item(), loss_pi.item()

        if algo == 'npg':
            # npg has no backtracking or hard kl constraint enforcement
            kl, pi_l_new = set_and_eval(step=1.)

        elif algo == 'dtrpo':
            # trpo augments npg with backtracking line search, hard kl
            for j in range(backtrack_iters):
                kl, pi_l_new = set_and_eval(step=backtrack_coeff ** j)
                if kl <= delta and pi_l_new <= pi_l_old:
                    logger.log('Accepting new params at step %d of line search.' % j)
                    logger.store(BacktrackIters=j)
                    break

                if j == backtrack_iters - 1:
                    logger.log('Line search failed! Keeping old params.')
                    logger.store(BacktrackIters=j)
                    kl, pi_l_new = set_and_eval(step=0.)

        # Value function learning
        for i in range(train_v_iters):
            d_vf_optimizer.zero_grad()
            loss_d_v = compute_loss_d_v(data, shuffled_mask)
            loss_d_v.backward()
            mpi_avg_grads(ac.ensemble_d_v)
            mpi_avg_grads(ac.ensemble_std)
            d_vf_optimizer.step()

        # Log changes from update
        logger.store(LossPi=pi_l_old, KL=kl,
                     DeltaLossPi=(pi_l_new - pi_l_old))
        
        logger.store(LossDV=d_v_l_old,
            DeltaLossDV=(loss_d_v.item() - d_v_l_old))

    # Prepare for interaction with environment
    start_time = time.time()
    o, ep_ret, ep_len = env.reset(), 0, 0

    # Main loop: collect experience in env and update/log each epoch
    base_loc = compute_standard_normal_locations(d_output)
    ep_rewards = []
    ep_lens = []

    for epoch in range(epochs):
        ep_time = 0
        order_count = 0
        diffs = []
        pred_stds = []
        for t in range(local_steps_per_epoch):
            a, d_v, logp = ac.step(torch.as_tensor(o, dtype=torch.float32))
            order_count += ac.order_check(torch.as_tensor(o, dtype=torch.float32), num_net)
            with torch.no_grad():                
                pred_std = ac.ensemble_std(torch.as_tensor(o, dtype=torch.float32)) # no gradien
                pred_std = torch.max(pred_std, epsilon * torch.ones_like(pred_std))
                pred_stds.append(pred_std)
            next_o, r, d, _ = env.step(a)
            d_v_bars = ac.ensemble_d_v.get_d_v_bars(torch.as_tensor(o, dtype=torch.float32)) 

            diff = []
            for d_v_bar in d_v_bars:
                diff.append(compute_diff(d_v_bar))
            diffs.append(diff)

            ep_ret += r
            ep_len += 1

            buf.store(o, a, r, logp, d_v)
            logger.store(VVals=np.mean(d_v))
            ep_time += 1

            o = next_o

            timeout = ep_len == max_ep_len
            terminal = d or timeout
            epoch_ended = t == local_steps_per_epoch - 1

            if terminal or epoch_ended:
                if epoch_ended and not (terminal):
                    print('Warning: trajectory cut off by epoch at %d steps.' % ep_len, flush=True)
                # if trajectory didn't reach terminal state, bootstrap value target
                if timeout or epoch_ended:
                    _, v, _ = ac.step(torch.as_tensor(o, dtype=torch.float32))
                    last_d_v_bars = ac.ensemble_d_v.get_d_v_bars(torch.as_tensor(o, dtype=torch.float32)) 
                    last_diff = []
                    for last_d_v_bar in last_d_v_bars:
                        last_diff.append(compute_diff(last_d_v_bar))

                else:
                    v = 0
                    last_diff = 0
                buf.save_cur_ptr_and_last_values(v, last_diff)
                if terminal:
                    # only save EpRet / EpLen if trajectory finished
                    logger.store(EpRet=ep_ret, EpLen=ep_len)
                    if len(ep_rewards) >= 50:
                        ep_rewards.pop(0)
                    ep_rewards.append(ep_ret)
                    if len(ep_lens) >= 50:
                        ep_lens.pop(0)
                    ep_lens.append(ep_len)
                o, ep_ret, ep_len, ep_time = env.reset(), 0, 0, 0

        # Save model
        # if (epoch % save_freq == 0) or (epoch == epochs - 1):
        #     logger.save_state({'env': env}, None)

        avg_u_w = 0.0
        if u_weight_update:
            l_w, h_w = target_weight - interval, target_weight + interval
            left, right = 0, 2**12
            diffs = np.array(diffs)
            while left <= right:
                T = (left + right) / 2
                T = max(T, 0)
                if uncertainty_func == 'sig':
                    scaled_diffs = (T * diffs)
                    uncertainty_weights = ((1-min_weight)/0.5)*sig(-scaled_diffs) + min_weight
                    uncertainty_weights = uncertainty_weights.mean(axis=-1)
                else:
                    raise Exception('uncertainty_func is not tan or sig') 
                avg_u_w = np.mean(uncertainty_weights)
                # print(avg_u_w, T, left, right)
                
                if avg_u_w <= h_w and avg_u_w >= l_w:
                    buf.store_u_weight(uncertainty_weights)
                    break
                elif avg_u_w > target_weight:
                    left = T + epsilon * epsilon
                else:
                    right = T - epsilon * epsilon
                

            if left > right or avg_u_w == 0.0:
                T = -1
                uncertainty_weights = np.ones_like(buf.u_weight_buf)
                avg_u_w = 1.0
                buf.store_u_weight(uncertainty_weights)
        else:
            T = -1
            uncertainty_weights = np.ones_like(buf.u_weight_buf)
            avg_u_w = 1.0
            buf.store_u_weight(uncertainty_weights)

        buf.compute_advantage(T, u_weight_update, min_weight)
        update()

        eval_returns, eval_ep_lens = eval(ac , env_eval, max_ep_len, epochs==(epoch + 1), logger.output_file.name)

        logger.log_tabular('Epoch', epoch)
        logger.log_tabular('AverageEpRet', np.mean(eval_returns))
        logger.log_tabular('StdEpRet', np.std(eval_returns))
        logger.log_tabular('MaxEpRet', np.max(eval_returns))
        logger.log_tabular('MinEpRet', np.min(eval_returns))
        logger.log_tabular('EpLen', np.mean(eval_ep_lens))
        logger.log_tabular('VVals', with_min_and_max=True)
        logger.log_tabular('TotalEnvInteracts', (epoch + 1) * steps_per_epoch)
        logger.log_tabular('LossPi', average_only=True)
        logger.log_tabular('DeltaLossPi', average_only=True)
        logger.log_tabular('KL', average_only=True)
        logger.log_tabular('LossDV', average_only=True)
        logger.log_tabular('DeltaLossDV', average_only=True)
        logger.log_tabular('MeanUncertaintyWeight', avg_u_w)
        logger.log_tabular('AfterTemperature', T)
        logger.log_tabular('PredStdMean', np.mean(pred_stds))
        logger.log_tabular('PredStdMinimum', np.min(pred_stds))
        logger.log_tabular('PredStdMaximum', np.max(pred_stds))
        logger.log_tabular('OrderRatio', order_count / (num_net * local_steps_per_epoch))
        
        if algo == 'dtrpo':
            logger.log_tabular('BacktrackIters', average_only=True)
        logger.log_tabular('Time', time.time() - start_time)
        logger.dump_tabular()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='HalfCheetah-v2')
    parser.add_argument('--hid', type=int, default=64)
    parser.add_argument('--l', type=int, default=2)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--cpu', type=int, default=4)
    parser.add_argument('--steps', type=int, default=4000)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--exp_name', type=str, default='dtrpo')
    args = parser.parse_args()

    mpi_fork(args.cpu)  # run parallel code with mpi

    from spinup.utils.run_utils import setup_logger_kwargs

    logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed)

    rn_trpo(lambda: gym.make(args.env), actor_critic=core.MLPActorCritic,
         ac_kwargs=dict(hidden_sizes=[args.hid] * args.l), gamma=args.gamma,
         seed=args.seed, steps_per_epoch=args.steps, epochs=args.epochs,
         logger_kwargs=logger_kwargs)
