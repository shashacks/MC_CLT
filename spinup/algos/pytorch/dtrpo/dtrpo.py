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

import spinup.algos.pytorch.dtrpo.core as core
from spinup.utils.eval import eval
from spinup.utils.logx import EpochLogger
from spinup.utils.mpi_pytorch import mpi_avg_grads, sync_params, setup_pytorch_for_mpi
from spinup.utils.mpi_tools import mpi_fork, proc_id, mpi_statistics_scalar, num_procs
import pybulletgym 

EPS = 1e-8


class GAEBuffer:
    """
    A buffer for storing trajectories experienced by a PPO agent interacting
    with the environment, and using Generalized Advantage Estimation (GAE-Lambda)
    for calculating the advantages of state-action pairs.
    """

    def __init__(self, obs_dim, act_dim, size, gamma=0.99, lam=0.95, d_output=8):
        self.obs_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(core.combined_shape(size, act_dim), dtype=np.float32)
        self.adv_buf = np.zeros(size, dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.ret_buf = np.zeros(core.combined_shape(size, d_output), dtype=np.float32)
        self.ue_val_buf = np.zeros(size, dtype=np.float32) # uncertainty bonus from ue network
        self.logp_buf = np.zeros(size, dtype=np.float32)
        self.d_val_buf = np.zeros(core.combined_shape(size, d_output), dtype=np.float32)

        self.gamma, self.lam = gamma, lam
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size
        self.d_output = d_output


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

    def finish_path(self, last_val=0):
        """
        Call this at the end of a trajectory, or when one gets cut off
        by an epoch ending. This looks back in the buffer to where the
        trajectory started, and uses rewards and value estimates from
        the whole trajectory to compute advantage estimates with GAE-Lambda,
        as well as compute the rewards-to-go for each state, to use as
        the targets for the value function.

        The "last_val" argument should be 0 if the trajectory ended
        because the agent reached a terminal state (died), and otherwise
        should be V(s_T), the value function estimated for the last state.
        This allows us to bootstrap the reward-to-go calculation to account
        for timesteps beyond the arbitrary episode horizon (or epoch cutoff).
        """
        
        path_slice = slice(self.path_start_idx, self.ptr)
        rews = np.append(self.rew_buf[path_slice], np.mean(last_val))
        d_last_val = np.reshape(last_val, (1, self.d_output))
        d_vals = np.append(self.d_val_buf[path_slice], d_last_val, axis=0)
        d_mean_vals = np.mean(d_vals, axis=1)

        # the next two lines implement GAE-Lambda advantage calculation 
        deltas = rews[:-1] + self.gamma * d_mean_vals[1:] - d_mean_vals[:-1] 
        self.adv_buf[path_slice] = core.discount_cumsum(deltas, self.gamma * self.lam)
        
        # the next line computes rewards-to-go, to be targets for the value function
        # self.ret_buf[path_slice] = core.discount_cumsum(rews, self.gamma)[:-1]
        self.ret_buf[path_slice] = np.reshape(self.adv_buf[path_slice], (-1, 1)) + d_vals[:-1]
        self.path_start_idx = self.ptr

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


def dtrpo(env_fn, actor_critic=core.MLPActorCritic, ac_kwargs=dict(), seed=0,
         steps_per_epoch=4000, epochs=250, gamma=0.99, delta=0.01, d_vf_lr=1e-3,
         train_v_iters=80, damping_coeff=0.1, cg_iters=10, backtrack_iters=10,
         backtrack_coeff=0.8, lam=0.97, max_ep_len=1000, logger_kwargs=dict(),
         save_freq=10, algo='dtrpo', d_output=8, d_h_size=64, d_act='relu'):
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

    # Instantiate environment
    env = gym.make(env_fn)
    env.seed(seed)
    obs_dim = env.observation_space.shape
    act_dim = env.action_space.shape
    env_eval = gym.make(env_fn)
    env_eval.seed(seed)

    d_act = nn.ReLU if d_act == 'relu' else nn.Tanh

    # Create actor-critic module
    ac = actor_critic(env.observation_space, env.action_space, **ac_kwargs, d_ouput=d_output, d_h_size=(d_h_size, d_h_size), d_activation=d_act)

    # Sync params across processes
    sync_params(ac)

    # Count variables
    var_counts = tuple(core.count_vars(module) for module in [ac.pi, ac.d_v])
    logger.log('\nNumber of parameters: \t pi: %d, \t v: %d\n' % var_counts)

    # Set up experience buffer
    local_steps_per_epoch = int(steps_per_epoch / num_procs())
    buf = GAEBuffer(obs_dim, act_dim, local_steps_per_epoch, gamma, lam, d_output)
    cum_prob = []
    for i in range(d_output):
        cum_prob.append((i + 1) / (d_output + 1))
    cum_prob = torch.from_numpy(np.array(cum_prob).reshape(1, -1))

    def compute_loss_pi(data):
        obs, act, adv, logp_old = data['obs'], data['act'], data['adv'], data['logp']
        # Policy loss
        _, logp = ac.pi(obs, act)
        ratio = torch.exp(logp - logp_old)
        loss_pi = -(ratio * adv).mean()
        return loss_pi

    def compute_loss_quantile_huber_d_v(data):
        obs, ret = data['obs'], data['ret']
        pairwise_delta = ac.d_v(obs) - ret
        abs_pairwise_delta = torch.abs(pairwise_delta)
        huber_loss = torch.where(abs_pairwise_delta > 1, abs_pairwise_delta - 0.5, pairwise_delta ** 2 * 0.5)
        loss = torch.abs(cum_prob - (pairwise_delta.detach() < 0).float()) * huber_loss
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
    d_vf_optimizer = Adam(ac.d_v.parameters(), lr=d_vf_lr)

    # Set up model saving
    logger.setup_pytorch_saver(ac)

    def update():
        data = buf.get()

        # compute old pi distribution
        obs, act = data['obs'], data['act']
        with torch.no_grad():
            old_pi, _ = ac.pi(obs, act)

        pi_loss = compute_loss_pi(data)
        pi_l_old = pi_loss.item()
        d_v_l_old = compute_loss_quantile_huber_d_v(data).item()

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
            loss_d_v = compute_loss_quantile_huber_d_v(data)
            loss_d_v.backward()
            mpi_avg_grads(ac.d_v)  # average grads across MPI processes
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
    for epoch in range(epochs):
        ep_time = 0
        order_count = 0
        for t in range(local_steps_per_epoch):
            a, d_v, logp = ac.step(torch.as_tensor(o, dtype=torch.float32))
            order_count += ac.order_check(torch.as_tensor(o, dtype=torch.float32))
            next_o, r, d, _ = env.step(a)
            ep_ret += r
            ep_len += 1

            # save and log
            buf.store(o, a, r, logp, d_v)
            logger.store(VVals=np.mean(d_v))
            ep_time += 1

            # Update obs (critical!)
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
                else:
                    v = np.zeros_like(d_v)
                buf.finish_path(v)
                if terminal:
                    # only save EpRet / EpLen if trajectory finished
                    logger.store(EpRet=ep_ret, EpLen=ep_len)
                o, ep_ret, ep_len = env.reset(), 0, 0

        # Save model
        # if (epoch % save_freq == 0) or (epoch == epochs - 1):
        #     logger.save_state({'env': env}, None)

        # Perform TRPO update!
        update()
        eval_returns, eval_ep_lens = eval(ac , env_eval, max_ep_len, epochs==(epoch + 1), logger.output_file.name)

        # Log info about epoch
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
        logger.log_tabular('OrderRatio', order_count / local_steps_per_epoch)
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

    dtrpo(lambda: gym.make(args.env), actor_critic=core.MLPActorCritic,
         ac_kwargs=dict(hidden_sizes=[args.hid] * args.l), gamma=args.gamma,
         seed=args.seed, steps_per_epoch=args.steps, epochs=args.epochs,
         logger_kwargs=logger_kwargs)
