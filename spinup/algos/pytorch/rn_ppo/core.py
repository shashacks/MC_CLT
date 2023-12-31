import numpy as np
import scipy.signal
from gym.spaces import Box, Discrete

import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from torch.distributions.categorical import Categorical
import math

def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)


def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)


def count_vars(module):
    return sum([np.prod(p.shape) for p in module.parameters()])


def discount_cumsum(x, discount):
    """
    magic from rllab for computing discounted cumulative sums of vectors.

    input: 
        vector x, 
        [x0, 
         x1, 
         x2]

    output:
        [x0 + discount * x1 + discount^2 * x2,  
         x1 + discount * x2,
         x2]
    """
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]

def discount_cumsum_with_uncertainty_weight(x, w, discount):
    last_gae_lam = w[-1] * x[-1]
    res = np.zeros_like(x)
    res[-1] = last_gae_lam
    for step in reversed(range(len(x) - 1)):
        last_gae_lam = w[step] * x[step] + discount * last_gae_lam
        res[step] = last_gae_lam
    return res


class Actor(nn.Module):

    def _distribution(self, obs):
        raise NotImplementedError

    def _log_prob_from_distribution(self, pi, act):
        raise NotImplementedError

    def forward(self, obs, act=None):
        # Produce action distributions for given observations, and 
        # optionally compute the log likelihood of given actions under
        # those distributions.
        pi = self._distribution(obs)
        logp_a = None
        if act is not None:
            logp_a = self._log_prob_from_distribution(pi, act)
        return pi, logp_a


class MLPCategoricalActor(Actor):
    
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        self.logits_net = mlp([obs_dim] + list(hidden_sizes) + [act_dim], activation)

    def _distribution(self, obs):
        logits = self.logits_net(obs)
        return Categorical(logits=logits)

    def _log_prob_from_distribution(self, pi, act):
        return pi.log_prob(act)


class MLPGaussianActor(Actor):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        log_std = -0.5 * np.ones(act_dim, dtype=np.float32)
        self.log_std = torch.nn.Parameter(torch.as_tensor(log_std))
        self.mu_net = mlp([obs_dim] + list(hidden_sizes) + [act_dim], activation)

    def _distribution(self, obs):
        mu = self.mu_net(obs)
        std = torch.exp(self.log_std)
        return Normal(mu, std)
    
    def _get_mean(self, obs):
        return self.mu_net(obs)

    def _log_prob_from_distribution(self, pi, act):
        return pi.log_prob(act).sum(axis=-1)    # Last axis sum needed for Torch Normal distribution


class MLPCritic(nn.Module):

    def __init__(self, obs_dim, hidden_sizes, activation):
        super().__init__()
        self.v_net = mlp([obs_dim] + list(hidden_sizes) + [1], activation)

    def forward(self, obs):
        return torch.squeeze(self.v_net(obs), -1) # Critical to ensure v has right shape.

class MLPDCritic(nn.Module):

    def __init__(self, obs_dim, d_hidden_sizes, d_ouput, d_activation):
        super().__init__()
        self.v_net = mlp([obs_dim] + list(d_hidden_sizes) + [d_ouput], d_activation)

    def forward(self, obs):
        return torch.squeeze(self.v_net(obs), -1)

class EnsembleStd(nn.Module):
    def __init__(self, num_net, input_dim, hidden_dim, output_dim, activation):
        super(EnsembleStd, self).__init__()
        
        self.models = nn.ModuleList()
        for i in range(num_net):
            self.models.append(nn.Sequential(
                nn.Linear(input_dim, hidden_dim[0]),
                activation(),
                nn.Linear(hidden_dim[0], hidden_dim[1]),
                activation(),
                nn.Linear(hidden_dim[1], output_dim),
            ))
        
    def forward(self, x):
        outputs = []
        for model in self.models:
            outputs.append(model(x))
        outputs = torch.stack(outputs).mean(dim=0)
        return torch.squeeze(outputs, -1)

    def forward_idx(self, x, i):
        output = torch.squeeze(self.models[i](x), -1)
        return output

class EnsembleDV(nn.Module):
    def __init__(self, num_net, input_dim, hidden_dim, output_dim, activation):
        super(EnsembleDV, self).__init__()
        self.num_net = num_net
        self.models = nn.ModuleList()
        for i in range(num_net):
            self.models.append(nn.Sequential(
                nn.Linear(input_dim, hidden_dim[0]),
                activation(),
                nn.Linear(hidden_dim[0], hidden_dim[1]),
                activation(),
                nn.Linear(hidden_dim[1], output_dim),
            ))
        
    def forward(self, x):
        outputs = []
        for model in self.models:
            outputs.append(model(x).mean(dim=-1))
        outputs = torch.stack(outputs).mean(dim=0)
        return torch.squeeze(outputs, -1)

    def get_d_v_bars(self, x):
        outputs = []
        for model in self.models:
            outputs.append(model(x))
        return torch.stack(outputs).detach().numpy()

    def forward_idx(self, x, i):
        output = torch.squeeze(self.models[i](x), -1)
        return output

class MLPRNDCritic(nn.Module):

    def __init__(self, obs_dim, d_hidden_sizes, d_ouput, d_activation):
        super().__init__()

        d_v_layers_size = [obs_dim] + list(d_hidden_sizes) +  [d_ouput]
        d_v_layers_act = [d_activation] * len(d_hidden_sizes) + [nn.Identity]
        d_v_layers = []
        for j in range(len(d_v_layers_size)-1):
            d_v_layers += [nn.Linear(d_v_layers_size[j], d_v_layers_size[j+1]), d_v_layers_act[j]()]
        self.back_d_v = nn.Sequential(*d_v_layers)

    def forward(self, obs):    
        return torch.squeeze(self.back_d_v(obs), -1)


class MLPActorCritic(nn.Module):


    def __init__(self, observation_space, action_space, num_net,
                 hidden_sizes=(64,64), activation=nn.Tanh, d_h_size=(512, 512), std_h_size=(128,128),d_ouput=8, d_activation=nn.ReLU):
        super().__init__()
        self.d_ouput = d_ouput
        obs_dim = observation_space.shape[0]

        # policy builder depends on action space
        if isinstance(action_space, Box):
            self.pi = MLPGaussianActor(obs_dim, action_space.shape[0], hidden_sizes, activation)
        elif isinstance(action_space, Discrete):
            self.pi = MLPCategoricalActor(obs_dim, action_space.n, hidden_sizes, activation)

        # build distributional value function V^{D, \pi}
        # self.d_v  = MLPRNDCritic(obs_dim, d_h_size, d_ouput, d_activation)
        self.ensemble_d_v = EnsembleDV(num_net, obs_dim, d_h_size, d_ouput, d_activation)
        self.ensemble_std = EnsembleStd(num_net, obs_dim, std_h_size, 1, d_activation)

    def step(self, obs, training=True):
        with torch.no_grad():
            if training:
                pi = self.pi._distribution(obs)
                a = pi.sample()
                logp_a = self.pi._log_prob_from_distribution(pi, a)
                v = self.ensemble_d_v(obs)
            else:
                return self.pi._get_mean(obs).numpy(), None, None
        return a.numpy(), np.mean(v.numpy()), logp_a.numpy()

    def act(self, obs):
        return self.step(obs)[0]

    # def get_std(self, obs):
    #     return self.d_v.forward_std(obs)

    def order_check(self, obs, num_net):
        d_vs = self.ensemble_d_v.get_d_v_bars(obs)
        if np.any(np.isnan(d_vs)):
            return 0.0

        orderd = 0
        for d_v in d_vs:
            flag = True
            for i in range(len(d_v) - 1):
                if d_v[i] >= d_v[i+1]:
                    flag = False
            if flag:
                orderd += 1

            flag_rev = True
            for i in range(len(d_v) - 1):
                if d_v[i] <= d_v[i+1]:
                    flag_rev = False

            if flag_rev:
                orderd += 1
        
        return min(orderd, num_net)
