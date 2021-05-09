import numpy as np
import scipy.signal
from gym.spaces import Box, Discrete

import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from torch.distributions.categorical import Categorical



class Net(nn.Module):
    def __init__(self, obs_dim,hidden_size,n_acts):
        super(Net, self).__init__()
        self.linear1 = nn.Linear(obs_dim,hidden_size)
        self.linear2 = nn.Linear(hidden_size,n_acts)
        
    def forward(self,x):
        x = self.linear1(x)
        x = nn.Tanh()(x)
        x= self.linear2(x)
        return nn.Identity()(x)

        
class Actor(nn.Module):
    def __init__(self,device, obs_dim, hidden_size,act_dim):
        super().__init__()
        self.logits_net = Net(obs_dim,hidden_size,act_dim).to(device)

    def _policy(self, obs):
        logits = self.logits_net(obs)
        # Use categorical to get distribution over actions i.e Plicy
        return Categorical(logits=logits)
    
    @staticmethod
    def _log_prob_from_distribution(pi, act):
        return pi.log_prob(act)
    
class Critic(nn.Module):
    def __init__(self, device, obs_dim, hidden_size):
        super().__init__()
        self.v_net = Net(obs_dim,hidden_size,1).to(device)

    def forward(self, obs):
        return torch.squeeze(self.v_net(obs), -1) # Squeeze to one dimension to ensure v has right shape.


class ActorCritic(nn.Module):
    def __init__(self,device, env, hidden_size=32):
        super().__init__()
        obs_dim = env.observation_space.shape[0]
        n_acts = env.action_space.n
        
        # build policy NN 
        self.pi = Actor(device, obs_dim, hidden_size,n_acts)

        # value function NN 
        self.v  = Critic(device, obs_dim, hidden_size)

    def step(self, obs):
        with torch.no_grad():
            # Get distrib over actions
            pi = self.pi._policy(obs)
            # Sample one action
            a = pi.sample()
            # get log proba over distrib
            logp_a = self.pi._log_prob_from_distribution(pi, a)
            # Get estimated value from Critic
            v = self.v(obs)
        return a.cpu().numpy(), v.cpu().numpy(), logp_a.cpu().numpy()

    def act(self, obs):
        return self.step(obs)[0]