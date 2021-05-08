import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
import numpy as np

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
    
    def _log_prob_from_distribution(self,obs, act):
        pi = self._policy(obs)
        return pi.log_prob(act)    
    
    def step(self, obs):
        with torch.no_grad():
            pi = self._policy(obs)
            a = pi.sample()#.item()
        return a.cpu().numpy()