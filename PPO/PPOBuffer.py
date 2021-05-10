import numpy as np
import scipy.signal

import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from torch.optim import Adam
from torch.distributions.categorical import Categorical

class Buffer:
    def __init__(self, env, T, gamma=0.99, lam=0.95):     
        obs_dim = int(env.observation_space.shape[0])
        # TODO : add raise errer if env type != Sample
        act_shape = env.action_space.shape
        if np.isscalar(act_shape) :
            batch_action_shape = (T, act_shape) 
        else : 
            batch_action_shape =  (T, *act_shape)
         
        self.obs_buf = np.zeros((T, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(batch_action_shape, dtype=np.float32)
        self.adv_buf = np.zeros(T, dtype=np.float32)
        self.rew_buf = np.zeros(T, dtype=np.float32)
        self.ret_buf = np.zeros(T, dtype=np.float32)
        self.val_buf = np.zeros(T, dtype=np.float32)
        self.logp_buf = np.zeros(T, dtype=np.float32)
        
        # Params for GAE estimation
        self.gamma, self.lam = gamma, lam
        # Pointer : useful in same trajectory to get two 
        self.ptr, self.path_start_idx, self.max_size = 0, 0, T

    def append(self, t, obs, act, rew, val, logp):
        """
        Append one timestep of agent-environment interaction to the buffer.
        """
        assert self.ptr < self.max_size   
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.val_buf[self.ptr] = val
        self.logp_buf[self.ptr] = logp
        self.ptr = t

    @staticmethod
    def discount_cumsum(x, discount):
        """ 
        Computes the discounted sum for each timestep 
        TODO : can be written in parrallel for the subprocess
        """
        return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]

    def finish(self, last_val=0):
        # T step trajectory
        path_slice = slice(self.path_start_idx, self.ptr)
        # Append the last bootstraped value 
        rews = np.append(self.rew_buf[path_slice], last_val)
        vals = np.append(self.val_buf[path_slice], last_val)
        
        # Compute one-step TD error
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        # GAE Advantage for the slice
        self.adv_buf[path_slice] = self.discount_cumsum(deltas, self.gamma*self.lam)
        
        # Rewards-to-go : targets for the value function
        self.ret_buf[path_slice] = self.discount_cumsum(rews, self.gamma)[:-1]
        
        # Move index of next trajectory 
        self.path_start_idx = self.ptr

    def get(self):
        self.ptr, self.path_start_idx = 0, 0
        # the next two lines implement the advantage normalization trick
        # TODO : parallel 
        # adv_mean, adv_std = mpi_statistics_scalar(self.adv_buf)
        # self.adv_buf = (self.adv_buf - adv_mean) / adv_std
        # TODO : noramlize across multiple processes
        self.adv_buf = (self.adv_buf - self.adv_buf.mean()) / self.adv_buf.std()
        return dict(obs=self.obs_buf, act=self.act_buf, ret=self.ret_buf,
                    adv=self.adv_buf, logp=self.logp_buf)
