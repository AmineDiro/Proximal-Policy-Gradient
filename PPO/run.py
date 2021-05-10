import os
import torch
import time 
import numpy as np

def run_policy(device,env, player, max_ep_len=None, num_episodes=10, render=True):
    o, r, d, ep_ret, ep_len, n = env.reset(), 0, False, 0, 0, 0
    while n < num_episodes:
        if render:
            env.render()
            time.sleep(1e-3)

        obs_cuda = torch.as_tensor(o, dtype=torch.float32, device=device)
        a, v, logp = player.step(
                    torch.as_tensor(o, dtype=torch.float32, device=device)
                )
        o, r, d, _ = env.step(a)
        ep_ret += r
        ep_len += 1

        if d or (ep_len == max_ep_len):
            print("Episode %d \t EpRet %.3f \t EpLen %d" % (n, ep_ret, ep_len))
            o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0
            n += 1
    env.close()
