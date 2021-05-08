import torch
import torch.nn as nn
from SimplePG.Actor import Actor
from torch.optim import Adam
import numpy as np
import gym
from gym.spaces import Discrete, Box


# One epoch training 
def train_one_epoch(env,actor,optimizer,output_path_model,device):
    # make some empty lists for logging.
    batch_obs = []          # for observations
    batch_acts = []         # for actions
    batch_weights = []      # for R(tau) weighting in policy gradient
    batch_rets = []         # for measuring episode returns
    batch_lens = []         # for measuring episode lengths

    # reset episode-specific variables
    obs = env.reset()       # first obs comes from starting distribution
    done = False            # signal from environment that episode is over
    ep_rews = []            # list for rewards accrued throughout ep

    # render first episode of each epoch
    finished_rendering_this_epoch = False

    # collect experience by acting in the environment with current policy
    while True:
        # rendering
        if (not finished_rendering_this_epoch) and render:
            env.render()

        # save obs
        batch_obs.append(obs.copy())

        # act in the environment
        obs_cuda= torch.as_tensor(obs, dtype=torch.float32,device=device)
        act = actor.step(obs_cuda)
        # Step then get state, reward and if done
        obs, rew, done, _ = env.step(act)

        # save action, reward
        batch_acts.append(act.item())
        ep_rews.append(rew)

        if done:
            # if episode is over, record info about episode
            ep_ret, ep_len = sum(ep_rews), len(ep_rews)
            batch_rets.append(ep_ret)
            batch_lens.append(ep_len)

            # the weight for each logprob(a|s) == R(episode)
            batch_weights += [ep_ret] * ep_len

            # reset episode-specific variables
            obs, done, ep_rews = env.reset(), False, []

            # won't render again this epoch
            finished_rendering_this_epoch = True

            # end experience loop if we have enough of it
            if len(batch_obs) > batch_size:
                break
        env.close()

    # take a single policy gradient update step
    optimizer.zero_grad()
    batch_obs = torch.as_tensor(batch_obs, dtype=torch.float32,device =device)
    batch_acts = torch.as_tensor(batch_acts, dtype=torch.float32,device=device)
    weights = torch.as_tensor(batch_weights, dtype=torch.float32,device =device)

    batch_loss = compute_loss(obs=batch_obs,
                              act=batch_acts,
                              weights=weights
                              )
    batch_loss.backward()
    optimizer.step()
    
    # Save model ever N epochs
    if (epoch % save_epochs == 0):
        torch.save(
                {
                    "actor": actor.state_dict(),
                    "optimizer": optimizer.state_dict(),
                },
                os.path.join(output_path_model, "simple_pg_model.pth"))
            
    return batch_loss, batch_rets, batch_lens