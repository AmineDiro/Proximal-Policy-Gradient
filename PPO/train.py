import numpy as np
import scipy.signal

import torch
import torch.nn as nn


# Set up function for computing PPO policy loss
def _policy_loss(device, data, player, clip_ratio):
    obs, act, adv, logp_old = data["obs"], data["act"], data["adv"], data["logp"]
    logp_old = torch.as_tensor(logp_old, device=device)
    adv = torch.as_tensor(adv, device=device)
    obs_torch = torch.as_tensor(obs, dtype=torch.float32, device=device)
    act_torch = torch.as_tensor(act, dtype=torch.int32, device=device)
    # Policy loss
    pi, logp = player.pi(obs_torch, act_torch)
    ratio = torch.exp(logp - logp_old)
    clip_adv = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio) * adv
    loss_pi = -(torch.min(ratio * adv, clip_adv)).mean()

    # NOTE : Important Info
    approx_kl = (logp_old - logp).mean().item()
    ent = pi.entropy().mean().item()

    clipped = ratio.gt(1 + clip_ratio) | ratio.lt(1 - clip_ratio)
    clipfrac = torch.as_tensor(clipped, dtype=torch.float32).mean().item()
    pi_info = dict(kl=approx_kl, ent=ent, cf=clipfrac)
    return loss_pi, pi_info


# Set up function for computing value loss
def _v_loss(device, data, player):
    obs, ret = data["obs"], data["ret"]
    obs_torch = torch.as_tensor(obs, dtype=torch.float32, device=device)
    ret_torch = torch.as_tensor(ret, dtype=torch.int32, device=device)
    return ((player.v(obs_torch) - ret_torch) ** 2).mean()


def train(
    device,
    buffer,
    player,
    pi_optimizer,
    v_optimizer,
    clip_ratio=0.2,
    target_kl=0.01,
    train_pi_iters=80,
    train_v_iters=80
):
    data = buffer.get()

    # Train policy with multiple steps of gradient descent
    for i in range(train_pi_iters):
        pi_optimizer.zero_grad()
        loss_pi, pi_info = _policy_loss(device, data, player, clip_ratio)

        # kl = mpi_avg(pi_info['kl'])
        # kl = np.mean(pi_info['kl'])
        #kl = pi_info['kl']
        # if kl > 1.5 * target_kl:
        #      print('Early stopping at step %d due to reaching max kl.'%i)
        #      break
        # loss_pi.backward()
        # mpi_avg_grads(ac.pi)    # average grads across MPI processes
        pi_optimizer.step()

    # Value function learning
    for i in range(train_v_iters):
        v_optimizer.zero_grad()
        loss_v = _v_loss(device, data, player)
        loss_v.backward()
        # mpi_avg_grads(ac.v)    # average grads across MPI processes
        v_optimizer.step()
