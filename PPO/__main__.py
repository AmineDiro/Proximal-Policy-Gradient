import os
import torch
import argparse
import pickle 

import numpy as np
import gym
from gym.spaces import Box, Discrete

import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from torch.optim import Adam
from torch.distributions.categorical import Categorical

from PPO.ActorCritic import ActorCritic
from PPO.PPOBuffer import Buffer
from PPO.train import train
from PPO.run import run_policy


if __name__ == "__main__":

    use_cuda = True

    # Get device to use GPU if available and use_cuda =True
    device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")

    parser = argparse.ArgumentParser(description="Parameters")
    parser.add_argument("--env", default="CartPole-v0",help="Discrete action type env",)
    parser.add_argument(
        "-e",
        "--epochs",
        type=int,
        default=10,
        metavar="N",
        help="Epochs to run training",
    )
    parser.add_argument(
        "-b",
        "--batch_size",
        type=int,
        default=5000,
        metavar="N",
        help="Batch size for training (N*T)",
    )
    parser.add_argument(
        "-se",
        "--save_epochs",
        type=int,
        default=10,
        metavar="N",
        help="Saving model every N epoch",
    )
    parser.add_argument(
        "--train", default=False, action="store_true", help="Flag to train"
    )

    args = parser.parse_args()
    name_env = args.env  # CartPole-v0"

    ## Hyperparameters
    seed = 0
    gamma = 0.99
    clip_ratio = 0.2
    pi_lr = 3e-4
    vf_lr = 1e-3
    train_pi_iters = 80
    train_v_iters = 80
    lam = 0.97
    max_ep_len = 1000
    target_kl = 0.01
    # Episode sampling
    N = 1  # Number of process TODO : multiprocessing
    # steps_per_epoch = 5000  ##  N*T
    T = args.batch_size // N
    output_path_model = "./models"
    output_results = "./results"

    ##  Init env
    env = gym.make(name_env).env  # "CartPole-v0"
    if args.train:
        print("###################### STARTING TRAINING #################")

        #  player : actor critic
        player = ActorCritic(device, env)

        # Set up optimizers for policy and value function
        pi_optimizer = Adam(player.pi.parameters(), lr=pi_lr)
        v_optimizer = Adam(player.v.parameters(), lr=vf_lr)

        # Buffer TODO : set up in parallel with shared buffer (N,T,obs_dim) , (N,T,)
        buffer = Buffer(env, T)

        o, ep_ret, ep_len = env.reset(), 0, 0

        ## REsults
        batch_avg_len = np.zeros(args.epochs)
        batch_std_len = np.zeros(args.epochs)
        batch_avg_return = np.zeros(args.epochs)
        batch_std_return = np.zeros(args.epochs)

        # Main loop: collect experience in env and update policy and value function/log each epoch
        for epoch in range(args.epochs):
            batch_ret = []
            batch_ep_len = []
            for t in range(T):
                a, v, logp = player.step(
                    torch.as_tensor(o, dtype=torch.float32, device=device)
                )

                o, r, d, _ = env.step(a)
                ep_ret += r
                ep_len += 1

                # Save in buffer
                buffer.append(t, o, a, r, v, logp)

                # Two times of termination : done ==True or Time limit termination
                timeout = ep_len == max_ep_len
                terminal = d or timeout
                # Early termination induced by fixed trajectory.
                end_epoch = t == T - 1

                if terminal or end_epoch:
                    # if end_epoch and not(terminal):
                    # print('Warning: trajectory cut off by epoch at %d steps.'%ep_len, flush=True)
                    # TODO : ma3reftch chno ndir hna ?

                    # if trajectory didn't reach terminal state, bootstrap value target to get next value for advantage
                    if timeout or end_epoch:
                        _, v, _ = player.step(
                            torch.as_tensor(o, dtype=torch.float32, device=device)
                        )
                    else:
                        v = 0

                    # Compute  GAE advantages and Reward to go
                    buffer.finish(v)

                    ## Save and  Reset for next epoch
                    batch_ret.append(ep_ret)
                    batch_ep_len.append(ep_len)
                    o, ep_ret, ep_len = env.reset(), 0, 0

            # Save mean rturns and std and mean lengths and std
            batch_avg_return[epoch] = np.mean(batch_ret)
            batch_std_return[epoch] = np.std(batch_ret)
            batch_avg_len[epoch] = np.mean(batch_ep_len)
            batch_std_len[epoch] = np.std(batch_ep_len)

            print(
                "epoch: %3d \t return: %.3f \t ep_len: %.3f"
                % (epoch, np.mean(batch_ret), np.mean(batch_ep_len))
            )
            if epoch % args.save_epochs == 0:
                torch.save(
                    {
                        "actorcritic": player.state_dict(),
                        "pi_optimizer": pi_optimizer.state_dict(),
                        "v_optimizer": v_optimizer.state_dict(),
                    },
                    os.path.join(
                        output_path_model, "ppo_model_{}.pth".format(name_env)
                    ),
                )

            # Perform PPO update
            train(device, buffer, player, pi_optimizer, v_optimizer)

        with open("./results/ppo_results_{}.pkl".format(name_env), "wb") as f:
            pickle.dump(
                {
                    "lens": batch_avg_len,
                    "std_lens": batch_std_len,
                    "returns": batch_avg_return,
                    "std_return": batch_std_return,
                },
                f,
            )
    else:
        print("###################### Evaluating model #################")
        # Get device to use GPU if available and use_cuda =True
        # TODO : refactor this
        use_cuda = False
        device = torch.device(
            "cuda" if (use_cuda and torch.cuda.is_available()) else "cpu"
        )
        player = ActorCritic(device, env)
        state_dict = torch.load(
            "./models/ppo_model_{}.pth".format(name_env), map_location="cpu"
        )
        player.load_state_dict(state_dict["actorcritic"])
        run_policy(device, env, player)

