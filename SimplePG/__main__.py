import os
import torch
import argparse

import torch
import numpy as np
import pandas as pd
from torch.optim import Adam

import gym
from gym.spaces import Discrete, Box
from SimplePG.Actor import Actor
from SimplePG.train import train_one_epoch
from SimplePG.run import run_policy
import pickle

if __name__ == "__main__":
    # TODO : if time change to args
    hidden_size = 32

    output_path_model = "./models"
    output_results = "./results"
    use_cuda = True

    # Get device to use GPU if available and use_cuda =True
    device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")

    parser = argparse.ArgumentParser(description="Parameters")
    parser.add_argument(
        "-e",
        "--epochs",
        type=int,
        default=100,
        metavar="N",
        help="Batch size for training and scoring(default: 64)",
    )
    parser.add_argument(
        "-b",
        "--batch_size",
        type=int,
        default=5000,
        metavar="N",
        help="Batch size for training and scoring(default: 64)",
    )
    parser.add_argument(
        "-lr",
        "--lr",
        type=int,
        default=1e-2,
        metavar="N",
        help="Batch size for training and scoring(default: 64)",
    )
    parser.add_argument(
        "-se",
        "--save_epoch",
        type=int,
        default=10,
        metavar="N",
        help="Saving model every N epoch",
    )
    parser.add_argument(
        "--max_len",
        type=int,
        default=1000,
        metavar="N",
        help="Saving model every N epoch",
    )
    parser.add_argument("--norender", "-nr", action="store_true")
    parser.add_argument(
        "--train", default=False, action="store_true", help="Flag to train"
    )

    parser.add_argument("--env", default="CartPole-v0")
    args = parser.parse_args()
    name_env = args.env  # CartPole-v0"

    # Create Environment and get obs dim and nb of actions
    print(name_env)
    env = gym.make(name_env).env
    obs_dim = env.observation_space.shape[0]
    n_acts = env.action_space.n

    ## TODO : check env type and raise error if continuous action space
    # Init action network in device
    actor = Actor(device, obs_dim, hidden_size, n_acts).to(device)

    if args.train:
        # TODO : Refactor in train
        print("###### Training the policy ######")
        # Instantiate optimizer
        optimizer = Adam(actor.logits_net.parameters(), lr=args.lr)

        # Training loop
        batch_avg_len = np.zeros(args.epochs)
        batch_std_len = np.zeros(args.epochs)
        batch_avg_return = np.zeros(args.epochs)
        batch_std_return = np.zeros(args.epochs)

        for epoch in range(args.epochs):
            batch_loss, batch_rets, batch_lens = train_one_epoch(
                env,
                actor,
                optimizer,
                output_path_model,
                device,
                epoch,
                render=not (args.norender),
                batch_size=args.batch_size,
                save_epochs=args.save_epoch,
                max_len=args.max_len,
            )
            batch_avg_return[epoch] = np.mean(batch_rets)
            batch_std_return[epoch] = np.std(batch_rets)
            batch_avg_len[epoch] = np.mean(batch_lens)
            batch_std_len[epoch] = np.std(batch_lens)
            print(
                "epoch: %3d \t loss: %.3f \t return: %.3f \t ep_len: %.3f"
                % (epoch, batch_loss, np.mean(batch_rets), np.mean(batch_lens))
            )
        # Save results for visualization and comparison
        with open("./results/simple_pg_results_{}.pkl".format(name_env), "wb") as f:
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
        print("###### Testing with latest policy ######")
        # Get device to use GPU if available and use_cuda =True
        # TODO : refactor this
        use_cuda = False
        device = torch.device(
            "cuda" if (use_cuda and torch.cuda.is_available()) else "cpu"
        )
        actor = Actor(device, obs_dim, hidden_size, n_acts)
        state_dict = torch.load(
            "./models/simple_pg_model_{}.pth".format(name_env), map_location="cpu"
        )
        actor.load_state_dict(state_dict["actor"])
        run_policy(device, env, actor)

