import os
import argparse
import torch
import numpy as np
from torch.optim import Adam
import gym
from gym.spaces import Discrete, Box
from SimplePG.Actor import Actor
from SimplePG.train import train_one_epoch
import pickle

if __name__ == "__main__":
    # TODO : if time change to args
    hidden_size = 32
    lr = 1e-2
    epochs = 100
    batch_size = 5000
    use_cuda = True
    output_path_model = "./models"
    output_results = "./results"

    # Get device to use GPU if available and use_cuda =True
    device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")
    # Create Environment and get obs dim and nb of actions
    env = gym.make("CartPole-v0")
    obs_dim = env.observation_space.shape[0]
    n_acts = env.action_space.n

    # Init action network in device
    actor = Actor(device, obs_dim, hidden_size, n_acts).to(device)

    # Instantiate optimizer
    optimizer = Adam(actor.logits_net.parameters(), lr=lr)

    # Training loop
    batch_avg_len = []
    batch_avg_return = []

    for epoch in range(epochs):
        batch_loss, batch_rets, batch_lens = train_one_epoch(
            env,
            actor,
            optimizer,
            output_path_model,
            device,
            epoch,
            render=True,
            batch_size=5000,
            save_epochs=10,
        )
        batch_avg_len.append(np.mean(batch_lens))
        batch_avg_return.append(np.mean(batch_avg_return))
        print(
            "epoch: %3d \t loss: %.3f \t return: %.3f \t ep_len: %.3f"
            % (epoch, batch_loss, np.mean(batch_rets), np.mean(batch_lens))
        )

    # Save results for visualization and comparison
    with open("simple_pg_results.pkl", "wb") as f:
        pickle.dump({"avg_len": batch_avg_len, "avg_return": batch_avg_return},f)

