
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1DE_1Sv6RIbQuK3Pl8EQh4G9OI9JJRozL?usp=sharing)

# Implementation of Proximal Policy paper

- PPO is a policy gradient method for reinforcement learning.
- PPO is motivated by two challenge:
    - **reducing the sample estimate variance by implementing a modified version of the GAE algorithm**
    - **taking the biggest possible improvement step on a policy using the data we currently have, without stepping so far that we accidentally cause performance collapse**
- PPO lets us do **multiple gradient updates per sample** by trying to **keep the policy close to the policy that was used to sample data.** It does so by **clipping gradient flow if the updated policy is not close to the policy used to sample the data.**

- References to PPO : 
    * [Proximal Policy ](https://arxiv.org/abs/1707.06347), Schulman et al. 2017.
    * [Trust Region Policy Optimization](https://arxiv.org/abs/1502.05477), Schulman et al. 2015.
    * [High Dimensional Continuous Control Using Generalized Advantage Estimation](https://arxiv.org/abs/1506.02438), Schulman et al. 2016.


# Training / Testing


Project Organization
------------

`PPO_Project` directory is structured as follows  


    ├── models                  <- Saved pytorch models, loaded when testing 
    └── SimplePG                <- Simple Policy Gradient 
        ├── Actor.py            <- Policy architecture, 2 layer NN
        ├── run.py              <- Used when testing trained model
        └── train.py            <- Code for updating policy
    ├── PPO
    │   ├── ActorCritic.py      <- Policy and Value function architecture, 2 layer NN
    │   ├── PPOBuffer.py        <- Buffer class needed to store obs,ac,pi and compute advantage
    │   ├── run.py              <- Used when testing trained model
    │   └── train.py            <- Code for updating policy
    ├── ppo.yaml                <- Miniconda env dependencies
    ├── results                 <- Directory for results
    ├── setup.py                <- Run to setup environement



To run the training follow these steps :
1. Clone the repository and cd to the directory
    ```bash
    git clone https://github.com/AmineDiro/Proximal-Policy-Gradient.git 
    cd ./Proximal-Policy-Gradient
    ```
2. Create conda env
    ```bash
    conda env create -f environment.yml
    conda activate test
    ```
3. The training has different arguments , for running ppo use command `python -m PPO --train`, you can choose from the list of arguments below, some are only available for Simple Gradient Policy 

4. You can also test pretrained  `SPG` or `PPO` algorithms by running  `python -m PPO --env `, followed by  the name of the environement : `CartPole-v0` or `LunarLander-v2`


## Arguments 

| Short     | Long         | Description                                                | Default       | PPO                | SPG                |
|-----------|--------------|------------------------------------------------------------|---------------|--------------------|--------------------|
| --env     |              | Discrete action type environement                          | "CartPole-v0" | :heavy_check_mark: | :heavy_check_mark: |
| -e        | --epochs     | Epochs to run training                                     | 5000          | :heavy_check_mark: | :heavy_check_mark: |
| -b        | --batch_size | Batch size for training (N*T)                              | 2             | :heavy_check_mark: | :heavy_check_mark: |
| -se       | --save_epoch | Saving model every N epoch                                 | 10            | :heavy_check_mark: | :heavy_check_mark: |
| --train   |              | Put this Flag to train model                               | False         | :heavy_check_mark: | :heavy_check_mark: |
| -r        | --render     | Put this flag to avoid visualizing first epoch of training | False         | :white_check_mark: | :heavy_check_mark: |
| --max_len |              | Max episode length                                         | 1000          | :white_check_mark: | :heavy_check_mark: |
| --lr      |              | Learning rate default 1e-2                                 | 1e-2          | :white_check_mark: | :heavy_check_mark: |
