
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1rFNs_dYyU_gTCL216TvN8WTFCRAlKmHR?usp=sharing)

# Implementation of Proximal Policy paper

PPO-Clip doesn’t have a KL-divergence term in the objective and doesn’t have a constraint at all. Instead relies on specialized clipping in the objective function to remove incentives for the new policy to get far from the old policy.
* [Proximal Policy ](https://arxiv.org/abs/1707.06347), Schulman et al. 2017.
* [Trust Region Policy Optimization](https://arxiv.org/abs/1502.05477), Schulman et al. 2015.
* [High Dimensional Continuous Control Using Generalized Advantage Estimation](https://arxiv.org/abs/1506.02438), Schulman et al. 2016.


# Training / Testing

To test .... 


Project Organization
------------

`PPO_Project` directory is structured as follows  

   
    ├── OTGAN
    │   ├── dataset.py          <- Loading the two batch datasets : MNIST or CIFAR10
    │   ├── Discriminator.py    <- Critic architecture (1 or 3 input channels)
    │   ├── Generator.py        <- Generator architecture  (1 or 3 output channels)
    │   ├── InceptionScore.py   <- Returns inception score for batchs or for dataset
    │   ├── __init__.py
    │   ├── __main__.py
    │   ├── SinkhornDistance.py <- Computes the Minibatch Energy distance
    │   ├── train.py            <- Main training loop
    │   └── utils.py            <- Weight init, plotting and sampling



To run the training follow these steps :
1. Clone the repository and cd to the directory
    ```bash
    git clone https://github.com/AmineDiro/OT-GAN.git && cd OT-GAN/
    ```
2. The training has different arguments , run  the command `python -m OTGAN` with the proper arguments : 

    | Short                | Long         | Description                                                       | Default |
    |----------------------|--------------|-------------------------------------------------------------------|---------|
    | -c                   | --channels   | Nb of channels 1 for MNIST,3 for CIFAR , 3 by default             | 3       |
    | -b                   | --batch_size | Batch size for training (default: 64)                             | 64      |
    | -se                  | --save_epoch | Saving model every N epoch                                        | 2       |
    | -si                  | --sample_interval| Interval number for sampling image from generator and saving them | 1       |
    | --score / --no-score |              | Boolean args to get Inception score or not                        | True        | 

**NOTE :** The Notebook `Results.ipynb`  presents the main results from training on the CIFAR10 dataset. We plot the generated images from training, the loss of generator, critic and the inception score while training. You can click on the **[Open In Colab]** to access the notebook on google collab.

## TODO : 
- implement ppo buffer
- Implement training : 
    - GAE loss 
    - Figure out with 1 
    - Do parallel processing using MPI or Cuda ?? :O
    -  
