# The authors' official PyTorch SigCWGAN implementation.

This repository is the official implementation of [Conditional Sig-Wasserstein GANs for Time Series Generation]

Authors:

Paper Link:

## Requirements

To setup the conda enviroment:

```setup
conda env create -f requirements.yml
```

## Datasets

This repository contains implementations of synthetic and empirical datasets.

- Synthetic:
    - Vector autoregressive (VAR) data
    - Autoregressive conditionally heteroscedastic (ARCH)
- Real-world data:
    - Stock data: https://realized.oxford-man.ox.ac.uk/data

## Baselines

We compare our SigCGAN with several baselines including: TimeGAN, RCGAN, GMMN(GAN with MMD). The baselines functions are in sig_lib/baselines.py


## Training

To reproduce the numerical results in the paper, save weights and produce a training summaries, run the following line:

```train
python train.py -use_cuda -total_steps 1000
```
Optionally drop the flag ```-use_cuda``` to run the experiments on CPU.


## Evaluation

To evaluate models on different metrics and GPU, run:

```eval
python evaluate.py -use_cuda
```
As above, optionally drop the flag ```-use_cuda``` to run the evaluation on CPU.

## Numerical Results

The numerical results will be saved in the 'numerical_results' folder during training process. Running evaluate.py will produce the 'summary.csv' files.
