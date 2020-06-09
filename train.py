import itertools
import os
from os import path as pt
from tqdm import tqdm

import matplotlib.pyplot as plt
import numpy as np
import torch

from sig_lib.baselines import train_baseline
from sig_lib.sig_conditional import train_SigCGAN
from sig_lib.tools import pickle_it


def savefig(path):
    plt.savefig(path)
    plt.close()


def download_man_ahl_dataset():
    import requests
    from zipfile import ZipFile
    url = 'https://realized.oxford-man.ox.ac.uk/images/oxfordmanrealizedvolatilityindices.zip'
    r = requests.get(url)
    with open('./oxford.zip', 'wb') as f:
        pbar = tqdm( unit="B", total=int( r.headers['Content-Length'] ) )
        for chunk in r.iter_content(chunk_size=100*1024):
            if chunk:
                pbar.update(len(chunk))
                f.write(r.content)
    zf = ZipFile('./oxford.zip')
    zf.extractall(path='./data')
    zf.close()
    os.remove('./oxford.zip')


def create_and_run_experiment(experiment_directory, algo, use_cuda=True, **kwargs):
    """

    Args:
        experiment_directory: path to directory where generator weights and training summaries shall be saved
        baseline: whether to train a baseline model
        use_cuda: bool, if True run code code on GPU, else on CPU
        **kwargs: HYPERPARAMETERS used for training

    Returns:

    """
    if not os.path.exists('./data/oxfordmanrealizedvolatilityindices.csv'):
        print('Downloading Oxford MAN AHL realised library...')
        download_man_ahl_dataset()

    if not os.path.exists(experiment_directory):
        # if the experiment directory does not exist we create the directory
        os.makedirs(experiment_directory)
    if use_cuda:
        # if use_cuda is True we set torch.cuda.FloatTensor as the default tensor
        torch.set_default_tensor_type(torch.cuda.FloatTensor)
        device = 'cuda'
    else:
        device = 'cpu'
    # pickle all HYPERPARAMETERS
    pickle_it(kwargs, pt.join(experiment_directory, 'hyperparameters.dict'))
    # Set seed for exact reproducibility of the experiments
    torch.manual_seed(0)
    np.random.seed(0)
    # Train the algorithm
    train_foo = train_baseline if algo in ['RCGAN', 'TimeGAN', 'GMMN'] else train_SigCGAN
    G, training_loss, x_real, aux_loss_list = train_foo(experiment_directory=experiment_directory, device=device,
                                                        **kwargs)
    # Pickle generator weights and real path.
    pickle_it(training_loss, pt.join(experiment_directory, 'training_loss.pkl'))
    pickle_it(x_real, pt.join(experiment_directory, 'x_real.torch'))
    pickle_it(G.state_dict(), pt.join(experiment_directory, 'G_weights.torch'))
    del G
    del x_real
    # Log some results at the end of training
    fig, axes = plt.subplots(len(aux_loss_list) + 1, 1, figsize=(10, 8))
    if algo in ['TimeGAN', 'RCGAN']:
        axes[0].plot(training_loss['G_loss'], label='G_loss')
        axes[0].plot(training_loss['D_loss'], label='D_loss')
    elif algo == 'GMMN':
        axes[0].plot(training_loss['MMD'], label='MMD')
    elif algo == 'SigCGAN':
        axes[0].plot(training_loss['loss'], label='sig_w1_loss')
        axes[0].set_yscale('log')
    else:
        raise NotImplementedError()
    axes[0].grid()
    axes[0].legend()
    for i, aux_loss in enumerate(aux_loss_list):
        axes[i + 1].plot(training_loss[aux_loss.name], label=aux_loss.name)
        axes[i + 1].grid()
        axes[i + 1].legend()
        if i + 1 == len(aux_loss_list):
            axes[i + 1].set_xlabel('Generator weight updates')
    savefig(pt.join(experiment_directory, 'losses.png'))


def main(numerical_root='./numerical_results', algos=('SigCGAN', 'GMMN', 'RCGAN', 'TimeGAN'), use_cuda=True):
    """
    Main procedure to run the full benchmark, including SigCGAN and baseline models TimeGAN and RCGAN.

    Args:
        numerical_root:
        algos:

    Returns:
    """
    print('Start of training. CUDA: %s' % use_cuda)
    for algo in algos:
        # Stocks benchmark
        for asset in [('SPX',), ('SPX', 'DJI')]:
            create_and_run_experiment(
                experiment_directory=os.path.join(numerical_root, 'stocks', algo, '_'.join(asset)),
                algo=algo,
                data_type='stocks',
                data_params=dict(assets=asset, with_vol=True),
                latent_dim=2 * len(asset),
                use_cuda=use_cuda,
                **STOCKS_DEFAULTS.copy(),
                **HYPERPARAMETERS[algo]['_'.join(asset)].copy()
            )
        # VAR benchmark
        for dim, (phi, sigma) in itertools.product([1], [(0.2, 0.8), (0.5, 0.8), (0.8, 0.8)]):
            print('Algo: %s. Dataset: VAR(1). Dimension: %s. Parameters: phi=%s.' % (algo, dim, phi))
            create_and_run_experiment(
                experiment_directory=os.path.join(numerical_root, 'VAR(1)_dim=%s' % dim, algo, 'phi=%s' % (phi)),
                algo=algo,
                data_type='VAR',
                dim=dim,
                latent_dim=dim,
                data_params=dict(phi=phi, sigma=sigma),
                use_cuda=use_cuda,
                **VAR_DEFAULTS.copy(),
                **HYPERPARAMETERS[algo]['VAR'].copy()
            )
        for dim, (phi, sigma) in itertools.product(
                [2, 3], [(0.2, 0.8), (0.5, 0.8), (0.8, 0.8), (0.8, 0.2), (0.8, 0.5), ]
        ):
            print(
                'Algo: %s. Dataset: VAR(1). Dimension: %s. Parameters: phi=%s, sigma=%s.' % (algo, dim, phi, sigma))
            create_and_run_experiment(
                experiment_directory=os.path.join(numerical_root, 'VAR(1)_dim=%s' % dim, algo, 'phi=%s_sigma=%s' % (phi, sigma)),
                algo=algo,
                data_type='VAR',
                dim=dim,
                latent_dim=dim,
                data_params=dict(phi=phi, sigma=sigma),
                use_cuda=use_cuda,
                **VAR_DEFAULTS.copy(),
                **HYPERPARAMETERS[algo]['VAR'].copy()
            )
        # ARCH benchmark
        for lag in itertools.product([2, 3, 4]):
            print('Algo: %s. Dataset: ARCH. Lag: %s. ' % (algo, lag))
            create_and_run_experiment(
                experiment_directory=os.path.join(numerical_root, 'ARCH_lag=%s' % lag, algo),
                algo=algo,
                data_type='ARCH',
                latent_dim=1,
                data_params=dict(lag=lag),
                **ARCH_DEFAULTS.copy(),
                **HYPERPARAMETERS[algo]['ARCH'].copy()
            )


VAR_DEFAULTS = dict(p=3, q=3, hidden_dims=3 * [50], total_steps=1000, batch_size=200, )
STOCKS_DEFAULTS = dict(p=3, q=3, hidden_dims=3 * [50], total_steps=1000, batch_size=400, )
ARCH_DEFAULTS = dict(p=3, q=3, hidden_dims=3 * [50], total_steps=1000, batch_size=200, )


SigCGAN = dict(
    VAR=dict(
        lr=1e-3,
        sig_degree_p=2,
        sig_degree_q=2,
        scalar_p=0.5,
        scalar_q=0.5,
        mc_size=1500,
        sig_params_p=dict(
            m=2, with_lift=True, with_concat=True, with_time=False,
        ),
        sig_params_q=dict(
            m=2, with_lift=True, with_concat=True, with_time=False,
        )
    ),
    SPX=dict(
        lr=1e-3,
        sig_degree_p=2,
        sig_degree_q=2,
        scalar_p=0.4,
        scalar_q=0.4,
        mc_size=600,
        sig_params_p=dict(
            m=2, with_lift=True, with_concat=True, with_time=False,
        ),
        sig_params_q=dict(
            m=2, with_lift=True, with_concat=True, with_time=False,
        )
    ),
    SPX_DJI=dict(
        lr=1e-3,
        sig_degree_p=2,
        sig_degree_q=2,
        scalar_p=0.8,
        scalar_q=0.8,
        mc_size=500,
        sig_params_p=dict(
            m=2, with_lift=True, with_concat=True, with_time=False,
        ),
        sig_params_q=dict(
            m=2, with_lift=True, with_concat=True, with_time=False,
        )
    ),
    ARCH=dict(
        lr=1e-3,
        sig_degree_p=2,
        sig_degree_q=2,
        scalar_p=0.2,
        scalar_q=0.2,
        mc_size=600,
        sig_params_p=dict(
            m=2, with_lift=True, with_concat=True, with_time=False,
        ),
        sig_params_q=dict(
            m=2, with_lift=True, with_concat=True, with_time=False,
        )
    )
)


RCGAN = dict(
    VAR=dict(
        gan_algo='RCGAN',
        hidden_dims_D=3 * [50, ]
    ),
    SPX=dict(
        gan_algo='RCGAN',
        hidden_dims_D=3 * [50, ]
    ),
    SPX_DJI=dict(
        gan_algo='RCGAN',
        hidden_dims_D=3 * [50, ]
    ),
    ARCH=dict(
        gan_algo='RCGAN',
        hidden_dims_D=3 * [50, ]
    )
)


TimeGAN = dict(
    VAR=dict(
        gan_algo='TimeGAN',
        hidden_dims_D=3 * [50, ]
    ),
    SPX=dict(
        gan_algo='TimeGAN',
        hidden_dims_D=3 * [50, ]
    ),
    SPX_DJI=dict(
        gan_algo='TimeGAN',
        hidden_dims_D=3 * [50, ]
    ),
    ARCH=dict(
        gan_algo='TimeGAN',
        hidden_dims_D=3 * [50, ]
    )
)


GMMN = dict(
    VAR=dict(
        gan_algo='GMMN',
        hidden_dims_D=3 * [50, ]
    ),
    SPX=dict(
        gan_algo='GMMN',
        hidden_dims_D=3 * [50, ]
    ),
    SPX_DJI=dict(
        gan_algo='GMMN',
        hidden_dims_D=3 * [50, ]
    ),
    ARCH=dict(
        gan_algo='GMMN',
        hidden_dims_D=3 * [50, ]
    )
)


HYPERPARAMETERS = dict(RCGAN=RCGAN, SigCGAN=SigCGAN, TimeGAN=TimeGAN, GMMN=GMMN)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Turn CUDA off / on during training. ')
    parser.add_argument('-use_cuda', action='store_true')
    args = parser.parse_args()
    main('./numerical_results/', use_cuda=args.use_cuda)
