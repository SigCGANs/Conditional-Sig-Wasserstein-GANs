import os

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LinearRegression

from train import HYPERPARAMETERS, VAR_DEFAULTS, STOCKS_DEFAULTS, ARCH_DEFAULTS
from sig_lib.ResFNN import ResFNN
from sig_lib.auxilliary_losses import supervised_losses
from sig_lib.sig_conditional import compute_signature, sample, sigcgan_loss
from sig_lib.sig_conditional import to_numpy
from sig_lib.tools import load_pickle
from sig_lib.plot import plot_var, plot_spot, compare_cross_corr

import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")


def evaluate_generator(model_name,benchmark_directory, experiment_directory, data_type, use_cuda=True):
    """

    Args:
        model_name:
        experiment_directory:
        data_type:
        use_cuda:

    Returns:

    """
    torch.random.manual_seed(0)
    if use_cuda:
        device = 'cuda'
    else:
        device = 'cpu'
    experiment_summary = dict()
    experiment_summary['model_id'] = model_name
    # shorthands
    sig_params_p = HYPERPARAMETERS['SigCGAN'][data_type]['sig_params_p']
    sig_params_q = HYPERPARAMETERS['SigCGAN'][data_type]['sig_params_q']
    if data_type == 'var':
        defaults = VAR_DEFAULTS
    elif data_type == 'stocks':
        defaults = STOCKS_DEFAULTS
    else:
        defaults = ARCH_DEFAULTS
#     defaults = VAR_DEFAULTS if data_type == 'var' else STOCKS_DEFAULTS
    if data_type=='ARCH':
        p, q = int(benchmark_directory.split('=')[-1]), defaults['q']
    else:
        p, q = defaults['p'], defaults['q']
#     p, q = defaults['p'], defaults['q']
    scalar_p = HYPERPARAMETERS['SigCGAN'][data_type]['scalar_p']
    sig_degree_p = HYPERPARAMETERS['SigCGAN'][data_type]['sig_degree_p']
    scalar_q = HYPERPARAMETERS['SigCGAN'][data_type]['scalar_q']
    sig_degree_q = HYPERPARAMETERS['SigCGAN'][data_type]['sig_degree_q']
    plot_comparison = plot_spot if data_type == 'stocks' else plot_var
    # ----------------------------------------------
    # Load and prepare real path.
    # ----------------------------------------------
    x_real = load_pickle(os.path.join(experiment_directory, 'x_real.torch')).to(device)
    x_past = x_real[:, :p]
    x_future = x_real[:, p:p + q]
    dim = latent_dim = x_real.shape[-1]
    # ----------------------------------------------
    # Load generator weights and hyperparameters
    # ----------------------------------------------
    G_weights = load_pickle(os.path.join(experiment_directory, 'G_weights.torch'))
    G = ResFNN(dim * p + latent_dim, dim, 3*[50,]).to(device)
    G.load_state_dict(G_weights)
    # ----------------------------------------------
    # Compute predictive score - TSTR (train on synthetic, test on real)
    # ----------------------------------------------
    with torch.no_grad():
        # generate 1 fake path for each condition
        x_fake = sample(G, torch.randn(x_past.size(0), 1, latent_dim).to(device), x_past)
    size = x_fake.shape[0]
    X = to_numpy(x_past.reshape(size, -1))
    Y = to_numpy(x_fake.reshape(size, -1))
    size = x_real.shape[0]
    X_test = X.copy()
    Y_test = to_numpy(x_future[:, :1].reshape(size, -1))
    model = LinearRegression()
    model.fit(X, Y)  # TSTR
    experiment_summary['r2_tstr'] = model.score(X_test, Y_test)
    model = LinearRegression()
    model.fit(X_test, Y_test)  # TRTR
    experiment_summary['r2_trtr'] = model.score(X_test, Y_test)
    # ----------------------------------------------
    # Compute metrics and scores of the unconditional distribution.
    # ----------------------------------------------
    with torch.no_grad():
        x_fake = sample(G, torch.randn(x_past.size(0), q, latent_dim).to(device), x_past)
    density_metric = supervised_losses['abs_metric'](x_real, reg=0.)
    acf_loss = supervised_losses['acf_id'](x_real, max_lag=2, reg=0.)
    cross_correl = supervised_losses['cross_correl'](x_real)
    acf_loss(x_fake)
    density_metric(x_fake)
    cross_correl(x_fake)
    experiment_summary['acf_id_lag=1'] = np.mean(to_numpy(acf_loss.loss_componentwise))
    experiment_summary['abs_metric'] = np.mean(to_numpy(density_metric.loss_componentwise))
    experiment_summary['cross_correl'] = np.mean(to_numpy(cross_correl.loss_componentwise))
    del x_fake
    # ----------------------------------------------
    # Compute SigW1 distance.
    # ----------------------------------------------
    sigs_past = compute_signature(x_past, scalar_p, sig_degree_p, **sig_params_p)
    sigs_future = compute_signature(x_future, scalar_q, sig_degree_q, **sig_params_q)
    X, Y = sigs_past, sigs_future
    X = to_numpy(X)
    Y = to_numpy(Y)
    from sklearn.linear_model import Ridge
    model = Ridge(alpha=1)
    model.fit(X, Y)
    sigs_pred = torch.from_numpy(model.predict(X)).float().to(device)
    # generate fake paths
    mc_size = 1000  # fixed MC size
    sigs_conditional = list()
    with torch.no_grad():
        steps = 200
        size = x_past.size(0)//steps
        for i in range(steps):
            x_past_sample = x_past[i*size:(i+1)*size] if i < (steps-1) else x_past[i*size:]
            x_past_sample_mc = x_past_sample.repeat(mc_size, 1, 1)
            x_fake_future = sample(G, torch.randn(x_past_sample_mc.size(0), q, latent_dim).to(device), x_past_sample_mc)
            sig_fake_future = compute_signature(x_fake_future, scalar_q, sig_degree_q, **sig_params_q)
            sig_fake_conditional_expectation = sig_fake_future.reshape(mc_size, x_past_sample.size(0), -1).transpose(0, 1).mean(1)
            sigs_conditional.append(sig_fake_conditional_expectation)
        sigs_conditional = torch.cat(sigs_conditional, dim=0)
        sig_w1_metric = sigcgan_loss(sigs_pred, sigs_conditional)
    experiment_summary['sig_w1_metric'] = sig_w1_metric.item()
    # ----------------------------------------------
    # Create the relevant summary plots.
    # ----------------------------------------------
    with torch.no_grad():
        x_p = x_past.clone().repeat(5, 1, 1) if data_type == 'stocks' else x_past.clone()
        x_fake_future = sample(G, torch.randn(x_p.shape[0], q, latent_dim).to(device), x_p.to(device))
        plot_comparison(x_fake=x_fake_future, x_real=x_real, max_lag=q)
    plt.savefig(os.path.join(experiment_directory, 'summary.png'))
    plt.close()
    if dim > 1:
        compare_cross_corr(x_fake=x_fake_future, x_real=x_real)
        plt.savefig(os.path.join(experiment_directory, 'cross_correl.png'))
        plt.close()
    # ----------------------------------------------
    # Generate long paths when VAR(1) dataset is considered.
    # ----------------------------------------------
    if data_type == 'VAR':
        with torch.no_grad():
            x_fake_future = sample(G, torch.randn(1, 8000, latent_dim).to(device), x_past[0:1].to(device))
        plot_comparison(x_fake=x_fake_future, x_real=x_real, max_lag=q)
        plt.savefig(os.path.join(experiment_directory, 'summary_long_path.png'))
        plt.close()
    return experiment_summary


def create_benchmark_specific_dataframe(benchmark_directory):
    if benchmark_directory == 'VAR(1)_dim=1':
        df = pd.DataFrame(
            columns=['model_id', 'phi', 'abs_metric', 'acf_id_lag=1', 'cross_correl', 'sig_w1_metric',
                     'r2_tstr', 'r2_trtr']
        )
    elif benchmark_directory in ['VAR(1)_dim=2', 'VAR(1)_dim=3']:
        df = pd.DataFrame(
            columns=['model_id', 'phi', 'sigma', 'abs_metric', 'acf_id_lag=1', 'cross_correl', 'sig_w1_metric',
                     'r2_tstr', 'r2_trtr']
        )
    else:
        df = pd.DataFrame(
            columns=['model_id', 'asset', 'abs_metric', 'acf_id_lag=1', 'cross_correl', 'sig_w1_metric',
                     'r2_tstr', 'r2_trtr']
        )
    return df


def complete_experiment_summary(benchmark_directory, experiment_directory, experiment_summary):
    if benchmark_directory == 'VAR(1)_dim=1':
        experiment_summary['phi'] = float(experiment_directory.split('_')[0].split('=')[-1])
    elif benchmark_directory in ['VAR(1)_dim=2', 'VAR(1)_dim=3']:
        experiment_summary['phi'] = float(experiment_directory.split('_')[0].split('=')[-1])
        experiment_summary['sigma'] = float(experiment_directory.split('_')[1].split('=')[-1])
    elif benchmark_directory in ['ARCH_lag=2', 'ARCH_lag=3','ARCH_lag=4']:
        experiment_summary['lag'] = int(benchmark_directory.split('=')[-1])
    else:
        experiment_summary['asset'] = experiment_directory
    return experiment_summary


def evaluate_benchmarks(use_cuda=False):
    msg = 'Running evalution on GPU.' if use_cuda else 'Running evalution on CPU.'
    print(msg)
    root = './numerical_results'
    for benchmark_directory in os.listdir(root):
        if benchmark_directory.startswith('VAR'):
            data_type = 'VAR'
        elif benchmark_directory.startswith('stocks'):
            data_type = 'stocks'
        elif benchmark_directory.startswith('ARCH'):
            data_type = 'ARCH'
        else:
            raise NotImplementedError()
        df_dst_path = os.path.join(root, benchmark_directory, 'summary.csv')
        df = create_benchmark_specific_dataframe(benchmark_directory)
        benchmark_path = os.path.join(root, benchmark_directory)
        algo_directories = [
            directory for directory in os.listdir(benchmark_path) if
            os.path.isdir(os.path.join(benchmark_path, directory))
        ]
        for algo_directory in algo_directories:
            algo_path = os.path.join(benchmark_path, algo_directory)
            if data_type!='ARCH':
                for experiment_directory in os.listdir(algo_path):
                    experiment_path = os.path.join(algo_path, experiment_directory)
                    print(algo_directory + '_' + benchmark_directory + '_' + experiment_directory)
                    # evaluate the generator
                    experiment_summary = evaluate_generator(
                        model_name=algo_directory,
                        benchmark_directory=benchmark_directory,
                        experiment_directory=experiment_path,
                        data_type=experiment_directory[:7] if data_type == 'stocks' else data_type,
                        use_cuda=use_cuda
                    )
                    # add relevant parameters used during training to the experiment summary
                    experiment_summary = complete_experiment_summary(
                        benchmark_directory, experiment_directory, experiment_summary
                    )
                    df = df.append(experiment_summary, ignore_index=True, )
            else:
                experiment_path = algo_path
                experiment_summary = evaluate_generator(
                        model_name=algo_directory,
                        benchmark_directory=benchmark_directory,
                        experiment_directory=experiment_path,
                        data_type=data_type,
                        use_cuda=use_cuda
                    )
                experiment_summary = complete_experiment_summary(
                        benchmark_directory, '', experiment_summary
                    )
                df = df.append(experiment_summary, ignore_index=True, )
        df.to_csv(df_dst_path, decimal=',', sep=';', float_format='%.5f')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Turn cuda off / on during evalution.')
    parser.add_argument('-use_cuda', action='store_true')
    args = parser.parse_args()
    evaluate_benchmarks(use_cuda=args.use_cuda)
