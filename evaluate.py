import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LinearRegression

from hyperparameters import SIGCWGAN_CONFIGS
from lib.algos.base import BaseConfig
from lib.algos.base import is_multivariate
from lib.algos.sigcwgan import calibrate_sigw1_metric, sample_sig_fake
from lib.algos.sigcwgan import sigcwgan_loss
from lib.arfnn import SimpleGenerator
from lib.plot import plot_summary, compare_cross_corr
from lib.test_metrics import test_metrics
from lib.utils import load_pickle, to_numpy

warnings.filterwarnings("ignore")


def compute_predictive_score(x_past, x_future, x_fake):
    size = x_fake.shape[0]
    X = to_numpy(x_past.reshape(size, -1))
    Y = to_numpy(x_fake.reshape(size, -1))
    size = x_past.shape[0]
    X_test = X.copy()
    Y_test = to_numpy(x_future[:, :1].reshape(size, -1))
    model = LinearRegression()
    model.fit(X, Y)  # TSTR
    r2_tstr = model.score(X_test, Y_test)
    model = LinearRegression()
    model.fit(X_test, Y_test)  # TRTR
    r2_trtr = model.score(X_test, Y_test)
    return dict(r2_tstr=r2_tstr, r2_trtr=r2_trtr, predictive_score=np.abs(r2_trtr - r2_tstr))


def compute_test_metrics(x_fake, x_real):
    res = dict()
    res['abs_metric'] = test_metrics['abs_metric'](x_real)(x_fake).item()
    res['acf_id_lag=1'] = test_metrics['acf_id'](x_real, max_lag=2)(x_fake).item()
    res['kurtosis'] = test_metrics['kurtosis'](x_real)(x_fake).item()
    res['skew'] = test_metrics['skew'](x_real)(x_fake).item()
    if is_multivariate(x_real):
        res['cross_correl'] = test_metrics['cross_correl'](x_real)(x_fake).item()
    return res


def get_algo_config(dataset, experiment_dir):
    key = dataset
    if dataset == 'VAR':
        key += experiment_dir.split('/')[2][4]
    elif dataset == 'STOCKS':
        key += '_' + experiment_dir.split('/')[2]
    sig_config = SIGCWGAN_CONFIGS[key]
    return sig_config


def evaluate_generator(model_name, seed, experiment_dir, dataset, use_cuda=True):
    """

    Args:
        model_name:
        experiment_dir:
        dataset:
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
    experiment_summary['seed'] = seed

    sig_config = get_algo_config(dataset, experiment_dir)
    # shorthands
    base_config = BaseConfig(device=device)
    p, q = base_config.p, base_config.q
    # ----------------------------------------------
    # Load and prepare real path.
    # ----------------------------------------------
    x_real = load_pickle(os.path.join(os.path.dirname(experiment_dir), 'x_real_test.torch')).to(device)
    x_past, x_future = x_real[:, :p], x_real[:, p:p + q]
    x_future = x_real[:, p:p + q]
    dim = x_real.shape[-1]
    # ----------------------------------------------
    # Load generator weights and hyperparameters
    # ----------------------------------------------
    G_weights = load_pickle(os.path.join(experiment_dir, 'G_weights.torch'))
    G = SimpleGenerator(dim * p, dim, 3 * (50,), dim).to(device)
    G.load_state_dict(G_weights)
    # ----------------------------------------------
    # Compute predictive score - TSTR (train on synthetic, test on real)
    # ----------------------------------------------
    with torch.no_grad():
        x_fake = G.sample(1, x_past)
    predict_score_dict = compute_predictive_score(x_past, x_future, x_fake)
    experiment_summary.update(predict_score_dict)
    # ----------------------------------------------
    # Compute metrics and scores of the unconditional distribution.
    # ----------------------------------------------
    with torch.no_grad():
        x_fake = G.sample(q, x_past)
    test_metrics_dict = compute_test_metrics(x_fake, x_real)
    experiment_summary.update(test_metrics_dict)
    # ----------------------------------------------
    # Compute Sig-W_1 distance.
    # ----------------------------------------------
    if dataset in ['VAR', 'ARCH']:
        x_past = x_past[::10]
        x_future = x_future[::10]
    sigs_pred = calibrate_sigw1_metric(sig_config, x_future, x_past)
    # generate fake paths
    sigs_conditional = list()
    with torch.no_grad():
        steps = 100
        size = x_past.size(0) // steps
        for i in range(steps):
            x_past_sample = x_past[i * size:(i + 1) * size] if i < (steps - 1) else x_past[i * size:]
            sigs_fake_ce = sample_sig_fake(G, q, sig_config, x_past_sample)[0]
            sigs_conditional.append(sigs_fake_ce)
        sigs_conditional = torch.cat(sigs_conditional, dim=0)
        sig_w1_metric = sigcwgan_loss(sigs_pred, sigs_conditional)
    experiment_summary['sig_w1_metric'] = sig_w1_metric.item()
    # ----------------------------------------------
    # Create the relevant summary plots.
    # ----------------------------------------------
    with torch.no_grad():
        _x_past = x_past.clone().repeat(5, 1, 1) if dataset in ['STOCKS', 'ECG'] else x_past.clone()
        x_fake_future = G.sample(q, _x_past)
        plot_summary(x_fake=x_fake_future, x_real=x_real, max_lag=q)
    plt.savefig(os.path.join(experiment_dir, 'summary.png'))
    plt.close()
    if is_multivariate(x_real):
        compare_cross_corr(x_fake=x_fake_future, x_real=x_real)
        plt.savefig(os.path.join(experiment_dir, 'cross_correl.png'))
        plt.close()
    # ----------------------------------------------
    # Generate long paths
    # ----------------------------------------------
    with torch.no_grad():
        x_fake = G.sample(8000, x_past[0:1])
    plot_summary(x_fake=x_fake, x_real=x_real, max_lag=q)
    plt.savefig(os.path.join(experiment_dir, 'summary_long.png'))
    plt.close()
    plt.plot(to_numpy(x_fake[0, :1000]))
    plt.savefig(os.path.join(experiment_dir, 'long_path.png'))
    plt.close()
    return experiment_summary


def complete_experiment_summary(benchmark_directory, experiment_directory, experiment_summary):
    if benchmark_directory == 'VAR':
        experiment_summary['phi'] = float(experiment_directory.split('_')[1].split('=')[-1])
        experiment_summary['sigma'] = float(experiment_directory.split('_')[2].split('=')[-1])
    elif benchmark_directory in ['lag=3']:
        experiment_summary['lag'] = int(benchmark_directory.split('=')[-1])
    elif benchmark_directory == 'STOCKS':
        experiment_summary['asset'] = experiment_directory
    return experiment_summary


def get_top_dirs(path):
    return [directory for directory in os.listdir(path) if os.path.isdir(os.path.join(path, directory))]


def evaluate_benchmarks(algos, base_dir, datasets, use_cuda=False):
    msg = 'Running evalution on GPU.' if use_cuda else 'Running evalution on CPU.'
    print(msg)
    for dataset_dir in os.listdir(base_dir):
        dataset_path = os.path.join(base_dir, dataset_dir)
        if dataset_dir not in datasets:
            continue
        for experiment_dir in os.listdir(dataset_path):
            df = pd.DataFrame(columns=[])
            experiment_path = os.path.join(dataset_path, experiment_dir)
            for seed_dir in get_top_dirs(experiment_path):
                seed_path = os.path.join(experiment_path, seed_dir)
                for algo_dir in get_top_dirs(seed_path):
                    if algo_dir not in algos:
                        continue
                    print(dataset_dir, experiment_dir, algo_dir, )
                    algo_path = os.path.join(seed_path, algo_dir)
                    # evaluate the generator
                    experiment_summary = evaluate_generator(
                        model_name=algo_dir,
                        seed=seed_dir.split('_')[-1],
                        experiment_dir=algo_path,
                        dataset=dataset_dir,
                        use_cuda=use_cuda
                    )
                    # add relevant parameters used during training to the experiment summary
                    experiment_summary = complete_experiment_summary(dataset_dir, experiment_dir, experiment_summary)
                    df = df.append(experiment_summary, ignore_index=True, )
            df_dst_path = os.path.join(base_dir, dataset_dir, experiment_dir, 'summary.csv')
            df.to_csv(df_dst_path, decimal=',', sep=';', float_format='%.5f', index=False)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Turn cuda off / on during evalution.')
    parser.add_argument('-base_dir', default='./numerical_results', type=str)
    parser.add_argument('-use_cuda', action='store_true')
    parser.add_argument('-datasets', default=['ARCH', 'STOCKS', 'ECG', 'VAR', ], nargs="+")
    parser.add_argument('-algos', default=['SigCWGAN', 'GMMN', 'RCGAN', 'TimeGAN', 'RCWGAN', ], nargs="+")
    args = parser.parse_args()
    evaluate_benchmarks(base_dir=args.base_dir, use_cuda=args.use_cuda, datasets=args.datasets, algos=args.algos)
