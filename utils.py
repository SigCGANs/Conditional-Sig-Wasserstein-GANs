import csv
import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

msg = {
    'sig_w1_metric': 'Sig-$W_1$ metric',
    'abs_metric': 'density metric',
    'acf_id_lag=1': 'ACF(1) score',
    'cross_correl': 'cross-correlation score',
    'pred_score': 'predictive score'
}


def aggregate_dfs(numerical_root, admissible_datasets, prefix):
    dfs = None
    for dirpath, dirnames, filenames in os.walk(numerical_root):
        for filename in [f for f in filenames if f.endswith(".csv")]:
            if any([True for part in dirpath.split('/') if part in admissible_datasets]):
                df = pd.read_csv(os.path.join(dirpath, filename), decimal=',', sep=';')
                df = df.rename(columns={'model_id': 'Model'})
                admissible_columns = [col for col in df.columns if
                                      col not in ['asset', 'phi', 'sigma', 'seed', 'r2_tstr', 'r2_trtr']]
                df = df[admissible_columns]
                dataset = dirpath.split('/')[-2]
                if dataset == 'VAR':
                    import re
                    par = re.split('_ |=|_', dirpath.split('/')[-1])[1::2]
                    if prefix != 'VAR_only_':
                        df['Dataset'] = 'VAR(' + ','.join(par[:1]) + ')'
                    else:
                        df['Dataset'] = 'VAR(' + ','.join(par) + ')'
                elif dataset == 'MIT':
                    df['Dataset'] = 'ECG'
                elif dataset == 'STOCKS':
                    df['Dataset'] = dirpath.split('/')[-1].replace('_', '+')
                elif dataset == 'ARCH':
                    df['Dataset'] = 'ARCH'
                else:
                    df['Dataset'] = '_'.join(dirpath.split('/')[-2:])
                if dfs is None:
                    dfs = df
                else:
                    dfs = dfs.append(df)
    test_metric_names = [col for col in dfs.columns if col not in ['Model', 'Dataset']]
    return dfs, test_metric_names


def plot_accross_benchmarks(numerical_root, admissible_datasets=('STOCKS', 'VAR', 'ARCH', 'MIT'), prefix='', rot=0):
    dfs, test_metric_names = aggregate_dfs(numerical_root, admissible_datasets, prefix)
    for test_metric in test_metric_names:
        df_sub = dfs[['Model', 'Dataset', test_metric]]
        df_pivot_mean = df_sub.pivot_table(index='Model', columns='Dataset', values=test_metric).transpose()
        df_pivot_std = df_sub.pivot_table(
            index='Model', columns='Dataset', values=test_metric, aggfunc='std'
        ).transpose()

        msg2 = 'VAR' if prefix == 'VAR_only' else 'benchmark'
        if len(df_pivot_std) == 0:
            ax = df_pivot_std.plot(kind='bar', legend=True, figsize=(6, 4), rot=rot, capsize=2, zorder=3)
        else:
            ax = df_pivot_mean.plot(
                kind='bar', yerr=df_pivot_std,
                error_kw=dict(ecolor='black', elinewidth=1.), legend=True,
                figsize=(6, 4), rot=rot, capsize=2, zorder=3,
                # title='Comparison of the %s across different %s datasets' % (msg[test_metric], msg2)
            )
        # box = ax.get_position()
        # ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        # ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.grid()
        plt.ylabel(msg[test_metric])
        plt.tight_layout()
        fn = prefix + 'model_comparison_thin_%s.pdf' % test_metric
        plt.savefig(os.path.join(numerical_root, fn))
        plt.close()
    return dfs


def create_aggregated_tables(numerical_root, admissible_datasets=('STOCKS', 'VAR', 'ARCH', 'MIT'), prefix=''):
    dfs, test_metric_names = aggregate_dfs(numerical_root, admissible_datasets, prefix)

    def aggfunc(x):
        return '${:.4f} (\pm {:.4f})$'.format(np.mean(x), np.std(x))

    msg = {
        'sig_w1_metric': 'Sig-$W_1$ metric',
        'abs_metric': 'density metric',
        'acf_id_lag=1': 'ACF(1) score',
        'cross_correl': 'cross-correlation score',
        'pred_score': 'predictive score',
        'kurtosis': 'kurtosis score',
        'skew': 'skewness score',
    }

    test_metric = test_metric_names[0]
    with open(os.path.join(numerical_root, prefix + 'model_comparison.csv'), 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        df_pivot = dfs.pivot_table(
            index='Model', columns='Dataset', aggfunc=aggfunc, values=test_metric
        )
        names = df_pivot.columns.tolist()
        names.insert(0, 'Model')
        writer.writerow(names)
        for test_metric in test_metric_names:
            writer.writerow([msg[test_metric]] + (len(df_pivot.columns)) * [''])
            df_pivot = dfs.pivot_table(
                index='Model', columns='Dataset', aggfunc=aggfunc, values=test_metric
            )
            df_pivot = df_pivot.reset_index()
            for i in range(df_pivot.shape[0]):
                writer.writerow(df_pivot[i:i + 1].values.tolist()[0])


def plot(numerical_root):
    for dirpath, dirnames, filenames in os.walk(numerical_root):
        for filename in [f for f in filenames if f.endswith(".csv")]:
            print(os.path.join(dirpath, filename))
            df = pd.read_csv(os.path.join(dirpath, filename), decimal=',', sep=';')
            admissible_columns = [col for col in df.columns if
                                  col not in ['phi', 'sigma', 'seed', 'r2_tstr', 'r2_trtr']]
            df[admissible_columns].groupby('model_id').mean().transpose().plot(
                kind='bar', yerr=df.groupby('model_id').std().transpose(),
                error_kw=dict(ecolor='black', elinewidth=1.), legend=True
            )
            plt.tight_layout()
            plt.savefig(os.path.join(dirpath, 'summary.pdf'))
            plt.close()
            break


if __name__ == '__main__':
    plot(root)
    # create_aggregated_tables(root, admissible_datasets=('STOCKS', 'ARCH', 'ECG'), prefix='')
    # create_aggregated_tables(root, admissible_datasets=('VAR',), prefix='VAR_only_')
    # plot_accross_benchmarks(root, admissible_datasets=('STOCKS', 'ARCH', 'ECG'), rot=0)
    # plot_accross_benchmarks(root, admissible_datasets=('VAR',), prefix='VAR_only_')
