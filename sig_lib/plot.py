from os.path import join

import numpy as np
from matplotlib import pyplot as plt

from sig_lib.auxilliary_losses import *


def to_numpy(x):
    """
    Casts torch.Tensor to a numpy ndarray.

    The function detaches the tensor from its gradients, then puts it onto the cpu and at last casts it to numpy.
    """
    return x.detach().cpu().numpy()


def set_style(ax):
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)


def compare_hists(x_real, x_fake, ax=None, log=False, label=None):
    """ Computes histograms and plots those. """
    if ax is None:
        _, ax = plt.subplots(1, 1)
    if label is not None:
        label_historical = 'Historical ' + label
        label_generated = 'Generated ' + label
    else:
        label_historical = 'Historical'
        label_generated = 'Generated'
    bin_edges = ax.hist(x_real.flatten(), bins=80, alpha=0.6, density=True, label=label_historical)[1]
    ax.hist(x_fake.flatten(), bins=bin_edges, alpha=0.6, density=True, label=label_generated)
    ax.grid()
    set_style(ax)
    ax.legend()
    if log:
        ax.set_ylabel('log-pdf')
        ax.set_yscale('log')
    else:
        ax.set_ylabel('pdf')
    return ax


def compare_acf(x_real, x_fake, ax=None, max_lag=64, CI=True, dim=(0, 1), drop_first_n_lags=0):
    """ Computes ACF of historical and (mean)-ACF of generated and plots those. """
    if ax is None:
        _, ax = plt.subplots(1, 1)
    acf_real_list = cacf_torch(x_real, max_lag=max_lag, dim=dim).cpu().numpy()
    acf_real = np.mean(acf_real_list, axis=0)

    acf_fake_list = cacf_torch(x_fake, max_lag=max_lag, dim=dim).cpu().numpy()
    acf_fake = np.mean(acf_fake_list, axis=0)

    ax.plot(acf_real[drop_first_n_lags:], label='Historical')
    ax.plot(acf_fake[drop_first_n_lags:], label='Generated', alpha=0.8)

    if CI:
        acf_fake_std = np.std(acf_fake_list, axis=0)
        ub = acf_fake + acf_fake_std
        lb = acf_fake - acf_fake_std

        for i in range(acf_real.shape[-1]):
            ax.fill_between(
                range(acf_fake[:, i].shape[0]),
                ub[:, i], lb[:, i],
                color='orange',
                alpha=.3
            )
    set_style(ax)
    ax.set_xlabel('Lags')
    ax.set_ylabel('ACF')
    from matplotlib.ticker import MaxNLocator
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(True)
    ax.legend()
    return ax


def compare_cacf(x_real, x_fake, assets, ax=None, legend=False, max_lag=128, figsize=(10, 8)):
    """ Computes ACF of historical and (mean)-ACF of generated and plots those. """
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=figsize)
    acf_real_list = cacf_torch(x_real, max_lag=max_lag, dim=(1)).cpu().numpy()
    acf_real = np.mean(acf_real_list, axis=0)

    acf_fake_list = cacf_torch(x_fake, max_lag=max_lag, dim=(1)).cpu().numpy()
    acf_fake = np.mean(acf_fake_list, axis=0)
    acf_fake_std = np.std(acf_fake_list, axis=0)

    n = x_real.shape[2]
    ind = torch.tril_indices(n, n).transpose(0, 1).cpu().numpy()
    for i, (j, k) in enumerate(ind):
        ax.plot(acf_real[:, i], label='Historical: ' + ' '.join([assets[j], assets[k]]))
        ax.plot(acf_fake[:, i], label='Generated: ' + ' '.join([assets[j], assets[k]]), alpha=0.8)
    # ax.set_ylim([-0.1, 0.35])
    """
    ub = acf_fake + acf_fake_std
    lb = acf_fake - acf_fake_std
    for i in range(acf_real.shape[-1]):
        ax.fill_between(
            range(acf_fake[1:, i].shape[0]),
            ub[1:, i], lb[1:, i],
            color='orange',
            alpha=.3
        )
    """
    set_style(ax)
    ax.set_xlabel('Lags')
    ax.set_ylabel('ACF')
    ax.grid(True)
    if legend:
        ax.legend()
    return ax


def compare_lev_eff(x_real, x_fake, ax=None, CI=True, drop_first_n_lags=1, max_lag=None):
    """ Computes ACF of historical and (mean)-ACF of generated and plots those. """
    if ax is None:
        _, ax = plt.subplots(1, 1)
    acf_real_list = lev_eff_torch(x_real, max_lag=max_lag, dim=(1)).cpu().numpy()
    acf_real = np.mean(acf_real_list, axis=0)
    acf_fake_list = lev_eff_torch(x_fake, max_lag=max_lag, dim=(1)).cpu().numpy()
    acf_fake = np.mean(acf_fake_list, axis=0)
    acf_fake_std = np.std(acf_fake_list, axis=0)

    ax.plot(acf_real[drop_first_n_lags:], label='Historical')
    ax.plot(acf_fake[drop_first_n_lags:], label='Generated', alpha=0.8)

    if CI:
        ub = acf_fake + acf_fake_std
        lb = acf_fake - acf_fake_std

        ax.fill_between(range(acf_fake[1:].shape[0]), ub[1:], lb[1:],
                        color='orange', alpha=.3)
    set_style(ax)
    ax.set_xlabel('Lags')
    ax.set_ylabel('Leverage Effect')
    ax.grid(True)
    ax.legend()
    return ax


def plot_spot(x_fake, x_real, max_lag=None):
    if max_lag is None:
        max_lag = min(128, x_fake.shape[1])

    from sig_lib.auxilliary_losses import skew_torch, kurtosis_torch
    dim = x_real.shape[2]
    _, axes = plt.subplots(dim, 5, figsize=(25, dim * 5))

    if len(axes.shape) == 1:
        axes = axes[None, ...]
    for i in range(dim):
        x_real_i = x_real[..., i:i + 1]
        x_fake_i = x_fake[..., i:i + 1]

        compare_hists(x_real=to_numpy(x_real_i), x_fake=to_numpy(x_fake_i), ax=axes[i, 0])

        def text_box(x, height, title):
            textstr = '\n'.join((
                r'%s' % (title,),
                # t'abs_metric=%.2f' % abs_metric
                r'$s=%.2f$' % (skew_torch(x).item(),),
                r'$\kappa=%.2f$' % (kurtosis_torch(x).item(),))
            )
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
            axes[i, 0].text(
                0.05, height, textstr,
                transform=axes[i, 0].transAxes,
                fontsize=14,
                verticalalignment='top',
                bbox=props
            )

        text_box(x_real_i, 0.95, 'Historical')
        text_box(x_fake_i, 0.70, 'Generated')

        compare_hists(x_real=to_numpy(x_real_i), x_fake=to_numpy(x_fake_i), ax=axes[i, 1], log=True)
        compare_acf(x_real=x_real_i, x_fake=x_fake_i, ax=axes[i, 2], max_lag=max_lag, CI=False, dim=(0, 1),
                    drop_first_n_lags=0)
        compare_acf(x_real=torch.abs(x_real_i), x_fake=torch.abs(x_fake_i), ax=axes[i, 3], max_lag=max_lag, CI=False,
                    drop_first_n_lags=0,
                    dim=(0, 1))
        compare_lev_eff(x_real_i, x_fake_i, ax=axes[i, 4], CI=False, max_lag=max_lag)


def plot_var(x_fake, x_real, max_lag=None, labels=None):
    if max_lag is None:
        max_lag = min(128, x_fake.shape[1])

    from sig_lib.auxilliary_losses import skew_torch, kurtosis_torch
    dim = x_real.shape[2]
    _, axes = plt.subplots(dim, 3, figsize=(25, dim * 5))

    if len(axes.shape) == 1:
        axes = axes[None, ...]
    for i in range(dim):
        x_real_i = x_real[..., i:i + 1]
        x_fake_i = x_fake[..., i:i + 1]

        compare_hists(x_real=to_numpy(x_real_i), x_fake=to_numpy(x_fake_i), ax=axes[i, 0])

        def text_box(x, height, title):
            textstr = '\n'.join((
                r'%s' % (title,),
                # t'abs_metric=%.2f' % abs_metric
                r'$s=%.2f$' % (skew_torch(x).item(),),
                r'$\kappa=%.2f$' % (kurtosis_torch(x).item(),))
            )
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
            axes[i, 0].text(
                0.05, height, textstr,
                transform=axes[i, 0].transAxes,
                fontsize=14,
                verticalalignment='top',
                bbox=props
            )

        text_box(x_real_i, 0.95, 'Historical')
        text_box(x_fake_i, 0.70, 'Generated')

        compare_hists(x_real=to_numpy(x_real_i), x_fake=to_numpy(x_fake_i), ax=axes[i, 1], log=True)
        compare_acf(x_real=x_real_i, x_fake=x_fake_i, ax=axes[i, 2], max_lag=max_lag, CI=False, dim=(0, 1))


def compare_cross_corr(x_real, x_fake):
    """ Computes cross correlation matrices of x_real and x_fake and plots them. """
    x_real = x_real.reshape(-1, x_real.shape[2])
    x_fake = x_fake.reshape(-1, x_fake.shape[2])
    cc_real = np.corrcoef(to_numpy(x_real).T)
    cc_fake = np.corrcoef(to_numpy(x_fake).T)

    vmin = min(cc_fake.min(), cc_real.min())
    vmax = max(cc_fake.max(), cc_real.max())

    fig, axes = plt.subplots(1, 2)
    axes[0].matshow(cc_real, vmin=vmin, vmax=vmax)
    im = axes[1].matshow(cc_fake, vmin=vmin, vmax=vmax)

    axes[0].set_title('Real')
    axes[1].set_title('Generated')
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    fig.colorbar(im, cax=cbar_ax)


def log_aux_losses(aux_losses, training_loss):
    num_subplots = len(aux_losses)
    i = 1
    _, axes = plt.subplots(1, num_subplots, figsize=(24, 6))
    for (name, _), ax in zip(aux_losses, axes):
        ax.plot(training_loss[name], alpha=0.8)
        ax.set_title(name)
        ax.grid(True)
        if name == 'skew':
            ax.set_ylim([0, 2])
        elif name == 'kurtosis':
            ax.set_ylim([0, 10])
        i += 1


def plot_signature(signature_tensor, alpha=0.2):
    plt.plot(to_numpy(signature_tensor).T, alpha=alpha, linestyle='None', marker='o')
    plt.grid()


def savefig(filename, directory):
    plt.savefig(join(directory, filename))
    plt.close()


def to_numpy(x):
    """
    Casts torch.Tensor to a numpy ndarray.

    The function detaches the tensor from its gradients, then puts it onto the cpu and at last casts it to numpy.
    """
    return x.detach().cpu().numpy()


def subplot_multiple_series(x):
    fig, axes = plt.subplots(x.shape[-1], 1, figsize=(10, 15))
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])
    for i in range(x.shape[-1]):
        axes[i].plot(to_numpy(x[:, i]))
        axes[i].grid()
