

from os import path as pt

import matplotlib.pyplot as plt
import signatory
import torch
from torch import optim
from tqdm import tqdm

from sig_lib.ResFNN import ResFNN
from sig_lib.augmentations import lift, lead_lag_transform_2
from sig_lib.auxilliary_losses import supervised_losses
from sig_lib.data import rolling_window, get_var_dataset, get_equities_dataset,get_arch_dataset
from sig_lib.plot import plot_spot, plot_var, compare_cross_corr
from sig_lib.plot import to_numpy
from sig_lib.tools import sample_indices


def get_dataset(data_type, p, q, device, **data_params):
    """

    Args:
        data_type:
        p:
        q:
        device:
        path_length:
        num_paths:
        **data_params:

    Returns:

    """
    if data_type == 'VAR':
        pipeline, x_real_raw, x_real = get_var_dataset(
            40000, batch_size=1, **data_params
        )
        plot_comparison = plot_var
        x_real = rolling_window(x_real[0], p + q)[::p + q].to(device)
    elif data_type == 'stocks':
        data_params.pop('dim')
        pipeline, x_real_raw, x_real = get_equities_dataset(**data_params)
        plot_comparison = plot_spot
        x_real = rolling_window(x_real[0], p + q).to(device)
    elif data_type == 'ARCH':
        pipeline, x_real_raw, x_real = get_arch_dataset(
            40000, N=1, **data_params
        )
        plot_comparison = plot_var
        x_real = rolling_window(x_real[0], p + q).to(device)
    else:
        raise NotImplementedError('')
    x_real_raw = x_real_raw.to(device)
    return pipeline, x_real_raw, x_real, plot_comparison


def postprocess(x, scalar, with_time, with_lift, with_concat, m):
    x = scalar * x  # scaling factor to increase the variance of the process --> signature decays slower for larger factors
    x_cumsum = torch.cumsum(x, dim=1)
    if with_concat:
        x = torch.cat([x_cumsum, x], dim=-1)
    else:
        x = x_cumsum
    if with_lift and x.shape[1] > m:
        x = lift(x, m)
    return lead_lag_transform_2(x, with_time)  # compute


def compute_signature(x, scalar, sig_depth, with_time, with_lift, with_concat, m):
    y = postprocess(x, scalar, with_time, with_lift, with_concat, m)
    return signatory.signature(y, sig_depth)


def sample(G, z, x_past_sample):
    """
    Sampling function. Uses the generator function, sampled noise and a condition of the real process.
    """
    x_generated = list()
    for t in range(z.shape[1]):
        z_t = z[:, t:t + 1]
        x_in = torch.cat([z_t, x_past_sample.reshape(x_past_sample.shape[0], 1, -1)], dim=-1)
        x_gen = G(x_in).unsqueeze(1)
        x_past_sample = torch.cat([x_past_sample[:, :-1], x_gen], dim=1)
        x_generated.append(x_gen)
    x_fake = torch.cat(x_generated, dim=1)
    return x_fake


def sigcgan_loss(sig_pred, sig_fake_conditional_expectation):
    return torch.norm(sig_pred - sig_fake_conditional_expectation, p=2, dim=1).mean()


def savefig(path):
    plt.savefig(path)
    plt.close()


def train_SigCGAN(
        data_type,
        latent_dim,
        hidden_dims,
        lr,  # learning rate of the generator
        p,  # lookback and future
        q,
        scalar_p,  # scalar for postprocessing the generated path
        scalar_q,  # scalar for postprocessing the generated path
        sig_degree_p,  # depth of the signature
        sig_degree_q,
        total_steps,
        experiment_directory,  # directory where logs should be saved
        batch_size,
        mc_size,
        sig_params_p,
        sig_params_q,
        dim=None,
        data_params=dict(),
        device='cpu',
):
    # ------------------------------------------------------------
    # Load the dataset.
    # ------------------------------------------------------------
    if data_type=='ARCH':
        p = data_params['lag'][0]
    pipeline, x_real_raw, x_real, plot_comparison = get_dataset(data_type, p, q, device=device, dim=dim, **data_params)
    if dim is None:
        dim = x_real.shape[-1]
        latent_dim = dim
    x_past = x_real[:, :p]
    x_future = x_real[:, p:]
    sigs_past = compute_signature(x_past, scalar_p, sig_degree_p, **sig_params_p)
    sigs_future = compute_signature(x_future, scalar_q, sig_degree_q, **sig_params_q)
    assert sigs_past.size(0) == sigs_future.size(0)
    # ------------------------------------------------------------
    # apply linear regression to past and future signatures
    # ------------------------------------------------------------
    X, Y = sigs_past, sigs_future
    X = to_numpy(X)
    Y = to_numpy(Y)
    from sklearn.linear_model import Ridge
    model = Ridge(alpha=1)
    model.fit(X, Y)
    sigs_pred = torch.from_numpy(model.predict(X)).float().to(device)
    # ------------------------------------------------------------
    # Initialise generator and optimiser.
    # ------------------------------------------------------------
    # input dimension is p, the lookback, + the latent dimension
    input_dim = p * dim + latent_dim
    G = ResFNN(input_dim, dim, hidden_dims)
    G_optimizer = optim.Adam(G.parameters(), lr=lr)
    G_scheduler = optim.lr_scheduler.StepLR(G_optimizer, step_size=100, gamma=0.9)
    # ------------------------------------------------------------
    # Initialise auxilliary loss functions / metrics.
    # ------------------------------------------------------------
    aux_loss_list = [
        supervised_losses['abs_metric'](x_real, reg=0.1),
        supervised_losses['acf_id'](x_real, max_lag=2, reg=0.3),
    ]
    if x_real.shape[-1] > 1:
        aux_loss_list.append(supervised_losses['cross_correl'](x_real, reg=0.1))
    # ------------------------------------------------------------
    # Start of optimization
    # ------------------------------------------------------------
    from collections import defaultdict
    training_loss = defaultdict(list)
    # always using the same noise and conditioning vector for testing
    if batch_size == -1:
        batch_size = x_past.shape[0]
    for step in tqdm(range(total_steps)):  # sig_pred, x_past, x_real
        G.train()
        G_optimizer.zero_grad()  # empty 'cache' of gradients
        random_indices = sample_indices(sigs_pred.shape[0], batch_size)  # sample indices
        # sample the least squares signature and the log-rtn condition
        sig_pred_sample = sigs_pred[random_indices].clone()
        x_past_sample = x_past[random_indices].clone()
        # sample noise - here we take batch_size * mc_size samples because we need to approximate later the conditional
        # expectation E[S(X_future) | S(X_past)]
        z = torch.randn(batch_size * mc_size, q, latent_dim)
        # we therefore also repeat the conditioning vector
        x_past_sample = x_past_sample.repeat(mc_size, 1, 1)
        x_past_sample.requires_grad_()
        # next we sample the 'fake' time series by using the noise and the conditioning vector
        x_fake_future = sample(G, z, x_past_sample)
        # then the 'fake' time series is postprocessed, i.e. scalar multiplication, cumsum, leadlag
        sig_fake_futu = compute_signature(x_fake_future, scalar_q, sig_degree_q, **sig_params_q)
        # then reshape the 'fake' signature such that the conditional expectation can be computed and take the mean
        sig_fake_conditional_expectation = sig_fake_futu.reshape(mc_size, batch_size, -1).transpose(0, 1).mean(1)
        # compute sig_w1 loss function
        loss = sigcgan_loss(sig_pred_sample, sig_fake_conditional_expectation)
        loss.backward()
        total_norm = torch.nn.utils.clip_grad_norm_(G.parameters(), 10)
        training_loss['loss'].append(loss.item())
        training_loss['total_norm'].append(total_norm)
        for aux_loss in aux_loss_list:
            with torch.no_grad():
                aux_loss(x_fake_future[:10000])
            training_loss[aux_loss.name].append(to_numpy(aux_loss.loss_componentwise))
        G_optimizer.step()
        G_scheduler.step()  # decaying learning rate slowly.
    return G, training_loss, x_real, aux_loss_list
