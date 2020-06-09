import os.path as pt

import matplotlib.pyplot as plt
import torch
from torch import autograd
from tqdm import tqdm

from sig_lib.ResFNN import ResFNN
from sig_lib.sig_conditional import get_dataset, supervised_losses, to_numpy, compare_cross_corr
from sig_lib.sig_conditional import sample
from sig_lib.tools import sample_indices


def train_baseline(
        experiment_directory,
        device,
        total_steps,
        gan_algo,
        batch_size,
        p,
        q,
        latent_dim,
        hidden_dims,
        data_type,
        data_params,
        dim=None,
        D_steps_per_G_step=2,
        hidden_dims_D=None,
):
    assert gan_algo in ['RCGAN', 'TimeGAN', 'GMMN'], 'Algo %s not allowed' % gan_algo
    # ---------------------------------------
    # initialise all variables
    # ---------------------------------------
    if data_type=='ARCH':
        p = data_params['lag'][0]
    pipeline, x_real_raw, x_real, plot_comparison = get_dataset(data_type, p, q, device=device, dim=dim, **data_params)
    if dim is None:
        dim = x_real.shape[-1]
        latent_dim = dim

    G = ResFNN(latent_dim + p * dim, dim, hidden_dims)
    G_optimizer = torch.optim.Adam(G.parameters(), lr=1e-4, betas=(0, 0.9))

    if gan_algo in ['RCGAN', 'TimeGAN']:
        D = ResFNN(dim * (p + q), 1, hidden_dims_D)
        D_optimizer = torch.optim.Adam(D.parameters(), lr=2e-4, betas=(0, 0.9))  # Using TTUR
    else:
        D, D_optimizer = None, None

    aux_loss_list = [
        supervised_losses['abs_metric'](x_real, reg=0.),
        supervised_losses['acf_id'](x_real, max_lag=2, reg=0.),
    ]
    if x_real.shape[-1] > 1:
        aux_loss_list.append(supervised_losses['cross_correl'](x_real))
    trainer = CGANTrainer(  # central object to tune the GAN
        G=G, D=D, G_optimizer=G_optimizer, D_optimizer=D_optimizer,
        gan_algo=gan_algo, p=p, q=q,
    )

    # ---------------------------------------
    # start of training
    # ---------------------------------------

    from collections import defaultdict
    training_loss = defaultdict(list)
    dataset_size = x_real.shape[0]
    for _ in tqdm(range(total_steps)):
        if gan_algo in ['RCGAN', 'TimeGAN']:
            for _ in range(D_steps_per_G_step):
                # generate x_fake
                z = torch.randn(batch_size, q, latent_dim)
                indices = sample_indices(dataset_size, batch_size)
                x_past = x_real[indices, :p].clone()
                with torch.no_grad():
                    x_fake = sample(G, z, x_past.clone())
                    x_fake = torch.cat([x_past, x_fake], dim=1)
                D_loss_real, D_loss_fake = trainer.D_trainstep(x_fake, x_real[indices])
                training_loss['D_loss_fake'].append(D_loss_fake)
                training_loss['D_loss_real'].append(D_loss_real)
                training_loss['D_loss'].append(D_loss_fake + D_loss_real)

        # Generator step
        z = torch.randn(batch_size, q, latent_dim)
        indices = sample_indices(dataset_size, batch_size)
        x_past = x_real[indices, :p].clone()
        x_past.requires_grad_()
        z.requires_grad_()
        x_fake = sample(G, z, x_past)
        x_fake_past = torch.cat([x_past, x_fake], dim=1)
        G_loss = trainer.G_trainstep(x_fake_past, x_real[indices].clone())
        if gan_algo == 'GMMN':
            training_loss['MMD'].append(G_loss)
        else:
            training_loss['G_loss'].append(G_loss)
        for aux_loss in aux_loss_list:
            with torch.no_grad():
                aux_loss(x_fake[:10000])
            training_loss[aux_loss.name].append(to_numpy(aux_loss.loss_componentwise))
    return G, training_loss, x_real, aux_loss_list


class CGANTrainer(object):
    def __init__(
            self,
            G,
            D,
            G_optimizer,
            D_optimizer,
            p,
            q,
            gan_algo,
    ):
        self.G = G
        self.D = D
        self.G_optimizer = G_optimizer
        self.D_optimizer = D_optimizer

        self.p = p
        self.q = q

        self.gan_algo = gan_algo

    def G_trainstep(self, x_fake, x_real):
        toggle_grad(self.G, True)
        self.G.train()
        self.G_optimizer.zero_grad()
        if self.gan_algo in ['RCGAN', 'TimeGAN']:
            d_fake = self.D(x_fake)
            self.D.train()
            gloss = self.compute_loss(d_fake, 1)

            if self.gan_algo == 'TimeGAN':
                gloss = gloss + torch.mean((x_fake - x_real) ** 2)
        elif self.gan_algo in ['GMMN']:
            gloss = mmd_loss(x_real, x_fake)
        gloss.backward()
        self.G_optimizer.step()
        return gloss.item()

    def D_trainstep(self, x_fake, x_real):
        toggle_grad(self.D, True)
        self.D.train()
        self.D_optimizer.zero_grad()

        # On real data
        x_real.requires_grad_()
        d_real = self.D(x_real)
        dloss_real = self.compute_loss(d_real, 1)

        # On fake data
        x_fake.requires_grad_()
        d_fake = self.D(x_fake)
        dloss_fake = self.compute_loss(d_fake, 0)

        # Compute regularizer on fake/real
        dloss = dloss_fake + dloss_real
        dloss.backward()
        # Step discriminator params
        self.D_optimizer.step()

        # Toggle gradient to False
        toggle_grad(self.D, False)

        return dloss_real.item(), dloss_fake.item()

    @staticmethod
    def compute_loss(d_out, target):
        targets = d_out.new_full(size=d_out.size(), fill_value=target)
        return torch.nn.functional.binary_cross_entropy_with_logits(d_out, targets)


def toggle_grad(model, requires_grad):
    for p in model.parameters():
        p.requires_grad_(requires_grad)


def compute_grad2(d_out, x_in):
    batch_size = x_in.size(0)
    grad_dout = autograd.grad(
        outputs=d_out.sum(), inputs=x_in,
        create_graph=True, retain_graph=True, only_inputs=True
    )[0]
    grad_dout2 = grad_dout.pow(2)
    assert (grad_dout2.size() == x_in.size())
    reg = grad_dout2.view(batch_size, -1).sum(1)
    return reg


def _rbf(norm, sigma):
    return torch.exp(-norm / (2 * sigma ** 2))


def pairwise_distance(X, Y=None):
    n = X.size(0)
    X = X.contiguous().view(n, -1)
    if Y is None:
        Y = X
    else:
        m = Y.size(0)
        Y = Y.contiguous().view(m, -1)
    pairwise_distance = torch.pow(X.unsqueeze(0) - Y.unsqueeze(1), 2)
    l2_dist = pairwise_distance.sum(2)
    return l2_dist


def median_pairwise_distance(X, Y=None):
    """
    Heuristic for bandwidth of the RBF. Median pairwise distance of joint data.
    If Y is missing, just calculate it from X:
        this is so that, during training, as Y changes, we can use a fixed
        bandwidth (and save recalculating this each time we evaluated the mmd)
    At the end of training, we do the heuristic "correctly" by including
    both X and Y.
    Note: most of this code is assuming tensorflow, but X and Y are just ndarrays
    """
    if Y is None:
        Y = X
    if len(X.shape) == 2:
        # matrix
        X_sqnorms = torch.einsum('...i,...i', X, X)
        Y_sqnorms = torch.einsum('...i,...i', Y, Y)
        XY = torch.einsum('ia,ja', X, Y)
    elif len(X.shape) == 3:
        # tensor -- this is computing the Frobenius norm
        X_sqnorms = torch.einsum('...ij,...ij', X, X)
        Y_sqnorms = torch.einsum('...ij,...ij', Y, Y)
        XY = torch.einsum('iab,jab', X, Y)
    distances = torch.sqrt(X_sqnorms.reshape(-1, 1) - 2 * XY + Y_sqnorms.reshape(1, -1))
    return torch.median(distances)


def _partial_mmd(X, Y=None, bandwidth=None, heuristic=True):
    l2_dist = pairwise_distance(X, Y)
    if heuristic:
        heuristic_sigma = median_pairwise_distance(X, Y).detach()
        return torch.mean(_rbf(l2_dist, heuristic_sigma))
    else:
        return torch.mean(_rbf(l2_dist, bandwidth))


def mmd_loss(real_data, fake_data, bandwidths=(0.1, 1, 5), heuristic=False):
    if heuristic:
        mmd_gen_real = _partial_mmd(real_data, fake_data, bandwidth=None, heuristic=heuristic)
        mmd_gen = _partial_mmd(fake_data, bandwidth=None, heuristic=heuristic)
        mmd_real = _partial_mmd(real_data, bandwidth=None, heuristic=heuristic)
        mmd = mmd_real - 2 * mmd_gen_real + mmd_gen
    else:
        mmd = 0
        for bandwidth in bandwidths:
            mmd_gen_real = _partial_mmd(real_data, fake_data, bandwidth=bandwidth, heuristic=heuristic)
            mmd_gen = _partial_mmd(fake_data, bandwidth=bandwidth, heuristic=heuristic)
            mmd_real = _partial_mmd(real_data, bandwidth=bandwidth, heuristic=heuristic)
            mmd += mmd_real - 2 * mmd_gen_real + mmd_gen
    return mmd
