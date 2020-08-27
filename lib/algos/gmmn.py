import torch

from lib.algos.base import BaseAlgo
from lib.algos.gans import toggle_grad
from lib.utils import sample_indices


class GMMN(BaseAlgo):
    def __init__(
            self,
            base_config,
            x_real,
    ):
        super(GMMN, self).__init__(base_config, x_real)
        self.G_optimizer = torch.optim.Adam(self.G.parameters(), lr=1e-4, betas=(0, 0.9))

    def step(self):
        indices = sample_indices(self.x_real.shape[0], self.batch_size)
        x_past = self.x_real[indices, :self.p].clone().to(self.device)
        x_fake = self.G.sample(self.q, x_past)
        x_fake_past = torch.cat([x_past, x_fake], dim=1)
        G_loss = self.train_step(x_fake_past, self.x_real[indices].clone().to(self.device))
        self.training_loss['MMD'].append(G_loss)
        self.evaluate(x_fake)

    def train_step(self, x_fake, x_real):
        toggle_grad(self.G, True)
        self.G.train()
        self.G_optimizer.zero_grad()
        gloss = mmd_loss(x_real, x_fake)
        gloss.backward()
        self.G_optimizer.step()
        return gloss.item()


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
