"""
Simple augmentations to enhance the capability of capturing important features in the first components of the
signature and thereby improving the convergence of the generator.
"""

import torch


def get_time_vector(size, length):
    return torch.linspace(0, 1, length).reshape(1, -1, 1).repeat(size, 1, 1)


def add_time(x):
    t = get_time_vector(x.shape[0], x.shape[1])
    return torch.cat([x, t], dim=2)


def lift(x, m):
    q = x.shape[1]
    assert q >= m, 'Lift cannot be performed. q < m : (%s < %s)' % (q, m)
    x_lifted = list()
    for i in range(q - m + 1):
        x_lifted.append(x[:, i:i + m])
    return torch.cat(x_lifted, dim=-1)


def lead_lag_transform(x, with_time=True):
    """
    Lead-lag transformation for a multivariate paths.
    """
    dim = x.shape[-1]
    if with_time:
        x = add_time(x)
        repeats = 1 + 2 * dim
    else:
        repeats = 2 * dim
    x_rep = torch.repeat_interleave(x, repeats=repeats, dim=1)
    x_cat = list()
    dims = repeats - 1 if with_time else repeats
    for i in range(0, dims):
        dimension = i // 2
        x_partial = x_rep[:, i:-(repeats - 1 - i), dimension:dimension + 1] if (repeats - 1 - i) != 0 else x_rep[:, i:,
                                                                                                           dimension:dimension + 1]
        x_cat.append(x_partial)
    if with_time:
        x_cat.append(x_rep[:, repeats - 1:, -1:])
    x_ = torch.cat(x_cat, dim=-1)
    return x_


def lead_lag_transform_2(x, with_time=False):
    """
    Lead-lag transformation for a multivariate paths.
    """
    dim = x.shape[-1]
    x_rep = torch.repeat_interleave(x, repeats=2, dim=1)
    x_ll = list()
    for i in range(dim):
        x_ll.append(x_rep[:, 1:, i:i + 1])
        x_ll.append(x_rep[:, :-1, i:i + 1])
    x_ = torch.cat(x_ll, dim=-1)
    return x_


def multidelayed_lead_lag_transform(x, delay=0):
    """
    Multi-delayed lead-lag transformation for a one-dimensional path.
    """
    repeats = delay + 1
    x_rep = torch.repeat_interleave(x, repeats=repeats, dim=1)
    x_cat = list()
    for i in range(repeats):
        x_cat.append(x_rep[:, i:-(repeats - i)] if i != repeats else x_rep[:, i:])
    return torch.cat(x_cat, dim=2)

