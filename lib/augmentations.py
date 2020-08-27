"""
Simple augmentations to enhance the capability of capturing important features in the first components of the
signature.
"""
from dataclasses import dataclass
from typing import List, Tuple

import signatory
import torch

__all__ = ['AddLags', 'Concat', 'Cumsum', 'LeadLag', 'Scale']


def get_time_vector(size: int, length: int) -> torch.Tensor:
    return torch.linspace(0, 1, length).reshape(1, -1, 1).repeat(size, 1, 1)


def lead_lag_transform(x: torch.Tensor) -> torch.Tensor:
    """
    Lead-lag transformation for a multivariate paths.
    """
    x_rep = torch.repeat_interleave(x, repeats=2, dim=1)
    x_ll = torch.cat([x_rep[:, :-1], x_rep[:, 1:]], dim=2)
    return x_ll


def lead_lag_transform_with_time(x: torch.Tensor) -> torch.Tensor:
    """
    Lead-lag transformation for a multivariate paths.
    """
    t = get_time_vector(x.shape[0], x.shape[1]).to(x.device)
    t_rep = torch.repeat_interleave(t, repeats=3, dim=1)
    x_rep = torch.repeat_interleave(x, repeats=3, dim=1)
    x_ll = torch.cat([
        t_rep[:, 0:-2],
        x_rep[:, 1:-1],
        x_rep[:, 2:],
    ], dim=2)
    return x_ll


def cat_lags(x: torch.Tensor, m: int) -> torch.Tensor:
    q = x.shape[1]
    assert q >= m, 'Lift cannot be performed. q < m : (%s < %s)' % (q, m)
    x_lifted = list()
    for i in range(m):
        x_lifted.append(x[:, i:i + m])
    return torch.cat(x_lifted, dim=-1)


@dataclass
class BaseAugmentation:
    pass

    def apply(self, *args: List[torch.Tensor]) -> torch.Tensor:
        raise NotImplementedError('Needs to be implemented by child.')


@dataclass
class Scale(BaseAugmentation):
    scale: float = 1

    def apply(self, x: torch.Tensor):
        return self.scale * x


@dataclass
class Concat(BaseAugmentation):

    @staticmethod
    def apply(x: torch.Tensor, y: torch.Tensor):
        return torch.cat([x, y], dim=-1)


@dataclass
class Cumsum(BaseAugmentation):
    dim: int = 1

    def apply(self, x: torch.Tensor):
        return x.cumsum(dim=self.dim)


@dataclass
class AddLags(BaseAugmentation):
    m: int = 2

    def apply(self, x: torch.Tensor):
        return cat_lags(x, self.m)


@dataclass
class LeadLag(BaseAugmentation):
    with_time: bool = False

    def apply(self, x: torch.Tensor):
        if self.with_time:
            return lead_lag_transform_with_time(x)
        else:
            return lead_lag_transform(x)


def _apply_augmentation(x: torch.Tensor, y: torch.Tensor, augmentation) \
        -> Tuple[torch.Tensor, torch.Tensor]:
    if type(augmentation).__name__ == 'Concat':  # todo
        return y, augmentation.apply(x, y)
    else:
        return y, augmentation.apply(y)


def apply_augmentations(x: torch.Tensor, augmentations: Tuple) -> torch.Tensor:
    y = x
    for augmentation in augmentations:
        x, y = _apply_augmentation(x, y, augmentation)
    return y


@dataclass
class SignatureConfig:
    augmentations: Tuple
    depth: int
    basepoint: bool = False


def augment_path_and_compute_signatures(x: torch.Tensor, config: SignatureConfig) -> torch.Tensor:
    y = apply_augmentations(x, config.augmentations)
    return signatory.signature(y, config.depth, basepoint=config.basepoint)


def get_standard_augmentation(scale: float) -> Tuple:
    return tuple([Scale(scale), Cumsum(), Concat(), AddLags(m=2), LeadLag(with_time=False)])
