import numpy as np
import torch
from torchmetrics.functional import image_gradients

from .strategy import QueryStrategy


def _var_grad(prob: torch.Tensor, _: torch.Tensor):
    dy, dx = image_gradients(prob)  # [N, C, H, W]
    grad = dy**2 + dx**2  # [N, C, H, W]
    var_grad = torch.var(grad, dim=(-1, -2))  # [N, C]
    return torch.sum(var_grad, dim=1)  # [N]


class VarianceGradientSamplingSeg(QueryStrategy):
    def query(self, strategy, n):
        idxs_unlabeled = np.arange(strategy.n_pool)[~strategy.idxs_lb]
        print(f"{idxs_unlabeled=!r}")
        var_grad = strategy.avg_fn(
            strategy.X[idxs_unlabeled],
            strategy.Y.numpy()[idxs_unlabeled],
            _var_grad,
        )
        print(var_grad)
        return idxs_unlabeled[var_grad.sort()[1][:n]]
