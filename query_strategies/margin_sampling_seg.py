import numpy as np
import torch

from .strategy import QueryStrategy, Strategy


def _margin(prob: torch.Tensor, _: torch.Tensor):
    prob_sorted, _ = prob.sort(dim=1, descending=True)
    margin = prob_sorted[:, 0] - prob_sorted[:, 1]
    return torch.mean(torch.mean(margin, -1), -1)


class MarginSamplingSeg(QueryStrategy):
    def query(self, strategy: Strategy, n):
        idxs_unlabeled = np.arange(strategy.n_pool)[~strategy.idxs_lb]
        avg_margin = strategy.avg_fn(
            strategy.X[idxs_unlabeled],
            strategy.Y.numpy()[idxs_unlabeled],
            _margin,
        )
        return idxs_unlabeled[avg_margin.sort()[1][:n]]
