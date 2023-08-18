import numpy as np
import torch

from .strategy import QueryStrategy


def _confidence(prob: torch.Tensor, _: torch.Tensor):
    max, _ = torch.max(prob, dim=1)
    return torch.mean(torch.mean(max, dim=-1), dim=-1)


class LeastConfidenceSeg(QueryStrategy):
    def query(self, strategy, n):
        idxs_unlabeled = np.arange(strategy.n_pool)[~strategy.idxs_lb]

        # average least confidence over an image
        avg_conf = strategy.avg_fn(
            strategy.X[idxs_unlabeled],
            strategy.Y.numpy()[idxs_unlabeled],
            _confidence,
        )
        return idxs_unlabeled[avg_conf.sort()[1][:n]]
