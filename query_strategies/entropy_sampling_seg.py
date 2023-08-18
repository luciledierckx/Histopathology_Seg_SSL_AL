import numpy as np
import torch

from .strategy import QueryStrategy


def _entropy(prob: torch.Tensor, _: torch.Tensor):
    log_prob = torch.log(prob)
    entropy = (prob * log_prob).sum(dim=1)
    return torch.mean(torch.mean(entropy, -1), -1)


class EntropySamplingSeg(QueryStrategy):
    def query(self, strategy, n):
        idxs_unlabeled = np.arange(strategy.n_pool)[~strategy.idxs_lb]
        avg_entropy = strategy.avg_fn(
            strategy.X[idxs_unlabeled],
            strategy.Y.numpy()[idxs_unlabeled],
            _entropy,
        )
        return idxs_unlabeled[avg_entropy.sort()[1][:n]]
