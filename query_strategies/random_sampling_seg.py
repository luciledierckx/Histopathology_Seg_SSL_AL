"random sampling: adaptations for segmentation are structural only (i.e. code architecture)"

import numpy as np

from .strategy import QueryStrategy, Strategy


class RandomSamplingSeg(QueryStrategy):
    def query(self, strategy: Strategy, n):
        inds = np.where(strategy.idxs_lb == 0)[0]
        return inds[np.random.permutation(len(inds))][:n]
